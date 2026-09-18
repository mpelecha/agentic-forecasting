"""The LLM door: probe once, retry only what is transient, never fabricate, always price."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from litellm import exceptions as litellm_exceptions
from pydantic import BaseModel

from energy_oil_forecasting.cfm_coach.review.llm import (
    BudgetExceededError,
    CostLedger,
    FakeClient,
    LlmFailureError,
    ProxyClient,
    assert_ref_free,
    inline_refs,
    response_format_for,
)
from energy_oil_forecasting.cfm_coach.review.settings import ReviewSettings


SETTINGS = ReviewSettings(max_concurrency=2)


# -- schemas -------------------------------------------------------------------


class Tag(BaseModel):
    code: str
    severity: int


class TriageOut(BaseModel):
    tags: list[Tag]
    notes: str | None = None


def test_pydantic_schemas_are_flattened_for_the_proxy():
    schema = TriageOut.model_json_schema()
    assert "$defs" in schema  # the thing the proxy rejects
    flat = inline_refs(schema)
    assert_ref_free(flat)
    assert flat["properties"]["tags"]["items"]["properties"]["code"]["type"] == "string"
    fmt = response_format_for("TriageOut", schema)
    assert fmt["type"] == "json_schema"
    assert "additionalProperties" not in str(fmt)


def test_recursive_schema_cannot_be_flattened():
    schema = {
        "$defs": {"Node": {"type": "object", "properties": {"child": {"$ref": "#/$defs/Node"}}}},
        "$ref": "#/$defs/Node",
    }
    with pytest.raises(ValueError, match="recursive"):
        inline_refs(schema)


# -- ledger ---------------------------------------------------------------------


def test_ledger_prices_from_the_table_and_hard_stops():
    ledger = CostLedger(0.01, SETTINGS)
    # 1M input at $1.50 is far past a one-cent budget.
    with pytest.raises(BudgetExceededError):
        ledger.check_estimate(stage="triage", prompt_tokens=1_000_000, completion_tokens=0)
    assert ledger.price(1_000_000, 0) == pytest.approx(1.50)
    assert ledger.price(0, 1_000_000) == pytest.approx(9.00)


# -- proxy client, with a scripted acompletion ---------------------------------


def _resp(content="ok", *, finish="stop", prompt=100, completion=20, reasoning=None, cost=None, tool_calls=None):
    details = SimpleNamespace(reasoning_tokens=reasoning) if reasoning is not None else None
    usage = SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion, completion_tokens_details=details)
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    resp = SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish)], usage=usage)
    resp._hidden_params = {"response_cost": cost}
    return resp


class Scripted:
    """An acompletion stand-in: pops outcomes in order; an exception instance is raised."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


async def _no_sleep(_seconds: float) -> None:
    return None


def _client(scripted, budget=10.0):
    return ProxyClient(
        SETTINGS,
        CostLedger(budget, SETTINGS),
        api_base="https://proxy.test/v1",
        api_key="k",
        acompletion=scripted,
        sleep=_no_sleep,
    )


def test_probe_records_support_and_calls_carry_effort_via_extra_body():
    scripted = Scripted(_resp(), _resp(), _resp(), _resp())  # minimal, medium, high probes + one triage call
    client = _client(scripted)
    assert asyncio.run(client.probe_reasoning_effort()) is True
    assert [c["extra_body"]["reasoning_effort"] for c in scripted.calls] == ["high", "medium", "minimal"]
    out = asyncio.run(client.complete(stage="triage", messages=[{"role": "user", "content": "x"}]))
    call = scripted.calls[-1]
    assert call["model"] == "openai/gemini-3.5-flash"
    assert call["api_base"] == "https://proxy.test/v1"
    assert call["extra_body"] == {"reasoning_effort": "minimal"}
    assert call["temperature"] == 1.0 and call["max_tokens"] == SETTINGS.triage.max_tokens
    assert out.content == "ok" and out.usage.reasoning_effort_sent == "minimal"


def test_probe_400_on_reasoning_effort_disables_the_parameter_for_the_run():
    err = litellm_exceptions.BadRequestError("reasoning_effort is not supported", model="m", llm_provider="openai")
    scripted = Scripted(err, err, err, _resp())
    client = _client(scripted)
    assert asyncio.run(client.probe_reasoning_effort()) is False
    assert client.reasoning_effort_supported is False
    assert all(entry["supported"] is False for entry in client.probe_log)
    asyncio.run(client.complete(stage="synthesis", messages=[{"role": "user", "content": "x"}]))
    assert "extra_body" not in scripted.calls[-1]


def test_a_probe_error_that_is_not_about_reasoning_propagates():
    err = litellm_exceptions.BadRequestError("invalid api key", model="m", llm_provider="openai")
    client = _client(Scripted(err))
    with pytest.raises(litellm_exceptions.BadRequestError):
        asyncio.run(client.probe_reasoning_effort())


def test_judgment_calls_refuse_to_run_before_the_probe():
    client = _client(Scripted(_resp()))
    with pytest.raises(RuntimeError, match="probe"):
        asyncio.run(client.complete(stage="triage", messages=[]))


def test_lookup_calls_need_no_probe_and_are_pinned_at_zero():
    scripted = Scripted(_resp())
    client = _client(scripted)
    asyncio.run(client.complete(stage="lookup", messages=[{"role": "user", "content": "x"}]))
    assert scripted.calls[-1]["temperature"] == 0.0
    assert "extra_body" not in scripted.calls[-1]


def test_transient_errors_are_retried_and_bad_requests_are_not():
    transient = litellm_exceptions.ServiceUnavailableError("503", model="m", llm_provider="openai")
    scripted = Scripted(transient, _resp())
    client = _client(scripted)
    out = asyncio.run(client.complete(stage="lookup", messages=[]))
    assert out.content == "ok" and out.usage.attempts == 2

    bad = litellm_exceptions.BadRequestError("schema invalid", model="m", llm_provider="openai")
    scripted = Scripted(bad, _resp())
    client = _client(scripted)
    with pytest.raises(LlmFailureError, match="not retried"):
        asyncio.run(client.complete(stage="lookup", messages=[]))
    assert len(scripted.calls) == 1


def test_token_starvation_is_retried_once_at_double_budget_then_recorded_as_failure():
    scripted = Scripted(_resp(content=None, finish="length"), _resp(content="late", finish="stop"))
    client = _client(scripted)
    out = asyncio.run(client.complete(stage="lookup", messages=[]))
    assert out.content == "late"
    assert scripted.calls[0]["max_tokens"] * 2 == scripted.calls[1]["max_tokens"]

    scripted = Scripted(_resp(content="", finish="length"), _resp(content="", finish="length"))
    client = _client(scripted)
    with pytest.raises(LlmFailureError, match="length"):
        asyncio.run(client.complete(stage="lookup", messages=[]))
    # Both starved calls were still priced: spend is spend.
    assert len(client.ledger.entries) == 2


def test_cost_comes_from_the_table_when_the_proxy_reports_nothing_and_reasoning_tokens_are_kept():
    scripted = Scripted(
        _resp(prompt=1_000_000, completion=0, reasoning=1234, cost=0.0), _resp(prompt=10, completion=10, cost=0.42)
    )
    client = _client(scripted, budget=100.0)
    a = asyncio.run(client.complete(stage="lookup", messages=[])).usage
    b = asyncio.run(client.complete(stage="lookup", messages=[])).usage
    assert (a.cost_source, a.cost_usd, a.reasoning_tokens) == ("table", pytest.approx(1.50), 1234)
    assert (b.cost_source, b.cost_usd) == ("proxy", 0.42)
    assert client.ledger.by_stage()["lookup"]["calls"] == 2


def test_tools_and_response_format_never_share_a_call():
    client = _client(Scripted(_resp()))
    client.reasoning_effort_supported = True
    with pytest.raises(ValueError, match="submit tool"):
        asyncio.run(
            client.complete(
                stage="synthesis",
                messages=[],
                response_schema={"type": "object"},
                tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
            )
        )


def test_budget_pre_flight_stops_a_call_before_it_is_made():
    scripted = Scripted(_resp())
    client = _client(scripted, budget=0.001)
    with pytest.raises(BudgetExceededError):
        asyncio.run(client.complete(stage="lookup", messages=[{"role": "user", "content": "x" * 4000}]))
    assert scripted.calls == []


# -- fake client -----------------------------------------------------------------


def test_fake_client_scripts_per_stage_and_records_calls():
    fake = FakeClient(settings=SETTINGS, script={"triage": ['{"tags": []}']}, default=None)
    first = asyncio.run(
        fake.complete(
            stage="triage",
            messages=[{"role": "user", "content": "card"}],
            response_schema=TriageOut.model_json_schema(),
        )
    )
    second = asyncio.run(fake.complete(stage="triage", messages=[]))
    assert first.content == '{"tags": []}' and first.ok
    assert second.content is None and not second.ok and second.usage.finish_reason == "length"
    assert [c.stage for c in fake.calls] == ["triage", "triage"]
    assert fake.ledger.spent_usd > 0
