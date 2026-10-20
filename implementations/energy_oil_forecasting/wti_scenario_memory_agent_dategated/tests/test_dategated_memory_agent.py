import inspect
from types import SimpleNamespace

import pytest
from energy_oil_forecasting.cfm_agent_v_5_2.schemas import ActiveSource
from energy_oil_forecasting.cfm_agent_v_5_2_dategated import date_gate
from energy_oil_forecasting.wti_scenario_memory_agent import build_agent, memory
from energy_oil_forecasting.wti_scenario_memory_agent_dategated import build_agent_dategated


def _search_tool(predictor):
    return next(t for t in predictor.inner._agent.tools if getattr(t, "__name__", None) == "search_web")


def test_only_the_search_internals_and_the_id_differ_from_the_ungated_agent(monkeypatch, tmp_path):
    monkeypatch.setattr(memory, "MEMORY_DIR", tmp_path)
    plain, gated = build_agent(), build_agent_dategated()

    # What the LLM sees: instruction, tool order, and search_web's name/signature/description.
    assert gated.inner._agent.instruction == plain.inner._agent.instruction
    assert [getattr(t, "__name__", type(t).__name__) for t in gated.inner._agent.tools] == [
        getattr(t, "__name__", type(t).__name__) for t in plain.inner._agent.tools
    ]
    plain_search, gated_search = _search_tool(plain), _search_tool(gated)
    assert gated_search is not plain_search
    assert inspect.signature(gated_search) == inspect.signature(plain_search)
    assert inspect.getdoc(gated_search) == inspect.getdoc(plain_search)

    assert gated.predictor_id == plain.predictor_id.replace(
        "scenario_schema_anchored_enhanced", "scenario_schema_anchored_enhanced_dategated"
    )
    # Memory is keyed by predictor ID: the gated agent never reads the ungated agent's frameworks.
    assert memory._memory_path(gated.predictor_id, "t") != memory._memory_path(plain.predictor_id, "t")


class _Choice:
    def __init__(self, rows, supports):
        self.message = SimpleNamespace(content="raw")
        self.provider_specific_fields = {
            "grounding_metadata": {
                "groundingChunks": [{"web": {"uri": url, "title": None}} for url in rows],
                "groundingSupports": supports,
            }
        }


@pytest.mark.asyncio
async def test_gated_search_returns_only_dated_text_or_the_failure_sentinel(monkeypatch):
    early, late = "https://site.test/early", "https://site.test/late"
    responses = {
        "both": _Choice(
            [early, late],
            [
                {"segment": {"text": "Early fact."}, "groundingChunkIndices": [0]},
                {"segment": {"text": "Late fact."}, "groundingChunkIndices": [1]},
            ],
        ),
        "late_only": _Choice([late], [{"segment": {"text": "Late fact."}, "groundingChunkIndices": [0]}]),
    }
    verified: list[str] = []

    async def fake_completion(*, messages, **_):
        key = "late_only" if "late only" in messages[-1]["content"] else "both"
        return SimpleNamespace(choices=[responses[key]])

    async def fake_verify(*, text, **_):
        verified.append(text)
        return SimpleNamespace(clean=True, confidence=10, flagged_claims=[], filtered_text=text)

    dates = {early: "2025-07-01", late: "2026-08-20"}
    monkeypatch.setattr("litellm.acompletion", fake_completion)
    monkeypatch.setattr("aieng.forecasting.methods.agentic.agent_factory._verify_no_leakage", fake_verify)
    monkeypatch.setattr(
        date_gate,
        "_inspect_source",
        lambda source, *_: SimpleNamespace(
            publication_date=dates.get(source.resolved_url),
            publication_date_provenance="html_metadata",
            potentially_mutable=False,
            extraction_warnings=[],
        ),
    )
    gated = build_agent_dategated()
    monkeypatch.setattr(
        gated.gate,
        "_resolve_uncached",
        lambda url, title: ActiveSource(
            source_id="x", rank=1, grounding_url=url, resolved_url=url, resolution_status="resolved"
        ),
    )
    search_web = _search_tool(gated)

    text = await search_web("oil news", cutoff_date="2025-07-07")
    assert text == f"Early fact.\n\nSources:\n{early}"
    assert verified == ["Early fact."]  # the late page's sentence never reaches the verifier or the agent

    failed = await search_web("late only", cutoff_date="2025-07-07")
    assert failed.startswith("[SEARCH_VERIFICATION_FAILED]")
    assert [entry["outcome"] for entry in gated.gate_log] == [
        "accepted",
        "no_dated_text",
        "no_dated_text",
        "no_dated_text",
    ]
