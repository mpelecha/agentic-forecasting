"""The review agent's only door to an LLM.

Reuses the repo's proxy call shape (``llm_processes/_client.py``): model
prefixed ``openai/`` when an ``api_base`` is set, ``reasoning_effort`` injected
through ``extra_body`` so litellm cannot silently drop it, JSON-schema
``response_format`` built by `make_json_schema_response_format`.

What this module adds on top, each because of something observed on the proxy:

- **Capability probe.** The proxy has returned 400 for ``reasoning_effort`` on
  ``gemini-3.5-flash`` (HANDOFF.md:457). One tiny call per run decides whether
  the parameter is sent at all; a 400 is never retried per call.
- **Retry classes.** Transient errors (503, 429, timeout, connection) are
  retried with backoff. ``BadRequestError`` is not: it means the request shape
  is wrong and retrying it is spend without information.
- **Token starvation.** A thinking model spends output budget on thinking; at
  a small ``max_tokens`` it returns ``finish_reason="length"`` with empty
  content (HANDOFF.md:408). That is retried once at double the budget, then
  recorded as a stage failure. Never fabricated.
- **Cost ledger.** ``response_cost`` is zero through the proxy, so every call
  is priced from the settings table unless the proxy reports a number. The
  ledger hard-stops at its budget.
- **Flat schemas.** Gemini via the proxy rejects ``$ref``/``$defs`` in both
  ``response_schema`` and tool declarations; `inline_refs` flattens them.

Nothing here knows what a card or a hypothesis is. Stages call
:meth:`LlmClient.complete` and parse what comes back.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from aieng.forecasting.methods.llm_processes._client import (
    bootstrap_litellm,
    make_json_schema_response_format,
    strip_markdown_fence,
)
from energy_oil_forecasting.cfm_coach.review.settings import (
    DEFAULT_REVIEW_SETTINGS,
    ReviewSettings,
    Stage,
)


logger = logging.getLogger(__name__)

#: Exception class names litellm raises for conditions worth retrying.
TRANSIENT_ERROR_NAMES = frozenset(
    {"ServiceUnavailableError", "RateLimitError", "Timeout", "APIConnectionError", "InternalServerError"}
)
#: Backoff between transient retries, seconds: 5, 15, 45 (`_client.py:352`).
BACKOFF_BASE_SECONDS = 5.0
BACKOFF_FACTOR = 3.0


class BudgetExceededError(RuntimeError):
    """The cost ledger refused a call. The run stops; nothing is fabricated."""


class LlmFailureError(RuntimeError):
    """A call failed in a way the stage must record rather than retry."""


# --------------------------------------------------------------- schemas ----


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Return ``schema`` with every ``$ref`` replaced by its ``$defs`` target.

    Pydantic's ``model_json_schema()`` emits ``$defs`` for any nested model. The
    proxy's Gemini route rejects both keys, so schemas are flattened before they
    leave this module. Recursive definitions cannot be flattened and raise.
    """
    defs = schema.get("$defs", {})

    def resolve(node: Any, stack: tuple[str, ...]) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if ref is not None:
                if not ref.startswith("#/$defs/"):
                    raise ValueError(f"unsupported $ref {ref!r}; only local #/$defs refs are flattened")
                name = ref.removeprefix("#/$defs/")
                if name in stack:
                    raise ValueError(f"recursive schema definition {name!r} cannot be flattened for the proxy")
                if name not in defs:
                    raise ValueError(f"$ref to unknown definition {name!r}")
                target = resolve(defs[name], (*stack, name))
                extras = {k: v for k, v in node.items() if k != "$ref"}
                return {**target, **resolve(extras, stack)}
            return {k: resolve(v, stack) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(v, stack) for v in node]
        return node

    return resolve(copy.deepcopy(schema), ())


def assert_ref_free(schema: Any, *, where: str = "schema") -> None:
    """Raise if any ``$ref`` or ``$defs`` survives; used by tests and at call time."""
    if isinstance(schema, dict):
        for key, value in schema.items():
            if key in ("$ref", "$defs"):
                raise ValueError(f"{where} still contains {key!r}; flatten it with inline_refs()")
            assert_ref_free(value, where=where)
    elif isinstance(schema, list):
        for value in schema:
            assert_ref_free(value, where=where)


def response_format_for(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Flattened, ``additionalProperties``-stripped ``response_format`` dict."""
    flat = inline_refs(schema)
    assert_ref_free(flat, where=f"response schema {name!r}")
    return make_json_schema_response_format(name, flat)


def estimate_tokens(text: str) -> int:
    """Pre-flight estimate: ~4 characters per token. Coarse on purpose; the ledger uses real usage."""
    return max(1, len(text) // 4)


# ------------------------------------------------------------------ usage ----


@dataclass(frozen=True)
class LlmUsage:
    stage: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    #: Present only when the proxy reports ``completion_tokens_details.reasoning_tokens``.
    reasoning_tokens: int | None
    cost_usd: float
    #: ``proxy`` when the proxy reported a non-zero cost, else ``table``.
    cost_source: str
    finish_reason: str | None
    elapsed_seconds: float
    attempts: int
    reasoning_effort_sent: str | None

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class LlmResponse:
    content: str | None
    tool_calls: list[dict[str, Any]]
    usage: LlmUsage
    #: The raw message, for the week log. Never parsed twice.
    raw: dict[str, Any]

    @property
    def ok(self) -> bool:
        return bool(self.content) or bool(self.tool_calls)


class CostLedger:
    """Append-only record of every call's usage, with a hard budget.

    ``check_estimate`` is the pre-flight: a stage asks before it starts whether
    its estimated spend fits. ``record`` is the truth after each call.
    """

    def __init__(self, budget_usd: float, settings: ReviewSettings = DEFAULT_REVIEW_SETTINGS):
        if budget_usd <= 0:
            raise ValueError("budget must be positive")
        self.budget_usd = float(budget_usd)
        self.settings = settings
        self.entries: list[LlmUsage] = []

    @property
    def spent_usd(self) -> float:
        return float(sum(entry.cost_usd for entry in self.entries))

    @property
    def remaining_usd(self) -> float:
        return self.budget_usd - self.spent_usd

    def price(self, prompt_tokens: int, completion_tokens: int) -> float:
        s = self.settings
        return (
            prompt_tokens * s.price_usd_per_million_input + completion_tokens * s.price_usd_per_million_output
        ) / 1_000_000

    def check_estimate(self, *, stage: str, prompt_tokens: int, completion_tokens: int) -> float:
        estimate = self.price(prompt_tokens, completion_tokens)
        if self.spent_usd + estimate > self.budget_usd:
            raise BudgetExceededError(
                f"{stage}: estimated ${estimate:.3f} would exceed the ${self.budget_usd:.2f} budget "
                f"(spent ${self.spent_usd:.3f})"
            )
        return estimate

    def record(self, usage: LlmUsage) -> None:
        self.entries.append(usage)
        if self.spent_usd > self.budget_usd:
            raise BudgetExceededError(
                f"{usage.stage}: spent ${self.spent_usd:.3f} exceeds the ${self.budget_usd:.2f} budget"
            )

    def by_stage(self) -> dict[str, dict[str, float | int]]:
        out: dict[str, dict[str, float | int]] = {}
        for entry in self.entries:
            row = out.setdefault(entry.stage, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0})
            row["calls"] += 1
            row["prompt_tokens"] += entry.prompt_tokens
            row["completion_tokens"] += entry.completion_tokens
            row["cost_usd"] = float(row["cost_usd"]) + entry.cost_usd
        return out

    def write_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for entry in self.entries:
                handle.write(json.dumps(entry.as_dict(), sort_keys=True) + "\n")


# -------------------------------------------------------------- tool loops ----


def schema_text(model: Any) -> str:
    """Compact, `$ref`-free JSON schema of a pydantic model, for pasting into a prompt."""
    return json.dumps(inline_refs(model.model_json_schema()), separators=(",", ":"), sort_keys=True)


def submit_contract(tool: str, argument: str, model: Any) -> str:
    """The exact shape a submit tool's JSON-string argument must take. Strict models reject any other key."""
    return (
        f"The `{argument}` argument of {tool} is a JSON string that must match this schema exactly. "
        "Every key not listed is rejected, so use these names and no others:\n" + schema_text(model)
    )


def forced_tool_choice(name: str) -> dict[str, Any]:
    """OpenAI-shaped `tool_choice` that requires one named function call."""
    return {"type": "function", "function": {"name": name}}


def final_turn_notice(submit_tool: str) -> str:
    return (
        f"Your tool budget is spent. Call {submit_tool} now with your best assessment from what you "
        "already have; no other tool call will be executed."
    )


def turn_notice(remaining: int, submit_tool: str) -> str:
    return f"[{remaining} tool turn(s) remain before you must call {submit_tool}.]"


async def complete_with_forced_submit(
    client: "LlmClient",
    *,
    stage: "Stage",
    messages: Sequence[dict[str, Any]],
    tools: Sequence[dict[str, Any]],
    submit_tool: str,
    force: bool,
) -> "LlmResponse":
    """One loop turn; the final turn requires the submit tool, falling back to `auto` if the proxy rejects it."""
    if not force:
        return await client.complete(stage=stage, messages=messages, tools=tools, tool_choice="auto")
    try:
        return await client.complete(
            stage=stage, messages=messages, tools=tools, tool_choice=forced_tool_choice(submit_tool)
        )
    except LlmFailureError:
        return await client.complete(stage=stage, messages=messages, tools=tools, tool_choice="auto")


# ----------------------------------------------------------------- client ----


class LlmClient(Protocol):
    settings: ReviewSettings
    ledger: CostLedger

    async def complete(
        self,
        *,
        stage: Stage,
        messages: Sequence[dict[str, Any]],
        response_schema: dict[str, Any] | None = None,
        schema_name: str = "Response",
        tools: Sequence[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> LlmResponse: ...


def load_proxy_credentials(env_path: Path | None = None) -> tuple[str, str]:
    """``(OPENAI_BASE_URL, OPENAI_API_KEY)`` from the environment, loading the repo ``.env`` if present.

    Mirrors ``run_daily.load_credentials``: the daily agent runs already use
    these two variables, so the coach uses the same ones and nothing else.
    """
    from dotenv import load_dotenv  # noqa: PLC0415

    if env_path is None:
        env_path = Path(__file__).resolve().parents[4] / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
    base = os.environ.get("OPENAI_BASE_URL")
    key = os.environ.get("OPENAI_API_KEY")
    if not base or not key:
        raise RuntimeError("OPENAI_BASE_URL and OPENAI_API_KEY must be set (the repo .env carries both)")
    return base, key


def _error_name(exc: BaseException) -> str:
    return type(exc).__name__


def _is_transient(exc: BaseException) -> bool:
    return _error_name(exc) in TRANSIENT_ERROR_NAMES


def _is_bad_request(exc: BaseException) -> bool:
    return _error_name(exc) == "BadRequestError"


class ProxyClient:
    """litellm against the Vector proxy. One instance per run; the probe result is per instance."""

    def __init__(
        self,
        settings: ReviewSettings,
        ledger: CostLedger,
        *,
        api_base: str,
        api_key: str,
        acompletion: Callable[..., Awaitable[Any]] | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self.settings = settings
        self.ledger = ledger
        self.api_base = api_base
        self.api_key = api_key
        self._acompletion = acompletion
        self._sleep = sleep
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)
        #: ``None`` until `probe_reasoning_effort` has run; then True/False.
        self.reasoning_effort_supported: bool | None = None
        self.probe_log: list[dict[str, Any]] = []

    # -- wiring ---------------------------------------------------------------

    async def _call(self, **kwargs: Any) -> Any:
        if self._acompletion is None:
            bootstrap_litellm()
            import litellm  # noqa: PLC0415

            self._acompletion = litellm.acompletion
        return await self._acompletion(**kwargs)

    def _kwargs(
        self,
        *,
        stage: Stage,
        messages: Sequence[dict[str, Any]],
        response_schema: dict[str, Any] | None,
        schema_name: str,
        tools: Sequence[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        max_tokens: int,
    ) -> tuple[dict[str, Any], str | None]:
        s = self.settings
        stage_llm = s.stage(stage)
        model = s.model if s.model.startswith("openai/") else f"openai/{s.model}"
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "temperature": s.temperature_for(stage),
            "max_tokens": max_tokens,
            "timeout": s.timeout_seconds,
            "api_base": self.api_base,
            "api_key": self.api_key,
            "drop_params": True,
        }
        if response_schema is not None and tools:
            raise ValueError("a tool-bearing call must not also set response_format; end the loop with a submit tool")
        if response_schema is not None:
            kwargs["response_format"] = response_format_for(schema_name, response_schema)
        if tools:
            for tool in tools:
                assert_ref_free(tool, where="tool declaration")
            kwargs["tools"] = list(tools)
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
        effort_sent: str | None = None
        if stage_llm.reasoning_effort is not None and self.reasoning_effort_supported is not False:
            kwargs["extra_body"] = {"reasoning_effort": stage_llm.reasoning_effort}
            effort_sent = stage_llm.reasoning_effort
        return kwargs, effort_sent

    # -- the probe ------------------------------------------------------------

    async def probe_reasoning_effort(self) -> bool:
        """Send one tiny call per distinct effort value; decide once whether to send the parameter.

        Recorded in ``probe_log`` for the week manifest. A 400 that mentions
        ``reasoning_effort`` switches the run to sending nothing; any other
        error propagates, because it is not what the probe is testing.
        """
        efforts = sorted(
            {
                s.reasoning_effort
                for s in (self.settings.triage, self.settings.deep_dive, self.settings.synthesis, self.settings.critic)
                if s.reasoning_effort is not None
            }
        )
        if not efforts:
            self.reasoning_effort_supported = False
            return False
        supported = True
        for effort in efforts:
            kwargs, _ = self._kwargs(
                stage="probe",
                messages=[{"role": "user", "content": "Reply with the single word: ok"}],
                response_schema=None,
                schema_name="Probe",
                tools=None,
                tool_choice=None,
                max_tokens=self.settings.probe.max_tokens,
            )
            kwargs["extra_body"] = {"reasoning_effort": effort}
            started = time.monotonic()
            try:
                resp = await self._call(**kwargs)
            except Exception as exc:  # noqa: BLE001 - classified below
                if _is_bad_request(exc) and "reasoning" in str(exc).lower():
                    supported = False
                    self.probe_log.append({"effort": effort, "supported": False, "error": str(exc)[:300]})
                    continue
                raise
            usage = self._usage_from(resp, stage="probe", started=started, attempts=1, effort_sent=effort)
            self.ledger.record(usage)
            self.probe_log.append({"effort": effort, "supported": True, "usage": usage.as_dict()})
        self.reasoning_effort_supported = supported
        return supported

    # -- one call -------------------------------------------------------------

    def _usage_from(self, resp: Any, *, stage: str, started: float, attempts: int, effort_sent: str | None) -> LlmUsage:
        usage = getattr(resp, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage is not None else 0
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage is not None else 0
        details = getattr(usage, "completion_tokens_details", None) if usage is not None else None
        reasoning = getattr(details, "reasoning_tokens", None) if details is not None else None
        if reasoning is None and isinstance(details, dict):
            reasoning = details.get("reasoning_tokens")
        hidden = getattr(resp, "_hidden_params", None) or {}
        reported = float(hidden.get("response_cost") or 0.0) if isinstance(hidden, dict) else 0.0
        if reported > 0:
            cost, source = reported, "proxy"
        else:
            cost, source = self.ledger.price(prompt_tokens, completion_tokens), "table"
        choices = getattr(resp, "choices", None) or []
        finish = getattr(choices[0], "finish_reason", None) if choices else None
        return LlmUsage(
            stage=stage,
            model=self.settings.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=int(reasoning) if reasoning is not None else None,
            cost_usd=cost,
            cost_source=source,
            finish_reason=finish,
            elapsed_seconds=time.monotonic() - started,
            attempts=attempts,
            reasoning_effort_sent=effort_sent,
        )

    @staticmethod
    def _message_from(resp: Any) -> tuple[str | None, list[dict[str, Any]], dict[str, Any]]:
        choices = getattr(resp, "choices", None) or []
        if not choices:
            return None, [], {}
        message = choices[0].message
        raw_content = getattr(message, "content", None)
        content = strip_markdown_fence(raw_content) if raw_content else None
        calls: list[dict[str, Any]] = []
        for call in getattr(message, "tool_calls", None) or []:
            fn = getattr(call, "function", None)
            calls.append(
                {
                    "id": getattr(call, "id", None),
                    "name": getattr(fn, "name", None),
                    "arguments": getattr(fn, "arguments", None),
                }
            )
        # Grounded search: the proxy passes Gemini's grounding metadata through
        # `provider_specific_fields`, the same field `research_pipeline._do_search` reads.
        fields = getattr(choices[0], "provider_specific_fields", None) or {}
        metadata = fields.get("grounding_metadata") if isinstance(fields, dict) else None
        grounding = []
        for chunk in (metadata or {}).get("groundingChunks", []) or []:
            web = chunk.get("web") or {}
            if web.get("uri"):
                grounding.append({"url": web["uri"], "title": web.get("title")})
        raw = {
            "content": raw_content,
            "tool_calls": calls,
            "finish_reason": getattr(choices[0], "finish_reason", None),
            "grounding": grounding,
        }
        return content, calls, raw

    async def _call_with_transient_retry(self, kwargs: dict[str, Any], *, stage: str) -> tuple[Any, int]:
        attempts = self.settings.transient_retry_attempts
        for attempt in range(attempts):
            try:
                return await self._call(**kwargs), attempt + 1
            except Exception as exc:  # noqa: BLE001 - classified below
                if _is_bad_request(exc):
                    raise LlmFailureError(f"{stage}: bad request, not retried: {str(exc)[:500]}") from exc
                if not _is_transient(exc) or attempt == attempts - 1:
                    raise
                delay = BACKOFF_BASE_SECONDS * (BACKOFF_FACTOR**attempt)
                logger.warning(
                    "%s: transient %s on attempt %d/%d; sleeping %.0fs",
                    stage,
                    _error_name(exc),
                    attempt + 1,
                    attempts,
                    delay,
                )
                await self._sleep(delay)
        raise AssertionError("unreachable")

    async def complete(
        self,
        *,
        stage: Stage,
        messages: Sequence[dict[str, Any]],
        response_schema: dict[str, Any] | None = None,
        schema_name: str = "Response",
        tools: Sequence[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> LlmResponse:
        if self.reasoning_effort_supported is None and self.settings.stage(stage).reasoning_effort is not None:
            raise RuntimeError("call probe_reasoning_effort() once before any judgment call")
        budget_tokens = max_tokens or self.settings.stage(stage).max_tokens
        prompt_text = json.dumps(list(messages))
        self.ledger.check_estimate(
            stage=stage, prompt_tokens=estimate_tokens(prompt_text), completion_tokens=budget_tokens
        )

        async with self._semaphore:
            total_attempts = 0
            for starvation_round in range(2):
                kwargs, effort_sent = self._kwargs(
                    stage=stage,
                    messages=messages,
                    response_schema=response_schema,
                    schema_name=schema_name,
                    tools=tools,
                    tool_choice=tool_choice,
                    max_tokens=budget_tokens,
                )
                started = time.monotonic()
                resp, attempts = await self._call_with_transient_retry(kwargs, stage=stage)
                total_attempts += attempts
                usage = self._usage_from(
                    resp, stage=stage, started=started, attempts=total_attempts, effort_sent=effort_sent
                )
                self.ledger.record(usage)
                content, calls, raw = self._message_from(resp)
                starved = usage.finish_reason == "length" and not content and not calls
                if starved and starvation_round == 0:
                    logger.warning(
                        "%s: empty content with finish_reason=length at max_tokens=%d; retrying at 2x",
                        stage,
                        budget_tokens,
                    )
                    budget_tokens *= 2
                    continue
                if starved:
                    raise LlmFailureError(
                        f"{stage}: empty content with finish_reason=length twice (max_tokens={budget_tokens})"
                    )
                return LlmResponse(content=content, tool_calls=calls, usage=usage, raw=raw)
        raise AssertionError("unreachable")


# ------------------------------------------------------------------- fake ----


@dataclass
class FakeCall:
    stage: str
    messages: list[dict[str, Any]]
    response_schema: dict[str, Any] | None
    tools: list[dict[str, Any]] | None
    max_tokens: int
    tool_choice: str | dict[str, Any] | None = None


@dataclass
class FakeClient:
    """Scripted responses for ``--no-llm`` runs and tests. Records every call.

    ``script`` maps a stage to a queue of contents; a callable receives the
    `FakeCall` and returns content. When the queue is empty the client returns
    ``default`` (``None`` means an empty-content response, which stages must
    treat as a failure -- exactly what a real starvation looks like).
    """

    settings: ReviewSettings = DEFAULT_REVIEW_SETTINGS
    ledger: CostLedger = field(default_factory=lambda: CostLedger(1_000.0))
    script: dict[str, list[str | dict[str, Any]]] = field(default_factory=dict)
    responder: Callable[[FakeCall], str | dict[str, Any] | None] | None = None
    default: str | None = None
    calls: list[FakeCall] = field(default_factory=list)
    reasoning_effort_supported: bool | None = True

    async def probe_reasoning_effort(self) -> bool:
        return bool(self.reasoning_effort_supported)

    async def complete(
        self,
        *,
        stage: Stage,
        messages: Sequence[dict[str, Any]],
        response_schema: dict[str, Any] | None = None,
        schema_name: str = "Response",
        tools: Sequence[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> LlmResponse:
        if response_schema is not None:
            assert_ref_free(inline_refs(response_schema), where=f"response schema {schema_name!r}")
        call = FakeCall(
            stage=stage,
            messages=list(messages),
            response_schema=response_schema,
            tools=list(tools) if tools else None,
            max_tokens=max_tokens or self.settings.stage(stage).max_tokens,
            tool_choice=tool_choice,
        )
        self.calls.append(call)
        scripted: str | dict[str, Any] | None
        if self.responder is not None:
            scripted = self.responder(call)
        elif self.script.get(stage):
            scripted = self.script[stage].pop(0)
        else:
            scripted = self.default
        tool_calls: list[dict[str, Any]] = []
        content: str | None
        grounding: list[dict[str, Any]] = []
        if isinstance(scripted, dict) and ("tool_calls" in scripted or "grounding" in scripted):
            tool_calls = list(scripted.get("tool_calls", []))
            grounding = list(scripted.get("grounding", []))
            content = scripted.get("content")
        elif isinstance(scripted, dict):
            content = json.dumps(scripted)
        else:
            content = scripted
        prompt_tokens = estimate_tokens(json.dumps(call.messages))
        completion_tokens = estimate_tokens(content or json.dumps(tool_calls))
        usage = LlmUsage(
            stage=stage,
            model="fake",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=None,
            cost_usd=self.ledger.price(prompt_tokens, completion_tokens),
            cost_source="table",
            finish_reason="stop" if (content or tool_calls) else "length",
            elapsed_seconds=0.0,
            attempts=1,
            reasoning_effort_sent=None,
        )
        self.ledger.record(usage)
        return LlmResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            raw={"content": content, "tool_calls": tool_calls, "grounding": grounding},
        )


__all__ = [
    "BACKOFF_BASE_SECONDS",
    "BACKOFF_FACTOR",
    "TRANSIENT_ERROR_NAMES",
    "BudgetExceededError",
    "CostLedger",
    "FakeCall",
    "FakeClient",
    "LlmClient",
    "LlmFailureError",
    "LlmResponse",
    "LlmUsage",
    "ProxyClient",
    "assert_ref_free",
    "estimate_tokens",
    "inline_refs",
    "load_proxy_credentials",
    "response_format_for",
]
