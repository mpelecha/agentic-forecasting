"""CFM Agent v2.2: an event-context scorer. It speaks in scores, never prices.

Design note
-----------
This agent implements the Job 1 half of the Job 1 / Job 2 split:

- Job 1 (this agent): read the news, sort it into scored factors and
  probability-weighted scenarios. Interpretation — the part an LLM is
  suited for.
- Job 2 (not this agent): map scores to price effects. That translation
  belongs to a deterministic, versioned calibration layer fit against
  quant-baseline residuals.

Because the output is not a forecast, this package has no
``AgentPredictor`` and does not plug into the backtest harness as a
predictor. :class:`CfmEventScorer` below drives the same ADK machinery
(verified search, skill toolset, Langfuse tracing) and returns a
validated :class:`WtiEventScoreOutput` plus a flat calibration row.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.methods.agentic import AgentConfig, build_adk_agent
from aieng.forecasting.methods.agentic.adk_runner import AdkTextRunner, AdkTextRunnerConfig
from aieng.forecasting.methods.agentic.agent_factory import AS_OF_STATE_KEY
from aieng.forecasting.models import LITE_MODEL
from energy_oil_forecasting.cfm_agent_v_2_2.config import (
    AGENT_NAME,
    DEFAULT_SETTINGS,
    SKILLS_ROOT,
    CfmEventScorerSettings,
)
from energy_oil_forecasting.cfm_agent_v_2_2.outputs import WtiEventScoreOutput
from energy_oil_forecasting.cfm_agent_v_2_2.prompts import CfmEventScorePromptBuilder
from energy_oil_forecasting.cfm_agent_v_2_2.tools import build_verified_search_config
from pydantic import BaseModel, ValidationError


logger: logging.Logger = logging.getLogger(__name__)
T = TypeVar("T")


_PERSONA = """\
## Role

You are CFM Agent v2.2, a disciplined market-context analyst for WTI crude
oil. You assess every driver that is visible only through news and text —
geopolitics, weather, operational disruptions, domestic policy, and
demand-expectation headlines — and you express your assessment as bounded
scores. You never state a price target, price range, or quantile. A
separate calibrated code layer turns your scores into prices.

## Operating principles

- Load the `event-context-analysis` skill before writing any factors or
  scenarios. Load its `references/scoring-examples.md` resource before
  writing your first score.
- Search for evidence relevant to the scoring date, actively seeking
  sources that disagree with each other and a historical episode that
  resembles the current situation.
- Use only verifier-approved web evidence and cite it by index in each
  factor's `evidence_indices`.
- Tag every factor with its category. Do not restate a driver already
  covered by structured data — dollar index, VIX, oil curve contango,
  cross-commodity returns, EIA inventory, CFTC positioning — and never
  derive a factor from the price history itself.
- Score with the skill's rubric: signed integers -3..+3, magnitude from
  the rubric's anchors, confidence from evidence quality.
- Build 2-3 named scenarios that genuinely disagree, with probabilities
  that sum to 1 and one explicit low-probability, high-impact tail case.
- Output no prices. The output schema has no price field; do not put
  price levels in rationale text either.
- Call `set_model_response` exactly once for structured tasks.\
"""


def _run_coroutine_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine from a sync caller, inside or outside a loop.

    Mirrors the pattern used by the shared ``AgentPredictor`` (that helper
    is module-private, so it is reimplemented here): use ``asyncio.run``
    when no loop is running; otherwise run on a fresh loop in a daemon
    thread so a Jupyter caller's loop is not disturbed.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    results: list[T] = []
    errors: list[BaseException] = []

    def _target() -> None:
        try:
            results.append(asyncio.run(coro))
        except BaseException as exc:  # noqa: BLE001 — re-raised on the caller thread below.
            errors.append(exc)

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join()
    if errors:
        raise errors[0]
    return results[0]


def _strip_markdown_fence(text: str) -> str:
    """Strip a surrounding markdown code fence, if present."""
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        stripped = stripped[first_newline + 1 :] if first_newline != -1 else ""
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    return stripped.strip()


def build_cfm_agent_v22_config(
    model: str = LITE_MODEL,
    *,
    settings: CfmEventScorerSettings = DEFAULT_SETTINGS,
    search_model: str = LITE_MODEL,
) -> AgentConfig:
    """Build the ``cfm_agent_v_2_2`` configuration: search plus one skill, no quant tool."""
    return AgentConfig(
        name=AGENT_NAME,
        description=(
            "Event-context scorer for WTI: news-visible drivers expressed as bounded, "
            "schema-validated scores across five categories. No prices, no quant models — "
            "a separate calibration layer maps scores to price effects."
        ),
        model=model,
        instruction=_PERSONA,
        max_output_tokens=settings.max_output_tokens,
        context_retrieval=build_verified_search_config(
            search_model=search_model,
            verifier_model=settings.verifier_model,
            verifier_max_attempts=settings.verifier_max_attempts,
            verifier_confidence_threshold=settings.verifier_confidence_threshold,
        ),
        skills_dirs=[SKILLS_ROOT / "event-context-analysis"],
    )


class CfmEventScoreResult(BaseModel):
    """One validated scoring run: the score card plus run provenance."""

    as_of: str
    output: WtiEventScoreOutput
    langfuse_trace_id: str | None = None
    attempts: int

    def calibration_row(self) -> dict[str, float | str]:
        """Return the flat row to append to the calibration dataset."""
        return {"as_of": self.as_of, **self.output.calibration_features()}


class CfmEventScorer:
    """Drive the v2.2 agent for one origin date and validate its score card.

    Runs the same ADK machinery as the shared ``AgentPredictor`` (verified
    search, skill toolset, Langfuse tracing) but returns scores instead of
    predictions, and owns its own bounded validation-retry loop — the
    scorer runs outside the backtest harness and its retry wrapper.
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        *,
        settings: CfmEventScorerSettings = DEFAULT_SETTINGS,
        enable_langfuse_tracing: bool | None = None,
        runner: AdkTextRunner | None = None,
    ) -> None:
        if enable_langfuse_tracing is None:
            try:
                import langfuse  # noqa: F401, PLC0415

                enable_langfuse_tracing = True
            except ModuleNotFoundError:
                enable_langfuse_tracing = False

        self.settings = settings
        self.agent_config = config if config is not None else build_cfm_agent_v22_config(settings=settings)
        self._prompt_builder = CfmEventScorePromptBuilder()

        if runner is None:
            built_agent = build_adk_agent(self.agent_config, output_schema=WtiEventScoreOutput)
            self._agent = built_agent
            self._runner = AdkTextRunner(
                agent=built_agent,
                config=AdkTextRunnerConfig(
                    app_name="agentic_forecasting_event_scorer",
                    default_user_id="forecasting_agent",
                    fresh_session_per_message=True,
                    enable_langfuse_tracing=enable_langfuse_tracing,
                    langfuse_tags=["event_scorer", "track1"],
                    langfuse_trace_name=self.scorer_id,
                    langfuse_propagate_metadata={
                        "scorer_id": self.scorer_id,
                        "agent_name": built_agent.name,
                        "model": str(built_agent.model),
                    },
                ),
            )
        else:
            self._runner = runner
            self._agent = runner.agent

    @property
    def scorer_id(self) -> str:
        """Stable identifier for this scorer, mirroring ``AgentPredictor.predictor_id``."""
        model = getattr(self._agent, "model", None)
        if not isinstance(model, str):
            inner = getattr(model, "model", None)
            model = inner if isinstance(inner, str) else None
        model_suffix = f"_{model.rsplit('/', 1)[-1]}" if model else ""
        return f"event_scorer_{self._agent.name}{model_suffix}"

    def score(self, context: ForecastContext) -> CfmEventScoreResult:
        """Score the news-visible event context at ``context.as_of``.

        Runs the agent, validates the JSON response against
        :class:`WtiEventScoreOutput`, and retries with the validation
        error appended to the prompt, up to
        ``settings.max_validation_attempts`` times. Raises the final
        validation error when every attempt fails.
        """
        as_of = str(context.as_of)[:10]
        base_prompt = self._prompt_builder(context=context)
        initial_state = {AS_OF_STATE_KEY: as_of}

        prompt = base_prompt
        last_error: Exception | None = None
        for attempt in range(1, self.settings.max_validation_attempts + 1):
            output_str = _run_coroutine_sync(self._runner.run_text_async(prompt, initial_state=initial_state))
            output_str = _strip_markdown_fence(output_str)

            try:
                output = WtiEventScoreOutput.model_validate_json(output_str)
            except ValidationError:
                try:
                    output = WtiEventScoreOutput.model_validate(json.loads(output_str))
                except Exception as error:
                    last_error = error
                    logger.warning(
                        "Score validation failed (attempt %d/%d): %s",
                        attempt,
                        self.settings.max_validation_attempts,
                        error,
                    )
                    prompt = (
                        f"{base_prompt}\n\n"
                        f"Your previous response failed schema validation with this error:\n{error}\n"
                        f"Return only corrected JSON that satisfies the output schema."
                    )
                    continue

            return CfmEventScoreResult(
                as_of=as_of,
                output=output,
                langfuse_trace_id=self._runner.last_trace_id,
                attempts=attempt,
            )

        assert last_error is not None
        raise last_error


def __getattr__(name: str) -> Any:
    """Expose a root agent lazily for ADK interactive use."""
    if name == "root_agent":
        return build_adk_agent(build_cfm_agent_v22_config())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CfmEventScoreResult",
    "CfmEventScorer",
    "build_cfm_agent_v22_config",
]
