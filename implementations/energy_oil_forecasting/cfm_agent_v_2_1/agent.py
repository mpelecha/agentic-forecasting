"""Geopolitical-only CFM Agent v2.1: one search tool, one skill, no quant models."""

from __future__ import annotations

from typing import Any

from aieng.forecasting.methods.agentic import AgentConfig, AgentPredictor, build_adk_agent
from aieng.forecasting.models import LITE_MODEL
from energy_oil_forecasting.cfm_agent_v_2_1.config import (
    AGENT_NAME,
    DEFAULT_SETTINGS,
    SKILLS_ROOT,
    CfmGeoAgentSettings,
)
from energy_oil_forecasting.cfm_agent_v_2_1.outputs import WtiGeoForecastOutput
from energy_oil_forecasting.cfm_agent_v_2_1.prompts import CfmGeoPromptBuilder
from energy_oil_forecasting.cfm_agent_v_2_1.tools import build_verified_search_config


_PERSONA = """\
## Role

You are CFM Agent v2.1, a disciplined geopolitical market forecaster. You
hold no built-in quant models — you reason from compressed target price
history plus verified web evidence, decomposed into named, competing
scenarios.

## Operating principles

- Load the `geopolitical-analysis` skill before writing any factors or
  scenarios. Load its `references/factor-examples.md` resource before
  writing your first factor or scenario.
- Search for evidence relevant to the forecast, actively seeking sources
  that disagree with each other and a historical episode that resembles
  the current situation.
- Use only verifier-approved web evidence and cite its URL in
  `verified_evidence`.
- Identify the shared core/transitory factor set once. Do not restate a
  factor already implied by price history alone — ground factors in
  geopolitical developments.
- Build 2-3 named scenarios that genuinely disagree, including one
  explicit tail case, each with a real price range at the longest
  horizon.
- Your final quantile grid must be consistent with the spread across your
  own scenarios — a violation is rejected and retried, not silently
  accepted.
- Call `set_model_response` exactly once for structured tasks.\
"""


def build_cfm_agent_v21_config(
    model: str = LITE_MODEL,
    *,
    settings: CfmGeoAgentSettings = DEFAULT_SETTINGS,
    search_model: str = LITE_MODEL,
) -> AgentConfig:
    """Build the ``cfm_agent_v_2_1`` configuration: search plus one skill, no quant tool."""
    return AgentConfig(
        name=AGENT_NAME,
        description=(
            "Geopolitical-only probabilistic WTI forecaster with verified research and "
            "a structured, schema-validated scenario decomposition. No quant models."
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
        skills_dirs=[SKILLS_ROOT / "geopolitical-analysis"],
    )


def build_cfm_agent_v21_predictor(config: AgentConfig) -> AgentPredictor:
    """Wrap a v2.1 config in the shared :class:`AgentPredictor`.

    Unlike ``cfm_agent_v_2_0``, there is no quant tool and therefore no
    authoritative-tool-truth merge step to layer on top — the shared
    predictor already attaches Langfuse trace identifiers to
    ``Prediction.metadata``.
    """
    return AgentPredictor(
        agent_config=config,
        prompt_builder=CfmGeoPromptBuilder(),
        output_schema=WtiGeoForecastOutput,
    )


def __getattr__(name: str) -> Any:
    """Expose a root agent lazily for ADK interactive use."""
    if name == "root_agent":
        return build_adk_agent(build_cfm_agent_v21_config())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["build_cfm_agent_v21_config", "build_cfm_agent_v21_predictor"]
