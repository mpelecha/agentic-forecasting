"""Three-tool, three-model CFM Agent v2.0 with rich attribution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aieng.forecasting.data import DataService
from aieng.forecasting.methods.agentic import AgentConfig, build_adk_agent
from aieng.forecasting.models import LITE_MODEL
from energy_oil_forecasting.cfm_agent_v_2_0.config import (
    AGENT_NAME,
    DEFAULT_SETTINGS,
    SKILLS_ROOT,
    CfmAgentSettings,
)
from energy_oil_forecasting.cfm_agent_v_2_0.predictor import CfmRichAgentPredictor
from energy_oil_forecasting.cfm_agent_v_2_0.tools import (
    MarketDataTool,
    build_code_execution_config,
    build_verified_search_config,
)
from energy_oil_forecasting.data import (
    DEFAULT_WTI_COVARIATE_SERIES_IDS,
    build_wti_multivariate_service,
)


_CONFIG_TO_MARKET_TOOL: dict[int, MarketDataTool] = {}

_PERSONA = """\
## Role

You are CFM Agent v2.0, a disciplined probabilistic market forecaster. You
combine cutoff-safe structured data, ARIMA, Kalman, LightGBM, their transparent
ensemble, verified web evidence, and optional sandboxed analysis.

## Operating principles

- The prompt contains identifiers and task metadata, not raw price history.
- Retrieve structured data and all standard model results through
  `query_market_data` exactly once per forecast.
- Copy component-model values exactly into the rich response. Never alter or
  invent a component result.
- Use the deterministic ensemble as the default center. Quantify and explain any
  final adjustment from its median.
- Explain model disagreement, failures, active weights, evidence, uncertainty,
  and limitations with concise auditable rationale.
- Use only verifier-approved web evidence and cite its URL.
- E2B is optional and must not recreate the standard models.
- Call `set_model_response` exactly once for structured tasks.\
"""


def build_cfm_agent_config(
    model: str = LITE_MODEL,
    *,
    data_service: DataService | None = None,
    settings: CfmAgentSettings = DEFAULT_SETTINGS,
    search_model: str = LITE_MODEL,
    covariate_series_ids: list[str] | None = None,
) -> AgentConfig:
    """Build the fully enabled ``cfm_agent_v_2_0`` configuration."""
    service = data_service or build_wti_multivariate_service()
    requested_covariates = (
        list(covariate_series_ids)
        if covariate_series_ids is not None
        else [series_id for series_id in DEFAULT_WTI_COVARIATE_SERIES_IDS if series_id in set(service.series_ids)]
    )
    market_tool = MarketDataTool(
        service,
        settings=settings,
        covariate_series_ids=requested_covariates,
    )
    skill_names = ["forecasting", "model-selection", "research", "code-analysis"]
    skill_dirs: list[Path] = [SKILLS_ROOT / name for name in skill_names]
    config = AgentConfig(
        name=AGENT_NAME,
        description=(
            "Verified-research, E2B-enabled probabilistic market forecaster with "
            "rich model attribution and training-data audit."
        ),
        model=model,
        instruction=_PERSONA,
        max_output_tokens=24_576,
        context_retrieval=build_verified_search_config(search_model=search_model),
        code_execution=build_code_execution_config(),
        function_tools=[market_tool.as_function_tool()],
        skills_dirs=skill_dirs,
    )
    _CONFIG_TO_MARKET_TOOL[id(config)] = market_tool
    return config


def build_cfm_agent_predictor(
    config: AgentConfig,
    *,
    settings: CfmAgentSettings = DEFAULT_SETTINGS,
) -> CfmRichAgentPredictor:
    """Build the rich-output predictor for a config created by this package."""
    del settings
    market_tool = _CONFIG_TO_MARKET_TOOL.get(id(config))
    if market_tool is None:
        raise ValueError(
            "The configuration was not created by build_cfm_agent_config in this "
            "process; its MarketDataTool cannot be recovered."
        )
    return CfmRichAgentPredictor(config, market_tool)


def __getattr__(name: str) -> Any:
    """Expose a fully enabled root agent lazily for ADK interactive use."""
    if name == "root_agent":
        return build_adk_agent(build_cfm_agent_config())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["build_cfm_agent_config", "build_cfm_agent_predictor"]
