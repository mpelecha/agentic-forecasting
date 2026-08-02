"""Self-contained CFM Agent v3.3 configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aieng.forecasting.data import DataService
from aieng.forecasting.methods.agentic import AgentConfig, build_adk_agent
from aieng.forecasting.models import LITE_MODEL
from energy_oil_forecasting.cfm_agent_v_3_3.config import (
    AGENT_NAME,
    DEFAULT_SETTINGS,
    SKILLS_ROOT,
    CfmV33Settings,
)
from energy_oil_forecasting.cfm_agent_v_3_3.predictor import CfmV33Predictor
from energy_oil_forecasting.cfm_agent_v_3_3.tools import (
    AuthoritativeSuiteTool,
    build_code_execution_config,
    build_verified_search_config,
)
from energy_oil_forecasting.data import (
    DEFAULT_WTI_COVARIATE_SERIES_IDS,
    build_wti_multivariate_service,
)


_CONFIG_TO_MARKET_TOOL: dict[int, AuthoritativeSuiteTool] = {}
_CONFIG_TO_SETTINGS: dict[int, CfmV33Settings] = {}

_PERSONA = """\
### Role
You are CFM Agent v3.3, a disciplined market-context assessment agent.
Python owns all final forecast arithmetic. You never author a final price,
quantile grid, dollar adjustment, or uncertainty multiplier.

### Required operating procedure
1. Load evidence-assessment, market-context-assessment, and
   forecast-action-selection.
2. Call run_authoritative_suite exactly once with all required arguments. Treat its component models, ensemble,
   diagnostics, dates, and sample count as authoritative.
3. Execute the fixed multi-query research plan. One eligible high-quality publisher may support only a small bounded action; independent corroboration is required for greater authority. Broker notes and trading websites cannot establish material facts by themselves.
4. Separate confirmed facts from source interpretation and your own assessment.
5. Record material claims with source IDs, support status, and contradictions.
6. Do not claim that information is "already priced." Assess only whether it is
   likely new relative to the latest model data, partly reflected, reflected,
   or indeterminate.
7. No-change is the default. A non-neutral action requires direct eligible evidence and a clear transmission mechanism. Python assigns limited, corroborated, or strong evidence authority and may cap or reject the action.
8. Select only the authorized center, uncertainty, and persistence categories.
9. E2B is optional and diagnostic only; it must not recreate the model suite or
   calculate the final forecast.
10. Call set_model_response exactly once.
"""


def build_cfm_agent_config(
    model: str = LITE_MODEL,
    *,
    data_service: DataService | None = None,
    settings: CfmV33Settings = DEFAULT_SETTINGS,
    search_model: str = LITE_MODEL,
    covariate_series_ids: list[str] | None = None,
) -> AgentConfig:
    service = data_service or build_wti_multivariate_service()
    requested_covariates = (
        list(covariate_series_ids)
        if covariate_series_ids is not None
        else [series_id for series_id in DEFAULT_WTI_COVARIATE_SERIES_IDS if series_id in set(service.series_ids)]
    )
    market_tool = AuthoritativeSuiteTool(
        service,
        settings=settings,
        covariate_series_ids=requested_covariates,
    )
    skill_names = [
        "evidence-assessment",
        "market-context-assessment",
        "forecast-action-selection",
        "code-analysis",
    ]
    skill_dirs: list[Path] = [SKILLS_ROOT / name for name in skill_names]
    config = AgentConfig(
        name=AGENT_NAME,
        description=(
            "Policy-controlled WTI forecaster: stronger corroborated research, "
            "structured contextual actions, and Python-owned final arithmetic."
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
    _CONFIG_TO_SETTINGS[id(config)] = settings
    return config


def build_cfm_agent_predictor(config: AgentConfig) -> CfmV33Predictor:
    market_tool = _CONFIG_TO_MARKET_TOOL.get(id(config))
    settings = _CONFIG_TO_SETTINGS.get(id(config))
    if market_tool is None or settings is None:
        raise ValueError("The configuration was not created by build_cfm_agent_config in this process.")
    return CfmV33Predictor(config, market_tool, settings)


def __getattr__(name: str) -> Any:
    if name == "root_agent":
        return build_adk_agent(build_cfm_agent_config())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["build_cfm_agent_config", "build_cfm_agent_predictor"]
