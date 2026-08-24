"""CFM Agent v5.2.2 Delta-Governed: ARIMA-only ensemble + real-history-governed LLM actions.

Builds on CFM Agent v5.2.2 (ARIMA-only numerical suite — see
``cfm_agent_v_5_2.tools.market_data_arima_only``) but replaces the fixed
fraction-of-ensemble-width rule that turns the LLM's categorical action into
a dollar adjustment with something grounded in actual market history: the
LLM proposes a discrete rank in ``{-2,-1,0,1,2}``, which Python maps to a
target percentile of WTI's REAL historical h-day price-move distribution
(``delta_distribution.py``) — never a number the LLM invents. Same
"Python owns the arithmetic, LLM proposes only categorical actions"
discipline as CFM v5.2, just with a better-grounded governor for what those
categories mean in dollar terms.

Does not modify ``cfm_agent_v_5_2`` in any way — every file in this package
is new.
"""

from pathlib import Path
from typing import Any

from aieng.forecasting.data import DataService
from aieng.forecasting.methods.agentic import AgentConfig, build_adk_agent
from aieng.forecasting.models import LITE_MODEL
from energy_oil_forecasting.cfm_agent_v_5_2.config import (
    AGENT_NAME,
    DEFAULT_SETTINGS,
    SKILLS_ROOT,
    CfmV52Settings,
)
from energy_oil_forecasting.cfm_agent_v_5_2.tools import (
    AuditedCodeExecutionTool,
    ResearchPipelineTool,
)
from energy_oil_forecasting.cfm_agent_v_5_2.tools.market_data_arima_only import (
    AuthoritativeSuiteToolArimaOnly,
)
from energy_oil_forecasting.cfm_agent_v_5_2_2_delta_governed.predictor import (
    CfmDeltaGovernedPredictor,
)
from energy_oil_forecasting.data import (
    DEFAULT_WTI_COVARIATE_SERIES_IDS,
    build_wti_multivariate_service,
)


_LOCAL_SKILLS_ROOT = Path(__file__).resolve().parent / "skills"


_CONFIG: dict[
    int,
    tuple[
        AgentConfig,
        AuthoritativeSuiteToolArimaOnly,
        ResearchPipelineTool,
        AuditedCodeExecutionTool,
        CfmV52Settings,
    ],
] = {}

_PERSONA = (
    "You are CFM Agent v5.2.2 Delta-Governed. Python owns task binding, source identity, evidence policy, "
    "and final arithmetic. Load source-selection before research-planning and claim-building. Run the "
    "authoritative numerical suite once using exactly the task target, horizons, frequency, and cutoff — "
    "this suite runs ARIMA only. Run the package-local research pipeline once using exactly four neutral "
    "queries. Only main-verifier-cleaned summaries are substantive research evidence. Claims must cite "
    "accepted verified-summary IDs and provenance-correct associated source IDs, preferring the strongest "
    "sources under source-selection. Never use unfiltered text, removed claims, destination-page content, "
    "or model memory as factual evidence. Propose only categorical actions: your center_action is a "
    "discrete rank in {-2,-1,0,1,2}, not a price or percentage — Python converts your rank into a dollar "
    "adjustment using the real historical distribution of WTI price moves, never a number you compute "
    "yourself. Optional run_code use is diagnostics-only and must state its purpose. Do not use run_code "
    "to verify, recompute, or reinterpret task-bound values already supplied by Python, including the "
    "cutoff date, target series, horizons, and frequency; treat those values as authoritative. Audit-only "
    "components cannot affect Components #9 or #10. Call set_model_response exactly once."
)


def build_cfm_agent_config_delta_governed(
    model: str = LITE_MODEL,
    *,
    data_service: DataService | None = None,
    settings: CfmV52Settings = DEFAULT_SETTINGS,
    search_model: str = LITE_MODEL,
    covariate_series_ids: list[str] | None = None,
) -> AgentConfig:
    """Build CFM Agent v5.2.2 Delta-Governed config.

    Same ARIMA-only numerical suite as CFM Agent v5.2.2. The LLM's output
    schema differs (``center_action`` is a discrete rank, not a named
    category) and this config must be paired with
    :func:`build_cfm_agent_predictor_delta_governed` — NOT
    ``build_cfm_agent_predictor``/``build_cfm_agent_predictor_arima_only`` —
    since that's what wires in the history-governed policy and forecast
    engine that know how to interpret the rank.
    """
    service = data_service or build_wti_multivariate_service()
    available = set(service.series_ids)
    covariates = (
        list(covariate_series_ids)
        if covariate_series_ids is not None
        else [value for value in DEFAULT_WTI_COVARIATE_SERIES_IDS if value in available]
    )
    market = AuthoritativeSuiteToolArimaOnly(service, settings=settings, covariate_series_ids=covariates)
    research = ResearchPipelineTool(settings, search_model=search_model)
    code_execution = AuditedCodeExecutionTool(settings)
    tools = [market.as_function_tool(), research.as_function_tool()]
    if settings.code_execution_enabled:
        tools.append(code_execution.as_function_tool())
    config = AgentConfig(
        name="cfm_agent_v_5_2_2_delta_governed",
        description="ARIMA-only CFM Agent whose LLM actions are governed by real historical price-move percentiles.",
        model=model,
        instruction=_PERSONA,
        max_output_tokens=24576,
        function_tools=tools,
        skills_dirs=[
            # Shared with cfm_agent_v_5_2 -- none of these reference center_action's
            # format, so they apply unchanged.
            *(SKILLS_ROOT / name for name in ["source-selection", "research-planning", "claim-building"]),
            # Local override: the shared action-proposal skill describes v5.2's
            # string-category center_action, which contradicts this agent's
            # integer-rank schema. See skills/action-proposal/SKILL.md here.
            _LOCAL_SKILLS_ROOT / "action-proposal",
            SKILLS_ROOT / "code-analysis",
        ],
    )
    _CONFIG[id(config)] = (config, market, research, code_execution, settings)
    return config


def build_cfm_agent_predictor_delta_governed(config: AgentConfig) -> CfmDeltaGovernedPredictor:
    bundle = _CONFIG.pop(id(config), None)
    if bundle is None or bundle[0] is not config:
        raise ValueError("Configuration was not created here or was already consumed.")
    _, market, research, code_execution, settings = bundle
    return CfmDeltaGovernedPredictor(config, market, research, code_execution, settings)


def __getattr__(name: str) -> Any:
    if name == "root_agent":
        return build_adk_agent(build_cfm_agent_config_delta_governed())
    raise AttributeError(name)


__all__ = [
    "AGENT_NAME",
    "build_cfm_agent_config_delta_governed",
    "build_cfm_agent_predictor_delta_governed",
]
