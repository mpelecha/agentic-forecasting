"""Self-contained CFM Agent v5.2 configuration."""

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
from energy_oil_forecasting.cfm_agent_v_5_2.predictor import CfmV51Predictor
from energy_oil_forecasting.cfm_agent_v_5_2.tools import (
    AuditedCodeExecutionTool,
    AuthoritativeSuiteTool,
    ResearchPipelineTool,
)
from energy_oil_forecasting.cfm_agent_v_5_2.tools.market_data_arima_only import (
    AuthoritativeSuiteToolArimaOnly,
)
from energy_oil_forecasting.data import (
    DEFAULT_WTI_COVARIATE_SERIES_IDS,
    build_wti_multivariate_service,
)


_CONFIG: dict[
    int,
    tuple[
        AgentConfig,
        AuthoritativeSuiteTool,
        ResearchPipelineTool,
        AuditedCodeExecutionTool,
        CfmV52Settings,
    ],
] = {}
_PERSONA = """You are CFM Agent v5.2. Python owns task binding, source identity, evidence policy, and final arithmetic. Load source-selection before research-planning and claim-building. Run the authoritative numerical suite once using exactly the task target, horizons, frequency, and cutoff. Run the package-local research pipeline once using exactly four neutral queries. Only main-verifier-cleaned summaries are substantive research evidence. Claims must cite accepted verified-summary IDs and provenance-correct associated source IDs, preferring the strongest sources under source-selection. Never use unfiltered text, removed claims, destination-page content, or model memory as factual evidence. Propose only categorical actions. Optional run_code use is diagnostics-only and must state its purpose. Do not use run_code to verify, recompute, or reinterpret task-bound values already supplied by Python, including the cutoff date, target series, horizons, and frequency; treat those values as authoritative. Audit-only components cannot affect Components #9 or #10. Call set_model_response exactly once."""


def build_cfm_agent_config(
    model: str = LITE_MODEL,
    *,
    data_service: DataService | None = None,
    settings: CfmV52Settings = DEFAULT_SETTINGS,
    search_model: str = LITE_MODEL,
    covariate_series_ids: list[str] | None = None,
) -> AgentConfig:
    service = data_service or build_wti_multivariate_service()
    available = set(service.series_ids)
    covariates = (
        list(covariate_series_ids)
        if covariate_series_ids is not None
        else [value for value in DEFAULT_WTI_COVARIATE_SERIES_IDS if value in available]
    )
    market = AuthoritativeSuiteTool(service, settings=settings, covariate_series_ids=covariates)
    research = ResearchPipelineTool(settings, search_model=search_model)
    code_execution = AuditedCodeExecutionTool(settings)
    tools = [market.as_function_tool(), research.as_function_tool()]
    if settings.code_execution_enabled:
        tools.append(code_execution.as_function_tool())
    config = AgentConfig(
        name=AGENT_NAME,
        description="Task-bound, cutoff-verified-summary, policy-controlled WTI forecaster.",
        model=model,
        instruction=_PERSONA,
        max_output_tokens=24576,
        function_tools=tools,
        skills_dirs=[
            SKILLS_ROOT / name
            for name in [
                "source-selection",
                "research-planning",
                "claim-building",
                "action-proposal",
                "code-analysis",
            ]
        ],
    )
    _CONFIG[id(config)] = (config, market, research, code_execution, settings)
    return config


def build_cfm_agent_predictor(config: AgentConfig) -> CfmV51Predictor:
    bundle = _CONFIG.pop(id(config), None)
    if bundle is None or bundle[0] is not config:
        raise ValueError("Configuration was not created here or was already consumed.")
    _, market, research, code_execution, settings = bundle
    return CfmV51Predictor(config, market, research, code_execution, settings)


def build_cfm_agent_config_arima_only(
    model: str = LITE_MODEL,
    *,
    data_service: DataService | None = None,
    settings: CfmV52Settings = DEFAULT_SETTINGS,
    search_model: str = LITE_MODEL,
    covariate_series_ids: list[str] | None = None,
) -> AgentConfig:
    """Build CFM Agent v5.2 config with ONLY ARIMA in the ensemble (no Kalman, no LightGBM).

    Identical to build_cfm_agent_config() except uses AuthoritativeSuiteToolArimaOnly,
    which skips the expensive Kalman and LightGBM models. Saves hours of computation
    while keeping the full LLM agent, policy, and research components.
    """
    service = data_service or build_wti_multivariate_service()
    available = set(service.series_ids)
    covariates = (
        list(covariate_series_ids)
        if covariate_series_ids is not None
        else [value for value in DEFAULT_WTI_COVARIATE_SERIES_IDS if value in available]
    )
    # Use ARIMA-only variant instead of full ensemble
    market = AuthoritativeSuiteToolArimaOnly(service, settings=settings, covariate_series_ids=covariates)
    research = ResearchPipelineTool(settings, search_model=search_model)
    code_execution = AuditedCodeExecutionTool(settings)
    tools = [market.as_function_tool(), research.as_function_tool()]
    if settings.code_execution_enabled:
        tools.append(code_execution.as_function_tool())
    config = AgentConfig(
        name="cfm_agent_v_5_2_arima_only",
        description="CFM Agent v5.2 with ONLY ARIMA ensemble (no Kalman/LightGBM).",
        model=model,
        instruction=_PERSONA,
        max_output_tokens=24576,
        function_tools=tools,
        skills_dirs=[
            SKILLS_ROOT / name
            for name in [
                "source-selection",
                "research-planning",
                "claim-building",
                "action-proposal",
                "code-analysis",
            ]
        ],
    )
    _CONFIG[id(config)] = (config, market, research, code_execution, settings)
    return config


def build_cfm_agent_predictor_arima_only(config: AgentConfig) -> CfmV51Predictor:
    """Build predictor for ARIMA-only CFM Agent config (companion to build_cfm_agent_config_arima_only)."""
    bundle = _CONFIG.pop(id(config), None)
    if bundle is None or bundle[0] is not config:
        raise ValueError("Configuration was not created here or was already consumed.")
    _, market, research, code_execution, settings = bundle
    return CfmV51Predictor(config, market, research, code_execution, settings)


def __getattr__(name: str) -> Any:
    if name == "root_agent":
        return build_adk_agent(build_cfm_agent_config())
    raise AttributeError(name)
