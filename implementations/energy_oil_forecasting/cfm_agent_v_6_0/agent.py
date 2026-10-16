"""Self-contained CFM Agent v6.0 configuration.

Builds the ADK ``AgentConfig`` for the main assessment role from settings
(Section 4.3), wires the three package tools, and hands the configuration to the
predictor. The main model object is constructed here by the package, not by the
shared factory, so that every generation parameter in ``[llm.main]`` and
``[llm.common]`` reaches the invocation layer.
"""

from __future__ import annotations

from typing import Any

from aieng.forecasting.data import DataService
from aieng.forecasting.methods.agentic import AgentConfig, build_adk_agent
from energy_oil_forecasting.cfm_agent_v_6_0.config import (
    AGENT_NAME,
    DEFAULT_SETTINGS,
    SKILLS_ROOT,
    CfmV60Settings,
)
from energy_oil_forecasting.cfm_agent_v_6_0.llm_roles import main_agent_config_kwargs
from energy_oil_forecasting.cfm_agent_v_6_0.predictor import CfmV60Predictor
from energy_oil_forecasting.cfm_agent_v_6_0.tools import (
    AuditedCodeExecutionTool,
    AuthoritativeSuiteTool,
    ResearchPipelineTool,
)
from energy_oil_forecasting.data import build_wti_multivariate_service


SKILL_NAMES: tuple[str, ...] = (
    "source-selection",
    "research-planning",
    "claim-building",
    "action-proposal",
    "code-analysis",
)
AGENT_DESCRIPTION = "Task-bound, cutoff-verified, policy-controlled WTI forecaster with a Python-owned numerical suite."

_CONFIG: dict[
    int,
    tuple[
        AgentConfig,
        AuthoritativeSuiteTool,
        ResearchPipelineTool,
        AuditedCodeExecutionTool,
        CfmV60Settings,
    ],
] = {}

_PERSONA = """You are CFM Agent v6.0, a WTI crude oil forecasting analyst. Python owns task binding, the numerical model suite, source identity, evidence policy, the forecast transformation, and all final arithmetic. You assess evidence and propose categorical actions only.

Roles. Five separately configured LLM roles exist: you are the main assessment role. The research search, the cutoff verifier, the structured-output correction, and the claim-support verifier are separate calls that you do not control and cannot address.

Workflow. Load source-selection before research-planning and claim-building. Call run_authoritative_suite exactly once using exactly the task target, horizons, frequency, and cutoff. Construct exactly four neutral queries and call run_research_pipeline exactly once. Then build claims, assess novelty, propose actions, and call set_model_response exactly once.

Evidence. Only accepted, verifier-cleaned summaries in the returned active packet are substantive evidence. Each summary is split into numbered passages with ids of the form "SS.NN". Every claim must cite the accepted verified-summary ids, the specific passage ids that state the fact, and provenance-correct associated source ids, preferring the strongest sources under source-selection. Never use unfiltered search text, removed claims, destination-page content, audit-only findings, or model memory as factual evidence. Do not invent or alter summaries, passages, ids, URLs, domains, publishers, or verification results.

Diagnostics. The market diagnostics returned by run_authoritative_suite (returns, realised volatility, z-scores, drawdowns, jumps, curve state) are informational context for your assessment. They are not evidence: they cannot satisfy any evidence tier, and an action that rests on diagnostics alone will be rejected by the evidence policy.

Novelty. Classify incremental_novelty as one of likely_new_relative_to_model_data, possibly_partly_reflected, likely_reflected_in_model_data, or indeterminate. All four classes act: Python scales the centre adjustment by a settings-defined multiplier for the class, so the class changes the size of an action, never whether it applies.

Code execution. run_code is optional and diagnostics-only. Plan the whole diagnostic before calling: exactly one accepted call is available, so submit one self-contained script with a stated purpose that reads the preamble names DIAGNOSTICS, MODEL_SUITE_QUANTILES, and TASK_BINDING. Prohibited: forecasting or recomputing the model suite, recomputing or verifying task-bound values (cutoff date, target series, horizons, frequency: treat them as authoritative), evidence decisions, forecast transformations, final pricing, filesystem, package, environment, skill, or runtime discovery, and network access. A rejected request names every failed rule and counts against a separate budget.

Output. Propose only the exact categorical values in action-proposal. Do not calculate prices, quantiles, multipliers, coverages, or adjustments. Audit-only components cannot affect the forecast. Call set_model_response exactly once."""


def _apply_overrides(settings: CfmV60Settings, **updates: str | None) -> CfmV60Settings:
    """Re-validate settings with the coach-compatibility model overrides applied."""
    changes = {name: value for name, value in updates.items() if value is not None}
    if not changes:
        return settings
    return CfmV60Settings.model_validate(settings.model_dump() | changes)


def build_cfm_agent_config(
    *,
    data_service: DataService | None = None,
    settings: CfmV60Settings = DEFAULT_SETTINGS,
    covariate_series_ids: list[str] | None = None,
    model: str | None = None,
    search_model: str | None = None,
) -> AgentConfig:
    """Build the agent configuration; ``model`` and ``search_model`` exist for coach compatibility only.

    When given, they replace ``main_model`` / ``search_model`` in a re-validated copy
    of ``settings``. Every other parameter comes from settings.
    """
    settings = _apply_overrides(settings, main_model=model, search_model=search_model)
    service = data_service or build_wti_multivariate_service()
    covariates = list(covariate_series_ids) if covariate_series_ids is not None else list(settings.covariate_series_ids)
    market = AuthoritativeSuiteTool(service, settings=settings, covariate_series_ids=covariates)
    research = ResearchPipelineTool(settings)
    code_execution = AuditedCodeExecutionTool(
        settings, payload_provider=getattr(market, "code_execution_payload", None)
    )
    tools = [market.as_function_tool(), research.as_function_tool()]
    if settings.code_execution_enabled:
        tools.append(code_execution.as_function_tool())
    config = AgentConfig(
        name=AGENT_NAME,
        description=AGENT_DESCRIPTION,
        instruction=_PERSONA,
        function_tools=tools,
        skills_dirs=[SKILLS_ROOT / name for name in SKILL_NAMES],
        **main_agent_config_kwargs(settings),
    )
    _CONFIG[id(config)] = (config, market, research, code_execution, settings)
    return config


def build_cfm_agent_predictor(config: AgentConfig) -> CfmV60Predictor:
    bundle = _CONFIG.pop(id(config), None)
    if bundle is None or bundle[0] is not config:
        raise ValueError("Configuration was not created here or was already consumed.")
    _, market, research, code_execution, settings = bundle
    return CfmV60Predictor(config, market, research, code_execution, settings)


def __getattr__(name: str) -> Any:
    if name == "root_agent":
        return build_adk_agent(build_cfm_agent_config())
    raise AttributeError(name)


__all__ = [
    "AGENT_DESCRIPTION",
    "SKILL_NAMES",
    "build_cfm_agent_config",
    "build_cfm_agent_predictor",
]
