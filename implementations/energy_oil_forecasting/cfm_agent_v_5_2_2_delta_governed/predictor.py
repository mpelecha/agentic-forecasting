"""ARIMA-only, real-history-governed CFM predictor variant.

Identical control flow to
:class:`energy_oil_forecasting.cfm_agent_v_5_2.predictor.CfmV51Predictor`
(audit trails, validation, placeholder resolution) — only the pieces that
decide *how much* the LLM's proposal moves the forecast are swapped: the
LLM's ``center_action`` is a discrete rank ``{-2,-1,0,1,2}`` instead of a
named category, and both the center-shift and the uncertainty-width are
governed by the empirical distribution of real historical h-day WTI price
deltas (see ``delta_distribution.py``) rather than by the numerical
ensemble's own self-reported quantile width.

``predict()`` overrides only enough to compute and stash that empirical
distribution before delegating to the parent's well-tested ``predict()`` —
everything else (evidence-tier gating call sites, audit metadata
construction) is reused unchanged via inheritance.
"""

from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import Prediction
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic import AgentConfig, AgentPredictor
from energy_oil_forecasting.cfm_agent_v_5_2.config import CfmV52Settings
from energy_oil_forecasting.cfm_agent_v_5_2.predictor import CfmV51Predictor
from energy_oil_forecasting.cfm_agent_v_5_2.tools import AuditedCodeExecutionTool, ResearchPipelineTool
from energy_oil_forecasting.cfm_agent_v_5_2.tools.market_data_arima_only import AuthoritativeSuiteToolArimaOnly
from energy_oil_forecasting.cfm_agent_v_5_2_2_delta_governed.forecast_engine import PythonForecastEngineDeltaGoverned
from energy_oil_forecasting.cfm_agent_v_5_2_2_delta_governed.outputs import CfmContextAssessmentOutputDeltaGoverned
from energy_oil_forecasting.cfm_agent_v_5_2_2_delta_governed.policy import (
    EnsembleLockedPolicyDeltaGoverned,
    EvidencePolicyDeltaGoverned,
)
from energy_oil_forecasting.cfm_agent_v_5_2_2_delta_governed.prompts import CfmDeltaGovernedPromptBuilder
from energy_oil_forecasting.cfm_agent_v_5_2_2_delta_governed.response_control import (
    StructuredAssessmentControllerDeltaGoverned,
)


class CfmDeltaGovernedPredictor(CfmV51Predictor):
    def __init__(
        self,
        config: AgentConfig,
        market: AuthoritativeSuiteToolArimaOnly,
        research: ResearchPipelineTool,
        code_execution: AuditedCodeExecutionTool,
        settings: CfmV52Settings,
    ):
        self.market = market
        self.research = research
        self.code_execution = code_execution
        self.settings = settings
        self.policy = (
            EnsembleLockedPolicyDeltaGoverned(settings)
            if settings.policy_mode == "ensemble_locked"
            else EvidencePolicyDeltaGoverned(settings)
        )
        self.engine = PythonForecastEngineDeltaGoverned(settings)
        self.inner = AgentPredictor(
            agent_config=config,
            prompt_builder=CfmDeltaGovernedPromptBuilder(),
            output_schema=CfmContextAssessmentOutputDeltaGoverned,
        )
        self.assessment_controller = StructuredAssessmentControllerDeltaGoverned(self.inner, settings)

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        self.engine.prepare(context, task)
        return super().predict(task, context)


__all__ = ["CfmDeltaGovernedPredictor"]
