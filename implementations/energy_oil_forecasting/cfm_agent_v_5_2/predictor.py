"""End-to-end v5.1 controller with task binding and isolated audits."""

from typing import Any

import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import ContinuousForecast, Prediction
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic import AgentConfig, AgentPredictor
from energy_oil_forecasting.cfm_agent_v_5_2.claim_support_verifier import (
    audit_claim_support,
)
from energy_oil_forecasting.cfm_agent_v_5_2.config import CfmV52Settings
from energy_oil_forecasting.cfm_agent_v_5_2.forecast_engine import PythonForecastEngine
from energy_oil_forecasting.cfm_agent_v_5_2.outputs import CfmContextAssessmentOutput
from energy_oil_forecasting.cfm_agent_v_5_2.policy import (
    EnsembleLockedPolicy,
    EvidencePolicy,
)
from energy_oil_forecasting.cfm_agent_v_5_2.prompts import CfmV51PromptBuilder
from energy_oil_forecasting.cfm_agent_v_5_2.response_control import (
    StructuredAssessmentController,
)
from energy_oil_forecasting.cfm_agent_v_5_2.tools import (
    AuditedCodeExecutionTool,
    AuthoritativeSuiteTool,
    ResearchPipelineTool,
)


class CfmV51Predictor(Predictor):
    def __init__(
        self,
        config: AgentConfig,
        market: AuthoritativeSuiteTool,
        research: ResearchPipelineTool,
        code_execution: AuditedCodeExecutionTool,
        settings: CfmV52Settings,
    ):
        self.market = market
        self.research = research
        self.code_execution = code_execution
        self.settings = settings
        self.policy = (
            EnsembleLockedPolicy(settings) if settings.policy_mode == "ensemble_locked" else EvidencePolicy(settings)
        )
        self.engine = PythonForecastEngine(settings)
        self.inner = AgentPredictor(
            agent_config=config,
            prompt_builder=CfmV51PromptBuilder(),
            output_schema=CfmContextAssessmentOutput,
        )
        self.assessment_controller = StructuredAssessmentController(self.inner, settings)

    @property
    def predictor_id(self):
        return self.inner.predictor_id

    @property
    def _runner(self) -> Any:
        return self.inner._runner

    def predict(  # noqa: PLR0912, PLR0915
        self, task: ForecastingTask, context: ForecastContext
    ) -> list[Prediction]:
        cutoff = str(pd.Timestamp(context.as_of).date())
        expected_horizons = [int(horizon) for horizon in task.horizons]
        if expected_horizons != sorted(set(expected_horizons)):
            raise RuntimeError("task horizons must be strictly increasing and unique")
        self.market.prepare_workflow(task, context)
        self.research.prepare_workflow(cutoff)
        self.code_execution.prepare_workflow()
        # The initial agent turn executes both one-shot tools. Resolve the raw
        # structured response only after the active research packet exists.
        first_raw = self.assessment_controller._initial_response(task, context)
        result = self.market.last_result
        packet = self.research.last_result
        if result is None or result.model_suite is None or result.model_suite.ensemble is None:
            raise RuntimeError("one authoritative suite result required")
        if packet is None:
            raise RuntimeError("one research pipeline result required")
        research_audit = self.research.audit
        if research_audit["successful_execution_count"] != 1:
            raise RuntimeError("exactly one successful research execution is required")
        suite = result.model_suite
        if suite.cutoff_date != cutoff or packet.cutoff_date != cutoff:
            raise RuntimeError("numerical and research cutoffs must equal the harness-controlled cutoff")
        if suite.target_series_id != task.target_series_id or suite.frequency != task.frequency:
            raise RuntimeError("numerical-suite target or frequency does not match the task")
        if suite.horizons != expected_horizons:
            raise RuntimeError("numerical-suite horizons do not exactly match the task")
        if [item.horizon for item in suite.ensemble.forecasts] != expected_horizons:
            raise RuntimeError("ensemble horizons do not exactly match the task")
        # Validate the initial raw response, perform at most one correction-only
        # LLM retry, and fall back to a Python-owned neutral assessment if needed.
        resolution = self.assessment_controller.resolve(
            task=task,
            context=context,
            packet_id=packet.packet_id,
            initial_raw_response=first_raw,
        )
        placeholders = resolution.predictions
        assessment = resolution.assessment
        structured_output_audit = resolution.audit
        if len(placeholders) != len(expected_horizons):
            raise RuntimeError("placeholder count does not match task horizons")
        if self.settings.audit_enabled:
            findings, support_audit = audit_claim_support(assessment, packet, self.settings)
        else:
            findings = []
            support_audit = {
                "verifier_id": "claim_support_verifier_v52_audit_only",
                "audit_only": True,
                "executed": False,
                "completed": False,
                "reason": "audit switch is off",
                "findings": [],
            }
        source_bundle = self.research.last_source_audit
        source_audit = source_bundle.model_dump(mode="json")
        ensemble = {item.horizon: item for item in suite.ensemble.forecasts}
        decisions = {horizon: self.policy.apply(assessment, packet, horizon) for horizon in expected_horizons}
        transforms = {
            horizon: self.engine.transform(
                ensemble[horizon],
                decisions[horizon],
                assessment.incremental_novelty,
                suite.diagnostics.latest_value,
            )
            for horizon in expected_horizons
        }
        by_date = {str(value.forecast_date.date()): value for value in placeholders}
        numerical_audit = {
            "model_suite": suite.model_dump(mode="json"),
            "execution": self.market.audit,
        }
        for item in suite.ensemble.forecasts:
            prediction = by_date.get(item.forecast_date)
            if prediction is None:
                raise RuntimeError(f"missing placeholder for forecast date {item.forecast_date}")
            transformation = transforms[item.horizon]
            prediction.payload = ContinuousForecast(
                point_forecast=transformation.final_point_forecast,
                quantiles=transformation.final_quantiles,
            )
            prediction.metadata.update(
                {
                    "model_suite_id": suite.model_suite_id,
                    "numerical_suite_audit": numerical_audit,
                    "research_execution_audit": research_audit,
                    "code_execution_audit": self.code_execution.audit,
                    "llm_context_assessment": assessment.model_dump(mode="json"),
                    "structured_output_control_audit": structured_output_audit,
                    "active_research_packet": packet.model_dump(mode="json"),
                    "cutoff_verification_audit": self.research.last_verification_audit.model_dump(mode="json"),
                    "source_validation_audit": source_audit,
                    "claim_support_audit": support_audit,
                    "claim_support_findings": [finding.model_dump(mode="json") for finding in findings],
                    "policy_decision": decisions[item.horizon].model_dump(mode="json"),
                    "forecast_transformation": transformation.model_dump(mode="json"),
                    "unadjusted_ensemble": {
                        "point_forecast": item.point_forecast,
                        "quantiles": item.quantiles,
                    },
                    "market_diagnostics": suite.diagnostics.model_dump(mode="json"),
                    "task_binding": {
                        "cutoff": cutoff,
                        "target_series_id": task.target_series_id,
                        "horizons": expected_horizons,
                        "frequency": task.frequency,
                        "validated": True,
                    },
                    "audit_switch": self.settings.audit_enabled,
                    "audit_only_controls": {
                        "source_validator_executed": source_bundle.executed,
                        "claim_support_verifier_executed": bool(support_audit.get("executed")),
                        "claim_support_verifier_completed": bool(support_audit.get("completed")),
                    },
                    "active_evidence_declaration": (
                        "The LLM used the authoritative numerical-suite output and main-verifier-cleaned summaries "
                        "with mechanically resolved source identities. Destination-page and audit-only content were "
                        "not exposed. Optional diagnostic code use is separately declared in code_execution_audit."
                    ),
                    "audit_findings_changed_forecast": False,
                    "llm_authored_final_numbers": False,
                }
            )
        return placeholders
