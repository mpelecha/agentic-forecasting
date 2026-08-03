"""Policy-controlled predictor for CFM Agent v3.4."""

from __future__ import annotations

from typing import Any

from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import ContinuousForecast, Prediction
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic import AgentConfig, AgentPredictor
from energy_oil_forecasting.cfm_agent_v_3_4.config import CfmV34Settings
from energy_oil_forecasting.cfm_agent_v_3_4.outputs import CfmContextAssessmentOutput
from energy_oil_forecasting.cfm_agent_v_3_4.policy import ConstrainedActionPolicy, EnsembleLockedPolicy
from energy_oil_forecasting.cfm_agent_v_3_4.prompts import CfmV34PromptBuilder
from energy_oil_forecasting.cfm_agent_v_3_4.sanitizer import sanitize_assessment
from energy_oil_forecasting.cfm_agent_v_3_4.schemas import ModelForecastResult, PolicyDecision
from energy_oil_forecasting.cfm_agent_v_3_4.tools import AuthoritativeSuiteTool


class CfmV34Predictor(Predictor):
    """Run the assessment agent, then let Python approve actions and construct every final number."""

    def __init__(
        self,
        config: AgentConfig,
        market_data_tool: AuthoritativeSuiteTool,
        settings: CfmV34Settings,
    ) -> None:
        self._market_data_tool = market_data_tool
        self._settings = settings
        self._policy = (
            EnsembleLockedPolicy(settings)
            if settings.policy_mode == "ensemble_locked"
            else ConstrainedActionPolicy(settings)
        )
        self._inner = AgentPredictor(
            agent_config=config,
            prompt_builder=CfmV34PromptBuilder(),
            output_schema=CfmContextAssessmentOutput,
        )

    @property
    def predictor_id(self) -> str:
        return self._inner.predictor_id

    @property
    def _runner(self) -> Any:
        return self._inner._runner

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        placeholders = self._inner.predict(task, context)
        result = self._market_data_tool.last_result
        if result is None or result.model_suite is None or result.model_suite.ensemble is None:
            raise RuntimeError("CFM v3.4 requires one successful authoritative numerical-suite result.")
        suite = result.model_suite
        if suite.horizons != sorted(task.horizons):
            raise RuntimeError("Numerical-suite horizons do not match the requested task.")

        assessment_raw = placeholders[0].metadata.get("context_assessment") if placeholders else None
        original_assessment = CfmContextAssessmentOutput.model_validate(assessment_raw)
        sanitized_assessment, sanitization_audit = sanitize_assessment(
            original_assessment,
            cutoff_date=suite.cutoff_date,
            settings=self._settings,
        )
        by_horizon = {action.horizon: action for action in sanitized_assessment.horizon_actions}
        if set(by_horizon) != set(task.horizons):
            raise RuntimeError("Context assessment does not cover the requested horizons.")

        ensemble_by_horizon = {item.horizon: item for item in suite.ensemble.forecasts}
        decisions = {
            horizon: self._policy.apply(
                ensemble=ensemble_by_horizon[horizon],
                assessment=sanitized_assessment,
                horizon=horizon,
                horizons=task.horizons,
                latest_price=suite.diagnostics.latest_value,
                cutoff_date=suite.cutoff_date,
            )
            for horizon in task.horizons
        }
        approved_assessment = self._approved_assessment(sanitized_assessment, decisions)
        prediction_by_date = {str(prediction.forecast_date.date()): prediction for prediction in placeholders}

        for ensemble_forecast in suite.ensemble.forecasts:
            prediction = prediction_by_date[ensemble_forecast.forecast_date]
            decision = decisions[ensemble_forecast.horizon]
            prediction.payload = ContinuousForecast(
                point_forecast=decision.final_point_forecast,
                quantiles=decision.final_quantiles,
            )
            prediction.metadata.update(
                {
                    "model_suite_id": suite.model_suite_id,
                    "llm_context_assessment": original_assessment.model_dump(mode="json"),
                    "llm_proposed_context_assessment": original_assessment.model_dump(mode="json"),
                    "sanitized_context_assessment": sanitized_assessment.model_dump(mode="json"),
                    "policy_approved_context_assessment": approved_assessment.model_dump(mode="json"),
                    "context_assessment": approved_assessment.model_dump(mode="json"),
                    "llm_proposed_action": original_assessment.action_for(
                        ensemble_forecast.horizon
                    ).model_dump(mode="json"),
                    "policy_approved_action": approved_assessment.action_for(
                        ensemble_forecast.horizon
                    ).model_dump(mode="json"),
                    "assessment_sanitization_audit": sanitization_audit,
                    "workflow_audit": {
                        **self._market_data_tool.audit,
                        "reported_search_queries": list(original_assessment.reported_search_queries),
                        "reported_search_call_count": len(original_assessment.reported_search_queries),
                        "validated_structured_response_count": 1,
                        "graduated_evidence_policy_id": self._policy.policy_id,
                    },
                    "model_num_samples": suite.model_num_samples,
                    "authoritative_component_models": {
                        name: self._component_at_horizon(model, ensemble_forecast.horizon)
                        for name, model in suite.models.items()
                    },
                    "authoritative_ensemble": self._component_at_horizon(
                        suite.ensemble, ensemble_forecast.horizon
                    ),
                    "unadjusted_ensemble": self._component_at_horizon(
                        suite.ensemble, ensemble_forecast.horizon
                    ),
                    "configured_ensemble_weights": suite.configured_ensemble_weights,
                    "active_ensemble_weights": suite.active_ensemble_weights,
                    "successful_models": suite.successful_models,
                    "failed_models": suite.failed_models,
                    "model_disagreement_std": suite.model_disagreement_std[ensemble_forecast.horizon],
                    "training_data": {
                        name: model.training_data.model_dump(mode="json")
                        for name, model in suite.models.items()
                    },
                    "market_diagnostics": suite.diagnostics.model_dump(mode="json"),
                    "policy_decision": decision.model_dump(mode="json"),
                    # Fixed-width numeric row for the future score-to-price
                    # calibration study. Built from the sanitized assessment, so
                    # sanitizer-rejected evidence cannot inflate a feature. Pair
                    # it with unadjusted_ensemble above to form the regression
                    # target: actual move minus the quant baseline's expectation.
                    "calibration_features": sanitized_assessment.calibration_features(),
                    "llm_authored_final_numbers": False,
                    "market_data_tool_status": result.status,
                    "market_data_tool_warnings": result.warnings,
                }
            )
        return placeholders

    @staticmethod
    def _approved_assessment(
        sanitized: CfmContextAssessmentOutput,
        decisions: dict[int, PolicyDecision],
    ) -> CfmContextAssessmentOutput:
        raw = sanitized.model_dump(mode="json")
        approved_actions = []
        for action in sanitized.horizon_actions:
            decision = decisions[action.horizon]
            approved = action.model_dump(mode="json")
            approved["center_action"] = decision.center_action
            approved["uncertainty_action"] = decision.uncertainty_action
            approved["cited_claim_ids"] = [
                claim_id for claim_id in action.cited_claim_ids
                if claim_id in decision.qualifying_claim_ids
            ]
            if not decision.eligible:
                approved["persistence_profile"] = "unknown"
                approved["cited_claim_ids"] = []
                approved["rationale"] = (
                    "The LLM proposal was rejected by Python's graduated evidence policy; "
                    + " ".join(decision.eligibility_reasons)
                )
            elif decision.evidence_tier_level > 0:
                approved["rationale"] = (
                    f"Python approved evidence Tier {decision.evidence_tier_level} "
                    f"({decision.evidence_tier}) and bounded the proposed categorical action."
                )
            approved_actions.append(approved)
        raw["horizon_actions"] = approved_actions
        raw["research_summary"] = (
            "Python-approved assessment derived from the sanitized evidence record; "
            "unsupported citations and rejected actions were removed."
        )
        raw["overall_rationale"] = (
            "Final contextual actions are assigned and bounded by Python under the graduated evidence policy."
        )
        return CfmContextAssessmentOutput.model_validate(raw)

    @staticmethod
    def _component_at_horizon(result: ModelForecastResult, horizon: int) -> dict[str, Any]:
        if result.status == "error":
            return {
                "status": "error",
                "predictor_id": result.predictor_id,
                "num_samples": result.num_samples,
                "point_forecast": None,
                "quantiles": {},
                "forecast_date": None,
                "error": result.error,
            }
        forecast = next(item for item in result.forecasts if item.horizon == horizon)
        return {
            "status": "ok",
            "predictor_id": result.predictor_id,
            "num_samples": result.num_samples,
            "point_forecast": forecast.point_forecast,
            "quantiles": {str(q): value for q, value in forecast.quantiles.items()},
            "forecast_date": forecast.forecast_date,
            "error": None,
        }


__all__ = ["CfmV34Predictor"]
