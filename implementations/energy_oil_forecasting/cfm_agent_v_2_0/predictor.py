"""Harness-compatible predictor that attaches authoritative model attribution."""

from __future__ import annotations

from typing import Any

from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import Prediction
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic import AgentConfig, AgentPredictor
from energy_oil_forecasting.cfm_agent_v_2_0.outputs import CfmRichForecastOutput
from energy_oil_forecasting.cfm_agent_v_2_0.prompts import CfmForecastPromptBuilder
from energy_oil_forecasting.cfm_agent_v_2_0.schemas import ModelForecastResult
from energy_oil_forecasting.cfm_agent_v_2_0.tools import MarketDataTool


class CfmRichAgentPredictor(Predictor):
    """Run the rich-output agent and merge authoritative tool results.

    The agent is asked to reproduce component forecasts for readability. This
    wrapper does not trust that copy as the system of record: after the run it
    reads the exact ``MarketDataTool.last_result`` object produced during the
    same call and writes those values into ``Prediction.metadata``. Both copies
    remain available, and ``component_copy_matches_tool`` flags discrepancies.
    """

    def __init__(self, config: AgentConfig, market_data_tool: MarketDataTool) -> None:
        self._market_data_tool = market_data_tool
        self._inner = AgentPredictor(
            agent_config=config,
            prompt_builder=CfmForecastPromptBuilder(),
            output_schema=CfmRichForecastOutput,
        )

    @property
    def predictor_id(self) -> str:
        """Delegate the stable identifier to the underlying agent predictor."""
        return self._inner.predictor_id

    @property
    def _runner(self) -> Any:
        """Expose the runner for compatibility with notebook trace inspection."""
        return self._inner._runner

    def predict(
        self,
        task: ForecastingTask,
        context: ForecastContext,
    ) -> list[Prediction]:
        """Return standard predictions enriched with authoritative attribution."""
        predictions = self._inner.predict(task, context)
        result = self._market_data_tool.last_result
        if not predictions:
            return predictions
        if result is None or result.model_suite is None:
            for prediction in predictions:
                warnings = list(prediction.metadata.get("warnings", []))
                warnings.append("No authoritative model-suite result was captured.")
                prediction.metadata["warnings"] = warnings
                prediction.metadata["component_copy_matches_tool"] = False
            return predictions
        suite = result.model_suite
        by_forecast_date = {str(prediction.forecast_date.date()): prediction for prediction in predictions}
        for horizon in suite.horizons:
            component_models: dict[str, Any] = {}
            for name, model_result in suite.models.items():
                component_models[name] = self._component_at_horizon(model_result, horizon)
            if suite.ensemble is not None:
                component_models["ensemble"] = self._component_at_horizon(
                    suite.ensemble,
                    horizon,
                )
            ensemble_entry = component_models.get("ensemble", {})
            forecast_date = ensemble_entry.get("forecast_date")
            prediction = by_forecast_date.get(forecast_date)
            if prediction is None:
                continue
            agent_copy = prediction.metadata.get("agent_reported_component_models", {})
            prediction.metadata["component_models"] = component_models
            prediction.metadata["component_copy_matches_tool"] = self._copies_match(
                agent_copy,
                component_models,
            )
            prediction.metadata["training_data"] = {
                name: model.training_data.model_dump(mode="json") for name, model in suite.models.items()
            }
            if suite.ensemble is not None:
                prediction.metadata["training_data"]["ensemble"] = suite.ensemble.training_data.model_dump(mode="json")
            prediction.metadata["configured_ensemble_weights"] = suite.configured_ensemble_weights
            prediction.metadata["active_ensemble_weights"] = suite.active_ensemble_weights
            prediction.metadata["successful_models"] = suite.successful_models
            prediction.metadata["failed_models"] = suite.failed_models
            prediction.metadata["model_disagreement_std"] = suite.model_disagreement_std[horizon]
            prediction.metadata["agent_reported_ensemble_to_final_adjustment"] = prediction.metadata.get(
                "ensemble_to_final_adjustment"
            )
            ensemble_point = ensemble_entry.get("point_forecast")
            if ensemble_point is not None:
                prediction.metadata["ensemble_to_final_adjustment"] = prediction.payload.point_forecast - ensemble_point
            prediction.metadata["market_data_tool_status"] = result.status
            prediction.metadata["market_data_tool_warnings"] = result.warnings
        return predictions

    @staticmethod
    def _component_at_horizon(
        result: ModelForecastResult,
        horizon: int,
    ) -> dict[str, Any]:
        if result.status == "error":
            return {
                "status": "error",
                "predictor_id": result.predictor_id,
                "point_forecast": None,
                "quantiles": {},
                "forecast_date": None,
                "error": result.error,
            }
        forecast = next(item for item in result.forecasts if item.horizon == horizon)
        return {
            "status": "ok",
            "predictor_id": result.predictor_id,
            "point_forecast": forecast.point_forecast,
            "quantiles": {str(q): value for q, value in forecast.quantiles.items()},
            "forecast_date": forecast.forecast_date,
            "error": None,
        }

    @staticmethod
    def _copies_match(agent_copy: dict[str, Any], tool_copy: dict[str, Any]) -> bool:
        """Compare the material component values, ignoring descriptive fields."""
        for name, authoritative in tool_copy.items():
            reported = agent_copy.get(name)
            if reported is None or reported.get("status") != authoritative.get("status"):
                return False
            if authoritative.get("status") == "error":
                continue
            if reported.get("point_forecast") != authoritative.get("point_forecast"):
                return False
            if reported.get("quantiles") != authoritative.get("quantiles"):
                return False
        return True


__all__ = ["CfmRichAgentPredictor"]
