"""Prompt builder that embeds the ARIMA anchor into the payload."""

from __future__ import annotations

import json

from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import ContinuousForecast
from aieng.forecasting.evaluation.task import ForecastingTask
from energy_oil_forecasting.analyst_agent.agent import WtiPriceForecastPromptBuilder
from pydantic import BaseModel


class AnchoredPromptBuilder(BaseModel):
    """Wraps :class:`WtiPriceForecastPromptBuilder`, adding an `arima_anchor` block.

    The anchor is a precomputed, deterministic AutoARIMA forecast (point +
    quantiles per horizon) that the LLM is told to reason from, rather than
    inventing price numbers with no statistical grounding.

    ``arima_anchor`` is mutable — :class:`~energy_oil_forecasting.scenario_schema_anchored.predictor.ScenarioSchemaAnchoredPredictor`
    computes it fresh each call and assigns it here immediately before
    invoking the inner :class:`AgentPredictor`, the same "compute, stash,
    then call" pattern CFM Agent v5.2 uses for its own tools.
    """

    model_config = {"extra": "forbid"}

    arima_anchor: dict[int, ContinuousForecast] = {}

    def __call__(self, *, task: ForecastingTask, context: ForecastContext) -> str:
        base_payload = json.loads(WtiPriceForecastPromptBuilder()(task=task, context=context))
        base_payload["arima_anchor"] = {
            str(horizon): {
                "point_forecast": cf.point_forecast,
                "quantiles": cf.quantiles,
            }
            for horizon, cf in self.arima_anchor.items()
        }
        return json.dumps(base_payload, indent=2)


__all__ = ["AnchoredPromptBuilder"]
