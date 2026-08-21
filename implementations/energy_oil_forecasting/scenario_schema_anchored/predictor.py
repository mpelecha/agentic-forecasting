"""ARIMA-anchored, scenario-widened WTI Scenario Schema predictor.

Orchestrates three steps per call to :meth:`ScenarioSchemaAnchoredPredictor.predict`:

1. Run AutoARIMA once, deterministically, via
   :func:`~energy_oil_forecasting.scenario_schema_anchored.arima_anchor.compute_arima_anchor`.
2. Run the news-grounded LLM agent (via :class:`AgentPredictor`, wired with
   :class:`~energy_oil_forecasting.scenario_schema_anchored.prompt.AnchoredPromptBuilder`
   so the ARIMA numbers are visible in its prompt) to get factors, scenarios,
   and probabilities — the LLM's own point_forecast/quantiles are discarded.
3. Compute the final point_forecast/quantiles in Python: take the ARIMA
   anchor's own quantile grid and widen its outermost quantiles toward the
   scenario-implied price range, scaled by ``sqrt(horizon / max_horizon)`` —
   the same widen-only formula already used in
   :meth:`WtiScenarioForecastOutput.to_predictions`, just applied to a
   deterministic statistical baseline instead of the LLM's self-reported
   numbers.

This keeps the LLM doing what it's suited for (reading news, reasoning about
qualitative scenarios) and keeps Python owning all final arithmetic — the
same "Python owns the numbers" philosophy as CFM Agent v5.2's authoritative
numerical suite.
"""

from __future__ import annotations

from math import sqrt

from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import ContinuousForecast, Prediction
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic import AgentConfig, AgentPredictor
from energy_oil_forecasting.analyst_agent.agent import WtiScenarioForecastOutput
from energy_oil_forecasting.scenario_schema_anchored.arima_anchor import (
    compute_arima_anchor,
    horizon_for,
)
from energy_oil_forecasting.scenario_schema_anchored.prompt import AnchoredPromptBuilder


def _widen_toward_scenarios(
    quantiles: dict[float, float],
    scenario_low: float,
    scenario_high: float,
    *,
    scale: float,
) -> dict[float, float]:
    """Widen (never narrow) the outermost quantiles toward a scenario price range.

    Same formula as :meth:`WtiScenarioForecastOutput.to_predictions`, applied
    here to the ARIMA anchor's quantiles instead of the LLM's self-reported
    ones.
    """
    lowest_q = min(quantiles)
    highest_q = max(quantiles)
    widened = dict(quantiles)

    if widened[lowest_q] > scenario_low:
        widened[lowest_q] -= (widened[lowest_q] - scenario_low) * scale
    if widened[highest_q] < scenario_high:
        widened[highest_q] += (scenario_high - widened[highest_q]) * scale

    return widened


class ScenarioSchemaAnchoredPredictor(Predictor):
    """ARIMA-anchored Scenario Schema — Python owns the final numbers.

    The LLM produces factors, scenarios, and probabilities from live search;
    its own point_forecast/quantiles are used only as a self-consistency
    forcing function during generation (the existing
    :class:`WtiScenarioForecastOutput` validators still require them to
    agree with its stated scenarios) and are discarded afterward. The
    predictions actually returned are built from a deterministic AutoARIMA
    anchor, widened toward the LLM's scenario price range.
    """

    def __init__(self, config: AgentConfig, *, arima_num_samples: int = 1_000):
        self.arima_num_samples = arima_num_samples
        self._prompt_builder = AnchoredPromptBuilder(arima_anchor={})
        self.inner = AgentPredictor(
            agent_config=config,
            prompt_builder=self._prompt_builder,
            output_schema=WtiScenarioForecastOutput,
        )

    @property
    def predictor_id(self) -> str:
        return self.inner.predictor_id

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        arima_anchor = compute_arima_anchor(task, context, num_samples=self.arima_num_samples)
        # Stashed here so AnchoredPromptBuilder.__call__ can read it when the
        # inner AgentPredictor invokes the prompt builder below.
        self._prompt_builder.arima_anchor = arima_anchor

        llm_predictions = self.inner.predict(task, context)
        if not llm_predictions or not llm_predictions[0].metadata:
            raise RuntimeError("Scenario Schema Anchored: LLM call produced no scenarios to anchor against.")

        scenarios = llm_predictions[0].metadata.get("scenarios", [])
        if not scenarios:
            raise RuntimeError("Scenario Schema Anchored: no scenarios in LLM output metadata.")

        scenario_low = min(scenario["price_low"] for scenario in scenarios)
        scenario_high = max(scenario["price_high"] for scenario in scenarios)
        max_horizon = max(task.horizons)

        for pred in llm_predictions:
            horizon = horizon_for(pred.as_of, pred.forecast_date, task.horizons)
            anchor_cf = arima_anchor[horizon]
            scale = sqrt(horizon / max_horizon)

            final_quantiles = _widen_toward_scenarios(
                anchor_cf.quantiles, scenario_low, scenario_high, scale=scale
            )
            pred.payload = ContinuousForecast(
                point_forecast=anchor_cf.point_forecast,
                quantiles=final_quantiles,
            )
            pred.metadata["arima_anchor"] = {
                "point_forecast": anchor_cf.point_forecast,
                "quantiles": anchor_cf.quantiles,
            }
            pred.metadata["mixture_method"] = "arima_anchor_widened_by_scenario_range"

        return llm_predictions


def build_wti_scenario_schema_anchored_predictor(
    config: AgentConfig, *, arima_num_samples: int = 1_000
) -> ScenarioSchemaAnchoredPredictor:
    """Wrap an :class:`AgentConfig` in a :class:`ScenarioSchemaAnchoredPredictor`.

    Parameters
    ----------
    config : AgentConfig
        Config from :func:`~energy_oil_forecasting.scenario_schema_anchored.agent.build_wti_news_scenario_schema_anchored_config`.
    arima_num_samples : int, default=1_000
        Monte Carlo sample count for the AutoARIMA anchor.

    Returns
    -------
    ScenarioSchemaAnchoredPredictor
    """
    return ScenarioSchemaAnchoredPredictor(config, arima_num_samples=arima_num_samples)


__all__ = ["ScenarioSchemaAnchoredPredictor", "build_wti_scenario_schema_anchored_predictor"]
