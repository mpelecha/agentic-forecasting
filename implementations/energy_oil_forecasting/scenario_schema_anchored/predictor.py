"""ARIMA-anchored, scenario-shifted, scenario-widened WTI Scenario Schema predictor.

Orchestrates four steps per call to :meth:`ScenarioSchemaAnchoredPredictor.predict`:

1. Run AutoARIMA once, deterministically, via
   :func:`~energy_oil_forecasting.scenario_schema_anchored.arima_anchor.compute_arima_anchor`.
2. Run the news-grounded LLM agent (via :class:`AgentPredictor`, wired with
   :class:`~energy_oil_forecasting.scenario_schema_anchored.prompt.AnchoredPromptBuilder`
   so the ARIMA numbers are visible in its prompt) to get factors, scenarios,
   and probabilities — the LLM's own point_forecast/quantiles are discarded,
   but its scenarios (each with a probability and a price range) are not.
3. Shift the ARIMA anchor's quantile grid toward the scenarios' own
   probability-weighted price (see :func:`_probability_weighted_scenario_price`
   — the same quantity :meth:`WtiScenarioForecastOutput` already validates
   the LLM's point_forecast against, at the longest horizon). The shift
   itself is never the LLM's raw dollar figure: it's re-grounded in the real
   empirical distribution of historical h-day WTI price deltas (see
   :func:`_implied_target_percentile` and :func:`_grounded_center_shift`),
   the same discipline used by
   :mod:`energy_oil_forecasting.cfm_agent_v_5_2_2_delta_governed`.
4. Widen (never narrow) the shifted grid's outermost quantiles toward the
   scenario-implied price range, scaled by ``sqrt(horizon / max_horizon)`` —
   the same widen-only formula already used in
   :meth:`WtiScenarioForecastOutput.to_predictions`, just applied to the
   shifted anchor instead of the LLM's self-reported numbers.

Earlier versions of this predictor discarded the scenarios' probability
weighting entirely and pinned point_forecast to the raw, unshifted ARIMA
anchor unconditionally — meaning news the LLM read could only ever widen
the interval, never move its center. That defeated the purpose of reading
news at all for anything but tail-width: the LLM's job narrows to what it's
suited for (reading live news, weighing competing scenarios), while Python
still owns all the arithmetic, but now uses the scenario information as
intended rather than discarding the direction and magnitude it encodes.

The tail-widening step (step 4) still uses each scenario's raw
price_low/price_high, unweighted by probability, same as before — so a
low-probability tail scenario can still stretch the interval as much as a
high-probability one. That's a known, separate gap from the center-shift
fix here and is not addressed by this change.
"""

from __future__ import annotations

from math import sqrt

import numpy as np
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import ContinuousForecast, Prediction
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic import AgentConfig, AgentPredictor
from energy_oil_forecasting.analyst_agent.agent import WtiScenarioForecastOutput
from energy_oil_forecasting.price_deltas import PERCENTILE_LEVELS, compute_horizon_delta_percentiles
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
    here to the (already center-shifted) ARIMA anchor's quantiles instead of
    the LLM's self-reported ones.
    """
    lowest_q = min(quantiles)
    highest_q = max(quantiles)
    widened = dict(quantiles)

    if widened[lowest_q] > scenario_low:
        widened[lowest_q] -= (widened[lowest_q] - scenario_low) * scale
    if widened[highest_q] < scenario_high:
        widened[highest_q] += (scenario_high - widened[highest_q]) * scale

    return widened


def _probability_weighted_scenario_price(scenarios: list[dict]) -> float:
    """Probability-weighted average of each scenario's own price midpoint.

    Mirrors :meth:`WtiScenarioForecastOutput._point_forecast_consistent_with_scenarios`
    exactly, so this is the same quantity the LLM's own point_forecast was
    already required to match (within tolerance) at the longest horizon —
    not a new number, just one Python no longer throws away.
    """
    total_probability = sum(scenario["probability"] for scenario in scenarios)
    if total_probability <= 0:
        raise ValueError("Scenario probabilities must sum to a positive value.")
    return (
        sum(scenario["probability"] * (scenario["price_low"] + scenario["price_high"]) / 2 for scenario in scenarios)
        / total_probability
    )


def _implied_target_percentile(llm_weighted_price: float, anchor_quantiles: dict[float, float]) -> float:
    """Turn the LLM's probability-weighted scenario price into a percentile position.

    Expresses how far ``llm_weighted_price`` sits from the ARIMA anchor's own
    center as a fraction of the anchor's own one-sided spread (p50->p90 if
    the price is above center, p50->p10 if below), clipped to +/-1 span —
    the LLM cannot imply more conviction than the anchor's own quantile grid
    already allows for its center. That normalized position in [-1, 1] maps
    linearly onto a percentile in [10, 90], centered on 50.
    """
    p10, p50, p90 = anchor_quantiles[0.1], anchor_quantiles[0.5], anchor_quantiles[0.9]
    raw = llm_weighted_price - p50
    span = (p90 - p50) if raw >= 0 else (p50 - p10)
    fraction = max(-1.0, min(1.0, raw / span)) if span > 0 else 0.0
    return 50.0 + fraction * 40.0


def _grounded_center_shift(target_percentile: float, delta_percentiles: dict[int, float]) -> float:
    """Interpolate the REAL historical h-day price-delta value at ``target_percentile``.

    ``delta_percentiles`` is one horizon's entry from
    :func:`~energy_oil_forecasting.price_deltas.compute_horizon_delta_percentiles`
    — actual historical price moves, never an LLM-invented dollar figure.
    Returned relative to the historical median (delta_percentiles[50]) so it
    can be added directly to the anchor's own p50 as a shift.
    """
    levels = list(PERCENTILE_LEVELS)
    values = [delta_percentiles[level] for level in levels]
    interpolated = float(np.interp(target_percentile, levels, values))
    return interpolated - delta_percentiles[50]


class ScenarioSchemaAnchoredPredictor(Predictor):
    """ARIMA-anchored Scenario Schema — Python owns the final numbers.

    The LLM produces factors, scenarios, and probabilities from live search;
    its own point_forecast/quantiles are used only as a self-consistency
    forcing function during generation (the existing
    :class:`WtiScenarioForecastOutput` validators still require them to
    agree with its stated scenarios) and are discarded afterward. The
    predictions actually returned are built from a deterministic AutoARIMA
    anchor, shifted toward the scenarios' own probability-weighted price
    (re-grounded in real historical price-delta percentiles, never the LLM's
    raw dollar figures) and then widened toward the LLM's scenario price
    range.
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
        # log_returns=True: the paired offline swap (see
        # scripts/compare_anchor_level_vs_logret.py) put log returns ahead at
        # h=10 and h=21 in BOTH windows -- CRPS -0.087/-0.059 at h=10 and
        # -0.035/-0.166 at h=21 (backtest/eval) -- and behind at h=63 by
        # +0.338/+0.131. The pooled sign flips between windows purely because
        # those two effects trade places in size, so the pooled figure was
        # never the signal; the per-horizon pattern is what replicates.
        #
        # Adopted globally rather than per horizon: h=63 is not a horizon this
        # project targets, and a single specification is worth more than
        # squeezing the quarterly column. It also removes a real mismatch --
        # the centre-shift cap in this same file already works in log-return
        # space, so the band was the only part left in levels.
        #
        # One caveat on the measured size. AnchoredPromptBuilder shows the
        # anchor to the LLM (see the assignment below), while that comparison
        # held the cached scenarios fixed and swapped only the arithmetic
        # after the call. It therefore measures the numerical effect alone;
        # the LLM now also reads different anchor numbers and may reason
        # differently from them. The measured gain is a lower bound on the
        # change, not the whole of it.
        arima_anchor = compute_arima_anchor(
            task, context, num_samples=self.arima_num_samples, log_returns=True
        )
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

        # The LLM's probability-weighted scenario view, re-expressed as a
        # percentile position against the anchor's own quantile grid, then
        # translated per-horizon into a real historical-delta dollar shift.
        weighted_price = _probability_weighted_scenario_price(scenarios)
        target_percentile = _implied_target_percentile(weighted_price, arima_anchor[max_horizon].quantiles)
        delta_percentiles = compute_horizon_delta_percentiles(context, task.target_series_id, task.horizons)

        for pred in llm_predictions:
            horizon = horizon_for(pred.as_of, pred.forecast_date, task.horizons)
            anchor_cf = arima_anchor[horizon]
            scale = sqrt(horizon / max_horizon)

            center_shift = _grounded_center_shift(target_percentile, delta_percentiles[horizon])
            shifted_quantiles = {level: value + center_shift for level, value in anchor_cf.quantiles.items()}

            final_quantiles = _widen_toward_scenarios(
                shifted_quantiles, scenario_low, scenario_high, scale=scale
            )
            pred.payload = ContinuousForecast(
                point_forecast=anchor_cf.point_forecast + center_shift,
                quantiles=final_quantiles,
            )
            pred.metadata["arima_anchor"] = {
                "point_forecast": anchor_cf.point_forecast,
                "quantiles": anchor_cf.quantiles,
            }
            pred.metadata["scenario_probability_weighted_price_max_horizon"] = weighted_price
            pred.metadata["scenario_implied_target_percentile"] = target_percentile
            pred.metadata["scenario_center_shift"] = center_shift
            pred.metadata["historical_delta_percentiles"] = delta_percentiles[horizon]
            pred.metadata["mixture_method"] = "arima_anchor_shifted_by_grounded_scenario_percentile_widened_by_scenario_range"

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
