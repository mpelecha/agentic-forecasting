"""Deterministic AutoARIMA anchor for scenario_schema_anchored.

Runs AutoARIMA once, in Python, before the LLM call. Its point forecast and
quantiles become the statistical baseline the LLM's scenarios are asked to
reason from, and the base the final predictions are built on — instead of
letting the LLM invent price numbers with no numerical grounding.
"""

from __future__ import annotations

import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import ContinuousForecast
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.numerical import DartsAutoARIMAPredictor


def horizon_for(as_of, forecast_date, horizons: list[int]) -> int:
    """Match a prediction's forecast_date back to its horizon in business days."""
    as_of = pd.Timestamp(as_of)
    forecast_date = pd.Timestamp(forecast_date)
    offset = pd.tseries.offsets.BDay()
    for horizon in horizons:
        if (as_of + offset * horizon).normalize() == forecast_date.normalize():
            return horizon
    raise ValueError(f"Could not match forecast_date {forecast_date} to any horizon in {horizons}")


def compute_arima_anchor(
    task: ForecastingTask,
    context: ForecastContext,
    *,
    num_samples: int = 1_000,
) -> dict[int, ContinuousForecast]:
    """Run AutoARIMA once and index its per-horizon forecasts by horizon.

    Parameters
    ----------
    task : ForecastingTask
        Same task passed to the overall predictor — horizons, target series,
        frequency.
    context : ForecastContext
        Same context passed to the overall predictor — enforces the same
        information cutoff ARIMA and the LLM both see.
    num_samples : int, default=1_000
        Monte Carlo sample count for AutoARIMA's quantile estimation.

    Returns
    -------
    dict[int, ContinuousForecast]
        Maps each horizon in ``task.horizons`` to its deterministic ARIMA
        point forecast and quantiles.
    """
    predictor = DartsAutoARIMAPredictor(num_samples=num_samples)
    predictions = predictor.predict(task, context)

    anchor: dict[int, ContinuousForecast] = {}
    for pred in predictions:
        if not isinstance(pred.payload, ContinuousForecast):
            continue
        horizon = horizon_for(pred.as_of, pred.forecast_date, task.horizons)
        anchor[horizon] = pred.payload

    missing = set(task.horizons) - set(anchor)
    if missing:
        raise RuntimeError(f"AutoARIMA anchor missing horizons: {sorted(missing)}")
    return anchor


__all__ = ["compute_arima_anchor", "horizon_for"]
