"""Darts AutoARIMA predictor — probabilistic forecast via Monte Carlo sampling.

``DartsAutoARIMAPredictor`` wraps Darts ``AutoARIMA`` on the target series only
(univariate). Darts' ``AutoARIMA`` implementation used here does not support
exogenous covariates; this class does not expose any covariate parameters.

The probabilistic forecast is produced via Monte Carlo sampling (``num_samples``
draws from the predictive distribution).  Point forecast is the median;
quantiles use :data:`~aieng.forecasting.evaluation.prediction.STANDARD_QUANTILES`
levels.

For multi-horizon tasks, the model is fitted once to ``n = max(task.horizons)``
and samples are extracted at each requested horizon index from the resulting
trajectory. This is more efficient than fitting once per horizon.

Usage::

    from aieng.forecasting.methods.darts_arima import DartsAutoARIMAPredictor
    from aieng.forecasting.evaluation import backtest, BacktestSpec

    predictor = DartsAutoARIMAPredictor()
    result = backtest(predictor=predictor, spec=spec, data_service=svc)
    print(f"Mean CRPS: {result.mean_score:.4f}")
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES, ContinuousForecast, Prediction
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask


class DartsAutoARIMAPredictor(Predictor):
    """Probabilistic predictor wrapping Darts AutoARIMA (univariate).

    Fits AutoARIMA on the target series history available at the forecast
    origin, then generates a probabilistic trajectory via Monte Carlo sampling.
    One :class:`~aieng.forecasting.evaluation.prediction.Prediction` is
    returned per horizon step declared in ``task.horizons``.

    Parameters
    ----------
    num_samples : int
        Number of Monte Carlo samples used to build the predictive distribution.
        Higher values give smoother quantile estimates at the cost of compute.
        Default: 500.
    log_transform : bool, default=False
        Fit on ``log(value)`` and exponentiate the predictive samples back to
        the original scale.  Because AutoARIMA selects its own differencing
        order on whatever series it is given, fitting on log levels makes the
        model one of **log differences** (log returns) rather than of price
        changes — the innovation variance is then proportional to the level
        instead of constant in absolute units, which matters for a series that
        has traded across a wide range of price levels.

        Off by default so existing callers are unaffected; the two settings are
        separate predictors with separate ``predictor_id`` values, so their
        cached results never collide and can be compared side by side.
    price_floor : float, default=1.0
        Lower bound applied to the series before taking logs.  Only used when
        ``log_transform`` is set.  ``log`` of a non-positive value is undefined
        and WTI printed negative in April 2020, so one unfloored observation
        would otherwise produce ``nan`` and poison the fit.

    Notes
    -----
    - **Darts AutoARIMA** requires ``statsforecast`` (already a project
      dependency).  No additional install is needed.
    - AutoARIMA can be slow (seconds to tens of seconds per origin). For rapid
      iteration use
      :class:`~aieng.forecasting.methods.darts_regression.DartsLinearRegressionPredictor`
      instead.
    """

    def __init__(self, num_samples: int = 500, *, log_transform: bool = False, price_floor: float = 1.0) -> None:
        self._num_samples = num_samples
        self._log_transform = log_transform
        self._price_floor = price_floor

    @property
    def predictor_id(self) -> str:
        """Return a stable string identifier for this predictor."""
        return "darts_autoarima_log" if self._log_transform else "darts_autoarima"

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Produce probabilistic AutoARIMA forecasts for every horizon in the task.

        Parameters
        ----------
        task : ForecastingTask
            Defines the target series, horizons, and frequency.
        context : ForecastContext
            Cutoff-scoped data view.  All series returned respect
            ``context.as_of``.

        Returns
        -------
        list[Prediction]
            One ``ContinuousForecast`` per horizon step in ``task.horizons``,
            with ``point_forecast`` equal to the median of the predictive
            sample at that step.
        """
        from darts import TimeSeries  # noqa: PLC0415
        from darts.models import AutoARIMA  # noqa: PLC0415  # type: ignore[import-untyped]

        series_df = context.get_series(task.target_series_id)
        if self._log_transform:
            series_df = series_df.copy()
            series_df["value"] = np.log(series_df["value"].clip(lower=self._price_floor))

        ts = TimeSeries.from_dataframe(
            series_df,
            time_col="timestamp",
            value_cols="value",
            fill_missing_dates=True,
            freq=task.frequency,
        )

        model = AutoARIMA()
        model.fit(ts)

        # Fit once to max horizon; extract samples at each requested step.
        # all_values() shape: (n_steps, n_components, n_samples), 0-indexed.
        forecast_ts: Any = model.predict(
            n=task.horizon,
            num_samples=self._num_samples,
        )

        offset = pd.tseries.frequencies.to_offset(task.frequency)
        issued_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        predictions: list[Prediction] = []

        for h in task.horizons:
            samples: np.ndarray = forecast_ts.all_values()[h - 1, 0, :]
            if self._log_transform:
                # Back to price space. ``exp`` is monotonic, so exponentiating
                # the samples and then taking quantiles is equivalent to taking
                # quantiles in log space and exponentiating — no ordering is
                # disturbed, and the resulting interval is asymmetric in price
                # terms, which is the point of modelling returns.
                samples = np.exp(samples)
            payload = ContinuousForecast(
                point_forecast=float(np.median(samples)),
                quantiles={q: float(np.quantile(samples, q)) for q in STANDARD_QUANTILES},
            )
            forecast_date: datetime = (pd.Timestamp(context.as_of) + offset * h).to_pydatetime()
            predictions.append(
                Prediction(
                    predictor_id=self.predictor_id,
                    task_id=task.task_id,
                    issued_at=issued_at,
                    as_of=context.as_of,
                    forecast_date=forecast_date,
                    payload=payload,
                )
            )

        return predictions
