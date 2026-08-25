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
    log_returns : bool, default=False
        Model the series in **log-return space**: the fitted target ``y`` is
        ``diff(log(value))``, i.e. the first difference of log price, not the
        price and not the log price.  Forecast sample paths are cumulated and
        applied to the last observed price
        (``last_price * exp(cumsum(predicted returns))``) to return to price
        space.

        This is deliberately *not* "fit AutoARIMA on log prices and let it
        choose ``d``".  That alternative is equivalent only when AutoARIMA
        happens to select ``d=1``, and it anchors the forecast on ARIMA's own
        fitted level structure rather than on the last observed price — a real
        difference in behaviour once drift or AR terms are present, and it
        grows with horizon.  Here ``y`` is the return series itself, so
        AutoARIMA fits a stationary model directly on it and the price path is
        reconstructed explicitly from the last close.

        Off by default so existing callers are unaffected; the two settings are
        separate predictors with separate ``predictor_id`` values, so their
        cached results never collide and can be compared side by side.
    price_floor : float, default=1.0
        Lower bound applied to the price series before taking logs.  Only used
        when ``log_returns`` is set.  ``log`` of a non-positive value is
        undefined and WTI printed negative in April 2020, so one unfloored
        observation would otherwise produce ``nan`` and poison the fit.

    Notes
    -----
    - **Darts AutoARIMA** requires ``statsforecast`` (already a project
      dependency).  No additional install is needed.
    - AutoARIMA can be slow (seconds to tens of seconds per origin). For rapid
      iteration use
      :class:`~aieng.forecasting.methods.darts_regression.DartsLinearRegressionPredictor`
      instead.
    """

    def __init__(self, num_samples: int = 500, *, log_returns: bool = False, price_floor: float = 1.0) -> None:
        self._num_samples = num_samples
        self._log_returns = log_returns
        self._price_floor = price_floor

    @property
    def predictor_id(self) -> str:
        """Return a stable string identifier for this predictor."""
        return "darts_autoarima_logret" if self._log_returns else "darts_autoarima"

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
        anchor_price: float | None = None
        if self._log_returns:
            # Non-positive prices are DROPPED, not clipped to the floor.
            # Clipping WTI's April-2020 -$37.63 print up to $1 manufactures a
            # log return of log(1/25) ~ -3.2 going in and ~ +3.0 coming out.
            # Against typical daily returns near 0.02 those two points carry
            # ~19 of squared error versus ~2.3 for the other 5,600 combined, so
            # a maximum-likelihood fit inflates sigma roughly threefold and the
            # intervals come out 2-3x too wide. (Percentile-based users of the
            # same floor are unaffected -- order statistics ignore two extreme
            # points -- which is why this only bites here.)
            # Dropping instead leaves one return spanning the gap, which is
            # large but real, rather than an artifact of the floor.
            series_df = series_df[series_df["value"] > self._price_floor].copy()
            log_price = np.log(series_df["value"])
            anchor_price = float(np.exp(log_price.iloc[-1]))
            series_df["value"] = log_price.diff()
            series_df = series_df.iloc[1:]

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

        all_values: np.ndarray = forecast_ts.all_values()
        if self._log_returns:
            # Each sample index is a coherent trajectory, so cumulating along
            # the time axis per sample gives that path's cumulative log return.
            # Price is then reconstructed from the last observed close. ``exp``
            # is monotonic, so exponentiating the samples and taking quantiles
            # equals taking quantiles first — no ordering is disturbed — and
            # the resulting interval is asymmetric in price space, which is the
            # point of modelling returns.
            all_values = np.cumsum(all_values, axis=0)

        for h in task.horizons:
            samples: np.ndarray = all_values[h - 1, 0, :]
            if self._log_returns:
                samples = anchor_price * np.exp(samples)
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
