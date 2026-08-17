"""Single-equation Engle-Granger error-correction predictor.

``ErrorCorrectionRegressionPredictor`` models the target series as being tied
to one or more covariate series by a long-run equilibrium relationship, with
short-run dynamics that correct back toward that equilibrium whenever the
series drifts away from it. This is the classic two-step Engle-Granger ECM,
scoped to a single equation (the target) rather than a full VECM system —
appropriate for any :class:`~aieng.forecasting.evaluation.task.ForecastingTask`,
since the harness only ever asks a predictor to forecast
``task.target_series_id``, not the covariates themselves. Like the Darts
regression predictors in this package, it is **task-agnostic**: point it at
any target with one or more covariate series that plausibly share a long-run
equilibrium (gasoline CPI vs. crude oil, an exchange rate vs. rate
differentials, equity levels vs. a valuation anchor, etc.).

Step 1 (long-run / cointegrating relationship)
    OLS of the target level on covariate levels, fit on all data available at
    ``context.as_of``::

        y_t = c + b_1 * x1_t + b_2 * x2_t + ... + e_t

    The residual ``e_t`` is the "equilibrium error" — how far the target
    currently sits above or below the level implied by its long-run
    relationship with the covariates.

Step 2 (short-run / error-correction regression)
    OLS of the target's period-over-period change on the covariates' changes
    and the *lagged* equilibrium error::

        dy_{t+1} = a * e_t + b_1 * dx1_t + b_2 * dx2_t + ... + eps_t

    The coefficient ``a`` is the error-correction speed: how much of the
    current disequilibrium gets closed off in the next period. Only
    already-realized values (``e_t``, ``dx_t``) are used as inputs, so
    forecasting ``dy_{t+1}`` from information available at ``t`` requires no
    future covariate data.

Multi-step horizons
    Forecasts beyond one step are produced by iterating the fitted short-run
    equation forward, holding each covariate's rate of change fixed at its
    last observed value (a flat / random-walk-in-differences assumption for
    the exogenous drivers) and updating the equilibrium error at each step
    from the projected covariate levels. Forecast uncertainty is widened with
    the square root of the horizon, the standard random-walk error-growth
    heuristic.

Log levels
    Price-like series (CPI, commodity prices, equity levels) are often
    modeled in logs so the long-run relationship is linear in growth rates
    rather than levels. Set ``use_log_levels=True`` when the target and every
    covariate are strictly positive; this is **off by default** since not
    every series qualifies (returns, rate differentials, and other series
    that can go negative would raise on the log transform). A clear
    ``ValueError`` is raised if you request logs on non-positive data.

Choosing covariates
    Any series available from the ``ForecastContext`` can be used, but the
    ECM specification only makes economic sense when the target and
    covariates are plausibly cointegrated — i.e. individually non-stationary
    but tied together by a stable long-run relationship. Feeding it
    unrelated or already-stationary covariates won't break the code, but the
    "long-run equilibrium" the model fits won't mean anything.

Usage::

    from aieng.forecasting.methods.numerical import ErrorCorrectionRegressionPredictor
    from aieng.forecasting.evaluation import backtest

    # Gasoline CPI vs. crude oil + industrial production (log levels)
    predictor = ErrorCorrectionRegressionPredictor(
        covariate_series_ids=["crude", "indpro"],
        use_log_levels=True,
    )
    result = backtest(predictor=predictor, spec=spec, data_service=svc)
    print(f"ECM mean CRPS: {result.mean_score:.4f}")

    # A different task: e.g. an FX rate vs. a rate differential (raw levels,
    # since a rate differential can be negative)
    fx_predictor = ErrorCorrectionRegressionPredictor(
        covariate_series_ids=["rate_differential"],
    )
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES, ContinuousForecast, Prediction
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask


class ErrorCorrectionRegressionPredictor(Predictor):
    """Single-equation Engle-Granger ECM: long-run levels + short-run corrections.

    Parameters
    ----------
    covariate_series_ids : list[str]
        Series ids fetched from the ``ForecastContext`` and used as the
        long-run / short-run explanatory variables. At least one required.
        Choose series plausibly cointegrated with the target — see "Choosing
        covariates" above.
    use_log_levels : bool
        If True, fit the long-run relationship in log space (appropriate for
        strictly positive price/level series, and the conventional choice
        for that case). Defaults to False (raw levels), since not every
        target/covariate is guaranteed positive. Raises ``ValueError`` at
        predict time if True but the data contains non-positive values.
    min_observations : int
        Minimum number of overlapping observations required across the
        target and all covariates before fitting. Raises ``ValueError`` if
        the available history is shorter than this at a given origin. The
        default (24) is a reasonable floor for monthly data; tighten or
        loosen it to match your task's frequency and the degrees of freedom
        needed for a stable fit (roughly ``2 * (1 + n_covariates)`` at a
        minimum).
    """

    def __init__(
        self,
        covariate_series_ids: list[str],
        use_log_levels: bool = False,
        min_observations: int = 24,
    ) -> None:
        if not covariate_series_ids:
            raise ValueError("covariate_series_ids must contain at least one series id")
        self._covariate_series_ids = list(covariate_series_ids)
        self._use_log_levels = use_log_levels
        self._min_observations = min_observations

    @property
    def predictor_id(self) -> str:
        """Return a stable identifier, suffixed ``_log`` when fit in log space."""
        suffix = "_log" if self._use_log_levels else ""
        return f"ecm_regression{suffix}"

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Produce ECM forecasts for every horizon in the task."""
        merged = self._build_merged_frame(task, context)

        if len(merged) < self._min_observations:
            raise ValueError(
                f"ECM predictor needs at least {self._min_observations} overlapping "
                f"observations, got {len(merged)} as of {context.as_of}."
            )

        if self._use_log_levels:
            non_positive_cols = [
                col
                for col in ["y", *self._covariate_series_ids]
                if (merged[col] <= 0).any()
            ]
            if non_positive_cols:
                raise ValueError(
                    "use_log_levels=True requires strictly positive values, but "
                    f"found non-positive observations in: {non_positive_cols}. "
                    "Set use_log_levels=False to fit on raw levels instead."
                )

        transform, inverse_transform = self._get_transforms()
        y = transform(merged["y"].to_numpy())
        x = transform(merged[self._covariate_series_ids].to_numpy())

        # --- Step 1: long-run / cointegrating regression (levels) ---
        long_run_model = LinearRegression()
        long_run_model.fit(x, y)
        residuals = y - long_run_model.predict(x)  # equilibrium error e_t

        # --- Step 2: short-run error-correction regression (differences) ---
        dy = np.diff(y)  # dy[i] = y[i+1] - y[i]
        dx = np.diff(x, axis=0)  # dx[i] = x[i+1] - x[i]
        e_t = residuals[:-1]  # equilibrium error known at time of dx[i]

        # Features available at time t: [e_t, dx_t]; target: dy_{t+1}.
        short_run_features = np.column_stack([e_t, dx])[:-1]
        short_run_target = dy[1:]

        short_run_model = LinearRegression()
        short_run_model.fit(short_run_features, short_run_target)
        fitted = short_run_model.predict(short_run_features)
        residual_std = float(np.std(short_run_target - fitted, ddof=1))

        # --- Iteratively forecast forward to the max requested horizon ---
        last_y = float(y[-1])
        last_x = x[-1].copy()
        last_dx = dx[-1].copy()  # held flat for all future steps
        last_e = float(residuals[-1])

        max_horizon = max(task.horizons)
        point_forecasts: dict[int, float] = {}
        for step in range(1, max_horizon + 1):
            features = np.concatenate([[last_e], last_dx]).reshape(1, -1)
            pred_dy = float(short_run_model.predict(features)[0])

            new_y = last_y + pred_dy
            new_x = last_x + last_dx  # covariate levels drift by their last observed change
            new_long_run_fit = float(long_run_model.predict(new_x.reshape(1, -1))[0])
            new_e = new_y - new_long_run_fit

            point_forecasts[step] = new_y
            last_y, last_x, last_e = new_y, new_x, new_e

        return self._build_predictions(
            task=task,
            context=context,
            point_forecasts=point_forecasts,
            residual_std=residual_std,
            inverse_transform=inverse_transform,
        )

    def _get_transforms(self):
        """Return (forward, inverse) transforms for log-level or raw-level fitting."""
        if self._use_log_levels:
            return np.log, np.exp
        return (lambda a: a), (lambda a: a)

    def _build_merged_frame(self, task: ForecastingTask, context: ForecastContext) -> pd.DataFrame:
        """Fetch target + covariates and inner-join them on timestamp."""
        target_df = context.get_series(task.target_series_id)[["timestamp", "value"]].rename(
            columns={"value": "y"}
        )
        merged = target_df
        for cov_id in self._covariate_series_ids:
            cov_df = context.get_series(cov_id)[["timestamp", "value"]].rename(columns={"value": cov_id})
            merged = merged.merge(cov_df, on="timestamp", how="inner")
        return merged.sort_values("timestamp").reset_index(drop=True)

    def _build_predictions(
        self,
        *,
        task: ForecastingTask,
        context: ForecastContext,
        point_forecasts: dict[int, float],
        residual_std: float,
        inverse_transform,
    ) -> list[Prediction]:
        """Assemble one Prediction per requested horizon, with Gaussian quantiles.

        Uncertainty is widened with sqrt(horizon), the standard random-walk
        error-growth heuristic, since each additional step compounds the
        held-flat-covariate assumption.
        """
        offset = pd.tseries.frequencies.to_offset(task.frequency)
        issued_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)

        predictions = []
        for h in task.horizons:
            point_transformed = point_forecasts[h]
            step_std = residual_std * np.sqrt(h)

            quantiles = {
                q: float(inverse_transform(point_transformed + norm.ppf(q) * step_std))
                for q in STANDARD_QUANTILES
            }
            payload = ContinuousForecast(
                point_forecast=float(inverse_transform(point_transformed)),
                quantiles=quantiles,
            )
            predictions.append(
                Prediction(
                    predictor_id=self.predictor_id,
                    task_id=task.task_id,
                    issued_at=issued_at,
                    as_of=context.as_of,
                    forecast_date=(pd.Timestamp(context.as_of) + offset * h).to_pydatetime(),
                    payload=payload,
                    metadata={"covariates": self._covariate_series_ids},
                )
            )
        return predictions
