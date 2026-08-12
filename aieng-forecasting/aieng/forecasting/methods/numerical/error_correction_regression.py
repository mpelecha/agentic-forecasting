r"""Regularized single-equation error-correction model (ECM) for level targets.

:class:`ErrorCorrectionRegressionPredictor` is a bespoke Engle-Granger
two-step error-correction model, not a wrapper around a Darts estimator —
hence no ``darts_`` prefix, which this package reserves for thin adapters over
the Darts library.

The model
---------
Step 1 (**long run**, on levels) estimates a candidate cointegrating relation

.. math:: y_t = \alpha + \beta' x_t + u_t

and keeps the equilibrium error :math:`u_t`.  Step 2 (**short run**, on first
differences) regresses the change in the target on the *lagged* equilibrium
error plus contemporaneous covariate changes

.. math::

    \Delta y_t = \gamma + \varphi u_{t-1} + \theta' \Delta x_t + \varepsilon_t

:math:`\varphi` is the error-correction speed: negative means the target is
pulled back toward the long-run relation, and its magnitude sets how fast.

This targets the **price level** (e.g. ``wti_crude_oil_price`` at horizons
5/10/21 business days), not a cumulative return — the long-run/short-run
decomposition is only meaningful for a levels target.

Why regularized, and why these particular checks
------------------------------------------------
**ElasticNet instead of OLS, in both steps.**  A levels-on-levels regression
between trending series is the textbook setup for *spurious regression*: OLS
will report a high :math:`R^2` and confident coefficients between series that
share nothing but a trend.  Shrinkage does not make the relation real, but it
stops a wide covariate panel from manufacturing an arbitrarily good in-sample
fit, and the L1 component performs covariate selection so the caller need not
hand-pick a short list.

**TimeSeriesSplit, never plain k-fold.**  The penalty is chosen by
cross-validation, and ordinary k-fold trains on future blocks to score earlier
ones.  That leaks information the model would not have had at the forecast
origin — the penalty ends up tuned with hindsight.  ``TimeSeriesSplit`` only
ever trains on the past of each validation block.

**Feature standardization.**  A penalty on unscaled coefficients is really a
penalty on units: a covariate panel mixing log returns (:math:`10^{-3}`) with
inventory levels (:math:`10^{5}`) would have its small-scale members zeroed
out by L1 regardless of signal.  Both steps therefore run inside a
:class:`~sklearn.pipeline.Pipeline` with a
:class:`~sklearn.preprocessing.StandardScaler`, which also refits the scaler
within each CV fold so no scaling statistics cross the split boundary.

**Stationarity of the residual, reported not assumed.**  Fitting a regression
does not establish cointegration.  If :math:`u_t` carries a unit root, the
"equilibrium" is an artifact and the ECM rests on nothing.  The residual is
therefore tested two ways and both land in
:attr:`~aieng.forecasting.evaluation.prediction.Prediction.metadata`:

- ``adf_pvalue`` — a plain augmented Dickey-Fuller test on :math:`u_t`.
- ``coint_pvalue`` — the Engle-Granger test (:func:`statsmodels.tsa.stattools.coint`).

The two differ, and the difference matters.  Standard ADF critical values
assume the tested series is *observed*; :math:`u_t` is instead *estimated*
from a fitted regression, which shifts the null distribution.  Applied to
regression residuals, ADF is **anti-conservative** — it reports cointegration
more readily than it should.  The Engle-Granger test uses MacKinnon critical
values built for exactly this case and is the one to believe.  ``adf_pvalue``
is retained as a diagnostic, not as a valid hypothesis test.  Neither figure
is exact here anyway, because ElasticNet's data-dependent variable selection
distorts both null distributions further; treat them as flags, not proofs.

Non-stationary residuals never raise.  The forecast is still returned and
``cointegration_warning`` is set, leaving the caller to decide.

Known limitations
-----------------
**Covariates are held flat when forecasting forward.**  The short-run equation
needs :math:`\Delta x` at each future step, which is unknown.  By default
(``covariate_diff_path="zero"``) future covariate differences are set to zero,
i.e. every covariate is assumed to follow a random walk and stay at its last
observed level.  ``"last"`` instead repeats the most recent observed
difference at every step — be aware this extrapolates a single day's move
linearly, so over 21 steps one large last move produces a large drift.
Either way the covariate path is an assumption, not a forecast, and its error
is not reflected in the predictive intervals.

**Single-equation, so weak exogeneity is assumed.**  Estimating one equation
rather than a full VECM presumes the covariates are weakly exogenous for the
long-run parameters — that they do not themselves adjust to the equilibrium
error.  For a panel containing Brent or the crack spread, both plainly
co-determined with WTI, that assumption is violated in principle.  The
practical consequence is bias in :math:`\beta`; a VECM would be the
principled fix and is out of scope here.

**Intervals are Gaussian and scale as** :math:`\sqrt{h}`.  Uncertainty comes
from the in-sample short-run residual standard deviation widened by
:math:`\sqrt{h}`, which assumes homoskedastic, serially uncorrelated
:math:`\varepsilon`.  Oil returns are neither.  Expect intervals that are too
narrow in high-volatility regimes.

Usage
-----
::

    from aieng.forecasting.methods import ErrorCorrectionRegressionPredictor
    from aieng.forecasting.evaluation import backtest

    predictor = ErrorCorrectionRegressionPredictor(
        covariate_series_ids=["brent_log_ret_1b_l1b", "vix_level_l1b"],
    )
    result = backtest(predictor=predictor, spec=spec, data_service=svc)

    # Inspect the cointegration diagnostics carried on each prediction.
    meta = result.predictions[0].metadata
    print(meta["coint_pvalue"], meta["ecm_coefficient"])
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np
import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES, ContinuousForecast, Prediction
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask


_TARGET_COLUMN = "__target__"

DEFAULT_MIN_OBSERVATIONS = 120
"""Minimum aligned observations required to fit.

Set by the binding constraint, which is ADF power: the residual stationarity
test is the point of the design, and ADF has poor power below roughly 100
observations — it would fail to reject a unit root even under real
cointegration, making the flag noise.  ``TimeSeriesSplit(n_splits=5)`` also
needs workable folds (at n=120 the first fold trains on ~20 rows), and a
14-covariate panel puts ~15 regressors in the short-run step, so 120 leaves
~8 observations each.  120 business days is about six months and sits well
inside the ``warmup: 250`` used by the daily WTI specs.
"""

DEFAULT_ADF_PVALUE_THRESHOLD = 0.10
"""Residuals with ``p`` above this are flagged as not credibly stationary."""


class ErrorCorrectionRegressionPredictor(Predictor):
    """Engle-Granger two-step ECM with ElasticNet long-run and short-run steps.

    Forecasts the **level** of the target by iterating the short-run equation
    forward to ``max(task.horizons)``, re-deriving the equilibrium error at
    each step from the updated target level.

    Parameters
    ----------
    covariate_series_ids : list[str]
        Series ids fetched from the :class:`ForecastContext` and used as the
        long-run regressors (on levels) and short-run regressors (on
        differences).  Required, and must contain at least one id — an ECM
        with no covariates has no equilibrium relation to correct toward.
        ElasticNet's L1 component selects among them, so the list does not
        need to be pre-pruned; note however that the cointegration
        diagnostics then describe only the *selected* subset, not everything
        passed in.
    use_log_levels : bool
        When ``True``, the target and every covariate are converted to natural
        logs before fitting, so the long-run relation is a constant-elasticity
        one and forecasts are strictly positive.  Raises ``ValueError`` at fit
        time if any of those series contains a non-positive value rather than
        silently dropping or clipping rows.  Note this makes the flag
        unusable with log-return covariates, which are routinely negative.
        Default: ``False``.
    min_observations : int
        Minimum aligned rows required.  Below this, ``predict`` raises
        ``ValueError`` and the harness skips the origin.  Default:
        :data:`DEFAULT_MIN_OBSERVATIONS`.
    cv_splits : int
        Number of :class:`~sklearn.model_selection.TimeSeriesSplit` folds used
        to select the ElasticNet penalty in both steps.  Default: 5.
    l1_ratios : tuple[float, ...]
        Candidate ElasticNet mixing parameters.  Values near 1 favour lasso-like
        selection, near 0 favour ridge-like shrinkage.
    n_alphas : int
        Penalty-path length searched per ``l1_ratio``.  Default: 50.
    adf_pvalue_threshold : float
        Threshold above which residuals are flagged non-stationary.  Default:
        :data:`DEFAULT_ADF_PVALUE_THRESHOLD`.
    max_iter : int
        Coordinate-descent iteration cap for ElasticNet.  Default: 10000.
    covariate_diff_path : {"zero", "last"}
        How future covariate differences are assumed to evolve.  ``"zero"``
        (default) freezes covariates at their last observed level;  ``"last"``
        repeats the most recent observed difference at every step, which
        extrapolates one day's move linearly across the horizon.

    Raises
    ------
    ValueError
        If ``covariate_series_ids`` is empty; if fewer than
        ``min_observations`` aligned rows are available; or if
        ``use_log_levels`` is set and a non-positive value is present.
    """

    def __init__(
        self,
        covariate_series_ids: list[str],
        *,
        use_log_levels: bool = False,
        min_observations: int = DEFAULT_MIN_OBSERVATIONS,
        cv_splits: int = 5,
        l1_ratios: tuple[float, ...] = (0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0),
        n_alphas: int = 50,
        adf_pvalue_threshold: float = DEFAULT_ADF_PVALUE_THRESHOLD,
        max_iter: int = 10_000,
        covariate_diff_path: Literal["zero", "last"] = "zero",
    ) -> None:
        if not covariate_series_ids:
            raise ValueError(
                "ErrorCorrectionRegressionPredictor requires at least one covariate series id; "
                "an error-correction model with no covariates has no long-run relation."
            )
        self._covariate_series_ids = list(covariate_series_ids)
        self._use_log_levels = use_log_levels
        self._min_observations = min_observations
        self._cv_splits = cv_splits
        self._l1_ratios = tuple(l1_ratios)
        self._n_alphas = n_alphas
        self._adf_pvalue_threshold = adf_pvalue_threshold
        self._max_iter = max_iter
        self._covariate_diff_path = covariate_diff_path

    @property
    def predictor_id(self) -> str:
        """Return a stable identifier, suffixed ``_log`` under log levels."""
        suffix = "_log" if self._use_log_levels else ""
        return f"ecm_regression{suffix}"

    # ── data assembly ────────────────────────────────────────────────────────

    def _aligned_frame(self, task: ForecastingTask, context: ForecastContext) -> pd.DataFrame:
        """Inner-join target and covariates on ``timestamp``.

        An inner join is deliberate: it makes no assumption about the calendar
        grid, so a covariate carrying an off-grid stamp (a stray weekend bar,
        say) cannot break the fit the way a ``freq="B"`` reindex would.  The
        cost is that the fit is restricted to sessions where every series is
        observed.
        """
        target = context.get_series(task.target_series_id)[["timestamp", "value"]].rename(
            columns={"value": _TARGET_COLUMN}
        )
        merged = target
        for series_id in self._covariate_series_ids:
            cov = context.get_series(series_id)[["timestamp", "value"]].rename(columns={"value": series_id})
            merged = pd.merge(merged, cov, on="timestamp", how="inner")

        return merged.sort_values("timestamp").dropna().reset_index(drop=True)

    def _to_model_arrays(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(y, X)``, log-transformed when ``use_log_levels`` is set."""
        y = frame[_TARGET_COLUMN].to_numpy(dtype=float)
        x = frame[self._covariate_series_ids].to_numpy(dtype=float)

        if self._use_log_levels:
            if np.any(y <= 0):
                raise ValueError(
                    f"use_log_levels=True but target {_TARGET_COLUMN!r} contains non-positive values; "
                    "refusing to take logs. Use use_log_levels=False for series that can be <= 0."
                )
            offenders = [
                series_id for idx, series_id in enumerate(self._covariate_series_ids) if np.any(x[:, idx] <= 0)
            ]
            if offenders:
                raise ValueError(
                    f"use_log_levels=True but these covariates contain non-positive values: {offenders}. "
                    "Log-return style covariates are routinely negative and cannot be log-levelled."
                )
            y = np.log(y)
            x = np.log(x)
        return y, x

    # ── model steps ──────────────────────────────────────────────────────────

    def _build_pipeline(self) -> Any:
        """ElasticNetCV behind a StandardScaler, penalty picked on ordered folds."""
        from sklearn.linear_model import ElasticNetCV  # noqa: PLC0415
        from sklearn.model_selection import TimeSeriesSplit  # noqa: PLC0415
        from sklearn.pipeline import Pipeline  # noqa: PLC0415
        from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    ElasticNetCV(
                        l1_ratio=list(self._l1_ratios),
                        # An int here is the penalty-path length. The older
                        # ``n_alphas`` spelling is deprecated in sklearn 1.7 and
                        # removed in 1.9.
                        alphas=self._n_alphas,
                        cv=TimeSeriesSplit(n_splits=self._cv_splits),
                        max_iter=self._max_iter,
                    ),
                ),
            ]
        )

    def _stationarity_diagnostics(
        self,
        residuals: np.ndarray,
        y: np.ndarray,
        x_selected: np.ndarray,
        x_all: np.ndarray,
    ) -> dict[str, Any]:
        """Run ADF on the residual and the proper Engle-Granger test on the pair.

        Both are reported because they disagree in a predictable direction:
        ADF on estimated residuals over-rejects the unit root, so it will call
        cointegration more often than the Engle-Granger critical values do.
        ``coint_pvalue`` is the one to trust.

        The Engle-Granger test normally runs on the ElasticNet-*selected*
        covariates, since those are what the long-run relation actually uses.
        When L1 selects nothing — which is itself strong evidence against any
        relation — it falls back to the full input panel so a real p-value is
        still reported rather than a bare NaN.  ``coint_basis`` records which
        was used.
        """
        from statsmodels.tsa.stattools import adfuller, coint  # noqa: PLC0415

        adf_stat, adf_pvalue = float("nan"), float("nan")
        try:
            adf_result = adfuller(residuals, regression="c", autolag="AIC")
            adf_stat, adf_pvalue = float(adf_result[0]), float(adf_result[1])
        except (ValueError, np.linalg.LinAlgError):
            pass

        coint_basis = "selected" if x_selected.size else "all"
        coint_inputs = x_selected if x_selected.size else x_all

        coint_stat, coint_pvalue = float("nan"), float("nan")
        if coint_inputs.size:
            try:
                coint_result = coint(y, coint_inputs, trend="c", autolag="AIC")
                coint_stat, coint_pvalue = float(coint_result[0]), float(coint_result[1])
            except (ValueError, np.linalg.LinAlgError):
                pass

        adf_stationary = bool(adf_pvalue <= self._adf_pvalue_threshold) if np.isfinite(adf_pvalue) else False
        coint_stationary = bool(coint_pvalue <= self._adf_pvalue_threshold) if np.isfinite(coint_pvalue) else False
        return {
            "adf_stat": adf_stat,
            "adf_pvalue": adf_pvalue,
            "adf_stationary": adf_stationary,
            "coint_stat": coint_stat,
            "coint_pvalue": coint_pvalue,
            "coint_stationary": coint_stationary,
            "coint_basis": coint_basis,
            "adf_threshold": self._adf_pvalue_threshold,
            # Believe the Engle-Granger test; ADF on estimated residuals over-rejects.
            "cointegration_warning": not coint_stationary,
        }

    def _iterate_forward(
        self,
        *,
        long_run: Any,
        short_run: Any,
        y_last: float,
        x_last: np.ndarray,
        dx_future: np.ndarray,
        max_horizon: int,
    ) -> dict[int, float]:
        """Step the short-run equation forward, re-deriving ``u`` each step."""
        y_cur = float(y_last)
        x_cur = x_last.astype(float).copy()
        path: dict[int, float] = {}

        for step in range(1, max_horizon + 1):
            equilibrium_error = y_cur - float(long_run.predict(x_cur.reshape(1, -1))[0])
            features = np.concatenate(([equilibrium_error], dx_future))
            delta_y = float(short_run.predict(features.reshape(1, -1))[0])
            y_cur = y_cur + delta_y
            x_cur = x_cur + dx_future
            path[step] = y_cur

        return path

    # ── Predictor interface ──────────────────────────────────────────────────

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Fit the two-step ECM at the origin and forecast every requested horizon."""
        from scipy.stats import norm  # noqa: PLC0415

        frame = self._aligned_frame(task, context)
        if len(frame) < self._min_observations:
            raise ValueError(
                f"ErrorCorrectionRegressionPredictor needs at least {self._min_observations} aligned "
                f"observations across the target and {len(self._covariate_series_ids)} covariate(s); "
                f"only {len(frame)} available at as_of={context.as_of}."
            )

        y, x = self._to_model_arrays(frame)

        # Step 1 — long-run relation on levels; u is the equilibrium error.
        long_run = self._build_pipeline()
        long_run.fit(x, y)
        equilibrium_error = y - long_run.predict(x)

        coefficients = np.asarray(long_run["model"].coef_, dtype=float)
        selected_mask = coefficients != 0.0
        selected = [sid for sid, keep in zip(self._covariate_series_ids, selected_mask, strict=True) if keep]

        diagnostics = self._stationarity_diagnostics(equilibrium_error, y, x[:, selected_mask], x)

        # Step 2 — short-run dynamics on differences, driven by the lagged error.
        delta_y = np.diff(y)
        delta_x = np.diff(x, axis=0)
        lagged_error = equilibrium_error[:-1]
        short_run_features = np.column_stack([lagged_error, delta_x])

        short_run = self._build_pipeline()
        short_run.fit(short_run_features, delta_y)
        short_run_residuals = delta_y - short_run.predict(short_run_features)
        residual_std = float(np.std(short_run_residuals, ddof=1)) if len(short_run_residuals) > 1 else 0.0

        # Recover phi in original units: the pipeline fits on standardized inputs.
        scale = np.asarray(short_run["scale"].scale_, dtype=float)
        phi = float(np.asarray(short_run["model"].coef_, dtype=float)[0] / scale[0]) if scale[0] else 0.0

        dx_future = np.zeros(x.shape[1]) if self._covariate_diff_path == "zero" else delta_x[-1]
        path = self._iterate_forward(
            long_run=long_run,
            short_run=short_run,
            y_last=y[-1],
            x_last=x[-1],
            dx_future=dx_future,
            max_horizon=task.horizon,
        )

        metadata: dict[str, Any] = {
            **diagnostics,
            "n_observations": int(len(frame)),
            "covariates": list(self._covariate_series_ids),
            "selected_covariates": selected,
            "n_selected": len(selected),
            "ecm_coefficient": phi,
            # A zeroed phi means ElasticNet removed the error-correction term
            # entirely, leaving a plain differenced regression; phi >= 0 means
            # the "correction" pushes away from equilibrium rather than toward it.
            "ecm_coefficient_zeroed": bool(phi == 0.0),
            "ecm_sign_warning": bool(phi >= 0.0),
            "short_run_residual_std": residual_std,
            "use_log_levels": self._use_log_levels,
            "covariate_diff_path": self._covariate_diff_path,
        }

        offset = pd.tseries.frequencies.to_offset(task.frequency)
        issued_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        predictions: list[Prediction] = []

        for h in task.horizons:
            centre = path[h]
            sigma = residual_std * np.sqrt(h)
            quantiles = {q: float(centre + norm.ppf(q) * sigma) for q in STANDARD_QUANTILES}
            point = float(centre)

            if self._use_log_levels:
                # Quantiles survive monotone transforms exactly; the exponentiated
                # centre is the lognormal *median*, not its mean.
                quantiles = {q: float(np.exp(v)) for q, v in quantiles.items()}
                point = float(np.exp(point))

            predictions.append(
                Prediction(
                    predictor_id=self.predictor_id,
                    task_id=task.task_id,
                    issued_at=issued_at,
                    as_of=context.as_of,
                    forecast_date=(pd.Timestamp(context.as_of) + offset * h).to_pydatetime(),
                    payload=ContinuousForecast(point_forecast=point, quantiles=quantiles),
                    metadata=metadata,
                )
            )

        return predictions
