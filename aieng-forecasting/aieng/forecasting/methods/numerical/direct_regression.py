r"""Direct multi-horizon regression — one model per horizon, no recursion.

Every other numerical predictor here is *recursive*: it fits one model and
propagates it forward ``n = max(task.horizons)`` steps, reading each requested
horizon out of the resulting trajectory.  This module takes the other route.
For each horizon :math:`h` it fits a **separate** model whose target is the
:math:`h`-step log return itself

.. math:: w_t = \log p_{t+h} - \log p_t

regressed on information available at :math:`t`.  The model is trained on the
loss it is scored on, so nothing compounds across steps.

Three specifications
--------------------
One class, three registry rows, selected by what is passed in:

- ``direct_ar`` — lagged own log returns only.  The univariate baseline, and
  the direct counterpart of
  :class:`~aieng.forecasting.methods.numerical.darts_arima.DartsAutoARIMAPredictor`.
- ``direct_ar_cov`` — adds the covariate panel, each series taken at :math:`t`.
- ``direct_ecm`` — adds the equilibrium error :math:`u_t` from a long-run
  levels regression, the direct counterpart of
  :class:`~aieng.forecasting.methods.numerical.error_correction_regression.ErrorCorrectionRegressionPredictor`.

Why direct removes two documented ECM weaknesses
------------------------------------------------
The recursive ECM lists both in its own module docstring:

- *"Covariates are held flat when forecasting forward."*  A recursive model
  needs :math:`\Delta x` at every future step and has to assume one.  A direct
  model conditions only on :math:`t`, so **no future covariate path is
  required** and that assumption disappears rather than being improved.
- *"Intervals are Gaussian and scale as* :math:`\sqrt{h}`\ *."*  Here the band
  is bootstrapped from the model's own residuals, so it inherits their actual
  skew and tail weight instead of assuming away both.

Overlapping targets
-------------------
Consecutive :math:`w_t` share :math:`h - 1` days of price history.  Two
consequences, both handled deliberately:

- The overlap induces an MA(:math:`h-1`) error structure *by construction*,
  even under a white-noise price process.  Order selection would dutifully
  "discover" it, so the moving-average order is **fixed**, never searched --
  and defaults to zero on cost grounds (see :data:`DEFAULT_MA_ORDER`).
- Standard errors and model-implied intervals are invalid under that
  dependence.  Neither is used: the band comes from the residual bootstrap.

Leakage
-------
A training row at :math:`t` has a target reaching :math:`t + h`, so the final
:math:`h` rows of any history window are unusable — their outcomes have not
happened yet.  :func:`_build_design` drops them.  This is the module's most
important invariant and is pinned by the leakage tests in
``tests/.../test_direct_regression.py``.

Usage::

    from aieng.forecasting.methods.numerical import DirectRegressionPredictor

    univariate = DirectRegressionPredictor()
    with_cov = DirectRegressionPredictor(covariate_series_ids=COVARIATES)
    direct_ecm = DirectRegressionPredictor(
        covariate_series_ids=COVARIATES,
        ec_series_ids=LEVEL_SERIES,
    )
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES, ContinuousForecast, Prediction
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask


DEFAULT_AR_LAGS = 10
"""Lagged daily log returns used as regressors."""

DEFAULT_MA_ORDER = 0
"""Moving-average order for the regression errors when none is given.

Zero, i.e. plain OLS, on measured cost.  The overlap induces an MA(``h-1``)
error structure, and modelling it with SARIMAX is the textbook-efficient choice
-- but on this data it costs **29.1s per origin against 0.1s**, a factor of
290, which over a 129-origin dense grid is 62 minutes per predictor instead of
20 seconds.  (An earlier estimate of ~0.6s per fit was measured on synthetic
iid residuals, which converge in a couple of iterations; real overlapping
targets are autocorrelated by construction, which is precisely why the MLE
needs many more.)

What that buys is efficiency in the point estimate only.  OLS remains
*consistent* under correlated errors, and the band is bootstrapped from
residuals rather than model-implied, so nothing here depends on the error
covariance being right.  A 290x bill for a second-order gain on the centre is
not worth paying across a grid sweep.

Pass ``ma_order=1`` (or higher) to turn it back on when studying a single
origin, where the cost is seconds rather than hours.  Every prediction records
the order actually used in ``metadata["ma_order"]``, so a run is never
ambiguous about which it was.
"""

DEFAULT_BOOTSTRAP_DRAWS = 4000
"""Resamples drawn when building the predictive band."""

MAX_POOLED_SAMPLES = 200_000
"""Ceiling on the pooled bootstrap sample used to read quantiles off.

Drawing ``draws`` full-length resamples from ``n`` residuals materializes a
``draws x n`` array: at 4000 x 2500 that is 80 MB and ~1.8s, to extract eleven
numbers.  The quantiles of 200k pooled samples and of 10M agree to well inside
their own sampling error, so the array is capped.  The cap binds only when the
residual sample is already large, which is exactly where the bootstrap adds
least over the plain empirical quantiles.
"""

MIN_TRAIN_ROWS = 120
"""Fewest training rows accepted before refusing to fit a horizon.

Matches ``ErrorCorrectionRegressionPredictor.DEFAULT_MIN_OBSERVATIONS`` so the
two families decline on the same grounds rather than one quietly fitting a
model the other would have rejected.
"""


@dataclass(frozen=True)
class ResidualCalibration:
    """Externally supplied residual quantiles per horizon, in log-return space.

    An optional override for the built-in residual bootstrap, kept for the case
    where quantiles should come from a dedicated out-of-sample walk rather than
    from training residuals.  ``scripts/calibrate_direct_arima.py`` writes the
    file this reads.

    Parameters
    ----------
    offsets : dict[int, dict[float, float]]
        ``{horizon: {quantile_level: log-return offset}}``.
    """

    offsets: dict[int, dict[float, float]]

    @classmethod
    def from_json(cls, path: str | Path) -> "ResidualCalibration":
        """Load a calibration file, coercing JSON string keys back to numbers."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        offsets = {
            int(horizon): {float(level): float(value) for level, value in levels.items()}
            for horizon, levels in raw["offsets"].items()
        }
        return cls(offsets=offsets)

    def has(self, horizon: int) -> bool:
        """Return whether calibrated offsets exist for ``horizon``."""
        return horizon in self.offsets

    def offsets_for(self, horizon: int) -> dict[float, float]:
        """Return the residual offsets for ``horizon``, sorted by level."""
        return dict(sorted(self.offsets[horizon].items()))


def _aligned_frame(
    task: ForecastingTask,
    context: ForecastContext,
    series_ids: list[str],
) -> pd.DataFrame:
    """Inner-join the target with ``series_ids`` on ``timestamp``.

    Mirrors ``ErrorCorrectionRegressionPredictor._aligned_frame``: an inner join
    makes no assumption about the calendar grid, at the cost of restricting the
    fit to sessions where every series is observed.
    """
    frame = context.get_series(task.target_series_id)[["timestamp", "value"]].rename(columns={"value": "__price__"})
    for series_id in series_ids:
        column = context.get_series(series_id)[["timestamp", "value"]].rename(columns={"value": series_id})
        frame = pd.merge(frame, column, on="timestamp", how="inner")
    return frame.sort_values("timestamp").dropna().reset_index(drop=True)


def _log_prices(frame: pd.DataFrame, *, target_series_id: str) -> pd.Series:
    """Return log prices indexed by timestamp, dropping non-positive prints.

    WTI printed negative in April 2020, so this is a live concern on the very
    series this predictor targets.  Dropping rather than clipping follows the
    reasoning in ``error_correction_regression``: flooring -$37.63 to $1
    manufactures an enormous artificial move in the differenced series.
    """
    prices = pd.Series(
        frame["__price__"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(frame["timestamp"]),
    ).dropna()

    positive = prices > 0.0
    if not bool(positive.all()):
        warnings.warn(
            f"DirectRegressionPredictor: dropping {int((~positive).sum())} non-positive price observation(s) "
            f"from '{target_series_id}' before the log transform.",
            RuntimeWarning,
            stacklevel=2,
        )
        prices = prices[positive]
    return np.log(prices)


def _equilibrium_error(log_prices: pd.Series, levels: pd.DataFrame) -> pd.Series:
    r"""Return the long-run residual :math:`u_t` of ``log_prices`` on ``levels``.

    The Engle-Granger first step: a levels-on-levels regression whose residual
    is the deviation from long-run equilibrium.  Regularized and standardized
    for the reasons set out at length in ``error_correction_regression`` — a
    levels-on-levels fit between trending series is the textbook spurious
    regression, and an unscaled L1 penalty is really a penalty on units.

    Unlike the recursive ECM this does **not** test the residual for
    stationarity.  There it matters because the whole forecast rests on the
    error-correction dynamics; here :math:`u_t` is one regressor among many and
    a non-stationary one is simply a weak one, shrunk accordingly in step two.
    """
    from sklearn.linear_model import ElasticNetCV  # noqa: PLC0415
    from sklearn.model_selection import TimeSeriesSplit  # noqa: PLC0415
    from sklearn.pipeline import Pipeline  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    x = levels.to_numpy(dtype=float)
    y = log_prices.to_numpy(dtype=float)

    # TimeSeriesSplit, never plain k-fold: ordinary k-fold trains on future
    # blocks to score earlier ones, tuning the penalty with hindsight.
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("fit", ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=TimeSeriesSplit(n_splits=5), max_iter=5000)),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x, y)
    return pd.Series(y - model.predict(x), index=log_prices.index)


def _build_design(
    log_prices: pd.Series,
    *,
    horizon: int,
    ar_lags: int,
    extra: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the direct-regression design for one horizon.

    Parameters
    ----------
    log_prices : pd.Series
        Cutoff-scoped log prices, ascending by date.
    horizon : int
        Forecast horizon ``h`` in frequency units.
    ar_lags : int
        Lagged daily log returns used as regressors.
    extra : pd.DataFrame or None
        Additional regressors indexed like ``log_prices``, each column already
        known at ``t`` (the covariate panel is pre-lagged one business day, and
        the equilibrium error is computed from contemporaneous levels).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(X, y, x_origin)``.  ``x_origin`` is the regressor row at the forecast
        origin, whose values *are* known at the cutoff even though its target is
        not.

    Notes
    -----
    Row ``t`` pairs regressors known at ``t`` with ``log(p_{t+h}) - log(p_t)``.
    The last ``horizon`` rows are dropped: their targets extend past the end of
    the available history, so including them would be lookahead, not noise.
    """
    returns = log_prices.diff().dropna()
    if len(returns) < ar_lags + 1:
        raise ValueError(f"need at least {ar_lags + 1} returns to build {ar_lags} lags; got {len(returns)}")

    # Column j holds the return at t-j, so every column is known at time t.
    design = pd.concat({f"lag_{j}": returns.shift(j) for j in range(ar_lags)}, axis=1)
    if extra is not None and not extra.empty:
        design = design.join(extra, how="left")
    design = design.dropna()
    if design.empty:
        raise ValueError(f"no rows survive lag construction at horizon {horizon}.")

    x_origin = design.to_numpy(dtype=float)[-1]

    aligned = log_prices.reindex(design.index)
    # The shift leaves the final `horizon` rows NaN -- exactly the rows whose
    # outcome has not happened yet.
    target = aligned.shift(-horizon) - aligned

    usable = target.notna().to_numpy()
    return design.to_numpy(dtype=float)[usable], target.to_numpy(dtype=float)[usable], x_origin


def _fit_and_forecast(
    features: np.ndarray,
    values: np.ndarray,
    x_origin: np.ndarray,
    *,
    ma_order: int,
) -> tuple[float, np.ndarray]:
    """Fit the direct regression; return the origin forecast and residuals.

    SARIMAX with a fixed MA(``ma_order``) error term accounts for the
    overlap-induced dependence.  If ``statsmodels`` is missing or the fit fails
    to converge, this falls back to ordinary least squares: OLS stays consistent
    for the conditional mean under correlated errors, and only efficiency and
    the model-implied intervals suffer — and the intervals are not used.
    """
    if ma_order > 0:
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX  # noqa: PLC0415

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = SARIMAX(
                    values,
                    exog=features,
                    order=(0, 0, ma_order),
                    trend="c",
                    enforce_invertibility=False,
                ).fit(disp=False)
                forecast = fitted.forecast(steps=1, exog=x_origin.reshape(1, -1))
            return float(np.asarray(forecast).ravel()[0]), np.asarray(fitted.resid, dtype=float)
        except (ImportError, ValueError, np.linalg.LinAlgError):
            pass

    design = np.column_stack([np.ones(len(features)), features])
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    residuals = values - design @ coefficients
    return float(np.concatenate([[1.0], x_origin]) @ coefficients), residuals


def _bootstrap_offsets(
    residuals: np.ndarray,
    *,
    block: int,
    draws: int,
    rng: np.random.Generator,
) -> dict[float, float]:
    """Return predictive quantile offsets by moving-block bootstrap.

    Blocks of length ``block`` preserve the serial dependence the overlapping
    targets induce.  Note this mainly buys honesty about the *shape* of the
    error distribution: for a single horizon's marginal quantiles, and with
    training residuals numbering in the thousands, a block bootstrap converges
    to the plain empirical quantiles of ``residuals``.  It is used because it
    stays correct when the sample is small, not because it changes the answer
    when the sample is large.

    Parameter-estimation uncertainty is excluded.  With a few thousand rows
    against roughly a dozen coefficients it is negligible beside residual
    noise, and including it would mean refitting per draw.
    """
    finite = residuals[np.isfinite(residuals)]
    if len(finite) == 0:
        raise ValueError("no finite residuals to bootstrap.")

    block = max(1, min(block, len(finite)))
    # Cap total pooled samples rather than always drawing `draws` full-length
    # resamples; see MAX_POOLED_SAMPLES.
    target = min(draws * len(finite), MAX_POOLED_SAMPLES)
    n_blocks = max(1, int(np.ceil(target / block)))
    starts = rng.integers(0, len(finite) - block + 1, size=n_blocks)
    pooled = finite[(starts[:, None] + np.arange(block)[None, :]).ravel()]
    return {level: float(np.quantile(pooled, level)) for level in STANDARD_QUANTILES}


class DirectRegressionPredictor(Predictor):
    """Direct multi-horizon predictor — one regression per horizon.

    Parameters
    ----------
    ar_lags : int
        Lagged daily log returns used as regressors.  Default: 10.
    covariate_series_ids : list[str] or None
        Panel series added as regressors at ``t``.  The project's WTI panel is
        already lagged one business day, so its values are known at ``t``.
    ec_series_ids : list[str] or None
        Level series defining the long-run relation.  When set, the equilibrium
        error from that regression joins the design and ``predictor_id``
        becomes ``direct_ecm``.  These should be *levels*; a panel of log
        returns carries no long-run relation to a price level.
    ma_order : int or None
        Fixed MA order for the regression errors.  ``None`` (default) means
        :data:`DEFAULT_MA_ORDER`, i.e. plain OLS — see that constant for why.
        A supplied order is clipped to ``h - 1``, the most the overlap can
        induce.  Never selected from the data: the overlap guarantees an
        MA structure, so selection would be fitting an artifact.
    bootstrap_draws : int
        Resamples used to build the band.
    calibration : ResidualCalibration or None
        Optional override replacing the bootstrap band with externally supplied
        out-of-sample residual quantiles.
    random_state : int
        Seed for the bootstrap, so a backtest is reproducible.

    Notes
    -----
    Cost scales with the number of horizons, unlike the recursive predictors
    which fit once regardless.  Four horizons on a decade of daily data is
    still well under a second per origin.
    """

    def __init__(
        self,
        *,
        ar_lags: int = DEFAULT_AR_LAGS,
        covariate_series_ids: list[str] | None = None,
        ec_series_ids: list[str] | None = None,
        ma_order: int | None = None,
        bootstrap_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
        calibration: ResidualCalibration | None = None,
        random_state: int = 42,
    ) -> None:
        if ar_lags < 1:
            raise ValueError("ar_lags must be positive.")
        if ma_order is not None and ma_order < 0:
            raise ValueError("ma_order must be non-negative or None.")
        self._ar_lags = ar_lags
        self._covariate_series_ids = list(covariate_series_ids or [])
        self._ec_series_ids = list(ec_series_ids or [])
        self._ma_order = ma_order
        self._bootstrap_draws = bootstrap_draws
        self._calibration = calibration
        self._random_state = random_state

    @property
    def predictor_id(self) -> str:
        """Return an identifier naming the specification actually configured.

        Cached backtest files key on this, so the three variants must not share
        one id — that is how a covariate-bearing run silently reloads a
        univariate cache.
        """
        if self._ec_series_ids:
            return "direct_ecm"
        if self._covariate_series_ids:
            return "direct_ar_cov"
        return "direct_ar"

    def _resolve_ma_order(self, horizon: int) -> int:
        """Return the fixed MA order used at ``horizon``."""
        if self._ma_order is not None:
            return min(self._ma_order, horizon - 1)
        return DEFAULT_MA_ORDER

    def _prepare(self, task: ForecastingTask, context: ForecastContext) -> tuple[pd.Series, pd.DataFrame]:
        """Return cutoff-scoped log prices and the extra regressor columns."""
        needed = list(dict.fromkeys(self._covariate_series_ids + self._ec_series_ids))
        frame = _aligned_frame(task, context, needed)
        log_prices = _log_prices(frame, target_series_id=task.target_series_id)

        indexed = frame.set_index(pd.DatetimeIndex(frame["timestamp"])).reindex(log_prices.index)
        extra = pd.DataFrame(index=log_prices.index)
        for series_id in self._covariate_series_ids:
            extra[series_id] = indexed[series_id].astype(float)
        if self._ec_series_ids:
            levels = indexed[self._ec_series_ids].astype(float)
            extra["__ec__"] = _equilibrium_error(log_prices, levels)
        return log_prices, extra

    def _quantiles(
        self,
        *,
        last_price: float,
        log_return: float,
        horizon: int,
        residuals: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[dict[float, float], str]:
        """Map a log-return forecast to price-space quantiles.

        Returns the quantiles and the source used, which lands in metadata so a
        run's bands can be traced to how they were produced.
        """
        if self._calibration is not None and self._calibration.has(horizon):
            offsets = self._calibration.offsets_for(horizon)
            source = "calibration_file"
        else:
            offsets = _bootstrap_offsets(
                residuals,
                block=horizon,
                draws=self._bootstrap_draws,
                rng=rng,
            )
            source = "residual_bootstrap"

        quantiles = {level: last_price * float(np.exp(log_return + offset)) for level, offset in offsets.items()}
        # Bootstrap noise can invert adjacent levels; enforce non-crossing.
        running = float("-inf")
        for level in sorted(quantiles):
            running = max(running, quantiles[level])
            quantiles[level] = running
        return quantiles, source

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Produce a direct forecast for every horizon in the task.

        Returns
        -------
        list[Prediction]
            One ``ContinuousForecast`` per horizon in ``task.horizons``, with
            ``point_forecast`` in price space and a bootstrapped band.
            ``metadata`` records the band's source, the fitted log return, the
            fixed MA order, and the training-row count.
        """
        log_prices, extra = self._prepare(task, context)
        last_price = float(np.exp(log_prices.to_numpy(dtype=float)[-1]))
        rng = np.random.default_rng(self._random_state)

        offset = pd.tseries.frequencies.to_offset(task.frequency)
        issued_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        predictions: list[Prediction] = []

        for horizon in task.horizons:
            features, values, x_origin = _build_design(
                log_prices,
                horizon=horizon,
                ar_lags=self._ar_lags,
                extra=extra,
            )
            if len(values) < MIN_TRAIN_ROWS:
                raise ValueError(f"horizon {horizon}: {len(values)} training rows available, need {MIN_TRAIN_ROWS}.")

            ma_order = self._resolve_ma_order(horizon)
            log_return, residuals = _fit_and_forecast(features, values, x_origin, ma_order=ma_order)
            quantiles, band_source = self._quantiles(
                last_price=last_price,
                log_return=log_return,
                horizon=horizon,
                residuals=residuals,
                rng=rng,
            )

            predictions.append(
                Prediction(
                    predictor_id=self.predictor_id,
                    task_id=task.task_id,
                    issued_at=issued_at,
                    as_of=context.as_of,
                    forecast_date=(pd.Timestamp(context.as_of) + offset * horizon).to_pydatetime(),
                    payload=ContinuousForecast(
                        point_forecast=last_price * float(np.exp(log_return)),
                        quantiles=quantiles,
                    ),
                    metadata={
                        "band_source": band_source,
                        "log_return_forecast": log_return,
                        "ma_order": ma_order,
                        "train_rows": int(len(values)),
                        "n_regressors": int(features.shape[1]),
                    },
                )
            )

        return predictions


__all__ = ["DirectRegressionPredictor", "ResidualCalibration"]
