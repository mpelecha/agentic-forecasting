"""Tests for ``aieng.forecasting.methods.numerical.direct_regression``.

Beyond the usual probabilistic-predictor invariants, two checks earn their
keep here.

The **leakage** pair: the direct target ``log(p_{t+h}) - log(p_t)`` reaches
forward in time, so an off-by-``h`` slice would train on outcomes that have not
happened yet.  That mistake is invisible in the output and inflates backtest
skill, so it is pinned at the design level and again end to end through the
cutoff.

The **predictor_id** check: cached backtests key on that string, so if the
three specifications shared an id a covariate-bearing run would silently
reload a univariate cache and be reported as the same model.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.adapters.base import BaseAdapter
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES, Prediction
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.numerical import DirectRegressionPredictor, ResidualCalibration
from aieng.forecasting.methods.numerical.direct_regression import (
    MAX_MA_ORDER,
    _bootstrap_offsets,
    _build_design,
)


AS_OF = datetime(2021, 6, 1)
HORIZONS = [5, 10, 21, 63]


class _InMemoryAdapter(BaseAdapter):
    """Adapter that returns a supplied DataFrame unchanged."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df.copy()

    def fetch(self) -> pd.DataFrame:
        """Return the supplied DataFrame."""
        return self._df.copy()


def _dates(periods: int) -> pd.DatetimeIndex:
    # Must extend past AS_OF, or the cutoff tests have nothing to cut.
    return pd.date_range("2016-01-01", periods=periods, freq="B")


def _price_frame(seed: int = 7, periods: int = 1700) -> pd.DataFrame:
    """Build a business-frequency geometric random-walk price series."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0002, 0.02, periods)
    return pd.DataFrame({"timestamp": _dates(periods), "value": 60.0 * np.exp(np.cumsum(returns))})


def _return_covariate(seed: int, periods: int = 1700) -> pd.DataFrame:
    """Return a log-return style covariate: small, centred, routinely negative."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"timestamp": _dates(periods), "value": rng.normal(0.0, 0.015, periods)})


def _level_covariate(seed: int, periods: int = 1700) -> pd.DataFrame:
    """Return a level-style covariate, suitable for a long-run relation."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"timestamp": _dates(periods), "value": 50.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.018, periods)))}
    )


def _service(price: pd.DataFrame | None = None) -> DataService:
    """Register a target plus one return-style and one level-style covariate."""
    service = DataService()
    registrations = {
        "wti": (price if price is not None else _price_frame(), "USD"),
        "brent_ret": (_return_covariate(11), "log-return"),
        "brent_level": (_level_covariate(13), "USD"),
    }
    for series_id, (frame, units) in registrations.items():
        service.register(
            series_id,
            _InMemoryAdapter(frame),
            SeriesMetadata(
                series_id=series_id,
                description="Synthetic series",
                source="test",
                units=units,
                frequency="B",
            ),
        )
    return service


@pytest.fixture
def svc() -> DataService:
    """Build a DataService with a target and two covariates."""
    return _service()


def _task(horizons: list[int]) -> ForecastingTask:
    return ForecastingTask(
        task_id="wti_task",
        target_series_id="wti",
        horizons=horizons,
        frequency="B",
        description="Synthetic price forecast for unit tests.",
    )


def _assert_valid_probabilistic(pred: Prediction, horizon: int, predictor_id: str) -> None:
    assert pred.predictor_id == predictor_id
    assert pred.forecast_date == (pd.Timestamp(AS_OF) + horizon * pd.tseries.offsets.BDay()).to_pydatetime()
    quantiles = pred.payload.quantiles
    assert set(STANDARD_QUANTILES).issubset(quantiles)
    values = [quantiles[q] for q in sorted(quantiles)]
    assert all(a <= b + 1e-12 for a, b in zip(values, values[1:])), "Quantiles not monotonic."
    assert quantiles[0.95] - quantiles[0.05] > 1e-9, "Degenerate (point) distribution."
    assert pred.payload.point_forecast > 0.0, "Price-space forecast must be positive."


def test_univariate_multi_horizon(svc: DataService) -> None:
    """One valid Prediction per requested horizon, in order."""
    preds = DirectRegressionPredictor().predict(_task(HORIZONS), svc.context(AS_OF))
    assert len(preds) == len(HORIZONS)
    for pred, horizon in zip(preds, HORIZONS):
        _assert_valid_probabilistic(pred, horizon=horizon, predictor_id="direct_ar")


def test_covariate_variant(svc: DataService) -> None:
    """Covariates join the design and change the forecast."""
    task, context = _task([21]), svc.context(AS_OF)
    base = DirectRegressionPredictor().predict(task, context)[0]
    withcov = DirectRegressionPredictor(covariate_series_ids=["brent_ret"]).predict(task, context)[0]

    _assert_valid_probabilistic(withcov, horizon=21, predictor_id="direct_ar_cov")
    assert withcov.metadata["n_regressors"] == base.metadata["n_regressors"] + 1
    assert withcov.metadata["log_return_forecast"] != base.metadata["log_return_forecast"]


def test_ecm_variant(svc: DataService) -> None:
    """The equilibrium error adds one regressor and renames the predictor."""
    task, context = _task([21]), svc.context(AS_OF)
    base = DirectRegressionPredictor(covariate_series_ids=["brent_ret"]).predict(task, context)[0]
    ecm = DirectRegressionPredictor(
        covariate_series_ids=["brent_ret"],
        ec_series_ids=["brent_level"],
    ).predict(task, context)[0]

    _assert_valid_probabilistic(ecm, horizon=21, predictor_id="direct_ecm")
    assert ecm.metadata["n_regressors"] == base.metadata["n_regressors"] + 1


def test_predictor_ids_are_distinct() -> None:
    """Cached results key on this string, so the variants must not collide."""
    ids = {
        DirectRegressionPredictor().predictor_id,
        DirectRegressionPredictor(covariate_series_ids=["brent_ret"]).predictor_id,
        DirectRegressionPredictor(covariate_series_ids=["brent_ret"], ec_series_ids=["brent_level"]).predictor_id,
    }
    assert ids == {"direct_ar", "direct_ar_cov", "direct_ecm"}


def test_no_target_reaches_past_available_history() -> None:
    """No training row may have a target beyond the last available observation.

    Rows whose target lands *on* the final observations are fine — the series is
    already cutoff-scoped, so those outcomes are known.  It is rows reaching
    past the end that must be dropped.
    """
    horizon, ar_lags = 5, 10
    frame = _price_frame(seed=3, periods=400)
    log_prices = pd.Series(
        np.log(frame["value"].to_numpy(dtype=float)),
        index=pd.DatetimeIndex(frame["timestamp"]),
    )

    _, values, _ = _build_design(log_prices, horizon=horizon, ar_lags=ar_lags)

    assert len(values) == (len(log_prices) - ar_lags) - horizon

    last_origin = log_prices.index[-1 - horizon]
    expected = float(log_prices.iloc[-1] - log_prices.loc[last_origin])
    assert values[-1] == pytest.approx(expected)
    assert np.isfinite(values).all()


def test_future_data_beyond_cutoff_is_invisible(svc: DataService) -> None:
    """A spike after ``as_of`` must not move the forecast.

    End-to-end counterpart to the design-level check: this exercises the
    ``ForecastContext`` cutoff path the predictor relies on.
    """
    frame = _price_frame(seed=3)
    after_cutoff = frame["timestamp"] > pd.Timestamp(AS_OF)
    assert bool(after_cutoff.any()), "fixture must extend past the cutoff for this test to bite"

    tampered = frame.copy()
    tampered.loc[after_cutoff, "value"] *= 150.0

    predictor = DirectRegressionPredictor()
    baseline = predictor.predict(_task([21]), _service(frame).context(AS_OF))[0]
    spiked = predictor.predict(_task([21]), _service(tampered).context(AS_OF))[0]

    assert baseline.payload.point_forecast == pytest.approx(spiked.payload.point_forecast)


def test_design_row_targets_match_definition() -> None:
    """Each row pairs regressors at ``t`` with ``log(p_{t+h}) - log(p_t)``."""
    horizon, ar_lags = 3, 4
    frame = _price_frame(seed=11, periods=200)
    log_prices = pd.Series(
        np.log(frame["value"].to_numpy(dtype=float)),
        index=pd.DatetimeIndex(frame["timestamp"]),
    )
    features, values, _ = _build_design(log_prices, horizon=horizon, ar_lags=ar_lags)

    # Differencing consumes one observation and the lag stack ar_lags - 1 more,
    # so the first usable origin sits at index ar_lags.
    first_origin = log_prices.index[ar_lags]
    expected = float(log_prices.shift(-horizon).loc[first_origin] - log_prices.loc[first_origin])
    assert values[0] == pytest.approx(expected)
    assert features.shape == (len(values), ar_lags)


def test_band_comes_from_bootstrap_by_default(svc: DataService) -> None:
    """Without a calibration file the band is bootstrapped, and says so."""
    pred = DirectRegressionPredictor().predict(_task([21]), svc.context(AS_OF))[0]
    assert pred.metadata["band_source"] == "residual_bootstrap"


def test_band_width_grows_with_horizon(svc: DataService) -> None:
    """A 63-day band must be wider than a 5-day one on a random walk."""
    preds = DirectRegressionPredictor().predict(_task([5, 63]), svc.context(AS_OF))
    widths = [p.payload.quantiles[0.95] - p.payload.quantiles[0.05] for p in preds]
    assert widths[1] > widths[0]


def test_calibration_file_overrides_bootstrap(svc: DataService, tmp_path: Path) -> None:
    """A supplied calibration replaces the bootstrap and is recorded as such."""
    offsets = {level: float(np.log(1.0 + 0.1 * (level - 0.5))) for level in STANDARD_QUANTILES}
    path = tmp_path / "residuals.json"
    path.write_text(json.dumps({"offsets": {"21": {str(k): v for k, v in offsets.items()}}}), encoding="utf-8")

    predictor = DirectRegressionPredictor(calibration=ResidualCalibration.from_json(path))
    pred = predictor.predict(_task([21]), svc.context(AS_OF))[0]

    assert pred.metadata["band_source"] == "calibration_file"
    # quantile = p_T * exp(w_hat + offset) and point = p_T * exp(w_hat), so the
    # ratio recovers the offset exactly.
    for level, offset in offsets.items():
        ratio = pred.payload.quantiles[level] / pred.payload.point_forecast
        assert float(np.log(ratio)) == pytest.approx(offset, abs=1e-9)


def test_ma_order_is_capped_not_searched(svc: DataService) -> None:
    """Default MA order tracks ``h - 1`` up to the cap, and is never selected."""
    preds = DirectRegressionPredictor().predict(_task([1, 63]), svc.context(AS_OF))
    assert preds[0].metadata["ma_order"] == 0
    assert preds[1].metadata["ma_order"] == MAX_MA_ORDER


def test_bootstrap_matches_empirical_quantiles_at_scale() -> None:
    """The pooled-sample cap must not move the quantiles it exists to speed up.

    Guards the MAX_POOLED_SAMPLES optimisation: if capping the pool ever shifts
    a quantile materially, the speedup stopped being free.
    """
    rng = np.random.default_rng(0)
    residuals = rng.normal(0.0, 0.08, 2500)
    offsets = _bootstrap_offsets(residuals, block=21, draws=4000, rng=rng)
    for level in (0.05, 0.5, 0.95):
        assert offsets[level] == pytest.approx(float(np.quantile(residuals, level)), abs=2e-3)


def test_short_history_refuses_rather_than_fits() -> None:
    """Too few rows raises instead of fitting something meaningless."""
    service = _service(_price_frame(seed=5, periods=200))
    with pytest.raises(ValueError, match="training rows available"):
        DirectRegressionPredictor().predict(_task([63]), service.context(datetime(2016, 6, 1)))


def test_non_positive_prices_are_dropped() -> None:
    """Negative prints (April 2020 WTI) are dropped with a warning, not NaNed."""
    frame = _price_frame(seed=5)
    frame.loc[300, "value"] = -10.0

    with pytest.warns(RuntimeWarning, match="non-positive price"):
        preds = DirectRegressionPredictor().predict(_task([21]), _service(frame).context(AS_OF))

    assert np.isfinite(preds[0].payload.point_forecast)


def test_reproducible_across_runs(svc: DataService) -> None:
    """The bootstrap is seeded, so a backtest rerun reproduces its bands."""
    task, context = _task([21]), svc.context(AS_OF)
    first = DirectRegressionPredictor().predict(task, context)[0]
    second = DirectRegressionPredictor().predict(task, context)[0]
    assert first.payload.quantiles == second.payload.quantiles
