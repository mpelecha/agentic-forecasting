"""Tests for :class:`ErrorCorrectionRegressionPredictor`.

Two synthetic regimes drive most of the assertions:

- a **cointegrated** pair, where the target is tied to a random-walk covariate
  by a stationary error, so the diagnostics should report cointegration and a
  negative error-correction coefficient;
- **independent random walks**, where no equilibrium relation exists and the
  warning flag must fire rather than the model silently proceeding.

The second case is the one that matters: a two-step ECM will happily fit a
regression between unrelated trending series and report a flattering in-sample
fit, so the point of the stationarity check is to notice when that has
happened.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.adapters.base import BaseAdapter
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.numerical import ErrorCorrectionRegressionPredictor


AS_OF = datetime(2021, 12, 1)
N_DAYS = 500
HORIZONS = [5, 10, 21]


class _InMemoryAdapter(BaseAdapter):
    """Adapter that returns a supplied DataFrame unchanged."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df.copy()

    def fetch(self) -> pd.DataFrame:
        """Return the supplied DataFrame."""
        return self._df.copy()


def _dates(n: int = N_DAYS) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=n)


def _register(service: DataService, series_id: str, dates: pd.DatetimeIndex, values: np.ndarray) -> None:
    frame = pd.DataFrame({"timestamp": dates, "value": values, "released_at": dates})
    service.register(
        series_id,
        _InMemoryAdapter(frame),
        SeriesMetadata(
            series_id=series_id,
            description=series_id,
            source="synthetic",
            units="n/a",
            frequency="B",
        ),
    )


@pytest.fixture
def cointegrated_service() -> DataService:
    """Target tied to a random-walk covariate by a stationary (AR(1)) error."""
    rng = np.random.default_rng(0)
    dates = _dates()
    driver = 50.0 + np.cumsum(rng.normal(0, 0.5, N_DAYS))

    # Stationary equilibrium error: mean-reverting AR(1), so target - 2*driver
    # has no unit root and the pair is genuinely cointegrated.
    error = np.zeros(N_DAYS)
    for t in range(1, N_DAYS):
        error[t] = 0.85 * error[t - 1] + rng.normal(0, 0.4)
    target = 10.0 + 2.0 * driver + error

    service = DataService()
    _register(service, "target", dates, target)
    _register(service, "driver", dates, driver)
    return service


@pytest.fixture
def independent_service() -> DataService:
    """Target and covariate are unrelated random walks — no cointegration."""
    rng = np.random.default_rng(7)
    dates = _dates()
    service = DataService()
    _register(service, "target", dates, 60.0 + np.cumsum(rng.normal(0, 0.7, N_DAYS)))
    _register(service, "driver", dates, 20.0 + np.cumsum(rng.normal(0, 0.7, N_DAYS)))
    return service


@pytest.fixture
def task() -> ForecastingTask:
    """Multi-horizon business-day task against the synthetic target."""
    return ForecastingTask(
        task_id="synthetic_ecm",
        target_series_id="target",
        horizons=HORIZONS,
        frequency="B",
        description="Synthetic ECM test task.",
    )


# ---------------------------------------------------------------------------
# Interface conformance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("use_log_levels", "expected_id"),
    [(False, "ecm_regression"), (True, "ecm_regression_log")],
)
def test_predictor_id_is_config_aware(use_log_levels: bool, expected_id: str) -> None:
    """``predictor_id`` carries the log-levels flag, as ``_cov`` does for LightGBM."""
    predictor = ErrorCorrectionRegressionPredictor(["driver"], use_log_levels=use_log_levels)
    assert predictor.predictor_id == expected_id


def test_empty_covariate_list_rejected() -> None:
    """An ECM with no covariates has no long-run relation to correct toward."""
    with pytest.raises(ValueError, match="at least one covariate"):
        ErrorCorrectionRegressionPredictor([])


def test_returns_one_prediction_per_horizon_with_standard_quantiles(
    cointegrated_service: DataService, task: ForecastingTask
) -> None:
    """Shape, ids, dates and quantile grid match what the harness scores."""
    predictor = ErrorCorrectionRegressionPredictor(["driver"])
    preds = predictor.predict(task, cointegrated_service.context(AS_OF))

    assert len(preds) == len(HORIZONS)
    offset = pd.tseries.frequencies.to_offset("B")
    for pred, h in zip(preds, HORIZONS, strict=True):
        assert pred.predictor_id == "ecm_regression"
        assert pred.task_id == task.task_id
        assert pred.as_of == AS_OF
        assert pred.forecast_date == (pd.Timestamp(AS_OF) + offset * h).to_pydatetime()
        assert set(STANDARD_QUANTILES).issubset(pred.payload.quantiles)


def test_quantiles_are_monotone_and_widen_with_horizon(
    cointegrated_service: DataService, task: ForecastingTask
) -> None:
    """Intervals must be ordered, non-degenerate, and wider further out."""
    preds = ErrorCorrectionRegressionPredictor(["driver"]).predict(task, cointegrated_service.context(AS_OF))

    widths = []
    for pred in preds:
        quantiles = pred.payload.quantiles
        values = [quantiles[q] for q in sorted(quantiles)]
        assert all(a <= b + 1e-9 for a, b in zip(values, values[1:], strict=False)), "Quantiles not monotone."
        widths.append(quantiles[0.95] - quantiles[0.05])

    assert widths[0] > 0, "Degenerate (point) distribution."
    assert widths[0] < widths[1] < widths[2], "Uncertainty must grow with horizon."


def test_min_observations_is_enforced(cointegrated_service: DataService, task: ForecastingTask) -> None:
    """Too little history raises so the harness can skip the origin."""
    predictor = ErrorCorrectionRegressionPredictor(["driver"], min_observations=10_000)
    with pytest.raises(ValueError, match="at least 10000 aligned observations"):
        predictor.predict(task, cointegrated_service.context(AS_OF))


# ---------------------------------------------------------------------------
# Cointegration diagnostics
# ---------------------------------------------------------------------------


def test_cointegrated_pair_is_flagged_stationary_with_negative_phi(
    cointegrated_service: DataService, task: ForecastingTask
) -> None:
    """A genuine equilibrium relation should pass the test and pull back toward it."""
    preds = ErrorCorrectionRegressionPredictor(["driver"]).predict(task, cointegrated_service.context(AS_OF))
    meta = preds[0].metadata

    assert meta["coint_stationary"] is True
    assert meta["cointegration_warning"] is False
    assert meta["ecm_coefficient"] < 0, "Error correction must pull toward equilibrium, not away."
    assert meta["ecm_sign_warning"] is False
    assert meta["selected_covariates"] == ["driver"]


def test_independent_random_walks_raise_the_warning_not_an_error(
    independent_service: DataService, task: ForecastingTask
) -> None:
    """Spurious regression is surfaced in metadata; the forecast still returns."""
    preds = ErrorCorrectionRegressionPredictor(["driver"]).predict(task, independent_service.context(AS_OF))
    meta = preds[0].metadata

    assert len(preds) == len(HORIZONS), "Must still forecast when cointegration fails."
    assert meta["cointegration_warning"] is True
    assert meta["coint_stationary"] is False
    # L1 typically drops an unrelated covariate entirely; the diagnostic then
    # falls back to the full panel so a real p-value is still reported.
    assert np.isfinite(meta["coint_pvalue"])
    assert meta["coint_basis"] in {"selected", "all"}


def test_diagnostics_report_both_tests(cointegrated_service: DataService, task: ForecastingTask) -> None:
    """Both the raw ADF figure and the valid Engle-Granger figure are carried."""
    preds = ErrorCorrectionRegressionPredictor(["driver"]).predict(task, cointegrated_service.context(AS_OF))
    meta = preds[0].metadata

    for key in ("adf_stat", "adf_pvalue", "adf_stationary", "coint_stat", "coint_pvalue", "coint_stationary"):
        assert key in meta
    assert meta["n_observations"] == N_DAYS
    assert meta["covariate_diff_path"] == "zero"
    assert meta["use_log_levels"] is False


# ---------------------------------------------------------------------------
# Log levels
# ---------------------------------------------------------------------------


def test_log_levels_rejects_non_positive_covariate(task: ForecastingTask) -> None:
    """Log-return style covariates go negative and must fail loudly, not silently."""
    rng = np.random.default_rng(3)
    dates = _dates()
    service = DataService()
    _register(service, "target", dates, 60.0 + np.cumsum(rng.normal(0, 0.5, N_DAYS)))
    _register(service, "driver", dates, rng.normal(0, 0.01, N_DAYS))  # log returns, signed

    predictor = ErrorCorrectionRegressionPredictor(["driver"], use_log_levels=True)
    with pytest.raises(ValueError, match="non-positive values"):
        predictor.predict(task, service.context(AS_OF))


def test_log_levels_forecasts_are_positive(cointegrated_service: DataService, task: ForecastingTask) -> None:
    """Exponentiating the log-space path keeps every quantile strictly positive."""
    preds = ErrorCorrectionRegressionPredictor(["driver"], use_log_levels=True).predict(
        task, cointegrated_service.context(AS_OF)
    )
    for pred in preds:
        assert pred.payload.point_forecast > 0
        assert all(v > 0 for v in pred.payload.quantiles.values())
        assert pred.predictor_id == "ecm_regression_log"
