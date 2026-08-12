"""Smoke tests for ``aieng.forecasting.methods.numerical.darts_arima``.

AutoARIMA is comparatively slow to fit, so the synthetic series here is kept
short (monthly, ~8 years) and the horizon single-step — enough to exercise the
predictor id, quantile shape, and the log-levels transform without paying for
a large grid search on every test run.
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
from aieng.forecasting.methods.numerical import DartsAutoARIMAPredictor


HORIZON = 3
AS_OF = datetime(2019, 6, 1)


class _InMemoryAdapter(BaseAdapter):
    """Adapter that returns a supplied DataFrame unchanged."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df.copy()

    def fetch(self) -> pd.DataFrame:
        """Return the supplied DataFrame."""
        return self._df.copy()


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
            frequency="MS",
        ),
    )


@pytest.fixture
def positive_service() -> DataService:
    """Build a strictly positive trending random walk, safe under log levels."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2000-01-01", periods=100, freq="MS")
    values = 50.0 + np.cumsum(rng.normal(0.1, 1.0, len(dates)))
    values = np.clip(values, 1.0, None)  # keep strictly positive
    service = DataService()
    _register(service, "target", dates, values)
    return service


@pytest.fixture
def signed_service() -> DataService:
    """Build a series that crosses zero, so log levels must refuse it."""
    rng = np.random.default_rng(1)
    dates = pd.date_range("2000-01-01", periods=100, freq="MS")
    values = rng.normal(0, 1.0, len(dates))  # signed, routinely <= 0
    service = DataService()
    _register(service, "target", dates, values)
    return service


@pytest.fixture
def task() -> ForecastingTask:
    """Single-horizon monthly task against the synthetic target."""
    return ForecastingTask(
        task_id="synthetic_arima",
        target_series_id="target",
        horizons=[HORIZON],
        frequency="MS",
        description="Synthetic AutoARIMA test task.",
    )


def test_predictor_id_is_config_aware() -> None:
    """``predictor_id`` carries the log-levels flag, as it does for the ECM."""
    assert DartsAutoARIMAPredictor().predictor_id == "darts_autoarima"
    assert DartsAutoARIMAPredictor(use_log_levels=True).predictor_id == "darts_autoarima_log"


def test_returns_valid_probabilistic_forecast(positive_service: DataService, task: ForecastingTask) -> None:
    """Shape, id, date and quantile grid match what the harness scores."""
    preds = DartsAutoARIMAPredictor(num_samples=200).predict(task, positive_service.context(AS_OF))
    assert len(preds) == 1

    pred = preds[0]
    assert pred.predictor_id == "darts_autoarima"
    assert pred.forecast_date == (pd.Timestamp(AS_OF) + pd.DateOffset(months=HORIZON)).to_pydatetime()

    quantiles = pred.payload.quantiles
    assert set(STANDARD_QUANTILES).issubset(quantiles)
    values = [quantiles[q] for q in sorted(quantiles)]
    assert all(a <= b + 1e-9 for a, b in zip(values, values[1:], strict=False)), "Quantiles not monotone."
    assert quantiles[0.95] - quantiles[0.05] > 1e-6, "Degenerate (point) distribution."


def test_log_levels_rejects_non_positive_target(signed_service: DataService, task: ForecastingTask) -> None:
    """A signed target must fail loudly under log levels, not silently."""
    predictor = DartsAutoARIMAPredictor(use_log_levels=True)
    with pytest.raises(ValueError, match="non-positive values"):
        predictor.predict(task, signed_service.context(AS_OF))


def test_log_levels_forecasts_are_positive(positive_service: DataService, task: ForecastingTask) -> None:
    """Exponentiating the log-space samples keeps every quantile strictly positive."""
    preds = DartsAutoARIMAPredictor(num_samples=200, use_log_levels=True).predict(task, positive_service.context(AS_OF))
    pred = preds[0]
    assert pred.predictor_id == "darts_autoarima_log"
    assert pred.payload.point_forecast > 0
    assert all(v > 0 for v in pred.payload.quantiles.values())
