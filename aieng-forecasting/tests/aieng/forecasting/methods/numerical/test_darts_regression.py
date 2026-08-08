"""Smoke tests for ``aieng.forecasting.methods.darts_regression``.

One test per predictor.  Each fits with past covariates, which exercises the
full covariate path (the univariate path is a subset of the same helper).
We assert the key invariants that make a Darts-based predictor evaluable:
expected predictor id, standard quantile coverage, and monotone non-degenerate
quantiles.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.adapters.base import BaseAdapter
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES, Prediction
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.numerical import DartsLightGBMPredictor, DartsLinearRegressionPredictor


HORIZON = 6
AS_OF = datetime(2020, 12, 1)


class _InMemoryAdapter(BaseAdapter):
    """Adapter that returns a supplied DataFrame unchanged."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df.copy()

    def fetch(self) -> pd.DataFrame:
        """Return the supplied DataFrame."""
        return self._df.copy()


def _synthetic_series(seed: int, amplitude: float = 10.0) -> pd.DataFrame:
    """Build a 240-month trend+seasonal+noise series (deterministic via seed)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-01", periods=240, freq="MS")
    t = np.arange(240, dtype=float)
    values = 100.0 + 0.5 * t + amplitude * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1.0, 240)
    return pd.DataFrame({"timestamp": dates, "value": values})


@pytest.fixture
def svc() -> DataService:
    """Build a DataService with one target and two covariate series."""
    service = DataService()
    for series_id, seed, amp in [("target", 1, 10.0), ("cov_a", 2, 5.0), ("cov_b", 3, 2.0)]:
        service.register(
            series_id,
            _InMemoryAdapter(_synthetic_series(seed=seed, amplitude=amp)),
            SeriesMetadata(
                series_id=series_id,
                description=f"Synthetic {series_id}",
                source="test",
                units="index",
                frequency="MS",
            ),
        )
    return service


@pytest.fixture
def task() -> ForecastingTask:
    """Build a 6-month horizon task against the synthetic target."""
    return ForecastingTask(
        task_id="synthetic_6m",
        target_series_id="target",
        horizons=[HORIZON],
        frequency="MS",
        description="Synthetic 6-month forecast for unit tests.",
    )


def _assert_valid_probabilistic(pred: Prediction, expected_id: str) -> None:
    """Assert shape, id, date, quantile coverage and monotonicity with real spread."""
    assert pred.predictor_id == expected_id
    assert pred.forecast_date == (pd.Timestamp(AS_OF) + pd.DateOffset(months=HORIZON)).to_pydatetime()

    quantiles = pred.payload.quantiles
    assert set(STANDARD_QUANTILES).issubset(quantiles)

    values = [quantiles[q] for q in sorted(quantiles)]
    assert all(a <= b + 1e-9 for a, b in zip(values, values[1:])), "Quantiles not monotonic."
    assert quantiles[0.95] - quantiles[0.05] > 1e-6, "Degenerate (point) distribution."


def test_linear_regression_with_covariates(svc: DataService, task: ForecastingTask) -> None:
    """Linear regression yields a valid forecast with covariates."""
    preds = DartsLinearRegressionPredictor(
        lags=12,
        lags_past_covariates=12,
        covariate_series_ids=["cov_a", "cov_b"],
        num_samples=200,
    ).predict(task, svc.context(AS_OF))
    assert len(preds) == 1, "Single-horizon task should yield exactly one Prediction."
    _assert_valid_probabilistic(preds[0], "darts_linreg_cov")


def test_lightgbm_with_covariates(svc: DataService, task: ForecastingTask) -> None:
    """LightGBM predictor returns a valid probabilistic forecast with covariates."""
    preds = DartsLightGBMPredictor(
        lags=12,
        lags_past_covariates=12,
        covariate_series_ids=["cov_a", "cov_b"],
        num_samples=200,
    ).predict(task, svc.context(AS_OF))
    assert len(preds) == 1, "Single-horizon task should yield exactly one Prediction."
    _assert_valid_probabilistic(preds[0], "darts_lightgbm_cov")


# ---------------------------------------------------------------------------
# Recipe seam: variant_tag keeps distinct covariate panels off each other's cache
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("covariates", "variant_tag", "expected_id"),
    [
        (None, None, "darts_lightgbm"),
        (["cov_a"], None, "darts_lightgbm_cov"),
        (["cov_a"], "expanded", "darts_lightgbm_cov_expanded"),
        (None, "expanded", "darts_lightgbm_expanded"),
    ],
)
def test_lightgbm_predictor_id_folds_variant_tag(
    covariates: list[str] | None,
    variant_tag: str | None,
    expected_id: str,
) -> None:
    """``variant_tag`` is folded into ``predictor_id`` after the covariate suffix.

    ``cached_multi_backtest`` keys its cache on ``predictor_id`` alone (plus
    spec and task), so two covariate panels sharing an id would silently reuse
    each other's results.  The first two rows pin the pre-existing ids so
    callers that never pass a tag keep their cached backtests valid.
    """
    predictor = DartsLightGBMPredictor(covariate_series_ids=covariates, variant_tag=variant_tag)
    assert predictor.predictor_id == expected_id


def test_lightgbm_variant_tag_reaches_prediction_metadata(svc: DataService, task: ForecastingTask) -> None:
    """A tagged run records its recipe in metadata; an untagged run omits the key."""
    tagged = DartsLightGBMPredictor(
        lags=12,
        lags_past_covariates=12,
        covariate_series_ids=["cov_a", "cov_b"],
        num_samples=50,
        variant_tag="expanded",
    ).predict(task, svc.context(AS_OF))
    assert tagged[0].metadata["variant_tag"] == "expanded"
    _assert_valid_probabilistic(tagged[0], "darts_lightgbm_cov_expanded")

    untagged = DartsLightGBMPredictor(
        lags=12,
        lags_past_covariates=12,
        covariate_series_ids=["cov_a", "cov_b"],
        num_samples=50,
    ).predict(task, svc.context(AS_OF))
    assert "variant_tag" not in untagged[0].metadata
