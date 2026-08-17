"""Tests for energy/oil forecasting helper modules."""

from __future__ import annotations

import math
from datetime import datetime

import pandas as pd
import pytest
from aieng.forecasting.evaluation import BacktestSpec, ForecastingTask
from aieng.forecasting.evaluation.backtest import BacktestResult
from aieng.forecasting.evaluation.prediction import ContinuousForecast, Prediction
from energy_oil_forecasting.analysis import (
    _extract_agent_point,
    compute_brier_score,
    predictions_to_frame,
    rolling_coverage_pct,
    score_backtest_results,
)
from energy_oil_forecasting.prophet_baseline import prophet_prob_shock


def test_compute_brier_score_perfect() -> None:
    assert compute_brier_score([1.0, 0.0], [1, 0]) == 0.0


def test_compute_brier_score_worst() -> None:
    assert compute_brier_score([0.0, 1.0], [1, 0]) == 1.0


def test_rolling_coverage_pct() -> None:
    df = pd.DataFrame(
        {
            "resolution_date": pd.to_datetime(["2025-06-01", "2026-03-01"]),
            "actual_price": [70.0, 100.0],
            "inside_ci": [True, False],
        }
    )
    assert rolling_coverage_pct(df, year=2025) == 100.0
    assert rolling_coverage_pct(df, year=2026) == 0.0


# ---------------------------------------------------------------------------
# _extract_agent_point — dual-format contract
# ---------------------------------------------------------------------------


def test_extract_agent_point_reference_format() -> None:
    """Reference format: predictions list with payload dicts is parsed correctly."""
    rec = {
        "origin": "2024-01-01",
        "predictions": [
            {"payload": {"point_forecast": 85.0}, "horizon": 5},
            {"payload": {"point_forecast": 88.0}, "horizon": 10},
        ],
    }
    assert _extract_agent_point(rec, horizon_idx=0, horizon=5) == 85.0
    assert _extract_agent_point(rec, horizon_idx=1, horizon=10) == 88.0


def test_extract_agent_point_reference_format_out_of_bounds_returns_nan() -> None:
    """Horizon index beyond the predictions list returns NaN (graceful miss)."""
    rec = {"origin": "2024-01-01", "predictions": [{"payload": {"point_forecast": 85.0}}]}
    result = _extract_agent_point(rec, horizon_idx=5, horizon=5)
    assert math.isnan(result)


def test_extract_agent_point_legacy_flat_format() -> None:
    """Legacy flat format: day_N keys are read directly."""
    rec = {"origin": "2024-01-01", "day_5": 85.0, "day_10": 88.0}
    assert _extract_agent_point(rec, horizon_idx=0, horizon=5) == 85.0
    assert _extract_agent_point(rec, horizon_idx=1, horizon=10) == 88.0


def test_extract_agent_point_legacy_missing_horizon_returns_nan() -> None:
    """Missing day_N key in the legacy format returns NaN."""
    rec = {"origin": "2024-01-01", "day_5": 85.0}
    result = _extract_agent_point(rec, horizon_idx=1, horizon=21)
    assert math.isnan(result)


def test_prophet_prob_shock_high_when_mean_above_threshold() -> None:
    sub = pd.DataFrame(
        {
            "horizon": [5],
            "yhat": [80.0],
            "yhat_lower": [75.0],
            "yhat_upper": [85.0],
        }
    )
    prob = prophet_prob_shock(sub, origin_price=70.0, threshold=5.0, horizon=5)
    assert prob > 0.5


# ---------------------------------------------------------------------------
# 80% interval coverage — the band must be P10..P90, not P20..P80
#
# These pin the level explicitly because the failure is silent: a P20..P80 band
# reported as "80% coverage" still produces a plausible-looking percentage, and
# it inverts the diagnosis — an over-wide model reads as slightly too narrow.
# ---------------------------------------------------------------------------

_ACTUAL = 100.0
_FD = datetime(2025, 1, 8)


class _StubService:
    """Minimal DataService stand-in: one observation on the forecast date."""

    def get_series(self, series_id: str, as_of: datetime | None = None) -> pd.DataFrame:
        _ = series_id, as_of
        return pd.DataFrame({"timestamp": [pd.Timestamp(_FD)], "value": [_ACTUAL]})


def _result(quantiles: dict[float, float]) -> dict[str, BacktestResult]:
    task = ForecastingTask(
        task_id="t",
        target_series_id="s",
        horizons=[5],
        frequency="B",
        description="stub",
    )
    prediction = Prediction(
        predictor_id="p",
        task_id="t",
        issued_at=datetime(2025, 1, 1),
        as_of=datetime(2025, 1, 1),
        forecast_date=_FD,
        payload=ContinuousForecast(point_forecast=100.0, quantiles=quantiles),
    )
    return {
        "t": BacktestResult(
            spec=BacktestSpec(task=task, start="2025-01-01", end="2025-01-31", stride=5),
            predictor_id="p",
            predictions=[prediction],
            scores=[1.0],
            mean_score=1.0,
            ran_at=datetime(2025, 1, 1),
        )
    }


# Actual is 100. The P20..P80 band contains it; the P10..P90 band does not.
# Only a P10..P90 implementation reports 0% here.
_BAND_TRAP = {0.1: 101.0, 0.2: 99.0, 0.5: 100.0, 0.8: 101.0, 0.9: 102.0}


def test_coverage_uses_p10_p90_not_p20_p80() -> None:
    scores = score_backtest_results(_result(_BAND_TRAP), _StubService())
    assert scores["coverage_80"] == 0.0, "coverage_80 must be measured on the P10..P90 band"


def test_coverage_counts_a_hit_inside_p10_p90() -> None:
    band = {0.1: 95.0, 0.2: 97.0, 0.5: 100.0, 0.8: 103.0, 0.9: 105.0}
    scores = score_backtest_results(_result(band), _StubService())
    assert scores["coverage_80"] == 100.0


def test_prediction_frame_band_columns_use_p10_p90() -> None:
    frame = predictions_to_frame({"m": _result(_BAND_TRAP)}, _StubService())
    row = frame.iloc[0]
    assert row["inside80"] == 0.0
    assert row["width80"] == pytest.approx(1.0)  # P90 102 − P10 101
    assert row["q20"] == 99.0 and row["q80"] == 101.0  # still reported, unused for the band


def test_coverage_is_nan_when_the_band_quantiles_are_absent() -> None:
    scores = score_backtest_results(_result({0.5: 100.0}), _StubService())
    assert math.isnan(scores["coverage_80"])
