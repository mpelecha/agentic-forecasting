"""Tests for the calibration study's date and outcome arithmetic.

These functions decide which origins are usable and what the regression target
is. An off-by-one here would not crash anything — it would quietly produce a
table that looks fine and answers the wrong question. Hence the tests.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cfm_calibration_study import (  # noqa: E402
    build_origins,
    load_completed,
    load_rows,
    outcome_returns,
    write_csv,
)


@pytest.fixture
def prices() -> pd.DataFrame:
    """Return 100 business days priced 100, 101, 102, ... for exact arithmetic."""
    dates = pd.bdate_range("2024-01-01", periods=100)
    return pd.DataFrame({"timestamp": dates, "value": np.arange(100.0) + 100.0})


def test_origins_are_strided(prices: pd.DataFrame) -> None:
    origins = build_origins(
        prices,
        start=pd.Timestamp("2024-01-01"),
        end=pd.Timestamp("2024-03-01"),
        stride=5,
        max_horizon=1,
    )
    gaps = {(b - a).days for a, b in zip(origins, origins[1:], strict=False)}
    assert gaps == {7}  # five business days is one calendar week


def test_origins_without_enough_forward_data_are_dropped(prices: pd.DataFrame) -> None:
    """An origin too close to the end has no observable outcome, so it is unusable."""
    all_dates = pd.DatetimeIndex(sorted(prices["timestamp"].unique()))
    origins = build_origins(
        prices,
        start=all_dates[0],
        end=all_dates[-1],
        stride=1,
        max_horizon=21,
    )
    # The last 21 business days cannot supply a 21-day outcome.
    assert origins[-1] == all_dates[-22]
    assert all_dates[-1] not in origins


def test_outcome_returns_count_business_days_forward(prices: pd.DataFrame) -> None:
    """Horizon h must land on the h-th observation strictly after the origin."""
    all_dates = pd.DatetimeIndex(sorted(prices["timestamp"].unique()))
    origin = all_dates[10]
    base_price = 110.0  # prices increment by 1.0 per business day from 100.0

    outcomes = outcome_returns(prices, origin=origin, base_price=base_price, horizons=(1, 5, 21))

    assert outcomes["actual_price_h1"] == 111.0
    assert outcomes["actual_price_h5"] == 115.0
    assert outcomes["actual_price_h21"] == 131.0
    assert outcomes["actual_return_h5"] == pytest.approx(np.log(115.0 / 110.0))


def test_outcome_returns_are_nan_when_horizon_runs_off_the_end(prices: pd.DataFrame) -> None:
    all_dates = pd.DatetimeIndex(sorted(prices["timestamp"].unique()))
    outcomes = outcome_returns(prices, origin=all_dates[-2], base_price=198.0, horizons=(1, 21))
    assert outcomes["actual_price_h1"] == 199.0
    assert np.isnan(outcomes["actual_return_h21"])


def test_outcome_uses_origin_base_price_not_the_next_close(prices: pd.DataFrame) -> None:
    """The base price is supplied by the caller from cutoff-safe data.

    This guards the cutoff boundary: the return must be measured from what was
    knowable at the origin, never from a price released afterwards.
    """
    all_dates = pd.DatetimeIndex(sorted(prices["timestamp"].unique()))
    outcomes = outcome_returns(prices, origin=all_dates[10], base_price=100.0, horizons=(1,))
    assert outcomes["actual_return_h1"] == pytest.approx(np.log(111.0 / 100.0))


def test_resume_skips_only_successful_runs(tmp_path: Path) -> None:
    """A failed origin must be retried on resume; a successful one must not."""
    path = tmp_path / "score_runs.jsonl"
    path.write_text(
        json.dumps({"as_of": "2024-01-05", "sample": 0, "status": "ok"})
        + "\n"
        + json.dumps({"as_of": "2024-01-12", "sample": 0, "status": "error"})
        + "\n",
        encoding="utf-8",
    )
    assert load_completed(path) == {("2024-01-05", 0)}


def test_resume_on_missing_file_is_empty(tmp_path: Path) -> None:
    assert load_completed(tmp_path / "absent.jsonl") == set()


def test_corrupt_line_does_not_abort_resume(tmp_path: Path) -> None:
    """A truncated final line from a hard kill must not lose the whole sweep."""
    path = tmp_path / "score_runs.jsonl"
    path.write_text(
        json.dumps({"as_of": "2024-01-05", "sample": 0, "status": "ok"}) + "\n{ truncated",
        encoding="utf-8",
    )
    assert load_completed(path) == {("2024-01-05", 0)}


def test_csv_table_is_rectangular_when_a_run_failed(tmp_path: Path) -> None:
    """Failed runs carry no features; the table must still line up."""
    jsonl = tmp_path / "score_runs.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "as_of": "2024-01-05",
                "sample": 0,
                "status": "ok",
                "features": {"score_geopolitical": 3.0},
                "outcomes": {"actual_return_h5": 0.01},
            }
        )
        + "\n"
        + json.dumps(
            {
                "as_of": "2024-01-12",
                "sample": 0,
                "status": "error",
                "error": "boom",
                "outcomes": {"actual_return_h5": -0.02},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_rows(jsonl)
    csv_path = tmp_path / "calibration_table.csv"
    write_csv(rows, csv_path)

    table = pd.read_csv(csv_path)
    assert len(table) == 2
    assert "score_geopolitical" in table.columns
    assert table.loc[0, "score_geopolitical"] == 3.0
    assert pd.isna(table.loc[1, "score_geopolitical"])
    assert table.loc[1, "status"] == "error"
