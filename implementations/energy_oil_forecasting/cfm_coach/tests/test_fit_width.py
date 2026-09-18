"""Tests for the width-scale fit: the needed-stretch maths, the omega it implies, and its interval.

Synthetic histories with a known answer: outcomes drawn so the ensemble's P10-P90 is
exactly half as wide as it should be, which means the fit must recover omega ~= 2.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

# `tests/` is deliberately not a package (see test_m1_harness.py).
from conftest import quantile_grid
from energy_oil_forecasting.cfm_coach.fit_width import (
    TRAIN_THROUGH,
    block_bootstrap_omega,
    build_rows,
    coverage_at,
    fit_horizon,
    needed_stretch,
    omega_for_coverage,
    pinball_curve,
    resolve_actual,
)


#: conftest's grid puts P10/P90 at -1.0/+1.0 spread units, so a standard normal with
#: sd = spread / 1.2816 would sit exactly on the 80% interval. Doubling it means the
#: range is half as wide as the outcomes need.
Z90 = 1.2815515655446004


def _rows(
    n: int, *, true_scale: float, spread: float = 2.0, seed: int = 3, start: date = date(2010, 1, 4)
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    for index in range(n):
        centre = 80.0
        quantiles = quantile_grid(centre, spread)
        actual = centre + rng.normal(0.0, true_scale * spread / Z90)
        records.append(
            {
                "cutoff": start + timedelta(weeks=index),
                "horizon": 5,
                "actual": actual,
                "p10": quantiles[0.1],
                "p50": quantiles[0.5],
                "p90": quantiles[0.9],
                "last_value": centre,
                "vix": 10.0 + index % 30,
                "quantiles": quantiles,
            }
        )
    return pd.DataFrame(records)


def test_needed_stretch_uses_the_half_width_on_the_side_the_price_went():
    quantiles = {0.1: 79.0, 0.5: 84.0, 0.9: 89.0}
    rows = pd.DataFrame(
        [
            {"actual": 86.0, "p10": 79.0, "p50": 84.0, "p90": 89.0, "quantiles": quantiles},  # up, inside
            {"actual": 95.0, "p10": 79.0, "p50": 84.0, "p90": 89.0, "quantiles": quantiles},  # up, 11 / 5
            {"actual": 82.0, "p10": 81.0, "p50": 84.0, "p90": 89.0, "quantiles": quantiles},  # down, 2 / 3
        ]
    )
    assert needed_stretch(rows) == pytest.approx([0.4, 2.2, 2 / 3])


def test_coverage_omega_is_the_80th_percentile_of_needed_stretch():
    stretch = np.array([0.4, 0.6, 1.33, 2.2, 2.33])
    # Four of five must cover, so the fourth-smallest stretch is the answer.
    assert omega_for_coverage(stretch) == pytest.approx(2.2)
    assert omega_for_coverage(np.array([0.5, np.inf, 0.7, 0.9, 1.1])) == pytest.approx(1.1)


def test_fit_recovers_a_known_under_width():
    rows = _rows(4_000, true_scale=2.0)
    stretch = needed_stretch(rows)
    assert omega_for_coverage(stretch) == pytest.approx(2.0, abs=0.1)
    # Outcomes twice as dispersed: P(|Z| < 1.2816 / 2) = 2 * Phi(0.6408) - 1 ~= 0.478.
    assert coverage_at(rows, 1.0) == pytest.approx(0.478, abs=0.03)
    assert coverage_at(rows, omega_for_coverage(stretch)) == pytest.approx(0.8, abs=0.01)


def test_pinball_is_minimised_near_the_true_scale():
    rows = _rows(3_000, true_scale=1.5)
    grid = np.round(np.arange(0.5, 3.0001, 0.05), 2)
    best = float(grid[int(np.argmin(pinball_curve(rows, grid)))])
    assert best == pytest.approx(1.5, abs=0.15)


def test_block_bootstrap_interval_brackets_the_estimate_and_is_reproducible():
    stretch = needed_stretch(_rows(600, true_scale=2.0))
    first = block_bootstrap_omega(stretch, block=5, resamples=300)
    again = block_bootstrap_omega(stretch, block=5, resamples=300)
    assert first == again
    assert first[0] <= omega_for_coverage(stretch) <= first[1]


def test_fit_horizon_trains_before_the_holdout_boundary_only():
    rows = _rows(900, true_scale=2.0, start=date(2008, 1, 7))
    # Make the holdout years wildly wider; a leak would drag the proposed omega up.
    late = rows["cutoff"] > TRAIN_THROUGH
    rows.loc[late, "actual"] = rows.loc[late, "p50"] + 10 * (rows.loc[late, "actual"] - rows.loc[late, "p50"])
    fit = fit_horizon(rows, 5)
    assert fit["omega_proposed"] == pytest.approx(2.0, abs=0.15)
    assert fit["holdout"]["omega_needed_for_80"] > 10
    assert fit["train"]["through"] <= str(TRAIN_THROUGH)


def test_resolve_actual_applies_the_live_staleness_rule():
    prices = pd.Series([80.0, 81.0, 90.0], index=[date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 20)])
    assert resolve_actual(prices, date(2020, 1, 4)) == 81.0  # weekend target -> Friday close
    assert resolve_actual(prices, date(2020, 1, 15)) is None  # 12 days stale
    assert resolve_actual(prices, date(2020, 2, 1)) is None  # beyond the data


def test_build_rows_excludes_degraded_and_errored_origins():
    quantiles = {str(level): value for level, value in quantile_grid(80.0, 2.0).items()}
    forecast = {"5": {"forecast_date": "2020-01-10", "ensemble": quantiles, "components": {}}}
    base = {"status": "ok", "failed_models": [], "diagnostics": {"latest_value": 80.0}, "forecasts": forecast}
    history = [
        {**base, "cutoff": "2020-01-03"},
        {**base, "cutoff": "2020-01-06", "failed_models": ["lightgbm"]},
        {"status": "error", "cutoff": "2020-01-13"},
    ]
    prices = pd.Series([82.0], index=[date(2020, 1, 10)])
    rows, counts = build_rows(history, prices)
    assert len(rows) == 1
    assert counts["degraded"] == 1
    assert counts["errored"] == 1
