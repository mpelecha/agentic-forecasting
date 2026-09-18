"""Tests for the random-walk floor and directional scoring.

The assertions that matter most are the cutoff-honesty ones: a baseline that can see
one price after the cutoff is not a floor, it is a leak, and it would make every real
forecaster look worse than it is.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

# `tests/` is deliberately not a package (see test_m1_harness.py).
from conftest import FakeService, make_run_record, quantile_grid
from energy_oil_forecasting.cfm_coach.baselines import MIN_RANDOM_WALK_MOVES, RandomWalkBaseline
from energy_oil_forecasting.cfm_coach.config import CoachSettings
from energy_oil_forecasting.cfm_coach.ledger import CalibrationLedger
from energy_oil_forecasting.cfm_coach.outcomes import OutcomeResolver
from energy_oil_forecasting.cfm_coach.report import (
    AGENT,
    ENSEMBLE,
    RANDOM_WALK,
    CoachReporter,
    binomial_upper_tail,
    hit_rate_needed,
)
from energy_oil_forecasting.cfm_coach.run_store import RunRecordStore
from energy_oil_forecasting.cfm_coach.schemas import CalibrationVersion, ResolvedForecast
from energy_oil_forecasting.cfm_coach.scoring import FLAT_EPS, ForecastScorer, direction_of


CUTOFF = "2026-05-01"
V001 = CalibrationVersion(version="v001", effective_from=date(2004, 1, 1))


def _prices(end: str, n: int, *, level: float = 80.0, seed: int = 7) -> pd.Series:
    """Build a lognormal walk of `n` business days ending the business day before `end`."""
    dates = pd.bdate_range(end=pd.Timestamp(end) - pd.offsets.BDay(1), periods=n)
    moves = np.random.default_rng(seed).normal(0.0, 0.02, size=n)
    values = level * np.exp(np.cumsum(moves) - moves.sum())
    return pd.Series(values, index=[stamp.date() for stamp in dates])


def _resolved(record, horizon: int, realized: float) -> ResolvedForecast:
    return ResolvedForecast(
        run_id=record.run_id,
        horizon=horizon,
        cutoff=record.cutoff,
        forecast_date=record.horizon(horizon).forecast_date,
        realized_value=realized,
        realized_observation_date=record.horizon(horizon).forecast_date.date(),
        resolved_at=datetime(2026, 5, 20, 12, 0),
    )


# ------------------------------------------------------------ random walk ----


def test_random_walk_centres_on_the_last_close_and_never_crosses():
    record = make_run_record(cutoff=CUTOFF, centre=80.0)
    point, quantiles = RandomWalkBaseline(_prices(CUTOFF, 600)).for_record(record, 5)

    assert point == 80.0
    assert quantiles[0.5] == 80.0
    ordered = [quantiles[level] for level in sorted(quantiles)]
    assert ordered == sorted(ordered)
    assert quantiles[0.1] < 80.0 < quantiles[0.9]
    # Same grid as the forecast it is compared against, or pinball is not paired.
    assert set(quantiles) == set(record.horizon(5).ensemble_quantiles)


def test_random_walk_cannot_see_prices_on_or_after_the_cutoff():
    record = make_run_record(cutoff=CUTOFF)
    honest = _prices(CUTOFF, 600)
    crash_dates = [stamp.date() for stamp in pd.bdate_range(CUTOFF, periods=30)]
    leaked = pd.concat([honest, pd.Series(10.0, index=crash_dates)])

    _, clean = RandomWalkBaseline(honest).for_record(record, 21)
    _, with_future = RandomWalkBaseline(leaked).for_record(record, 21)
    assert with_future == pytest.approx(clean)


def test_random_walk_stops_at_the_records_own_latest_observation():
    """When a record says what it last saw, nothing after that date may shape the spread."""
    prices = _prices(CUTOFF, 600)
    seen_through = prices.index[-4]
    record = make_run_record(cutoff=CUTOFF)
    record = record.model_copy(
        update={"diagnostics": {"latest_value": 80.0, "latest_observation_date": seen_through.isoformat()}}
    )
    spiked = prices.copy()
    spiked.iloc[-3:] = 500.0

    _, clean = RandomWalkBaseline(prices[prices.index <= seen_through]).for_record(record, 5)
    _, guarded = RandomWalkBaseline(spiked).for_record(record, 5)
    assert guarded == pytest.approx(clean)


def test_random_walk_declines_when_history_is_too_short():
    record = make_run_record(cutoff=CUTOFF)
    short = _prices(CUTOFF, MIN_RANDOM_WALK_MOVES // 2)
    assert RandomWalkBaseline(short).for_record(record, 5) is None


def test_random_walk_spread_scales_with_the_price_level():
    """Log moves, not dollar moves: the same market at half the price has half the dollar range."""
    baseline = RandomWalkBaseline(_prices(CUTOFF, 600))
    history = baseline.prices
    levels = [0.1, 0.5, 0.9]
    high = baseline.quantiles(history, last_value=80.0, horizon=10, levels=levels)
    low = baseline.quantiles(history, last_value=40.0, horizon=10, levels=levels)
    assert (high[0.9] - high[0.1]) == pytest.approx(2.0 * (low[0.9] - low[0.1]))


# -------------------------------------------------------------- direction ----


def test_direction_treats_anything_within_the_flat_threshold_as_no_call():
    assert direction_of(FLAT_EPS / 2) == 0
    assert direction_of(-FLAT_EPS / 2) == 0
    assert direction_of(FLAT_EPS + 0.01) == 1
    assert direction_of(-(FLAT_EPS + 0.01)) == -1


@pytest.mark.parametrize(
    ("centre", "actual", "expected"),
    [
        (81.0, 82.0, True),  # called up, went up
        (81.0, 79.0, False),  # called up, went down
        (80.03, 82.0, None),  # a three-cent nudge is not a call
        (81.0, 80.02, None),  # a two-cent move cannot be called
    ],
)
def test_direction_hit_scores_calls_and_excludes_flats(centre, actual, expected):
    record = make_run_record(cutoff=CUTOFF, horizons=(5,))
    card = ForecastScorer().score_quantiles(
        quantile_grid(centre, 2.0),
        point_forecast=centre,
        resolved=_resolved(record, 5, actual),
        cutoff_variant=AGENT,
        calibration_version="v001",
        last_value=80.0,
    )
    assert card.direction_hit is expected


def test_summary_reports_always_up_on_the_same_calls():
    record = make_run_record(cutoff=CUTOFF, horizons=(5,))
    scorer = ForecastScorer()

    def card(centre: float, actual: float):
        return scorer.score_quantiles(
            quantile_grid(centre, 2.0),
            point_forecast=centre,
            resolved=_resolved(record, 5, actual),
            cutoff_variant=AGENT,
            calibration_version="v001",
            last_value=80.0,
        )

    # Two up calls that went up, one down call that went up, one flat call.
    summary = ForecastScorer.summarize([card(81, 82), card(81, 83), card(79, 81), card(80.01, 85)])
    assert summary.direction_calls == 3
    assert summary.direction_hits == 2
    assert summary.direction_no_calls == 1
    assert summary.always_up_rate == pytest.approx(1.0)
    assert summary.direction_hit_rate == pytest.approx(2 / 3)


def test_significance_bar_uses_the_binomial_tail_and_admits_when_it_is_unreachable():
    assert binomial_upper_tail(3, 3, 0.53) == pytest.approx(0.53**3)
    # Three independent observations cannot beat a 53% base rate at p<0.10, even at 3/3.
    assert hit_rate_needed(3.4, 0.53) is None
    # A hundred can: P(X>=57 | 100, 0.5) ~= 0.097, P(X>=56) ~= 0.136.
    assert hit_rate_needed(100, 0.5) == pytest.approx(0.57)


# ----------------------------------------------------------------- report ----


def test_report_adds_the_random_walk_floor_and_withholds_thin_verdicts(tmp_path):
    settings = CoachSettings(runs_dir=tmp_path / "runs", calibration_dir=tmp_path / "cal")
    CalibrationLedger(settings).save(V001)
    record = make_run_record(cutoff=CUTOFF, horizons=(5,), centre=80.0)
    RunRecordStore(settings).save(record)

    history = _prices(CUTOFF, 600)
    target = record.horizon(5).forecast_date.date()
    values = {day.isoformat(): float(value) for day, value in history.items()}
    values[(target - timedelta(days=0)).isoformat()] = 86.0

    reporter = CoachReporter(settings, resolver=OutcomeResolver(settings, service=FakeService(values)))
    report = reporter.build()
    variants = {card.variant: card for card in report.cards}

    assert {RANDOM_WALK, ENSEMBLE, AGENT} <= set(variants)
    assert variants[RANDOM_WALK].p50 == pytest.approx(80.0)
    assert variants[RANDOM_WALK].direction_call == 0
    # Every variant is scored against the same realized price.
    assert len({card.realized_value for card in report.cards}) == 1

    rendered = reporter.render(report)
    assert "DIRECTION" in rendered
    assert "never calls direction" in rendered
    assert "verdict withheld" in rendered
