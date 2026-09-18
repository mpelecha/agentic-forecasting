"""Stats: strata reconcile with the scoreboard, episodes cut the path honestly, counterfactuals price exactly."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest
from conftest import make_v52_run_record

from energy_oil_forecasting.cfm_coach.report import AGENT, ENSEMBLE
from energy_oil_forecasting.cfm_coach.review.stats import (
    CounterfactualEngine,
    episode_for,
    find_episodes,
    independent_windows,
    recombine,
    reweight_record,
    strata_frame,
    strata_table,
)
from energy_oil_forecasting.cfm_coach.schemas import CalibrationLayer
from test_review_cards import corpus_with, ten_runs


def _corpus():
    runs = ten_runs("2026-08-19") + [make_v52_run_record(cutoff="2026-08-26", suffix="z", price_shift=1.5)]
    return corpus_with(runs, through="2026-09-30")


def test_strata_totals_reconcile_with_the_scoreboard():
    corpus = _corpus()
    frame = strata_frame(corpus)
    agent_scores = [c for (_, _, v), c in corpus.scores.items() if v == AGENT]
    assert len(frame) == len(agent_scores)
    assert frame["pinball_agent"].mean() == pytest.approx(np.mean([c.pinball for c in agent_scores]))
    assert (frame["gap_vs_rw"] - frame["base_term"] - frame["overlay_term"]).abs().max() < 1e-9
    table = strata_table(frame, ["horizon", "tier"])
    assert table["rows"].sum() == len(frame)
    h5 = table[table["horizon"] == 5]
    assert h5["distinct_cutoffs"].max() == 2
    assert (h5["effective_n"] == h5["distinct_cutoffs"] / 5).all()
    assert set(frame["zeroed"]) == {True}  # the fixture's proposed move was zeroed on every run


def test_independent_windows_counts_non_overlapping_cutoffs():
    days = [date(2026, 8, d) for d in (19, 20, 21, 24, 25, 26)]
    assert independent_windows(days, 5) == 2
    assert independent_windows(days, 1) == 6
    assert independent_windows(days, 21) == 1
    assert independent_windows([], 5) == 0


def test_find_episodes_sees_one_rally_and_one_dip_not_many():
    days = pd.bdate_range("2024-01-02", periods=560)
    values = np.full(len(days), 80.0)
    values[520:530] += np.linspace(1, 15, 10)  # a two-week rally
    values[530:540] = values[529] - np.linspace(1, 15, 10)  # then a two-week dip
    values[540:] = values[539]
    prices = pd.Series(values, index=[d.date() for d in days])
    episodes = find_episodes(prices, start=date(2025, 12, 1))
    assert [e.direction for e in episodes] == [1, -1]
    assert episodes[0].end_price - episodes[0].start_price == pytest.approx(15.0)
    hit = episode_for(episodes, cutoff=episodes[0].start, forecast_date=episodes[0].end)
    assert hit is episodes[0]
    assert episode_for(episodes, cutoff=date(2024, 3, 1), forecast_date=date(2024, 3, 20)) is None


def test_recombiner_reproduces_the_recorded_ensemble():
    record = make_v52_run_record()
    weights = record.settings["ensemble_weights"]
    for item in record.forecasts:
        pooled = recombine({k: dict(v) for k, v in item.component_quantiles.items()}, weights)
        for level, value in item.ensemble_quantiles.items():
            assert pooled[level] == pytest.approx(value, abs=1e-9)
    heavy = reweight_record(record, {"arima": 1.0, "kalman": 0.0, "lightgbm": 0.0})
    assert heavy.horizon(5).ensemble_quantiles[0.5] == pytest.approx(
        record.horizon(5).component_quantiles["arima"][0.5]
    )
    with pytest.raises(ValueError, match="RandomWalkBaseline"):
        reweight_record(record, {"random_walk": 1.0})


def test_counterfactual_operations_are_priced_exactly():
    corpus = _corpus()
    engine = CounterfactualEngine(corpus)

    identity = engine.calibration(layer=CalibrationLayer())
    assert identity.pooled is not None and max(abs(d) for d in identity.deltas.values()) < 1e-9
    assert identity.fidelity == "exact"

    zero = engine.overlay_zero()
    ens = np.mean([c.pinball for (_, _, v), c in corpus.scores.items() if v == ENSEMBLE])
    assert zero.pooled.pinball_after == pytest.approx(ens)

    anchored = engine.calibration(layer=CalibrationLayer(rw_anchor={5: 1.0, 10: 1.0, 21: 1.0}))
    shifted = engine.rw_centre_keep_width(1.0)
    # Anchoring through the layer and translating the published quantiles are the same operation.
    for key, delta in anchored.deltas.items():
        assert shifted.deltas[key] == pytest.approx(delta, abs=1e-9)

    reweighted = engine.ensemble_reweight({"arima": 0.5, "kalman": 0.5, "lightgbm": 0.0})
    assert reweighted.fidelity == "approximate_ensemble_recombination"
    with_rw = engine.ensemble_reweight({"arima": 0.5, "random_walk": 0.5}, with_rw=True)
    assert with_rw.pooled is not None and with_rw.pooled.n == identity.pooled.n

    neutral = engine.action_override(
        {"horizon_actions_patch": {"center_action": "no_change", "uncertainty_action": "unchanged"}}
    )
    assert neutral.pooled.pinball_after == pytest.approx(ens)
    for row in identity.by_horizon:
        assert row.pinball_rw > 0 and row.hit_rate_needed is None or isinstance(row.hit_rate_needed, float)


def test_calibration_pricing_refuses_replay_blind_and_denied_fields(tmp_path):
    import pytest  # noqa: PLC0415

    from energy_oil_forecasting.cfm_coach.review.stats import REPLAY_BLIND_FIELDS  # noqa: PLC0415

    engine = CounterfactualEngine(_corpus())
    assert "ensemble_weights" in REPLAY_BLIND_FIELDS
    with pytest.raises(ValueError, match="ensemble_reweight"):
        engine.calibration(settings_overlay={"ensemble_weights": {}})
    with pytest.raises(ValueError, match="denylist"):
        engine.calibration(settings_overlay={"audit_enabled": True})
    assert engine.calibration(settings_overlay={"large_action_width_fraction": 0.4}).by_horizon
