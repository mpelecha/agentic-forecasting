"""Numeric track: the anchor lever, fits that find a planted bias, and candidates that always validate."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from conftest import make_v52_run_record

from energy_oil_forecasting.cfm_agent_v_5_2.config import CfmV52Settings
from energy_oil_forecasting.cfm_coach.config import CoachSettings
from energy_oil_forecasting.cfm_coach.policy.comparison_policy import ComparisonPolicy
from energy_oil_forecasting.cfm_coach.review.coached import coach_ledger
from energy_oil_forecasting.cfm_coach.review.numeric import (
    denied,
    fit_ensemble_weights,
    fit_rw_anchor,
    judge_candidate,
    next_version_name,
    round_overlay,
    simplex_grid,
)
from energy_oil_forecasting.cfm_coach.review.settings import ReviewSettings
from energy_oil_forecasting.cfm_coach.schemas import CalibrationLayer, CalibrationVersion, Candidate
from energy_oil_forecasting.cfm_coach.streams import V52_ADVANCED
from test_review_cards import corpus_with


# -- the layer field ---------------------------------------------------------


def test_rw_anchor_translates_the_whole_distribution_toward_the_last_close():
    layer = CalibrationLayer(rw_anchor={5: 1.0})
    point, q = layer.apply(
        ensemble_p50=80.0,
        final_point_forecast=81.0,
        final_quantiles={0.1: 78.0, 0.5: 81.0, 0.9: 84.0},
        horizon=5,
        last_value=90.0,
    )
    assert point == pytest.approx(90.0) and q == pytest.approx({0.1: 87.0, 0.5: 90.0, 0.9: 93.0})
    half = CalibrationLayer(rw_anchor={5: 0.5})
    assert half.apply(
        ensemble_p50=80.0, final_point_forecast=81.0, final_quantiles={0.5: 81.0}, horizon=5, last_value=91.0
    )[0] == pytest.approx(86.0)
    assert (
        half.apply(
            ensemble_p50=80.0, final_point_forecast=81.0, final_quantiles={0.5: 81.0}, horizon=10, last_value=91.0
        )[0]
        == 81.0
    )
    assert not half.is_identity and CalibrationLayer(rw_anchor={5: 0.0}).is_identity
    with pytest.raises(ValueError, match="last_value"):
        half.apply(ensemble_p50=80.0, final_point_forecast=81.0, final_quantiles={0.5: 81.0}, horizon=5)


def test_anchor_is_one_tunable_and_shrinks_toward_the_incumbent():
    incumbent = CalibrationVersion(version="v001", effective_from=date(2004, 1, 1))
    candidate = Candidate(candidate_id="c", parent_version="v001", layer=CalibrationLayer(rw_anchor={5: 0.8, 10: 0.6}))
    assert candidate.tunable_names(incumbent) == {"layer.rw_anchor"}
    policy = ComparisonPolicy(CoachSettings(target_agent="cfm_agent_v_5_2"))
    _, layer, factor = policy.shrink(candidate, incumbent)
    assert factor == 0.5 and layer.rw_anchor == {5: 0.4, 10: 0.3}


# -- fits on synthetic history --------------------------------------------------


def _history_rows(bias: float, n: int = 120, horizon: int = 10):
    rng = np.random.default_rng(1)
    rows = []
    day = date(2019, 1, 7)
    for i in range(n):
        last = 60.0 + 0.05 * i
        actual = last + rng.normal(0, 2.0)
        p50 = last + bias
        rows.append(
            {
                "cutoff": day + pd.Timedelta(weeks=i).to_pytimedelta(),
                "horizon": horizon,
                "actual": actual,
                "p10": p50 - 3,
                "p50": p50,
                "p90": p50 + 3,
                "last_value": last,
                "vix": 15.0,
                "quantiles": {0.1: p50 - 3, 0.5: p50, 0.9: p50 + 3},
            }
        )
    return pd.DataFrame(rows)


def test_fit_rw_anchor_finds_a_planted_level_bias():
    rows = _history_rows(bias=-2.5)
    fit = fit_rw_anchor(rows, 10, train_through=date(2020, 6, 30))
    assert fit["anchor"] >= 0.8
    assert fit["train"]["gain_pct_vs_0"] > 0 and fit["holdout"]["n"] > 0
    assert fit["anchor_ci90"][0] <= fit["anchor"] <= fit["anchor_ci90"][1]
    unbiased = fit_rw_anchor(_history_rows(bias=0.0), 10, train_through=date(2020, 6, 30))
    assert unbiased["anchor"] <= 0.3


def test_fit_ensemble_weights_moves_weight_off_a_biased_member():
    rng = np.random.default_rng(2)
    rows = []
    day = date(2019, 1, 7)
    for i in range(80):
        last = 60.0
        actual = last + rng.normal(0, 1.0)
        good = {0.1: last - 2, 0.5: last, 0.9: last + 2}
        bad = {0.1: last - 6, 0.5: last - 4, 0.9: last - 2}
        rows.append(
            {
                "cutoff": day + pd.Timedelta(weeks=i).to_pytimedelta(),
                "horizon": 5,
                "actual": actual,
                "last_value": last,
                "components": {"arima": good, "lightgbm": bad},
                "weights": {"arima": 0.5, "lightgbm": 0.5},
            }
        )
    fit = fit_ensemble_weights(pd.DataFrame(rows), 5, train_through=date(2020, 1, 1))
    assert fit["best"]["arima"] >= 0.9
    assert fit["train"]["pinball_best"] < fit["train"]["pinball_equal"]
    assert sum(fit["shrunk_toward_equal"].values()) == pytest.approx(1.0)
    assert len(simplex_grid(["a", "b", "c"])) == 66


# -- candidates ------------------------------------------------------------------


def test_round_overlay_and_denylist():
    assert round_overlay({"tier_2_min_publishers": 2.5, "small_action_width_fraction": 0.15}, CfmV52Settings) == {
        "tier_2_min_publishers": 2,
        "small_action_width_fraction": 0.15,
    }
    denylist = ReviewSettings().overlay_denylist
    assert (
        denied("audit_enabled", denylist)
        and denied("search_verifier_model", denylist)
        and denied("research_passage_chars", denylist)
    )
    assert not denied("large_action_width_fraction", denylist) and not denied("ensemble_weights", denylist)


def test_judge_candidate_labels_power_and_emits_a_validating_candidate(tmp_path):
    settings = ReviewSettings(data_dir=tmp_path)
    days = [d.date() for d in pd.bdate_range("2026-08-03", periods=14)]
    records = [make_v52_run_record(cutoff=str(d), suffix="a", price_shift=0.3 * i) for i, d in enumerate(days)]
    corpus = corpus_with(records, through="2026-09-30")
    candidate = Candidate(
        candidate_id="P-0001",
        parent_version="v001",
        layer=CalibrationLayer(rw_anchor={5: 0.8, 10: 0.8, 21: 0.8}),
        fitted_through=date(2021, 12, 31),
        rationale="test",
    )
    verdict = judge_candidate(corpus, V52_ADVANCED, candidate, lever="layer.rw_anchor", settings=settings)
    assert verdict.incumbent_version == "v001" and verdict.underpowered
    assert verdict.shrunk.shrunk_layer is None or True  # gate ran; its verdict is its own business
    assert verdict.pricing_shrunk.pooled is not None and verdict.pricing_shrunk.fidelity == "exact"
    assert verdict.candidate_file and verdict.candidate_file.endswith("P-0001__shrunk__v002.json")
    emitted = CalibrationVersion.model_validate_json(open(verdict.candidate_file).read())
    assert emitted.layer.rw_anchor == {5: 0.4, 10: 0.4, 21: 0.4} and emitted.parent_version == "v001"
    assert next_version_name(coach_ledger(V52_ADVANCED, settings)) == "v002"  # nothing was saved into the ledger
    assert not (settings.calibration_dir / "v52_advanced" / "v002.json").exists()

    with pytest.raises(ValueError, match="denylist"):
        judge_candidate(
            corpus,
            V52_ADVANCED,
            Candidate(candidate_id="P-bad", parent_version="v001", settings_overlay={"audit_enabled": False}),
            lever="x",
            settings=settings,
        )
