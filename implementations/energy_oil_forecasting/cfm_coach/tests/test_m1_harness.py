"""Tests for the M1 calibration harness: replay, scoring, outcomes, and the gate.

The two loudest assertions here are the ones that protect against silent wrongness:
``verify_fidelity`` proves the coach still models v5.0 exactly, and the synthetic
null proves the gate cannot be talked into calling noise an improvement.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pytest

# `tests/` is deliberately not a package here (matching `cfm_agent_v_5_0/tests`),
# so pytest puts this directory on sys.path and `conftest` imports as a top-level module.
from conftest import FakeService, make_run_record, quantile_grid
from energy_oil_forecasting.cfm_coach.config import CoachSettings
from energy_oil_forecasting.cfm_coach.ledger import CalibrationLedger
from energy_oil_forecasting.cfm_coach.outcomes import (
    OutcomeResolver,
    PendingForecast,
    ResolutionReport,
)
from energy_oil_forecasting.cfm_coach.policy.comparison_policy import ComparisonPolicy
from energy_oil_forecasting.cfm_coach.replay import FidelityError, ReplayEngine
from energy_oil_forecasting.cfm_coach.report import AGENT, CURRENT, ENSEMBLE, CoachReporter, effective_n
from energy_oil_forecasting.cfm_coach.run_store import RunRecordStore
from energy_oil_forecasting.cfm_coach.schemas import (
    CalibrationLayer,
    CalibrationVersion,
    Candidate,
    ResolvedForecast,
)
from energy_oil_forecasting.cfm_coach.scoring import (
    ForecastScorer,
    crps_from_quantiles,
    interval_covers,
    interval_width,
    pinball_by_level,
    pinball_loss,
)


V001 = CalibrationVersion(version="v001", effective_from=date(2004, 1, 1))


def _resolved(record, horizon: int, realized: float) -> ResolvedForecast:
    return ResolvedForecast(
        run_id=record.run_id,
        horizon=horizon,
        cutoff=record.cutoff,
        forecast_date=record.horizon(horizon).forecast_date,
        realized_value=realized,
        realized_observation_date=record.horizon(horizon).forecast_date.date(),
        resolved_at=datetime(2026, 8, 14, 12, 0),
    )


# ---------------------------------------------------------------- replay ----


def test_every_stored_record_replays_exactly():
    """The corpus must reproduce bit-for-bit under its own calibration.

    This is the check that keeps the coach's model of v5.0 from drifting. If it ever
    fails, every calibration fitted on this corpus is invalid -- so it runs against
    the real committed records, not synthetic ones.
    """
    records = RunRecordStore().load_all()
    assert records, "the committed corpus should not be empty"
    for report in ReplayEngine().verify_all(records, strict=True):
        assert report.faithful
        assert report.max_absolute_error == 0.0


def test_identity_layer_is_a_no_op():
    """lambda=1, omega=1 reproduces v5.0 exactly -- so v001 changes nothing."""
    record = make_run_record(cutoff="2026-05-01", centre=80.0, spread=3.0)
    replayed = ReplayEngine().replay(record, V001)
    assert V001.is_baseline
    for item in replayed:
        recorded = record.horizon(item.horizon)
        assert item.point_forecast == recorded.final_point_forecast
        assert item.quantiles == recorded.final_quantiles


def test_width_scale_widens_without_moving_the_centre():
    """The leading M1 hypothesis: width should scale with volatility, the centre should not."""
    record = make_run_record(cutoff="2026-05-01", centre=80.0, spread=3.0)
    wide = CalibrationVersion(
        version="vtest",
        effective_from=date(2004, 1, 1),
        layer=CalibrationLayer(width_scale={5: 1.5, 10: 1.5, 21: 1.5}),
    )
    for base, scaled in zip(ReplayEngine().replay(record, V001), ReplayEngine().replay(record, wide), strict=True):
        assert scaled.point_forecast == pytest.approx(base.point_forecast)
        assert scaled.p10_p90_width == pytest.approx(base.p10_p90_width * 1.5)


def test_centre_gain_of_zero_discards_the_overlay():
    """centre_gain=0 must return the raw ensemble centre, whatever the LLM proposed."""
    record = make_run_record(cutoff="2026-05-01", centre=80.0, spread=3.0)
    flat = CalibrationVersion(
        version="vflat",
        effective_from=date(2004, 1, 1),
        layer=CalibrationLayer(centre_gain={5: 0.0, 10: 0.0, 21: 0.0}),
    )
    for item in ReplayEngine().replay(record, flat):
        assert item.point_forecast == pytest.approx(item.ensemble_quantiles[0.5])
        assert item.overlay == pytest.approx(0.0)


def test_fidelity_fails_loudly_when_a_record_is_tampered_with(tmp_path):
    """A record whose stored output no longer matches its inputs must raise, not warn."""
    record = make_run_record(cutoff="2026-05-01")
    horizon = record.forecasts[0]
    tampered = record.model_copy(
        update={
            "forecasts": [
                horizon.model_copy(update={"final_point_forecast": horizon.final_point_forecast + 1.0}),
                *record.forecasts[1:],
            ]
        }
    )
    settings = CoachSettings(calibration_dir=tmp_path)
    CalibrationLedger(settings).save(V001)
    engine = ReplayEngine(settings)

    assert not engine.check_fidelity(tampered).faithful
    with pytest.raises(FidelityError, match="DRIFTED"):
        engine.verify_fidelity(tampered)


def test_replay_uses_the_records_own_settings_not_package_defaults(tmp_path):
    """A candidate must vary only the calibrated constants, never the run's other choices."""
    record = make_run_record(cutoff="2026-05-01")
    settings = CoachSettings(calibration_dir=tmp_path)
    CalibrationLedger(settings).save(V001)
    resolved = ReplayEngine(settings).agent_settings_for(record, V001)
    assert resolved.model_dump(mode="json") == record.settings


# --------------------------------------------------------------- scoring ----


def test_pinball_matches_hand_computation():
    quantiles = {0.1: 70.0, 0.5: 80.0, 0.9: 90.0}
    losses = pinball_by_level(quantiles, 85.0)
    assert losses[0.1] == pytest.approx(0.1 * (85.0 - 70.0))
    assert losses[0.5] == pytest.approx(0.5 * (85.0 - 80.0))
    assert losses[0.9] == pytest.approx((0.9 - 1.0) * (85.0 - 90.0))
    assert pinball_loss(quantiles, 85.0) == pytest.approx(np.mean([1.5, 2.5, 0.5]))


@pytest.mark.parametrize("level", [0.1, 0.5, 0.9])
def test_pinball_is_minimised_at_the_true_quantile(level):
    """The property that makes pinball a proper scoring rule, checked numerically."""
    sample = np.random.default_rng(0).normal(80.0, 5.0, 200_000)
    truth = float(np.quantile(sample, level))
    candidates = np.linspace(truth - 2.0, truth + 2.0, 201)
    losses = [np.mean((level - (sample < candidate)) * (sample - candidate)) for candidate in candidates]
    assert candidates[int(np.argmin(losses))] == pytest.approx(truth, abs=0.05)


def test_widening_helps_when_the_outcome_lands_outside_the_interval():
    """R1's premise: 0-of-6 coverage means the intervals are too narrow, and pinball should say so."""
    narrow = quantile_grid(80.0, 1.0)
    wide = quantile_grid(80.0, 2.0)
    actual = 86.0
    assert not interval_covers(narrow, actual)
    assert pinball_loss(wide, actual) < pinball_loss(narrow, actual)


def test_widening_hurts_when_the_outcome_is_already_covered():
    """The same lever must cost something when it is not needed, or it is not a trade-off."""
    narrow = quantile_grid(80.0, 1.0)
    wide = quantile_grid(80.0, 2.0)
    actual = 80.2
    assert interval_covers(narrow, actual)
    assert pinball_loss(wide, actual) > pinball_loss(narrow, actual)


def test_coverage_and_width_read_the_80_percent_interval():
    quantiles = quantile_grid(80.0, 2.0)
    assert interval_width(quantiles) == pytest.approx(quantiles[0.9] - quantiles[0.1])
    assert interval_covers(quantiles, quantiles[0.1])
    assert interval_covers(quantiles, quantiles[0.9])
    assert not interval_covers(quantiles, quantiles[0.9] + 0.01)


def test_crps_is_zero_for_a_point_mass_on_the_outcome():
    degenerate = dict.fromkeys((0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95), 80.0)
    assert crps_from_quantiles(degenerate, 80.0) == pytest.approx(0.0)
    assert crps_from_quantiles(degenerate, 85.0) > 0.0


def test_summarize_refuses_to_mix_variants():
    record = make_run_record(cutoff="2026-05-01")
    resolved = _resolved(record, 5, 84.0)
    scorer = ForecastScorer()
    cards = [
        scorer.score_recorded(record, resolved, variant=AGENT),
        scorer.score_ensemble(record, resolved, variant=ENSEMBLE),
    ]
    with pytest.raises(ValueError, match="mix variants"):
        ForecastScorer.summarize(cards)


# -------------------------------------------------------------- outcomes ----


def test_resolves_a_trading_day_and_pends_the_future():
    record = make_run_record(cutoff="2026-05-01", horizons=(5, 21))
    target = record.horizon(5).forecast_date.date()
    service = FakeService({target.isoformat(): 84.5})
    report = OutcomeResolver(service=service).resolve(record)

    assert len(report.resolved) == 1
    assert report.resolved[0].realized_value == pytest.approx(84.5)
    assert report.resolved[0].realized_observation_date == target
    assert len(report.pending) == 1
    assert "target date not reached" in report.pending[0].reason


def test_falls_back_to_the_last_observation_before_a_non_trading_target():
    """A business-day offset can land on a holiday; that is normal and must resolve."""
    record = make_run_record(cutoff="2026-05-01", horizons=(5,))
    target = record.horizon(5).forecast_date.date()
    service = FakeService(
        {
            (target - timedelta(days=1)).isoformat(): 83.0,
            (target + timedelta(days=3)).isoformat(): 90.0,
        }
    )
    resolved = OutcomeResolver(service=service).resolve(record).resolved
    assert len(resolved) == 1
    assert resolved[0].realized_value == pytest.approx(83.0)
    assert resolved[0].realized_observation_date == target - timedelta(days=1)


def test_refuses_to_score_against_a_stale_price():
    """A frozen cache must not silently resolve a forecast against a pre-issue price."""
    record = make_run_record(cutoff="2026-05-01", horizons=(5,))
    target = record.horizon(5).forecast_date.date()
    service = FakeService(
        {
            (target - timedelta(days=30)).isoformat(): 70.0,
            (target + timedelta(days=10)).isoformat(): 95.0,
        }
    )
    outcome = OutcomeResolver(service=service).resolve_horizon(record, 5)
    assert isinstance(outcome, PendingForecast)
    assert "stale price" in outcome.reason


# ---------------------------------------------------------------- report ----


def test_effective_n_collapses_repeat_runs_of_one_cutoff():
    """Two runs of the same cutoff are not two independent observations of the market."""
    record = make_run_record(cutoff="2026-05-01", horizons=(5,))
    resolved = _resolved(record, 5, 84.0)
    scorer = ForecastScorer()
    once = [scorer.score_recorded(record, resolved, variant=AGENT)]
    twice = [*once, scorer.score_ensemble(record, resolved, variant=ENSEMBLE)]
    assert effective_n(once) == pytest.approx(effective_n(twice))
    assert effective_n(once) == pytest.approx(1 / 5)


def test_report_scores_three_variants_and_current_matches_agent_under_v001(tmp_path):
    settings = CoachSettings(runs_dir=tmp_path / "runs", calibration_dir=tmp_path / "cal")
    CalibrationLedger(settings).save(V001)
    record = make_run_record(cutoff="2026-05-01", horizons=(5,))
    RunRecordStore(settings).save(record)
    target = record.horizon(5).forecast_date.date()

    reporter = CoachReporter(
        settings,
        resolver=OutcomeResolver(settings, service=FakeService({target.isoformat(): 86.0})),
    )
    report = reporter.build()
    variants = {card.variant: card for card in report.cards}
    assert set(variants) == {ENSEMBLE, AGENT, CURRENT}
    # v001 is a true no-op, so the calibration in force must reproduce what shipped.
    assert variants[CURRENT].pinball == pytest.approx(variants[AGENT].pinball)
    assert report.fitting_eligible_records == 1
    assert "cfm coach scoreboard" in reporter.render(report).lower()


# ------------------------------------------------------------ the gate ------


def _corpus(
    n_origins: int = 20,
    *,
    provenance: str = "live_forward",
    spread: float = 1.0,
    offset: float = 1.5,
):
    """Build a synthetic live corpus whose intervals are too narrow, missing both ways.

    Outcomes sit outside the P10-P90 on both sides in equal measure, so the centre is
    unbiased and *only* the width is wrong -- which is the calibration question M1 is
    actually built to answer. The default ``offset`` of 1.5x the incumbent half-width
    is deliberately just outside it: a 2x candidate then covers every outcome, so the
    positive control exercises coverage as well as pinball.
    """
    start = date(2026, 1, 5)
    records, resolutions = [], []
    for index in range(n_origins):
        cutoff = start + timedelta(days=7 * index)
        record = make_run_record(cutoff=cutoff.isoformat(), centre=80.0, spread=spread, provenance=provenance)
        records.append(record)
        signed = offset if index % 2 == 0 else -offset
        resolutions.extend(_resolved(record, horizon, 80.0 + signed) for horizon in record.horizons)
    return records, resolutions


def _resolution_report(resolved):
    return ResolutionReport(resolved=resolved, pending=[], data_through=date(2026, 12, 31))


@pytest.fixture
def gate(tmp_path) -> ComparisonPolicy:
    settings = CoachSettings(calibration_dir=tmp_path / "cal")
    CalibrationLedger(settings).save(V001)
    return ComparisonPolicy(settings)


def test_synthetic_null_returns_p_of_one(gate):
    """Candidate identical to incumbent: every paired delta is exactly zero, so p = 1."""
    deltas = {date(2026, 1, 5) + timedelta(days=7 * index): [0.0, 0.0, 0.0] for index in range(20)}
    p_value, ci_low, ci_high = gate.origin_blocked_bootstrap(deltas)
    assert p_value == pytest.approx(1.0)
    assert ci_low == pytest.approx(0.0)
    assert ci_high == pytest.approx(0.0)


def test_bootstrap_blocks_on_origin_not_on_origin_times_horizon(gate):
    """Resampling rows instead of origins would manufacture significance from correlation.

    Horizons within an origin are near-duplicates, so a row bootstrap sees 60
    independent draws where there are really 20. It should post a visibly smaller
    p-value on the identical data -- which is exactly the overclaim origin-blocking
    exists to prevent.
    """
    rng = np.random.default_rng(7)
    origins = [date(2026, 1, 5) + timedelta(days=7 * index) for index in range(20)]
    # A weak per-origin effect, repeated identically across that origin's 3 horizons.
    deltas = {origin: [float(value)] * 3 for origin, value in zip(origins, rng.normal(0.35, 1.0, 20), strict=True)}

    blocked_p, _, _ = gate.origin_blocked_bootstrap(deltas)

    rows = np.array([value for values in deltas.values() for value in values])
    row_rng = np.random.default_rng(gate.seed)
    naive = row_rng.choice(rows, size=(gate.s.bootstrap_resamples, rows.size), replace=True).mean(axis=1)
    naive_p = float((1 + np.sum(naive <= 0.0)) / (gate.s.bootstrap_resamples + 1))

    assert naive_p < blocked_p, "row-resampling should look more significant than it is"


def test_gate_rejects_replayed_history_as_fitting_evidence(gate):
    records, resolved = _corpus(provenance="replayed_live_search")
    candidate = Candidate(
        candidate_id="width", parent_version="v001", layer=CalibrationLayer(width_scale={5: 2.0, 10: 2.0, 21: 2.0})
    )
    verdict = gate.evaluate(candidate, V001, records=records, resolution=_resolution_report(resolved))

    assert not verdict.passed
    failed = {item.name for item in verdict.failed_conditions}
    assert "provenance_clean" in failed


def test_gate_rejects_a_candidate_fitted_on_everything(gate):
    """No holdout, no verdict: a candidate fitted through the last origin has nothing left to prove itself on."""
    records, resolved = _corpus()
    candidate = Candidate(
        candidate_id="width",
        parent_version="v001",
        layer=CalibrationLayer(width_scale={5: 2.0, 10: 2.0, 21: 2.0}),
        fitted_through=max(item.cutoff for item in resolved),
    )
    verdict = gate.evaluate(candidate, V001, records=records, resolution=_resolution_report(resolved))

    assert not verdict.passed
    assert "holdout_respected" in {item.name for item in verdict.failed_conditions}


def test_gate_rejects_two_tunables_moved_at_once(gate):
    records, resolved = _corpus()
    candidate = Candidate(
        candidate_id="both",
        parent_version="v001",
        layer=CalibrationLayer(width_scale={5: 2.0, 10: 2.0, 21: 2.0}, centre_gain={5: 0.5, 10: 0.5, 21: 0.5}),
    )
    verdict = gate.evaluate(candidate, V001, records=records, resolution=_resolution_report(resolved))

    assert not verdict.passed
    assert "single_tunable" in {item.name for item in verdict.failed_conditions}


def test_gate_rejects_a_corpus_spanning_two_prompt_versions(gate):
    """A different prompt is a different agent; pooling across them biases every fit."""
    records, resolved = _corpus()
    records[0] = make_run_record(cutoff=records[0].cutoff.isoformat(), spread=1.0, prompt_version="cfm_v5_0_m2_memory")
    candidate = Candidate(
        candidate_id="width", parent_version="v001", layer=CalibrationLayer(width_scale={5: 2.0, 10: 2.0, 21: 2.0})
    )
    verdict = gate.evaluate(candidate, V001, records=records, resolution=_resolution_report(resolved))

    assert not verdict.passed
    assert "single_prompt_version" in {item.name for item in verdict.failed_conditions}


def test_gate_rejects_an_improvement_too_small_to_matter(gate):
    """Detectable is not the same as worth applying to a live forecast."""
    records, resolved = _corpus()
    candidate = Candidate(
        candidate_id="tiny",
        parent_version="v001",
        layer=CalibrationLayer(width_scale={5: 1.01, 10: 1.01, 21: 1.01}),
        fitted_through=date(2026, 2, 20),
    )
    verdict = gate.evaluate(candidate, V001, records=records, resolution=_resolution_report(resolved))

    assert not verdict.passed
    assert "effect_size_sufficient" in {item.name for item in verdict.failed_conditions}
    assert 0.0 < verdict.effect_size_pct < gate.s.min_effect_size_pct


def test_gate_passes_a_genuine_improvement_and_shrinks_it(gate):
    """The positive control: a real, large, out-of-sample width fix clears all nine conditions."""
    records, resolved = _corpus()
    candidate = Candidate(
        candidate_id="width_2x",
        parent_version="v001",
        layer=CalibrationLayer(width_scale={5: 2.0, 10: 2.0, 21: 2.0}),
        fitted_through=date(2026, 2, 20),
        rationale="0-of-N coverage; intervals are far too narrow",
    )
    verdict = gate.evaluate(candidate, V001, records=records, resolution=_resolution_report(resolved))

    assert verdict.passed, verdict.describe()
    assert verdict.mean_paired_delta > 0
    assert verdict.p_value < gate.s.significance_alpha
    assert verdict.effect_size_pct >= gate.s.min_effect_size_pct
    assert verdict.n_holdout_origins >= gate.s.min_holdout_origins
    # R1's target moves too: the outcomes sit just outside the incumbent interval.
    assert verdict.coverage_incumbent == pytest.approx(0.0)
    assert verdict.coverage_candidate == pytest.approx(1.0)

    # Shrinkage: half way from the incumbent's 1.0, never the raw fitted 2.0.
    assert verdict.shrunk_layer.width_scale == {5: 1.5, 10: 1.5, 21: 1.5}
    assert verdict.shrinkage_factor == pytest.approx(0.5)


def test_a_worse_candidate_never_passes(gate):
    """Narrowing already-too-narrow intervals must be rejected on sign alone."""
    records, resolved = _corpus()
    candidate = Candidate(
        candidate_id="narrower",
        parent_version="v001",
        layer=CalibrationLayer(width_scale={5: 0.5, 10: 0.5, 21: 0.5}),
        fitted_through=date(2026, 2, 20),
    )
    verdict = gate.evaluate(candidate, V001, records=records, resolution=_resolution_report(resolved))

    assert not verdict.passed
    assert verdict.mean_paired_delta < 0
    assert verdict.p_value > gate.s.significance_alpha


def test_verdict_scores_both_variants_on_identical_keys(gate):
    records, resolved = _corpus()
    candidate = Candidate(
        candidate_id="width_2x",
        parent_version="v001",
        layer=CalibrationLayer(width_scale={5: 2.0, 10: 2.0, 21: 2.0}),
        fitted_through=date(2026, 2, 20),
    )
    verdict = gate.evaluate(candidate, V001, records=records, resolution=_resolution_report(resolved))
    pairing = next(item for item in verdict.conditions if item.name == "paired_on_identical_days")
    assert pairing.passed
    assert verdict.n_scored_rows == verdict.n_holdout_origins * 3


def test_gate_rejects_an_unsupported_metric(gate):
    records, resolved = _corpus()
    candidate = Candidate(candidate_id="x", parent_version="v001")
    with pytest.raises(ValueError, match="metric must be one of"):
        gate.evaluate(candidate, V001, records=records, resolution=_resolution_report(resolved), metric="rmse")
