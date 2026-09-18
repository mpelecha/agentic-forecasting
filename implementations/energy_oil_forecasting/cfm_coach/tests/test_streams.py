"""The invariants that keep two agents and three models from contaminating each other.

Every test here guards something that would fail *silently* if it broke: a run
charged to the wrong model, a corpus pooled across builds, a fingerprint asserting
an identity the record does not have. None of them would raise at run time -- they
would just produce a confident wrong calibration some weeks later.
"""

from __future__ import annotations

from datetime import date

import pytest
from aieng.forecasting.models import ADVANCED_MODEL, LITE_MODEL
from conftest import make_run_record
from energy_oil_forecasting.cfm_coach.ledger import CalibrationLedger
from energy_oil_forecasting.cfm_coach.outcomes import ResolutionReport
from energy_oil_forecasting.cfm_coach.policy.comparison_policy import ComparisonPolicy
from energy_oil_forecasting.cfm_coach.replay import FidelityError, ReplayEngine
from energy_oil_forecasting.cfm_coach.run_store import agent_package_fingerprint
from energy_oil_forecasting.cfm_coach.schemas import CalibrationVersion, Candidate
from energy_oil_forecasting.cfm_coach.streams import (
    SCHEDULED_STREAMS,
    STREAMS,
    V50_LITE,
    V52_ADVANCED,
    V52_LITE,
    stream_for,
)
from energy_oil_forecasting.cfm_coach.targets import V50, V52, target_for


# -- the streams are what was asked for ---------------------------------------


def test_the_three_streams_are_wired_as_specified():
    assert (V50_LITE.target, V50_LITE.model, V50_LITE.runs_per_day) == (V50, LITE_MODEL, 3)
    assert (V52_ADVANCED.target, V52_ADVANCED.model, V52_ADVANCED.runs_per_day) == (V52, ADVANCED_MODEL, 3)
    assert (V52_LITE.target, V52_LITE.model, V52_LITE.runs_per_day) == (V52, LITE_MODEL, 10)


def test_stream_lookup_rejects_an_unknown_id():
    assert stream_for("v52_lite") is V52_LITE
    with pytest.raises(KeyError):
        stream_for("v52_medium")


# -- every LLM call goes where the stream says --------------------------------


def test_v52_streams_bind_every_llm_call_to_one_model():
    """Five knobs, not one. Missing any leaves that call on the package default."""
    for stream in (V52_ADVANCED, V52_LITE):
        models = stream.models_in_use(stream.base_settings)
        assert set(models) == {
            "agent",
            "search",
            "search_verifier_model",
            "claim_support_verifier_model",
            "structured_output_retry_model",
        }
        assert set(models.values()) == {stream.model}, f"{stream.stream_id} is running a mixed-model workflow"


def test_v50_stream_keeps_the_verifier_models_it_shipped_with():
    """Binding them would change what the agent *is* partway through a live corpus.

    Nothing would catch it: `package_fingerprint` hashes the manifest, and the
    manifest does not record which model a verifier called.
    """
    models = V50_LITE.models_in_use(V50_LITE.base_settings)
    assert models["agent"] == models["search"] == LITE_MODEL
    assert models["search_verifier_model"] == ADVANCED_MODEL
    assert models["claim_support_verifier_model"] == ADVANCED_MODEL


def test_v52_streams_do_not_get_the_code_execution_tool():
    """`run_code` reaches E2B with no timeout at any layer, so it can block a run forever.

    Asserted on the *tool list*, not the settings flag: the flag is only meaningful
    because `build_cfm_agent_config` consults it, and that is the coupling that could
    silently break. Diagnostics-only and never once used in 11 lite runs, so removing
    it cannot change a forecast.
    """
    for stream in (V52_ADVANCED, V52_LITE):
        assert stream.base_settings.code_execution_enabled is False
        names = {t.func.__name__ for t in stream.build_config(settings=stream.base_settings).function_tools}
        assert "run_code" not in names, f"{stream.stream_id} still exposes run_code"


def test_v50_stream_keeps_the_code_execution_tool():
    """Unchanged for the same reason its verifier models are: 14 records already exist."""
    assert V50_LITE.base_settings.code_execution_enabled is True
    names = {t.func.__name__ for t in V50_LITE.build_config(settings=V50_LITE.base_settings).function_tools}
    assert "run_code" in names


def test_model_fields_are_discovered_not_assumed():
    """Guards the naming convention `AgentTarget.model_settings_fields` relies on.

    Rename a field off the ``_model`` suffix and that call silently stops being
    bound. This test is the reason that discovery is safe.
    """
    assert V50.model_settings_fields == ("claim_support_verifier_model", "search_verifier_model")
    assert V52.model_settings_fields == (
        "claim_support_verifier_model",
        "search_verifier_model",
        "structured_output_retry_model",
    )


# -- the corpora cannot collide -----------------------------------------------


def test_every_stream_has_its_own_corpus_and_ledger():
    runs = [stream.coach_settings.runs_dir for stream in STREAMS]
    ledgers = [stream.coach_settings.calibration_dir for stream in STREAMS]
    assert len(set(runs)) == len(STREAMS), "two streams would write into one corpus"
    assert len(set(ledgers)) == len(STREAMS), "a calibration fitted on one stream would apply to another"


def test_the_original_stream_still_points_at_the_original_directories():
    """The 14 existing records and the v001 ledger must stay where they are."""
    settings = V50_LITE.coach_settings
    assert settings.runs_dir.name == "runs"
    assert settings.calibration_dir.name == "calibration"
    assert (settings.calibration_dir / "v001.json").exists()


def test_each_stream_has_a_baseline_calibration_in_force():
    """A stream with an empty ledger raises at run time, on the morning it first fires."""
    for stream in STREAMS:
        current = CalibrationLedger(stream.coach_settings).current(date(2026, 8, 19))
        assert current.is_baseline, f"{stream.stream_id} baseline is not a no-op"


def test_v50_lite_is_retired_from_the_schedule_but_not_from_the_registry():
    """Retired 2026-09-08. Stopping the runs must not unpublish the corpus.

    The two halves are separable on purpose, and each half has a way to fail
    quietly: dropping `v50_lite` from `STREAMS` would break `stream_for` and
    the corpus page, while leaving it in `SCHEDULED_STREAMS` would keep
    spending on a stream nobody asked to keep running.
    """
    assert V50_LITE not in SCHEDULED_STREAMS
    assert V50_LITE in STREAMS
    assert stream_for("v50_lite") is V50_LITE
    assert set(SCHEDULED_STREAMS) == {V52_ADVANCED, V52_LITE}
    assert set(SCHEDULED_STREAMS) <= set(STREAMS), "the job cannot run a stream that does not exist"


def test_retiring_a_stream_is_not_expressible_as_a_closed_window():
    """Why `SCHEDULED_STREAMS` exists rather than an end date on `V50_LITE.window`.

    Outside its window a stream drops to `OFF_WINDOW_RUNS_PER_DAY`, which is one
    run a day, not none -- so closing the window would quietly keep running it.
    """
    long_past_the_window = date(2027, 1, 1)
    assert V50_LITE.repeats_for(long_past_the_window) == 1


def test_run_id_prefixes_are_distinct_across_streams():
    """Two v5.2 streams share an agent id; only the prefix keeps their run ids apart."""
    prefixes = [stream.run_id_prefix for stream in STREAMS]
    assert len(set(prefixes)) == len(STREAMS)
    # Unchanged for v5.0, so records already on disk stay addressable.
    assert V50_LITE.run_id_prefix == "cfm_agent_v_5_0"


# -- identity fields say what they mean ---------------------------------------


def test_package_fingerprints_distinguish_the_two_agents():
    """The bug this refactor exists to prevent: a v5.2 run stamped as v5.0."""
    v50 = agent_package_fingerprint(V50)
    v52 = agent_package_fingerprint(V52)
    assert v50.startswith("cfm_v5_0_package:sha256:")
    assert v52.startswith("cfm_v5_2_package:sha256:")
    assert v50 != v52
    assert "unavailable" not in v50 and "unavailable" not in v52


def test_target_lookup_is_by_agent_id():
    assert target_for("cfm_agent_v_5_0") is V50
    assert target_for("cfm_agent_v_5_2") is V52
    with pytest.raises(KeyError):
        target_for("cfm_agent_v_5_1")


# -- a calibration may not choose a model -------------------------------------


def test_a_calibration_overlay_may_not_name_an_llm():
    """Which model answered is identity, not a fitted constant."""
    rogue = CalibrationVersion(
        version="v999",
        effective_from=date(2026, 1, 1),
        settings_overlay={"search_verifier_model": ADVANCED_MODEL},
    )
    with pytest.raises(ValueError, match="LLM field"):
        CalibrationLedger.to_agent_settings(rogue, base=V52_LITE.base_settings, target=V52)


def test_a_normal_overlay_still_applies_and_leaves_the_models_alone():
    version = CalibrationVersion(
        version="v002",
        effective_from=date(2026, 1, 1),
        settings_overlay={"small_wider_multiplier": 1.15},
    )
    applied = CalibrationLedger.to_agent_settings(version, base=V52_LITE.base_settings, target=V52)
    assert applied.small_wider_multiplier == 1.15
    assert applied.structured_output_retry_model == LITE_MODEL
    # Run-level choices that are not calibration survive the overlay.
    assert applied.audit_enabled is True


# -- a record cannot be replayed through the wrong agent ----------------------


def test_replay_refuses_a_record_from_another_agent():
    """The one failure this refactor could introduce, and the cheapest to catch.

    v5.0's and v5.2's engines agree today, so a v5.2 record replayed through v5.0's
    engine would look perfectly faithful and start lying the moment they diverge.
    """
    record = make_run_record(cutoff="2026-08-19", agent_id="cfm_agent_v_5_2")
    engine = ReplayEngine(V50_LITE.coach_settings, target=V50)  # deliberately mismatched
    with pytest.raises(FidelityError, match="targets"):
        engine.replay_as_recorded(record)


def test_replay_accepts_a_record_from_its_own_agent():
    record = make_run_record(cutoff="2026-08-19")
    engine = ReplayEngine(V50_LITE.coach_settings, target=V50)
    assert engine.verify_fidelity(record).faithful


# -- the gate refuses a corpus that pooled two models -------------------------


def test_the_gate_rejects_a_corpus_spanning_two_models():
    """Refuse a fit whose corpus pooled two models.

    Defence in depth behind the directory layout, and the only check that could
    catch a hand-run with an overridden model: the package fingerprint cannot see
    the model, because the manifest does not record which one a verifier called.
    """
    records = [
        make_run_record(cutoff="2026-08-17", agent_model=LITE_MODEL),
        make_run_record(cutoff="2026-08-18", agent_model=ADVANCED_MODEL),
    ]
    verdict = ComparisonPolicy().evaluate(
        Candidate(candidate_id="c", parent_version="v001"),
        CalibrationVersion(version="v001", effective_from=date(2004, 1, 1)),
        records=records,
        resolution=ResolutionReport(resolved=[], pending=[], data_through=date(2026, 8, 19)),
    )
    failed = {item.name for item in verdict.failed_conditions}
    assert "single_agent_model" in failed
    assert not verdict.passed


def test_the_gate_accepts_a_single_model_corpus_on_that_condition():
    records = [make_run_record(cutoff=day, agent_model=LITE_MODEL) for day in ("2026-08-17", "2026-08-18")]
    verdict = ComparisonPolicy().evaluate(
        Candidate(candidate_id="c", parent_version="v001"),
        CalibrationVersion(version="v001", effective_from=date(2004, 1, 1)),
        records=records,
        resolution=ResolutionReport(resolved=[], pending=[], data_through=date(2026, 8, 19)),
    )
    assert "single_agent_model" not in {item.name for item in verdict.failed_conditions}
