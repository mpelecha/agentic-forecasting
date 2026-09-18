"""ReviewSettings: the coach never runs on the lite model, and effort values the proxy rejects never leave."""

from __future__ import annotations

import pytest
from aieng.forecasting.models import ADVANCED_MODEL, LITE_MODEL
from pydantic import ValidationError

from energy_oil_forecasting.cfm_coach.report import MIN_EFFECTIVE_N_FOR_R1
from energy_oil_forecasting.cfm_coach.review.settings import (
    MIN_EFFECTIVE_N_POWERED,
    ReviewSettings,
    StageLlm,
)


def test_defaults_to_the_advanced_model_for_every_stage():
    s = ReviewSettings()
    assert s.model == ADVANCED_MODEL
    assert s.challenger_model == ADVANCED_MODEL


def test_the_lite_model_is_rejected_for_coach_and_challenger():
    with pytest.raises(ValidationError, match="lite"):
        ReviewSettings(model=LITE_MODEL)
    with pytest.raises(ValidationError, match="lite"):
        ReviewSettings(challenger_model="gemini-9-lite")


@pytest.mark.parametrize("effort", ["low", "disable", "none", "LOW"])
def test_efforts_the_proxy_rejects_are_rejected_here(effort):
    with pytest.raises(ValidationError):
        StageLlm(call_class="judgment", reasoning_effort=effort, max_tokens=1024)


def test_stage_defaults_differ_by_effort_and_budget_not_model():
    s = ReviewSettings()
    assert s.triage.reasoning_effort == "minimal"
    assert s.deep_dive.reasoning_effort == "medium"
    assert s.synthesis.reasoning_effort == "high"
    assert s.critic.reasoning_effort == "high"
    assert s.lookup.reasoning_effort is None
    assert s.synthesis.max_tokens > s.triage.max_tokens


def test_judgment_calls_run_at_default_temperature_and_lookups_are_pinned():
    s = ReviewSettings()
    assert s.temperature_for("triage") == 1.0
    assert s.temperature_for("synthesis") == 1.0
    assert s.temperature_for("lookup") == 0.0
    assert s.temperature_for("probe") == 0.0


def test_lookup_stage_cannot_be_reclassified_as_judgment():
    with pytest.raises(ValidationError, match="lookup"):
        ReviewSettings(lookup=StageLlm(call_class="judgment", max_tokens=1024))


def test_frozen_and_extra_forbidden():
    s = ReviewSettings()
    with pytest.raises(ValidationError):
        s.model = "x"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        ReviewSettings(unknown_knob=1)  # type: ignore[call-arg]


def test_powered_threshold_matches_the_report():
    assert MIN_EFFECTIVE_N_POWERED == MIN_EFFECTIVE_N_FOR_R1


def test_data_layout_follows_data_dir(tmp_path):
    s = ReviewSettings(data_dir=tmp_path)
    assert s.cards_dir == tmp_path / "cards"
    assert s.coached_dir == tmp_path / "coached"
    assert s.calibration_dir == tmp_path / "calibration"
