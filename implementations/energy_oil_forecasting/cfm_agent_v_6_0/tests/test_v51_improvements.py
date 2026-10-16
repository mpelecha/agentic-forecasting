"""Structured-output correction and neutral fallback (Section 4.4), carried over from v5.2.

``test_task_derived_seed_and_suite_id_are_repeatable`` from v5.2 is not here: its
v6.0 equivalent (two fresh suite runs bit-equal, identity settings in the suite
ID) lives in ``test_determinism.py``.
"""

import json
from datetime import datetime
from types import SimpleNamespace

from aieng.forecasting.evaluation.task import ForecastingTask
from energy_oil_forecasting.cfm_agent_v_6_0.config import CfmV60Settings
from energy_oil_forecasting.cfm_agent_v_6_0.response_control import (
    StructuredAssessmentController,
)


def _valid(packet_id="packet", physical_status="unknown"):
    return json.dumps(
        {
            "research_packet_id": packet_id,
            "evidence_claims": [],
            "physical_status": physical_status,
            "incremental_novelty": "indeterminate",
            "material_evidence_conflict": False,
            "confidence": 0.0,
            "horizon_actions": [
                {
                    "horizon": 5,
                    "center_action": "no_change",
                    "uncertainty_action": "unchanged",
                    "persistence_profile": "unknown",
                    "cited_claim_ids": [],
                    "narrowing_uncertainty_resolved": "",
                    "rationale": "neutral",
                }
            ],
            "research_summary": "summary",
            "overall_rationale": "rationale",
            "warnings": [],
        }
    )


class FakeInner:
    predictor_id = "fake"
    _runner = None

    def prompt_builder(self, **kwargs):
        return "prompt"


def _task():
    return ForecastingTask(
        task_id="test",
        target_series_id="wti",
        horizons=[5],
        frequency="B",
        description="test",
    )


def _context():
    return SimpleNamespace(as_of=datetime(2026, 8, 13))


def test_invalid_initial_response_is_corrected_once(monkeypatch):
    controller = StructuredAssessmentController(FakeInner(), CfmV60Settings())
    calls = []

    def corrected(**kwargs):
        calls.append(kwargs)
        return _valid()

    monkeypatch.setattr(controller, "_corrected_response", corrected)
    result = controller.resolve(
        task=_task(),
        context=_context(),
        packet_id="packet",
        initial_raw_response=_valid(physical_status="tight"),
    )
    assert len(calls) == 1
    assert result.audit["control_id"] == "structured_assessment_retry_v60"
    assert result.audit["initial_validation_failed"] is True
    assert result.audit["initial_validation_errors"]
    assert result.audit["correction_retry_attempted"] is True
    assert result.audit["correction_request"]["required_research_packet_id"] == "packet"
    assert result.audit["correction_request"]["required_horizons_in_order"] == [5]
    assert result.audit["retry_response"] == _valid()
    assert result.audit["retry_succeeded"] is True
    assert result.audit["neutral_fallback_applied"] is False
    assert result.audit["final_disposition"] == "corrected_response_accepted"


def test_second_invalid_response_produces_neutral_fallback(monkeypatch):
    controller = StructuredAssessmentController(FakeInner(), CfmV60Settings())
    monkeypatch.setattr(controller, "_corrected_response", lambda **kwargs: "not json")
    result = controller.resolve(
        task=_task(),
        context=_context(),
        packet_id="packet",
        initial_raw_response=_valid(physical_status="tight"),
    )
    action = result.assessment.horizon_actions[0]
    assert action.center_action == "no_change"
    assert action.uncertainty_action == "unchanged"
    assert result.assessment.confidence == 0.0
    assert result.assessment.evidence_claims == []
    assert result.assessment.incremental_novelty == "indeterminate"
    assert "structured_output_neutral_fallback" in result.assessment.warnings
    assert result.audit["initial_response"] == _valid(physical_status="tight")
    assert result.audit["retry_response"] == "not json"
    assert result.audit["retry_validation_errors"]
    assert result.audit["neutral_fallback_applied"] is True
    assert result.audit["final_disposition"] == "neutral_fallback_after_retry_failure"
