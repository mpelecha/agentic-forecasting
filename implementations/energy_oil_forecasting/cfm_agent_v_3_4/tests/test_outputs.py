"""Assessment schema tests."""

import pytest
from energy_oil_forecasting.cfm_agent_v_3_4.outputs import CfmContextAssessmentOutput
from pydantic import ValidationError


def neutral() -> dict[str, object]:
    return {
        "reported_search_queries": [],
        "evidence_sources": [],
        "evidence_claims": [],
        "physical_status": "unknown",
        "incremental_novelty": "indeterminate",
        "material_evidence_conflict": False,
        "confidence": 0.2,
        "horizon_actions": [
            {
                "horizon": 5,
                "center_action": "no_change",
                "uncertainty_action": "unchanged",
                "persistence_profile": "unknown",
                "cited_claim_ids": [],
                "rationale": "Insufficient evidence.",
            }
        ],
        "research_summary": "No eligible evidence.",
        "overall_rationale": "Neutral assessment.",
    }


def test_neutral_assessment_can_abstain_without_sources() -> None:
    parsed = CfmContextAssessmentOutput.model_validate(neutral())
    assert parsed.action_for(5).center_action == "no_change"


def test_unknown_claim_reference_is_rejected() -> None:
    raw = neutral()
    raw["horizon_actions"][0]["cited_claim_ids"] = ["missing"]  # type: ignore[index]
    with pytest.raises(ValidationError, match="unknown claim_id"):
        CfmContextAssessmentOutput.model_validate(raw)
