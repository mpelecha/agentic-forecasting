import json
from types import SimpleNamespace

from energy_oil_forecasting.cfm_agent_v_5_2.claim_support_verifier import (
    audit_claim_support,
)
from energy_oil_forecasting.cfm_agent_v_5_2.config import CfmV52Settings
from energy_oil_forecasting.cfm_agent_v_5_2.outputs import CfmContextAssessmentOutput
from energy_oil_forecasting.cfm_agent_v_5_2.schemas import EvidenceClaim, HorizonAction
from energy_oil_forecasting.cfm_agent_v_5_2.source_validator import run_source_audit


def test_source_audit_off_does_not_fetch(packet):
    bundle = run_source_audit(packet.sources, packet.cutoff_date, CfmV52Settings())
    assert bundle.audit_enabled is False
    assert bundle.executed is False
    assert bundle.records == []


# def test_claim_audit_uses_summaries(packet):
#     summary = packet.verified_summaries[0]
#     assessment = CfmContextAssessmentOutput(
#         research_packet_id=packet.packet_id,
#         evidence_claims=[
#             EvidenceClaim(
#                 claim_id="c1",
#                 statement="Confirmed physical oil supply information",
#                 claim_type="physical_supply",
#                 supporting_summary_ids=[summary.verified_summary_id],
#                 supporting_source_ids=list(summary.associated_source_ids),
#             )
#         ],
#         confidence=0.8,
#         horizon_actions=[HorizonAction(horizon=5, rationale="test")],
#         research_summary="test",
#         overall_rationale="test",
#     )
#     findings, audit = audit_claim_support(assessment, packet, CfmV52Settings())
#     assert audit["audit_only"] is True
#     assert findings[0].verdict == "not_assessed"
#     assert findings[0].supporting_summary_ids == [summary.verified_summary_id]


def test_claim_audit_fails_closed_when_semantic_verifier_fails(
    monkeypatch,
    packet,
):
    summary = packet.verified_summaries[0]

    assessment = CfmContextAssessmentOutput(
        research_packet_id=packet.packet_id,
        evidence_claims=[
            EvidenceClaim(
                claim_id="c1",
                statement="Confirmed physical oil supply information",
                claim_type="physical_supply",
                supporting_summary_ids=[summary.verified_summary_id],
                supporting_source_ids=list(summary.associated_source_ids),
            )
        ],
        confidence=0.8,
        horizon_actions=[
            HorizonAction(
                horizon=5,
                rationale="test",
            )
        ],
        research_summary="test",
        overall_rationale="test",
    )

    def fail_verifier(**_):
        raise TimeoutError("forced semantic-verifier timeout")

    monkeypatch.setattr(
        "litellm.completion",
        fail_verifier,
    )

    findings, audit = audit_claim_support(
        assessment,
        packet,
        CfmV52Settings(audit_enabled=True),
    )

    assert audit["audit_only"] is True
    assert audit["executed"] is True
    assert audit["completed"] is False
    assert audit["semantic_verification_attempted"] is True
    assert "TimeoutError" in audit["semantic_verification_error"]
    assert findings[0].claim_id == "c1"
    assert findings[0].verdict == "not_assessed"
    assert findings[0].confidence == 0.0


def test_claim_audit_uses_summaries(monkeypatch, packet):
    summary = packet.verified_summaries[0]

    assessment = CfmContextAssessmentOutput(
        research_packet_id=packet.packet_id,
        evidence_claims=[
            EvidenceClaim(
                claim_id="c1",
                statement="Confirmed physical oil supply information",
                claim_type="physical_supply",
                supporting_summary_ids=[summary.verified_summary_id],
                supporting_source_ids=list(summary.associated_source_ids),
            )
        ],
        confidence=0.8,
        horizon_actions=[
            HorizonAction(
                horizon=5,
                rationale="test",
            )
        ],
        research_summary="test",
        overall_rationale="test",
    )

    payload = {
        "findings": [
            {
                "claim_id": "c1",
                "verdict": "direct",
                "unsupported_elements": [],
                "confidence": 0.98,
            }
        ]
    }

    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))])

    monkeypatch.setattr(
        "litellm.completion",
        lambda **_: response,
    )

    findings, audit = audit_claim_support(
        assessment,
        packet,
        CfmV52Settings(audit_enabled=True),
    )

    assert audit["audit_only"] is True
    assert audit["executed"] is True
    assert audit["completed"] is True
    assert audit["semantic_verification_attempted"] is True
    assert audit["semantic_verification_error"] is None
    assert findings[0].claim_id == "c1"
    assert findings[0].verdict == "direct"
    assert findings[0].confidence == 0.98
