"""Audit-only controls: the source validator switch and the claim-support verifier."""

from __future__ import annotations

import json
from types import SimpleNamespace

from energy_oil_forecasting.cfm_agent_v_6_0 import llm_roles
from energy_oil_forecasting.cfm_agent_v_6_0.claim_support_verifier import (
    VERIFIER_ID,
    audit_claim_support,
)
from energy_oil_forecasting.cfm_agent_v_6_0.config import CfmV60Settings
from energy_oil_forecasting.cfm_agent_v_6_0.outputs import CfmContextAssessmentOutput
from energy_oil_forecasting.cfm_agent_v_6_0.schemas import EvidenceClaim, HorizonAction
from energy_oil_forecasting.cfm_agent_v_6_0.source_validator import run_source_audit


def build_assessment(packet, *, passage_ids=None, contradicting_passage_ids=None) -> CfmContextAssessmentOutput:
    summary = packet.verified_summaries[0]
    return CfmContextAssessmentOutput(
        research_packet_id=packet.packet_id,
        evidence_claims=[
            EvidenceClaim(
                claim_id="c1",
                statement="Confirmed physical oil supply information",
                claim_type="physical_supply",
                supporting_summary_ids=[summary.verified_summary_id],
                supporting_passage_ids=["01.01"] if passage_ids is None else list(passage_ids),
                supporting_source_ids=list(summary.associated_source_ids),
                contradicting_passage_ids=list(contradicting_passage_ids or []),
            )
        ],
        confidence=0.8,
        horizon_actions=[HorizonAction(horizon=5, rationale="test")],
        research_summary="test",
        overall_rationale="test",
    )


def semantic_response(findings: list[dict]) -> SimpleNamespace:
    content = json.dumps({"findings": findings})
    return SimpleNamespace(
        model="gemini-3.5-flash", choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def test_source_audit_off_does_not_fetch(packet):
    bundle = run_source_audit(packet.sources, packet.cutoff_date, CfmV60Settings())
    assert bundle.audit_enabled is False
    assert bundle.executed is False
    assert bundle.records == []
    assert bundle.validator_id == "source_validator_v60_audit_only"


def test_claim_audit_fails_closed_when_semantic_verifier_fails(monkeypatch, packet):
    assessment = build_assessment(packet)

    def fail_verifier(**_):
        raise TimeoutError("forced semantic-verifier timeout")

    monkeypatch.setattr("litellm.completion", fail_verifier)
    llm_roles.RECORDER.reset()

    findings, audit = audit_claim_support(assessment, packet, CfmV60Settings(audit_enabled=True))

    assert audit["verifier_id"] == VERIFIER_ID == "claim_support_verifier_v60_audit_only"
    assert audit["audit_only"] is True
    assert audit["executed"] is True
    assert audit["completed"] is False
    assert audit["semantic_verification_attempted"] is True
    assert "TimeoutError" in audit["semantic_verification_error"]
    assert findings[0].claim_id == "c1"
    assert findings[0].verdict == "not_assessed"
    assert findings[0].confidence == 0.0
    assert findings[0].supporting_passage_ids == ["01.01"]
    record = llm_roles.RECORDER.snapshot()["claim_support_verifier"][0]
    assert record["note"] == "call failed: TimeoutError"
    assert record["parameters"]["model"] == "openai/gemini-3.5-flash"


def test_claim_audit_uses_only_the_cited_passages(monkeypatch, packet):
    summary = packet.verified_summaries[0]
    cited_text = summary.passages[0].text
    uncited_text = summary.passages[1].text
    assessment = build_assessment(packet, passage_ids=["01.01"])
    captured: dict = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return semantic_response(
            [{"claim_id": "c1", "verdict": "direct", "unsupported_elements": [], "confidence": 0.98}]
        )

    monkeypatch.setattr("litellm.completion", fake_completion)
    llm_roles.RECORDER.reset()

    findings, audit = audit_claim_support(assessment, packet, CfmV60Settings(audit_enabled=True))

    assert audit["audit_only"] is True
    assert audit["executed"] is True
    assert audit["completed"] is True
    assert audit["semantic_verification_attempted"] is True
    assert audit["semantic_verification_error"] is None
    assert findings[0].claim_id == "c1"
    assert findings[0].verdict == "direct"
    assert findings[0].confidence == 0.98
    assert findings[0].supporting_passage_ids == ["01.01"]
    assert findings[0].supporting_summary_ids == [summary.verified_summary_id]

    prompt = captured["messages"][-1]["content"]
    assert captured["messages"][-1]["role"] == "user"
    assert cited_text in prompt
    assert uncited_text not in prompt
    assert summary.cleaned_summary not in prompt
    payload = json.loads(prompt)
    assert payload["claims"][0]["supporting_passages"] == [{"passage_id": "01.01", "text": cited_text}]
    assert payload["claims"][0]["contradicting_passages"] == []
    assert "cleaned_summary" not in prompt

    assert captured["model"] == "openai/gemini-3.5-flash"
    assert captured["temperature"] == 0.0
    assert captured["max_tokens"] == 4096
    assert captured["timeout"] == 60
    assert captured["top_p"] == 1.0
    assert captured["n"] == 1
    assert captured["drop_params"] is True
    assert "parallel_tool_calls" not in captured
    assert captured["response_format"]["type"] == "json_schema"

    record = llm_roles.RECORDER.snapshot()["claim_support_verifier"][0]
    assert record["parameters"]["model"] == "openai/gemini-3.5-flash"
    assert record["parameters"]["temperature"] == 0.0
    assert record["parameters"]["max_tokens"] == 4096
    assert record["parameters"]["timeout"] == 60
    assert record["parameters"]["response_format_type"] == "json_schema"
    assert record["response_model"] == "gemini-3.5-flash"
    assert "messages" not in record["parameters"]


def test_claim_citing_a_passage_outside_its_summary_is_settled_without_a_call(monkeypatch, packet):
    assessment = build_assessment(packet, passage_ids=["01.01", "02.01"])

    def unexpected_call(**_):
        raise AssertionError("the semantic verifier must not be called for a structurally invalid claim")

    monkeypatch.setattr("litellm.completion", unexpected_call)

    findings, audit = audit_claim_support(assessment, packet, CfmV60Settings(audit_enabled=True))
    assert audit["semantic_verification_attempted"] is False
    assert audit["semantic_verification_error"] is None
    assert audit["completed"] is True
    assert findings[0].verdict == "unsupported"
    assert findings[0].confidence == 1.0
    assert findings[0].supporting_passage_ids == ["01.01"]
    assert any("passage" in element for element in findings[0].unsupported_elements)


def test_claim_without_passages_is_unsupported_without_a_call(monkeypatch, packet):
    assessment = build_assessment(packet, passage_ids=[])

    def unexpected_call(**_):
        raise AssertionError("no call expected")

    monkeypatch.setattr("litellm.completion", unexpected_call)
    findings, audit = audit_claim_support(assessment, packet, CfmV60Settings(audit_enabled=True))
    assert audit["semantic_verification_attempted"] is False
    assert findings[0].verdict == "unsupported"
    assert "No supporting passage cited" in findings[0].unsupported_elements


def test_claim_audit_rejects_a_response_that_does_not_cover_the_requested_ids(monkeypatch, packet):
    assessment = build_assessment(packet)
    monkeypatch.setattr(
        "litellm.completion",
        lambda **_: semantic_response(
            [{"claim_id": "c9", "verdict": "direct", "unsupported_elements": [], "confidence": 0.9}]
        ),
    )
    findings, audit = audit_claim_support(assessment, packet, CfmV60Settings(audit_enabled=True))
    assert audit["completed"] is False
    assert "requested claim IDs" in audit["semantic_verification_error"]
    assert findings[0].verdict == "not_assessed"
