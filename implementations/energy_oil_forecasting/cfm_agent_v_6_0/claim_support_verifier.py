"""Audit-only claim-support verifier for CFM Agent v6.0 (specification Section 4.1).

Each claim is audited against its cited passages only: the texts of its
``supporting_passage_ids`` and ``contradicting_passage_ids``, never the whole
summary and never destination-page text. Structural problems (a passage that
does not belong to a cited summary, a source not associated with a cited summary)
are settled in Python without an LLM call. The semantic call uses the
``claim_support_verifier`` role parameters from settings, records the exact
keyword arguments handed to ``litellm``, and fails closed to ``not_assessed``
findings on any error. The findings are audit-only and never change a forecast.
"""

from __future__ import annotations

import json
from typing import Literal

from aieng.forecasting.methods.llm_processes._client import (
    make_json_schema_response_format,
    strip_markdown_fence,
)
from energy_oil_forecasting.cfm_agent_v_6_0.config import CfmV60Settings
from energy_oil_forecasting.cfm_agent_v_6_0.llm_roles import (
    completion_kwargs,
    record_effective_parameters,
)
from energy_oil_forecasting.cfm_agent_v_6_0.outputs import CfmContextAssessmentOutput
from energy_oil_forecasting.cfm_agent_v_6_0.passages import passage_texts_for
from energy_oil_forecasting.cfm_agent_v_6_0.schemas import (
    ClaimSupportFinding,
    ResearchPacket,
    VerifiedSummary,
)
from pydantic import BaseModel, ConfigDict, Field


VERIFIER_ID = "claim_support_verifier_v60_audit_only"
ROLE = "claim_support_verifier"


class SemanticFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_id: str
    verdict: Literal["direct", "partial", "unsupported", "contradicted"]
    unsupported_elements: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class SemanticResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    findings: list[SemanticFinding]


SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": ["direct", "partial", "unsupported", "contradicted"],
                    },
                    "unsupported_elements": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": [
                    "claim_id",
                    "verdict",
                    "unsupported_elements",
                    "confidence",
                ],
            },
        }
    },
    "required": ["findings"],
}

INSTRUCTION = (
    "Audit every atomic claim against only the supplied cited passages. Evaluate the supporting passages "
    "and any contradicting passages. Return exactly one finding for every requested claim ID, with no "
    "unknown or duplicate IDs. Return direct only if every material element is entailed by the supporting "
    "passages and not contradicted. Return partial if only part is supported, unsupported if support is "
    "absent, and contradicted for a material conflict. Do not use outside knowledge. This audit cannot "
    "change the forecast."
)


def _accepted(summary_by: dict[str, VerifiedSummary], summary_ids: list[str]) -> list[VerifiedSummary]:
    return [
        summary_by[value] for value in summary_ids if value in summary_by and summary_by[value].status == "accepted"
    ]


def _semantic_call(settings: CfmV60Settings, inputs: list[dict[str, object]]) -> SemanticResponse:
    """One structured verifier call under the ``claim_support_verifier`` role parameters."""
    import litellm  # noqa: PLC0415

    kwargs = completion_kwargs(settings, ROLE)
    kwargs["messages"] = [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": json.dumps({"claims": inputs})},
    ]
    kwargs["response_format"] = make_json_schema_response_format("ClaimSupportAudit", SCHEMA)
    try:
        response = litellm.completion(**kwargs)
    except Exception as exc:
        record_effective_parameters(ROLE, kwargs, note=f"call failed: {type(exc).__name__}")
        raise
    record_effective_parameters(ROLE, kwargs, response_model=getattr(response, "model", None))
    return SemanticResponse.model_validate(
        json.loads(strip_markdown_fence(response.choices[0].message.content or "{}"))
    )


def audit_claim_support(  # noqa: PLR0912, PLR0914, PLR0915
    assessment: CfmContextAssessmentOutput,
    packet: ResearchPacket,
    settings: CfmV60Settings,
) -> tuple[list[ClaimSupportFinding], dict[str, object]]:
    """Audit every claim against its cited passages; audit-only and nonfatal."""
    summary_by = {item.verified_summary_id: item for item in packet.verified_summaries}
    source_ids = {item.source_id for item in packet.sources}
    structural: dict[str, ClaimSupportFinding] = {}
    inputs: list[dict[str, object]] = []
    for claim in assessment.evidence_claims:
        supporting = _accepted(summary_by, claim.supporting_summary_ids)
        contradicting = _accepted(summary_by, claim.contradicting_summary_ids)
        supporting_passages = passage_texts_for(supporting, claim.supporting_passage_ids)
        contradicting_passages = passage_texts_for(contradicting, claim.contradicting_passage_ids)
        valid_support_sources = [value for value in claim.supporting_source_ids if value in source_ids]
        valid_contradict_sources = [value for value in claim.contradicting_source_ids if value in source_ids]
        support_associated = {value for summary in supporting for value in summary.associated_source_ids}
        contradict_associated = {value for summary in contradicting for value in summary.associated_source_ids}
        errors: list[str] = []
        if not supporting:
            errors.append("No valid accepted supporting summary")
        if not claim.supporting_passage_ids:
            errors.append("No supporting passage cited")
        elif set(claim.supporting_passage_ids) - set(supporting_passages):
            errors.append("A supporting passage ID does not belong to a cited supporting summary")
        if not valid_support_sources:
            errors.append("No valid supporting source")
        if set(valid_support_sources) - support_associated:
            errors.append("A supporting source is not associated with a cited supporting summary")
        if claim.contradicting_summary_ids and len(contradicting) != len(set(claim.contradicting_summary_ids)):
            errors.append("A contradicting summary ID is invalid or not accepted")
        if set(claim.contradicting_passage_ids) - set(contradicting_passages):
            errors.append("A contradicting passage ID does not belong to a cited contradicting summary")
        if claim.contradicting_source_ids and set(valid_contradict_sources) - contradict_associated:
            errors.append("A contradicting source is not associated with a cited contradicting summary")
        if errors:
            structural[claim.claim_id] = ClaimSupportFinding(
                claim_id=claim.claim_id,
                verdict="unsupported",
                supporting_summary_ids=[item.verified_summary_id for item in supporting],
                supporting_passage_ids=sorted(supporting_passages),
                supporting_source_ids=valid_support_sources,
                unsupported_elements=errors,
                confidence=1.0,
            )
            continue
        inputs.append(
            {
                "claim_id": claim.claim_id,
                "claim": claim.statement,
                "supporting_passages": [
                    {"passage_id": passage_id, "text": text} for passage_id, text in supporting_passages.items()
                ],
                "contradicting_passages": [
                    {"passage_id": passage_id, "text": text} for passage_id, text in contradicting_passages.items()
                ],
            }
        )

    semantic: dict[str, SemanticFinding] = {}
    error: str | None = None
    requested_ids = [str(item["claim_id"]) for item in inputs]
    if inputs:
        try:
            parsed = _semantic_call(settings, inputs)
            returned_ids = [item.claim_id for item in parsed.findings]
            if len(returned_ids) != len(set(returned_ids)):
                raise ValueError("semantic verifier returned duplicate claim IDs")
            if set(returned_ids) != set(requested_ids):
                raise ValueError("semantic verifier response does not exactly cover requested claim IDs")
            semantic = {item.claim_id: item for item in parsed.findings}
        except Exception as exc:  # noqa: BLE001
            error = f"semantic_verifier_failed: {type(exc).__name__}: {exc}"

    ordered: list[ClaimSupportFinding] = []
    for claim in assessment.evidence_claims:
        if claim.claim_id in structural:
            ordered.append(structural[claim.claim_id])
            continue
        item = semantic.get(claim.claim_id)
        ordered.append(
            ClaimSupportFinding(
                claim_id=claim.claim_id,
                verdict="not_assessed" if item is None else item.verdict,
                supporting_summary_ids=claim.supporting_summary_ids,
                supporting_passage_ids=claim.supporting_passage_ids,
                supporting_source_ids=claim.supporting_source_ids,
                unsupported_elements=[error or "No semantic finding returned"]
                if item is None
                else item.unsupported_elements,
                confidence=0.0 if item is None else item.confidence,
            )
        )
    completed = error is None and all(item.verdict != "not_assessed" for item in ordered)
    audit = {
        "verifier_id": VERIFIER_ID,
        "audit_only": True,
        "executed": True,
        "completed": completed,
        "semantic_verification_attempted": bool(inputs),
        "semantic_verification_error": error,
        "requested_claim_ids": [claim.claim_id for claim in assessment.evidence_claims],
        "findings": [item.model_dump(mode="json") for item in ordered],
    }
    return ordered, audit


__all__ = ["VERIFIER_ID", "audit_claim_support"]
