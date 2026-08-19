"""Audit-only semantic claim support verifier for CFM Agent v5.2."""

from __future__ import annotations

import json
import os
from typing import Literal

from aieng.forecasting.methods.llm_processes._client import (
    make_json_schema_response_format,
    strip_markdown_fence,
)
from energy_oil_forecasting.cfm_agent_v_5_2.config import CfmV52Settings
from energy_oil_forecasting.cfm_agent_v_5_2.outputs import CfmContextAssessmentOutput
from energy_oil_forecasting.cfm_agent_v_5_2.schemas import (
    ClaimSupportFinding,
    ResearchPacket,
)
from pydantic import BaseModel, ConfigDict, Field


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

INSTRUCTION = """Audit every atomic claim against only the supplied cited cleaned summaries. Evaluate supporting and contradicting summaries. Return exactly one finding for every requested claim ID, with no unknown or duplicate IDs. Return direct only if every material element is entailed and not contradicted. Return partial if only part is supported, unsupported if support is absent, and contradicted for a material conflict. Do not use outside knowledge. This audit cannot change the forecast."""


def audit_claim_support(  # noqa: PLR0912, PLR0915
    assessment: CfmContextAssessmentOutput,
    packet: ResearchPacket,
    settings: CfmV52Settings,
) -> tuple[list[ClaimSupportFinding], dict[str, object]]:
    summaries = {item.verified_summary_id: item for item in packet.verified_summaries}
    sources = {item.source_id for item in packet.sources}
    structural: dict[str, ClaimSupportFinding] = {}
    inputs: list[dict[str, object]] = []
    for claim in assessment.evidence_claims:
        supporting = [
            summaries[value]
            for value in claim.supporting_summary_ids
            if value in summaries and summaries[value].status == "accepted"
        ]
        contradicting = [
            summaries[value]
            for value in claim.contradicting_summary_ids
            if value in summaries and summaries[value].status == "accepted"
        ]
        valid_support_sources = [value for value in claim.supporting_source_ids if value in sources]
        valid_contradict_sources = [value for value in claim.contradicting_source_ids if value in sources]
        support_associated = {value for summary in supporting for value in summary.associated_source_ids}
        contradict_associated = {value for summary in contradicting for value in summary.associated_source_ids}
        errors: list[str] = []
        if not supporting:
            errors.append("No valid accepted supporting summary")
        if not valid_support_sources:
            errors.append("No valid supporting source")
        if set(valid_support_sources) - support_associated:
            errors.append("A supporting source is not associated with a cited supporting summary")
        if claim.contradicting_summary_ids and len(contradicting) != len(set(claim.contradicting_summary_ids)):
            errors.append("A contradicting summary ID is invalid or not accepted")
        if claim.contradicting_source_ids and set(valid_contradict_sources) - contradict_associated:
            errors.append("A contradicting source is not associated with a cited contradicting summary")
        if errors:
            structural[claim.claim_id] = ClaimSupportFinding(
                claim_id=claim.claim_id,
                verdict="unsupported",
                supporting_summary_ids=[item.verified_summary_id for item in supporting],
                supporting_source_ids=valid_support_sources,
                unsupported_elements=errors,
                confidence=1.0,
            )
            continue
        inputs.append(
            {
                "claim_id": claim.claim_id,
                "claim": claim.statement,
                "supporting_cleaned_summaries": [
                    {
                        "verified_summary_id": item.verified_summary_id,
                        "text": item.cleaned_summary,
                    }
                    for item in supporting
                ],
                "contradicting_cleaned_summaries": [
                    {
                        "verified_summary_id": item.verified_summary_id,
                        "text": item.cleaned_summary,
                    }
                    for item in contradicting
                ],
            }
        )
    semantic: dict[str, SemanticFinding] = {}
    error: str | None = None
    requested_ids = [str(item["claim_id"]) for item in inputs]
    if inputs:
        try:
            import litellm  # noqa: PLC0415

            model = settings.claim_support_verifier_model
            model = model if model.startswith("openai/") else f"openai/{model}"
            response = litellm.completion(
                model=model,
                api_base=os.getenv("OPENAI_BASE_URL"),
                api_key=os.getenv("OPENAI_API_KEY"),
                messages=[
                    {"role": "system", "content": INSTRUCTION},
                    {"role": "user", "content": json.dumps({"claims": inputs})},
                ],
                response_format=make_json_schema_response_format("ClaimSupportAudit", SCHEMA),
                temperature=0.0,
                max_tokens=4096,
                timeout=settings.claim_support_verifier_timeout_seconds,
            )
            parsed = SemanticResponse.model_validate(
                json.loads(strip_markdown_fence(response.choices[0].message.content or "{}"))
            )
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
        if item is None:
            ordered.append(
                ClaimSupportFinding(
                    claim_id=claim.claim_id,
                    verdict="not_assessed",
                    supporting_summary_ids=claim.supporting_summary_ids,
                    supporting_source_ids=claim.supporting_source_ids,
                    unsupported_elements=[error or "No semantic finding returned"],
                    confidence=0.0,
                )
            )
        else:
            ordered.append(
                ClaimSupportFinding(
                    claim_id=claim.claim_id,
                    verdict=item.verdict,
                    supporting_summary_ids=claim.supporting_summary_ids,
                    supporting_source_ids=claim.supporting_source_ids,
                    unsupported_elements=item.unsupported_elements,
                    confidence=item.confidence,
                )
            )
    completed = error is None and all(item.verdict != "not_assessed" for item in ordered)
    audit = {
        "verifier_id": "claim_support_verifier_v52_audit_only",
        "audit_only": True,
        "executed": True,
        "completed": completed,
        "semantic_verification_attempted": bool(inputs),
        "semantic_verification_error": error,
        "requested_claim_ids": [claim.claim_id for claim in assessment.evidence_claims],
        "findings": [item.model_dump(mode="json") for item in ordered],
    }
    return ordered, audit


__all__ = ["audit_claim_support"]
