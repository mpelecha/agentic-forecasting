"""Deterministic evidence and narrative sanitization for CFM Agent v3.4."""

from __future__ import annotations

from contextlib import suppress
from datetime import datetime
from urllib.parse import urlparse

from energy_oil_forecasting.cfm_agent_v_3_4.config import CfmV34Settings
from energy_oil_forecasting.cfm_agent_v_3_4.outputs import CfmContextAssessmentOutput


_PLACEHOLDERS = {
    "various", "unknown", "multiple sources", "news reports", "several outlets",
    "google search", "search results", "vertex ai search", "independent market reporting",
    "maritime analytics reporting", "energy market news",
}
_REJECTED_HOSTS = {"vertexaisearch.cloud.google.com"}
_PHYSICAL_CLAIM_TYPES = {"physical_supply", "shipping", "production_policy", "strategic_reserves", "inventory"}


def _url_quality(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if not host:
        return "invalid"
    if host in _REJECTED_HOSTS:
        return "opaque_redirect"
    if not parsed.path.strip("/"):
        return "publisher_homepage"
    return "resolved_specific"


def sanitize_assessment(  # noqa: PLR0912, PLR0915 — one linear rejection pass; splitting it hides the order.
    assessment: CfmContextAssessmentOutput,
    *,
    cutoff_date: str,
    settings: CfmV34Settings,
) -> tuple[CfmContextAssessmentOutput, dict[str, object]]:
    """Return a conservative assessment plus claim- and source-level audit metadata."""
    raw = assessment.model_dump(mode="json")
    cutoff = datetime.strptime(cutoff_date, "%Y-%m-%d").date()
    eligible: set[str] = set()
    eligible_publishers: set[str] = set()
    source_audit: list[dict[str, object]] = []
    changes: list[str] = []
    warnings: list[str] = []

    for source in raw["evidence_sources"]:
        publisher = source["publisher"].strip()
        publisher_key = publisher.lower()
        quality = _url_quality(source["source_url"])
        reasons: list[str] = []
        publication_date = None
        with suppress(TypeError, ValueError):
            if source.get("publication_date"):
                publication_date = datetime.strptime(source["publication_date"], "%Y-%m-%d").date()
        if publisher_key in _PLACEHOLDERS:
            reasons.append("publisher is missing or a placeholder")
        if source.get("provenance_status") != "verified_from_tool":
            reasons.append("source identity is not verified_from_tool")
        if source.get("verifier_content_status") != "accepted_factual_content":
            reasons.append("verifier returned no accepted factual content")
        if not source.get("verified_evidence_excerpt", "").strip():
            reasons.append("verified evidence excerpt is empty")
        if publication_date is None:
            reasons.append("publication date is missing or invalid")
        elif publication_date >= cutoff:
            reasons.append("publication date is not strictly before cutoff")
        if not source.get("title", "").strip() or source["title"].lstrip().startswith("<"):
            reasons.append("document title is missing or unresolved")
        if quality == "invalid":
            reasons.append("source URL is invalid")
        elif quality in {"opaque_redirect", "publisher_homepage"}:
            complete_identity = bool(
                publisher and source.get("title") and publication_date and source.get("verified_evidence_excerpt", "").strip()
            )
            if not settings.allow_verified_identity_with_imperfect_url or not complete_identity:
                reasons.append("source URL is not a resolved auditable document")
            else:
                warnings.append(
                    f"Source {source['source_id']} accepted from verified identity metadata with URL quality {quality}."
                )

        source_ok = not reasons
        if source_ok:
            eligible.add(source["source_id"])
            eligible_publishers.add(publisher_key)
        else:
            if source["source_tier"] != "tier_4_other" or source.get("is_primary_or_official"):
                changes.append(f"Downgraded source {source['source_id']} classification.")
            source["source_tier"] = "tier_4_other"
            source["is_primary_or_official"] = False
        source_audit.append({
            "source_id": source["source_id"],
            "publisher": publisher,
            "eligible": source_ok,
            "url_quality": quality,
            "reasons": reasons,
        })

    claim_audit: list[dict[str, object]] = []
    qualifying_claim_ids: set[str] = set()
    for claim in raw["evidence_claims"]:
        eligible_support = [sid for sid in claim["supporting_source_ids"] if sid in eligible]
        eligible_contradictions = [sid for sid in claim["contradicting_source_ids"] if sid in eligible]
        claim["supporting_source_ids"] = eligible_support
        claim["contradicting_source_ids"] = eligible_contradictions
        direct_eligible_publishers = {
            source["publisher"].strip().lower()
            for source in raw["evidence_sources"]
            if source["source_id"] in eligible_support
        }
        if claim["support_status"] == "direct" and not direct_eligible_publishers:
            claim["support_status"] = "unsupported"
            changes.append(f"Downgraded claim {claim['claim_id']} to unsupported.")
        if claim["support_status"] == "direct" and direct_eligible_publishers:
            qualifying_claim_ids.add(claim["claim_id"])
        claim_audit.append({
            "claim_id": claim["claim_id"],
            "support_status": claim["support_status"],
            "eligible_supporting_source_ids": eligible_support,
            "eligible_supporting_publishers": sorted(direct_eligible_publishers),
            "eligible_support_count": len(direct_eligible_publishers),
            "eligible_contradicting_source_ids": eligible_contradictions,
        })

    qualifying_physical_claims = [
        claim for claim in raw["evidence_claims"]
        if claim["claim_type"] in _PHYSICAL_CLAIM_TYPES and claim["support_status"] == "direct"
    ]
    if raw["physical_status"] in {"partial_disruption", "confirmed_disruption"} and not qualifying_physical_claims:
        raw["physical_status"] = "unknown"
        changes.append("Downgraded physical_status to unknown.")
    if raw["physical_status"] == "elevated_risk" and not qualifying_claim_ids:
        raw["physical_status"] = "unknown"
        changes.append("Downgraded elevated_risk physical_status to unknown.")

    if not qualifying_claim_ids:
        raw["confidence"] = 0.0
        raw["research_summary"] = "No eligible directly supported material claim survives sanitization."
        raw["overall_rationale"] = (
            "The original LLM assessment is retained separately; the sanitized assessment does not establish a contextual deviation."
        )
        warnings.append("Assessment confidence reduced to zero because no eligible direct claim survived.")
    raw["warnings"] = list(dict.fromkeys(list(raw.get("warnings", [])) + warnings))

    sanitized = CfmContextAssessmentOutput.model_validate(raw)
    audit = {
        "sanitizer_id": "evidence_sanitizer_v2",
        "eligible_source_ids": sorted(eligible),
        "eligible_publishers": sorted(eligible_publishers),
        "eligible_publisher_count": len(eligible_publishers),
        "qualifying_claim_ids": sorted(qualifying_claim_ids),
        "qualifying_physical_claim_ids": sorted(claim["claim_id"] for claim in qualifying_physical_claims),
        "source_audit": source_audit,
        "claim_audit": claim_audit,
        "change_count": len(changes),
        "changes": changes,
        "warnings": warnings,
    }
    return sanitized, audit


__all__ = ["sanitize_assessment"]
