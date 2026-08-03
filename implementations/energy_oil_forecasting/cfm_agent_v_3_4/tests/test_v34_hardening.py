"""V3.2 dedicated-tool, audit, and sanitization tests."""

import inspect

from energy_oil_forecasting.cfm_agent_v_3_4 import (
    CfmContextAssessmentOutput,
    CfmV34Settings,
)
from energy_oil_forecasting.cfm_agent_v_3_4.sanitizer import (
    sanitize_assessment,
)
from energy_oil_forecasting.cfm_agent_v_3_4.tools import (
    AuthoritativeSuiteTool,
)


def _assessment() -> CfmContextAssessmentOutput:
    """Return an intentionally unsupported LLM assessment."""
    return CfmContextAssessmentOutput.model_validate(
        {
            "reported_search_queries": [
                "q1",
                "q2",
                "q3",
                "q4",
            ],
            "evidence_sources": [
                {
                    "source_id": "s1",
                    "title": "Claimed source",
                    "source_url": ("https://vertexaisearch.cloud.google.com/grounding-api-redirect/x"),
                    "publisher": "Energy Market News",
                    "source_tier": "tier_1_primary",
                    "publication_date": "2026-03-01",
                    "is_primary_or_official": True,
                    "provenance_status": "inferred_by_agent",
                    "verifier_content_status": ("accepted_factual_content"),
                    "verifier_processing_status": "unknown",
                    "verified_evidence_excerpt": "text",
                }
            ],
            "evidence_claims": [
                {
                    "claim_id": "c1",
                    "statement": "Disruption occurred.",
                    "claim_type": "shipping",
                    "support_status": "direct",
                    "supporting_source_ids": ["s1"],
                    "contradicting_source_ids": [],
                }
            ],
            "physical_status": "confirmed_disruption",
            "incremental_novelty": ("likely_new_relative_to_model_data"),
            "material_evidence_conflict": False,
            "confidence": 0.9,
            "horizon_actions": [
                {
                    "horizon": 5,
                    "center_action": "small_up",
                    "uncertainty_action": ("moderately_wider"),
                    "persistence_profile": "persistent",
                    "cited_claim_ids": ["c1"],
                    "rationale": "Claimed evidence.",
                }
            ],
            "research_summary": "Claimed disruption.",
            "overall_rationale": "Claimed action.",
        }
    )


def test_sanitizer_downgrades_inferred_source_and_claim() -> None:
    """Unsupported source metadata and conclusions must be downgraded."""
    original = _assessment()

    sanitized, audit = sanitize_assessment(
        original,
        cutoff_date="2026-03-02",
        settings=CfmV34Settings(),
    )

    source = sanitized.evidence_sources[0]
    claim = sanitized.evidence_claims[0]

    assert source.source_tier == "tier_4_other"
    assert source.is_primary_or_official is False
    assert claim.support_status == "unsupported"
    assert claim.supporting_source_ids == []
    assert sanitized.physical_status == "unknown"
    assert sanitized.confidence == 0.0

    assert audit["sanitizer_id"] == "evidence_sanitizer_v2"
    assert audit["eligible_source_ids"] == []
    assert audit["change_count"] >= 3

    # The original LLM assessment remains available unchanged.
    assert original.evidence_sources[0].source_tier == "tier_1_primary"
    assert original.evidence_sources[0].is_primary_or_official is True
    assert original.evidence_claims[0].support_status == "direct"
    assert original.physical_status == "confirmed_disruption"
    assert original.confidence == 0.9


def test_dedicated_tool_schema_requires_horizons(
    synthetic_service,
) -> None:
    """The dedicated tool must structurally require horizons."""
    tool = AuthoritativeSuiteTool(
        synthetic_service,
        settings=CfmV34Settings(
            model_num_samples=50,
        ),
    )

    signature = inspect.signature(tool.run_authoritative_suite)

    parameters = signature.parameters

    required_parameters = [
        name
        for name, parameter in parameters.items()
        if (name != "self" and parameter.default is inspect.Parameter.empty)
    ]

    assert required_parameters == [
        "cutoff_date",
        "series_ids",
        "target_series_id",
        "horizons",
    ]

    assert parameters["horizons"].default is inspect.Parameter.empty

    assert "operation" not in parameters

    assert tool.audit == {
        "tool_name": "run_authoritative_suite",
        "attempt_count": 0,
        "validation_failure_count": 0,
        "successful_execution_count": 0,
        "attempt_errors": [],
    }
