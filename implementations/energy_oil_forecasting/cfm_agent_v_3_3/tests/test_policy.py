"""Graduated evidence and bounded action-policy tests."""

import pytest

from energy_oil_forecasting.cfm_agent_v_3_3.config import CfmV33Settings
from energy_oil_forecasting.cfm_agent_v_3_3.outputs import CfmContextAssessmentOutput
from energy_oil_forecasting.cfm_agent_v_3_3.policy import ConstrainedActionPolicy, EnsembleLockedPolicy
from energy_oil_forecasting.cfm_agent_v_3_3.schemas import ModelHorizonForecast


def ensemble() -> ModelHorizonForecast:
    return ModelHorizonForecast(
        horizon=5,
        forecast_date="2026-03-09",
        point_forecast=67.0,
        quantiles={0.05: 60.0, 0.1: 62.0, 0.5: 67.0, 0.9: 72.0, 0.95: 74.0},
    )


def assessment(
    *,
    publishers: int,
    confidence: float,
    action: str = "moderate_up",
    support_count: int | None = None,
    primary: bool = True,
    novelty: str = "likely_new_relative_to_model_data",
) -> CfmContextAssessmentOutput:
    sources = [
        {
            "source_id": f"s{i}",
            "title": f"Source {i}",
            "source_url": f"https://publisher{i}.example/story",
            "publisher": f"Publisher {i}",
            "source_tier": "tier_1_primary" if i == 0 and primary else "tier_2_independent",
            "publication_date": "2026-03-01",
            "is_primary_or_official": i == 0 and primary,
            "provenance_status": "verified_from_tool",
            "verifier_content_status": "accepted_factual_content",
            "verifier_processing_status": "accepted_clean",
            "verified_evidence_excerpt": "Accepted factual evidence.",
        }
        for i in range(publishers)
    ]
    support_count = publishers if support_count is None else support_count
    return CfmContextAssessmentOutput.model_validate(
        {
            "reported_search_queries": ["q1", "q2", "q3", "q4"],
            "evidence_sources": sources,
            "evidence_claims": [
                {
                    "claim_id": "c1",
                    "statement": "A directly supported physical event occurred.",
                    "claim_type": "physical_supply",
                    "support_status": "direct",
                    "supporting_source_ids": [source["source_id"] for source in sources[:support_count]],
                    "contradicting_source_ids": [],
                }
            ],
            "physical_status": "partial_disruption",
            "incremental_novelty": novelty,
            "material_evidence_conflict": False,
            "confidence": confidence,
            "horizon_actions": [
                {
                    "horizon": 5,
                    "center_action": action,
                    "uncertainty_action": "substantially_wider",
                    "persistence_profile": "decaying",
                    "cited_claim_ids": ["c1"],
                    "rationale": "Direct evidence supports a bounded action.",
                }
            ],
            "research_summary": "Evidence record.",
            "overall_rationale": "Assessment only.",
        }
    )


def apply(value: CfmContextAssessmentOutput):
    return ConstrainedActionPolicy(CfmV33Settings()).apply(
        ensemble=ensemble(),
        assessment=value,
        horizon=5,
        horizons=[5],
        latest_price=67.0,
        cutoff_date="2026-03-02",
    )


def test_one_publisher_at_point_50_gets_limited_tier_and_small_cap() -> None:
    decision = apply(assessment(publishers=1, confidence=0.50))
    assert decision.evidence_tier == "limited"
    assert decision.evidence_tier_level == 1
    assert decision.center_action == "small_up"
    assert decision.applied_center_adjustment == pytest.approx(1.0)
    assert decision.uncertainty_action == "moderately_wider"
    assert decision.uncertainty_multiplier == pytest.approx(1.10)


def test_two_publishers_at_point_65_get_correlated_tier() -> None:
    decision = apply(assessment(publishers=2, confidence=0.65, support_count=1))
    assert decision.evidence_tier == "corroborated"
    assert decision.evidence_tier_level == 2
    assert decision.center_action == "moderate_up"
    assert decision.applied_center_adjustment == pytest.approx(2.0)
    assert decision.uncertainty_multiplier == pytest.approx(1.25)


def test_three_publishers_at_point_80_with_double_support_get_strong_tier() -> None:
    decision = apply(assessment(publishers=3, confidence=0.80, support_count=2))
    assert decision.evidence_tier == "strong"
    assert decision.evidence_tier_level == 3
    assert decision.center_action == "moderate_up"
    assert decision.applied_center_adjustment == pytest.approx(2.0)


def test_three_publishers_without_double_claim_support_remain_correlated() -> None:
    decision = apply(assessment(publishers=3, confidence=0.90, support_count=1))
    assert decision.evidence_tier == "corroborated"


def test_below_tier_one_confidence_is_rejected_and_neutralized() -> None:
    decision = apply(assessment(publishers=1, confidence=0.49))
    assert decision.eligible is False
    assert decision.evidence_tier == "none"
    assert decision.center_action == "no_change"
    assert decision.uncertainty_action == "unchanged"
    assert decision.applied_center_adjustment == 0.0


def test_partly_reflected_evidence_halves_center_adjustment() -> None:
    decision = apply(
        assessment(
            publishers=1,
            confidence=0.60,
            action="small_up",
            novelty="possibly_partly_reflected",
        )
    )
    assert decision.evidence_tier == "limited"
    assert decision.applied_center_adjustment == pytest.approx(0.5)


def test_material_conflict_blocks_non_neutral_action() -> None:
    raw = assessment(publishers=3, confidence=0.90).model_dump(mode="json")
    raw["material_evidence_conflict"] = True
    decision = apply(CfmContextAssessmentOutput.model_validate(raw))
    assert decision.eligible is False
    assert decision.final_point_forecast == ensemble().point_forecast


def test_reflected_or_indeterminate_novelty_blocks_center_action() -> None:
    for novelty in ("likely_reflected_in_model_data", "indeterminate"):
        decision = apply(assessment(publishers=3, confidence=0.90, novelty=novelty))
        assert decision.eligible is False
        assert decision.applied_center_adjustment == 0.0


def test_ensemble_lock_ignores_all_agent_actions() -> None:
    decision = EnsembleLockedPolicy(CfmV33Settings()).apply(
        ensemble=ensemble(),
        assessment=assessment(publishers=3, confidence=0.90),
        horizon=5,
        horizons=[5],
        latest_price=67.0,
        cutoff_date="2026-03-02",
    )
    assert decision.policy_id == "ensemble_locked_v1"
    assert decision.final_quantiles == ensemble().quantiles
    assert decision.evidence_tier == "none"
