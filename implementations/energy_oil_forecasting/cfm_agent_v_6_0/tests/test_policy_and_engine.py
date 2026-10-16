"""Policy tiers on the shared packet, and the policy-to-engine hand-off."""

from __future__ import annotations

import pytest
from energy_oil_forecasting.cfm_agent_v_6_0.config import DEFAULT_SETTINGS
from energy_oil_forecasting.cfm_agent_v_6_0.forecast_engine import PythonForecastEngine
from energy_oil_forecasting.cfm_agent_v_6_0.outputs import CfmContextAssessmentOutput
from energy_oil_forecasting.cfm_agent_v_6_0.policy import EvidencePolicy
from energy_oil_forecasting.cfm_agent_v_6_0.schemas import (
    ActiveSource,
    EvidenceClaim,
    HistoricalReturnProfile,
    HorizonAction,
    ModelHorizonForecast,
)


def assessment(  # noqa: PLR0917
    packet,
    *,
    confidence=0.8,
    center="large_up",
    unc="substantially_wider",
    novelty="likely_new_relative_to_model_data",
    source_ids=None,
    passage_ids=None,
):
    summary = packet.verified_summaries[0]
    selected = list(summary.associated_source_ids) if source_ids is None else list(source_ids)
    claim = EvidenceClaim(
        claim_id="c1",
        statement="Confirmed physical oil supply information",
        claim_type="physical_supply",
        supporting_summary_ids=[summary.verified_summary_id],
        supporting_passage_ids=["01.01"] if passage_ids is None else list(passage_ids),
        supporting_source_ids=selected,
        material_to_forecast=True,
    )
    return CfmContextAssessmentOutput(
        research_packet_id=packet.packet_id,
        evidence_claims=[claim],
        physical_status="confirmed_disruption",
        incremental_novelty=novelty,
        material_evidence_conflict=False,
        confidence=confidence,
        horizon_actions=[
            HorizonAction(
                horizon=5,
                center_action=center,
                uncertainty_action=unc,
                persistence_profile="temporary",
                cited_claim_ids=["c1"],
                rationale="test",
            )
        ],
        research_summary="test",
        overall_rationale="test",
    )


def profile() -> HistoricalReturnProfile:
    return HistoricalReturnProfile(
        horizon=5,
        n_eligible=2_500,
        min_required=DEFAULT_SETTINGS.min_historical_returns,
        eligible=True,
        median=0.0,
        return_percentiles={10: -0.06, 25: -0.03, 40: -0.01, 60: 0.015, 75: 0.03, 90: 0.06},
        absolute_deviation_bands={65: 0.03, 70: 0.034, 75: 0.038, 80: 0.045, 85: 0.052, 90: 0.062, 95: 0.08},
    )


def ensemble() -> ModelHorizonForecast:
    return ModelHorizonForecast(
        horizon=5,
        forecast_date="2026-08-04",
        point_forecast=80,
        quantiles={0.1: 70, 0.5: 80, 0.9: 90},
    )


def test_tier_3_unlocks_large_for_full_source_set(packet):
    decision = EvidencePolicy(DEFAULT_SETTINGS).apply(assessment(packet), packet, 5)
    assert decision.evidence_tier == "strong"
    assert decision.evidence_tier_level == 3
    assert decision.center_action == "large_up"
    assert decision.uncertainty_action == "substantially_wider"
    assert decision.resolved_source_ids == ["source_001", "source_002", "source_003"]
    assert decision.claim_rejection_reasons == {}


def test_two_publishers_at_high_confidence_reach_tier_3(packet):
    result = assessment(packet, confidence=0.8, source_ids=["source_001", "source_002"])
    decision = EvidencePolicy(DEFAULT_SETTINGS).apply(result, packet, 5)
    assert decision.evidence_tier == "strong"
    assert decision.center_action == "large_up"


def test_valid_source_subset_passes_and_counts_only_selected_publishers(packet):
    result = assessment(
        packet,
        confidence=0.7,
        center="large_up",
        source_ids=["source_001", "source_002"],
    )
    decision = EvidencePolicy(DEFAULT_SETTINGS).apply(result, packet, 5)
    assert decision.evidence_tier == "corroborated"
    assert decision.center_action == "moderate_up"
    assert decision.uncertainty_action == "moderately_wider"
    assert decision.resolved_source_ids == ["source_001", "source_002"]
    assert decision.resolved_publishers == [
        "reuters",
        "u.s. energy information administration",
    ]
    assert "bloomberg" not in decision.resolved_publishers


def test_single_selected_resolved_source_reaches_only_tier_1(packet):
    result = assessment(packet, source_ids=["source_001"])
    decision = EvidencePolicy(DEFAULT_SETTINGS).apply(result, packet, 5)
    assert decision.evidence_tier == "limited"
    assert decision.center_action == "small_up"
    assert decision.uncertainty_action == "small_wider"
    assert decision.resolved_source_ids == ["source_001"]


def test_empty_source_list_fails(packet):
    decision = EvidencePolicy(DEFAULT_SETTINGS).apply(assessment(packet, source_ids=[]), packet, 5)
    assert decision.evidence_tier == "none"
    assert decision.center_action == "no_change"
    assert decision.uncertainty_action == "unchanged"
    assert "lists_supporting_source" in decision.claim_rejection_reasons["c1"]


def test_unassociated_added_source_fails(packet):
    packet.sources.append(
        ActiveSource(
            source_id="source_999",
            rank=4,
            grounding_url="https://redirect.test/999",
            resolved_url="https://www.ft.com/a",
            resolved_domain="ft.com",
            publisher="Financial Times",
            source_quality="tier_2_independent",
            resolution_status="resolved",
        )
    )
    decision = EvidencePolicy(DEFAULT_SETTINGS).apply(
        assessment(packet, source_ids=["source_001", "source_999"]), packet, 5
    )
    assert decision.evidence_tier == "none"
    assert decision.center_action == "no_change"
    assert decision.claim_rejection_reasons["c1"] == ["all_sources_associated"]


def test_selected_unresolved_source_does_not_qualify(packet):
    packet.sources[0] = packet.sources[0].model_copy(
        update={
            "resolved_url": None,
            "resolved_domain": None,
            "publisher": None,
            "resolution_status": "unresolved",
        }
    )
    decision = EvidencePolicy(DEFAULT_SETTINGS).apply(assessment(packet, source_ids=["source_001"]), packet, 5)
    assert decision.evidence_tier == "none"
    assert decision.resolved_source_ids == []
    assert decision.claim_rejection_reasons["c1"] == ["resolved_associated_source"]


def test_claim_without_passages_fails(packet):
    decision = EvidencePolicy(DEFAULT_SETTINGS).apply(assessment(packet, passage_ids=[]), packet, 5)
    assert decision.evidence_tier == "none"
    assert decision.claim_rejection_reasons["c1"] == ["passages_belong_to_linked_summary"]


def test_neutral_exact_reproduction(packet):
    result = assessment(packet, center="no_change", unc="unchanged")
    decision = EvidencePolicy(DEFAULT_SETTINGS).apply(result, packet, 5)
    assert decision.eligible is True
    base = ensemble()
    transformed = PythonForecastEngine(DEFAULT_SETTINGS).transform(
        base, decision, result.incremental_novelty, 80.0, profile()
    )
    assert transformed.neutral is True
    assert transformed.final_quantiles == base.quantiles
    assert transformed.final_point_forecast == base.point_forecast
    assert transformed.applied_center_adjustment == 0.0


def test_large_up_moves_p50_by_the_historical_percentile_times_novelty(packet):
    result = assessment(packet)
    decision = EvidencePolicy(DEFAULT_SETTINGS).apply(result, packet, 5)
    assert decision.center_action == "large_up"
    base = ensemble()
    multiplier = DEFAULT_SETTINGS.novelty_multipliers[result.incremental_novelty]
    transformed = PythonForecastEngine(DEFAULT_SETTINGS).transform(
        base, decision, result.incremental_novelty, 80.0, profile()
    )
    expected = 80 * 0.06 * multiplier
    assert transformed.neutral is False
    assert transformed.selected_center_percentile == DEFAULT_SETTINGS.center_percentiles["large_up"]
    assert transformed.novelty_multiplier == multiplier
    assert transformed.applied_center_adjustment == pytest.approx(expected)
    assert transformed.adjusted_p50 == pytest.approx(80 + expected)
