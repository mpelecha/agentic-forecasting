from energy_oil_forecasting.cfm_agent_v_5_0.config import DEFAULT_SETTINGS
from energy_oil_forecasting.cfm_agent_v_5_0.forecast_engine import PythonForecastEngine
from energy_oil_forecasting.cfm_agent_v_5_0.outputs import CfmContextAssessmentOutput
from energy_oil_forecasting.cfm_agent_v_5_0.policy import EvidencePolicy
from energy_oil_forecasting.cfm_agent_v_5_0.schemas import (
    ActiveSource,
    EvidenceClaim,
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
):
    summary = packet.verified_summaries[0]
    selected = list(summary.associated_source_ids) if source_ids is None else list(source_ids)
    claim = EvidenceClaim(
        claim_id="c1",
        statement="Confirmed physical oil supply information",
        claim_type="physical_supply",
        supporting_summary_ids=[summary.verified_summary_id],
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


def test_tier_3_unlocks_large_for_full_source_set(packet):
    decision = EvidencePolicy(DEFAULT_SETTINGS).apply(assessment(packet), packet, 5)
    assert decision.evidence_tier == "strong"
    assert decision.center_action == "large_up"
    assert decision.resolved_source_ids == ["source_001", "source_002", "source_003"]


def test_valid_source_subset_passes_and_counts_only_selected_publishers(packet):
    result = assessment(
        packet,
        confidence=0.8,
        center="large_up",
        source_ids=["source_001", "source_002"],
    )
    decision = EvidencePolicy(DEFAULT_SETTINGS).apply(result, packet, 5)
    assert decision.evidence_tier == "corroborated"
    assert decision.center_action == "moderate_up"
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
    assert decision.resolved_source_ids == ["source_001"]


def test_empty_source_list_fails(packet):
    decision = EvidencePolicy(DEFAULT_SETTINGS).apply(assessment(packet, source_ids=[]), packet, 5)
    assert decision.evidence_tier == "none"
    assert decision.center_action == "no_change"


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


def test_neutral_exact_reproduction(packet):
    result = assessment(packet, center="no_change", unc="unchanged")
    decision = EvidencePolicy(DEFAULT_SETTINGS).apply(result, packet, 5)
    ensemble = ModelHorizonForecast(
        horizon=5,
        forecast_date="2026-08-04",
        point_forecast=80,
        quantiles={0.1: 70, 0.5: 80, 0.9: 90},
    )
    transformed = PythonForecastEngine(DEFAULT_SETTINGS).transform(ensemble, decision, result.incremental_novelty, 80)
    assert transformed.final_quantiles == ensemble.quantiles


def test_large_is_thirty_percent_width(packet):
    result = assessment(packet)
    decision = EvidencePolicy(DEFAULT_SETTINGS).apply(result, packet, 5)
    ensemble = ModelHorizonForecast(
        horizon=5,
        forecast_date="2026-08-04",
        point_forecast=80,
        quantiles={0.1: 70, 0.5: 80, 0.9: 90},
    )
    transformed = PythonForecastEngine(DEFAULT_SETTINGS).transform(ensemble, decision, result.incremental_novelty, 80)
    assert transformed.raw_center_adjustment == 6.0
