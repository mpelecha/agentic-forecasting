"""Graduated evidence policy (specification Section 4.2)."""

from __future__ import annotations

import pytest
from energy_oil_forecasting.cfm_agent_v_6_0.config import DEFAULT_SETTINGS, NOVELTY_CLASSES, CfmV60Settings
from energy_oil_forecasting.cfm_agent_v_6_0.outputs import CfmContextAssessmentOutput
from energy_oil_forecasting.cfm_agent_v_6_0.policy import EnsembleLockedPolicy, EvidencePolicy
from energy_oil_forecasting.cfm_agent_v_6_0.policy.evidence_policy import CLAIM_CLAUSES
from energy_oil_forecasting.cfm_agent_v_6_0.schemas import (
    ActiveSource,
    EvidenceClaim,
    HorizonAction,
    SummaryPassage,
    VerifiedSummary,
)


HORIZON = 5


def claim(
    claim_id: str,
    summary_id: str,
    source_ids: list[str],
    *,
    passage_ids: list[str] | None = None,
    material: bool = True,
) -> EvidenceClaim:
    return EvidenceClaim(
        claim_id=claim_id,
        statement=f"statement {claim_id}",
        claim_type="physical_supply",
        supporting_summary_ids=[summary_id],
        supporting_passage_ids=["01.01"] if passage_ids is None else passage_ids,
        supporting_source_ids=source_ids,
        material_to_forecast=material,
    )


def assessment(
    packet,
    claims: list[EvidenceClaim],
    *,
    confidence: float = 0.8,
    center: str = "large_up",
    unc: str = "substantially_wider",
    novelty: str = "likely_new_relative_to_model_data",
    conflict: bool = False,
    cited: list[str] | None = None,
    narrowing_resolved: str = "",
) -> CfmContextAssessmentOutput:
    return CfmContextAssessmentOutput(
        research_packet_id=packet.packet_id,
        evidence_claims=claims,
        physical_status="confirmed_disruption",
        incremental_novelty=novelty,
        material_evidence_conflict=conflict,
        confidence=confidence,
        horizon_actions=[
            HorizonAction(
                horizon=HORIZON,
                center_action=center,
                uncertainty_action=unc,
                persistence_profile="temporary",
                cited_claim_ids=[item.claim_id for item in claims] if cited is None else cited,
                narrowing_uncertainty_resolved=narrowing_resolved,
                rationale="test",
            )
        ],
        research_summary="test",
        overall_rationale="test",
    )


def one_publisher(packet) -> list[EvidenceClaim]:
    return [claim("c1", packet.verified_summaries[0].verified_summary_id, ["source_001"])]


def two_publishers(packet) -> list[EvidenceClaim]:
    summary_id = packet.verified_summaries[0].verified_summary_id
    return [claim("c1", summary_id, ["source_001"]), claim("c2", summary_id, ["source_002"])]


def apply(packet, result, settings=DEFAULT_SETTINGS):
    return EvidencePolicy(settings).apply(result, packet, HORIZON)


# ------------------------------------------------------------- tier boundaries


@pytest.mark.parametrize(
    ("publishers", "confidence", "expected_level", "expected_name"),
    [
        (0, 0.99, 0, "none"),
        (1, 0.39, 0, "none"),
        (1, 0.40, 1, "limited"),
        (1, 0.60, 1, "limited"),
        (1, 0.80, 1, "limited"),
        (2, 0.59, 1, "limited"),
        (2, 0.60, 2, "corroborated"),
        (2, 0.79, 2, "corroborated"),
        (2, 0.80, 3, "strong"),
        (3, 0.80, 3, "strong"),
    ],
)
def test_tier_boundaries(packet, publishers, confidence, expected_level, expected_name):
    summary_id = packet.verified_summaries[0].verified_summary_id
    source_ids = ["source_001", "source_002", "source_003"][:publishers]
    claims = [claim(f"c{index}", summary_id, [source_id]) for index, source_id in enumerate(source_ids, 1)]
    decision = apply(packet, assessment(packet, claims, confidence=confidence, cited=[c.claim_id for c in claims]))
    assert decision.evidence_tier_level == expected_level
    assert decision.evidence_tier == expected_name
    assert decision.eligible is (expected_level > 0)
    assert set(decision.tier_evaluation) == {1, 2, 3}
    for tier, evaluation in decision.tier_evaluation.items():
        assert evaluation.publishers == publishers
        assert evaluation.confidence == confidence
        assert evaluation.min_publishers == getattr(DEFAULT_SETTINGS, f"tier_{tier}_min_publishers")
        assert evaluation.min_confidence == getattr(DEFAULT_SETTINGS, f"tier_{tier}_min_confidence")
        assert evaluation.passed is (tier <= expected_level)


def test_tier_thresholds_come_from_settings(packet):
    settings = CfmV60Settings(tier_3_min_publishers=3)
    decision = apply(packet, assessment(packet, two_publishers(packet), confidence=0.8), settings)
    assert decision.evidence_tier_level == 2
    assert decision.tier_evaluation[3].min_publishers == 3
    assert decision.tier_evaluation[3].passed is False


def test_publishers_are_the_union_across_valid_claims(packet):
    summary_id = packet.verified_summaries[0].verified_summary_id
    claims = [
        claim("c1", summary_id, ["source_001"]),
        claim("c2", summary_id, ["source_001", "source_002"]),
        claim("c3", summary_id, ["source_002"]),
    ]
    decision = apply(packet, assessment(packet, claims, confidence=0.8))
    assert decision.resolved_publishers == ["reuters", "u.s. energy information administration"]
    assert decision.resolved_source_ids == ["source_001", "source_002"]
    assert decision.qualifying_claim_ids == ["c1", "c2", "c3"]
    assert decision.evidence_tier_level == 3


def test_publisher_spelling_variants_count_once(packet):
    packet.sources[1] = packet.sources[1].model_copy(update={"publisher": "  U.S. Energy  Information Administration "})
    decision = apply(packet, assessment(packet, two_publishers(packet), confidence=0.8))
    assert decision.resolved_publishers == ["u.s. energy information administration"]
    assert decision.evidence_tier_level == 1


def test_rejected_claims_do_not_count_but_valid_ones_still_qualify(packet):
    summary_id = packet.verified_summaries[0].verified_summary_id
    claims = [claim("c1", summary_id, ["source_001"]), claim("c2", summary_id, ["source_002"], material=False)]
    decision = apply(packet, assessment(packet, claims, confidence=0.8))
    assert decision.qualifying_claim_ids == ["c1"]
    assert decision.claim_rejection_reasons == {"c2": ["material"]}
    assert decision.evidence_tier_level == 1
    assert decision.center_action == "small_up"


# --------------------------------------------------------------- conflict


def test_material_conflict_forces_tier_0_and_neutral_actions(packet):
    decision = apply(packet, assessment(packet, two_publishers(packet), confidence=0.9, conflict=True))
    assert decision.evidence_tier_level == 0
    assert decision.evidence_tier == "none"
    assert decision.eligible is False
    assert (decision.center_action, decision.uncertainty_action) == ("no_change", "unchanged")
    assert any("conflict" in reason.lower() for reason in decision.eligibility_reasons)
    assert decision.tier_evaluation[3].passed is True
    assert decision.qualifying_claim_ids == ["c1", "c2"]


# ---------------------------------------------------------- narrowing gate


def test_narrowing_requires_a_resolved_uncertainty(packet):
    result = assessment(packet, two_publishers(packet), confidence=0.8, unc="substantially_narrower")
    decision = apply(packet, result)
    assert decision.evidence_tier_level == 3
    assert decision.center_action == "large_up"
    assert decision.uncertainty_action == "unchanged"
    assert any("narrowing" in reason.lower() for reason in decision.eligibility_reasons)


def test_narrowing_allowed_when_resolved_and_capped_by_tier(packet):
    strong = assessment(
        packet,
        two_publishers(packet),
        confidence=0.8,
        unc="substantially_narrower",
        narrowing_resolved="OPEC meeting outcome now known",
    )
    assert apply(packet, strong).uncertainty_action == "substantially_narrower"
    limited = assessment(
        packet,
        one_publisher(packet),
        confidence=0.8,
        unc="substantially_narrower",
        narrowing_resolved="OPEC meeting outcome now known",
    )
    assert apply(packet, limited).uncertainty_action == "small_narrower"


def test_narrowing_gate_applies_at_tier_0(packet):
    result = assessment(packet, one_publisher(packet), confidence=0.2, unc="small_narrower")
    decision = apply(packet, result)
    assert decision.evidence_tier_level == 0
    assert decision.uncertainty_action == "unchanged"
    assert any("narrowing" in reason.lower() for reason in decision.eligibility_reasons)


# ---------------------------------------------------------- claim clauses


def test_failed_clause_names_are_recorded_in_clause_order(packet):
    bad = EvidenceClaim(
        claim_id="c1",
        statement="unsupported",
        claim_type="other",
        supporting_summary_ids=["verified_summary:sha256:missing"],
        supporting_passage_ids=[],
        supporting_source_ids=["source_001"],
        material_to_forecast=False,
    )
    decision = apply(packet, assessment(packet, [bad]))
    assert decision.claim_rejection_reasons == {
        "c1": [
            "material",
            "links_accepted_summary",
            "resolved_associated_source",
            "all_sources_associated",
            "passages_belong_to_linked_summary",
        ]
    }
    assert decision.evidence_tier_level == 0
    assert decision.qualifying_claim_ids == []
    summary_id = packet.verified_summaries[0].verified_summary_id
    no_sources = apply(packet, assessment(packet, [claim("c2", summary_id, [])]))
    covered = set(decision.claim_rejection_reasons["c1"]) | set(no_sources.claim_rejection_reasons["c2"])
    assert covered == set(CLAIM_CLAUSES)


def test_non_material_claim_is_rejected_by_name(packet):
    summary_id = packet.verified_summaries[0].verified_summary_id
    decision = apply(packet, assessment(packet, [claim("c1", summary_id, ["source_001"], material=False)]))
    assert decision.claim_rejection_reasons["c1"] == ["material"]


def test_claim_linking_a_failed_closed_summary_is_rejected(packet):
    packet.verified_summaries[0] = packet.verified_summaries[0].model_copy(update={"status": "failed_closed"})
    summary_id = packet.verified_summaries[0].verified_summary_id
    decision = apply(packet, assessment(packet, [claim("c1", summary_id, ["source_001"])]))
    assert "links_accepted_summary" in decision.claim_rejection_reasons["c1"]
    assert decision.evidence_tier_level == 0


def test_claim_without_sources_is_rejected(packet):
    summary_id = packet.verified_summaries[0].verified_summary_id
    decision = apply(packet, assessment(packet, [claim("c1", summary_id, [])]))
    assert decision.claim_rejection_reasons["c1"] == ["lists_supporting_source", "resolved_associated_source"]


def test_claim_listing_an_unassociated_source_is_rejected(packet):
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
    summary_id = packet.verified_summaries[0].verified_summary_id
    decision = apply(packet, assessment(packet, [claim("c1", summary_id, ["source_001", "source_999"])]))
    assert decision.claim_rejection_reasons["c1"] == ["all_sources_associated"]
    assert decision.evidence_tier_level == 0


def test_claim_with_only_an_unresolved_source_is_rejected(packet):
    packet.sources[0] = packet.sources[0].model_copy(
        update={"resolved_url": None, "resolved_domain": None, "publisher": None, "resolution_status": "unresolved"}
    )
    summary_id = packet.verified_summaries[0].verified_summary_id
    decision = apply(packet, assessment(packet, [claim("c1", summary_id, ["source_001"])]))
    assert decision.claim_rejection_reasons["c1"] == ["resolved_associated_source"]
    assert decision.resolved_source_ids == []


def test_passage_of_an_unlinked_summary_is_rejected(packet):
    other = VerifiedSummary(
        verified_summary_id="verified_summary:sha256:other",
        query_area="market_reaction",
        query="other",
        cutoff_date=packet.cutoff_date,
        status="accepted",
        cleaned_summary="Prices rose after the announcement was made to the market.",
        passages=[
            SummaryPassage(passage_id="02.01", text="Prices rose after the announcement was made to the market.")
        ],
        associated_source_ids=["source_003"],
    )
    packet.verified_summaries.append(other)
    summary_id = packet.verified_summaries[0].verified_summary_id
    decision = apply(packet, assessment(packet, [claim("c1", summary_id, ["source_001"], passage_ids=["02.01"])]))
    assert decision.claim_rejection_reasons["c1"] == ["passages_belong_to_linked_summary"]
    mixed = claim("c1", summary_id, ["source_001"], passage_ids=["01.01", "02.01"])
    assert apply(packet, assessment(packet, [mixed])).claim_rejection_reasons["c1"] == [
        "passages_belong_to_linked_summary"
    ]


def test_material_claim_without_passages_is_rejected(packet):
    summary_id = packet.verified_summaries[0].verified_summary_id
    decision = apply(packet, assessment(packet, [claim("c1", summary_id, ["source_001"], passage_ids=[])]))
    assert decision.claim_rejection_reasons["c1"] == ["passages_belong_to_linked_summary"]
    assert decision.evidence_tier_level == 0


def test_valid_claim_has_no_rejection_entry(packet):
    decision = apply(packet, assessment(packet, one_publisher(packet)))
    assert decision.claim_rejection_reasons == {}
    assert decision.qualifying_claim_ids == ["c1"]


def test_non_neutral_action_without_cited_claims_is_tier_0(packet):
    decision = apply(packet, assessment(packet, one_publisher(packet), cited=[]))
    assert decision.evidence_tier_level == 0
    assert (decision.center_action, decision.uncertainty_action) == ("no_change", "unchanged")


# ---------------------------------------------------------------- cap ladder


@pytest.mark.parametrize(
    ("tier_claims", "confidence", "center", "unc", "expected_center", "expected_unc"),
    [
        (one_publisher, 0.8, "large_up", "substantially_wider", "small_up", "small_wider"),
        (one_publisher, 0.8, "moderate_down", "moderately_wider", "small_down", "small_wider"),
        (two_publishers, 0.7, "large_down", "substantially_wider", "moderate_down", "moderately_wider"),
        (two_publishers, 0.7, "small_up", "small_wider", "small_up", "small_wider"),
        (two_publishers, 0.8, "large_up", "substantially_wider", "large_up", "substantially_wider"),
        (two_publishers, 0.8, "small_down", "unchanged", "small_down", "unchanged"),
        (one_publisher, 0.3, "large_up", "substantially_wider", "no_change", "unchanged"),
        (one_publisher, 0.3, "small_down", "small_wider", "no_change", "unchanged"),
    ],
)
def test_cap_ladder_for_both_levers(packet, tier_claims, confidence, center, unc, expected_center, expected_unc):
    result = assessment(packet, tier_claims(packet), confidence=confidence, center=center, unc=unc)
    decision = apply(packet, result)
    assert decision.proposed_center_action == center
    assert decision.proposed_uncertainty_action == unc
    assert decision.center_action == expected_center
    assert decision.uncertainty_action == expected_unc


def test_capping_is_recorded_in_the_reasons(packet):
    decision = apply(packet, assessment(packet, one_publisher(packet), confidence=0.8))
    assert any("large_up" in reason and "small_up" in reason for reason in decision.eligibility_reasons)
    assert any("substantially_wider" in reason and "small_wider" in reason for reason in decision.eligibility_reasons)


# --------------------------------------------------------------- neutral


def test_neutral_proposal_needs_no_evidence(packet):
    result = assessment(packet, [], center="no_change", unc="unchanged", confidence=0.1, cited=[])
    decision = apply(packet, result)
    assert decision.eligible is True
    assert decision.evidence_tier == "none"
    assert decision.evidence_tier_level == 0
    assert (decision.center_action, decision.uncertainty_action) == ("no_change", "unchanged")
    assert set(decision.tier_evaluation) == {1, 2, 3}
    assert all(not evaluation.passed for evaluation in decision.tier_evaluation.values())
    assert decision.claim_rejection_reasons == {}


def test_neutral_proposal_with_conflict_stays_eligible(packet):
    result = assessment(packet, [], center="no_change", unc="unchanged", conflict=True, cited=[])
    decision = apply(packet, result)
    assert decision.eligible is True
    assert (decision.center_action, decision.uncertainty_action) == ("no_change", "unchanged")


# ---------------------------------------------------------------- novelty


@pytest.mark.parametrize("novelty", NOVELTY_CLASSES)
def test_novelty_never_changes_the_tier_or_the_actions(packet, novelty):
    decision = apply(packet, assessment(packet, two_publishers(packet), confidence=0.8, novelty=novelty))
    assert decision.evidence_tier_level == 3
    assert (decision.center_action, decision.uncertainty_action) == ("large_up", "substantially_wider")


# ------------------------------------------------------------- decision shape


def test_decision_records_policy_id_and_proposals(packet):
    decision = apply(packet, assessment(packet, two_publishers(packet), confidence=0.8))
    assert decision.policy_id == "graduated_evidence_policy_v60"
    assert decision.horizon == HORIZON
    assert decision.proposed_center_action == "large_up"
    assert decision.proposed_uncertainty_action == "substantially_wider"
    assert decision.model_dump(mode="json")["tier_evaluation"].keys() == {"1", "2", "3"}


def test_ensemble_locked_policy_is_always_neutral(packet):
    decision = EnsembleLockedPolicy(DEFAULT_SETTINGS).apply(
        assessment(packet, two_publishers(packet), confidence=0.9), packet, HORIZON
    )
    assert decision.policy_id == "ensemble_locked_v60"
    assert decision.eligible is False
    assert decision.evidence_tier_level == 0
    assert (decision.center_action, decision.uncertainty_action) == ("no_change", "unchanged")
    assert decision.proposed_center_action == "large_up"
