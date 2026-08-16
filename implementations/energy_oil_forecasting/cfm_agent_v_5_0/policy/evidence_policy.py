"""Locked Component #9 evidence policy for CFM Agent v5.0."""

from energy_oil_forecasting.cfm_agent_v_5_0.config import CfmV50Settings
from energy_oil_forecasting.cfm_agent_v_5_0.outputs import CfmContextAssessmentOutput
from energy_oil_forecasting.cfm_agent_v_5_0.schemas import PolicyDecision, ResearchPacket


PHYSICAL = {
    "physical_supply",
    "shipping",
    "production_policy",
    "strategic_reserves",
    "inventory",
}
CENTER_LEVEL = {
    "no_change": 0,
    "small_up": 1,
    "small_down": 1,
    "moderate_up": 2,
    "moderate_down": 2,
    "large_up": 3,
    "large_down": 3,
}
UNC_LEVEL = {
    "unchanged": 0,
    "small_wider": 1,
    "small_narrower": 1,
    "moderately_wider": 2,
    "moderately_narrower": 2,
    "substantially_wider": 3,
    "substantially_narrower": 3,
}


class EvidencePolicy:
    policy_id = "graduated_evidence_policy_v50"

    def __init__(self, settings: CfmV50Settings):
        self.s = settings

    def apply(  # noqa: PLR0912, PLR0915
        self,
        assessment: CfmContextAssessmentOutput,
        packet: ResearchPacket,
        horizon: int,
    ) -> PolicyDecision:  # noqa: PLR0912, PLR0915
        action = assessment.action_for(horizon)
        if action.center_action == "no_change" and action.uncertainty_action == "unchanged":
            return PolicyDecision(
                policy_id=self.policy_id,
                horizon=horizon,
                eligible=True,
                evidence_tier="none",
                evidence_tier_level=0,
                eligibility_reasons=["Neutral proposal requires no evidence."],
                center_action="no_change",
                uncertainty_action="unchanged",
            )

        source_by = {source.source_id: source for source in packet.sources}
        summary_by = {summary.verified_summary_id: summary for summary in packet.verified_summaries}
        claim_by = {claim.claim_id: claim for claim in assessment.evidence_claims}
        cited = [claim_by[claim_id] for claim_id in action.cited_claim_ids if claim_id in claim_by]
        valid = []
        reasons: list[str] = []

        for claim in cited:
            linked_summaries = [
                summary_by[summary_id]
                for summary_id in claim.supporting_summary_ids
                if summary_id in summary_by and summary_by[summary_id].status == "accepted"
            ]
            associated_ids = {source_id for summary in linked_summaries for source_id in summary.associated_source_ids}
            claimed_source_ids = set(claim.supporting_source_ids)
            selected_resolved_ids = {
                source_id
                for source_id in claimed_source_ids
                if (source_id in associated_ids and source_id in source_by and source_by[source_id].resolved_domain)
            }
            no_unassociated_listed = claimed_source_ids <= associated_ids

            if (
                claim.material_to_forecast
                and linked_summaries
                and claimed_source_ids
                and selected_resolved_ids
                and no_unassociated_listed
            ):
                valid.append(claim)

        if not cited or len(valid) != len(cited):
            reasons.append(
                "Every cited material claim must link to an accepted verified "
                "summary and cite at least one mechanically resolved source "
                "associated with that summary."
            )
        if assessment.material_evidence_conflict:
            reasons.append("Material evidence conflict declared.")
        if assessment.incremental_novelty not in {
            "likely_new_relative_to_model_data",
            "possibly_partly_reflected",
        }:
            reasons.append("Novelty requirement failed.")

        publishers = {}
        source_ids_by_claim: dict[str, set[str]] = {}
        for claim in valid:
            linked_summaries = [summary_by[summary_id] for summary_id in claim.supporting_summary_ids]
            associated_ids = {source_id for summary in linked_summaries for source_id in summary.associated_source_ids}
            claimed_source_ids = set(claim.supporting_source_ids)
            selected_resolved_ids = {
                source_id
                for source_id in claimed_source_ids
                if (source_id in associated_ids and source_id in source_by and source_by[source_id].resolved_domain)
            }
            source_ids_by_claim[claim.claim_id] = selected_resolved_ids
            for source_id in selected_resolved_ids:
                source = source_by[source_id]
                publisher = (source.publisher or source.resolved_domain).strip().lower()
                publishers[publisher] = source

        tier = 0
        tier_name = "none"
        if (
            not reasons
            and len(publishers) >= self.s.tier_1_min_publishers
            and assessment.confidence >= self.s.tier_1_min_confidence
        ):
            tier, tier_name = 1, "limited"
        if (
            not reasons
            and len(publishers) >= self.s.tier_2_min_publishers
            and assessment.confidence >= self.s.tier_2_min_confidence
        ):
            tier, tier_name = 2, "corroborated"

        per_claim_publishers = []
        for claim in valid:
            per_claim_publishers.append(
                len(
                    {
                        (source_by[source_id].publisher or source_by[source_id].resolved_domain).strip().lower()
                        for source_id in source_ids_by_claim[claim.claim_id]
                    }
                )
            )
        primary = any(source.source_quality == "tier_1_primary" for source in publishers.values())
        physical = any(claim.claim_type in PHYSICAL for claim in valid)
        if (
            not reasons
            and len(publishers) >= self.s.tier_3_min_publishers
            and assessment.confidence >= self.s.tier_3_min_confidence
            and per_claim_publishers
            and min(per_claim_publishers) >= self.s.tier_3_min_publishers_per_claim
            and primary
            and physical
            and assessment.incremental_novelty == "likely_new_relative_to_model_data"
        ):
            tier, tier_name = 3, "strong"

        if tier == 0 and not reasons:
            reasons.append("Publisher/confidence thresholds did not reach Tier 1.")
        if tier > 0:
            reasons.append(f"Evidence Tier {tier} ({tier_name}) passed.")

        max_level = tier
        center = action.center_action
        if CENTER_LEVEL[center] > max_level:
            direction = "up" if center.endswith("up") else "down"
            center = (
                {
                    1: f"small_{direction}",
                    2: f"moderate_{direction}",
                    3: f"large_{direction}",
                }[max_level]
                if max_level
                else "no_change"
            )

        uncertainty = action.uncertainty_action
        narrowing = uncertainty.endswith("narrower")
        if narrowing and not action.narrowing_uncertainty_resolved.strip():
            uncertainty = "unchanged"
            reasons.append("Narrowing rejected because no specific resolved uncertainty was identified.")
        elif UNC_LEVEL[uncertainty] > max_level:
            wider = uncertainty.endswith("wider")
            uncertainty = (
                {
                    1: "small_",
                    2: "moderately_",
                    3: "substantially_",
                }[max_level]
                + ("wider" if wider else "narrower")
                if max_level
                else "unchanged"
            )
        if tier == 0:
            center = "no_change"
            uncertainty = "unchanged"

        resolved_source_ids = sorted({source_id for ids in source_ids_by_claim.values() for source_id in ids})
        return PolicyDecision(
            policy_id=self.policy_id,
            horizon=horizon,
            eligible=tier > 0,
            evidence_tier=tier_name,
            evidence_tier_level=tier,
            resolved_source_ids=resolved_source_ids,
            resolved_publishers=sorted(publishers),
            qualifying_claim_ids=sorted(claim.claim_id for claim in valid),
            eligibility_reasons=reasons,
            center_action=center,
            uncertainty_action=uncertainty,
        )


class EnsembleLockedPolicy(EvidencePolicy):
    policy_id = "ensemble_locked_v5"

    def apply(self, assessment, packet, horizon):  # noqa: ANN001, ANN201
        return PolicyDecision(
            policy_id=self.policy_id,
            horizon=horizon,
            eligible=False,
            evidence_tier="none",
            evidence_tier_level=0,
            eligibility_reasons=["Ensemble-locked control."],
            center_action="no_change",
            uncertainty_action="unchanged",
        )
