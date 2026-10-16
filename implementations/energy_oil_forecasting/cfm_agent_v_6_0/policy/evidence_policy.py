"""Graduated evidence policy for CFM Agent v6.0 (specification Section 4.2).

The policy is the only place where the LLM's proposed actions are translated
into the actions the transformation engine applies. It validates every cited
claim against six mechanical clauses, counts the distinct publishers behind the
valid claims, evaluates the three evidence tiers, and caps both levers at the
qualified tier. It never upgrades a proposal, never reads the audit-only
outputs, and does not consult the novelty assessment.

Every threshold comes from settings; the tier numbers, action ladders, and tier
names are structural enumerations.
"""

from __future__ import annotations

from energy_oil_forecasting.cfm_agent_v_6_0.config import CfmV60Settings
from energy_oil_forecasting.cfm_agent_v_6_0.outputs import CfmContextAssessmentOutput
from energy_oil_forecasting.cfm_agent_v_6_0.passages import passage_ids_for
from energy_oil_forecasting.cfm_agent_v_6_0.schemas import (
    ActiveSource,
    CenterAction,
    EvidenceClaim,
    EvidenceTier,
    PolicyDecision,
    ResearchPacket,
    TierEvaluation,
    UncertaintyAction,
    VerifiedSummary,
)
from energy_oil_forecasting.cfm_agent_v_6_0.source_registry import normalize_publisher


#: The graduated tiers in ascending order (Section 4.2); Tier 0 is the absence of a tier.
TIERS: tuple[int, ...] = (1, 2, 3)  # structural: the three evidence tiers
TIER_NAMES: dict[int, EvidenceTier] = {
    0: "none",
    1: "limited",
    2: "corroborated",
    3: "strong",  # structural: tier 3 name
}

#: Action ladders indexed by level: position 0 is neutral, position N is the largest action Tier N allows.
CENTER_LADDER: dict[str, tuple[CenterAction, ...]] = {
    "up": ("no_change", "small_up", "moderate_up", "large_up"),
    "down": ("no_change", "small_down", "moderate_down", "large_down"),
}
UNCERTAINTY_LADDER: dict[str, tuple[UncertaintyAction, ...]] = {
    "wider": ("unchanged", "small_wider", "moderately_wider", "substantially_wider"),
    "narrower": ("unchanged", "small_narrower", "moderately_narrower", "substantially_narrower"),
}

#: The six per-claim validity clauses, in evaluation order (Section 4.2).
CLAIM_CLAUSES: tuple[str, ...] = (
    "material",
    "links_accepted_summary",
    "lists_supporting_source",
    "resolved_associated_source",
    "all_sources_associated",
    "passages_belong_to_linked_summary",
)
UNKNOWN_CLAIM_CLAUSE = "unknown_claim_id"


def _direction(action: str) -> str:
    """Return the ladder a non-neutral action belongs to (``up``/``down`` or ``wider``/``narrower``)."""
    return action.rsplit("_", 1)[-1]


def _level(action: str, ladders: dict[str, tuple[str, ...]]) -> int:
    """Return the rung of ``action`` on its ladder; neutral actions are level 0."""
    for ladder in ladders.values():
        if action in ladder:
            return ladder.index(action)
    raise ValueError(f"unknown action {action!r}")


def _cap(action: str, ladders: dict[str, tuple[str, ...]], max_level: int) -> str:
    """Step ``action`` down to the largest rung its ladder allows at ``max_level``."""
    level = _level(action, ladders)
    if level <= max_level:
        return action
    if max_level == 0:
        return ladders[next(iter(ladders))][0]
    return ladders[_direction(action)][max_level]


class ClaimCheck:
    """The mechanical validity of one cited claim against the packet."""

    def __init__(
        self,
        claim: EvidenceClaim,
        summary_by: dict[str, VerifiedSummary],
        source_by: dict[str, ActiveSource],
    ) -> None:
        self.claim = claim
        self.linked_summaries = [
            summary_by[summary_id]
            for summary_id in claim.supporting_summary_ids
            if summary_id in summary_by and summary_by[summary_id].status == "accepted"
        ]
        associated_ids = {source_id for summary in self.linked_summaries for source_id in summary.associated_source_ids}
        linked_passage_ids = set().union(*(passage_ids_for(summary) for summary in self.linked_summaries))
        listed_ids = list(dict.fromkeys(claim.supporting_source_ids))
        self.resolved_source_ids = [
            source_id
            for source_id in listed_ids
            if source_id in associated_ids
            and source_id in source_by
            and source_by[source_id].resolution_status == "resolved"
        ]
        self.failed_clauses = [
            name
            for name, passed in (
                ("material", claim.material_to_forecast),
                ("links_accepted_summary", bool(self.linked_summaries)),
                ("lists_supporting_source", bool(listed_ids)),
                ("resolved_associated_source", bool(self.resolved_source_ids)),
                ("all_sources_associated", set(listed_ids) <= associated_ids),
                (
                    "passages_belong_to_linked_summary",
                    bool(claim.supporting_passage_ids) and set(claim.supporting_passage_ids) <= linked_passage_ids,
                ),
            )
            if not passed
        ]

    @property
    def valid(self) -> bool:
        return not self.failed_clauses


class EvidencePolicy:
    """Validate claims, evaluate the evidence tiers, and cap the proposed actions."""

    policy_id = "graduated_evidence_policy_v60"

    def __init__(self, settings: CfmV60Settings):
        self.s = settings

    def _tier_threshold(self, tier: int) -> tuple[int, float]:
        return (
            getattr(self.s, f"tier_{tier}_min_publishers"),
            getattr(self.s, f"tier_{tier}_min_confidence"),
        )

    def _evaluate_tiers(self, publisher_count: int, confidence: float) -> dict[int, TierEvaluation]:
        evaluation: dict[int, TierEvaluation] = {}
        for tier in TIERS:
            min_publishers, min_confidence = self._tier_threshold(tier)
            evaluation[tier] = TierEvaluation(
                min_publishers=min_publishers,
                min_confidence=min_confidence,
                publishers=publisher_count,
                confidence=confidence,
                passed=publisher_count >= min_publishers and confidence >= min_confidence,
            )
        return evaluation

    @staticmethod
    def _qualified_tier(evaluation: dict[int, TierEvaluation]) -> int:
        """Return the highest tier passed with every lower tier also passed (tiers are cumulative)."""
        qualified = 0
        for tier in TIERS:
            if not evaluation[tier].passed:
                break
            qualified = tier
        return qualified

    def apply(  # noqa: PLR0912, PLR0914
        self,
        assessment: CfmContextAssessmentOutput,
        packet: ResearchPacket,
        horizon: int,
    ) -> PolicyDecision:
        action = assessment.action_for(horizon)
        proposed_center = action.center_action
        proposed_uncertainty = action.uncertainty_action
        neutral_proposal = proposed_center == "no_change" and proposed_uncertainty == "unchanged"

        source_by = {source.source_id: source for source in packet.sources}
        summary_by = {summary.verified_summary_id: summary for summary in packet.verified_summaries}
        claim_by = {claim.claim_id: claim for claim in assessment.evidence_claims}

        rejections: dict[str, list[str]] = {}
        checks: list[ClaimCheck] = []
        for claim_id in dict.fromkeys(action.cited_claim_ids):
            if claim_id not in claim_by:
                rejections[claim_id] = [UNKNOWN_CLAIM_CLAUSE]
                continue
            check = ClaimCheck(claim_by[claim_id], summary_by, source_by)
            if check.valid:
                checks.append(check)
            else:
                rejections[claim_id] = list(check.failed_clauses)

        resolved_source_ids = sorted({source_id for check in checks for source_id in check.resolved_source_ids})
        publishers = {
            key
            for source_id in resolved_source_ids
            if (key := normalize_publisher(source_by[source_id].publisher, source_by[source_id].resolved_domain))
        }
        evaluation = self._evaluate_tiers(len(publishers), assessment.confidence)

        reasons: list[str] = []
        if neutral_proposal:
            tier = 0
            reasons.append("Neutral proposal requires no evidence.")
        elif assessment.material_evidence_conflict:
            tier = 0
            reasons.append("Material evidence conflict declared; Tier 0 forces neutral actions.")
        else:
            tier = self._qualified_tier(evaluation)
            if not action.cited_claim_ids:
                reasons.append("Non-neutral proposal cites no claims.")
            elif not checks:
                reasons.append("No cited claim passed all six validity clauses.")
            elif rejections:
                reasons.append(f"Rejected claims excluded from the publisher count: {sorted(rejections)}.")
            if tier == 0:
                reasons.append("Publisher and confidence thresholds did not reach Tier 1.")
            else:
                reasons.append(f"Evidence Tier {tier} ({TIER_NAMES[tier]}) passed.")

        gated_uncertainty = proposed_uncertainty
        narrowing = (
            _level(proposed_uncertainty, UNCERTAINTY_LADDER) > 0 and _direction(proposed_uncertainty) == "narrower"
        )
        if narrowing and not action.narrowing_uncertainty_resolved.strip():
            gated_uncertainty = "unchanged"
            reasons.append("Narrowing rejected because no resolved uncertainty was identified.")
        center = _cap(proposed_center, CENTER_LADDER, tier)
        uncertainty = _cap(gated_uncertainty, UNCERTAINTY_LADDER, tier)
        if center != proposed_center:
            reasons.append(f"Centre action capped from {proposed_center} to {center} at Tier {tier}.")
        if uncertainty != gated_uncertainty:
            reasons.append(f"Uncertainty action capped from {gated_uncertainty} to {uncertainty} at Tier {tier}.")

        return PolicyDecision(
            policy_id=self.policy_id,
            horizon=horizon,
            eligible=neutral_proposal or tier > 0,
            evidence_tier=TIER_NAMES[tier],
            evidence_tier_level=tier,
            resolved_source_ids=resolved_source_ids,
            resolved_publishers=sorted(publishers),
            qualifying_claim_ids=sorted(check.claim.claim_id for check in checks),
            claim_rejection_reasons=rejections,
            tier_evaluation=evaluation,
            eligibility_reasons=reasons,
            proposed_center_action=proposed_center,
            proposed_uncertainty_action=proposed_uncertainty,
            center_action=center,
            uncertainty_action=uncertainty,
        )


class EnsembleLockedPolicy(EvidencePolicy):
    """Control policy: the ensemble is never adjusted."""

    policy_id = "ensemble_locked_v60"

    def apply(self, assessment, packet, horizon):  # noqa: ANN001, ANN201
        action = assessment.action_for(horizon)
        return PolicyDecision(
            policy_id=self.policy_id,
            horizon=horizon,
            eligible=False,
            evidence_tier="none",
            evidence_tier_level=0,
            eligibility_reasons=["Ensemble-locked control."],
            proposed_center_action=action.center_action,
            proposed_uncertainty_action=action.uncertainty_action,
            center_action="no_change",
            uncertainty_action="unchanged",
        )


__all__ = ["CLAIM_CLAUSES", "EnsembleLockedPolicy", "EvidencePolicy"]
