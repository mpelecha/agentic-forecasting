"""Graduated, bounded, auditable transformation policy for CFM Agent v3.4."""

from __future__ import annotations

import numpy as np
from energy_oil_forecasting.cfm_agent_v_3_4.config import CfmV34Settings
from energy_oil_forecasting.cfm_agent_v_3_4.outputs import CfmContextAssessmentOutput
from energy_oil_forecasting.cfm_agent_v_3_4.schemas import (
    HorizonAction,
    ModelHorizonForecast,
    PolicyDecision,
)


_PHYSICAL_TYPES = {"physical_supply", "shipping", "production_policy", "strategic_reserves", "inventory"}
_HIGH_QUALITY = {"tier_1_primary", "tier_2_independent"}


class ConstrainedActionPolicy:
    """Convert categorical LLM actions into evidence-tiered bounded transformations."""

    def __init__(self, settings: CfmV34Settings) -> None:
        self._settings = settings

    @property
    def policy_id(self) -> str:
        return "graduated_evidence_actions_v1"

    def _evidence_tier(  # noqa: PLR0912 — one linear gate sequence; splitting it hides the ladder.
        self,
        assessment: CfmContextAssessmentOutput,
        horizon: int,
    ) -> tuple[int, str, list[str], list[str], list[str], list[str]]:
        action = assessment.action_for(horizon)
        if action.center_action == "no_change" and action.uncertainty_action == "unchanged":
            return 0, "none", ["Neutral action requires no evidence gate."], [], [], []

        reasons: list[str] = []
        eligible_sources = {
            source.source_id: source
            for source in assessment.evidence_sources
            if source.provenance_status == "verified_from_tool"
            and source.verifier_content_status == "accepted_factual_content"
            and bool(source.verified_evidence_excerpt.strip())
            and source.source_tier != "tier_4_other"
        }
        publishers = sorted({source.publisher.strip().lower() for source in eligible_sources.values()})
        high_quality = any(source.source_tier in _HIGH_QUALITY for source in eligible_sources.values())
        tier_1_primary = any(source.source_tier == "tier_1_primary" for source in eligible_sources.values())

        claim_by_id = {claim.claim_id: claim for claim in assessment.evidence_claims}
        cited_claims = [claim_by_id[cid] for cid in action.cited_claim_ids if cid in claim_by_id]
        claim_support_counts: dict[str, int] = {}
        qualifying_claim_ids: list[str] = []
        for claim in cited_claims:
            supporting_publishers = {
                eligible_sources[sid].publisher.strip().lower()
                for sid in claim.supporting_source_ids
                if sid in eligible_sources
            }
            claim_support_counts[claim.claim_id] = len(supporting_publishers)
            if claim.support_status == "direct" and supporting_publishers and not claim.contradicting_source_ids:
                qualifying_claim_ids.append(claim.claim_id)

        every_claim_direct = bool(cited_claims) and len(qualifying_claim_ids) == len(cited_claims)
        every_claim_double = every_claim_direct and all(
            claim_support_counts[claim.claim_id] >= self._settings.tier_3_min_claim_support for claim in cited_claims
        )
        confirmed_material_physical = assessment.physical_status in {
            "partial_disruption", "confirmed_disruption"
        } and any(
            claim.claim_type in _PHYSICAL_TYPES
            and claim.claim_id in qualifying_claim_ids
            and claim_support_counts[claim.claim_id] >= self._settings.tier_3_min_claim_support
            for claim in cited_claims
        )
        novelty_ok = assessment.incremental_novelty in {
            "likely_new_relative_to_model_data", "possibly_partly_reflected"
        }

        if assessment.material_evidence_conflict:
            reasons.append("Material evidence conflict is unresolved.")
        if not novelty_ok:
            reasons.append("Incremental novelty is not sufficiently established.")
        if not cited_claims:
            reasons.append("Non-neutral action must cite at least one material claim.")
        if not every_claim_direct:
            reasons.append("Every cited material claim requires direct support from an eligible source.")
        if self._settings.require_high_quality_source and not high_quality:
            reasons.append("At least one eligible tier-1 or tier-2 source is required.")

        tier = 0
        tier_name = "none"
        if not reasons:
            if (
                len(publishers) >= self._settings.tier_1_min_publishers
                and assessment.confidence >= self._settings.tier_1_min_confidence
            ):
                tier, tier_name = 1, "limited"
            if (
                len(publishers) >= self._settings.tier_2_min_publishers
                and assessment.confidence >= self._settings.tier_2_min_confidence
            ):
                tier, tier_name = 2, "corroborated"
            if (
                len(publishers) >= self._settings.tier_3_min_publishers
                and assessment.confidence >= self._settings.tier_3_min_confidence
                and every_claim_double
                and tier_1_primary
                and confirmed_material_physical
                and assessment.incremental_novelty == "likely_new_relative_to_model_data"
            ):
                tier, tier_name = 3, "strong"

        if tier == 0 and not reasons:
            reasons.append(
                "Evidence is valid but does not meet the minimum publisher and confidence threshold for Tier 1."
            )
        if tier > 0:
            reasons.append(f"Graduated evidence Tier {tier} ({tier_name}) passed.")
        return (
            tier,
            tier_name,
            reasons,
            sorted(eligible_sources),
            publishers,
            sorted(qualifying_claim_ids),
        )

    def _uncertainty(
        self,
        *,
        action: HorizonAction,
        assessment: CfmContextAssessmentOutput,
        tier: int,
        eligible: bool,
    ) -> tuple[float, float, bool, str]:
        """Return the interval multiplier, scenario disagreement mass, and floor state.

        The tier caps how much the LLM's proposed uncertainty action may widen
        the interval. On top of that, a two-sided scenario set imposes a floor:
        it is the assessment's own admission that the situation could resolve
        either way, so unchanged or narrowed uncertainty contradicts it. The
        floor is deliberately independent of the evidence tier, because it
        widens only and never shifts the center; it therefore cannot
        manufacture a directional view out of weak evidence.
        """
        requested = {
            "unchanged": 1.0,
            "moderately_wider": self._settings.moderately_wider_multiplier,
            "substantially_wider": self._settings.substantially_wider_multiplier,
            "moderately_narrower": self._settings.moderately_narrower_multiplier,
        }[action.uncertainty_action]
        tier_cap = {
            0: 1.0,
            1: self._settings.moderately_wider_multiplier,
            2: self._settings.substantially_wider_multiplier,
            3: self._settings.substantially_wider_multiplier,
        }[tier]

        if not eligible:
            multiplier = 1.0
        elif requested > 1.0:
            multiplier = min(requested, tier_cap)
        elif requested < 1.0 and tier == 3:
            multiplier = requested
        else:
            multiplier = 1.0

        disagreement_mass = assessment.scenario_disagreement_mass()
        floor = self._settings.moderately_wider_multiplier
        if (
            self._settings.enforce_scenario_uncertainty_floor
            and disagreement_mass >= self._settings.scenario_disagreement_floor_mass
            and multiplier < floor
        ):
            return (
                floor,
                disagreement_mass,
                True,
                f"Scenario disagreement mass {disagreement_mass:.2f} met the floor; "
                "uncertainty was widened to at least moderately_wider.",
            )
        return multiplier, disagreement_mass, False, ""

    @staticmethod
    def _profile_multiplier(profile: str, horizon: int, horizons: list[int]) -> float:
        ordered = sorted(horizons)
        position = 0.0 if len(ordered) == 1 else ordered.index(horizon) / (len(ordered) - 1)
        if profile == "temporary":
            return 1.0 - 0.80 * position
        if profile == "decaying":
            return 1.0 - 0.60 * position
        if profile == "persistent":
            return 0.80 + 0.20 * position
        if profile == "delayed":
            return 0.40 + 0.60 * position
        return 0.0

    def apply(
        self,
        *,
        ensemble: ModelHorizonForecast,
        assessment: CfmContextAssessmentOutput,
        horizon: int,
        horizons: list[int],
        latest_price: float | None,
        cutoff_date: str,
    ) -> PolicyDecision:
        del cutoff_date  # Sources were deterministically cutoff-sanitized before policy evaluation.
        action = assessment.action_for(horizon)
        tier, tier_name, reasons, source_ids, publishers, claim_ids = self._evidence_tier(assessment, horizon)
        neutral = action.center_action == "no_change" and action.uncertainty_action == "unchanged"
        eligible = neutral or tier > 0
        quantiles = dict(ensemble.quantiles)
        median = quantiles[0.5]
        width = max(0.0, quantiles.get(0.9, median) - quantiles.get(0.1, median))

        requested_fraction = 0.0
        if action.center_action.startswith("small"):
            requested_fraction = self._settings.small_action_width_fraction
        elif action.center_action.startswith("moderate"):
            requested_fraction = self._settings.moderate_action_width_fraction
        allowed_fraction = {
            0: 0.0,
            1: self._settings.small_action_width_fraction,
            2: self._settings.moderate_action_width_fraction,
            3: self._settings.moderate_action_width_fraction,
        }[tier]
        fraction = min(requested_fraction, allowed_fraction)
        if assessment.incremental_novelty == "possibly_partly_reflected":
            fraction *= self._settings.partly_reflected_adjustment_multiplier

        direction = -1.0 if action.center_action.endswith("down") else 1.0
        if action.center_action == "no_change":
            direction = 0.0
        profile = self._profile_multiplier(action.persistence_profile, horizon, horizons)
        raw_adjustment = direction * fraction * width * profile
        price_bound = (
            abs(latest_price) * self._settings.max_center_adjustment_price_fraction
            if latest_price is not None
            else self._settings.max_center_adjustment_usd
        )
        bound = min(self._settings.max_center_adjustment_usd, price_bound)
        applied_adjustment = float(np.clip(raw_adjustment, -bound, bound)) if eligible else 0.0

        uncertainty_multiplier, disagreement_mass, floor_applied, floor_reason = self._uncertainty(
            action=action,
            assessment=assessment,
            tier=tier,
            eligible=eligible,
        )
        if floor_reason:
            reasons.append(floor_reason)

        transformed = {
            q: float(median + applied_adjustment + uncertainty_multiplier * (value - median))
            for q, value in quantiles.items()
        }
        previous = float("-inf")
        for q in sorted(transformed):
            transformed[q] = max(previous, transformed[q])
            previous = transformed[q]
        transformed[0.5] = float(median + applied_adjustment)

        approved_center = action.center_action
        approved_uncertainty = action.uncertainty_action
        if not eligible:
            approved_center = "no_change"
            approved_uncertainty = "unchanged"
        elif tier == 1 and approved_center in {"moderate_up", "moderate_down"}:
            approved_center = "small_up" if approved_center.endswith("up") else "small_down"
        if tier == 1 and approved_uncertainty == "substantially_wider":
            approved_uncertainty = "moderately_wider"
        if tier < 3 and approved_uncertainty == "moderately_narrower":
            approved_uncertainty = "unchanged"
        if floor_applied:
            approved_uncertainty = "moderately_wider"

        return PolicyDecision(
            policy_id=self.policy_id,
            horizon=horizon,
            eligible=eligible,
            evidence_tier=tier_name,
            evidence_tier_level=tier,
            eligible_source_ids=source_ids,
            eligible_publishers=publishers,
            qualifying_claim_ids=claim_ids,
            eligibility_reasons=reasons,
            center_action=approved_center,
            uncertainty_action=approved_uncertainty,
            raw_center_adjustment=float(raw_adjustment),
            applied_center_adjustment=applied_adjustment,
            uncertainty_multiplier=float(uncertainty_multiplier),
            scenario_disagreement_mass=float(disagreement_mass),
            scenario_uncertainty_floor_applied=floor_applied,
            final_point_forecast=transformed[0.5],
            final_quantiles=transformed,
        )


class EnsembleLockedPolicy(ConstrainedActionPolicy):
    """Control policy that always applies the authoritative ensemble unchanged."""

    @property
    def policy_id(self) -> str:
        return "ensemble_locked_v1"

    def apply(self, **kwargs) -> PolicyDecision:  # type: ignore[no-untyped-def]
        ensemble: ModelHorizonForecast = kwargs["ensemble"]
        horizon: int = kwargs["horizon"]
        return PolicyDecision(
            policy_id=self.policy_id,
            horizon=horizon,
            eligible=False,
            evidence_tier="none",
            evidence_tier_level=0,
            eligibility_reasons=["Ensemble-locked control policy."],
            center_action="no_change",
            uncertainty_action="unchanged",
            raw_center_adjustment=0.0,
            applied_center_adjustment=0.0,
            uncertainty_multiplier=1.0,
            final_point_forecast=ensemble.point_forecast,
            final_quantiles=dict(ensemble.quantiles),
        )


__all__ = ["ConstrainedActionPolicy", "EnsembleLockedPolicy"]
