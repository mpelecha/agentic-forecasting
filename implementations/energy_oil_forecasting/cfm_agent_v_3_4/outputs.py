"""Structured contextual-assessment output for CFM Agent v3.4."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import ClassVar, Literal, get_args

import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES, ContinuousForecast, Prediction
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic.outputs import AgentForecastOutput
from energy_oil_forecasting.cfm_agent_v_3_4.schemas import (
    ContextScenario,
    DriverCategory,
    EvidenceClaim,
    EvidenceSource,
    HorizonAction,
    NoveltyAssessment,
    PhysicalStatus,
)
from pydantic import Field, model_validator


# Scenario probabilities must sum to 1 within this absolute tolerance.
_PROBABILITY_SUM_TOLERANCE = 0.02

# Encodes the categorical center action as a signed magnitude for the
# calibration feature row. These are ordinal codes, not price effects — the
# fitted mapping supplies the price scale.
_CENTER_ACTION_CODE = {
    "no_change": 0.0,
    "small_up": 1.0,
    "moderate_up": 2.0,
    "small_down": -1.0,
    "moderate_down": -2.0,
}
_UNCERTAINTY_ACTION_CODE = {
    "moderately_narrower": -1.0,
    "unchanged": 0.0,
    "moderately_wider": 1.0,
    "substantially_wider": 2.0,
}
_MAGNITUDE_CODE = {"small": 1.0, "moderate": 2.0, "large": 3.0}
_DIRECTION_SIGN = {"up": 1.0, "down": -1.0, "neutral": 0.0}


class CfmContextAssessmentOutput(AgentForecastOutput):
    """LLM research assessment; final numerical arithmetic is deliberately absent."""

    modality: ClassVar[Literal["continuous", "discrete", "categorical"]] = "continuous"
    model_config = {"extra": "ignore"}

    reported_search_queries: list[str] = Field(default_factory=list)
    evidence_sources: list[EvidenceSource] = Field(default_factory=list)
    evidence_claims: list[EvidenceClaim] = Field(default_factory=list)
    physical_status: PhysicalStatus = "unknown"
    incremental_novelty: NoveltyAssessment = "indeterminate"
    material_evidence_conflict: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
    scenarios: list[ContextScenario] = Field(
        default_factory=list,
        description="2-3 competing storylines with probabilities summing to 1, including one tail case.",
    )
    horizon_actions: list[HorizonAction]
    research_summary: str = Field(min_length=1)
    overall_rationale: str = Field(min_length=1)
    e2b_used: bool = False
    e2b_summary: str = ""
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_assessment(self) -> "CfmContextAssessmentOutput":
        horizons = [action.horizon for action in self.horizon_actions]
        if not horizons or len(horizons) != len(set(horizons)):
            raise ValueError("horizon_actions must be nonempty and unique by horizon.")
        source_ids = [source.source_id for source in self.evidence_sources]
        claim_ids = [claim.claim_id for claim in self.evidence_claims]
        if len(source_ids) != len(set(source_ids)) or len(claim_ids) != len(set(claim_ids)):
            raise ValueError("source_id and claim_id values must be unique.")
        known_sources = set(source_ids)
        for claim in self.evidence_claims:
            referenced = set(claim.supporting_source_ids) | set(claim.contradicting_source_ids)
            if not referenced.issubset(known_sources):
                raise ValueError(f"claim {claim.claim_id} references an unknown source_id.")
        known_claims = set(claim_ids)
        for action in self.horizon_actions:
            if not set(action.cited_claim_ids).issubset(known_claims):
                raise ValueError(f"horizon {action.horizon} references an unknown claim_id.")
        if self.e2b_used and not self.e2b_summary.strip():
            raise ValueError("e2b_summary is required when e2b_used is true.")
        return self

    @model_validator(mode="after")
    def validate_scenarios(self) -> "CfmContextAssessmentOutput":
        """Check the scenario set's well-formedness when the model supplies one.

        An absent scenario set is tolerated rather than rejected: the policy
        then applies no scenario-derived uncertainty floor, and the agent
        degrades to the v3.3 behaviour. A *malformed* set is rejected, because
        Python reads it to size uncertainty and must be able to trust it.
        """
        if not self.scenarios:
            return self

        if len(self.scenarios) < 2:
            raise ValueError("A scenario set must contain at least two competing scenarios.")

        scenario_ids = [scenario.scenario_id for scenario in self.scenarios]
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("scenario_id values must be unique.")

        known_claims = {claim.claim_id for claim in self.evidence_claims}
        for scenario in self.scenarios:
            if not set(scenario.cited_claim_ids).issubset(known_claims):
                raise ValueError(f"scenario {scenario.scenario_id} references an unknown claim_id.")

        total = sum(scenario.probability for scenario in self.scenarios)
        if abs(total - 1.0) > _PROBABILITY_SUM_TOLERANCE:
            raise ValueError(f"Scenario probabilities must sum to 1, got {total:.3f}.")

        tails = [scenario for scenario in self.scenarios if scenario.is_tail_case]
        non_tails = [scenario for scenario in self.scenarios if not scenario.is_tail_case]
        if not tails:
            raise ValueError("At least one scenario must set is_tail_case=true.")
        if not non_tails:
            raise ValueError("At least one scenario must be a non-tail mainline case.")
        highest_non_tail = max(scenario.probability for scenario in non_tails)
        for tail in tails:
            if tail.probability > highest_non_tail:
                raise ValueError(
                    f"Tail scenario '{tail.name}' is more likely than every mainline scenario; "
                    "a tail case must be low-probability."
                )
            if tail.magnitude == "small":
                raise ValueError(
                    f"Tail scenario '{tail.name}' has small magnitude; a tail case must be high-impact."
                )

        shapes = [(scenario.direction, scenario.magnitude) for scenario in self.scenarios]
        if len(shapes) != len(set(shapes)):
            raise ValueError(
                "Two scenarios share the same direction and magnitude — that is one scenario "
                "written twice. Differentiate them by direction or by magnitude."
            )
        return self

    def action_for(self, horizon: int) -> HorizonAction:
        return next(action for action in self.horizon_actions if action.horizon == horizon)

    def scenario_disagreement_mass(self) -> float:
        """Return the probability mass on the less-favoured price direction.

        This is ``min(P(up), P(down))``: the share of the model's own scenario
        set that argues against its central story. Zero when the scenarios all
        point one way, or when no scenario set was supplied. The policy uses
        this to floor the uncertainty multiplier, so a genuinely two-sided
        situation cannot be reported with unchanged uncertainty.
        """
        up = sum(scenario.probability for scenario in self.scenarios if scenario.direction == "up")
        down = sum(scenario.probability for scenario in self.scenarios if scenario.direction == "down")
        return float(min(up, down))

    def calibration_features(self) -> dict[str, float]:
        """Return the fixed-width numeric row the calibration layer consumes.

        Every key exists on every run, whatever the LLM found. That is the
        point: factor and claim text changes between origins, so a learned
        score-to-price mapping can only be fitted on stable slots. Category
        totals count *eligible direct* claims only, so sanitizer-rejected
        evidence cannot inflate a feature.

        Pair these features with the ``unadjusted_ensemble`` recorded in
        prediction metadata to build the regression target: the residual of
        the actual move against the quant baseline's expectation.
        """
        direct_claims = [claim for claim in self.evidence_claims if claim.support_status == "direct"]
        features: dict[str, float] = {
            "confidence": float(self.confidence),
            "direct_claim_count": float(len(direct_claims)),
            "eligible_source_count": float(len(self.evidence_sources)),
            "material_evidence_conflict": 1.0 if self.material_evidence_conflict else 0.0,
            "physical_disruption": 1.0
            if self.physical_status in {"partial_disruption", "confirmed_disruption"}
            else 0.0,
            "novelty_is_new": 1.0
            if self.incremental_novelty == "likely_new_relative_to_model_data"
            else 0.0,
            "scenario_count": float(len(self.scenarios)),
            "scenario_disagreement_mass": self.scenario_disagreement_mass(),
            "expected_scenario_impact": float(
                sum(
                    scenario.probability
                    * _DIRECTION_SIGN[scenario.direction]
                    * _MAGNITUDE_CODE[scenario.magnitude]
                    for scenario in self.scenarios
                )
            ),
            "tail_probability": float(
                sum(scenario.probability for scenario in self.scenarios if scenario.is_tail_case)
            ),
        }
        for category in get_args(DriverCategory):
            features[f"claims_{category}"] = float(
                sum(1 for claim in direct_claims if claim.driver_category == category)
            )
        for action in self.horizon_actions:
            features[f"center_action_h{action.horizon}"] = _CENTER_ACTION_CODE[action.center_action]
            features[f"uncertainty_action_h{action.horizon}"] = _UNCERTAINTY_ACTION_CODE[
                action.uncertainty_action
            ]
        return features

    @classmethod
    def prompt_schema_json(cls) -> str:
        template = {
            "reported_search_queries": [
                "underlying event and official statements",
                "confirmed physical oil-flow impact",
                "production or strategic-reserve response",
                "independent market reaction reporting",
            ],
            "evidence_sources": [
                {
                    "source_id": "source_001",
                    "title": "<title>",
                    "source_url": "<url>",
                    "publisher": "<publisher>",
                    "source_tier": "<tier_1_primary|tier_2_independent|tier_3_commentary|tier_4_other>",
                    "publication_date": "<YYYY-MM-DD|null>",
                    "is_primary_or_official": False,
                    "provenance_status": "<verified_from_tool|inferred_by_agent|unresolved>",
                    "verifier_content_status": "<accepted_factual_content|empty|unknown>",
                    "verifier_processing_status": "<accepted_clean|accepted_after_removal|empty_after_verification|unknown>",
                    "verified_evidence_excerpt": "<exact concise excerpt from accepted tool output, or empty>",
                }
            ],
            "evidence_claims": [
                {
                    "claim_id": "claim_001",
                    "statement": "<normalized fact, not interpretation>",
                    "claim_type": "<physical_supply|shipping|production_policy|strategic_reserves|inventory|demand|market_reaction|other>",
                    "driver_category": "<geopolitical|weather|operational|policy|demand_expectations>",
                    "support_status": "<direct|partial|unsupported|conflicting>",
                    "supporting_source_ids": ["source_001"],
                    "contradicting_source_ids": [],
                    "material_to_forecast": True,
                }
            ],
            "physical_status": "<normal|elevated_risk|partial_disruption|confirmed_disruption|unknown>",
            "incremental_novelty": "<likely_new_relative_to_model_data|possibly_partly_reflected|likely_reflected_in_model_data|indeterminate>",
            "material_evidence_conflict": False,
            "confidence": "<0..1>",
            "scenarios": [
                {
                    "scenario_id": "scenario_001",
                    "name": "<short storyline name>",
                    "probability": "<0..1; the set must sum to 1>",
                    "direction": "<up|down|neutral>",
                    "magnitude": "<small|moderate|large; neutral must be small>",
                    "is_tail_case": "<true for exactly the low-probability, high-impact storyline>",
                    "cited_claim_ids": ["claim_001"],
                    "rationale": "<one or two sentences>",
                }
            ],
            "horizon_actions": [
                {
                    "horizon": "<requested integer>",
                    "center_action": "<no_change|small_up|moderate_up|small_down|moderate_down>",
                    "uncertainty_action": "<unchanged|moderately_wider|substantially_wider|moderately_narrower>",
                    "persistence_profile": "<temporary|decaying|persistent|delayed|unknown>",
                    "cited_claim_ids": ["claim_001"],
                    "rationale": "<concise explanation>",
                }
            ],
            "research_summary": "<facts, conflicts, and limitations>",
            "overall_rationale": "<assessment only; do not calculate final prices>",
            "e2b_used": False,
            "e2b_summary": "",
            "warnings": [],
        }
        return json.dumps(template, indent=2)

    def to_predictions(
        self,
        *,
        task: ForecastingTask,
        context: ForecastContext,
        predictor_id: str,
        metadata: dict[str, object] | None = None,
    ) -> list[Prediction]:
        """Create temporary placeholders; the v3 wrapper replaces them from policy output."""
        if {action.horizon for action in self.horizon_actions} != set(task.horizons):
            raise ValueError("assessment must contain exactly the requested horizons.")
        issued_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        offset = pd.tseries.frequencies.to_offset(task.frequency)
        assessment_metadata = dict(metadata or {})
        assessment_metadata["context_assessment"] = self.model_dump(mode="json")
        return [
            Prediction(
                predictor_id=predictor_id,
                task_id=task.task_id,
                issued_at=issued_at,
                as_of=context.as_of,
                forecast_date=(pd.Timestamp(context.as_of) + offset * horizon).to_pydatetime(),
                payload=ContinuousForecast(
                    point_forecast=0.0,
                    quantiles=dict.fromkeys(STANDARD_QUANTILES, 0.0),
                ),
                metadata=dict(assessment_metadata),
            )
            for horizon in task.horizons
        ]


__all__ = ["CfmContextAssessmentOutput"]
