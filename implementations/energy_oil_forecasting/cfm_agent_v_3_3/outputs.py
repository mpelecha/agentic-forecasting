"""Structured contextual-assessment output for CFM Agent v3.3."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import ClassVar, Literal

import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES, ContinuousForecast, Prediction
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic.outputs import AgentForecastOutput
from energy_oil_forecasting.cfm_agent_v_3_3.schemas import (
    EvidenceClaim,
    EvidenceSource,
    HorizonAction,
    NoveltyAssessment,
    PhysicalStatus,
)
from pydantic import Field, model_validator


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

    def action_for(self, horizon: int) -> HorizonAction:
        return next(action for action in self.horizon_actions if action.horizon == horizon)

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
