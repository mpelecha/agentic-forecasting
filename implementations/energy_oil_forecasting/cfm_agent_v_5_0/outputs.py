"""Structured LLM assessment output for CFM Agent v5.0."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import ClassVar, Literal

import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import (
    STANDARD_QUANTILES,
    ContinuousForecast,
    Prediction,
)
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic.outputs import AgentForecastOutput
from energy_oil_forecasting.cfm_agent_v_5_0.schemas import (
    EvidenceClaim,
    HorizonAction,
    NoveltyAssessment,
    PhysicalStatus,
)
from pydantic import Field, model_validator


class CfmContextAssessmentOutput(AgentForecastOutput):
    modality: ClassVar[Literal["continuous", "discrete", "categorical"]] = "continuous"
    model_config = {"extra": "ignore"}
    research_packet_id: str
    evidence_claims: list[EvidenceClaim] = Field(default_factory=list)
    physical_status: PhysicalStatus = "unknown"
    incremental_novelty: NoveltyAssessment = "indeterminate"
    material_evidence_conflict: bool = False
    confidence: float = Field(ge=0, le=1)
    horizon_actions: list[HorizonAction]
    research_summary: str = Field(min_length=1)
    overall_rationale: str = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_links(self):
        ids = [c.claim_id for c in self.evidence_claims]
        if len(ids) != len(set(ids)):
            raise ValueError("claim IDs must be unique")
        known = set(ids)
        hs = [a.horizon for a in self.horizon_actions]
        if not hs or len(hs) != len(set(hs)):
            raise ValueError("horizon actions must be nonempty and unique")
        for a in self.horizon_actions:
            if not set(a.cited_claim_ids) <= known:
                raise ValueError("action references unknown claim")
        return self

    def action_for(self, horizon: int) -> HorizonAction:
        return next(a for a in self.horizon_actions if a.horizon == horizon)

    @classmethod
    def prompt_schema_json(cls) -> str:
        return json.dumps(
            {
                "research_packet_id": "<exact packet id returned by tool>",
                "evidence_claims": [
                    {
                        "claim_id": "claim_001",
                        "statement": "<one atomic fact>",
                        "claim_type": "physical_supply",
                        "supporting_summary_ids": ["verified_summary:sha256:<id>"],
                        "supporting_source_ids": ["source_001"],
                        "contradicting_summary_ids": [],
                        "contradicting_source_ids": [],
                        "material_to_forecast": True,
                    }
                ],
                "physical_status": "unknown",
                "incremental_novelty": "indeterminate",
                "material_evidence_conflict": False,
                "confidence": "<0..1>",
                "horizon_actions": [
                    {
                        "horizon": "<requested>",
                        "center_action": "<no_change|small_up|moderate_up|large_up|small_down|moderate_down|large_down>",
                        "uncertainty_action": "<unchanged|small_wider|moderately_wider|substantially_wider|small_narrower|moderately_narrower|substantially_narrower>",
                        "persistence_profile": "<temporary|decaying|persistent|delayed|unknown>",
                        "cited_claim_ids": ["claim_001"],
                        "narrowing_uncertainty_resolved": "",
                        "rationale": "<concise>",
                    }
                ],
                "research_summary": "<summary>",
                "overall_rationale": "<assessment only>",
                "warnings": [],
            },
            indent=2,
        )

    def to_predictions(
        self,
        *,
        task: ForecastingTask,
        context: ForecastContext,
        predictor_id: str,
        metadata: dict | None = None,
    ) -> list[Prediction]:
        issued = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        off = pd.tseries.frequencies.to_offset(task.frequency)
        md = dict(metadata or {})
        md["context_assessment"] = self.model_dump(mode="json")
        return [
            Prediction(
                predictor_id=predictor_id,
                task_id=task.task_id,
                issued_at=issued,
                as_of=context.as_of,
                forecast_date=(pd.Timestamp(context.as_of) + off * h).to_pydatetime(),
                payload=ContinuousForecast(point_forecast=0.0, quantiles=dict.fromkeys(STANDARD_QUANTILES, 0.0)),
                metadata=dict(md),
            )
            for h in task.horizons
        ]
