"""Typed contracts for CFM Agent v5.2."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ToolStatus = Literal["ok", "partial", "error"]
SourceQuality = Literal[
    "tier_1_primary",
    "tier_2_independent",
    "tier_3_commentary",
    "tier_4_other",
    "unclassified",
]
SourceTier = Literal["tier_1_primary", "tier_2_independent", "tier_3_commentary", "tier_4_other"]
ForecastEligibility = Literal["eligible", "ineligible", "date_unknown", "unresolved", "not_assessed"]
CenterAction = Literal[
    "no_change",
    "small_up",
    "moderate_up",
    "large_up",
    "small_down",
    "moderate_down",
    "large_down",
]
UncertaintyAction = Literal[
    "unchanged",
    "small_wider",
    "moderately_wider",
    "substantially_wider",
    "small_narrower",
    "moderately_narrower",
    "substantially_narrower",
]
PersistenceProfile = Literal["temporary", "decaying", "persistent", "delayed", "unknown"]
NoveltyAssessment = Literal[
    "likely_new_relative_to_model_data",
    "possibly_partly_reflected",
    "likely_reflected_in_model_data",
    "indeterminate",
]
PhysicalStatus = Literal["normal", "elevated_risk", "partial_disruption", "confirmed_disruption", "unknown"]
EvidenceTier = Literal["none", "limited", "corroborated", "strong"]


class ResearchQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    area: str
    query: str


class SearchVerificationAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    attempt: int = Field(ge=1)
    clean: bool
    confidence: int = Field(ge=1, le=10)
    flagged_claims: list[str] = Field(default_factory=list)
    source_count: int = Field(ge=0)


class ActiveSource(BaseModel):
    """Mechanically resolved source identity available to the decision LLM."""

    model_config = ConfigDict(extra="forbid")
    source_id: str
    rank: int = Field(ge=1)
    grounding_url: str
    provider_title: str | None = None
    resolved_url: str | None = None
    resolved_domain: str | None = None
    publisher: str | None = None
    source_quality: SourceQuality = "unclassified"
    is_primary_or_official: bool = False
    resolution_status: Literal["resolved", "unresolved"]
    resolution_warning: str | None = None


class VerifiedSummary(BaseModel):
    """Main-verifier-approved active evidence and all associated sources."""

    model_config = ConfigDict(extra="forbid")
    verified_summary_id: str
    query_area: str
    query: str
    cutoff_date: str
    status: Literal["accepted", "failed_closed"]
    cleaned_summary: str = ""
    associated_source_ids: list[str] = Field(default_factory=list)
    verifier_confidence: int | None = Field(default=None, ge=1, le=10)
    verifier_attempt_count: int = Field(default=0, ge=0)


class QueryVerificationAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verified_summary_id: str
    query_area: str
    query: str
    cutoff_date: str
    status: Literal["accepted", "failed_closed"]
    attempts: list[SearchVerificationAttempt] = Field(default_factory=list)
    removed_flagged_claims: list[str] = Field(default_factory=list)
    audit_only: bool = True


class VerificationAuditBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verifier_id: str = "main_code_cutoff_verifier"
    queries: list[QueryVerificationAudit] = Field(default_factory=list)
    audit_only: bool = True


class ResearchPacket(BaseModel):
    """Active packet visible to the decision-making LLM."""

    model_config = ConfigDict(extra="forbid")
    packet_id: str
    cutoff_date: str
    queries: list[ResearchQuery]
    verified_summaries: list[VerifiedSummary]
    sources: list[ActiveSource]
    warnings: list[str] = Field(default_factory=list)
    active_evidence_type: Literal["main_verifier_cleaned_summaries"] = "main_verifier_cleaned_summaries"
    destination_page_content_included: bool = False


class AuditPassage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    passage_id: str
    source_id: str
    text: str
    extraction_method: str


class AuditSourceRecord(BaseModel):
    """Destination-page material that is never visible to the decision LLM."""

    model_config = ConfigDict(extra="forbid")
    source_id: str
    resolved_url: str | None = None
    title: str | None = None
    publisher_metadata: str | None = None
    publication_date: str | None = None
    publication_date_provenance: str = "missing"
    passages: list[AuditPassage] = Field(default_factory=list)
    extraction_warnings: list[str] = Field(default_factory=list)
    forecast_eligibility: ForecastEligibility = "not_assessed"
    source_quality: SourceQuality = "unclassified"
    potentially_mutable: bool = False
    audit_only: bool = True


class SourceAuditBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    audit_enabled: bool
    executed: bool
    validator_id: str = "source_validator_v52_audit_only"
    records: list[AuditSourceRecord] = Field(default_factory=list)
    audit_only: bool = True


class EvidenceClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    claim_type: Literal[
        "physical_supply",
        "shipping",
        "production_policy",
        "strategic_reserves",
        "inventory",
        "demand",
        "market_reaction",
        "other",
    ]
    supporting_summary_ids: list[str] = Field(default_factory=list)
    supporting_source_ids: list[str] = Field(default_factory=list)
    contradicting_summary_ids: list[str] = Field(default_factory=list)
    contradicting_source_ids: list[str] = Field(default_factory=list)
    material_to_forecast: bool = True


class ClaimSupportFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_id: str
    verdict: Literal["direct", "partial", "unsupported", "contradicted", "not_assessed"]
    supporting_summary_ids: list[str] = Field(default_factory=list)
    supporting_source_ids: list[str] = Field(default_factory=list)
    unsupported_elements: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0, le=1)
    audit_only: bool = True


class HorizonAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    horizon: int = Field(gt=0)
    center_action: CenterAction = "no_change"
    uncertainty_action: UncertaintyAction = "unchanged"
    persistence_profile: PersistenceProfile = "unknown"
    cited_claim_ids: list[str] = Field(default_factory=list)
    narrowing_uncertainty_resolved: str = ""
    rationale: str = Field(min_length=1)


class PolicyDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    policy_id: str
    horizon: int
    eligible: bool
    evidence_tier: EvidenceTier = "none"
    evidence_tier_level: int = 0
    resolved_source_ids: list[str] = Field(default_factory=list)
    resolved_publishers: list[str] = Field(default_factory=list)
    qualifying_claim_ids: list[str] = Field(default_factory=list)
    eligibility_reasons: list[str] = Field(default_factory=list)
    center_action: CenterAction
    uncertainty_action: UncertaintyAction


class ForecastTransformation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    engine_id: str = "python_forecast_engine_v52"
    horizon: int
    original_point_forecast: float
    original_quantiles: dict[float, float]
    p10_p90_width: float
    center_action: CenterAction
    action_fraction: float
    raw_center_adjustment: float
    novelty_multiplier: float
    novelty_adjusted_amount: float
    emergency_cap: float
    applied_center_adjustment: float
    uncertainty_action: UncertaintyAction
    uncertainty_multiplier: float
    pre_floor_quantiles: dict[float, float]
    final_point_forecast: float
    final_quantiles: dict[float, float]
    floor_applied: bool = False
    warnings: list[str] = Field(default_factory=list)


class SeriesObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    date: str
    value: float


class SeriesSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    series_id: str
    description: str
    source: str
    units: str
    frequency: str
    n_observations_at_cutoff: int = Field(ge=0)
    returned_observations: int = Field(ge=0)
    first_available_date: str | None = None
    last_available_date: str | None = None
    last_value: float | None = None
    observations: list[SeriesObservation] = Field(default_factory=list)


class ModelTrainingData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_observations: int = Field(ge=0)
    first_target_date: str | None = None
    last_target_date: str | None = None
    aligned_observations: int | None = Field(default=None, ge=0)
    effective_training_examples_estimate: int | None = Field(default=None, ge=0)
    target_lags: int | None = Field(default=None, ge=1)
    covariate_lags: int | None = Field(default=None, ge=1)
    covariates: list[str] = Field(default_factory=list)
    covariate_observations: dict[str, int] = Field(default_factory=dict)
    notes: str = ""


class ModelHorizonForecast(BaseModel):
    model_config = ConfigDict(extra="forbid")
    horizon: int = Field(gt=0)
    forecast_date: str
    point_forecast: float
    quantiles: dict[float, float]

    @model_validator(mode="after")
    def validate_quantiles(self) -> ModelHorizonForecast:
        missing = {0.1, 0.5, 0.9} - set(self.quantiles)
        if missing:
            raise ValueError(f"required quantiles missing: {sorted(missing)}")
        if not all(math.isfinite(value) for value in self.quantiles.values()):
            raise ValueError("quantiles must be finite.")
        ordered = [self.quantiles[q] for q in sorted(self.quantiles)]
        if ordered != sorted(ordered):
            raise ValueError("quantiles must be non-decreasing.")
        if 0.5 not in self.quantiles or self.point_forecast != self.quantiles[0.5]:
            raise ValueError("point_forecast must equal the 0.5 quantile.")
        return self


class ModelForecastResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_name: str
    predictor_id: str
    status: Literal["ok", "error"]
    forecasts: list[ModelHorizonForecast] = Field(default_factory=list)
    training_data: ModelTrainingData
    num_samples: int = Field(ge=1)
    error: str | None = None


class MarketDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")
    latest_observation_date: str | None = None
    latest_value: float | None = None
    calendar_gap_days: int | None = Field(default=None, ge=0)
    return_1b: float | None = None
    return_5b: float | None = None
    return_21b: float | None = None
    realized_volatility_21b: float | None = Field(default=None, ge=0.0)
    drawdown_63b: float | None = None
    jump_zscore_63b: float | None = None
    covariate_latest_values: dict[str, float] = Field(default_factory=dict)


class ModelSuiteResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_suite_id: str
    target_series_id: str
    cutoff_date: str
    horizons: list[int]
    frequency: str = Field(min_length=1)
    model_num_samples: int = Field(ge=1)
    rng_seed: int = Field(ge=0, le=4_294_967_295)
    rng_seed_derivation: str
    rng_seed_input_fingerprint: str
    models: dict[str, ModelForecastResult]
    ensemble: ModelForecastResult | None = None
    configured_ensemble_weights: dict[str, float]
    active_ensemble_weights: dict[str, float] = Field(default_factory=dict)
    successful_models: list[str] = Field(default_factory=list)
    failed_models: list[str] = Field(default_factory=list)
    model_disagreement_std: dict[int, float] = Field(default_factory=dict)
    diagnostics: MarketDiagnostics


class MarketDataToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: ToolStatus
    operation: str
    cutoff_date: str
    available_series: list[str] = Field(default_factory=list)
    series: list[SeriesSnapshot] = Field(default_factory=list)
    model_suite: ModelSuiteResult | None = None
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
