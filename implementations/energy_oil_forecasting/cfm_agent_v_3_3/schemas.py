"""Typed contracts for CFM Agent v3.3 numerical, research, and policy layers."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ToolStatus = Literal["ok", "partial", "error"]
ProvenanceStatus = Literal["verified_from_tool", "inferred_by_agent", "unresolved"]
VerifierContentStatus = Literal["accepted_factual_content", "empty", "unknown"]
VerifierProcessingStatus = Literal["accepted_clean", "accepted_after_removal", "empty_after_verification", "unknown"]
SourceTier = Literal["tier_1_primary", "tier_2_independent", "tier_3_commentary", "tier_4_other"]
CenterAction = Literal["no_change", "small_up", "moderate_up", "small_down", "moderate_down"]
UncertaintyAction = Literal["unchanged", "moderately_wider", "substantially_wider", "moderately_narrower"]
PersistenceProfile = Literal["temporary", "decaying", "persistent", "delayed", "unknown"]
NoveltyAssessment = Literal[
    "likely_new_relative_to_model_data",
    "possibly_partly_reflected",
    "likely_reflected_in_model_data",
    "indeterminate",
]
PhysicalStatus = Literal["normal", "elevated_risk", "partial_disruption", "confirmed_disruption", "unknown"]
EvidenceTier = Literal["none", "limited", "corroborated", "strong"]


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
    def validate_quantiles(self) -> "ModelHorizonForecast":
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
    model_num_samples: int = Field(ge=1)
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


class EvidenceSource(BaseModel):
    """One source used by the assessment."""

    model_config = ConfigDict(extra="ignore")
    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    source_tier: SourceTier
    publication_date: str | None = None
    is_primary_or_official: bool = False
    provenance_status: ProvenanceStatus = "unresolved"
    verifier_content_status: VerifierContentStatus = "unknown"
    verifier_processing_status: VerifierProcessingStatus = "unknown"
    verified_evidence_excerpt: str = ""


class EvidenceClaim(BaseModel):
    """One normalized material claim and its supporting source IDs."""

    model_config = ConfigDict(extra="ignore")
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
    support_status: Literal["direct", "partial", "unsupported", "conflicting"]
    supporting_source_ids: list[str] = Field(default_factory=list)
    contradicting_source_ids: list[str] = Field(default_factory=list)
    material_to_forecast: bool = True


class HorizonAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    horizon: int = Field(gt=0)
    center_action: CenterAction = "no_change"
    uncertainty_action: UncertaintyAction = "unchanged"
    persistence_profile: PersistenceProfile = "unknown"
    cited_claim_ids: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)


class PolicyDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    policy_id: str
    horizon: int
    eligible: bool
    evidence_tier: EvidenceTier = "none"
    evidence_tier_level: int = Field(default=0, ge=0, le=3)
    eligible_source_ids: list[str] = Field(default_factory=list)
    eligible_publishers: list[str] = Field(default_factory=list)
    qualifying_claim_ids: list[str] = Field(default_factory=list)
    eligibility_reasons: list[str] = Field(default_factory=list)
    center_action: CenterAction
    uncertainty_action: UncertaintyAction
    raw_center_adjustment: float
    applied_center_adjustment: float
    uncertainty_multiplier: float = Field(gt=0.0)
    final_point_forecast: float
    final_quantiles: dict[float, float]


__all__ = [
    "CenterAction",
    "EvidenceClaim",
    "EvidenceTier",
    "EvidenceSource",
    "HorizonAction",
    "MarketDataToolResult",
    "MarketDiagnostics",
    "ModelForecastResult",
    "ModelHorizonForecast",
    "ModelSuiteResult",
    "ModelTrainingData",
    "NoveltyAssessment",
    "PersistenceProfile",
    "PhysicalStatus",
    "PolicyDecision",
    "ProvenanceStatus",
    "SeriesObservation",
    "SeriesSnapshot",
    "SourceTier",
    "ToolStatus",
    "VerifierContentStatus",
    "VerifierProcessingStatus",
    "UncertaintyAction",
]
