"""Typed contracts for CFM Agent v6.0.

Frozen after step 1 of the build: a schema change is reported, not made in place.
"""

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
ComponentName = Literal["bootstrap", "sparse_ar", "lightgbm"]
CurveState = Literal["contango", "backwardation", "flat"]

#: Fixed failure-code vocabulary (Section 1.6). ``library_error`` carries the
#: exception class name after a colon; the message stays audit-only.
FAILURE_CODES: tuple[str, ...] = (
    "insufficient_history",
    "insufficient_aligned_rows",
    "covariate_missing",
    "non_finite_output",
    "floor_violation",
    "shape_mismatch",
    "quantile_crossing",
    "library_error",
)


def is_failure_code(value: str) -> bool:
    """Return whether ``value`` is a failure code from the fixed vocabulary."""
    head = value.split(":", 1)[0]
    return head in FAILURE_CODES and (head == "library_error" or ":" not in value)


# --------------------------------------------------------------- research ----


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


class SummaryPassage(BaseModel):
    """One sentence-level slice of a verifier-cleaned summary (Section 4.1)."""

    model_config = ConfigDict(extra="forbid")
    passage_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class VerifiedSummary(BaseModel):
    """Main-verifier-approved active evidence and all associated sources."""

    model_config = ConfigDict(extra="forbid")
    verified_summary_id: str
    query_area: str
    query: str
    cutoff_date: str
    status: Literal["accepted", "failed_closed"]
    cleaned_summary: str = ""
    passages: list[SummaryPassage] = Field(default_factory=list)
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
    validator_id: str = "source_validator_v60_audit_only"
    records: list[AuditSourceRecord] = Field(default_factory=list)
    audit_only: bool = True


# ----------------------------------------------------------------- claims ----


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
    supporting_passage_ids: list[str] = Field(default_factory=list)
    supporting_source_ids: list[str] = Field(default_factory=list)
    contradicting_summary_ids: list[str] = Field(default_factory=list)
    contradicting_passage_ids: list[str] = Field(default_factory=list)
    contradicting_source_ids: list[str] = Field(default_factory=list)
    material_to_forecast: bool = True


class ClaimSupportFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_id: str
    verdict: Literal["direct", "partial", "unsupported", "contradicted", "not_assessed"]
    supporting_summary_ids: list[str] = Field(default_factory=list)
    supporting_passage_ids: list[str] = Field(default_factory=list)
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


# ----------------------------------------------------------------- policy ----


class TierEvaluation(BaseModel):
    """One evidence tier's requirement against the horizon's valid claims (Section 4.2)."""

    model_config = ConfigDict(extra="forbid")
    min_publishers: int = Field(ge=0)
    min_confidence: float = Field(ge=0, le=1)
    publishers: int = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    passed: bool


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
    claim_rejection_reasons: dict[str, list[str]] = Field(default_factory=dict)
    tier_evaluation: dict[int, TierEvaluation] = Field(default_factory=dict)
    eligibility_reasons: list[str] = Field(default_factory=list)
    proposed_center_action: CenterAction = "no_change"
    proposed_uncertainty_action: UncertaintyAction = "unchanged"
    center_action: CenterAction
    uncertainty_action: UncertaintyAction


# --------------------------------------------------- historical profile -----


class HistoricalReturnProfile(BaseModel):
    """Per-horizon transformation inputs computed from cutoff-safe history (Section 3.4).

    ``return_percentiles`` is keyed by percentile (the values of
    ``center_percentiles``); ``absolute_deviation_bands`` by coverage (the values
    of ``uncertainty_coverage`` plus the anchor). A recorded forecast can be
    replayed from this record and the settings alone.
    """

    model_config = ConfigDict(extra="forbid")
    horizon: int = Field(gt=0)
    n_eligible: int = Field(ge=0)
    min_required: int = Field(ge=0)
    eligible: bool
    window_start: str | None = None
    window_end: str | None = None
    conditioning: Literal["none", "volatility_quintile"] = "none"
    conditioning_bin: int | None = None
    median: float | None = None
    mean: float | None = None
    return_percentiles: dict[int, float] = Field(default_factory=dict)
    absolute_deviation_bands: dict[int, float] = Field(default_factory=dict)
    ineligibility_reason: str | None = None


# ------------------------------------------------------------ transformation


class ForecastTransformation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    engine_id: str = "python_forecast_engine_v60"
    horizon: int
    original_point_forecast: float
    original_quantiles: dict[float, float]
    p10_p90_width: float
    latest_price: float | None = None
    neutral: bool
    neutral_reason: str | None = None
    center_action: CenterAction
    selected_center_percentile: int | None = None
    selected_center_value: float | None = None
    center_sign_guard_applied: bool = False
    raw_center_adjustment: float
    novelty: NoveltyAssessment
    novelty_multiplier: float
    novelty_adjusted_amount: float
    emergency_cap: float
    applied_center_adjustment: float
    adjusted_p50: float
    uncertainty_action: UncertaintyAction
    uncertainty_coverage_target: int | None = None
    anchor_coverage: int | None = None
    anchor_band: float | None = None
    target_band: float | None = None
    uncertainty_multiplier: float
    uncertainty_clamp_applied: bool = False
    pre_floor_quantiles: dict[float, float]
    final_price_floor: float
    floor_applied: bool = False
    final_point_forecast: float
    final_quantiles: dict[float, float]
    historical_profile: HistoricalReturnProfile | None = None
    warnings: list[str] = Field(default_factory=list)


# -------------------------------------------------------------- market data


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
    """Training audit for one component (Sections 1.3 to 1.5)."""

    model_config = ConfigDict(extra="forbid")
    target_observations: int = Field(ge=0)
    first_target_date: str | None = None
    last_target_date: str | None = None
    returns_available: int = Field(default=0, ge=0)
    minimum_returns_required: int | None = Field(default=None, ge=0)
    eligible_blocks: int | None = Field(default=None, ge=0)
    block_length: int | None = Field(default=None, ge=1)
    return_lags: list[int] = Field(default_factory=list)
    covariates: list[str] = Field(default_factory=list)
    covariate_observations: dict[str, int] = Field(default_factory=dict)
    rows_before_alignment: int | None = Field(default=None, ge=0)
    covariate_fill_counts: dict[str, int] = Field(default_factory=dict)
    rows_dropped: int | None = Field(default=None, ge=0)
    rows_remaining: int | None = Field(default=None, ge=0)
    fitted_parameters: dict[str, float] = Field(default_factory=dict)
    residual_std: float | None = None
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


class ComponentDisposition(BaseModel):
    """Why a component is active or excluded; the code enters the suite identity, the reason does not."""

    model_config = ConfigDict(extra="forbid")
    status: Literal["ok", "error"]
    failure_code: str | None = None
    reason: str | None = None


class ModelForecastResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_name: str
    predictor_id: str
    status: Literal["ok", "error"]
    forecasts: list[ModelHorizonForecast] = Field(default_factory=list)
    training_data: ModelTrainingData
    num_paths: int = Field(ge=0)
    component_seed: int | None = Field(default=None, ge=0, le=4_294_967_295)
    simulated_floor_activations: int = Field(default=0, ge=0)
    failure_code: str | None = None
    error: str | None = None


class HistoryWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    setting_business_days: int = Field(ge=0)
    start_date: str | None = None
    end_date: str | None = None
    rows: int = Field(ge=0)


class FloorRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_price_floor: float
    simulated_price_floor: float
    final_price_floor: float
    input_floor_activation_count: int = Field(default=0, ge=0)
    input_floor_activation_dates: list[str] = Field(default_factory=list)
    simulated_floor_activation_counts: dict[str, int] = Field(default_factory=dict)


class DataQuality(BaseModel):
    """Data-quality diagnostics (Section 2)."""

    model_config = ConfigDict(extra="forbid")
    cutoff_date: str | None = None
    observation_counts: dict[str, int] = Field(default_factory=dict)
    first_dates: dict[str, str | None] = Field(default_factory=dict)
    last_dates: dict[str, str | None] = Field(default_factory=dict)
    input_floor_activation_count: int = Field(default=0, ge=0)
    input_floor_activation_dates: list[str] = Field(default_factory=list)
    covariate_fill_counts: dict[str, int] = Field(default_factory=dict)
    rows_before_alignment: int | None = Field(default=None, ge=0)
    rows_dropped: int | None = Field(default=None, ge=0)
    rows_remaining: int | None = Field(default=None, ge=0)
    missing_covariates: list[str] = Field(default_factory=list)


class MarketDiagnostics(BaseModel):
    """Deterministic, cutoff-safe diagnostics (Section 2). Every value is ``None`` when history is too short."""

    model_config = ConfigDict(extra="forbid")
    latest_observation_date: str | None = None
    latest_value: float | None = None
    calendar_gap_days: int | None = Field(default=None, ge=0)
    cumulative_return: dict[int, float | None] = Field(default_factory=dict)
    realized_volatility: dict[int, float | None] = Field(default_factory=dict)
    volatility_zscore: dict[int, float | None] = Field(default_factory=dict)
    drawdown: dict[int, float | None] = Field(default_factory=dict)
    jump_zscore: dict[int, float | None] = Field(default_factory=dict)
    covariate_latest_values: dict[str, float] = Field(default_factory=dict)
    curve_state: CurveState | None = None
    vix_level: float | None = None
    historical_profiles: dict[int, HistoricalReturnProfile] = Field(default_factory=dict)
    model_disagreement_std: dict[int, float] = Field(default_factory=dict)
    data_quality: DataQuality = Field(default_factory=DataQuality)


class ModelSuiteResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_suite_id: str
    target_series_id: str
    cutoff_date: str
    horizons: list[int]
    frequency: str = Field(min_length=1)
    model_num_paths: int = Field(ge=1)
    simulation_length_business_days: int = Field(ge=1)
    suite_seed: int = Field(ge=0, le=4_294_967_295)
    seed_derivation: str
    seed_input_fingerprint: str
    data_fingerprint: str
    history_window: HistoryWindow
    component_seeds: dict[str, int] = Field(default_factory=dict)
    components: dict[str, ModelForecastResult]
    ensemble: ModelForecastResult | None = None
    configured_ensemble_weights: dict[str, float]
    active_ensemble_weights: dict[str, float] = Field(default_factory=dict)
    successful_components: list[str] = Field(default_factory=list)
    failed_components: list[str] = Field(default_factory=list)
    dispositions: dict[str, ComponentDisposition] = Field(default_factory=dict)
    floors: FloorRecord
    ensemble_guard_applied: bool = False
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


# ---------------------------------------------------------- code execution


class CodeExecutionRequest(BaseModel):
    """One E2B request, accepted or rejected, as recorded in the audit (Section 4.5)."""

    model_config = ConfigDict(extra="forbid")
    request_index: int = Field(ge=1)
    purpose: str
    code: str
    code_chars: int = Field(ge=0)
    accepted: bool
    rejection_reasons: list[str] = Field(default_factory=list)
    launched: bool = False
    duration_seconds: float | None = Field(default=None, ge=0)
    stdout: str | None = None
    stderr: str | None = None
    error: str | None = None
    output_truncated: bool = False


__all__ = [
    "FAILURE_CODES",
    "ActiveSource",
    "AuditPassage",
    "AuditSourceRecord",
    "CenterAction",
    "ClaimSupportFinding",
    "CodeExecutionRequest",
    "ComponentDisposition",
    "ComponentName",
    "CurveState",
    "DataQuality",
    "EvidenceClaim",
    "EvidenceTier",
    "FloorRecord",
    "ForecastEligibility",
    "ForecastTransformation",
    "HistoricalReturnProfile",
    "HistoryWindow",
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
    "QueryVerificationAudit",
    "ResearchPacket",
    "ResearchQuery",
    "SearchVerificationAttempt",
    "SeriesObservation",
    "SeriesSnapshot",
    "SourceAuditBundle",
    "SourceQuality",
    "SourceTier",
    "SummaryPassage",
    "TierEvaluation",
    "ToolStatus",
    "UncertaintyAction",
    "VerificationAuditBundle",
    "VerifiedSummary",
    "is_failure_code",
]
