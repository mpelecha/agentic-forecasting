"""Data contracts for CFM Coach.

``RunRecord`` is the coach's only coupling to ``cfm_agent_v_5_0``: a schema-versioned
snapshot of one agent execution, complete enough that the forecast can be *replayed*
under a different calibration without re-running the LLM or re-searching the web.

Everything the replay needs is already produced by v5.0 today -- the categorical
assessment, the research packet, the per-model and ensemble forecasts. What is not
produced today is the *configuration* that turned those into numbers, which is why
``settings``, ``calibration_version``, ``agent_model``, ``package_fingerprint`` and
``prompt_version`` are required fields here (R6).
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


#: Where a run's research came from. Only ``live_forward`` is fitting evidence:
#: a run re-executed at a historical cutoff searches today's web, and even when
#: every cited source predates the cutoff, the *selection* is shaped by what
#: turned out to matter. That bias is invisible to v5.0's leakage verifier,
#: which inspects content rather than retrieval.
Provenance = Literal[
    "live_forward",
    "backfill_cached_news",
    "replayed_live_search",
    "ensemble_only",
    # The coach's own forecast: a live record replayed under the coach-owned
    # ledger the same day. An output, never evidence -- `fitting_eligible_provenance`
    # keeps it out of every fit.
    "coached_replay",
]

#: Whether a record's settings were captured at run time or reconstructed after
#: the fact. Backfilled records predate R6, so their settings are inferred from
#: the package defaults and must never be treated as observed.
SettingsSource = Literal["recorded", "inferred"]

TrustTier = Literal["high", "medium", "low", "no_basis"]

Quantiles = dict[float, float]


class HorizonRecord(BaseModel):
    """One horizon of one run: the ensemble input, the recorded output, and the decision between."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    horizon: int = Field(gt=0)
    forecast_date: datetime

    # Component forecasts are kept so ensemble weights are replayable offline.
    component_quantiles: dict[str, Quantiles] = Field(default_factory=dict)

    # The unadjusted ensemble -- the baseline every candidate calibration is
    # measured against, and the input `PythonForecastEngine.transform` consumes.
    ensemble_point_forecast: float
    ensemble_quantiles: Quantiles

    # What the agent actually published. `ReplayEngine.verify_fidelity` requires
    # that replaying under this run's own calibration reproduces these exactly.
    final_point_forecast: float
    final_quantiles: Quantiles

    policy_decision: dict[str, Any] = Field(default_factory=dict)
    forecast_transformation: dict[str, Any] = Field(default_factory=dict)


class RunRecord(BaseModel):
    """One complete `cfm_agent_v_5_0` execution, replayable under any calibration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int
    run_id: str

    # -- identity of the system that produced this forecast (R6) --------------
    agent_id: str
    agent_model: str
    predictor_id: str
    package_fingerprint: str
    calibration_version: str
    prompt_version: str
    settings: dict[str, Any]
    settings_source: SettingsSource = "recorded"
    provenance: Provenance

    # -- task binding ----------------------------------------------------------
    task_id: str
    cutoff: date
    horizons: list[int]
    issued_at: datetime

    # -- replay inputs ---------------------------------------------------------
    assessment: dict[str, Any]
    research_packet: dict[str, Any]
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    # -- trust-bearing signals v5.0 computes and then discards ------------------
    # Audit-only by design: these may inform the trust report (R2) and may never
    # reach the evidence policy or the forecast engine (R9).
    audit_signals: dict[str, Any] = Field(default_factory=dict)

    forecasts: list[HorizonRecord]

    def horizon(self, horizon: int) -> HorizonRecord:
        for record in self.forecasts:
            if record.horizon == horizon:
                return record
        raise KeyError(f"run {self.run_id} has no horizon {horizon}")

    @property
    def is_fitting_eligible_provenance(self) -> bool:
        return self.provenance == "live_forward"


class ResolvedForecast(BaseModel):
    """One (run, horizon) paired with what WTI actually did."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    horizon: int
    cutoff: date
    forecast_date: datetime
    realized_value: float
    # The observation the price was taken from -- may precede `forecast_date`
    # when the target date is not a trading day.
    realized_observation_date: date
    resolved_at: datetime


class ScoreCard(BaseModel):
    """One (run, horizon, variant) scored against what WTI actually did.

    ``variant`` names *which* forecast was scored for this run and horizon -- the raw
    ensemble, the agent as it published, or a candidate calibration replayed over the
    same stored assessment. Every variant of one (run, horizon) shares a realized
    value, which is what makes the comparison paired (R10): the days are identical by
    construction, so a difference in score is a difference in calibration and not a
    difference in which days each variant happened to be scored on.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    horizon: int
    cutoff: date
    forecast_date: datetime
    variant: str
    calibration_version: str

    realized_value: float
    point_forecast: float
    p10: float
    p50: float
    p90: float

    #: Mean pinball loss over the quantile grid -- the primary metric. Proper for a
    #: quantile forecast and directly comparable across variants on a shared grid.
    pinball: float
    #: Quantile-integrated CRPS (2 * the integral of pinball over tau).
    crps: float
    #: `properscoring.crps_ensemble` over the quantile values, matching the
    #: framework's own scorer so coach numbers stay comparable with the repo leaderboard.
    crps_ensemble: float
    absolute_error: float
    interval_width: float
    covered_80: bool

    #: Last close visible at the cutoff -- the reference every direction is measured
    #: from. None when the caller had no `diagnostics.latest_value` to pass.
    last_value: float | None = None
    #: Sign of ``p50 - last_value``: +1 up, -1 down, 0 no call (within `scoring.FLAT_EPS`).
    direction_call: int | None = None
    #: Sign of ``realized_value - last_value``, on the same convention.
    direction_outcome: int | None = None

    @property
    def direction_hit(self) -> bool | None:
        """Whether a directional call matched the move. None when either side is flat or unknown.

        A flat call is not a wrong call and a flat move cannot be called, so both are
        excluded rather than counted as misses -- the same rule the corpus page uses.
        """
        if not self.direction_call or not self.direction_outcome:
            return None
        return self.direction_call == self.direction_outcome

    @property
    def is_paired_key(self) -> tuple[str, int]:
        """The identity a paired comparison joins on."""
        return (self.run_id, self.horizon)


class CalibrationLayer(BaseModel):
    """Coach-owned post-engine transform, applied to v5.0's final quantiles.

    Exists because the highest-leverage lever -- the *base* interval width -- has
    no in-engine control: v5.0's six uncertainty multipliers only fire when the
    LLM asks for a width change, and a neutral decision bypasses them entirely.
    Owning the layer here keeps `cfm_agent_v_5_0` unmodified.

        centre' = ens_p50 + centre_gain * (final_p50 - ens_p50)
        q'      = centre' + width_scale * (q - final_p50)
        q''     = q' + rw_anchor * (last_close - centre')

    ``centre_gain`` of 0 discards the LLM overlay entirely; 1 is v5.0 unchanged.
    ``rw_anchor`` pulls the whole distribution toward the last close: 0 leaves
    it, 1 puts the median on today's price (the random walk's centre) while
    keeping the shape. Added 2026-09-13 because the ensemble's level bias
    (IMPROVEMENTS.md A1) is the largest part of the agent's loss to the random
    walk and neither existing lever can express "trust today's price more".
    An empty layer is the identity, so ``v001`` reproduces the agent bit-for-bit.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    centre_gain: dict[int, float] = Field(default_factory=dict)
    width_scale: dict[int, float] = Field(default_factory=dict)
    rw_anchor: dict[int, float] = Field(default_factory=dict)

    def gain_for(self, horizon: int) -> float:
        return self.centre_gain.get(horizon, 1.0)

    def scale_for(self, horizon: int) -> float:
        return self.width_scale.get(horizon, 1.0)

    def anchor_for(self, horizon: int) -> float:
        return self.rw_anchor.get(horizon, 0.0)

    @property
    def is_identity(self) -> bool:
        return all(value == 1.0 for value in (*self.centre_gain.values(), *self.width_scale.values())) and all(
            value == 0.0 for value in self.rw_anchor.values()
        )

    def apply(
        self,
        *,
        ensemble_p50: float,
        final_point_forecast: float,
        final_quantiles: Quantiles,
        horizon: int,
        last_value: float | None = None,
    ) -> tuple[float, Quantiles]:
        """Re-centre, re-scale and anchor one horizon's quantiles. Identity when unset.

        Guards mirror `PythonForecastEngine`: the result must be finite and
        non-crossing, or the calibration is wrong and should fail loudly rather
        than publish a malformed distribution. A non-zero anchor with no
        ``last_value`` raises rather than silently skipping the step.
        """
        gain = self.gain_for(horizon)
        scale = self.scale_for(horizon)
        anchor = self.anchor_for(horizon)
        if gain == 1.0 and scale == 1.0 and anchor == 0.0:
            return final_point_forecast, dict(final_quantiles)

        centre = ensemble_p50 + gain * (final_point_forecast - ensemble_p50)
        adjusted = {level: centre + scale * (value - final_point_forecast) for level, value in final_quantiles.items()}
        if anchor != 0.0:
            if last_value is None:
                raise ValueError(f"rw_anchor={anchor} at horizon {horizon} needs last_value, and none was passed")
            delta = anchor * (last_value - centre)
            centre += delta
            adjusted = {level: value + delta for level, value in adjusted.items()}

        ordered = [adjusted[level] for level in sorted(adjusted)]
        if not all(math.isfinite(value) for value in ordered):
            raise ValueError(f"calibration layer produced a non-finite quantile at horizon {horizon}")
        if ordered != sorted(ordered):
            raise ValueError(f"calibration layer produced crossing quantiles at horizon {horizon}")
        return centre, adjusted


class CalibrationVersion(BaseModel):
    """A dated, versioned set of constants -- the coach's learned artifact.

    ``settings_overlay`` names the ``CfmV50Settings`` fields to override; anything
    absent keeps the package default. Because every version carries
    ``effective_from``, any past run can be replayed under the configuration that
    actually produced it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str
    effective_from: date
    parent_version: str | None = None
    settings_overlay: dict[str, Any] = Field(default_factory=dict)
    layer: CalibrationLayer = Field(default_factory=CalibrationLayer)
    source_proposal_id: str | None = None
    notes: str = ""

    @property
    def is_baseline(self) -> bool:
        """True when this version changes nothing -- the agent runs exactly as shipped."""
        return not self.settings_overlay and self.layer.is_identity


class Candidate(BaseModel):
    """A proposed change to the calibration, awaiting judgment.

    ``fitted_through`` is the load-bearing field: it declares the last origin whose
    outcome was used to *choose* this candidate, and therefore where the holdout
    begins. A candidate fitted on everything has no holdout and cannot pass -- which
    is the intended answer, not an obstacle. ``None`` means nothing was fitted (a
    hand-proposed constant), so every origin is out-of-sample and all of it counts.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str
    parent_version: str
    settings_overlay: dict[str, Any] = Field(default_factory=dict)
    layer: CalibrationLayer = Field(default_factory=CalibrationLayer)
    fitted_through: date | None = None
    rationale: str = ""

    def tunable_names(self, incumbent: "CalibrationVersion") -> set[str]:
        """Which knobs this candidate actually moves, relative to the incumbent.

        Layer entries are counted per parameter rather than per horizon: setting
        ``width_scale`` at h=5/10/21 is one hypothesis about width, not three.
        """
        names = {
            key
            for key in set(self.settings_overlay) | set(incumbent.settings_overlay)
            if self.settings_overlay.get(key) != incumbent.settings_overlay.get(key)
        }
        if self.layer.centre_gain != incumbent.layer.centre_gain:
            names.add("layer.centre_gain")
        if self.layer.width_scale != incumbent.layer.width_scale:
            names.add("layer.width_scale")
        if self.layer.rw_anchor != incumbent.layer.rw_anchor:
            names.add("layer.rw_anchor")
        return names

    def as_version(self, version: str, *, effective_from: date) -> "CalibrationVersion":
        return CalibrationVersion(
            version=version,
            effective_from=effective_from,
            parent_version=self.parent_version,
            settings_overlay=dict(self.settings_overlay),
            layer=self.layer,
            source_proposal_id=self.candidate_id,
            notes=self.rationale,
        )


class ConditionResult(BaseModel):
    """One gate condition and why it did or did not hold."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    passed: bool
    detail: str

    def describe(self) -> str:
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name}: {self.detail}"


class ComparisonVerdict(BaseModel):
    """The locked gate's decision on one candidate.

    ``passed`` is true only when every condition held. Nothing else in the coach may
    construct this with ``passed=True``: the gate is the single place where "this is
    an improvement" can be asserted, so that the standard cannot be relaxed by
    accident somewhere else.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str
    incumbent_version: str
    passed: bool
    conditions: list[ConditionResult]
    metric: str

    n_origins: int = 0
    n_holdout_origins: int = 0
    n_scored_rows: int = 0
    incumbent_score: float | None = None
    candidate_score: float | None = None
    mean_paired_delta: float | None = None
    effect_size_pct: float | None = None
    p_value: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    coverage_incumbent: float | None = None
    coverage_candidate: float | None = None

    #: What to actually apply if a human approves -- the candidate shrunk toward the
    #: incumbent. Never the raw fitted value.
    shrunk_settings_overlay: dict[str, Any] = Field(default_factory=dict)
    shrunk_layer: "CalibrationLayer | None" = None
    shrinkage_factor: float = 1.0

    computed_at: datetime

    @property
    def failed_conditions(self) -> list[ConditionResult]:
        return [item for item in self.conditions if not item.passed]

    def describe(self) -> str:
        head = (
            f"{self.candidate_id} vs {self.incumbent_version}: "
            f"{'PASSED' if self.passed else 'REJECTED'} ({self.metric})"
        )
        if self.mean_paired_delta is not None:
            head += (
                f"\n  delta={self.mean_paired_delta:+.4f} ({self.effect_size_pct:+.1f}%)"
                f"  p={self.p_value:.4f}  90% CI [{self.ci_low:+.4f}, {self.ci_high:+.4f}]"
                f"  holdout origins={self.n_holdout_origins}, rows={self.n_scored_rows}"
            )
        return "\n".join([head, *(f"  {item.describe()}" for item in self.conditions)])


class TrustReport(BaseModel):
    """How much to believe one forecast, computed strictly after it is final (R9)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    tier: TrustTier
    driver: str
    # Every input to the tier, kept so R4 can later test whether the tier
    # actually separates realized error -- and so a tier that does not can be retired.
    signals: dict[str, Any] = Field(default_factory=dict)
    computed_at: datetime


__all__ = [
    "CalibrationLayer",
    "CalibrationVersion",
    "Candidate",
    "ComparisonVerdict",
    "ConditionResult",
    "HorizonRecord",
    "Provenance",
    "Quantiles",
    "ResolvedForecast",
    "RunRecord",
    "ScoreCard",
    "SettingsSource",
    "TrustReport",
    "TrustTier",
]
