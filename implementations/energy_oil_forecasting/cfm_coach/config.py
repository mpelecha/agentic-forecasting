"""Configuration for the self-contained CFM Coach package.

The coach reads what ``cfm_agent_v_5_0`` produces and proposes calibration
corrections. It never writes to that package: a calibration version is applied
by *injecting* a ``CfmV50Settings`` at construction time, which the agent's own
``build_cfm_agent_config(settings=...)`` already supports.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


COACH_NAME = "cfm_coach"
PACKAGE_ROOT = Path(__file__).resolve().parent
RUNS_ROOT = PACKAGE_ROOT / "runs"
CALIBRATION_ROOT = PACKAGE_ROOT / "calibration"
PROPOSALS_ROOT = PACKAGE_ROOT / "proposals"

TARGET_AGENT = "cfm_agent_v_5_0"
BASELINE_CALIBRATION_VERSION = "v001"

# The prompt version in force while `CfmV50PromptBuilder` is used unmodified.
# M2 (self-audit memory) will introduce a new value here; fits must never pool
# records across prompt versions, because a different prompt is a different agent.
BUILTIN_PROMPT_VERSION = "cfm_v5_0_builtin"

# Bumped whenever `RunRecord`'s shape changes. `RunRecordStore.load_all` refuses
# records it was not written to understand rather than silently mis-reading them.
RUN_RECORD_SCHEMA_VERSION = 1


class CoachSettings(BaseModel):
    """Locked settings for one coach installation.

    Frozen and ``extra="forbid"`` for the same reason ``CfmV50Settings`` is: the
    constants that decide what counts as evidence should not be mutable by the
    thing being evaluated.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_agent: str = TARGET_AGENT
    run_record_schema_version: int = RUN_RECORD_SCHEMA_VERSION

    runs_dir: Path = RUNS_ROOT
    calibration_dir: Path = CALIBRATION_ROOT
    proposals_dir: Path = PROPOSALS_ROOT

    # Task binding the daily run issues. Must match what the agent is asked for.
    target_series_id: str = "wti_crude_oil_price"
    horizons: tuple[int, ...] = (5, 10, 21)
    frequency: str = "B"

    # -- locked comparison gate (stage 1c) ------------------------------------
    # Declared here so one settings object describes the whole coach, even
    # though nothing reads these until the calibration harness lands.
    min_resolved_origins: int = Field(default=12, ge=1)
    min_holdout_origins: int = Field(default=6, ge=1)
    bootstrap_resamples: int = Field(default=2_000, ge=200)
    significance_alpha: float = Field(default=0.10, gt=0.0, lt=1.0)
    min_effect_size_pct: float = Field(default=3.0, ge=0.0)
    shrinkage_factor: float = Field(default=0.50, gt=0.0, le=1.0)
    max_tunables_per_cycle: int = Field(default=1, ge=1)

    # Only live, forward-dated runs are fitting evidence. A run re-executed at a
    # historical cutoff re-searches today's web, so its research packet is shaped
    # by what turned out to matter -- excluded by construction, not by discipline.
    fitting_eligible_provenance: tuple[str, ...] = ("live_forward",)

    # -- trust report (stage 1b) ----------------------------------------------
    # Every threshold is named here rather than buried in `trust.py`, because the
    # tier is an untested hypothesis until R4 can rank it against realized error --
    # and a hypothesis should be easy to read and easy to change.
    ood_percentile: float = Field(default=99.0, gt=50.0, le=100.0)
    analogue_k: int = Field(default=5, ge=1)

    # Comparable days moved this much more than the interval allows for.
    trust_width_ratio_low: float = Field(default=1.5, gt=1.0)
    # ARIMA/Kalman/LightGBM spread, as a fraction of the ensemble P10-P90 width.
    trust_model_disagreement_ratio: float = Field(default=0.25, gt=0.0)


DEFAULT_SETTINGS = CoachSettings()

__all__ = [
    "BASELINE_CALIBRATION_VERSION",
    "BUILTIN_PROMPT_VERSION",
    "CALIBRATION_ROOT",
    "COACH_NAME",
    "DEFAULT_SETTINGS",
    "PACKAGE_ROOT",
    "PROPOSALS_ROOT",
    "RUNS_ROOT",
    "RUN_RECORD_SCHEMA_VERSION",
    "TARGET_AGENT",
    "CoachSettings",
]
