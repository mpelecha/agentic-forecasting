"""Locked settings for the review agent.

Frozen and ``extra="forbid"`` like `CoachSettings`: the constants that decide
what counts as evidence must not be mutable by the thing being evaluated.

Two facts about the Vector proxy shape this module:

- ``reasoning_effort`` values ``disable`` and ``low`` are rejected with a 400
  for Gemini models (``llm_processes/base.py``); the valid set is
  ``minimal`` / ``medium`` / ``high``. HANDOFF.md:457 saw a 400 on *any*
  value for ``gemini-3.5-flash``, so the client probes once per run and falls
  back to sending nothing.
- litellm's cost map has no ``openai/gemini-3.5-flash`` entry, so the proxy
  route reports ``response_cost`` of zero. The price table below is therefore
  the *default* cost source, not a fallback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from aieng.forecasting.models import ADVANCED_MODEL, LITE_MODEL
from energy_oil_forecasting.cfm_coach.config import PACKAGE_ROOT, PROPOSALS_ROOT
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


REVIEW_ROOT = PACKAGE_ROOT / "review"

#: The only ``reasoning_effort`` values the proxy accepts for Gemini.
ReasoningEffort = Literal["minimal", "medium", "high"]

#: Judgment calls run at the model default; lookup calls are pinned.
CallClass = Literal["judgment", "lookup"]

Stage = Literal["triage", "deep_dive", "synthesis", "critic", "lookup", "probe"]

#: Mirrors `report.MIN_EFFECTIVE_N_FOR_R1`; asserted equal in tests so the two
#: cannot drift apart silently (report.py imports the data layer, so it is not
#: imported here).
MIN_EFFECTIVE_N_POWERED = 10.0


class StageLlm(BaseModel):
    """How one stage calls the model. Stages differ by effort and budget, never by model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    call_class: CallClass
    reasoning_effort: ReasoningEffort | None = None
    max_tokens: int = Field(ge=256)


class ReviewSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # -- the model ------------------------------------------------------------
    model: str = ADVANCED_MODEL
    judgment_temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    lookup_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    timeout_seconds: float = Field(default=120.0, gt=0.0)
    max_concurrency: int = Field(default=4, ge=1)
    transient_retry_attempts: int = Field(default=3, ge=1)

    triage: StageLlm = StageLlm(call_class="judgment", reasoning_effort="minimal", max_tokens=8_192)
    deep_dive: StageLlm = StageLlm(call_class="judgment", reasoning_effort="medium", max_tokens=8_192)
    synthesis: StageLlm = StageLlm(call_class="judgment", reasoning_effort="high", max_tokens=16_384)
    critic: StageLlm = StageLlm(call_class="judgment", reasoning_effort="high", max_tokens=8_192)
    lookup: StageLlm = StageLlm(call_class="lookup", reasoning_effort=None, max_tokens=4_096)
    probe: StageLlm = StageLlm(call_class="lookup", reasoning_effort=None, max_tokens=512)

    # -- cost -----------------------------------------------------------------
    # litellm's `gemini-3.5-flash` entry at planning time (2026-09-13).
    price_usd_per_million_input: float = Field(default=1.50, ge=0.0)
    price_usd_per_million_output: float = Field(default=9.00, ge=0.0)
    weekly_budget_usd: float = Field(default=5.0, gt=0.0)
    bootstrap_budget_usd: float = Field(default=10.0, gt=0.0)

    # -- cadence --------------------------------------------------------------
    triage_after_horizons: tuple[int, ...] = (5, 10)
    rescore_at_horizons: tuple[int, ...] = (21,)
    deep_dive_k: int = Field(default=2, ge=0)
    deep_dive_max_turns: int = Field(default=8, ge=1)
    blind_retriage_fraction: float = Field(default=0.20, ge=0.0, le=1.0)

    # -- evidence bar: LLM track ---------------------------------------------
    min_independent_windows: int = Field(default=3, ge=1)
    min_episodes: int = Field(default=2, ge=1)
    min_process_cutoffs: int = Field(default=4, ge=1)
    min_support_ratio: float = Field(default=0.7, gt=0.0, le=1.0)
    contradiction_ratio: float = Field(default=0.5, gt=0.0, le=1.0)
    stale_after_reviews: int = Field(default=4, ge=1)
    retire_after_stale: int = Field(default=8, ge=1)
    reproposal_new_cutoffs: int = Field(default=3, ge=1)

    # -- evidence bar: numeric track -----------------------------------------
    base_dominated_share: float = Field(default=0.70, gt=0.5, le=1.0)
    overlay_implicated_share: float = Field(default=0.50, gt=0.0, le=1.0)
    min_effective_n_powered: float = MIN_EFFECTIVE_N_POWERED
    pairing_min_cutoffs: int = Field(default=12, ge=2)
    #: Settings fields a numeric candidate may never name: they change the run,
    #: not the arithmetic, so a replay cannot price them.
    overlay_denylist: tuple[str, ...] = (
        "*_model",
        "audit_enabled",
        "code_execution_enabled",
        "policy_mode",
        "max_pre_execution_attempts",
        "research_*",
        "source_resolution_*",
        "*_timeout_seconds",
        "search_verifier_*",
        "max_data_rows_per_series",
    )

    # -- challenger ------------------------------------------------------------
    challenger_model: str = ADVANCED_MODEL
    challenger_runs_per_day: int = Field(default=3, ge=1)

    # -- storage ----------------------------------------------------------------
    data_dir: Path = REVIEW_ROOT
    proposals_dir: Path = PROPOSALS_ROOT

    # -- validation -------------------------------------------------------------

    @field_validator("model", "challenger_model")
    @classmethod
    def _not_the_lite_model(cls, value: str) -> str:
        if value == LITE_MODEL or "lite" in value.lower():
            raise ValueError(f"the review coach must not run on the lite model (got {value!r})")
        return value

    @model_validator(mode="after")
    def _lookup_stages_are_pinned(self) -> ReviewSettings:
        for name in ("lookup", "probe"):
            if getattr(self, name).call_class != "lookup":
                raise ValueError(f"stage {name!r} must be a lookup call")
        return self

    # -- helpers ----------------------------------------------------------------

    def stage(self, name: Stage) -> StageLlm:
        return getattr(self, name)

    def temperature_for(self, name: Stage) -> float:
        return self.judgment_temperature if self.stage(name).call_class == "judgment" else self.lookup_temperature

    # Data layout, in one place so tests can point everything at ``tmp_path``.
    @property
    def cards_dir(self) -> Path:
        return self.data_dir / "cards"

    @property
    def annotations_dir(self) -> Path:
        return self.data_dir / "annotations"

    @property
    def hypotheses_path(self) -> Path:
        return self.data_dir / "hypotheses.jsonl"

    @property
    def weeks_dir(self) -> Path:
        return self.data_dir / "weeks"

    @property
    def calibration_dir(self) -> Path:
        return self.data_dir / "calibration"

    @property
    def candidates_dir(self) -> Path:
        return self.data_dir / "candidates"

    @property
    def coached_dir(self) -> Path:
        return self.data_dir / "coached"

    @property
    def challengers_dir(self) -> Path:
        return self.data_dir / "challengers"

    @property
    def state_path(self) -> Path:
        return self.data_dir / "state.json"


DEFAULT_REVIEW_SETTINGS = ReviewSettings()

__all__ = [
    "DEFAULT_REVIEW_SETTINGS",
    "MIN_EFFECTIVE_N_POWERED",
    "REVIEW_ROOT",
    "CallClass",
    "ReasoningEffort",
    "ReviewSettings",
    "Stage",
    "StageLlm",
]
