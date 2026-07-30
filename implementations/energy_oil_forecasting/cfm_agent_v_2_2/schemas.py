"""Event factor, scenario, and evidence contracts for CFM Agent v2.2.

CFM Agent v2.2 speaks in scores, never in prices. Each factor and scenario
carries a signed, bounded ``impact_score`` (-3 to +3); a separate,
deterministic calibration layer — not this package, and not an LLM — maps
scores to price effects. No schema in this module has a price field.

The scope is wider than ``cfm_agent_v_2_1``: not geopolitics only, but
every driver that is visible only through news and text — anything the
quant pillar (price/financial covariates) and the physical pillar (EIA
supply/demand data) cannot see. Each factor declares which ``category``
it belongs to so the calibration layer can fit per-category coefficients.

Cross-object checks that need the full factor or scenario list live on
:class:`~energy_oil_forecasting.cfm_agent_v_2_2.outputs.WtiEventScoreOutput`.
"""

from __future__ import annotations

from math import isfinite
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


EventCategory = Literal[
    "geopolitical",
    "weather",
    "operational",
    "policy",
    "demand_expectations",
]


class WtiEventFactor(BaseModel):
    """One news-visible driver, scored but never priced.

    Attributes
    ----------
    name : str
        Short factor label.
    category : EventCategory
        Which news-visible channel the factor belongs to. The calibration
        layer uses this to fit per-category score-to-price coefficients.
    tier : Literal["core", "transitory"]
        ``"core"`` for durable themes (plausibly relevant in five years or
        more); ``"transitory"`` for situational developments that could
        resolve, reverse, or become irrelevant within months.
    rationale : str
        Why this factor is currently relevant, grounded in retrieved evidence.
    impact_score : int
        Signed score from -3 to +3. Sign is price direction (+ = upward
        pressure on WTI); magnitude is strength. Core factors may score 0
        (a dormant theme still worth tracking); transitory factors must
        not — a transitory factor with no impact should not be listed.
    confidence : float
        0 to 1. How well the retrieved evidence supports the score's sign
        and magnitude, not how likely the factor is to matter.
    evidence_indices : list[int]
        Zero-based indices into the output's ``verified_evidence`` list.
    """

    model_config = {"extra": "ignore"}

    name: str = Field(min_length=1, description="Short factor label.")
    category: EventCategory = Field(
        description="News-visible channel: geopolitical, weather, operational, policy, or demand_expectations."
    )
    tier: Literal["core", "transitory"] = Field(
        description="'core' for durable themes; 'transitory' for situational developments."
    )
    rationale: str = Field(min_length=1, description="Why this factor is currently relevant.")
    impact_score: int = Field(
        ge=-3,
        le=3,
        description="Signed score -3..+3. Sign is price direction (+ = up); magnitude is strength.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="0..1: how well the evidence supports the score's sign and magnitude.",
    )
    evidence_indices: list[int] = Field(
        default_factory=list,
        description="Zero-based indices into verified_evidence.",
    )

    @field_validator("confidence")
    @classmethod
    def _confidence_is_finite(cls, value: float) -> float:
        """Reject NaN and infinite confidence values."""
        if not isfinite(value):
            raise ValueError("confidence must be a finite number.")
        return value

    @model_validator(mode="after")
    def _transitory_factors_have_nonzero_impact(self) -> "WtiEventFactor":
        """Require a nonzero score for transitory factors; core factors may be dormant (0)."""
        if self.tier == "transitory" and self.impact_score == 0:
            raise ValueError(
                "Transitory factors must have a nonzero impact_score — "
                "a transitory factor with no impact should not be listed."
            )
        return self


class WtiEventScenario(BaseModel):
    """One named, competing scenario with a probability and a conditional score.

    Attributes
    ----------
    name : str
        Short scenario label, e.g. ``"Disruption clears"``.
    stances : dict[str, Literal["bullish", "bearish", "neutral"]]
        This scenario's price-direction stance on each factor in the
        output's shared ``factors`` list, keyed by factor name.
    probability : float
        Probability of this scenario, in (0, 1]. Probabilities across the
        scenario set must sum to 1 (checked on the output).
    impact_score : int
        Signed score from -3 to +3: the net price pressure if this
        scenario plays out. Not a price — the calibration layer prices it.
    is_tail_case : bool
        ``True`` for the required low-probability, high-impact scenario.
    rationale : str
        One or two sentences describing the scenario's storyline.
    """

    model_config = {"extra": "ignore"}

    name: str = Field(min_length=1, description="Short scenario name.")
    stances: dict[str, Literal["bullish", "bearish", "neutral"]] = Field(
        description="Price-direction stance on each shared factor, keyed by factor name."
    )
    probability: float = Field(
        gt=0.0,
        le=1.0,
        description="Scenario probability in (0, 1]; the set must sum to 1.",
    )
    impact_score: int = Field(
        ge=-3,
        le=3,
        description="Signed score -3..+3: net price pressure if this scenario plays out.",
    )
    is_tail_case: bool = Field(
        default=False, description="True for the required low-probability, high-impact scenario."
    )
    rationale: str = Field(min_length=1, description="Short storyline for this scenario.")

    @field_validator("probability")
    @classmethod
    def _probability_is_finite(cls, value: float) -> float:
        """Reject NaN and infinite probabilities."""
        if not isfinite(value):
            raise ValueError("probability must be a finite number.")
        return value


class WtiEventVerifiedEvidence(BaseModel):
    """One concise, verifier-approved piece of evidence cited by the scores."""

    model_config = {"extra": "ignore"}

    title: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    forecast_effect: Literal[
        "center_up",
        "center_down",
        "uncertainty_wider",
        "uncertainty_narrower",
        "context_only",
    ]


__all__ = [
    "EventCategory",
    "WtiEventFactor",
    "WtiEventScenario",
    "WtiEventVerifiedEvidence",
]
