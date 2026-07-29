"""Geopolitical factor, scenario, and evidence contracts for CFM Agent v2.1.

Cross-object checks that need the full factor list or the full scenario
list — tier counts, tail-case presence, stance coverage, genuine
disagreement, and point-forecast/scenario-spread consistency — live on
:class:`~energy_oil_forecasting.cfm_agent_v_2_1.outputs.WtiGeoForecastOutput`,
the same placement ``analyst_agent.agent.WtiScenarioForecastOutput`` uses,
because a single factor or scenario object cannot see its siblings.
"""

from __future__ import annotations

from math import isfinite
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class WtiGeoFactor(BaseModel):
    """One core or transitory geopolitical factor identified for this forecast.

    Attributes
    ----------
    name : str
        Short factor label.
    tier : Literal["core", "transitory"]
        ``"core"`` for themes durable enough to plausibly matter in five
        years or more; ``"transitory"`` for situational developments that
        could resolve, reverse, or become irrelevant within months.
    rationale : str
        Why this factor is currently relevant, grounded in retrieved evidence.
    impact_score : Literal["low", "medium", "high"] or None
        Required for transitory factors (magnitude of potential price
        effect, independent of direction); must be omitted for core factors.
    """

    model_config = {"extra": "ignore"}

    name: str = Field(min_length=1, description="Short factor label.")
    tier: Literal["core", "transitory"] = Field(
        description="'core' for durable geopolitical themes; 'transitory' for situational developments."
    )
    rationale: str = Field(min_length=1, description="Why this factor is currently relevant.")
    impact_score: Literal["low", "medium", "high"] | None = Field(
        default=None, description="Required for transitory factors; omit for core factors."
    )

    @model_validator(mode="after")
    def _transitory_factors_have_impact_score(self) -> "WtiGeoFactor":
        """Require an impact score for transitory factors; forbid it for core factors."""
        if self.tier == "transitory" and self.impact_score is None:
            raise ValueError("Transitory factors must set impact_score.")
        if self.tier == "core" and self.impact_score is not None:
            raise ValueError("Core factors must not set impact_score (only transitory factors carry one).")
        return self


class WtiGeoScenario(BaseModel):
    """One named, competing geopolitical scenario with stances against the shared factor set.

    Attributes
    ----------
    name : str
        Short scenario label, e.g. ``"Strait remains open"``.
    stances : dict[str, Literal["bullish", "bearish", "neutral"]]
        This scenario's price-direction stance on each factor in the
        forecast's shared ``factors`` list, keyed by factor name.
    price_low : float
        Lower end of this scenario's implied WTI price range at the
        forecast's longest horizon.
    price_high : float
        Upper end of this scenario's implied price range. Must exceed
        ``price_low`` — a single point value is not a valid range.
    is_tail_case : bool
        ``True`` for the required low-probability, high-impact scenario.
    """

    model_config = {"extra": "ignore"}

    name: str = Field(min_length=1, description="Short scenario name.")
    stances: dict[str, Literal["bullish", "bearish", "neutral"]] = Field(
        description="This scenario's price-direction stance on each shared factor, keyed by factor name."
    )
    price_low: float = Field(description="Lower end of this scenario's implied price range at the longest horizon.")
    price_high: float = Field(description="Upper end of this scenario's implied price range.")
    is_tail_case: bool = Field(
        default=False, description="True for the required low-probability, high-impact scenario."
    )

    @field_validator("price_low", "price_high")
    @classmethod
    def _prices_are_finite(cls, value: float) -> float:
        """Reject NaN and infinite prices."""
        if not isfinite(value):
            raise ValueError("Scenario prices must be finite numbers.")
        return value

    @model_validator(mode="after")
    def _price_range_is_a_genuine_range(self) -> "WtiGeoScenario":
        """Require a real range, not a collapsed point estimate."""
        if self.price_low >= self.price_high:
            raise ValueError(f"price_low ({self.price_low}) must be strictly less than price_high ({self.price_high}).")
        return self


class WtiGeoVerifiedEvidence(BaseModel):
    """One concise, verifier-approved piece of evidence cited by the forecast."""

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


__all__ = ["WtiGeoFactor", "WtiGeoScenario", "WtiGeoVerifiedEvidence"]
