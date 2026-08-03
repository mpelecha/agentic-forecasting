"""Structured, schema-validated score output for CFM Agent v2.2.

This output is deliberately NOT an
:class:`~aieng.forecasting.methods.agentic.outputs.AgentForecastOutput`:
it contains no forecast, no price, and no quantile grid, so it has no
``to_predictions`` and does not plug into the backtest harness as a
predictor. The division of labor (Kam's Job 1 / Job 2 split):

- Job 1 — assess context. The LLM reads news, sorts it into scored
  factors and probability-weighted scenarios. That is this package.
- Job 2 — turn context into a number. A calibration layer (plain code,
  fit against quant-baseline residuals, versioned and backtested) maps
  scores to price effects. That layer consumes ``calibration_features()``
  below and lives outside this package.

Factor tier counts are 2-4 core and 1-3 transitory — wider than
``cfm_agent_v_2_1``'s geopolitical-only bounds, because the scope now
spans five news-visible categories and simultaneous transitory events
(a hurricane during a shipping disruption) are common.
"""

from __future__ import annotations

import json
from math import isclose
from typing import get_args

from energy_oil_forecasting.cfm_agent_v_2_2.schemas import (
    EventCategory,
    WtiEventFactor,
    WtiEventScenario,
    WtiEventVerifiedEvidence,
)
from pydantic import BaseModel, Field, model_validator


# Scenario probabilities must sum to 1 within this absolute tolerance.
_PROBABILITY_SUM_TOLERANCE = 0.02

# A tail case must be genuinely high-impact: |impact_score| at or above this.
_MIN_TAIL_IMPACT = 2


class WtiEventScoreOutput(BaseModel):
    """Scored event context for WTI: factors, scenarios, evidence — no prices.

    Attributes
    ----------
    factors : list[WtiEventFactor]
        2-4 core factors and 1-3 transitory factors, identified once.
    scenarios : list[WtiEventScenario]
        2 or more named, competing scenarios with probabilities that sum
        to 1. At least one tail case; at least one non-tail scenario; at
        least two scenarios must differ in stance on two or more factors.
    verified_evidence : list[WtiEventVerifiedEvidence]
        Verifier-approved sources the factors cite by index.
    """

    model_config = {"extra": "ignore"}

    factors: list[WtiEventFactor] = Field(
        description="The shared factor set: 2-4 core and 1-3 transitory, identified once."
    )
    scenarios: list[WtiEventScenario] = Field(
        min_length=2,
        description="2-3 named, competing scenarios; probabilities sum to 1.",
    )
    verified_evidence: list[WtiEventVerifiedEvidence] = Field(default_factory=list)
    research_summary: str = ""
    overall_rationale: str = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _factor_names_are_unique(self) -> "WtiEventScoreOutput":
        """Reject duplicate factor names — stances key on them."""
        names = [factor.name for factor in self.factors]
        if len(names) != len(set(names)):
            raise ValueError("Factor names must be unique.")
        return self

    @model_validator(mode="after")
    def _factor_tier_counts_are_valid(self) -> "WtiEventScoreOutput":
        """Require 2-4 core factors and 1-3 transitory factors."""
        core = [factor for factor in self.factors if factor.tier == "core"]
        transitory = [factor for factor in self.factors if factor.tier == "transitory"]
        if not (2 <= len(core) <= 4):
            raise ValueError(f"Expected 2-4 core factors, got {len(core)}.")
        if not (1 <= len(transitory) <= 3):
            raise ValueError(f"Expected 1-3 transitory factors, got {len(transitory)}.")
        return self

    @model_validator(mode="after")
    def _evidence_indices_are_in_range(self) -> "WtiEventScoreOutput":
        """Reject a factor evidence index that does not point into verified_evidence."""
        for factor in self.factors:
            if any(index < 0 or index >= len(self.verified_evidence) for index in factor.evidence_indices):
                raise ValueError(f"Factor '{factor.name}' has an out-of-range evidence index.")
        return self

    @model_validator(mode="after")
    def _scenario_probabilities_sum_to_one(self) -> "WtiEventScoreOutput":
        """Require scenario probabilities to sum to 1 within tolerance."""
        total = sum(scenario.probability for scenario in self.scenarios)
        if not isclose(total, 1.0, abs_tol=_PROBABILITY_SUM_TOLERANCE):
            raise ValueError(f"Scenario probabilities must sum to 1, got {total:.3f}.")
        return self

    @model_validator(mode="after")
    def _scenarios_include_tail_and_non_tail(self) -> "WtiEventScoreOutput":
        """Require at least one tail case and at least one non-tail scenario."""
        tails = [scenario for scenario in self.scenarios if scenario.is_tail_case]
        non_tails = [scenario for scenario in self.scenarios if not scenario.is_tail_case]
        if not tails:
            raise ValueError("At least one scenario must set is_tail_case=True.")
        if not non_tails:
            raise ValueError("At least one scenario must be a non-tail (mainline) case.")

        max_non_tail_probability = max(scenario.probability for scenario in non_tails)
        for tail in tails:
            if tail.probability > max_non_tail_probability:
                raise ValueError(
                    f"Tail scenario '{tail.name}' has probability {tail.probability:.2f}, higher than "
                    f"every non-tail scenario — a tail case must be low-probability."
                )
            if abs(tail.impact_score) < _MIN_TAIL_IMPACT:
                raise ValueError(
                    f"Tail scenario '{tail.name}' has |impact_score| {abs(tail.impact_score)} — "
                    f"a tail case must be high-impact (|impact_score| >= {_MIN_TAIL_IMPACT})."
                )
        return self

    @model_validator(mode="after")
    def _scenario_stances_cover_every_factor(self) -> "WtiEventScoreOutput":
        """Require each scenario's stances to cover exactly the shared factor names."""
        expected = {factor.name for factor in self.factors}
        for scenario in self.scenarios:
            actual = set(scenario.stances)
            if actual != expected:
                missing = sorted(expected - actual)
                extra = sorted(actual - expected)
                raise ValueError(
                    f"Scenario '{scenario.name}' stances must cover exactly the shared factors. "
                    f"Missing: {missing}; extra: {extra}."
                )
        return self

    @model_validator(mode="after")
    def _scenarios_genuinely_disagree(self) -> "WtiEventScoreOutput":
        """Require at least two scenarios to differ in stance on at least two shared factors."""
        factor_names = [factor.name for factor in self.factors]
        for i in range(len(self.scenarios)):
            for j in range(i + 1, len(self.scenarios)):
                first, second = self.scenarios[i], self.scenarios[j]
                differences = sum(1 for name in factor_names if first.stances.get(name) != second.stances.get(name))
                if differences >= 2:
                    return self
        raise ValueError(
            "No two scenarios differ in stance on at least two shared factors — "
            "scenarios must genuinely disagree, not just differ in tone."
        )

    @model_validator(mode="after")
    def _scenarios_are_not_duplicates(self) -> "WtiEventScoreOutput":
        """Reject scenarios identical in both stances and conditional impact.

        Same stances alone are allowed — an intensity ladder (de-escalation /
        sustained / escalation) legitimately shares stances and differs only
        in impact_score. Identical on both axes means the scenario is a
        reworded copy, not a different storyline.
        """
        seen: dict[tuple, str] = {}
        for scenario in self.scenarios:
            key = (tuple(sorted(scenario.stances.items())), scenario.impact_score)
            if key in seen:
                raise ValueError(
                    f"Scenarios '{seen[key]}' and '{scenario.name}' have identical stances and "
                    "identical impact_score — differentiate the storylines by stance or by "
                    "conditional impact."
                )
            seen[key] = scenario.name
        return self

    def calibration_features(self) -> dict[str, float]:
        """Return the flat numeric features the calibration layer consumes.

        The calibration layer regresses these against quant-baseline
        residuals (actual move minus the quant pillar's expected move) to
        learn a score-to-price mapping. Keep this row format stable:
        every stored row feeds the future fit.
        """
        core = [factor for factor in self.factors if factor.tier == "core"]
        transitory = [factor for factor in self.factors if factor.tier == "transitory"]
        tails = [scenario for scenario in self.scenarios if scenario.is_tail_case]

        features: dict[str, float] = {
            "net_core_score": float(sum(factor.impact_score for factor in core)),
            "net_transitory_score": float(sum(factor.impact_score for factor in transitory)),
            "confidence_weighted_core_score": float(
                sum(factor.impact_score * factor.confidence for factor in core)
            ),
            "confidence_weighted_transitory_score": float(
                sum(factor.impact_score * factor.confidence for factor in transitory)
            ),
            "expected_scenario_impact": float(
                sum(scenario.probability * scenario.impact_score for scenario in self.scenarios)
            ),
            "tail_probability": float(sum(tail.probability for tail in tails)),
            "tail_impact_score": float(max((tail.impact_score for tail in tails), key=abs)),
        }
        for category in get_args(EventCategory):
            features[f"score_{category}"] = float(
                sum(factor.impact_score for factor in self.factors if factor.category == category)
            )
        return features

    @classmethod
    def prompt_schema_json(cls) -> str:
        """Return a JSON template for use in agent instruction strings."""
        template: dict[str, object] = {
            "factors": [
                {
                    "name": "<string>",
                    "category": (
                        "<'geopolitical' | 'weather' | 'operational' | 'policy' | 'demand_expectations'>"
                    ),
                    "tier": "<'core' | 'transitory'>",
                    "rationale": "<string>",
                    "impact_score": "<integer -3..+3; sign is price direction, + = up; core may be 0, transitory must not>",
                    "confidence": "<float 0..1 — evidence quality behind the score>",
                    "evidence_indices": ["<zero-based index into verified_evidence>"],
                }
            ],
            "scenarios": [
                {
                    "name": "<string>",
                    "stances": {"<factor name>": "<'bullish' | 'bearish' | 'neutral'>"},
                    "probability": "<float in (0, 1]; the set must sum to 1>",
                    "impact_score": "<integer -3..+3: net price pressure if this scenario plays out>",
                    "is_tail_case": "<true for the low-probability, high-impact scenario (|impact_score| >= 2)>",
                    "rationale": "<one or two sentences>",
                }
            ],
            "verified_evidence": [
                {
                    "title": "<source title>",
                    "source_url": "<URL>",
                    "claim": "<verified claim>",
                    "forecast_effect": (
                        "<center_up|center_down|uncertainty_wider|uncertainty_narrower|context_only>"
                    ),
                }
            ],
            "research_summary": "<concise summary>",
            "overall_rationale": "<string>",
            "warnings": ["<warning strings>"],
        }
        return json.dumps(template, indent=2)


__all__ = ["WtiEventScoreOutput"]
