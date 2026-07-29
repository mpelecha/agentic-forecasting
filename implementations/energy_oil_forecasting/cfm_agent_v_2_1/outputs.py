"""Structured, schema-validated output contract for CFM Agent v2.1.

There is no quant tool in this package, so unlike
``cfm_agent_v_2_0.outputs.CfmRichForecastOutput`` there is no
``component_models`` field and no authoritative-tool-truth merge step.
The trust mechanism here is the same one ``analyst_agent.agent.
WtiScenarioForecastOutput`` uses: the model must state named, competing
scenarios, and the final forecast is rejected — not silently accepted —
if it is not consistent with those scenarios.

Design note on the spread check
--------------------------------
``analyst_agent.agent.WtiScenarioForecastOutput`` checks the point forecast
against a *probability-weighted* scenario price, because its scenarios
carry a ``probability`` field. ``WtiGeoScenario`` in this package has no
probability field (the caller's schema spec omits it), so a weighted
average is not available. Instead, ``_final_forecast_matches_scenario_spread``
below checks two weaker but still meaningful things at the longest
horizon: (1) the point forecast falls within the union of every scenario's
stated price range, and (2) the 90% interval (0.05-0.95) is not narrower
than a floor fraction of the scenario spread. This is a deliberate
substitution, not a literal port — flagged here so a future reader does
not assume it is numerically identical to the analyst_agent check.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from math import isclose
from typing import Any, ClassVar, Literal

import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import (
    STANDARD_QUANTILES,
    ContinuousForecast,
    Prediction,
)
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic.outputs import (
    AgentForecastOutput,
    AgentQuantileForecast,
)
from energy_oil_forecasting.cfm_agent_v_2_1.schemas import (
    WtiGeoFactor,
    WtiGeoScenario,
    WtiGeoVerifiedEvidence,
)
from pydantic import BaseModel, Field, model_validator


# Floor on the 90% interval width, as a fraction of the scenario price
# spread. See the "Design note" above for why this replaces a
# probability-weighted check.
_SCENARIO_SPREAD_TOLERANCE = 0.15

_MIN_QUANTILE = min(STANDARD_QUANTILES)
_MAX_QUANTILE = max(STANDARD_QUANTILES)


class WtiGeoHorizonForecast(BaseModel):
    """Final forecast for one horizon, with evidence references and rationale."""

    model_config = {"extra": "ignore"}

    horizon: int = Field(ge=1)
    point_forecast: float
    quantiles: list[AgentQuantileForecast]
    evidence_indices: list[int] = Field(default_factory=list)
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_quantile_contract(self) -> "WtiGeoHorizonForecast":
        """Require the standard grid, non-decreasing, with p50 equal to point_forecast."""
        by_level = {item.quantile: item.value for item in self.quantiles}
        if set(by_level) != set(STANDARD_QUANTILES):
            raise ValueError("forecast must include exactly the standard quantile levels.")

        values = [by_level[q] for q in STANDARD_QUANTILES]
        if values != sorted(values):
            raise ValueError("quantiles must be non-decreasing.")

        if not isclose(self.point_forecast, by_level[0.5], rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("point_forecast must equal the 0.5 quantile.")

        return self

    def quantile_dict(self) -> dict[float, float]:
        """Return quantiles in the standard order."""
        by_level = {item.quantile: item.value for item in self.quantiles}
        return {q: by_level[q] for q in STANDARD_QUANTILES}


class WtiGeoForecastOutput(AgentForecastOutput):
    """Geopolitical-only forecast with a required, structured scenario decomposition.

    Attributes
    ----------
    forecasts : list[WtiGeoHorizonForecast]
        One entry per requested horizon.
    factors : list[WtiGeoFactor]
        2-5 core factors and 1-2 transitory factors, identified once for
        the whole forecast.
    scenarios : list[WtiGeoScenario]
        2 or more named, competing scenarios. At least one must set
        ``is_tail_case=True``, and at least two scenarios must differ in
        their stance on at least two shared factors.
    """

    modality: ClassVar[Literal["continuous", "discrete", "categorical"]] = "continuous"
    model_config = {"extra": "ignore"}

    forecasts: list[WtiGeoHorizonForecast]
    factors: list[WtiGeoFactor] = Field(
        description="The shared core/transitory geopolitical factor set, identified once."
    )
    scenarios: list[WtiGeoScenario] = Field(
        min_length=2,
        description="2-3 named, competing scenarios, each tagging the shared factor set.",
    )
    verified_evidence: list[WtiGeoVerifiedEvidence] = Field(default_factory=list)
    research_summary: str = ""
    overall_rationale: str = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _forecasts_are_present_and_unique(self) -> "WtiGeoForecastOutput":
        """Reject an empty forecast list or duplicate horizons."""
        horizons = [forecast.horizon for forecast in self.forecasts]
        if not horizons:
            raise ValueError("forecasts cannot be empty.")
        if len(horizons) != len(set(horizons)):
            raise ValueError("forecast horizons must be unique.")
        return self

    @model_validator(mode="after")
    def _evidence_indices_are_in_range(self) -> "WtiGeoForecastOutput":
        """Reject an evidence_indices entry that does not point into verified_evidence."""
        for forecast in self.forecasts:
            if any(index < 0 or index >= len(self.verified_evidence) for index in forecast.evidence_indices):
                raise ValueError("evidence_indices contains an out-of-range index.")
        return self

    @model_validator(mode="after")
    def _factor_tier_counts_are_valid(self) -> "WtiGeoForecastOutput":
        """Require 2-5 core factors and 1-2 transitory factors."""
        core = [factor for factor in self.factors if factor.tier == "core"]
        transitory = [factor for factor in self.factors if factor.tier == "transitory"]
        if not (2 <= len(core) <= 5):
            raise ValueError(f"Expected 2-5 core factors, got {len(core)}.")
        if not (1 <= len(transitory) <= 2):
            raise ValueError(f"Expected 1-2 transitory factors, got {len(transitory)}.")
        return self

    @model_validator(mode="after")
    def _scenarios_include_a_tail_case(self) -> "WtiGeoForecastOutput":
        """Require at least one scenario explicitly marked as the tail case."""
        if not any(scenario.is_tail_case for scenario in self.scenarios):
            raise ValueError("At least one scenario must set is_tail_case=True.")
        return self

    @model_validator(mode="after")
    def _scenario_stances_cover_every_factor(self) -> "WtiGeoForecastOutput":
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
    def _scenarios_genuinely_disagree(self) -> "WtiGeoForecastOutput":
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
    def _final_forecast_matches_scenario_spread(self) -> "WtiGeoForecastOutput":
        """Require the longest horizon's forecast to fall inside the scenario spread.

        See the module docstring's "Design note" for why this checks range
        membership and interval width instead of a probability-weighted
        price — ``WtiGeoScenario`` has no probability field.
        """
        scenario_low = min(scenario.price_low for scenario in self.scenarios)
        scenario_high = max(scenario.price_high for scenario in self.scenarios)

        max_horizon = max(forecast.horizon for forecast in self.forecasts)
        longest = next(forecast for forecast in self.forecasts if forecast.horizon == max_horizon)

        if not (scenario_low <= longest.point_forecast <= scenario_high):
            raise ValueError(
                f"point_forecast ({longest.point_forecast:.2f}) at horizon {max_horizon} falls outside "
                f"the scenario price spread [{scenario_low:.2f}, {scenario_high:.2f}]. "
                "The model's point forecast is inconsistent with its own stated scenarios."
            )

        by_level = longest.quantile_dict()
        interval_width = by_level[_MAX_QUANTILE] - by_level[_MIN_QUANTILE]
        scenario_spread = scenario_high - scenario_low
        floor_width = scenario_spread * _SCENARIO_SPREAD_TOLERANCE
        if interval_width < floor_width:
            raise ValueError(
                f"The {_MIN_QUANTILE:.0%}-{_MAX_QUANTILE:.0%} interval at horizon {max_horizon} is "
                f"{interval_width:.2f} wide, narrower than the required floor of {floor_width:.2f} "
                f"({_SCENARIO_SPREAD_TOLERANCE:.0%} of the {scenario_spread:.2f}-wide scenario spread). "
                "The stated uncertainty is too narrow given the model's own scenarios."
            )

        return self

    @classmethod
    def prompt_schema_json(cls) -> str:
        """Return a JSON template for use in agent instruction strings."""
        quantile_entries = [{"quantile": float(q), "value": "<float>"} for q in STANDARD_QUANTILES]
        template: dict[str, object] = {
            "forecasts": [
                {
                    "horizon": "<integer — one entry per horizon from the task>",
                    "point_forecast": "<float — must equal the 0.50 quantile value>",
                    "quantiles": quantile_entries,
                    "evidence_indices": ["<zero-based index into verified_evidence>"],
                    "rationale": "<string>",
                }
            ],
            "factors": [
                {
                    "name": "<string>",
                    "tier": "<'core' | 'transitory'>",
                    "rationale": "<string>",
                    "impact_score": "<'low' | 'medium' | 'high' — required for transitory, omit for core>",
                }
            ],
            "scenarios": [
                {
                    "name": "<string>",
                    "stances": {"<factor name>": "<'bullish' | 'bearish' | 'neutral'>"},
                    "price_low": "<float>",
                    "price_high": "<float — must be strictly greater than price_low>",
                    "is_tail_case": "<true for exactly one low-probability/high-impact scenario>",
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

    def to_predictions(
        self,
        *,
        task: ForecastingTask,
        context: ForecastContext,
        predictor_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Prediction]:
        """Convert to harness-compatible predictions, stamping factors/scenarios onto metadata."""
        by_horizon = {forecast.horizon: forecast for forecast in self.forecasts}
        if set(by_horizon) != set(task.horizons):
            raise ValueError("output must contain exactly the requested horizons.")

        issued_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        offset = pd.tseries.frequencies.to_offset(task.frequency)
        evidence = [item.model_dump(mode="json") for item in self.verified_evidence]
        factors = [factor.model_dump(mode="json") for factor in self.factors]
        scenarios = [scenario.model_dump(mode="json") for scenario in self.scenarios]
        predictions: list[Prediction] = []

        for horizon in task.horizons:
            forecast = by_horizon[horizon]
            prediction_metadata = dict(metadata or {})
            prediction_metadata.update(
                {
                    "factors": factors,
                    "scenarios": scenarios,
                    "verified_evidence": evidence,
                    "evidence_indices": forecast.evidence_indices,
                    "research_summary": self.research_summary,
                    "overall_rationale": self.overall_rationale,
                    "rationale": forecast.rationale,
                    "warnings": list(self.warnings),
                }
            )
            predictions.append(
                Prediction(
                    predictor_id=predictor_id,
                    task_id=task.task_id,
                    issued_at=issued_at,
                    as_of=context.as_of,
                    forecast_date=(pd.Timestamp(context.as_of) + offset * horizon).to_pydatetime(),
                    payload=ContinuousForecast(
                        point_forecast=forecast.point_forecast,
                        quantiles=forecast.quantile_dict(),
                    ),
                    metadata=prediction_metadata,
                )
            )

        return predictions


__all__ = ["WtiGeoForecastOutput", "WtiGeoHorizonForecast"]
