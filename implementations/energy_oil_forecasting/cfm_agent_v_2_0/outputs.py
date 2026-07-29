"""Rich, auditable output contract for CFM Agent v2.0."""

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
from pydantic import BaseModel, Field, model_validator


_REQUIRED_COMPONENTS = {"arima", "kalman", "lightgbm", "ensemble"}


class CfmComponentForecast(BaseModel):
    """Agent-reported copy of one component forecast for one horizon."""

    model_config = {"extra": "ignore"}

    status: Literal["ok", "error"]
    point_forecast: float | None = None
    quantiles: list[AgentQuantileForecast] = Field(default_factory=list)
    error: str | None = None

    @model_validator(mode="after")
    def validate_status_contract(self) -> "CfmComponentForecast":
        """Require complete forecasts for success and an error message for failure."""
        if self.status == "error":
            if not self.error or not self.error.strip():
                raise ValueError("failed components must include an error.")
            return self

        if self.point_forecast is None:
            raise ValueError("successful components must include point_forecast.")

        by_level = {item.quantile: item.value for item in self.quantiles}
        if set(by_level) != set(STANDARD_QUANTILES):
            raise ValueError("successful components must include all standard quantiles.")

        values = [by_level[q] for q in STANDARD_QUANTILES]
        if values != sorted(values):
            raise ValueError("component quantiles must be non-decreasing.")

        if not isclose(self.point_forecast, by_level[0.5], rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("component point_forecast must equal p50.")

        return self

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation for prediction metadata."""
        return {
            "status": self.status,
            "point_forecast": self.point_forecast,
            "quantiles": {str(item.quantile): item.value for item in self.quantiles},
            "error": self.error,
        }


class CfmVerifiedEvidence(BaseModel):
    """Concise verifier-approved evidence cited by the final forecast."""

    model_config = {"extra": "ignore"}

    title: str
    source_url: str
    claim: str
    forecast_effect: Literal[
        "center_up",
        "center_down",
        "uncertainty_wider",
        "uncertainty_narrower",
        "context_only",
    ]


class CfmRichHorizonForecast(BaseModel):
    """Final forecast plus component attribution for one horizon."""

    model_config = {"extra": "ignore"}

    horizon: int = Field(ge=1)
    point_forecast: float
    quantiles: list[AgentQuantileForecast]
    component_models: dict[str, CfmComponentForecast]
    model_disagreement_std: float = Field(ge=0.0)
    included_models: list[str] = Field(default_factory=list)
    discounted_models: list[str] = Field(default_factory=list)
    ensemble_to_final_adjustment: float
    model_selection_rationale: str
    final_forecast_rationale: str
    evidence_indices: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_horizon_contract(self) -> "CfmRichHorizonForecast":
        """Validate quantiles, components, and ensemble adjustment arithmetic."""
        by_level = {item.quantile: item.value for item in self.quantiles}
        if set(by_level) != set(STANDARD_QUANTILES):
            raise ValueError("final forecast must include all standard quantiles.")

        values = [by_level[q] for q in STANDARD_QUANTILES]
        if values != sorted(values):
            raise ValueError("final quantiles must be non-decreasing.")

        if not isclose(self.point_forecast, by_level[0.5], rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("final point_forecast must equal p50.")

        if set(self.component_models) != _REQUIRED_COMPONENTS:
            raise ValueError("component_models must contain exactly arima, kalman, lightgbm, and ensemble.")

        ensemble = self.component_models["ensemble"]
        if ensemble.status != "ok" or ensemble.point_forecast is None:
            raise ValueError("ensemble must be successful in a submitted final forecast.")

        expected_adjustment = self.point_forecast - ensemble.point_forecast
        if not isclose(
            self.ensemble_to_final_adjustment,
            expected_adjustment,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("ensemble_to_final_adjustment is numerically inconsistent.")

        if not self.model_selection_rationale.strip() or not self.final_forecast_rationale.strip():
            raise ValueError("rationale fields cannot be empty.")

        return self

    def quantile_dict(self) -> dict[float, float]:
        """Return final quantiles in the standard order."""
        by_level = {item.quantile: item.value for item in self.quantiles}
        return {q: by_level[q] for q in STANDARD_QUANTILES}


class CfmRichForecastOutput(AgentForecastOutput):
    """Rich forecast converted to standard payloads plus detailed metadata."""

    modality: ClassVar[Literal["continuous", "discrete", "categorical"]] = "continuous"
    model_config = {"extra": "ignore"}

    forecasts: list[CfmRichHorizonForecast]
    verified_evidence: list[CfmVerifiedEvidence] = Field(default_factory=list)
    research_summary: str = ""
    overall_rationale: str
    e2b_used: bool = False
    e2b_summary: str = ""
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_output_contract(self) -> "CfmRichForecastOutput":
        """Reject duplicate horizons and bad evidence references."""
        horizons = [forecast.horizon for forecast in self.forecasts]
        if not horizons:
            raise ValueError("forecasts cannot be empty.")
        if len(horizons) != len(set(horizons)):
            raise ValueError("forecast horizons must be unique.")

        for forecast in self.forecasts:
            if any(index < 0 or index >= len(self.verified_evidence) for index in forecast.evidence_indices):
                raise ValueError("evidence_indices contains an out-of-range index.")

        if not self.overall_rationale.strip():
            raise ValueError("overall_rationale cannot be empty.")
        if self.e2b_used and not self.e2b_summary.strip():
            raise ValueError("e2b_summary is required when e2b_used is true.")

        return self

    @classmethod
    def prompt_schema_json(cls) -> str:
        """Return a compact exact template for ``set_model_response``."""
        quantiles = [{"quantile": q, "value": "<float>"} for q in STANDARD_QUANTILES]
        component = {
            "status": "<ok|error>",
            "point_forecast": "<float|null>",
            "quantiles": quantiles,
            "error": ("<null, empty, or omitted for status=ok; nonempty string required for status=error>"),
        }
        template = {
            "forecasts": [
                {
                    "horizon": "<requested integer>",
                    "point_forecast": "<float equal to final p50>",
                    "quantiles": quantiles,
                    "component_models": {
                        "arima": component,
                        "kalman": component,
                        "lightgbm": component,
                        "ensemble": component,
                    },
                    "model_disagreement_std": "<non-negative float>",
                    "included_models": ["<model names>"],
                    "discounted_models": ["<model names>"],
                    "ensemble_to_final_adjustment": "<final p50 minus ensemble p50>",
                    "model_selection_rationale": "<concise auditable explanation>",
                    "final_forecast_rationale": "<concise auditable explanation>",
                    "evidence_indices": ["<zero-based index into verified_evidence>"],
                }
            ],
            "verified_evidence": [
                {
                    "title": "<source title>",
                    "source_url": "<URL>",
                    "claim": "<verified claim>",
                    "forecast_effect": ("<center_up|center_down|uncertainty_wider|uncertainty_narrower|context_only>"),
                }
            ],
            "research_summary": "<concise summary>",
            "overall_rationale": "<concise overall explanation>",
            "e2b_used": False,
            "e2b_summary": "<empty unless E2B was used>",
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
        """Convert rich output to harness-compatible predictions and metadata."""
        by_horizon = {forecast.horizon: forecast for forecast in self.forecasts}
        if set(by_horizon) != set(task.horizons):
            raise ValueError("rich output must contain exactly the requested horizons.")

        issued_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        offset = pd.tseries.frequencies.to_offset(task.frequency)
        evidence = [item.model_dump(mode="json") for item in self.verified_evidence]
        predictions: list[Prediction] = []

        for horizon in task.horizons:
            forecast = by_horizon[horizon]
            prediction_metadata = dict(metadata or {})
            prediction_metadata.update(
                {
                    "agent_reported_component_models": {
                        name: result.as_dict() for name, result in forecast.component_models.items()
                    },
                    "model_disagreement_std": forecast.model_disagreement_std,
                    "included_models": forecast.included_models,
                    "discounted_models": forecast.discounted_models,
                    "ensemble_to_final_adjustment": forecast.ensemble_to_final_adjustment,
                    "model_selection_rationale": forecast.model_selection_rationale,
                    "final_forecast_rationale": forecast.final_forecast_rationale,
                    "verified_evidence": evidence,
                    "evidence_indices": forecast.evidence_indices,
                    "research_summary": self.research_summary,
                    "overall_rationale": self.overall_rationale,
                    "e2b_used": self.e2b_used,
                    "e2b_summary": self.e2b_summary,
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


__all__ = [
    "CfmComponentForecast",
    "CfmRichForecastOutput",
    "CfmRichHorizonForecast",
    "CfmVerifiedEvidence",
]
