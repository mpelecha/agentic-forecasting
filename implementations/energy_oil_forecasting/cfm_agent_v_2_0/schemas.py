"""Typed market-data and model-audit contracts for CFM Agent v2.0."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ToolStatus = Literal["ok", "partial", "error"]


class SeriesObservation(BaseModel):
    """One cutoff-safe time-series observation."""

    model_config = ConfigDict(extra="forbid")
    date: str
    value: float


class SeriesSnapshot(BaseModel):
    """Metadata and bounded history for one registered series."""

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
    """Auditable data window and feature settings used by one model."""

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
    """One model's probabilistic forecast at one horizon."""

    model_config = ConfigDict(extra="forbid")
    horizon: int = Field(gt=0)
    forecast_date: str
    point_forecast: float
    quantiles: dict[float, float]

    @model_validator(mode="after")
    def validate_quantiles(self) -> "ModelHorizonForecast":
        """Require non-crossing quantiles and median equality."""
        ordered = [self.quantiles[q] for q in sorted(self.quantiles)]
        if ordered != sorted(ordered):
            raise ValueError("quantiles must be non-decreasing.")
        if 0.5 not in self.quantiles:
            raise ValueError("quantiles must include 0.5.")
        if self.point_forecast != self.quantiles[0.5]:
            raise ValueError("point_forecast must equal the 0.5 quantile.")
        return self


class ModelForecastResult(BaseModel):
    """Forecast trajectory and training data for one deterministic model."""

    model_config = ConfigDict(extra="forbid")
    model_name: str
    predictor_id: str
    status: Literal["ok", "error"]
    forecasts: list[ModelHorizonForecast] = Field(default_factory=list)
    training_data: ModelTrainingData
    error: str | None = None


class ModelSuiteResult(BaseModel):
    """ARIMA, Kalman, LightGBM, ensemble, and model-audit results."""

    model_config = ConfigDict(extra="forbid")
    target_series_id: str
    cutoff_date: str
    horizons: list[int]
    models: dict[str, ModelForecastResult]
    ensemble: ModelForecastResult | None = None
    configured_ensemble_weights: dict[str, float]
    active_ensemble_weights: dict[str, float] = Field(default_factory=dict)
    successful_models: list[str] = Field(default_factory=list)
    failed_models: list[str] = Field(default_factory=list)
    model_disagreement_std: dict[int, float] = Field(default_factory=dict)


class MarketDataToolResult(BaseModel):
    """Top-level response from the market-data function tool."""

    model_config = ConfigDict(extra="forbid")
    status: ToolStatus
    operation: str
    cutoff_date: str
    available_series: list[str] = Field(default_factory=list)
    series: list[SeriesSnapshot] = Field(default_factory=list)
    model_suite: ModelSuiteResult | None = None
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


__all__ = [
    "MarketDataToolResult",
    "ModelForecastResult",
    "ModelHorizonForecast",
    "ModelSuiteResult",
    "ModelTrainingData",
    "SeriesObservation",
    "SeriesSnapshot",
    "ToolStatus",
]
