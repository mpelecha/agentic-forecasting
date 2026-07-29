"""Rich output conversion and component error-contract regression tests."""

from __future__ import annotations

from datetime import datetime

import pytest
import yaml
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES
from aieng.forecasting.evaluation.task import ForecastingTask
from energy_oil_forecasting.cfm_agent_v_2_0.outputs import (
    CfmComponentForecast,
    CfmRichForecastOutput,
)
from pydantic import ValidationError


def quantiles(center: float) -> list[dict[str, float]]:
    """Return a valid standard grid centered on ``center``."""
    return [{"quantile": q, "value": center + (q - 0.5) * 10.0} for q in STANDARD_QUANTILES]


def component(center: float, *, error: str | None = "") -> dict[str, object]:
    """Return one valid successful component."""
    return {
        "status": "ok",
        "point_forecast": center,
        "quantiles": quantiles(center),
        "error": error,
    }


def rich_output(*, component_error: str | None = "") -> dict[str, object]:
    """Return a one-horizon rich output fixture."""
    return {
        "forecasts": [
            {
                "horizon": 5,
                "point_forecast": 70.5,
                "quantiles": quantiles(70.5),
                "component_models": {
                    "arima": component(69.0, error=component_error),
                    "kalman": component(70.0, error=component_error),
                    "lightgbm": component(71.0, error=component_error),
                    "ensemble": component(70.0, error=component_error),
                },
                "model_disagreement_std": 0.8164965809,
                "included_models": ["arima", "kalman", "lightgbm"],
                "discounted_models": [],
                "ensemble_to_final_adjustment": 0.5,
                "model_selection_rationale": "The three models are reasonably aligned.",
                "final_forecast_rationale": "A small verified upward adjustment was applied.",
                "evidence_indices": [0],
            }
        ],
        "verified_evidence": [
            {
                "title": "Example source",
                "source_url": "https://example.com/evidence",
                "claim": "A verified supply event tightened the near-term balance.",
                "forecast_effect": "center_up",
            }
        ],
        "research_summary": "One material verified event was found.",
        "overall_rationale": "The ensemble remains the anchor.",
        "e2b_used": False,
        "e2b_summary": "",
        "warnings": [],
    }


def test_rich_output_becomes_standard_prediction_with_metadata(synthetic_service) -> None:
    output = CfmRichForecastOutput.model_validate(rich_output())
    task = ForecastingTask(
        task_id="rich",
        target_series_id="wti_crude_oil_price",
        horizons=[5],
        frequency="B",
        description="test",
    )
    context = synthetic_service.context(datetime(2021, 6, 1))
    prediction = output.to_predictions(
        task=task,
        context=context,
        predictor_id="cfm_test",
    )[0]

    assert prediction.payload.point_forecast == 70.5
    assert prediction.payload.quantiles[0.5] == 70.5
    assert prediction.metadata["ensemble_to_final_adjustment"] == 0.5
    assert prediction.metadata["model_selection_rationale"]
    assert prediction.metadata["verified_evidence"][0]["source_url"]
    assert prediction.metadata["e2b_used"] is False

    serialized = yaml.safe_dump(prediction.model_dump(mode="json"))
    assert "agent_reported_component_models" in serialized
    assert "final_forecast_rationale" in serialized


def test_successful_component_accepts_missing_error() -> None:
    raw = component(70.0)
    raw.pop("error")
    parsed = CfmComponentForecast.model_validate(raw)
    assert parsed.status == "ok"
    assert parsed.error is None


def test_successful_component_accepts_empty_error() -> None:
    parsed = CfmComponentForecast.model_validate(component(70.0, error=""))
    assert parsed.status == "ok"
    assert parsed.error == ""


def test_successful_component_accepts_null_error() -> None:
    parsed = CfmComponentForecast.model_validate(component(70.0, error=None))
    assert parsed.status == "ok"
    assert parsed.error is None


def test_failed_component_accepts_nonempty_error() -> None:
    parsed = CfmComponentForecast.model_validate(
        {
            "status": "error",
            "point_forecast": None,
            "quantiles": [],
            "error": "RuntimeError: model failed",
        }
    )
    assert parsed.status == "error"
    assert parsed.error == "RuntimeError: model failed"


@pytest.mark.parametrize("error_payload", [None, "", "   "])
def test_failed_component_rejects_missing_or_empty_error(error_payload: str | None) -> None:
    with pytest.raises(ValidationError, match="failed components must include an error"):
        CfmComponentForecast.model_validate(
            {
                "status": "error",
                "point_forecast": None,
                "quantiles": [],
                "error": error_payload,
            }
        )


def test_full_rich_output_accepts_null_errors_and_converts(synthetic_service) -> None:
    output = CfmRichForecastOutput.model_validate(rich_output(component_error=None))
    task = ForecastingTask(
        task_id="null_error_regression",
        target_series_id="wti_crude_oil_price",
        horizons=[5],
        frequency="B",
        description="Regression fixture copied from the live JSON boundary.",
    )
    context = synthetic_service.context(datetime(2021, 6, 1))
    prediction = output.to_predictions(
        task=task,
        context=context,
        predictor_id="cfm_test",
    )[0]

    assert prediction.payload.point_forecast == 70.5
    models = prediction.metadata["agent_reported_component_models"]
    assert set(models) == {"arima", "kalman", "lightgbm", "ensemble"}
    assert all(model["error"] is None for model in models.values())
