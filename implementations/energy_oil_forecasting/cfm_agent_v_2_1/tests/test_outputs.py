"""Schema and cross-object validator tests for CFM Agent v2.1 output."""

from __future__ import annotations

import copy
from datetime import datetime

import pytest
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES
from aieng.forecasting.evaluation.task import ForecastingTask
from energy_oil_forecasting.cfm_agent_v_2_1.outputs import WtiGeoForecastOutput
from energy_oil_forecasting.cfm_agent_v_2_1.schemas import WtiGeoFactor, WtiGeoScenario
from pydantic import ValidationError


def quantiles(center: float, *, half_width: float = 4.5) -> list[dict[str, float]]:
    """Return a valid standard grid centered on ``center``."""
    return [{"quantile": q, "value": center + (q - 0.5) * (2 * half_width)} for q in STANDARD_QUANTILES]


def valid_output() -> dict[str, object]:
    """Return a fully valid ``WtiGeoForecastOutput`` fixture.

    Scenario spread: [50, 100]. Point forecast 75 sits inside it, and the
    90% interval width (9.0) exceeds the 15%-of-spread floor (7.5).
    """
    return {
        "forecasts": [
            {
                "horizon": 21,
                "point_forecast": 75.0,
                "quantiles": quantiles(75.0),
                "evidence_indices": [0],
                "rationale": "Baseline drift with a real chance of escalation.",
            }
        ],
        "factors": [
            {"name": "A", "tier": "core", "rationale": "Durable driver A."},
            {"name": "B", "tier": "core", "rationale": "Durable driver B."},
            {"name": "C", "tier": "transitory", "rationale": "Situational driver C.", "impact_score": "high"},
        ],
        "scenarios": [
            {
                "name": "Baseline",
                "stances": {"A": "bullish", "B": "neutral", "C": "neutral"},
                "price_low": 60.0,
                "price_high": 70.0,
                "is_tail_case": False,
            },
            {
                "name": "De-escalation",
                "stances": {"A": "bearish", "B": "bullish", "C": "neutral"},
                "price_low": 50.0,
                "price_high": 58.0,
                "is_tail_case": False,
            },
            {
                "name": "Full escalation",
                "stances": {"A": "bullish", "B": "neutral", "C": "bullish"},
                "price_low": 80.0,
                "price_high": 100.0,
                "is_tail_case": True,
            },
        ],
        "verified_evidence": [
            {
                "title": "Example source",
                "source_url": "https://example.com/evidence",
                "claim": "A verified development affecting factor A.",
                "forecast_effect": "center_up",
            }
        ],
        "research_summary": "One material verified event was found.",
        "overall_rationale": "Scenarios span de-escalation through full escalation.",
        "warnings": [],
    }


def test_valid_output_converts_to_predictions(synthetic_service) -> None:
    output = WtiGeoForecastOutput.model_validate(valid_output())
    task = ForecastingTask(
        task_id="geo_test",
        target_series_id="wti_crude_oil_price",
        horizons=[21],
        frequency="B",
        description="test",
    )
    context = synthetic_service.context(datetime(2021, 6, 1))
    prediction = output.to_predictions(task=task, context=context, predictor_id="cfm_geo_test")[0]

    assert prediction.payload.point_forecast == 75.0
    assert prediction.payload.quantiles[0.5] == 75.0
    assert prediction.metadata["factors"][0]["name"] == "A"
    assert prediction.metadata["scenarios"][2]["is_tail_case"] is True
    assert prediction.metadata["overall_rationale"]


def test_missing_tail_case_is_rejected() -> None:
    raw = copy.deepcopy(valid_output())
    for scenario in raw["scenarios"]:
        scenario["is_tail_case"] = False
    with pytest.raises(ValidationError, match="is_tail_case=True"):
        WtiGeoForecastOutput.model_validate(raw)


def test_stance_coverage_mismatch_is_rejected() -> None:
    raw = copy.deepcopy(valid_output())
    del raw["scenarios"][0]["stances"]["C"]
    with pytest.raises(ValidationError, match="cover exactly the shared factors"):
        WtiGeoForecastOutput.model_validate(raw)


def test_scenarios_must_genuinely_disagree() -> None:
    raw = copy.deepcopy(valid_output())
    for scenario in raw["scenarios"]:
        scenario["stances"] = {"A": "bullish", "B": "neutral", "C": "neutral"}
    with pytest.raises(ValidationError, match="genuinely disagree"):
        WtiGeoForecastOutput.model_validate(raw)


def test_factor_tier_counts_out_of_range_is_rejected() -> None:
    raw = copy.deepcopy(valid_output())
    raw["factors"] = [{"name": "A", "tier": "core", "rationale": "Only one core factor."}]
    for scenario in raw["scenarios"]:
        scenario["stances"] = {"A": "bullish"}
    with pytest.raises(ValidationError, match="2-5 core factors"):
        WtiGeoForecastOutput.model_validate(raw)


def test_point_forecast_outside_scenario_spread_is_rejected() -> None:
    raw = copy.deepcopy(valid_output())
    raw["forecasts"][0]["point_forecast"] = 40.0
    raw["forecasts"][0]["quantiles"] = quantiles(40.0)
    with pytest.raises(ValidationError, match="falls outside the scenario price spread"):
        WtiGeoForecastOutput.model_validate(raw)


def test_interval_narrower_than_scenario_floor_is_rejected() -> None:
    raw = copy.deepcopy(valid_output())
    raw["forecasts"][0]["quantiles"] = quantiles(75.0, half_width=0.5)
    with pytest.raises(ValidationError, match="narrower than the required floor"):
        WtiGeoForecastOutput.model_validate(raw)


def test_scenario_price_range_must_be_strictly_ordered() -> None:
    with pytest.raises(ValidationError, match="strictly less than"):
        WtiGeoScenario.model_validate(
            {
                "name": "Point estimate",
                "stances": {"A": "bullish"},
                "price_low": 70.0,
                "price_high": 70.0,
                "is_tail_case": False,
            }
        )


def test_transitory_factor_requires_impact_score() -> None:
    with pytest.raises(ValidationError, match="must set impact_score"):
        WtiGeoFactor.model_validate({"name": "C", "tier": "transitory", "rationale": "Situational."})


def test_core_factor_forbids_impact_score() -> None:
    with pytest.raises(ValidationError, match="must not set impact_score"):
        WtiGeoFactor.model_validate(
            {"name": "A", "tier": "core", "rationale": "Durable.", "impact_score": "low"}
        )
