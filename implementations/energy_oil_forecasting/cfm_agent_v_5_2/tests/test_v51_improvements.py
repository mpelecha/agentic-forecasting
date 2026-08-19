import json
from datetime import datetime
from types import SimpleNamespace

import pandas as pd
from aieng.forecasting.evaluation.prediction import ContinuousForecast, Prediction
from aieng.forecasting.evaluation.task import ForecastingTask
from energy_oil_forecasting.cfm_agent_v_5_2.config import CfmV52Settings
from energy_oil_forecasting.cfm_agent_v_5_2.response_control import (
    StructuredAssessmentController,
)
from energy_oil_forecasting.cfm_agent_v_5_2.tools.market_data import (
    AuthoritativeSuiteTool,
)


def _valid(packet_id="packet", physical_status="unknown"):
    return json.dumps(
        {
            "research_packet_id": packet_id,
            "evidence_claims": [],
            "physical_status": physical_status,
            "incremental_novelty": "indeterminate",
            "material_evidence_conflict": False,
            "confidence": 0.0,
            "horizon_actions": [
                {
                    "horizon": 5,
                    "center_action": "no_change",
                    "uncertainty_action": "unchanged",
                    "persistence_profile": "unknown",
                    "cited_claim_ids": [],
                    "narrowing_uncertainty_resolved": "",
                    "rationale": "neutral",
                }
            ],
            "research_summary": "summary",
            "overall_rationale": "rationale",
            "warnings": [],
        }
    )


class FakeInner:
    predictor_id = "fake"
    _runner = None

    def prompt_builder(self, **kwargs):
        return "prompt"


def _task():
    return ForecastingTask(
        task_id="test",
        target_series_id="wti",
        horizons=[5],
        frequency="B",
        description="test",
    )


def _context():
    return SimpleNamespace(as_of=datetime(2026, 8, 13))


def test_invalid_initial_response_is_corrected_once(monkeypatch):
    controller = StructuredAssessmentController(FakeInner(), CfmV52Settings())
    monkeypatch.setattr(controller, "_corrected_response", lambda **kwargs: _valid())
    result = controller.resolve(
        task=_task(),
        context=_context(),
        packet_id="packet",
        initial_raw_response=_valid(physical_status="tight"),
    )
    assert result.audit["initial_validation_failed"] is True
    assert result.audit["correction_retry_attempted"] is True
    assert result.audit["retry_succeeded"] is True
    assert result.audit["neutral_fallback_applied"] is False


def test_second_invalid_response_produces_neutral_fallback(monkeypatch):
    controller = StructuredAssessmentController(FakeInner(), CfmV52Settings())
    monkeypatch.setattr(controller, "_corrected_response", lambda **kwargs: "not json")
    result = controller.resolve(
        task=_task(),
        context=_context(),
        packet_id="packet",
        initial_raw_response=_valid(physical_status="tight"),
    )
    action = result.assessment.horizon_actions[0]
    assert action.center_action == "no_change"
    assert action.uncertainty_action == "unchanged"
    assert result.audit["neutral_fallback_applied"] is True
    assert result.audit["final_disposition"] == "neutral_fallback_after_retry_failure"


class Service:
    series_ids = ["wti"]


def _prediction():
    return Prediction(
        predictor_id="stub",
        task_id="stub",
        issued_at=datetime(2026, 8, 13),
        as_of=datetime(2026, 8, 13),
        forecast_date=datetime(2026, 8, 20),
        payload=ContinuousForecast(
            point_forecast=80.0,
            quantiles={0.1: 70.0, 0.5: 80.0, 0.9: 90.0},
        ),
    )


def test_task_derived_seed_and_suite_id_are_repeatable(monkeypatch):
    tool = AuthoritativeSuiteTool(Service(), settings=CfmV52Settings(), covariate_series_ids=[])
    dates = pd.bdate_range("2025-01-01", periods=260)
    frame = pd.DataFrame({"timestamp": dates, "value": range(260)})
    context = SimpleNamespace(as_of=pd.Timestamp("2026-08-13"), get_series=lambda _: frame.copy())
    prediction = _prediction()
    monkeypatch.setattr(tool._ensemble, "collect", lambda task, ctx: ({"arima": [prediction]}, {}))
    monkeypatch.setattr(tool._ensemble, "combine", lambda task, ctx, successful: [prediction])
    first = tool._run_model_suite(
        context=context,
        target_series_id="wti",
        horizons=[5],
        frequency="B",
        cutoff_date="2026-08-13",
    )
    second = tool._run_model_suite(
        context=context,
        target_series_id="wti",
        horizons=[5],
        frequency="B",
        cutoff_date="2026-08-13",
    )
    assert first.rng_seed == second.rng_seed
    assert first.rng_seed_input_fingerprint == second.rng_seed_input_fingerprint
    assert first.model_suite_id == second.model_suite_id
