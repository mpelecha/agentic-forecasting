"""Regression coverage for the v4.4 live failure and the v6.0 suite contract (TRACEABILITY: v5.2 file kept by name)."""

import ast
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from aieng.forecasting.evaluation.task import ForecastingTask
from energy_oil_forecasting.cfm_agent_v_6_0.config import CfmV60Settings
from energy_oil_forecasting.cfm_agent_v_6_0.outputs import CfmContextAssessmentOutput
from energy_oil_forecasting.cfm_agent_v_6_0.policy import EvidencePolicy
from energy_oil_forecasting.cfm_agent_v_6_0.schemas import (
    ComponentDisposition,
    EvidenceClaim,
    FloorRecord,
    HistoryWindow,
    HorizonAction,
    MarketDiagnostics,
    ModelForecastResult,
    ModelHorizonForecast,
    ModelSuiteResult,
    ModelTrainingData,
)
from energy_oil_forecasting.cfm_agent_v_6_0.tools.market_data import (
    AuthoritativeSuiteTool,
)
from energy_oil_forecasting.data import WTI_SERIES_ID


class FakeService:
    series_ids = ["wti"]


def _forecast_result(name: str = "ensemble") -> ModelForecastResult:
    return ModelForecastResult(
        model_name=name,
        predictor_id=f"{name}_predictor",
        status="ok",
        forecasts=[
            ModelHorizonForecast(
                horizon=5,
                forecast_date="2026-08-04",
                point_forecast=80.0,
                quantiles={0.1: 70.0, 0.5: 80.0, 0.9: 90.0},
            )
        ],
        training_data=ModelTrainingData(target_observations=260),
        num_paths=1000,
        component_seed=None if name == "ensemble" else 7,
    )


def _suite() -> ModelSuiteResult:
    result = _forecast_result()
    return ModelSuiteResult(
        model_suite_id="model_suite:test",
        target_series_id="wti",
        cutoff_date="2026-07-28",
        horizons=[5],
        frequency="B",
        model_num_paths=1000,
        simulation_length_business_days=21,
        suite_seed=1,
        seed_derivation="test",
        seed_input_fingerprint="suite_seed_input:sha256:test",
        data_fingerprint="data:sha256:test",
        history_window=HistoryWindow(setting_business_days=0, start_date="2025-07-28", end_date="2026-07-28", rows=260),
        component_seeds={"bootstrap": 7},
        components={"bootstrap": _forecast_result("bootstrap")},
        ensemble=result,
        configured_ensemble_weights={"bootstrap": 0.5, "sparse_ar": 0.25, "lightgbm": 0.25},
        active_ensemble_weights={"bootstrap": 1.0},
        successful_components=["bootstrap"],
        failed_components=[],
        dispositions={"bootstrap": ComponentDisposition(status="ok")},
        floors=FloorRecord(input_price_floor=1.0, simulated_price_floor=1.0, final_price_floor=1.0),
        model_disagreement_std={5: 0.0},
        diagnostics=MarketDiagnostics(latest_value=82.61),
    )


def test_model_suite_schema_includes_frequency():
    assert "frequency" in ModelSuiteResult.model_fields
    assert ModelSuiteResult.model_fields["frequency"].annotation is str


def test_model_suite_frequency_constructs_and_round_trips():
    suite = _suite()
    assert suite.frequency == "B"
    restored = ModelSuiteResult.model_validate_json(suite.model_dump_json())
    assert restored == suite


def test_model_suite_constructor_keywords_match_schema():
    path = Path(__file__).parents[1] / "tools" / "market_data.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "ModelSuiteResult"
    ]
    assert len(calls) == 1
    supplied = {keyword.arg for keyword in calls[0].keywords if keyword.arg is not None}
    declared = set(ModelSuiteResult.model_fields)
    assert supplied <= declared
    assert "frequency" in supplied
    required = {name for name, field in ModelSuiteResult.model_fields.items() if field.is_required()}
    assert required <= supplied


def test_public_suite_path_serializes_frequency(monkeypatch):
    tool = AuthoritativeSuiteTool(FakeService(), settings=CfmV60Settings(), covariate_series_ids=[])
    task = ForecastingTask(
        task_id="test",
        target_series_id="wti",
        horizons=[5],
        frequency="B",
        description="test",
    )
    tool.prepare_workflow(task, SimpleNamespace(as_of="2026-07-28"))
    fake_context = SimpleNamespace()
    monkeypatch.setattr(tool._data_service, "context", lambda as_of: fake_context, raising=False)
    monkeypatch.setattr(tool, "_snapshot", lambda *args: None)
    monkeypatch.setattr(tool, "_run_model_suite", lambda **kwargs: _suite())
    result = json.loads(
        tool.run_authoritative_suite(
            cutoff_date="2026-07-28",
            series_ids=[],
            target_series_id="wti",
            horizons=[5],
            frequency="B",
        )
    )
    assert result["status"] == "ok"
    assert result["model_suite"]["frequency"] == "B"
    assert tool.audit["successful_execution_count"] == 1
    assert tool.audit["suite_seed"] == 1
    assert tool.audit["component_dispositions"] == {"bootstrap": {"status": "ok", "failure_code": None, "reason": None}}


def test_public_suite_rejects_wrong_frequency_before_execution(monkeypatch):
    tool = AuthoritativeSuiteTool(FakeService(), settings=CfmV60Settings(), covariate_series_ids=[])
    task = ForecastingTask(
        task_id="test",
        target_series_id="wti",
        horizons=[5],
        frequency="B",
        description="test",
    )
    tool.prepare_workflow(task, SimpleNamespace(as_of="2026-07-28"))
    monkeypatch.setattr(tool, "_run_model_suite", lambda **kwargs: pytest.fail("suite must not run"))
    result = json.loads(
        tool.run_authoritative_suite(
            cutoff_date="2026-07-28",
            series_ids=[],
            target_series_id="wti",
            horizons=[5],
            frequency="D",
        )
    )
    assert result["status"] == "error"
    assert "frequency must equal" in result["error"]
    assert tool.audit["successful_execution_count"] == 0


def test_real_internal_model_suite_constructs_frequency(synthetic_service):
    tool = AuthoritativeSuiteTool(synthetic_service, settings=CfmV60Settings())
    suite = tool._run_model_suite(
        context=synthetic_service.context(as_of=datetime(2026, 9, 14)),
        target_series_id=WTI_SERIES_ID,
        horizons=[5],
        frequency="B",
        cutoff_date="2026-09-14",
    )
    assert suite.frequency == "B"
    assert suite.horizons == [5]
    assert suite.target_series_id == WTI_SERIES_ID
    assert suite.successful_components == ["bootstrap", "sparse_ar", "lightgbm"]
    assert [forecast.forecast_date for forecast in suite.ensemble.forecasts] == ["2026-09-21"]


def test_march_2_claim_source_subset_regression(packet):
    """A proper subset of summary-associated sources remains qualifying evidence.

    Confidence 0.70 with two publishers is Tier 2 under the v6.0 tiering (Tier 3 needs 0.80).
    """
    summary = packet.verified_summaries[0]
    claim = EvidenceClaim(
        claim_id="march_2_subset",
        statement="Confirmed material physical oil-flow disruption.",
        claim_type="shipping",
        supporting_summary_ids=[summary.verified_summary_id],
        supporting_passage_ids=[passage.passage_id for passage in summary.passages],
        supporting_source_ids=["source_001", "source_002"],
        material_to_forecast=True,
    )
    assessment = CfmContextAssessmentOutput(
        research_packet_id=packet.packet_id,
        evidence_claims=[claim],
        physical_status="confirmed_disruption",
        incremental_novelty="likely_new_relative_to_model_data",
        material_evidence_conflict=False,
        confidence=0.7,
        horizon_actions=[
            HorizonAction(
                horizon=5,
                center_action="moderate_up",
                uncertainty_action="moderately_wider",
                persistence_profile="persistent",
                cited_claim_ids=[claim.claim_id],
                rationale="March 2 source-subset regression.",
            )
        ],
        research_summary="Fixed regression fixture.",
        overall_rationale="Fixed regression fixture.",
    )
    decision = EvidencePolicy(CfmV60Settings()).apply(assessment, packet, 5)
    assert decision.eligible is True
    assert decision.evidence_tier == "corroborated"
    assert decision.qualifying_claim_ids == [claim.claim_id]
    assert decision.resolved_source_ids == ["source_001", "source_002"]
