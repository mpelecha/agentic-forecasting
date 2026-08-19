import inspect
import json
from datetime import datetime
from types import SimpleNamespace

import pandas as pd
import pytest
from aieng.forecasting.evaluation.prediction import ContinuousForecast, Prediction
from aieng.forecasting.evaluation.task import ForecastingTask
from energy_oil_forecasting.cfm_agent_v_5_2.agent import (
    build_cfm_agent_config,
    build_cfm_agent_predictor,
)
from energy_oil_forecasting.cfm_agent_v_5_2.config import CfmV52Settings
from energy_oil_forecasting.cfm_agent_v_5_2.forecast_engine import PythonForecastEngine
from energy_oil_forecasting.cfm_agent_v_5_2.schemas import (
    ModelHorizonForecast,
    ModelSuiteResult,
    PolicyDecision,
)
from energy_oil_forecasting.cfm_agent_v_5_2.tools.code_execution import (
    AuditedCodeExecutionTool,
)
from energy_oil_forecasting.cfm_agent_v_5_2.tools.market_data import (
    AuthoritativeSuiteTool,
)
from energy_oil_forecasting.cfm_agent_v_5_2.tools.research_pipeline import (
    ResearchPipelineTool,
)


class SyntheticContext:
    as_of = datetime(2026, 7, 28)

    def __init__(self):
        timestamps = pd.bdate_range("2025-01-01", "2026-07-27")
        self.frame = pd.DataFrame(
            {
                "timestamp": timestamps,
                "value": [70.0 + index * 0.03 for index in range(len(timestamps))],
            }
        )

    def get_series(self, _series_id):
        return self.frame.copy()

    def get_metadata(self, series_id):
        return SimpleNamespace(
            description=f"Synthetic {series_id}",
            source="synthetic-test",
            units="USD/bbl",
            frequency="B",
        )


class SyntheticService:
    series_ids = ["wti"]

    def __init__(self):
        self._context = SyntheticContext()

    def context(self, *, as_of):
        assert pd.Timestamp(as_of).date() == datetime(2026, 7, 28).date()
        return self._context


def _predictions(predictor_id):
    output = []
    for horizon in [5, 10, 21]:
        p50 = 80.0 + horizon / 10.0
        output.append(
            Prediction(
                predictor_id=predictor_id,
                task_id="synthetic",
                issued_at=datetime(2026, 7, 28),
                as_of=datetime(2026, 7, 28),
                forecast_date=(pd.Timestamp("2026-07-28") + pd.offsets.BDay(horizon)).to_pydatetime(),
                payload=ContinuousForecast(
                    point_forecast=p50,
                    quantiles={0.1: p50 - 10.0, 0.5: p50, 0.9: p50 + 10.0},
                ),
                metadata={},
            )
        )
    return output


def _stub_models(tool):
    successful = {name: _predictions(name) for name in ["arima", "kalman", "lightgbm"]}
    tool._ensemble.collect = lambda task, context: (successful, {})
    tool._ensemble.combine = lambda task, context, successful_models: _predictions("ensemble")


def _task(horizons=None):
    return ForecastingTask(
        task_id="synthetic",
        target_series_id="wti",
        horizons=horizons or [5, 10, 21],
        frequency="B",
        description="synthetic contract test",
    )


def _find_key(value, target):
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == target:
                found.append(key)
            found.extend(_find_key(child, target))
    elif isinstance(value, list):
        for child in value:
            found.extend(_find_key(child, target))
    return found


def test_model_suite_schema_declares_frequency():
    assert "frequency" in ModelSuiteResult.model_fields
    assert ModelSuiteResult.model_fields["frequency"].annotation is str


def test_real_model_suite_construction_includes_frequency_and_round_trips():
    tool = AuthoritativeSuiteTool(SyntheticService(), settings=CfmV52Settings(), covariate_series_ids=[])
    _stub_models(tool)
    suite = tool._run_model_suite(
        context=SyntheticContext(),
        target_series_id="wti",
        horizons=[5, 10, 21],
        frequency="B",
        cutoff_date="2026-07-28",
    )
    assert suite.frequency == "B"
    assert suite.horizons == [5, 10, 21]
    assert suite.target_series_id == "wti"
    restored = ModelSuiteResult.model_validate_json(suite.model_dump_json())
    assert restored == suite


def test_public_authoritative_tool_constructs_and_serializes_suite():
    service = SyntheticService()
    tool = AuthoritativeSuiteTool(service, settings=CfmV52Settings(), covariate_series_ids=[])
    _stub_models(tool)
    task = _task()
    context = service.context(as_of=datetime(2026, 7, 28))
    tool.prepare_workflow(task, context)
    payload = json.loads(
        tool.run_authoritative_suite(
            cutoff_date="2026-07-28",
            series_ids=["wti"],
            target_series_id="wti",
            horizons=[5, 10, 21],
            frequency="B",
        )
    )
    assert payload["status"] == "ok"
    assert payload["model_suite"]["frequency"] == "B"
    assert payload["model_suite"]["horizons"] == [5, 10, 21]
    assert tool.audit["successful_execution_count"] == 1


def test_model_suite_identity_changes_with_frequency():
    tool = AuthoritativeSuiteTool(SyntheticService(), settings=CfmV52Settings(), covariate_series_ids=[])
    _stub_models(tool)
    suite_b = tool._run_model_suite(
        context=SyntheticContext(),
        target_series_id="wti",
        horizons=[5, 10, 21],
        frequency="B",
        cutoff_date="2026-07-28",
    )
    suite_d = tool._run_model_suite(
        context=SyntheticContext(),
        target_series_id="wti",
        horizons=[5, 10, 21],
        frequency="D",
        cutoff_date="2026-07-28",
    )
    assert suite_b.model_suite_id != suite_d.model_suite_id


def test_malformed_horizons_return_structured_error():
    tool = AuthoritativeSuiteTool(SyntheticService(), settings=CfmV52Settings(), covariate_series_ids=[])
    context = SyntheticContext()
    tool.prepare_workflow(_task(), context)
    result = json.loads(
        tool.run_authoritative_suite(
            cutoff_date="2026-07-28",
            series_ids=["wti"],
            target_series_id="wti",
            horizons=["bad"],
            frequency="B",
        )
    )
    assert result["status"] == "error"
    assert "only integers" in result["error"]
    assert tool.audit["successful_execution_count"] == 0


def test_unsorted_task_horizons_are_rejected_before_execution():
    tool = AuthoritativeSuiteTool(SyntheticService(), settings=CfmV52Settings(), covariate_series_ids=[])
    with pytest.raises(ValueError, match="strictly increasing"):
        tool.prepare_workflow(_task([10, 5]), SyntheticContext())


def test_duplicate_numerical_execution_does_not_replace_result():
    service = SyntheticService()
    tool = AuthoritativeSuiteTool(service, settings=CfmV52Settings(), covariate_series_ids=[])
    _stub_models(tool)
    context = service.context(as_of=datetime(2026, 7, 28))
    tool.prepare_workflow(_task(), context)
    args = {
        "cutoff_date": "2026-07-28",
        "series_ids": ["wti"],
        "target_series_id": "wti",
        "horizons": [5, 10, 21],
        "frequency": "B",
    }
    first = json.loads(tool.run_authoritative_suite(**args))
    first_id = tool.last_result.model_suite.model_suite_id
    second = json.loads(tool.run_authoritative_suite(**args))
    assert first["status"] == "ok"
    assert second["status"] == "error"
    assert tool.last_result.model_suite.model_suite_id == first_id
    assert tool.audit["successful_execution_count"] == 1


def test_all_function_tool_schemas_are_adk_compatible():
    settings = CfmV52Settings()
    tools = [
        AuthoritativeSuiteTool(SyntheticService(), settings=settings, covariate_series_ids=[]).as_function_tool(),
        ResearchPipelineTool(settings, search_model="test").as_function_tool(),
        AuditedCodeExecutionTool(settings).as_function_tool(),
    ]
    for tool in tools:
        declaration = tool._get_declaration()
        assert not _find_key(declaration.parameters_json_schema, "additionalProperties")
        assert "tool_context" not in declaration.parameters_json_schema.get("properties", {})


def test_configuration_is_one_use():
    config = build_cfm_agent_config(
        data_service=SyntheticService(),
        settings=CfmV52Settings(),
        search_model="test",
        covariate_series_ids=[],
    )
    build_cfm_agent_predictor(config)
    with pytest.raises(ValueError, match="already consumed"):
        build_cfm_agent_predictor(config)


def test_code_execution_audit_defaults_to_unused():
    audit = AuditedCodeExecutionTool(CfmV52Settings()).audit
    assert audit == {
        "code_execution_available": True,
        "code_execution_used": False,
        "code_execution_call_count": 0,
        "code_execution_purposes": [],
    }


def test_market_tool_signature_hides_context_from_llm_schema():
    signature = inspect.signature(AuthoritativeSuiteTool.run_authoritative_suite)
    assert "tool_context" in signature.parameters
    schema = (
        AuthoritativeSuiteTool(SyntheticService(), settings=CfmV52Settings(), covariate_series_ids=[])
        .as_function_tool()
        ._get_declaration()
        .parameters_json_schema
    )
    assert "tool_context" not in schema.get("properties", {})


def test_neutral_engine_is_literal_reproduction():
    quantiles = {
        0.1: 15.423411805148268,
        0.5: 87.96894132007836,
        0.9: 224.4901508722186,
    }
    ensemble = ModelHorizonForecast(
        horizon=5,
        forecast_date="2026-08-04",
        point_forecast=quantiles[0.5],
        quantiles=quantiles,
    )
    decision = PolicyDecision(
        policy_id="test",
        horizon=5,
        eligible=True,
        center_action="no_change",
        uncertainty_action="unchanged",
    )
    result = PythonForecastEngine(CfmV52Settings()).transform(
        ensemble, decision, "likely_reflected_in_model_data", 82.0
    )
    assert result.final_quantiles == quantiles
    assert result.pre_floor_quantiles == quantiles


def test_code_execution_wrapper_is_async():
    assert inspect.iscoroutinefunction(AuditedCodeExecutionTool.run_code)


@pytest.mark.asyncio
async def test_code_execution_awaits_interpreter_and_returns_serializable_result(
    monkeypatch,
):
    tool = AuditedCodeExecutionTool(CfmV52Settings())

    async def fake_run_code(*, code: str) -> str:
        assert code == "print(2 + 2)"
        return "4"

    monkeypatch.setattr(tool._interpreter, "run_code", fake_run_code)

    result = await tool.run_code(
        code="print(2 + 2)",
        purpose="Verify a diagnostic calculation.",
    )

    assert result == "4"
    assert not inspect.isawaitable(result)
    assert tool.audit == {
        "code_execution_available": True,
        "code_execution_used": True,
        "code_execution_call_count": 1,
        "code_execution_purposes": ["Verify a diagnostic calculation."],
    }
