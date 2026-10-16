"""Corrective controls carried from v5.2, run against the synthetic data service (TRACEABILITY: v5.2 file kept by name)."""

import inspect
import json
from datetime import datetime

import aieng.agents.tools.code_interpreter as interpreter_module
import pytest
from aieng.forecasting.evaluation.task import ForecastingTask
from energy_oil_forecasting.cfm_agent_v_6_0.agent import (
    build_cfm_agent_config,
    build_cfm_agent_predictor,
)
from energy_oil_forecasting.cfm_agent_v_6_0.config import CfmV60Settings
from energy_oil_forecasting.cfm_agent_v_6_0.forecast_engine import PythonForecastEngine
from energy_oil_forecasting.cfm_agent_v_6_0.history import PriceHistory
from energy_oil_forecasting.cfm_agent_v_6_0.schemas import (
    ModelHorizonForecast,
    ModelSuiteResult,
    PolicyDecision,
)
from energy_oil_forecasting.cfm_agent_v_6_0.tools.code_execution import (
    AuditedCodeExecutionTool,
)
from energy_oil_forecasting.cfm_agent_v_6_0.tools.market_data import (
    AuthoritativeSuiteTool,
)
from energy_oil_forecasting.cfm_agent_v_6_0.tools.research_pipeline import (
    ResearchPipelineTool,
)
from energy_oil_forecasting.data import WTI_SERIES_ID


CUTOFF = datetime(2026, 9, 14)
CUTOFF_TEXT = "2026-09-14"


def _task(horizons=None):
    return ForecastingTask(
        task_id="synthetic",
        target_series_id=WTI_SERIES_ID,
        horizons=horizons or [5, 10, 21],
        frequency="B",
        description="synthetic contract test",
    )


def _args(**overrides):
    args = {
        "cutoff_date": CUTOFF_TEXT,
        "series_ids": [WTI_SERIES_ID],
        "target_series_id": WTI_SERIES_ID,
        "horizons": [5, 10, 21],
        "frequency": "B",
    }
    args.update(overrides)
    return args


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


def test_real_model_suite_construction_includes_frequency_and_round_trips(synthetic_service, synthetic_context):
    tool = AuthoritativeSuiteTool(synthetic_service, settings=CfmV60Settings())
    suite = tool._run_model_suite(
        context=synthetic_context,
        target_series_id=WTI_SERIES_ID,
        horizons=[5, 10, 21],
        frequency="B",
        cutoff_date=CUTOFF_TEXT,
    )
    assert suite.frequency == "B"
    assert suite.horizons == [5, 10, 21]
    assert suite.target_series_id == WTI_SERIES_ID
    restored = ModelSuiteResult.model_validate_json(suite.model_dump_json())
    assert restored == suite


def test_public_authoritative_tool_constructs_and_serializes_suite(synthetic_service, synthetic_context):
    tool = AuthoritativeSuiteTool(synthetic_service, settings=CfmV60Settings())
    tool.prepare_workflow(_task(), synthetic_context)
    payload = json.loads(tool.run_authoritative_suite(**_args()))
    assert payload["status"] == "ok"
    assert payload["model_suite"]["frequency"] == "B"
    assert payload["model_suite"]["horizons"] == [5, 10, 21]
    assert payload["model_suite"]["successful_components"] == ["bootstrap", "sparse_ar", "lightgbm"]
    assert len(payload["series"]) == 1
    assert payload["series"][0]["returned_observations"] == 260
    assert tool.audit["successful_execution_count"] == 1
    assert tool.audit["suite_seed"] == payload["model_suite"]["suite_seed"]
    assert tool.audit["history_window"]["rows"] == 3000


def test_llm_snapshot_is_capped_at_max_data_rows_per_series(synthetic_service, synthetic_context):
    settings = CfmV60Settings()
    tool = AuthoritativeSuiteTool(synthetic_service, settings=settings)
    tool.prepare_workflow(_task(), synthetic_context)
    payload = json.loads(tool.run_authoritative_suite(**_args(lookback=10_000)))
    assert payload["series"][0]["returned_observations"] == settings.max_data_rows_per_series
    assert payload["series"][0]["n_observations_at_cutoff"] == 3000
    assert payload["model_suite"]["history_window"]["rows"] == 3000


def test_model_suite_identity_changes_with_frequency(synthetic_service, synthetic_context):
    tool = AuthoritativeSuiteTool(synthetic_service, settings=CfmV60Settings())
    kwargs = {
        "context": synthetic_context,
        "target_series_id": WTI_SERIES_ID,
        "horizons": [5, 10, 21],
        "cutoff_date": CUTOFF_TEXT,
    }
    suite_b = tool._run_model_suite(frequency="B", **kwargs)
    suite_d = tool._run_model_suite(frequency="D", **kwargs)
    assert suite_b.model_suite_id != suite_d.model_suite_id


def test_malformed_horizons_return_structured_error(synthetic_service, synthetic_context):
    tool = AuthoritativeSuiteTool(synthetic_service, settings=CfmV60Settings())
    tool.prepare_workflow(_task(), synthetic_context)
    result = json.loads(tool.run_authoritative_suite(**_args(horizons=["bad"])))
    assert result["status"] == "error"
    assert "only integers" in result["error"]
    assert tool.audit["successful_execution_count"] == 0
    assert tool.audit["suite_seed"] is None
    assert tool.audit["history_window"] is None


def test_horizons_beyond_the_simulation_length_are_rejected(synthetic_service, synthetic_context):
    tool = AuthoritativeSuiteTool(synthetic_service, settings=CfmV60Settings())
    tool.prepare_workflow(_task([5, 10, 42]), synthetic_context)
    result = json.loads(tool.run_authoritative_suite(**_args(horizons=[5, 10, 42])))
    assert result["status"] == "error"
    assert "simulation length" in result["error"]
    assert tool.audit["successful_execution_count"] == 0


def test_unsorted_task_horizons_are_rejected_before_execution(synthetic_service, synthetic_context):
    tool = AuthoritativeSuiteTool(synthetic_service, settings=CfmV60Settings())
    with pytest.raises(ValueError, match="strictly increasing"):
        tool.prepare_workflow(_task([10, 5]), synthetic_context)


def test_duplicate_numerical_execution_does_not_replace_result(synthetic_service, synthetic_context):
    tool = AuthoritativeSuiteTool(synthetic_service, settings=CfmV60Settings())
    tool.prepare_workflow(_task(), synthetic_context)
    first = json.loads(tool.run_authoritative_suite(**_args()))
    first_id = tool.last_result.model_suite.model_suite_id
    second = json.loads(tool.run_authoritative_suite(**_args()))
    assert first["status"] == "ok"
    assert second["status"] == "error"
    assert tool.last_result.model_suite.model_suite_id == first_id
    assert tool.audit["successful_execution_count"] == 1
    assert tool.audit["attempt_count"] == 2


def test_pre_execution_attempts_are_capped(synthetic_service, synthetic_context):
    settings = CfmV60Settings()
    tool = AuthoritativeSuiteTool(synthetic_service, settings=settings)
    tool.prepare_workflow(_task(), synthetic_context)
    for _ in range(settings.max_pre_execution_attempts):
        failed = json.loads(tool.run_authoritative_suite(**_args(frequency="D")))
        assert failed["status"] == "error"
    result = json.loads(tool.run_authoritative_suite(**_args()))
    assert result["status"] == "error"
    assert "attempt limit" in result["error"]
    assert tool.audit["successful_execution_count"] == 0


def test_code_execution_payload_carries_diagnostics_quantiles_and_binding(synthetic_service, synthetic_context):
    tool = AuthoritativeSuiteTool(synthetic_service, settings=CfmV60Settings())
    tool.prepare_workflow(_task(), synthetic_context)
    assert tool.code_execution_payload() is None
    json.loads(tool.run_authoritative_suite(**_args()))
    payload = tool.code_execution_payload()
    assert set(payload) == {"DIAGNOSTICS", "MODEL_SUITE_QUANTILES", "TASK_BINDING"}
    assert set(payload["MODEL_SUITE_QUANTILES"]) == {"bootstrap", "sparse_ar", "lightgbm", "ensemble"}
    assert set(payload["MODEL_SUITE_QUANTILES"]["ensemble"]) == {"5", "10", "21"}
    assert len(payload["MODEL_SUITE_QUANTILES"]["ensemble"]["21"]) == 11
    assert payload["TASK_BINDING"] == {
        "cutoff": CUTOFF_TEXT,
        "target_series_id": WTI_SERIES_ID,
        "horizons": [5, 10, 21],
        "frequency": "B",
    }
    assert "historical_profiles" in payload["DIAGNOSTICS"]
    text = json.dumps(payload)
    assert "settings" not in text and "paths" not in text
    json.dumps(payload)  # JSON-literal safe for the sandbox preamble


def test_all_function_tool_schemas_are_adk_compatible(synthetic_service):
    settings = CfmV60Settings()
    tools = [
        AuthoritativeSuiteTool(synthetic_service, settings=settings).as_function_tool(),
        ResearchPipelineTool(settings).as_function_tool(),
        AuditedCodeExecutionTool(settings).as_function_tool(),
    ]
    for tool in tools:
        declaration = tool._get_declaration()
        assert not _find_key(declaration.parameters_json_schema, "additionalProperties")
        assert "tool_context" not in declaration.parameters_json_schema.get("properties", {})


def test_configuration_is_one_use(synthetic_service):
    config = build_cfm_agent_config(
        data_service=synthetic_service,
        settings=CfmV60Settings(),
        search_model="test",
    )
    build_cfm_agent_predictor(config)
    with pytest.raises(ValueError, match="already consumed"):
        build_cfm_agent_predictor(config)


def test_code_execution_audit_defaults_to_unused():
    audit = AuditedCodeExecutionTool(CfmV60Settings()).audit
    expected = {
        "code_execution_available": True,
        "code_execution_used": False,
        "code_execution_call_count": 0,
        "code_execution_purposes": [],
    }
    assert {key: audit[key] for key in expected} == expected


def test_market_tool_signature_hides_context_from_llm_schema(synthetic_service):
    signature = inspect.signature(AuthoritativeSuiteTool.run_authoritative_suite)
    assert "tool_context" in signature.parameters
    schema = (
        AuthoritativeSuiteTool(synthetic_service, settings=CfmV60Settings())
        .as_function_tool()
        ._get_declaration()
        .parameters_json_schema
    )
    assert "tool_context" not in schema.get("properties", {})


def test_neutral_engine_is_literal_reproduction(synthetic_context):
    quantiles = {
        0.1: 15.423411805148268,
        0.5: 87.96894132007836,
        0.9: 224.4901508722186,
    }
    ensemble = ModelHorizonForecast(
        horizon=5,
        forecast_date="2026-09-21",
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
    settings = CfmV60Settings()
    engine = PythonForecastEngine(settings)
    without_profile = engine.transform(ensemble, decision, "likely_reflected_in_model_data", 82.0, None)
    assert without_profile.final_quantiles == quantiles
    assert without_profile.pre_floor_quantiles == quantiles
    profile = PriceHistory.from_context(synthetic_context, WTI_SERIES_ID, [], settings).profile(5)
    with_profile = engine.transform(ensemble, decision, "likely_reflected_in_model_data", 82.0, profile)
    assert with_profile.final_quantiles == quantiles
    assert with_profile.pre_floor_quantiles == quantiles
    assert with_profile.neutral is True
    assert with_profile.historical_profile == profile


def test_code_execution_wrapper_is_async():
    assert inspect.iscoroutinefunction(AuditedCodeExecutionTool.run_code)


@pytest.mark.asyncio
async def test_code_execution_awaits_interpreter_and_returns_serializable_result(
    monkeypatch, synthetic_service, synthetic_context
):
    market = AuthoritativeSuiteTool(synthetic_service, settings=CfmV60Settings())
    market.prepare_workflow(_task(), synthetic_context)
    json.loads(market.run_authoritative_suite(**_args()))
    tool = AuditedCodeExecutionTool(CfmV60Settings(), payload_provider=market.code_execution_payload)

    class FakeInterpreter:
        def __init__(self, **kwargs):
            pass

        async def run_code(self, *, code: str) -> str:
            assert code.endswith("print(2 + 2)")
            assert "DIAGNOSTICS" in code and "MODEL_SUITE_QUANTILES" in code and "TASK_BINDING" in code
            return "4"

    monkeypatch.setattr(interpreter_module, "CodeInterpreter", FakeInterpreter)

    result = await tool.run_code(
        code="print(2 + 2)",
        purpose="Verify a diagnostic calculation.",
    )

    assert not inspect.isawaitable(result)
    parsed = json.loads(result)
    assert parsed["status"] == "executed"
    assert parsed["stdout"] == "4"
    expected = {
        "code_execution_available": True,
        "code_execution_used": True,
        "code_execution_call_count": 1,
        "code_execution_purposes": ["Verify a diagnostic calculation."],
    }
    assert {key: tool.audit[key] for key in expected} == expected
