"""Agent composition and prompt contract tests."""

from __future__ import annotations

import json
from datetime import datetime

from aieng.forecasting.evaluation.task import ForecastingTask
from energy_oil_forecasting.cfm_agent_v_2_0 import (
    AGENT_NAME,
    build_cfm_agent_config,
)
from energy_oil_forecasting.cfm_agent_v_2_0.prompts import CfmForecastPromptBuilder


def test_agent_has_exact_identity_tools_and_skills(synthetic_service) -> None:
    config = build_cfm_agent_config(data_service=synthetic_service)
    assert config.name == AGENT_NAME == "cfm_agent_v_2_0"
    assert config.context_retrieval.enabled is True
    assert config.context_retrieval.enforce_cutoff is True
    assert config.code_execution.enabled is True
    assert [tool.name for tool in config.function_tools] == ["query_market_data"]
    assert [path.name for path in config.skills_dirs] == [
        "forecasting",
        "model-selection",
        "research",
        "code-analysis",
    ]


def test_prompt_carries_exact_task_contract(synthetic_service) -> None:
    task = ForecastingTask(
        task_id="wti_test",
        target_series_id="wti_crude_oil_price",
        horizons=[5, 10, 21],
        frequency="B",
        description="WTI test",
    )
    context = synthetic_service.context(datetime(2021, 6, 1))
    payload = json.loads(CfmForecastPromptBuilder()(task=task, context=context))
    assert payload["agent_name"] == "cfm_agent_v_2_0"
    assert payload["as_of"] == "2021-06-01"
    assert payload["horizons"] == [5, 10, 21]
    assert "query_market_data" in payload["instructions"]
    assert "output_schema" in payload
    assert payload["raw_history_in_prompt"] is False
    assert "recent_target_history_csv" not in payload
    assert "target_history_metadata" not in payload
    serialized = json.dumps(payload)
    assert "date,value" not in serialized
