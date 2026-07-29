"""Agent composition and prompt contract tests."""

from __future__ import annotations

import json
from datetime import datetime

from aieng.forecasting.evaluation.task import ForecastingTask
from energy_oil_forecasting.cfm_agent_v_2_1 import (
    AGENT_NAME,
    build_cfm_agent_v21_config,
)
from energy_oil_forecasting.cfm_agent_v_2_1.prompts import CfmGeoPromptBuilder


def test_agent_has_exact_identity_tools_and_skills() -> None:
    config = build_cfm_agent_v21_config()
    assert config.name == AGENT_NAME == "cfm_agent_v_2_1"
    assert config.context_retrieval.enabled is True
    assert config.context_retrieval.enforce_cutoff is True
    assert not (config.code_execution and config.code_execution.enabled)
    assert not config.function_tools
    assert [path.name for path in config.skills_dirs] == ["geopolitical-analysis"]


def test_prompt_carries_exact_task_contract(synthetic_service) -> None:
    task = ForecastingTask(
        task_id="wti_test",
        target_series_id="wti_crude_oil_price",
        horizons=[5, 10, 21],
        frequency="B",
        description="WTI test",
    )
    context = synthetic_service.context(datetime(2021, 6, 1))
    payload = json.loads(CfmGeoPromptBuilder()(task=task, context=context))
    assert payload["agent_name"] == "cfm_agent_v_2_1"
    assert payload["as_of"] == "2021-06-01"
    assert payload["horizons"] == [5, 10, 21]
    assert "output_schema" in payload
    assert payload["target_history_csv"].startswith("date,close")
