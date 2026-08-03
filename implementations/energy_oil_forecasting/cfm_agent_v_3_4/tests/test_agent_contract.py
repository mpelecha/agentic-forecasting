"""Construction and prompt-boundary tests."""

import json
from datetime import datetime

from aieng.forecasting.evaluation.task import ForecastingTask
from energy_oil_forecasting.cfm_agent_v_3_4 import (
    AGENT_NAME,
    DEFAULT_SETTINGS,
    build_cfm_agent_config,
)
from energy_oil_forecasting.cfm_agent_v_3_4.prompts import CfmV34PromptBuilder


def test_default_sample_count_and_agent_composition(synthetic_service) -> None:
    assert DEFAULT_SETTINGS.model_num_samples == 1_000
    config = build_cfm_agent_config(data_service=synthetic_service)
    assert config.name == AGENT_NAME == "cfm_agent_v_3_4"
    assert [tool.name for tool in config.function_tools] == ["run_authoritative_suite"]
    assert [path.name for path in config.skills_dirs] == [
        "evidence-assessment",
        "market-context-assessment",
        "forecast-action-selection",
        "code-analysis",
    ]


def test_prompt_forbids_llm_authored_final_numbers(synthetic_service) -> None:
    task = ForecastingTask(
        task_id="wti_v3",
        target_series_id="wti_crude_oil_price",
        horizons=[5, 10, 21],
        frequency="B",
        description="WTI v3 test",
    )
    payload = json.loads(CfmV34PromptBuilder()(task=task, context=synthetic_service.context(datetime(2021, 6, 1))))
    assert payload["raw_history_in_prompt"] is False
    assert "Do not calculate any final price" in payload["instructions"]
    assert "date,value" not in json.dumps(payload)
