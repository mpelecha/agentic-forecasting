"""Agent composition and prompt contract tests."""

from __future__ import annotations

import json
from datetime import datetime

from energy_oil_forecasting.cfm_agent_v_2_2 import (
    AGENT_NAME,
    build_cfm_agent_v22_config,
)
from energy_oil_forecasting.cfm_agent_v_2_2.prompts import CfmEventScorePromptBuilder


def test_agent_has_exact_identity_tools_and_skills() -> None:
    config = build_cfm_agent_v22_config()
    assert config.name == AGENT_NAME == "cfm_agent_v_2_2"
    assert config.context_retrieval.enabled is True
    assert config.context_retrieval.enforce_cutoff is True
    assert not (config.code_execution and config.code_execution.enabled)
    assert not config.function_tools
    assert [path.name for path in config.skills_dirs] == ["event-context-analysis"]


def test_prompt_carries_scoring_contract_and_no_forecast_fields(synthetic_service) -> None:
    context = synthetic_service.context(datetime(2021, 6, 1))
    payload = json.loads(CfmEventScorePromptBuilder()(context=context))
    assert payload["agent_name"] == "cfm_agent_v_2_2"
    assert payload["task"] == "wti_event_context_scores"
    assert payload["as_of"] == "2021-06-01"
    assert "output_schema" in payload
    assert payload["target_history_csv"].startswith("date,close")
    # A scorer has no forecast contract to carry.
    assert "horizons" not in payload
    assert "standard_quantiles" not in payload
