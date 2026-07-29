"""Tool configuration and market-data behavior tests."""

from __future__ import annotations

import json

import pandas as pd
from energy_oil_forecasting.cfm_agent_v_2_0.tools import (
    MarketDataTool,
    build_code_execution_config,
    build_verified_search_config,
)


def test_verified_search_and_e2b_are_enabled() -> None:
    search = build_verified_search_config()
    code = build_code_execution_config()
    assert search.enabled is True
    assert search.enforce_cutoff is True
    assert search.verifier_max_attempts >= 1
    assert code.enabled is True
    assert code.template_name


def test_market_data_lists_and_returns_bounded_history(synthetic_service) -> None:
    tool = MarketDataTool(synthetic_service)
    raw = tool.query_market_data(
        operation="get_series",
        cutoff_date="2021-06-01",
        series_ids=["wti_crude_oil_price", "vix_level_l1b"],
        lookback=10,
    )
    result = json.loads(raw)
    assert result["status"] == "ok"
    assert len(result["series"]) == 2
    assert all(item["returned_observations"] == 10 for item in result["series"])
    assert all(item["last_available_date"] <= "2021-06-01" for item in result["series"])


def test_market_data_rejects_bad_operation(synthetic_service) -> None:
    result = json.loads(
        MarketDataTool(synthetic_service).query_market_data(
            operation="unknown",
            cutoff_date="2021-06-01",
            series_ids=[],
        )
    )
    assert result["status"] == "error"
    assert "operation" in result["error"]


def test_training_audit_exposes_model_data_usage(synthetic_service) -> None:
    tool = MarketDataTool(
        synthetic_service,
        covariate_series_ids=["vix_level_l1b"],
    )
    context = synthetic_service.context(pd.Timestamp("2021-06-01").to_pydatetime())
    audit = tool._training_audit(context, "wti_crude_oil_price", 21)
    assert audit["arima"].target_observations > 0
    assert audit["kalman"].target_observations == audit["arima"].target_observations
    assert audit["lightgbm"].covariates == ["vix_level_l1b"]
    assert audit["lightgbm"].aligned_observations > 0
    assert audit["lightgbm"].effective_training_examples_estimate > 0
    assert audit["lightgbm"].target_lags == 21
