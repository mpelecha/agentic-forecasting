"""Cutoff-safety tests for the data-service tool."""

from __future__ import annotations

import json

from energy_oil_forecasting.cfm_agent_v_2_0.tools import MarketDataTool


def test_tool_never_returns_post_cutoff_rows(synthetic_service) -> None:
    cutoff = "2021-01-15"
    result = json.loads(
        MarketDataTool(synthetic_service).query_market_data(
            operation="get_series",
            cutoff_date=cutoff,
            series_ids=["wti_crude_oil_price"],
            lookback=500,
        )
    )
    dates = [row["date"] for row in result["series"][0]["observations"]]
    assert dates
    assert max(dates) <= cutoff
