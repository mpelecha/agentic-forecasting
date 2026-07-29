"""Identifier-plus-compressed-history prompt builder for ``cfm_agent_v_2_1``.

This package has no quant-model tool, so the prompt carries compressed
price history directly — unlike ``cfm_agent_v_2_0``, there is no
``query_market_data`` call to fetch it from. ``compress_history`` below is
reimplemented in this package rather than imported from
``analyst_agent.agent`` (read-only reference; see the caller's spec).
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES
from aieng.forecasting.evaluation.task import ForecastingTask
from energy_oil_forecasting.cfm_agent_v_2_1.outputs import WtiGeoForecastOutput


def compress_history(df: pd.DataFrame) -> str:
    """Compress WTI daily history to stay within context limits.

    Three-tier compression, most granular near the forecast origin:
    - last 63 trading days: daily bars
    - 63 trading days to 1 year back: weekly averages
    - older than 1 year: quarterly averages

    The CSV header is ``date,close``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ``timestamp`` and ``value``.

    Returns
    -------
    str
        CSV string with header ``date,close``.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    max_date = df["timestamp"].max()

    daily_cutoff = max_date - pd.tseries.offsets.BDay(63)
    weekly_cutoff = max_date - pd.DateOffset(years=1)

    daily = df[df["timestamp"] >= daily_cutoff].copy()
    weekly_band = df[(df["timestamp"] >= weekly_cutoff) & (df["timestamp"] < daily_cutoff)].copy()
    quarterly_band = df[df["timestamp"] < weekly_cutoff].copy()

    rows: list[str] = ["date,close"]

    if not quarterly_band.empty:
        quarterly_indexed = quarterly_band.set_index("timestamp")["value"]
        quarterly: pd.Series = quarterly_indexed.resample("QE").mean().dropna()
        for date, val in quarterly.items():
            rows.append(f"{date.date()},{val:.2f}")

    if not weekly_band.empty:
        weekly_indexed = weekly_band.set_index("timestamp")["value"]
        weekly: pd.Series = weekly_indexed.resample("W").mean().dropna()
        for date, val in weekly.items():
            rows.append(f"{date.date()},{val:.2f}")

    for _, row in daily.iterrows():
        rows.append(f"{row['timestamp'].date()},{row['value']:.2f}")

    return "\n".join(rows)


class CfmGeoPromptBuilder:
    """Serialize the task, compressed target history, and output schema."""

    def __init__(self) -> None:
        self._schema = WtiGeoForecastOutput.prompt_schema_json()

    def __call__(
        self,
        *,
        task: ForecastingTask,
        context: ForecastContext,
    ) -> str:
        """Build the identifier-plus-history forecasting payload."""
        df = context.get_series(task.target_series_id)
        compressed = compress_history(df)

        last_row = df.iloc[-1]
        last_close = float(last_row["value"])
        last_date = str(pd.Timestamp(last_row["timestamp"]).date())
        trailing_252 = df["value"].tail(252)

        payload: dict[str, Any] = {
            "agent_name": "cfm_agent_v_2_1",
            "task": task.task_id,
            "target_series_id": task.target_series_id,
            "as_of": str(context.as_of)[:10],
            "horizons": list(task.horizons),
            "frequency": task.frequency,
            "standard_quantiles": list(STANDARD_QUANTILES),
            "target_summary": {
                "last_close_usd_bbl": last_close,
                "last_date": last_date,
                "n_trading_days": int(len(df)),
                "52w_high": float(trailing_252.max()),
                "52w_low": float(trailing_252.min()),
            },
            "target_history_csv": compressed,
            "output_schema": self._schema,
        }

        return json.dumps(payload, indent=2)


__all__ = ["CfmGeoPromptBuilder", "compress_history"]
