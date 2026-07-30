"""Prompt builder for the event-context scoring task.

The payload carries compressed price history for situational awareness
only — the recent level and the 52-week range help the agent judge what
kind of regime it is scoring. The agent must not restate price history
as a factor and must not output a price; the output schema has no price
field to put one in.

``compress_history`` is reimplemented here rather than imported from
``cfm_agent_v_2_1`` (packages stay standalone by convention).
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from energy_oil_forecasting.cfm_agent_v_2_2.config import AGENT_NAME
from energy_oil_forecasting.cfm_agent_v_2_2.outputs import WtiEventScoreOutput
from energy_oil_forecasting.data import WTI_SERIES_ID


def compress_history(df: pd.DataFrame) -> str:
    """Compress WTI daily history to stay within context limits.

    Three-tier compression, most granular near the scoring date:
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


class CfmEventScorePromptBuilder:
    """Serialize the scoring task, compressed target history, and output schema."""

    def __init__(self, target_series_id: str = WTI_SERIES_ID) -> None:
        self._target_series_id = target_series_id
        self._schema = WtiEventScoreOutput.prompt_schema_json()

    def __call__(self, *, context: ForecastContext) -> str:
        """Build the event-scoring payload for one origin date."""
        df = context.get_series(self._target_series_id)
        compressed = compress_history(df)

        last_row = df.iloc[-1]
        last_close = float(last_row["value"])
        last_date = str(pd.Timestamp(last_row["timestamp"]).date())
        trailing_252 = df["value"].tail(252)

        payload: dict[str, Any] = {
            "agent_name": AGENT_NAME,
            "task": "wti_event_context_scores",
            "target_series_id": self._target_series_id,
            "as_of": str(context.as_of)[:10],
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


__all__ = ["CfmEventScorePromptBuilder", "compress_history"]
