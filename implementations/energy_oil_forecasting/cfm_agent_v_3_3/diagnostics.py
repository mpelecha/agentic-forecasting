"""Pure-Python deterministic market-state diagnostics for CFM v3."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from energy_oil_forecasting.cfm_agent_v_3_3.schemas import MarketDiagnostics


def _return(values: pd.Series, periods: int) -> float | None:
    if len(values) <= periods:
        return None
    prior = float(values.iloc[-periods - 1])
    latest = float(values.iloc[-1])
    return None if prior == 0 else latest / prior - 1.0


def compute_market_diagnostics(
    target: pd.DataFrame,
    *,
    as_of: datetime,
    covariates: dict[str, pd.DataFrame] | None = None,
) -> MarketDiagnostics:
    """Compute stable diagnostics solely from cutoff-safe frames."""
    if target.empty:
        return MarketDiagnostics()
    frame = target.copy().sort_values("timestamp")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    values = pd.to_numeric(frame["value"], errors="coerce").dropna()
    if values.empty:
        return MarketDiagnostics()

    latest_timestamp = pd.Timestamp(frame.loc[values.index[-1], "timestamp"])
    returns = values.pct_change().dropna()
    vol = float(returns.tail(21).std(ddof=0) * np.sqrt(252)) if len(returns) >= 2 else None
    trailing = values.tail(63)
    drawdown = float(values.iloc[-1] / trailing.max() - 1.0) if not trailing.empty else None
    jump_z = None
    if len(returns) >= 10:
        window = returns.tail(63)
        std = float(window.std(ddof=0))
        if std > 0:
            jump_z = float((returns.iloc[-1] - window.mean()) / std)

    cov_latest: dict[str, float] = {}
    for series_id, covariate in (covariates or {}).items():
        if not covariate.empty:
            value = pd.to_numeric(covariate["value"], errors="coerce").dropna()
            if not value.empty:
                cov_latest[series_id] = float(value.iloc[-1])

    return MarketDiagnostics(
        latest_observation_date=str(latest_timestamp.date()),
        latest_value=float(values.iloc[-1]),
        calendar_gap_days=max(0, (pd.Timestamp(as_of).date() - latest_timestamp.date()).days),
        return_1b=_return(values, 1),
        return_5b=_return(values, 5),
        return_21b=_return(values, 21),
        realized_volatility_21b=vol,
        drawdown_63b=drawdown,
        jump_zscore_63b=jump_z,
        covariate_latest_values=cov_latest,
    )


__all__ = ["compute_market_diagnostics"]
