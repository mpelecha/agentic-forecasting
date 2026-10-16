"""Deterministic, cutoff-safe market diagnostics (specification Section 2).

Every statistic is computed on the floored, windowed prices held by a
:class:`~energy_oil_forecasting.cfm_agent_v_6_0.history.PriceHistory`; windows
come from the settings; a window longer than the available history yields
``None``, never an error; no random numbers are used.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime

import numpy as np
import pandas as pd
from energy_oil_forecasting.cfm_agent_v_6_0.config import CfmV60Settings
from energy_oil_forecasting.cfm_agent_v_6_0.history import PriceHistory
from energy_oil_forecasting.cfm_agent_v_6_0.schemas import (
    CurveState,
    DataQuality,
    MarketDiagnostics,
)


#: Shared-data-layer identifiers of the two level covariates read for the current state.
CONTANGO_SERIES_ID = "oil_curve_contango_l1b"
VIX_SERIES_ID = "vix_level_l1b"


def _cumulative_return(prices: np.ndarray, window: int) -> float | None:
    if len(prices) <= window:
        return None
    prior = float(prices[-1 - window])
    return None if prior == 0 else float(prices[-1]) / prior - 1


def _realized_volatility(returns: np.ndarray, window: int, trading_days: int) -> float | None:
    if len(returns) < window:
        return None
    return float(np.std(returns[-window:], ddof=0) * np.sqrt(trading_days))


def _volatility_zscore(returns: np.ndarray, window: int) -> float | None:
    if len(returns) < window:
        return None
    rolling = pd.Series(returns).rolling(window).std(ddof=0).dropna().to_numpy()
    spread = float(np.std(rolling, ddof=0))
    if spread == 0 or not np.isfinite(spread):
        return None
    return float((rolling[-1] - np.mean(rolling)) / spread)


def _drawdown(prices: np.ndarray, window: int) -> float | None:
    if len(prices) < window:
        return None
    peak = float(np.max(prices[-window:]))
    return None if peak == 0 else float(prices[-1]) / peak - 1


def _jump_zscore(returns: np.ndarray, window: int) -> float | None:
    if len(returns) < window:
        return None
    recent = returns[-window:]
    spread = float(np.std(recent, ddof=0))
    if spread == 0 or not np.isfinite(spread):
        return None
    return float((returns[-1] - np.mean(recent)) / spread)


def curve_state_from(value: float | None, threshold: float) -> CurveState | None:
    """Label the curve from the latest log ratio: flat within the threshold, else by sign."""
    if value is None or not np.isfinite(value):
        return None
    if abs(value) < threshold:
        return "flat"
    return "contango" if value > 0 else "backwardation"


def compute_market_diagnostics(
    history: PriceHistory,
    *,
    as_of: datetime,
    settings: CfmV60Settings,
    model_disagreement: Mapping[int, float] | None = None,
    horizons: Iterable[int] | None = None,
) -> MarketDiagnostics:
    """Compute the Section 2 diagnostics from the floored windowed history.

    ``horizons`` selects the historical return profiles; when omitted the keys
    of ``model_disagreement`` are used, so the suite can pass either.
    """
    prices = np.asarray(history.prices, dtype=float)
    returns = np.asarray(history.returns, dtype=float)
    disagreement = {int(h): float(v) for h, v in (model_disagreement or {}).items()}
    profile_horizons = [int(h) for h in horizons] if horizons is not None else sorted(disagreement)
    covariate_latest = history.covariate_latest_values()
    latest_date = pd.Timestamp(history.dates[-1]) if len(history.dates) else None
    gap = max(0, (pd.Timestamp(as_of).date() - latest_date.date()).days) if latest_date is not None else None
    trading_days = settings.trading_days_per_year
    data_quality = DataQuality(
        cutoff_date=str(pd.Timestamp(as_of).date()),
        observation_counts={
            history.target_id: history.total_target_observations,
            **{
                series_id: history.covariates[series_id].observations
                for series_id in history.covariate_ids
                if not history.covariates[series_id].missing
            },
        },
        first_dates={
            history.target_id: history.first_served_date,
            **{series_id: history.covariates[series_id].first_date for series_id in history.covariate_ids},
        },
        last_dates={
            history.target_id: history.last_date,
            **{series_id: history.covariates[series_id].last_date for series_id in history.covariate_ids},
        },
        input_floor_activation_count=history.input_floor_activation_count,
        input_floor_activation_dates=list(history.input_floor_activation_dates),
        covariate_fill_counts=dict(history.covariate_fill_counts),
        rows_before_alignment=history.rows_before_alignment,
        rows_dropped=history.rows_dropped,
        rows_remaining=history.rows_remaining,
        missing_covariates=list(history.missing_covariates),
    )
    return MarketDiagnostics(
        latest_observation_date=history.last_date,
        latest_value=float(prices[-1]) if len(prices) else None,
        calendar_gap_days=gap,
        cumulative_return={n: _cumulative_return(prices, n) for n in settings.diagnostic_return_windows},
        realized_volatility={
            n: _realized_volatility(returns, n, trading_days) for n in settings.diagnostic_volatility_windows
        },
        volatility_zscore={n: _volatility_zscore(returns, n) for n in settings.diagnostic_volatility_windows},
        drawdown={n: _drawdown(prices, n) for n in settings.diagnostic_volatility_windows},
        jump_zscore={n: _jump_zscore(returns, n) for n in settings.diagnostic_jump_windows},
        covariate_latest_values=covariate_latest,
        curve_state=curve_state_from(covariate_latest.get(CONTANGO_SERIES_ID), settings.curve_flat_threshold),
        vix_level=covariate_latest.get(VIX_SERIES_ID),
        historical_profiles=history.profiles(profile_horizons),
        model_disagreement_std=disagreement,
        data_quality=data_quality,
    )


__all__ = ["CONTANGO_SERIES_ID", "VIX_SERIES_ID", "compute_market_diagnostics", "curve_state_from"]
