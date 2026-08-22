"""Empirical h-day price-delta distribution.

Shared statistical grounding for any predictor that needs to translate an
LLM's qualitative/probabilistic view into a real dollar shift without
trusting a number the LLM invented: the realized distribution of how far a
series has actually moved over each horizon, historically. Deliberately not
a model-fitted quantity (no ARIMA, no residuals) — cheaper than a
residual-based governor, and not circular (it isn't derived from the same
model whose output it's meant to keep honest).

Used by :mod:`energy_oil_forecasting.cfm_agent_v_5_2_2_delta_governed` (maps
a discrete rank to a target percentile) and
:mod:`energy_oil_forecasting.scenario_schema_anchored` (maps a
probability-weighted scenario price to a target percentile) — both agents
share this module rather than one importing the other's package.
"""

from __future__ import annotations

import numpy as np
from aieng.forecasting.data.context import ForecastContext


_PERCENTILE_LEVELS = (10, 25, 50, 75, 90)


def compute_horizon_delta_percentiles(
    context: ForecastContext,
    series_id: str,
    horizons: list[int],
) -> dict[int, dict[int, float]]:
    """For each horizon h, the empirical {10,25,50,75,90}th percentiles of
    historical h-day price deltas: ``price[t] - price[t-h]``, over all t
    available as of the cutoff.

    Parameters
    ----------
    context : ForecastContext
        Supplies the cutoff-safe price history via ``get_series``.
    series_id : str
        Target series id (e.g. the WTI series).
    horizons : list[int]
        Horizons (in series steps) to compute delta distributions for.

    Returns
    -------
    dict[int, dict[int, float]]
        Maps horizon -> {percentile level -> delta value}.
    """
    df = context.get_series(series_id)
    prices = df.sort_values("timestamp")["value"].to_numpy(dtype=float)

    result: dict[int, dict[int, float]] = {}
    for horizon in horizons:
        if len(prices) <= horizon:
            raise RuntimeError(f"Not enough history ({len(prices)} obs) to compute {horizon}-day deltas.")
        deltas = prices[horizon:] - prices[:-horizon]
        percentile_values = np.percentile(deltas, _PERCENTILE_LEVELS)
        result[horizon] = dict(zip(_PERCENTILE_LEVELS, (float(value) for value in percentile_values), strict=True))
    return result


__all__ = ["compute_horizon_delta_percentiles"]
