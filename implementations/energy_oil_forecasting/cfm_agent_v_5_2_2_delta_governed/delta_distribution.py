"""Empirical h-day price-delta distribution.

The statistical governor for CFM Agent v5.2.2 Delta-Governed's LLM-proposed
center shift and uncertainty width — deliberately not a model-fitted
quantity (no ARIMA, no residuals): just the realized distribution of how far
WTI has actually moved over each horizon, historically. Cheaper than a
residual-based governor and not circular (it isn't derived from the same
model whose output it's meant to keep honest).
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
