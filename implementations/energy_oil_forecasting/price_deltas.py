"""Empirical h-day price-move distribution.

Shared statistical grounding for any predictor that needs to translate an
LLM's qualitative/probabilistic view into a real dollar shift without
trusting a number the LLM invented: the realized distribution of how far a
series has actually moved over each horizon, historically. Deliberately not
a model-fitted quantity (no ARIMA, no residuals) — cheaper than a
residual-based governor, and not circular (it isn't derived from the same
model whose output it's meant to keep honest).

By default the distribution is built from **log returns**, then converted
back to dollars at the latest price. Raw dollar deltas pool moves taken at
wildly different price levels — a $4 move when WTI was $30 (a 13% move) and
a $4 move when WTI was $90 (4.4%) are not the same event, but a dollar-delta
percentile table treats them as identical. Log returns are scale-free, so
the percentile table describes *proportional* moves, and the conversion back
to dollars re-expresses them at today's level.

Set ``log_returns=False`` to recover the original dollar-delta behaviour
exactly (used to reproduce pre-change results for paired comparison).

Used by :mod:`energy_oil_forecasting.cfm_agent_v_5_2_2_delta_governed` (maps
a discrete rank to a target percentile) and
:mod:`energy_oil_forecasting.scenario_schema_anchored` (maps a
probability-weighted scenario price to a target percentile) — both agents
share this module rather than one importing the other's package.
"""

from __future__ import annotations

import numpy as np
from aieng.forecasting.data.context import ForecastContext


PERCENTILE_LEVELS = (10, 25, 50, 75, 90)

# WTI printed negative in April 2020, and log() of a non-positive price is
# undefined. Floor the input series before taking logs so one historical
# outlier cannot poison the whole percentile table.
DEFAULT_PRICE_FLOOR = 1.0


def compute_horizon_delta_percentiles(
    context: ForecastContext,
    series_id: str,
    horizons: list[int],
    *,
    log_returns: bool = True,
    price_floor: float = DEFAULT_PRICE_FLOOR,
) -> dict[int, dict[int, float]]:
    """For each horizon h, the empirical {10,25,50,75,90}th percentiles of
    historical h-day price moves, expressed in dollars at the latest price.

    With ``log_returns=True`` (the default) the percentiles are computed on
    ``log(price[t]) - log(price[t-h])`` and then converted back to a dollar
    move at the latest price: ``latest * (exp(pct) - 1)``. Because ``exp`` is
    monotonic, taking percentiles in log space and exponentiating is
    equivalent to taking percentiles of the exponentiated series, so no
    ordering is disturbed.

    With ``log_returns=False`` the percentiles are the raw dollar deltas
    ``price[t] - price[t-h]``, bit-identical to this function's original
    behaviour (``price_floor`` is not applied in that path, so legacy results
    reproduce exactly).

    Parameters
    ----------
    context : ForecastContext
        Supplies the cutoff-safe price history via ``get_series``.
    series_id : str
        Target series id (e.g. the WTI series).
    horizons : list[int]
        Horizons (in series steps) to compute move distributions for.
    log_returns : bool, default=True
        Build the distribution from scale-free log returns rather than raw
        dollar deltas.
    price_floor : float, default=1.0
        Lower bound applied to the price history before taking logs. Ignored
        when ``log_returns=False``.

    Returns
    -------
    dict[int, dict[int, float]]
        Maps horizon -> {percentile level -> dollar move}.
    """
    df = context.get_series(series_id)
    prices = df.sort_values("timestamp")["value"].to_numpy(dtype=float)

    if log_returns:
        floored = np.maximum(prices, price_floor)
        log_prices = np.log(floored)
        latest_price = float(floored[-1])

    result: dict[int, dict[int, float]] = {}
    for horizon in horizons:
        if len(prices) <= horizon:
            raise RuntimeError(f"Not enough history ({len(prices)} obs) to compute {horizon}-day moves.")
        if log_returns:
            log_deltas = log_prices[horizon:] - log_prices[:-horizon]
            percentile_values = latest_price * (np.exp(np.percentile(log_deltas, PERCENTILE_LEVELS)) - 1.0)
        else:
            deltas = prices[horizon:] - prices[:-horizon]
            percentile_values = np.percentile(deltas, PERCENTILE_LEVELS)
        result[horizon] = dict(zip(PERCENTILE_LEVELS, (float(value) for value in percentile_values), strict=True))
    return result


__all__ = ["DEFAULT_PRICE_FLOOR", "PERCENTILE_LEVELS", "compute_horizon_delta_percentiles"]
