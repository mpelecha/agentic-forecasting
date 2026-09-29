"""Empirical h-day price-move distribution.

Vendored (not imported) from ``energy_oil_forecasting.price_deltas`` so that
``scenario_schema_anchored`` / ``scenario_schema_anchored_enhanced`` do not
depend on a module shared with other, unrelated agent families (that
project-level module is deliberately shared between
``scenario_schema_anchored`` and ``cfm_agent_v_5_2_2_delta_governed`` — see
its own docstring). This copy exists purely so this package has no
dependency on anything outside itself and ``aieng.forecasting``.

Translates an LLM's qualitative/probabilistic view into a real dollar shift
without trusting a number the LLM invented: the realized distribution of how
far a series has actually moved over each horizon, historically. Built from
log returns (scale-free — a $4 move at $30 and a $4 move at $90 are not the
same event) and converted back to dollars at the latest price.
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
    ``price[t] - price[t-h]`` (``price_floor`` is not applied in that path).

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
