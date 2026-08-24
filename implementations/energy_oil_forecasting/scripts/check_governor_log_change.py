"""Diagnostic: what the log-return governor change does on real WTI data.

Answers three questions the synthetic tests could not:

1. How many observations actually sit at/below the price floor, and how many
   h-day windows they contaminate.
2. What the old (dollar-delta) and new (log-return) percentile tables
   actually look like at a real origin.
3. Whether the artificial log returns created by flooring the April-2020
   negative print reach the p10..p90 levels the governor actually uses --
   checked by recomputing with those windows masked out entirely and
   diffing against the floored table.

Run from the implementations/energy_oil_forecasting directory:

    python scripts/check_governor_log_change.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from energy_oil_forecasting.data import WTI_SERIES_ID, build_wti_multivariate_service
from energy_oil_forecasting.price_deltas import (
    DEFAULT_PRICE_FLOOR,
    PERCENTILE_LEVELS,
    compute_horizon_delta_percentiles,
)


ORIGIN = pd.Timestamp("2026-05-25")  # last biweekly eval origin
HORIZONS = [5, 10, 21]


def _masked_log_table(prices: np.ndarray, horizons: list[int], floor: float) -> dict[int, dict[int, float]]:
    """Log-return percentiles with every window touching a floored observation dropped."""
    needs_floor = prices <= floor
    floored = np.maximum(prices, floor)
    log_prices = np.log(floored)
    latest = float(floored[-1])

    table: dict[int, dict[int, float]] = {}
    for horizon in horizons:
        log_deltas = log_prices[horizon:] - log_prices[:-horizon]
        # Drop any window whose start or end endpoint had to be floored.
        contaminated = needs_floor[horizon:] | needs_floor[:-horizon]
        clean = log_deltas[~contaminated]
        values = latest * (np.exp(np.percentile(clean, PERCENTILE_LEVELS)) - 1.0)
        table[horizon] = dict(zip(PERCENTILE_LEVELS, (float(v) for v in values), strict=True))
    return table


def main() -> None:
    service = build_wti_multivariate_service()
    context = service.context(as_of=ORIGIN)
    prices = context.get_series(WTI_SERIES_ID).sort_values("timestamp")["value"].to_numpy(dtype=float)

    n_bad = int((prices <= DEFAULT_PRICE_FLOOR).sum())
    print(f"history: {len(prices)} observations up to {ORIGIN.date()}")
    print(f"min price: ${prices.min():.2f}   latest: ${prices[-1]:.2f}")
    print(f"observations at or below the ${DEFAULT_PRICE_FLOOR:.2f} floor: {n_bad}")
    for horizon in HORIZONS:
        contaminated = 2 * n_bad * horizon
        print(f"  h={horizon:>2}d: up to ~{contaminated} of {len(prices) - horizon} windows touch one "
              f"({100 * contaminated / (len(prices) - horizon):.2f}% -> {100 * contaminated / (len(prices) - horizon) / 2:.2f}% per tail)")
    print()

    old = compute_horizon_delta_percentiles(context, WTI_SERIES_ID, HORIZONS, log_returns=False)
    new = compute_horizon_delta_percentiles(context, WTI_SERIES_ID, HORIZONS)
    masked = _masked_log_table(prices, HORIZONS, DEFAULT_PRICE_FLOOR)

    latest = float(np.maximum(prices, DEFAULT_PRICE_FLOOR)[-1])
    for horizon in HORIZONS:
        print(f"h={horizon}d percentile table (dollars; latest price ${latest:.2f})")
        print(f"  {'level':>6} {'OLD dollar':>12} {'NEW log':>12} {'NEW masked':>12} {'masked-vs-new':>14}")
        for level in PERCENTILE_LEVELS:
            drift = masked[horizon][level] - new[horizon][level]
            print(
                f"  {level:>6} {old[horizon][level]:>12.2f} {new[horizon][level]:>12.2f} "
                f"{masked[horizon][level]:>12.2f} {drift:>14.4f}"
            )
        # What the governor actually consumes: distance from the median.
        print(f"  -> rank=+2 shift (p90-p50):  OLD ${old[horizon][90] - old[horizon][50]:+.2f}   "
              f"NEW ${new[horizon][90] - new[horizon][50]:+.2f}")
        print(f"  -> rank=-2 shift (p10-p50):  OLD ${old[horizon][10] - old[horizon][50]:+.2f}   "
              f"NEW ${new[horizon][10] - new[horizon][50]:+.2f}")
        print(f"  -> empirical width (p90-p10): OLD ${old[horizon][90] - old[horizon][10]:.2f}   "
              f"NEW ${new[horizon][90] - new[horizon][10]:.2f}")
        print()

    worst = max(
        abs(masked[h][level] - new[h][level]) for h in HORIZONS for level in PERCENTILE_LEVELS
    )
    print(f"largest masked-vs-floored difference across every level and horizon: ${worst:.4f}")
    if worst < 0.01:
        print("=> flooring artifacts do NOT reach p10..p90; the floor is safe as implemented.")
    else:
        print("=> flooring artifacts DO move the table; masking contaminated windows is worth adopting.")


if __name__ == "__main__":
    main()
