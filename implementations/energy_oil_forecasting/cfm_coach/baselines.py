"""Naive reference forecasts -- the floors a forecaster has to clear to be worth anything.

``ensemble`` was introduced as "the floor the overlay must beat", but on a near-driftless
series it is not the floor. The forecast that is genuinely hard to beat is *today's
price*: WTI's mean forward move is within a few cents of zero at every horizon, and no
price-based signal measured in this repo beats "no change" on direction. So the
scoreboard needs that row before any other row can be read as skill.

The random walk has two halves, and only one is a choice:

- **Centre.** The last close visible at the cutoff, exactly. Not a fitted drift -- a
  driftless walk is the claim being tested against.
- **Spread.** WTI's own empirical ``horizon``-business-day log moves over a trailing
  window, re-centred on zero and scaled to today's price. Log moves rather than dollar
  moves, so a $50 market and an $85 market get proportionally sized ranges; trailing
  rather than all-history, so the spread reflects the recent regime the way the
  numerical suite's own training window does.

**Cutoff honesty.** Only observations up to the record's ``latest_observation_date``
are used -- the last close the agent itself was shown. A record without that field
falls back to observations strictly before its cutoff, which is the same rule the
data service applies (each close is released the next business day).

No LLM, no network, no fitting: arithmetic over the cached price series.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from energy_oil_forecasting.cfm_coach.schemas import Quantiles, RunRecord


#: Trailing observations the spread is estimated from: ~2 years of business days,
#: matching `max_data_rows_per_series` in both agent packages.
RANDOM_WALK_WINDOW = 520

#: Below this many h-step moves the empirical tails are too thin to publish. The
#: baseline declines rather than inventing a spread.
MIN_RANDOM_WALK_MOVES = 60


class RandomWalkBaseline:
    """Today's price as the median, WTI's recent h-step moves as the spread."""

    baseline_id = "cfm_coach_random_walk_v1"

    def __init__(self, prices: pd.Series, *, window: int = RANDOM_WALK_WINDOW):
        """`prices` is the realized target series indexed by observation date, ascending."""
        self.prices = prices.sort_index()
        self.window = window

    def visible_history(self, record: RunRecord) -> pd.Series:
        """Observations the agent could have seen when `record` was issued."""
        latest = record.diagnostics.get("latest_observation_date")
        if latest:
            return self.prices[self.prices.index <= date.fromisoformat(str(latest)[:10])]
        return self.prices[self.prices.index < record.cutoff]

    def quantiles(
        self,
        history: pd.Series,
        *,
        last_value: float,
        horizon: int,
        levels: list[float],
    ) -> Quantiles | None:
        """Random-walk quantiles for one horizon, or None when history is too short.

        Non-positive prices (WTI settled at -$37 in April 2020) have no log and are
        dropped; the handful of adjacent moves lost to that is immaterial to a
        520-observation window.
        """
        if last_value <= 0:
            return None
        positive = history[history > 0].tail(self.window + horizon)
        log_levels = np.log(positive.to_numpy(dtype=float))
        if len(log_levels) - horizon < MIN_RANDOM_WALK_MOVES:
            return None

        moves = log_levels[horizon:] - log_levels[:-horizon]
        centre = float(np.quantile(moves, 0.5))
        quantiles = {
            float(level): float(last_value * np.exp(np.quantile(moves, float(level)) - centre))
            for level in sorted(levels)
        }
        # Exact by construction up to float rounding; pinned so the call is "no change" to the cent.
        if 0.5 in quantiles:
            quantiles[0.5] = float(last_value)
        return quantiles

    def for_record(self, record: RunRecord, horizon: int) -> tuple[float, Quantiles] | None:
        """(point forecast, quantiles) for one horizon of a stored run, on that run's own grid."""
        last_value = record.diagnostics.get("latest_value")
        if last_value is None:
            return None
        levels = list(record.horizon(horizon).ensemble_quantiles)
        quantiles = self.quantiles(
            self.visible_history(record),
            last_value=float(last_value),
            horizon=horizon,
            levels=levels,
        )
        if quantiles is None:
            return None
        return float(last_value), quantiles


__all__ = ["MIN_RANDOM_WALK_MOVES", "RANDOM_WALK_WINDOW", "RandomWalkBaseline"]
