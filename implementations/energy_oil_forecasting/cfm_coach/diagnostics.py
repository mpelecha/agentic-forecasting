"""The cutoff-state feature vector, shared by live runs and historical cases.

A "state" is how the market looked on one business day, described by exactly the
fields ``cfm_agent_v_5_0`` already computes at every cutoff. That shared representation
is the whole point: the trust report compares today's state against history, so today
and history must be described identically or the comparison is meaningless.

Identity is guaranteed by *calling v5.0's own* ``compute_market_diagnostics`` over
truncated history rather than reimplementing it. A reimplementation would drift.

**Deliberately not routed through `AgentTarget`,** unlike the replay engine. The state
vector is a description of the *market*, not of an agent, and the trust report's whole
premise is comparing today against history in one vocabulary. Giving each stream its
own diagnostics function would let two streams disagree about what a given day looked
like -- and v5.2's copy is byte-identical to v5.0's apart from its import line, so
there is nothing to gain by it. If the two ever genuinely diverge, that is the moment
to lift this to a shared module rather than to fork it per stream.

Nothing here involves an LLM. This is pandas over a cached parquet.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from aieng.forecasting.data import DataService
from energy_oil_forecasting.cfm_agent_v_5_0.diagnostics import compute_market_diagnostics


#: The state vector. Six target-derived fields plus two covariates that carry
#: information the price path alone does not: the futures-curve shape
#: (backwardation signals physical tightness) and broad risk sentiment.
STATE_FEATURES: tuple[str, ...] = (
    "return_1b",
    "return_5b",
    "return_21b",
    "realized_volatility_21b",
    "drawdown_63b",
    "jump_zscore_63b",
    "oil_curve_contango_l1b",
    "vix_level_l1b",
)

# Far enough ahead that `get_series` returns everything; history is then truncated
# per-date by `released_at`, which is what the cutoff enforcer does for a live run.
_ALL_HISTORY = datetime(2100, 1, 1)


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def state_from_diagnostics(diagnostics: Mapping[str, Any]) -> dict[str, float | None]:
    """Extract the state vector from a stored `market_diagnostics` block.

    Used for live runs, where v5.0 already computed the diagnostics and the
    `RunRecord` carries them -- so the state costs nothing to recover.
    """
    covariates = diagnostics.get("covariate_latest_values") or {}
    merged = {**dict(diagnostics), **dict(covariates)}
    return {name: _finite_or_none(merged.get(name)) for name in STATE_FEATURES}


def _visible_at(frame: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Rows a predictor could have seen on `as_of`, by release date not observation date.

    `released_at` is the next business day, so the close for `as_of` itself is not
    yet visible on `as_of` -- the same rule `CutoffEnforcer` applies live.
    """
    column = "released_at" if "released_at" in frame.columns else "timestamp"
    return frame[frame[column] <= as_of]


def historical_states(
    service: DataService,
    *,
    target_series_id: str = "wti_crude_oil_price",
    covariate_series_ids: tuple[str, ...] = ("oil_curve_contango_l1b", "vix_level_l1b"),
    start: str | None = None,
) -> pd.DataFrame:
    """Compute the state vector for every business day with complete data.

    Returns a frame indexed by `as_of` date with one column per `STATE_FEATURES`
    entry. Rows missing any feature are dropped: an incomplete state cannot be
    compared against a complete one, and silently imputing would invent history.
    """
    target = service.get_series(target_series_id, as_of=_ALL_HISTORY).sort_values("timestamp")
    covariates = {
        series_id: service.get_series(series_id, as_of=_ALL_HISTORY).sort_values("timestamp")
        for series_id in covariate_series_ids
        if series_id in service.series_ids
    }

    dates = pd.to_datetime(target["timestamp"]).dt.normalize().unique()
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]

    rows: list[dict[str, Any]] = []
    for as_of in dates:
        visible_target = _visible_at(target, as_of)
        if visible_target.empty:
            continue
        diagnostics = compute_market_diagnostics(
            visible_target,
            as_of=as_of.to_pydatetime(),
            covariates={key: _visible_at(frame, as_of) for key, frame in covariates.items()},
        )
        rows.append({"as_of": as_of.date(), **state_from_diagnostics(diagnostics.model_dump(mode="json"))})

    frame = pd.DataFrame(rows).set_index("as_of")
    return frame.dropna(how="any")


def forward_moves(
    service: DataService,
    horizons: tuple[int, ...] = (5, 10, 21),
    *,
    target_series_id: str = "wti_crude_oil_price",
) -> pd.DataFrame:
    """Return what the price actually did over each horizon, indexed by the same `as_of` date.

    Joined to `historical_states`, this is the case base: how the market looked, and
    what happened next.
    """
    target = service.get_series(target_series_id, as_of=_ALL_HISTORY).sort_values("timestamp")
    values = pd.to_numeric(target["value"], errors="coerce")
    index = pd.to_datetime(target["timestamp"]).dt.normalize()

    frame = pd.DataFrame({"level": values.to_numpy()}, index=index.dt.date)
    # `as_of` cannot see its own close, so the move runs from the last *visible*
    # level -- the previous business day -- to the level `horizon` business days
    # ahead, which is exactly the target date v5.0 forecasts for that horizon.
    base = frame["level"].shift(1)
    for horizon in horizons:
        frame[f"move_{horizon}b"] = frame["level"].shift(-horizon) - base
    return frame.drop(columns="level")


__all__ = ["STATE_FEATURES", "forward_moves", "historical_states", "state_from_diagnostics"]
