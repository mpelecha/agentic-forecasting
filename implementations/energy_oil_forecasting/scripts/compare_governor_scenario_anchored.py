"""Paired offline comparison for Scenario Schema Anchored: dollar vs log governor.

Companion to ``compare_governor_log_vs_dollar.py``, which covered
cfm_agent_v_5_2_2_delta_governed only. Both agents share the governor, so
both needed checking.

Same paired logic: the delta-percentile table is computed *after* the LLM
call in this predictor (see ``ScenarioSchemaAnchoredPredictor.predict``) and
never enters the prompt, so for a fixed cached LLM response the two
governors differ deterministically. Identical scenarios on both sides, only
the governor swapped -- no LLM calls, no sampling variance.

Reconstruction is fiddlier than the delta-governed case: the implied target
percentile is derived from the *longest* horizon's ARIMA anchor, which lives
in a different prediction's metadata than the one being recomputed, so
predictions are grouped by origin first.

Self-check: recomputing with ``log_returns=False`` must reproduce the cached
payload quantiles exactly, otherwise the reconstruction is wrong and the
comparison is meaningless.

Run from the implementations/energy_oil_forecasting directory:

    python scripts/compare_governor_scenario_anchored.py            # 2026 eval (default)
    python scripts/compare_governor_scenario_anchored.py backtest   # 2025 backtest
    python scripts/compare_governor_scenario_anchored.py 10yr       # 2014-2024 quarterly backtest
    python scripts/compare_governor_scenario_anchored.py 10yr-eval  # held-out half of the same grid
"""

from __future__ import annotations

import sys
from collections import defaultdict
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path

import pandas as pd
import yaml
from aieng.forecasting.evaluation.backtest import _crps_for_prediction
from aieng.forecasting.evaluation.prediction import ContinuousForecast, Prediction
from energy_oil_forecasting.data import WTI_SERIES_ID, build_wti_multivariate_service
from energy_oil_forecasting.price_deltas import compute_horizon_delta_percentiles

# The real post-LLM math, imported rather than reimplemented.
from energy_oil_forecasting.scenario_schema_anchored.predictor import (
    _grounded_center_shift,
    _implied_target_percentile,
    _probability_weighted_scenario_price,
    _widen_toward_scenarios,
)


WINDOW_DIRS = {
    "eval": "data/predictions/energy_oil_eval_biweekly",
    "backtest": "data/predictions/energy_oil_backtest_biweekly",
    "10yr": "data/predictions/energy_oil_backtest_10yr_quarterly",
    "10yr-eval": "data/predictions/energy_oil_eval_10yr_quarterly",
}
# The 10yr quarterly spec adds a fourth horizon; the biweekly specs stop at 21.
# max_horizon matters here beyond bookkeeping: the implied target percentile is
# read off the longest horizon's ARIMA anchor, so it must track the spec.
WINDOW_HORIZONS = {"10yr": [5, 10, 21, 63], "10yr-eval": [5, 10, 21, 63]}
CACHE_STEM = "agent_predictor_wti_analyst_news_scenario_schema_anchored_gemini-3.1-flash-lite-preview_" "continuous__wti_oil_price_forecast.yaml"
HORIZONS = [5, 10, 21]


def _float_keys(quantiles: dict) -> dict[float, float]:
    return {float(k): float(v) for k, v in quantiles.items()}


def _horizon_for(as_of: pd.Timestamp, forecast_date: pd.Timestamp, horizons: list[int]) -> int | None:
    offset = pd.tseries.offsets.BDay()
    for horizon in horizons:
        if (as_of + offset * horizon).normalize() == forecast_date.normalize():
            return horizon
    return None


def _score(quantiles: dict[float, float], point: float, actual: float, pred: dict) -> tuple[float, float]:
    prediction = Prediction(
        predictor_id="recompute",
        task_id=pred["task_id"],
        issued_at=datetime.now(tz=timezone.utc).replace(tzinfo=None),
        as_of=pd.Timestamp(pred["as_of"]).to_pydatetime(),
        forecast_date=pd.Timestamp(pred["forecast_date"]).to_pydatetime(),
        payload=ContinuousForecast(point_forecast=point, quantiles=quantiles),
    )
    return _crps_for_prediction(prediction, actual), float(quantiles[0.1] <= actual <= quantiles[0.9])


def main() -> None:
    window = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if window not in WINDOW_DIRS:
        raise SystemExit(f"usage: {sys.argv[0]} [{'|'.join(WINDOW_DIRS)}]")
    horizons = WINDOW_HORIZONS.get(window, HORIZONS)
    max_horizon = max(horizons)
    cache = Path(WINDOW_DIRS[window]) / CACHE_STEM
    print(f"window: {window}  ({cache})")
    data = yaml.safe_load(cache.read_text())
    service = build_wti_multivariate_service()
    resolved_as_of = datetime.now(tz=timezone.utc).replace(tzinfo=None)
    actual_df = service.get_series(WTI_SERIES_ID, as_of=resolved_as_of)
    actuals = {pd.Timestamp(r["timestamp"]).normalize(): float(r["value"]) for _, r in actual_df.iterrows()}

    by_origin: dict[str, list[dict]] = defaultdict(list)
    for pred in data["predictions"]:
        by_origin[str(pred["as_of"])[:10]].append(pred)

    rows = []
    mismatches = 0
    for origin, preds in sorted(by_origin.items()):
        context = service.context(as_of=pd.Timestamp(origin))
        tables = {
            "dollar": compute_horizon_delta_percentiles(context, WTI_SERIES_ID, horizons, log_returns=False),
            "log": compute_horizon_delta_percentiles(context, WTI_SERIES_ID, horizons),
        }

        # Anchor grid for this origin, indexed by horizon.
        anchors: dict[int, dict] = {}
        for pred in preds:
            horizon = _horizon_for(pd.Timestamp(pred["as_of"]), pd.Timestamp(pred["forecast_date"]), horizons)
            if horizon is not None:
                anchors[horizon] = pred["metadata"]["arima_anchor"]
        if max_horizon not in anchors:
            print(f"  (skipping {origin}: no h={max_horizon} anchor cached)")
            continue

        scenarios = preds[0]["metadata"].get("scenarios") or []
        if not scenarios:
            print(f"  (skipping {origin}: no scenarios cached)")
            continue
        scenario_low = min(s["price_low"] for s in scenarios)
        scenario_high = max(s["price_high"] for s in scenarios)
        weighted_price = _probability_weighted_scenario_price(scenarios)
        target_percentile = _implied_target_percentile(
            weighted_price, _float_keys(anchors[max_horizon]["quantiles"])
        )

        for pred in preds:
            horizon = _horizon_for(pd.Timestamp(pred["as_of"]), pd.Timestamp(pred["forecast_date"]), horizons)
            if horizon is None:
                continue
            actual = actuals.get(pd.Timestamp(pred["forecast_date"]).normalize())
            if actual is None:
                continue
            anchor = anchors[horizon]
            anchor_quantiles = _float_keys(anchor["quantiles"])
            anchor_point = float(anchor["point_forecast"])
            scale = sqrt(horizon / max_horizon)

            out = {}
            for label in ("dollar", "log"):
                shift = _grounded_center_shift(target_percentile, tables[label][horizon])
                shifted = {level: value + shift for level, value in anchor_quantiles.items()}
                final = _widen_toward_scenarios(shifted, scenario_low, scenario_high, scale=scale)
                out[label] = (anchor_point + shift, final, shift)

            cached = _float_keys(pred["payload"]["quantiles"])
            if any(abs(out["dollar"][1][q] - cached[q]) > 1e-6 for q in cached):
                mismatches += 1

            crps_old, in_old = _score(out["dollar"][1], out["dollar"][0], actual, pred)
            crps_new, in_new = _score(out["log"][1], out["log"][0], actual, pred)
            rows.append(
                {
                    "origin": origin,
                    "horizon": horizon,
                    "shift_old": out["dollar"][2],
                    "shift_new": out["log"][2],
                    "width_old": out["dollar"][1][0.9] - out["dollar"][1][0.1],
                    "width_new": out["log"][1][0.9] - out["log"][1][0.1],
                    "crps_old": crps_old,
                    "crps_new": crps_new,
                    "in_old": in_old,
                    "in_new": in_new,
                }
            )

    df = pd.DataFrame(rows)
    print(f"\nscored {len(df)} predictions from {df['origin'].nunique()} origins\n")
    if mismatches:
        print(f"!! SELF-CHECK FAILED: {mismatches}/{len(df)} dollar-path recomputes do not match the cache.")
        print("   The reconstruction is wrong; the comparison below is NOT trustworthy.\n")
    else:
        print("self-check OK: dollar-path recompute reproduces the cached quantiles exactly.\n")

    print("=" * 78)
    print("SCENARIO SCHEMA ANCHORED — paired, same LLM scenarios, governor swapped")
    print("=" * 78)
    print(f"{'':18} {'DOLLAR (cached)':>18} {'LOG (new)':>16} {'change':>12}")
    print(f"{'mean CRPS':18} {df['crps_old'].mean():>18.4f} {df['crps_new'].mean():>16.4f} "
          f"{df['crps_new'].mean() - df['crps_old'].mean():>+12.4f}")
    print(f"{'80% coverage %':18} {100 * df['in_old'].mean():>18.1f} {100 * df['in_new'].mean():>16.1f} "
          f"{100 * (df['in_new'].mean() - df['in_old'].mean()):>+12.1f}")
    print(f"{'mean width':18} {df['width_old'].mean():>18.2f} {df['width_new'].mean():>16.2f} "
          f"{df['width_new'].mean() - df['width_old'].mean():>+12.2f}")
    print(f"{'mean |shift|':18} {df['shift_old'].abs().mean():>18.2f} {df['shift_new'].abs().mean():>16.2f} "
          f"{df['shift_new'].abs().mean() - df['shift_old'].abs().mean():>+12.2f}")

    print("\nby horizon:")
    for horizon in horizons:
        sub = df[df["horizon"] == horizon]
        if sub.empty:
            continue
        print(
            f"  h={horizon:>2}d  CRPS {sub['crps_old'].mean():7.4f} -> {sub['crps_new'].mean():7.4f}"
            f"   coverage {100 * sub['in_old'].mean():5.1f}% -> {100 * sub['in_new'].mean():5.1f}%"
            f"   width {sub['width_old'].mean():6.2f} -> {sub['width_new'].mean():6.2f}"
        )


if __name__ == "__main__":
    main()
