"""Paired offline test: should Anchored's band come from level or log-return ARIMA?

Anchored is currently a hybrid. Its band is AutoARIMA fitted on the price
*level*; its centre-shift cap is computed from *log-return* percentiles scaled
to the current price. That mismatch is the question here.

The LLM's scenarios are cached and the anchor is pure numerics, so swapping the
anchor needs no LLM calls: same scenarios on both sides, only the ARIMA
specification differs. A new agent variant would instead mint a new
``predictor_id``, miss the cache, and re-run the whole grid through the model
for a change the LLM never sees.

**The noise floor is the point of this script.** ``compute_arima_anchor`` draws
``num_samples`` Monte Carlo paths with no seed, so re-running the *level* anchor
does not reproduce the cached numbers exactly. That jitter is measured here and
reported alongside the effect. If swapping to log-returns moves CRPS by less
than re-running the identical level anchor does, the comparison has found
nothing and says so, instead of reporting whichever way the coin landed.

Prior expectation, from the two specifications run standalone on this grid:
level holds 82.6% coverage / 4.275 CRPS on the backtest against log-returns'
81.4% / 4.394, and in the eval window log-returns over-covers harder (93.8% vs
87.5%). Anchored's eval coverage is already 96.4% — too wide, not too narrow —
so the expected effect is small and probably adverse.

Needs the price cache (it refits ARIMA at every origin), so run it locally:

    python scripts/compare_anchor_level_vs_logret.py 10yr
    python scripts/compare_anchor_level_vs_logret.py 10yr-eval
"""

from __future__ import annotations

import sys
from collections import defaultdict
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import properscoring as ps
import yaml
from aieng.forecasting.evaluation.task import ForecastingTask
from energy_oil_forecasting.data import WTI_SERIES_ID, build_wti_multivariate_service
from energy_oil_forecasting.price_deltas import compute_horizon_delta_percentiles
from energy_oil_forecasting.scenario_schema_anchored.arima_anchor import compute_arima_anchor

# The real post-LLM math, imported rather than reimplemented.
from energy_oil_forecasting.scenario_schema_anchored.predictor import (
    _grounded_center_shift,
    _implied_target_percentile,
    _probability_weighted_scenario_price,
    _widen_toward_scenarios,
)

from actuals_from_crps import recover_actuals


WINDOWS = {
    "10yr": Path("data/predictions/energy_oil_backtest_10yr_quarterly"),
    "10yr-eval": Path("data/predictions/energy_oil_eval_10yr_quarterly"),
}
CACHE_STEM = "agent_predictor_wti_analyst_news_scenario_schema_anchored_gemini-3.1-flash-lite-preview_continuous__wti_oil_price_forecast.yaml"
HORIZONS = [5, 10, 21, 63]
MAX_HORIZON = max(HORIZONS)


def _score(quantiles: dict[float, float], actual: float) -> tuple[float, float]:
    ensemble = np.array(sorted(quantiles.values()), dtype=float)
    return float(ps.crps_ensemble(actual, ensemble)), float(quantiles[0.1] <= actual <= quantiles[0.9])


def _final_quantiles(anchor, scenarios, deltas, horizon: int, target_percentile: float):
    """Reproduce ScenarioSchemaAnchoredPredictor.predict's post-LLM arithmetic."""
    shift = _grounded_center_shift(target_percentile, deltas[horizon])
    shifted = {level: value + shift for level, value in anchor[horizon].quantiles.items()}
    low = min(s["price_low"] for s in scenarios)
    high = max(s["price_high"] for s in scenarios)
    final = _widen_toward_scenarios(shifted, low, high, scale=sqrt(horizon / MAX_HORIZON))
    return final, anchor[horizon].point_forecast + shift


def main() -> None:
    window = sys.argv[1] if len(sys.argv) > 1 else "10yr"
    if window not in WINDOWS:
        raise SystemExit(f"usage: compare_anchor_level_vs_logret.py [{'|'.join(WINDOWS)}]")
    window_dir = WINDOWS[window]
    print(f"window: {window}  ({window_dir})\n")

    doc = yaml.safe_load((window_dir / CACHE_STEM).read_text())
    actuals = recover_actuals(window_dir).actuals
    service = build_wti_multivariate_service()

    by_origin: dict[str, list[dict]] = defaultdict(list)
    for pred in doc["predictions"]:
        by_origin[str(pred["as_of"])[:10]].append(pred)

    rows = []
    for origin, preds in sorted(by_origin.items()):
        scenarios = preds[0].get("metadata", {}).get("scenarios") or []
        if not scenarios:
            print(f"  (skipping {origin}: no scenarios cached)")
            continue

        as_of = pd.Timestamp(origin)
        context = service.context(as_of=as_of)
        task = ForecastingTask(
            task_id="wti_oil_price_forecast",
            target_series_id=WTI_SERIES_ID,
            horizons=HORIZONS,
            frequency="B",
            description="WTI price forecast",
        )
        deltas = compute_horizon_delta_percentiles(context, WTI_SERIES_ID, HORIZONS)
        weighted = _probability_weighted_scenario_price(scenarios)

        # Three anchors: the cached one, a fresh level refit (the noise floor),
        # and the log-return variant under test.
        anchors = {
            "level_refit": compute_arima_anchor(task, context),
            "logret": compute_arima_anchor(task, context, log_returns=True),
        }
        targets = {
            key: _implied_target_percentile(weighted, anchor[MAX_HORIZON].quantiles)
            for key, anchor in anchors.items()
        }

        for pred in preds:
            date = str(pred["forecast_date"])[:10]
            actual = actuals.get(date)
            if actual is None:
                continue
            horizon = int(np.busday_count(as_of.date(), pd.Timestamp(date).date()))
            if horizon not in HORIZONS:
                continue

            cached_q = {float(k): float(v) for k, v in pred["payload"]["quantiles"].items()}
            row = {"origin": origin, "horizon": horizon}
            row["cached_crps"], row["cached_in"] = _score(cached_q, actual)
            for key in ("level_refit", "logret"):
                q, _ = _final_quantiles(anchors[key], scenarios, deltas, horizon, targets[key])
                row[f"{key}_crps"], row[f"{key}_in"] = _score(q, actual)
                row[f"{key}_width"] = q[0.9] - q[0.1]
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        print("nothing scored")
        return

    noise = abs(df["level_refit_crps"].mean() - df["cached_crps"].mean())
    effect = df["logret_crps"].mean() - df["level_refit_crps"].mean()

    print(f"scored {len(df)} predictions from {df['origin'].nunique()} origins\n")
    print("=" * 84)
    print("SCENARIO SCHEMA ANCHORED — same cached LLM scenarios, ARIMA anchor swapped")
    print("=" * 84)
    print(f"{'':22} {'mean CRPS':>11} {'cov80':>8} {'mean width':>12}")
    for label, key in (("cached (level)", "cached"), ("level, refit", "level_refit"), ("log returns", "logret")):
        width = f"{df[key + '_width'].mean():>12.2f}" if key + "_width" in df else f"{'-':>12}"
        print(f"{label:22} {df[key + '_crps'].mean():>11.4f} {100 * df[key + '_in'].mean():>7.1f}% {width}")

    print()
    print(f"effect of the swap      : {effect:+.4f} CRPS  "
          f"({100 * (df['logret_in'].mean() - df['level_refit_in'].mean()):+.1f} coverage pts)")
    print(f"Monte-Carlo noise floor : {noise:.4f} CRPS  (identical level anchor, refit vs cached)")
    if abs(effect) <= noise:
        print("\n=> INCONCLUSIVE: the swap moves CRPS by less than re-running the same")
        print("   anchor does. Raise num_samples in compute_arima_anchor to shrink the")
        print("   floor, or accept that the anchor specification does not matter here.")
    else:
        direction = "better" if effect < 0 else "worse"
        print(f"\n=> log returns is {direction} by {abs(effect):.4f} CRPS, above the noise floor.")

    print("\nby horizon:")
    for horizon in HORIZONS:
        sub = df[df["horizon"] == horizon]
        if sub.empty:
            continue
        print(
            f"  h={horizon:>2}d  CRPS {sub['level_refit_crps'].mean():7.4f} -> {sub['logret_crps'].mean():7.4f}"
            f"   coverage {100 * sub['level_refit_in'].mean():5.1f}% -> {100 * sub['logret_in'].mean():5.1f}%"
            f"   width {sub['level_refit_width'].mean():6.2f} -> {sub['logret_width'].mean():6.2f}"
        )


if __name__ == "__main__":
    main()
