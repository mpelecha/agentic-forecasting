"""Rescale every predictor's interval to the width real WTI moves justify.

The under-covering agents miss with balanced sides and near-zero bias -- their
centres are fine, their bands are too tight. v5.2.2's governor already contains
the right correction:

    pre_floor = {k: p50 + applied + scale * (v - p50) for k, v in q.items()}
    empirical_width = percentiles[90] - percentiles[10]
    target_width    = uncertainty_multiplier * empirical_width

That is a shape-preserving rescale toward the spread of *actual* h-day price
moves observed up to the origin. It needs no fitting against outcomes, adapts
per horizon, and tracks regime and price level -- strictly better than tuning a
single constant on realised coverage.

But it is gated on the LLM's verdict, and the ``unchanged`` branch pins
``effective_target = ensemble_width`` (scale exactly 1.0), so the empirical
width is computed and then discarded on 78% of predictions. It has also never
been attached to the agents that actually under-cover, which do not use the
governor at all.

This applies that same mapping ungated, to every cached predictor:

- ``replace``  scale = empirical_width / model_width
               history sets the width outright
- ``floor``    scale = max(1, empirical_width / model_width)
               the model may be wider than history, never narrower

``floor`` is the conservative variant: it can only repair under-coverage, and
never overrides a model that has genuinely widened for elevated volatility.

Nothing here is fitted on realised outcomes, so there is no train/test split to
get wrong -- the empirical table at each origin uses only data available at that
cutoff. The tables are read from the delta-governed cache's per-prediction
metadata (``historical_delta_p10/p50/p90``), which is keyed by origin and
horizon, not by predictor, so it transfers to every predictor on the same grid.

Actuals are recovered from cached CRPS (see actuals_from_crps.py), so this needs
no price cache.

Run from the implementations/energy_oil_forecasting directory:

    python scripts/width_recalibration.py 10yr
    python scripts/width_recalibration.py 10yr-eval
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import properscoring as ps
import yaml

from actuals_from_crps import recover_actuals


WINDOWS = {
    "10yr": Path("data/predictions/energy_oil_backtest_10yr_quarterly"),
    "10yr-eval": Path("data/predictions/energy_oil_eval_10yr_quarterly"),
}
# Only this agent records the empirical delta table it computed, so it is the
# source of the per-(origin, horizon) widths every other predictor is rescaled
# against.
DELTA_CACHE = "agent_predictor_cfm_agent_v_5_2_2_delta_governed_gemini-3.1-flash-lite-preview_continuous__wti_oil_price_forecast.yaml"


def _empirical_widths(window_dir: Path) -> dict[tuple[str, int], float]:
    """Map (origin, horizon) -> p90-p10 of real h-day price moves at that origin."""
    doc = yaml.safe_load((window_dir / DELTA_CACHE).read_text())
    out: dict[tuple[str, int], float] = {}
    for pred in doc["predictions"]:
        ft = pred["metadata"]["forecast_transformation"]
        key = (str(pred["as_of"])[:10], int(ft["horizon"]))
        out[key] = float(ft["historical_delta_p90"]) - float(ft["historical_delta_p10"])
    return out


def _rescale(quantiles: dict[float, float], scale: float) -> dict[float, float]:
    """Stretch the band around its own median, leaving the centre untouched."""
    centre = quantiles[0.5]
    return {level: centre + scale * (value - centre) for level, value in quantiles.items()}


def _score(quantiles: dict[float, float], actual: float) -> tuple[float, float]:
    ensemble = np.array(sorted(quantiles.values()), dtype=float)
    crps = float(ps.crps_ensemble(actual, ensemble))
    return crps, float(quantiles[0.1] <= actual <= quantiles[0.9])


def main() -> None:
    window = sys.argv[1] if len(sys.argv) > 1 else "10yr"
    if window not in WINDOWS:
        raise SystemExit(f"usage: width_recalibration.py [{'|'.join(WINDOWS)}]")
    window_dir = WINDOWS[window]
    print(f"window: {window}  ({window_dir})\n")

    recovery = recover_actuals(window_dir)
    actuals = {k: v for k, v in recovery.actuals.items()}
    widths = _empirical_widths(window_dir)
    print(f"empirical delta tables for {len(widths)} (origin, horizon) pairs")
    print(f"actuals recovered for {len(actuals)} forecast dates "
          f"(worst cross-predictor disagreement ${recovery.worst_disagreement:.1e})\n")

    rows = []
    for path in sorted(window_dir.glob("*.yaml")):
        doc = yaml.safe_load(path.read_text())
        name = doc["predictor_id"].replace("agent_predictor_", "").replace(
            "_gemini-3.1-flash-lite-preview", "").replace("_continuous", "")
        for pred in doc["predictions"]:
            q = {float(k): float(v) for k, v in (pred.get("payload") or {}).get("quantiles", {}).items()}
            if not q:
                continue
            date = str(pred["forecast_date"])[:10]
            actual = actuals.get(date)
            if actual is None:
                continue
            origin = str(pred["as_of"])[:10]
            horizon = int(np.busday_count(pd.Timestamp(origin).date(), pd.Timestamp(date).date()))
            empirical = widths.get((origin, horizon))
            model_width = q[0.9] - q[0.1]
            if empirical is None or model_width <= 0:
                continue

            ratio = empirical / model_width
            base_crps, base_in = _score(q, actual)
            rep_crps, rep_in = _score(_rescale(q, ratio), actual)
            flo_crps, flo_in = _score(_rescale(q, max(1.0, ratio)), actual)
            rows.append(
                {
                    "predictor": name, "horizon": horizon, "ratio": ratio,
                    "base_crps": base_crps, "base_in": base_in,
                    "replace_crps": rep_crps, "replace_in": rep_in,
                    "floor_crps": flo_crps, "floor_in": flo_in,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        print("no predictions could be matched to an empirical width")
        return

    print("=" * 104)
    print("BASELINE vs HISTORY-WIDTH RESCALE (nominal coverage 80%)")
    print("=" * 104)
    print(f"{'predictor':38} {'n':>4} {'ratio':>6} "
          f"{'CRPS base':>10} {'replace':>9} {'floor':>9}   "
          f"{'cov base':>9} {'replace':>8} {'floor':>8}")
    for name, sub in sorted(df.groupby("predictor"), key=lambda kv: kv[1]["base_crps"].mean()):
        print(
            f"{name[:38]:38} {len(sub):>4} {sub['ratio'].mean():>6.2f} "
            f"{sub['base_crps'].mean():>10.3f} {sub['replace_crps'].mean():>9.3f} {sub['floor_crps'].mean():>9.3f}   "
            f"{100 * sub['base_in'].mean():>8.1f}% {100 * sub['replace_in'].mean():>7.1f}% "
            f"{100 * sub['floor_in'].mean():>7.1f}%"
        )

    print()
    print("ratio = empirical width / model width. Above 1 means history says the")
    print("model's band is too narrow; 'floor' only acts on those cases.")
    print("A degenerate point-mass predictor (naive) has zero width and is skipped.")


if __name__ == "__main__":
    main()
