"""Score every cached predictor before and after conformal calibration.

Companion to ``width_recalibration.py``. Both correct interval width without
touching the centre; they differ in where the correction comes from.

- width floor : the spread of historical h-day WTI moves. External to the
                model, identical for every predictor, widen-only.
- conformal   : the predictor's OWN past errors at that horizon. Specific to
                the model, adapts as it improves or degrades, and narrows an
                over-covering predictor as readily as it widens an
                under-covering one.

The second is the one with a coverage guarantee attached, which is why it is
worth the extra machinery.

Calibration uses only forecasts that had already resolved at the origin being
corrected (see ``conformal.calibrate_series``), so the numbers below are what
this would have produced live, not a hindsight fit. Early origins have no
history and are reported separately as uncalibrated rather than quietly scored
as though they had been corrected.

Actuals come from inverting the cached CRPS (see ``actuals_from_crps.py``), so
no price cache is needed.

Run from the implementations/energy_oil_forecasting directory:

    python scripts/conformal_calibration.py 10yr
    python scripts/conformal_calibration.py 10yr-eval
    python scripts/conformal_calibration.py backtest --window 20
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import properscoring as ps
import yaml

from actuals_from_crps import recover_actuals
from conformal import calibrate_series
from labels import label


WINDOWS = {
    "eval": Path("data/predictions/energy_oil_eval_biweekly"),
    "backtest": Path("data/predictions/energy_oil_backtest_biweekly"),
    "10yr": Path("data/predictions/energy_oil_backtest_10yr_quarterly"),
    "10yr-eval": Path("data/predictions/energy_oil_eval_10yr_quarterly"),
}

def _score(quantiles: dict[float, float], actual: float) -> tuple[float, float]:
    ensemble = np.array(sorted(quantiles.values()), dtype=float)
    return float(ps.crps_ensemble(actual, ensemble)), float(quantiles[0.1] <= actual <= quantiles[0.9])


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    window_name = args[0] if args else "10yr"
    if window_name not in WINDOWS:
        raise SystemExit(f"usage: conformal_calibration.py [{'|'.join(WINDOWS)}] [--window N]")
    lookback: int | None = None
    if "--window" in sys.argv:
        lookback = int(sys.argv[sys.argv.index("--window") + 1])

    window_dir = WINDOWS[window_name]
    print(f"window: {window_name}  ({window_dir})")
    print(f"calibration lookback: {'all history' if lookback is None else f'{lookback} origins'}\n")

    actuals = recover_actuals(window_dir).actuals

    rows = []
    for path in sorted(window_dir.glob("*.yaml")):
        doc = yaml.safe_load(path.read_text())
        name = label(doc["predictor_id"])

        # Group by horizon: a 5-day band and a 63-day band have entirely
        # different error scales and must never share a calibration set.
        by_horizon: dict[int, list[dict]] = defaultdict(list)
        for pred in doc.get("predictions", []):
            quantiles = (pred.get("payload") or {}).get("quantiles")
            if not quantiles:
                continue
            date = str(pred["forecast_date"])[:10]
            actual = actuals.get(date)
            if actual is None:
                continue
            origin = pd.Timestamp(str(pred["as_of"])[:10])
            resolves = pd.Timestamp(date)
            by_horizon[int(np.busday_count(origin.date(), resolves.date()))].append(
                {
                    "origin": origin,
                    "resolves": resolves,
                    "quantiles": {float(k): float(v) for k, v in quantiles.items()},
                    "actual": actual,
                }
            )

        for horizon, records in sorted(by_horizon.items()):
            calibrated, report = calibrate_series(records, window=lookback)
            for record in calibrated:
                base_crps, base_in = _score(record["quantiles"], record["actual"])
                conf_crps, conf_in = _score(record["conformal"], record["actual"])
                rows.append(
                    {
                        "predictor": name,
                        "horizon": horizon,
                        "base_crps": base_crps,
                        "base_in": base_in,
                        "conf_crps": conf_crps,
                        "conf_in": conf_in,
                        "base_width": record["quantiles"][0.9] - record["quantiles"][0.1],
                        "conf_width": record["conformal"][0.9] - record["conformal"][0.1],
                        "was_calibrated": bool(report.calibrated and record["conformal"] != record["quantiles"]),
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        print("nothing to score")
        return

    n_uncal = int((~df["was_calibrated"]).sum())
    print(f"{len(df)} predictions; {n_uncal} left uncalibrated for want of resolved history")
    print("Scored on the calibrated ones only -- including predictions the method")
    print("could not act on would understate both sides of the comparison.\n")

    df = df[df["was_calibrated"]]
    if df.empty:
        print("no prediction had enough resolved history to calibrate")
        return

    print("=" * 100)
    print("BEFORE vs AFTER CONFORMAL CALIBRATION (nominal 80%)")
    print("=" * 100)
    print(f"{'predictor':25} {'n':>4} {'CRPS base':>10} {'conformal':>10} {'change':>8}   "
          f"{'cov base':>9} {'conformal':>10} {'width x':>8}")
    for name, sub in sorted(df.groupby("predictor"), key=lambda kv: kv[1]["conf_crps"].mean()):
        ratio = sub["conf_width"].mean() / sub["base_width"].mean() if sub["base_width"].mean() else float("nan")
        print(
            f"{name[:25]:25} {len(sub):>4} {sub['base_crps'].mean():>10.3f} {sub['conf_crps'].mean():>10.3f} "
            f"{sub['conf_crps'].mean() - sub['base_crps'].mean():>+8.3f}   "
            f"{100 * sub['base_in'].mean():>8.1f}% {100 * sub['conf_in'].mean():>9.1f}% {ratio:>8.2f}"
        )

    print()
    # The overshoot below is the finite-sample correction, not a bug. The level
    # actually taken is ceil((n+1)(1-alpha))/n, which exceeds 1-alpha whenever
    # the calibration set is small: at n=34 it is 0.824, at n=16 it is 0.875.
    # That is why shrinking the window makes coverage worse, not better -- the
    # guarantee is one-sided (at least 1-alpha), and it buys that by being
    # conservative. More origins is the only fix; a quarterly grid over ten
    # years gives about 34 calibration points per horizon, which is not enough
    # to land on nominal.
    typical = int(df.groupby(["predictor", "horizon"]).size().median())
    effective = np.ceil((typical + 1) * 0.8) / typical
    print(f"calibration sets are ~{typical} points, so the 80% band is actually cut at "
          f"the {100 * effective:.1f}th percentile")
    print(f"of the conformity scores -- {100 * (effective - 0.8):.1f} points of built-in conservatism.")
    print()
    print(f"{'':25} {'|cov - 80|':>12}")
    before = (100 * df.groupby('predictor')['base_in'].mean() - 80).abs().mean()
    after = (100 * df.groupby('predictor')['conf_in'].mean() - 80).abs().mean()
    print(f"{'mean miss from nominal':25} {before:>11.1f} -> {after:.1f} points")
    print()
    print("width x below 1 means conformal NARROWED the band: that predictor was")
    print("too wide, not too narrow, and was paying CRPS for the excess.")


if __name__ == "__main__":
    main()
