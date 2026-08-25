"""Why are the 80% intervals under-covering — too narrow, or pointing the wrong way?

Widening the delta-governed agent's intervals by 12.5% moved its coverage by
exactly zero, which says the misses are not marginal. This separates the two
possible causes:

- misses split evenly above/below  -> the centre is fine, the bands are
  genuinely too narrow, and widening should help
- misses cluster on one side       -> the forecasts are biased, and widening
  is treating a symptom

Runs across every cached predictor in the chosen window, not just the agents.
That matters: if AutoARIMA misses the same way, the problem lives in the
numerical ensemble's uncertainty during the 2026 shock and no amount of
governor or agent work will fix it.

Run from the implementations/energy_oil_forecasting directory:

    python scripts/miss_structure.py            # 2026 eval window (default)
    python scripts/miss_structure.py backtest   # 2025 backtest window
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from energy_oil_forecasting.data import WTI_SERIES_ID, build_wti_multivariate_service


WINDOWS = {
    "eval": Path("data/predictions/energy_oil_eval_biweekly"),
    "backtest": Path("data/predictions/energy_oil_backtest_biweekly"),
}
HORIZONS = [5, 10, 21]


def _horizon_for(as_of: pd.Timestamp, forecast_date: pd.Timestamp) -> int | None:
    offset = pd.tseries.offsets.BDay()
    for horizon in HORIZONS:
        if (as_of + offset * horizon).normalize() == forecast_date.normalize():
            return horizon
    return None


def main() -> None:
    window = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if window not in WINDOWS:
        raise SystemExit(f"usage: miss_structure.py [{'|'.join(WINDOWS)}]")
    eval_dir = WINDOWS[window]
    print(f"window: {window}  ({eval_dir})\n")
    service = build_wti_multivariate_service()
    resolved_as_of = datetime.now(tz=timezone.utc).replace(tzinfo=None)
    actual_df = service.get_series(WTI_SERIES_ID, as_of=resolved_as_of)
    actuals = {pd.Timestamp(r["timestamp"]).normalize(): float(r["value"]) for _, r in actual_df.iterrows()}

    rows = []
    for path in sorted(eval_dir.glob("*.yaml")):
        data = yaml.safe_load(path.read_text())
        name = data.get("predictor_id", path.stem)
        for pred in data.get("predictions", []):
            payload = pred.get("payload") or {}
            quantiles = payload.get("quantiles")
            if not quantiles:
                continue
            q = {float(k): float(v) for k, v in quantiles.items()}
            forecast_date = pd.Timestamp(pred["forecast_date"]).normalize()
            actual = actuals.get(forecast_date)
            if actual is None:
                continue
            horizon = _horizon_for(pd.Timestamp(pred["as_of"]), forecast_date)
            half_width = (q[0.9] - q[0.1]) / 2.0
            inside80 = q[0.1] <= actual <= q[0.9]
            inside90 = q[0.05] <= actual <= q[0.95]
            if inside80:
                side, beyond = "inside", 0.0
            elif actual > q[0.9]:
                side, beyond = "above", actual - q[0.9]
            else:
                side, beyond = "below", q[0.1] - actual
            rows.append(
                {
                    "predictor": name,
                    "horizon": horizon,
                    "actual": actual,
                    "point": float(payload["point_forecast"]),
                    "signed_error": actual - float(payload["point_forecast"]),
                    "half_width": half_width,
                    "inside80": float(inside80),
                    "inside90": float(inside90),
                    "side": side,
                    "beyond": beyond,
                    "beyond_in_half_widths": beyond / half_width if half_width > 0 else np.nan,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"no scored predictions found under {eval_dir}")
        return

    print(f"{len(df)} scored predictions across {df['predictor'].nunique()} predictors\n")
    print("=" * 108)
    print("COVERAGE AND MISS STRUCTURE (80% band = p10..p90)")
    print("=" * 108)
    header = (
        f"{'predictor':52} {'n':>3} {'cov80':>6} {'cov90':>6} "
        f"{'above':>6} {'below':>6} {'mean signed err':>16} {'bias/halfwidth':>15}"
    )
    print(header)
    for name, sub in sorted(df.groupby("predictor"), key=lambda kv: -kv[1]["inside80"].mean()):
        above = int((sub["side"] == "above").sum())
        below = int((sub["side"] == "below").sum())
        bias = sub["signed_error"].mean()
        ratio = bias / sub["half_width"].mean() if sub["half_width"].mean() > 0 else np.nan
        print(
            f"{name[:52]:52} {len(sub):>3} {100 * sub['inside80'].mean():>5.1f}% "
            f"{100 * sub['inside90'].mean():>5.1f}% {above:>6} {below:>6} "
            f"{bias:>+16.2f} {ratio:>+15.2f}"
        )

    print()
    print("=" * 108)
    print("HOW FAR OUTSIDE THE MISSES LANDED (in multiples of the band's half-width)")
    print("=" * 108)
    missed = df[df["side"] != "inside"]
    if missed.empty:
        print("no misses")
    else:
        print(f"{'predictor':52} {'misses':>7} {'median':>8} {'mean':>8} {'max':>8}")
        for name, sub in sorted(missed.groupby("predictor"), key=lambda kv: -kv[1]["beyond_in_half_widths"].median()):
            b = sub["beyond_in_half_widths"]
            print(f"{name[:52]:52} {len(sub):>7} {b.median():>8.2f} {b.mean():>8.2f} {b.max():>8.2f}")
        print()
        print("A miss at 0.2 half-widths would be rescued by a 20% wider band; one at 2.0 would not.")

    print()
    print("=" * 108)
    print("BY HORIZON (all predictors pooled)")
    print("=" * 108)
    for horizon in HORIZONS:
        sub = df[df["horizon"] == horizon]
        if sub.empty:
            continue
        above = int((sub["side"] == "above").sum())
        below = int((sub["side"] == "below").sum())
        print(
            f"  h={horizon:>2}d  n={len(sub):>3}  cov80={100 * sub['inside80'].mean():5.1f}%  "
            f"above={above:>3} below={below:>3}  mean signed err={sub['signed_error'].mean():+7.2f}  "
            f"mean half-width={sub['half_width'].mean():6.2f}"
        )


if __name__ == "__main__":
    main()
