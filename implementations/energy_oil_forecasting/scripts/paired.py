"""Paired per-origin comparison against Naive. n=40, so unpaired SEs mislead."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_csv(
    "/tmp/claude-0/-home-user-agentic-forecasting/1b52bc58-6e35-5d9a-8f7d-7571837c0ef6/scratchpad/scored.csv"
)

pivot_crps = df.pivot(index="origin", columns="predictor", values="crps")
pivot_mae = df.pivot(index="origin", columns="predictor", values="abs_err")

BASE = "last_value_naive"

print("━" * 92)
print("PAIRED vs NAIVE — per-origin differences at h=63 (n=40). Negative = beats Naive.")
print("━" * 92)
print(f"{'predictor':30s} {'ΔCRPS':>8s} {'95% CI':>18s} {'win%':>6s} {'p':>8s} {'ΔMAE':>8s} {'MAEwin%':>8s}")

for col in sorted(pivot_crps.columns):
    if col == BASE:
        continue
    d = (pivot_crps[col] - pivot_crps[BASE]).dropna()
    dm = (pivot_mae[col] - pivot_mae[BASE]).dropna()
    lo, hi = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    p = stats.wilcoxon(d).pvalue
    print(
        f"{col:30s} {d.mean():+8.3f} [{lo:+7.2f},{hi:+7.2f}] "
        f"{100 * (d < 0).mean():5.0f}% {p:8.4f} {dm.mean():+8.3f} {100 * (dm < 0).mean():7.0f}%"
    )

print()
print("━" * 92)
print("HEAD-TO-HEAD: AutoARIMA vs each LightGBM variant (paired ΔCRPS, negative = ARIMA wins)")
print("━" * 92)
for col in ["darts_lightgbm_cov", "darts_lightgbm_cov_expanded"]:
    d = (pivot_crps["darts_autoarima"] - pivot_crps[col]).dropna()
    lo, hi = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    p = stats.wilcoxon(d).pvalue
    print(f"AutoARIMA − {col:28s} {d.mean():+7.3f} [{lo:+6.2f},{hi:+6.2f}]  p={p:.4f}  "
          f"ARIMA wins {100 * (d < 0).mean():.0f}% of origins")

# Interval calibration detail: nominal 80% coverage.
print()
print("━" * 92)
print("CALIBRATION at h=63 (nominal 80%)")
print("━" * 92)
cal = df.groupby("predictor").agg(
    coverage80=("inside80", lambda s: 100 * s.mean()),
    mean_width=("width80", "mean"),
    median_abs_err=("abs_err", "median"),
)
cal["verdict"] = np.where(
    cal.coverage80 < 60, "far too narrow",
    np.where(cal.coverage80 > 92, "too wide", "reasonable"),
)
print(cal.round(2).to_string())
