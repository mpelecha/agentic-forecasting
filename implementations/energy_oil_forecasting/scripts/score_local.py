"""Score the Mac's 10yr-quarterly cache without any external data.

Actuals are reconstructed from the Naive predictor: its point forecast at
origin t IS the observed WTI close at t. Because the origin stride (63 bd)
equals the longest horizon, every h=63 forecast_date lands on a later origin —
so those 40 predictions per predictor can be scored exactly.
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd
import properscoring as ps
import yaml

sys.path.insert(0, "implementations")

from aieng.forecasting.evaluation.backtest import BacktestResult  # noqa: E402
from energy_oil_forecasting.analysis import score_backtest_results  # noqa: E402

D = "implementations/energy_oil_forecasting/data/predictions/energy_oil_backtest_10yr_quarterly_localrun"

# ── Reconstruct the price series from Naive ──────────────────────────────────
naive_doc = yaml.safe_load(open(f"{D}/last_value_naive__wti_oil_price_forecast.yaml"))
price = {
    pd.Timestamp(p["as_of"]).normalize(): float(p["payload"]["point_forecast"])
    for p in naive_doc["predictions"]
}
print(f"Reconstructed {len(price)} origin prices "
      f"({min(price).date()} → {max(price).date()})\n")

_price_df = pd.DataFrame(
    {"timestamp": list(price.keys()), "value": list(price.values())}
).sort_values("timestamp")


class StubService:
    """DataService stand-in serving only the reconstructed origin closes."""

    def get_series(self, series_id: str, as_of=None) -> pd.DataFrame:
        _ = series_id, as_of
        return _price_df.copy()


stub = StubService()

# ── Load every cached predictor ──────────────────────────────────────────────
results: dict[str, dict[str, BacktestResult]] = {}
for path in sorted(glob.glob(f"{D}/*.yaml")):
    doc = yaml.safe_load(open(path))
    br = BacktestResult.model_validate(doc)
    name = os.path.basename(path).replace("__wti_oil_price_forecast.yaml", "")
    results[name] = {br.spec.task.task_id: br}

# ── Per-prediction scoring ───────────────────────────────────────────────────
def crps_from_quantiles(qs: dict, actual: float) -> float:
    """CRPS the same way the repo scores it: quantile values as an ensemble."""
    ensemble = np.array(sorted(qs.values()), dtype=float)
    return float(ps.crps_ensemble(actual, ensemble))


rows = []
for name, res in results.items():
    br = next(iter(res.values()))
    for p in br.predictions:
        fd = pd.Timestamp(p.forecast_date).normalize()
        actual = price.get(fd)
        if actual is None:
            continue  # h=5/10/21 targets are not origin dates — unscoreable here
        q = {float(k): float(v) for k, v in p.payload.quantiles.items()}
        lo, hi = q.get(0.1), q.get(0.9)
        rows.append(
            {
                "predictor": name,
                "origin": pd.Timestamp(p.as_of).normalize(),
                "forecast_date": fd,
                "actual": actual,
                "point": float(p.payload.point_forecast),
                "abs_err": abs(float(p.payload.point_forecast) - actual),
                "crps": crps_from_quantiles(q, actual),
                "inside80": float(lo <= actual <= hi) if lo is not None and hi is not None else np.nan,
                "width80": (hi - lo) if lo is not None and hi is not None else np.nan,
            }
        )

df = pd.DataFrame(rows)
df.to_csv("/tmp/claude-0/-home-user-agentic-forecasting/1b52bc58-6e35-5d9a-8f7d-7571837c0ef6/scratchpad/scored.csv", index=False)

print("Scored predictions per predictor (h=63 only):")
print(df.groupby("predictor").size().to_string())
print()

board = (
    df.groupby("predictor")
    .agg(
        n=("crps", "size"),
        mean_crps=("crps", "mean"),
        se_crps=("crps", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
        mae=("abs_err", "mean"),
        coverage80=("inside80", lambda s: 100 * s.mean()),
        mean_width80=("width80", "mean"),
    )
    .sort_values("mean_crps")
)
print("━" * 100)
print("h=63 LEADERBOARD — identical origin grid for naive/kalman/autoarima/lightgbm")
print("━" * 100)
print(board.round(3).to_string())
