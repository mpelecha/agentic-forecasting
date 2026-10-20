"""Check whether an agent's news layer moved its forecast the right way relative to its own anchor.

For each forecast the call is whether the agent's final median sits above, at (within
5 cents) or below the median of the numerical forecast it started from; it is right
when the price ended on the same side of that starting forecast. This isolates what
the LLM changed. The starting forecast is read from the prediction metadata:

* ``unadjusted_ensemble`` — CFM v5.x / v6.0 (the ensemble before the policy adjustment)
* ``arima_anchor`` — the scenario-memory agent (the AutoARIMA anchor before the shift)

Usage (repo root; needs the WTI price cache from scripts/fetch_wti.py)::

    S=implementations/energy_oil_forecasting/scripts/direction_vs_own_anchor.py
    uv run python $S "Scenario memory=path/to/scenario_memory.yaml" "CFM v6.0=path/to/v6.yaml"

Each argument is ``label=path`` to a cached backtest result (``*.yaml``). Prints the
call-by-outcome grid per horizon with calls right and an exact two-sided binomial
p-value against a coin flip, and saves a summary and a per-forecast CSV to
``results/``. No LLM calls.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from math import comb
from pathlib import Path

import pandas as pd
import yaml


_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))
RESULTS_DIR = _HERE.parents[1] / "results"
HORIZONS = (5, 10, 21)
EPS = 0.05  # gaps under 5 cents count as "same", as on cfm_coach's corpus page
ANCHOR_KEYS = ("unadjusted_ensemble", "arima_anchor")
CALLS = ["Above its anchor", "Same as its anchor", "Below its anchor"]


def _sign(x: float) -> int:
    return 0 if abs(x) < EPS else (1 if x > 0 else -1)


def _binom_p(k: int, n: int) -> float:
    """Two-sided exact binomial p-value against a fair coin."""
    if n == 0:
        return float("nan")
    probs = [comb(n, i) / 2**n for i in range(n + 1)]
    return min(1.0, sum(p for p in probs if p <= probs[k] + 1e-12))


def _horizon(as_of: pd.Timestamp, forecast_date: pd.Timestamp) -> int:
    steps = max(len(pd.bdate_range(as_of.normalize(), forecast_date.normalize())) - 1, 0)
    return min(HORIZONS, key=lambda h: abs(h - steps))


def rows_for(label: str, path: Path, actual: dict[pd.Timestamp, float]) -> list[dict]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    rows = []
    for pred in raw["predictions"]:
        meta = pred.get("metadata") or {}
        key = next((k for k in ANCHOR_KEYS if isinstance(meta.get(k), dict)), None)
        if key is None:
            raise SystemExit(f"{label}: no starting forecast in metadata (looked for {', '.join(ANCHOR_KEYS)}).")
        origin = pd.Timestamp(pred["as_of"]).normalize()
        fdate = pd.Timestamp(pred["forecast_date"]).normalize()
        if fdate not in actual:
            continue
        anchor = float(meta[key]["point_forecast"])
        final = float(pred["payload"]["point_forecast"])
        call, outcome = _sign(final - anchor), _sign(actual[fdate] - anchor)
        rows.append(
            {
                "agent": label,
                "origin": origin,
                "forecast_date": fdate,
                "horizon": _horizon(origin, fdate),
                "anchor_from": key,
                "anchor_p50": anchor,
                "final_p50": final,
                "actual": actual[fdate],
                "shift": final - anchor,
                "call": CALLS[1 - call],
                "outcome": {1: "Ended above", -1: "Ended below", 0: "Ended at"}[outcome],
                "right": None if call == 0 or outcome == 0 else call == outcome,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("results", nargs="+", help="label=path to a cached backtest result yaml")
    args = parser.parse_args()

    from energy_oil_forecasting.data import WTI_SERIES_ID, build_wti_multivariate_service  # noqa: PLC0415

    prices = build_wti_multivariate_service().get_series(WTI_SERIES_ID, as_of=datetime.now())
    actual = {pd.Timestamp(t).normalize(): float(v) for t, v in zip(prices["timestamp"], prices["value"], strict=True)}

    rows = []
    for item in args.results:
        label, sep, path = item.partition("=")
        if not sep:
            raise SystemExit(f"Expected label=path, got {item!r}")
        rows += rows_for(label, Path(path), actual)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    summary = []
    for agent, by_agent in df.groupby("agent", sort=False):
        print(f"\n================ {agent}: final median vs its own anchor ================")
        for h in HORIZONS:
            sub = by_agent[by_agent["horizon"] == h]
            scored = sub.dropna(subset=["right"])
            k, n = int(scored["right"].sum()), len(scored)
            grid = pd.crosstab(sub["call"], sub["outcome"]).reindex(index=CALLS).fillna(0).astype(int)
            print(f"\n--- {h}-day ({len(sub)} forecasts, {sub['origin'].nunique()} origins) ---")
            print(grid.to_string())
            print(f"Calls right: {k}/{n} = {100 * k / n if n else float('nan'):.0f}%  (p = {_binom_p(k, n):.2f})")
            summary.append(
                {
                    "agent": agent,
                    "horizon": f"{h}-day",
                    "forecasts": len(sub),
                    "moved": int((sub["call"] != CALLS[1]).sum()),
                    "called_above": int((sub["call"] == CALLS[0]).sum()),
                    "called_below": int((sub["call"] == CALLS[2]).sum()),
                    "calls_right": k,
                    "calls_scored": n,
                    "calls_right_pct": round(100 * k / n, 1) if n else None,
                    "p_value_vs_coin": round(_binom_p(k, n), 3),
                    "mean_abs_shift": round(sub["shift"].abs().mean(), 2),
                }
            )
    table = pd.DataFrame(summary)
    print("\nSummary:")
    print(table.to_string(index=False))

    RESULTS_DIR.mkdir(exist_ok=True)
    tag = "_".join(re.sub(r"[^a-z0-9]+", "-", a.lower()).strip("-") for a in df["agent"].unique())
    out = RESULTS_DIR / f"direction_vs_own_anchor__{tag}"
    table.to_csv(f"{out}.csv", index=False)
    df.to_csv(f"{out}__by_forecast.csv", index=False, float_format="%.4f", date_format="%Y-%m-%d")
    print(f"\nWrote {out}.csv\nWrote {out}__by_forecast.csv")


if __name__ == "__main__":
    main()
