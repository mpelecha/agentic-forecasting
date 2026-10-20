"""Run CFM Agent v5.2 ARIMA-only (lite) and AutoARIMA on the weekly 2026 eval grid.

The biweekly comparison (``energy_oil_eval_biweekly``) has 9 origins. This script
runs the same two predictors on the full weekly ``energy_oil_eval.yaml`` grid —
18 Monday origins, 2026-02-02 to 2026-06-01, horizons 5/10/21 — and compares
them on the identical forecasts:

* mean CRPS per horizon for each predictor,
* the paired per-origin difference (v5.2 ARIMA-only minus AutoARIMA) with its
  standard error, per horizon and averaged over horizons,
* the number of origins where ARIMA-only has the lower CRPS.

Settings match the biweekly run in ``04_news_contrarian_test.ipynb``:
``build_cfm_agent_config_arima_only()`` with its defaults (lite model) and
``DartsAutoARIMAPredictor()`` with its defaults; only the stride (5, not 10)
and the spec_id differ.

Results are cached under a NEW spec_id, ``energy_oil_eval_weekly`` — never
``energy_oil_eval_biweekly`` (kept untouched, checked by hash) and never
``energy_oil_eval`` (its AutoARIMA cache is a stale 8-origin run, and the cache
is keyed by spec_id alone, so reusing it would silently compare different
windows).

Usage (repo root, with .env holding OPENAI_BASE_URL / OPENAI_API_KEY /
E2B_API_KEY, and the WTI cache populated by ``scripts/fetch_wti.py``)::

    uv run python implementations/energy_oil_forecasting/scripts/run_v52_arima_only_weekly.py

    # Compare only, from caches already on disk (no LLM calls, no data fetch):
    uv run python implementations/energy_oil_forecasting/scripts/run_v52_arima_only_weekly.py \
        --compare-only --spec-id energy_oil_eval_biweekly

The agent takes roughly 4 minutes per origin, so expect over an hour. The
cache is written per task when the whole run finishes, so an interrupted run
starts over.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve()
_PKG = _HERE.parents[1]  # implementations/energy_oil_forecasting
_REPO = _HERE.parents[3]
sys.path.insert(0, str(_HERE.parents[2]))  # <repo>/implementations

# Absolute so the working directory cannot change where caches go (see
# run_autoarima_agent_grid.py for the failure this prevents).
STORE = _PKG / "data" / "predictions"
RESULTS_DIR = _PKG / "results"

WEEKLY_SPEC_ID = "energy_oil_eval_weekly"
PROTECTED_SPEC_ID = "energy_oil_eval_biweekly"
AGENT_ID = "agent_predictor_cfm_agent_v_5_2_arima_only_gemini-3.1-flash-lite-preview_continuous"
ARIMA_ID = "darts_autoarima"
TASK_ID = "wti_oil_price_forecast"
HORIZONS = (5, 10, 21)
EXPECTED_ORIGINS = pd.date_range("2026-02-02", "2026-06-01", freq="W-MON")


def _hash_dir(path: Path) -> dict[str, str]:
    """SHA-256 of every file under ``path`` — used to prove the biweekly cache is untouched."""
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(path.glob("*")) if p.is_file()}


def _business_horizon(as_of: pd.Timestamp, forecast_date: pd.Timestamp) -> int:
    # Same definition as energy_oil_forecasting.analysis._business_horizon.
    return max(len(pd.bdate_range(as_of.normalize(), forecast_date.normalize())) - 1, 0)


def _load_result(spec_id: str, predictor_id: str) -> dict:
    path = STORE / spec_id / f"{predictor_id}__{TASK_ID}.yaml"
    if not path.exists():
        raise SystemExit(f"Missing cache: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def _frame(raw: dict, name: str) -> pd.DataFrame:
    """One row per scored forecast: origin, horizon, CRPS (as stored by the backtest)."""
    rows = []
    for pred, score in zip(raw["predictions"], raw["scores"], strict=True):
        as_of = pd.Timestamp(pred["as_of"])
        fdate = pd.Timestamp(pred["forecast_date"])
        rows.append(
            {
                "predictor": name,
                "origin": as_of.normalize(),
                "forecast_date": fdate.normalize(),
                "horizon": _business_horizon(as_of, fdate),
                "crps": float(score),
            }
        )
    return pd.DataFrame(rows)


def _paired_stats(diff_by_origin: pd.Series) -> dict:
    d = diff_by_origin.dropna()
    n = len(d)
    return {
        "n_origins": n,
        "mean_diff": d.mean() if n else np.nan,
        "se_diff": d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan,
        "arima_only_wins": int((d < 0).sum()),
        "autoarima_wins": int((d > 0).sum()),
        "ties": int((d == 0).sum()),
    }


def compare(spec_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score the two predictors on the forecasts both produced."""
    agent = _frame(_load_result(spec_id, AGENT_ID), "v52_arima_only")
    arima = _frame(_load_result(spec_id, ARIMA_ID), "autoarima")

    keys = ["origin", "forecast_date"]
    both = agent.merge(arima, on=keys, suffixes=("_agent", "_arima"), how="inner")
    dropped = len(agent) + len(arima) - 2 * len(both)
    if (both["horizon_agent"] != both["horizon_arima"]).any():
        raise SystemExit("Horizon mismatch between predictors on the same forecast date.")
    both = both.rename(columns={"horizon_agent": "horizon"}).drop(columns="horizon_arima")
    both["diff"] = both["crps_agent"] - both["crps_arima"]

    print(
        f"spec_id={spec_id}: agent rows={len(agent)}, AutoARIMA rows={len(arima)}, "
        f"paired rows={len(both)}, unpaired rows dropped={dropped}"
    )
    origins = sorted(both["origin"].unique())
    print(f"paired origins={len(origins)}: {origins[0].date()} .. {origins[-1].date()}")

    # Holiday-adjacent forecast dates can shift the business-day count; map any
    # off-grid horizon to the nearest spec horizon rather than dropping it.
    both["horizon"] = both["horizon"].map(lambda h: min(HORIZONS, key=lambda s: abs(s - h)))

    summary_rows = []
    for h in HORIZONS:
        sub = both[both["horizon"] == h]
        summary_rows.append(
            {
                "spec_id": spec_id,
                "horizon": f"h={h}",
                "n_rows": len(sub),
                "crps_v52_arima_only": sub["crps_agent"].mean(),
                "crps_autoarima": sub["crps_arima"].mean(),
                **_paired_stats(sub.set_index("origin")["diff"]),
            }
        )
    # Averaged over horizons: one paired difference per origin, so the SE is
    # not inflated by treating overlapping horizons as independent.
    per_origin = both.groupby("origin").agg(
        n_horizons=("diff", "size"),
        crps_v52_arima_only=("crps_agent", "mean"),
        crps_autoarima=("crps_arima", "mean"),
        diff=("diff", "mean"),
    )
    summary_rows.append(
        {
            "spec_id": spec_id,
            "horizon": "all (per-origin mean)",
            "n_rows": len(both),
            "crps_v52_arima_only": both["crps_agent"].mean(),
            "crps_autoarima": both["crps_arima"].mean(),
            **_paired_stats(per_origin["diff"]),
        }
    )
    summary = pd.DataFrame(summary_rows)

    by_origin = both.pivot_table(index="origin", columns="horizon", values="diff")
    by_origin.columns = [f"diff_h{h}" for h in by_origin.columns]
    by_origin = per_origin.join(by_origin).reset_index()
    by_origin.insert(0, "spec_id", spec_id)
    by_origin["winner"] = np.where(
        by_origin["diff"] < 0, "v52_arima_only", np.where(by_origin["diff"] > 0, "autoarima", "tie")
    )
    return summary, by_origin


def run_weekly() -> None:
    """Compute both predictors on the weekly grid under WEEKLY_SPEC_ID."""
    for var in ("OPENAI_BASE_URL", "OPENAI_API_KEY"):
        if not os.getenv(var):
            raise SystemExit(f"{var} is not set (put it in {_REPO / '.env'}).")

    import energy_oil_forecasting  # noqa: PLC0415
    from aieng.forecasting.evaluation import MultiTargetBacktestSpec, cached_multi_backtest  # noqa: PLC0415
    from aieng.forecasting.methods.numerical.darts_arima import DartsAutoARIMAPredictor  # noqa: PLC0415
    from energy_oil_forecasting.cfm_agent_v_5_2 import (  # noqa: PLC0415
        build_cfm_agent_config_arima_only,
        build_cfm_agent_predictor_arima_only,
    )
    from energy_oil_forecasting.data import build_wti_multivariate_service  # noqa: PLC0415

    with open(Path(energy_oil_forecasting.__file__).parent / "specs" / "energy_oil_eval.yaml") as f:
        spec = MultiTargetBacktestSpec.model_validate(yaml.safe_load(f))
    spec.spec_id = WEEKLY_SPEC_ID
    if spec.stride != 5 or str(spec.start)[:10] != "2026-02-02" or str(spec.end)[:10] != "2026-06-01":
        raise SystemExit(f"energy_oil_eval.yaml changed: {spec.start}..{spec.end} stride={spec.stride}")

    data_service = build_wti_multivariate_service()
    agent = build_cfm_agent_predictor_arima_only(build_cfm_agent_config_arima_only())
    if agent.predictor_id != AGENT_ID:
        raise SystemExit(f"Unexpected agent predictor_id {agent.predictor_id!r}, expected {AGENT_ID!r}")

    for predictor in (DartsAutoARIMAPredictor(), agent):
        print(f"Running {predictor.predictor_id} on {WEEKLY_SPEC_ID} ...", flush=True)
        out = cached_multi_backtest(predictor, spec, data_service, store_dir=STORE)
        if TASK_ID not in out:
            raise SystemExit(f"{predictor.predictor_id} failed; see the WARNING log above.")
        got = sorted({pd.Timestamp(p.as_of).normalize() for p in out[TASK_ID].predictions})
        missing = sorted(set(EXPECTED_ORIGINS) - set(got))
        print(f"  {len(got)} origins" + (f"; MISSING {[d.date() for d in missing]}" if missing else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--compare-only", action="store_true", help="skip running; compare cached results")
    parser.add_argument("--spec-id", default=WEEKLY_SPEC_ID, help="cache to compare (default: weekly)")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv  # noqa: PLC0415

        load_dotenv(_REPO / ".env", override=False)
    except ImportError:
        pass

    protected = STORE / PROTECTED_SPEC_ID
    before = _hash_dir(protected)

    if not args.compare_only:
        if args.spec_id != WEEKLY_SPEC_ID:
            raise SystemExit("Running writes only to the weekly spec_id; use --compare-only for others.")
        run_weekly()

    summary, by_origin = compare(args.spec_id)

    if _hash_dir(protected) != before:
        raise SystemExit(f"{protected} changed during this run — it must not be touched.")
    print(f"{PROTECTED_SPEC_ID} cache unchanged ({len(before)} files, hashes verified).")

    RESULTS_DIR.mkdir(exist_ok=True)
    tag = args.spec_id
    summary_path = RESULTS_DIR / f"v52_arima_only_vs_autoarima__{tag}.csv"
    origin_path = RESULTS_DIR / f"v52_arima_only_vs_autoarima__{tag}__by_origin.csv"
    summary.to_csv(summary_path, index=False, float_format="%.4f")
    by_origin.to_csv(origin_path, index=False, float_format="%.4f", date_format="%Y-%m-%d")

    pd.set_option("display.width", 160)
    print("\nPaired diff = v5.2 ARIMA-only CRPS minus AutoARIMA CRPS (negative = ARIMA-only better)\n")
    print(summary.drop(columns="spec_id").round(3).to_string(index=False))
    print(f"\nWrote {summary_path}\nWrote {origin_path}")


if __name__ == "__main__":
    main()
