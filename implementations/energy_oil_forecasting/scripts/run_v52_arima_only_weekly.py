"""Run CFM Agent v5.2 ARIMA-only and AutoARIMA on the weekly 2026 eval grid.

The biweekly comparison (``energy_oil_eval_biweekly``) has 9 origins. This script
runs the same two predictors on the full weekly ``energy_oil_eval.yaml`` grid —
18 Monday origins, 2026-02-02 to 2026-06-01, horizons 5/10/21 — and compares
them on the identical forecasts:

* mean CRPS per horizon for each predictor,
* the paired per-origin difference (v5.2 ARIMA-only minus AutoARIMA) with its
  standard error, per horizon and averaged over horizons,
* the number of origins where ARIMA-only has the lower CRPS,
* a direction table in the layout of the deck's direction slide: the LLM's
  call (up / neutral / down) against what WTI did from the cutoff's last close,
  plus whether each applied adjustment moved the forecast toward the outcome.

``--model lite`` (default) matches the biweekly run in
``04_news_contrarian_test.ipynb``: ``build_cfm_agent_config_arima_only()`` with
its defaults. ``--model advanced`` puts every LLM call on gemini-3.5-flash —
agent, search, and every ``*_model`` settings field — as the live
``v52_advanced`` stream in ``cfm_coach/streams.py`` does; code execution keeps
the package default so the only change from the lite run is the model.
``DartsAutoARIMAPredictor()`` uses its defaults and is computed once.

Results are cached under a NEW spec_id, ``energy_oil_eval_weekly`` — never
``energy_oil_eval_biweekly`` (kept untouched, checked by hash) and never
``energy_oil_eval`` (its AutoARIMA cache is a stale 8-origin run, and the cache
is keyed by spec_id alone, so reusing it would silently compare different
windows).

Usage (repo root, with .env holding OPENAI_BASE_URL / OPENAI_API_KEY /
E2B_API_KEY, and the WTI cache populated by ``scripts/fetch_wti.py``)::

    uv run python implementations/energy_oil_forecasting/scripts/run_v52_arima_only_weekly.py
    uv run python implementations/energy_oil_forecasting/scripts/run_v52_arima_only_weekly.py --model advanced

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
MODELS = {"lite": "gemini-3.1-flash-lite-preview", "advanced": "gemini-3.5-flash"}
ARIMA_ID = "darts_autoarima"
TASK_ID = "wti_oil_price_forecast"
HORIZONS = (5, 10, 21)
FLAT_EPS = 0.05  # moves under 5 cents are "no call", as on cfm_coach's corpus page
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


def agent_id(model_key: str) -> str:
    return f"agent_predictor_cfm_agent_v_5_2_arima_only_{MODELS[model_key]}_continuous"


def compare(spec_id: str, model_key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score the two predictors on the forecasts both produced."""
    agent = _frame(_load_result(spec_id, agent_id(model_key)), "v52_arima_only")
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


def _call_label(center_action: str) -> str:
    if center_action.endswith("_up"):
        return "Called up"
    if center_action.endswith("_down"):
        return "Called down"
    return "Neutral"


def _sign(x: float) -> int:
    return 0 if abs(x) < FLAT_EPS else (1 if x > 0 else -1)


def direction_rows(spec_id: str, model_key: str, actual: dict[pd.Timestamp, float]) -> pd.DataFrame:
    """One row per agent forecast: the LLM's call, what WTI did, and where the adjustment went."""
    rows = []
    for pred in _load_result(spec_id, agent_id(model_key))["predictions"]:
        meta = pred["metadata"]
        horizon = meta["policy_decision"]["horizon"]
        last = meta["market_diagnostics"]["latest_value"]
        realised = actual[pd.Timestamp(pred["forecast_date"]).normalize()]
        calls = {
            int(a["horizon"]): a["center_action"] for a in meta["llm_context_assessment"].get("horizon_actions", [])
        }
        base = meta["unadjusted_ensemble"]["point_forecast"]
        final = pred["payload"]["point_forecast"]
        rows.append(
            {
                "model": model_key,
                "origin": pd.Timestamp(pred["as_of"]).normalize(),
                "horizon": horizon,
                "llm_center_action": calls.get(horizon, "no_change"),
                "call": _call_label(calls.get(horizon, "no_change")),
                "last_close": last,
                "unadjusted_p50": base,
                "final_p50": final,
                "actual": realised,
                "wti_move": _sign(realised - last),
                "base_dir": _sign(base - last),
                "adjustment": _sign(final - base),
                "adjustment_toward_actual": bool(np.sign(final - base) == np.sign(realised - base)),
                "ended_closer": bool(abs(final - realised) < abs(base - realised)),
            }
        )
    return pd.DataFrame(rows)


def direction_summary(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Print the slide-style grid per horizon and return the headline numbers."""
    out = []
    print(f"\n{'=' * 30} DIRECTION: {label} {'=' * 30}")
    print("Rows: the LLM's call. Columns: what WTI did vs the last close at the cutoff.")
    for h in HORIZONS:
        s = df[df["horizon"] == h]
        grid = pd.crosstab(s["call"], s["wti_move"].map({1: "Rose", -1: "Fell", 0: "Flat"}))
        grid = grid.reindex(index=["Called up", "Neutral", "Called down"]).fillna(0).astype(int)
        called = s[(s["call"] != "Neutral") & (s["wti_move"] != 0)]
        right = ((called["call"] == "Called up") & (called["wti_move"] == 1)) | (
            (called["call"] == "Called down") & (called["wti_move"] == -1)
        )
        base = s[(s["base_dir"] != 0) & (s["wti_move"] != 0)]
        adjusted = s[s["adjustment"] != 0]
        row = {
            "label": label,
            "horizon": f"h={h}",
            "forecasts": len(s),
            "origins": s["origin"].nunique(),
            "outcomes_up": int((s["wti_move"] == 1).sum()),
            "outcomes_down": int((s["wti_move"] == -1).sum()),
            "calls_up": int((s["call"] == "Called up").sum()),
            "calls_neutral": int((s["call"] == "Neutral").sum()),
            "calls_down": int((s["call"] == "Called down").sum()),
            "calls_right": int(right.sum()),
            "calls_scored": len(called),
            "calls_right_pct": right.mean() * 100 if len(called) else np.nan,
            "base_right_pct": (base["base_dir"] == base["wti_move"]).mean() * 100 if len(base) else np.nan,
            "always_up_right_pct": (s["wti_move"] == 1).sum() / max((s["wti_move"] != 0).sum(), 1) * 100,
            "adjusted": len(adjusted),
            "adjusted_toward_actual": int(adjusted["adjustment_toward_actual"].sum()),
            "adjusted_ended_closer": int(adjusted["ended_closer"].sum()),
        }
        out.append(row)
        print(
            f"\n--- {h}-day ({row['forecasts']} forecasts, {row['origins']} origins; "
            f"WTI {row['outcomes_up']} up / {row['outcomes_down']} down) ---"
        )
        print(grid.to_string())
        print(
            f"LLM calls right {row['calls_right']}/{row['calls_scored']} = {row['calls_right_pct']:.0f}%"
            f"  | base ARIMA {row['base_right_pct']:.0f}%  | always-up {row['always_up_right_pct']:.0f}%"
        )
        print(
            f"Adjustments: {row['adjusted']}/{row['forecasts']} moved; toward actual {row['adjusted_toward_actual']}"
            f"/{row['adjusted']}; ended closer {row['adjusted_ended_closer']}/{row['adjusted']}"
        )
    return pd.DataFrame(out)


def run_weekly(model_key: str) -> None:
    """Compute both predictors on the weekly grid under WEEKLY_SPEC_ID."""
    for var in ("OPENAI_BASE_URL", "OPENAI_API_KEY"):
        if not os.getenv(var):
            raise SystemExit(f"{var} is not set (put it in {_REPO / '.env'}).")

    import energy_oil_forecasting  # noqa: PLC0415
    from aieng.forecasting.evaluation import MultiTargetBacktestSpec, cached_multi_backtest  # noqa: PLC0415
    from aieng.forecasting.methods.numerical.darts_arima import DartsAutoARIMAPredictor  # noqa: PLC0415
    from energy_oil_forecasting.cfm_agent_v_5_2 import (  # noqa: PLC0415
        DEFAULT_SETTINGS,
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
    if model_key == "lite":
        config = build_cfm_agent_config_arima_only()  # exactly the biweekly run's settings
    else:
        model = MODELS[model_key]
        # Every settings field naming an LLM ends in "_model" (cfm_coach/targets.py
        # discovers them the same way), so none is left on the package default.
        fields = [name for name in type(DEFAULT_SETTINGS).model_fields if name.endswith("_model")]
        settings = DEFAULT_SETTINGS.model_copy(update=dict.fromkeys(fields, model))
        print(f"All LLM calls on {model}: agent, search, {', '.join(fields)}")
        config = build_cfm_agent_config_arima_only(model=model, search_model=model, settings=settings)
    agent = build_cfm_agent_predictor_arima_only(config)
    if agent.predictor_id != agent_id(model_key):
        raise SystemExit(f"Unexpected agent predictor_id {agent.predictor_id!r}, expected {agent_id(model_key)!r}")

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
    parser.add_argument("--model", choices=sorted(MODELS), default="lite", help="LLM for every agent call")
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
        run_weekly(args.model)

    summary, by_origin = compare(args.spec_id, args.model)

    from datetime import datetime  # noqa: PLC0415

    from energy_oil_forecasting.data import WTI_SERIES_ID, build_wti_multivariate_service  # noqa: PLC0415

    prices = build_wti_multivariate_service().get_series(WTI_SERIES_ID, as_of=datetime.now())
    actual = {pd.Timestamp(t).normalize(): float(v) for t, v in zip(prices["timestamp"], prices["value"], strict=True)}
    directions = direction_rows(args.spec_id, args.model, actual)
    dir_tables = [direction_summary(directions, args.model)]
    # Pool lite and advanced (as the deck's direction slide does) once both exist.
    if all((STORE / args.spec_id / f"{agent_id(k)}__{TASK_ID}.yaml").exists() for k in MODELS):
        pooled = pd.concat([direction_rows(args.spec_id, k, actual) for k in MODELS])
        dir_tables.append(direction_summary(pooled, "lite + advanced pooled"))

    if _hash_dir(protected) != before:
        raise SystemExit(f"{protected} changed during this run — it must not be touched.")
    print(f"{PROTECTED_SPEC_ID} cache unchanged ({len(before)} files, hashes verified).")

    RESULTS_DIR.mkdir(exist_ok=True)
    tag = f"{args.model}__{args.spec_id}"
    summary_path = RESULTS_DIR / f"v52_arima_only_vs_autoarima__{tag}.csv"
    origin_path = RESULTS_DIR / f"v52_arima_only_vs_autoarima__{tag}__by_origin.csv"
    dir_path = RESULTS_DIR / f"v52_arima_only_direction__{tag}.csv"
    dir_rows_path = RESULTS_DIR / f"v52_arima_only_direction__{tag}__by_forecast.csv"
    summary.to_csv(summary_path, index=False, float_format="%.4f")
    by_origin.to_csv(origin_path, index=False, float_format="%.4f", date_format="%Y-%m-%d")
    pd.concat(dir_tables).to_csv(dir_path, index=False, float_format="%.1f")
    directions.to_csv(dir_rows_path, index=False, float_format="%.4f", date_format="%Y-%m-%d")

    pd.set_option("display.width", 160)
    print("\nPaired diff = v5.2 ARIMA-only CRPS minus AutoARIMA CRPS (negative = ARIMA-only better)\n")
    print(summary.drop(columns="spec_id").round(3).to_string(index=False))
    print(f"\nWrote {summary_path}\nWrote {origin_path}\nWrote {dir_path}\nWrote {dir_rows_path}")


if __name__ == "__main__":
    main()
