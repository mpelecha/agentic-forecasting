"""Extend the v5.2 ARIMA-only lite vs AutoARIMA comparison to every Monday from June 2025.

The weekly comparison (``energy_oil_eval_weekly``) covers 18 Mondays, 2026-02-02 to
2026-06-01. ``specs/energy_oil_eval_weekly_extended.yaml`` extends the grid back to
2025-06-02: 53 Mondays, horizons 5/10/21, same task, stride, and warmup.

Nothing that already exists is rerun or overwritten (each existing cache folder is
hash-checked before and after). Results come from three places:

==========================  =====================================  ============================
Origins                     v5.2 ARIMA-only lite                   AutoARIMA
==========================  =====================================  ============================
2026-02-02 .. 2026-06-01    energy_oil_eval_weekly (cached)        energy_oil_eval_weekly (cached)
2025-06-02 .. 2026-01-26    energy_oil_backtest_biweekly (cached)  new run, every origin
                            where present; new run otherwise
==========================  =====================================  ============================

New runs are saved one origin at a time under
``data/predictions/energy_oil_eval_weekly_extended/<origin>/``, so an interrupted run
resumes from the next missing origin. The predictor and its settings are exactly the
weekly lite run's: ``build_cfm_agent_config_arima_only()`` with its defaults.

Before running, a one-token request to the proxy records which model actually serves
``gemini-3.1-flash-lite-preview`` (the predictions do not record it). A fresh
AutoARIMA run on the 18 cached Feb-Jun origins (not saved) checks that the price data
reproduces the cached forecasts before old and new results are combined.

Usage (repo root, on the branch with cfm_agent_v_5_2; keys in the environment)::

    S=implementations/energy_oil_forecasting/scripts/run_v52_weekly_extended.py
    uv run python $S --plan          # list what will run and what is reused; no calls
    uv run python $S                 # run the missing origins, then analyse
    uv run python $S --analyze-only  # analyse what is on disk
    uv run python $S --audit-dates   # check source publication dates (fetches pages)

The analysis prints and saves (to ``results/``) a summary per period (Jun 2025-Jan 2026,
Feb-Jun 2026, all) and horizon — 5-day is the primary result — plus a per-origin CSV.

``--audit-dates`` checks, for every origin, the publication date of each news source
the agent was given (``active_research_packet.sources``). It fetches each page once and
reads the date with v5.2's own audit extractor (HTML metadata, JSON-LD, then visible
text). It does not rerun or change any forecast. Limits: a page with no detectable date
stays unknown; a live page (quote pages, "latest" reports) shows today's content, not
what existed at the origin; and searching today's web for a past date can still shape
*which* pre-origin articles are found.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve()
_PKG = _HERE.parents[1]
_REPO = _HERE.parents[3]
sys.path.insert(0, str(_HERE.parents[2]))

STORE = _PKG / "data" / "predictions"
RESULTS_DIR = _PKG / "results"
SPEC_PATH = _PKG / "specs" / "energy_oil_eval_weekly_extended.yaml"
REFERENCE_SPEC_PATH = _PKG / "specs" / "energy_oil_eval.yaml"

NEW_SPEC_ID = "energy_oil_eval_weekly_extended"
WEEKLY_SPEC_ID = "energy_oil_eval_weekly"
BIWEEKLY_2025_SPEC_ID = "energy_oil_backtest_biweekly"
PROTECTED_SPEC_IDS = (WEEKLY_SPEC_ID, BIWEEKLY_2025_SPEC_ID, "energy_oil_eval_biweekly", "energy_oil_backtest")

MODEL = "gemini-3.1-flash-lite-preview"
AGENT_ID = f"agent_predictor_cfm_agent_v_5_2_arima_only_{MODEL}_continuous"
ARIMA_ID = "darts_autoarima"
TASK_ID = "wti_oil_price_forecast"
HORIZONS = (5, 10, 21)
PRIMARY_HORIZON = 5

PERIODS = {
    "Jun 2025-Jan 2026": ("2025-06-02", "2026-01-26"),
    "Feb-Jun 2026": ("2026-02-02", "2026-06-01"),
}
# Published knowledge cutoff of gemini-3.1-flash-lite-preview (Google model card as
# reported by third-party listings; see the conversation that added this script).
MODEL_KNOWLEDGE_CUTOFF = pd.Timestamp("2025-01-31")
MODEL_KNOWLEDGE_CUTOFF_NOTE = "2025-01-31 (January 2025), per published model listings"


# ── spec and origins ──────────────────────────────────────────────────────────


def load_spec() -> dict[str, Any]:
    with open(SPEC_PATH) as f:
        spec = yaml.safe_load(f)
    with open(REFERENCE_SPEC_PATH) as f:
        reference = yaml.safe_load(f)
    # The prompt includes task.description; the extended spec must not change it.
    if spec["tasks"] != reference["tasks"]:
        raise SystemExit("Task block differs from energy_oil_eval.yaml; the agent's prompt would change.")
    if (spec["stride"], spec["warmup"]) != (reference["stride"], reference["warmup"]):
        raise SystemExit("Stride or warmup differs from energy_oil_eval.yaml.")
    return spec


def all_origins(spec: dict[str, Any]) -> list[pd.Timestamp]:
    grid = list(pd.date_range(spec["start"], spec["end"], freq=f"{spec['stride']}B"))
    if any(d.weekday() != 0 for d in grid) or len(grid) != 53:
        raise SystemExit(f"Expected 53 Mondays, got {len(grid)}: check the spec window.")
    return grid


def period_of(origin: pd.Timestamp) -> str:
    for name, (lo, hi) in PERIODS.items():
        if pd.Timestamp(lo) <= origin <= pd.Timestamp(hi):
            return name
    raise ValueError(f"{origin.date()} is outside every period")


# ── cache reading ─────────────────────────────────────────────────────────────


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f)


def _new_path(origin: pd.Timestamp, predictor_id: str) -> Path:
    return STORE / NEW_SPEC_ID / str(origin.date()) / f"{predictor_id}.yaml"


def _origins_in(raw: dict[str, Any] | None) -> set[pd.Timestamp]:
    return {pd.Timestamp(p["as_of"]).normalize() for p in (raw or {}).get("predictions", [])}


def _horizon(as_of: pd.Timestamp, forecast_date: pd.Timestamp) -> int:
    steps = max(len(pd.bdate_range(as_of.normalize(), forecast_date.normalize())) - 1, 0)
    return min(HORIZONS, key=lambda h: abs(h - steps))


def _rows(raw: dict[str, Any], keep: set[pd.Timestamp], source: str) -> list[dict[str, Any]]:
    rows = []
    for pred, score in zip(raw["predictions"], raw["scores"], strict=True):
        as_of = pd.Timestamp(pred["as_of"]).normalize()
        if as_of not in keep:
            continue
        audit = (pred.get("metadata") or {}).get("structured_output_control_audit") or {}
        rows.append(
            {
                "origin": as_of,
                "forecast_date": pd.Timestamp(pred["forecast_date"]).normalize(),
                "horizon": _horizon(as_of, pd.Timestamp(pred["forecast_date"])),
                "crps": float(score),
                "source": source,
                "disposition": audit.get("final_disposition"),
                "issued_at": str(pred.get("issued_at", ""))[:10],
            }
        )
    return rows


def plan(origins: list[pd.Timestamp]) -> dict[str, list[pd.Timestamp]]:
    """Which origins each predictor still needs, after every reusable cache is counted."""
    new = [o for o in origins if period_of(o) == "Jun 2025-Jan 2026"]
    weekly = [o for o in origins if period_of(o) == "Feb-Jun 2026"]
    biweekly_agent = _origins_in(_load(STORE / BIWEEKLY_2025_SPEC_ID / f"{AGENT_ID}__{TASK_ID}.yaml"))
    for pid in (AGENT_ID, ARIMA_ID):
        have = _origins_in(_load(STORE / WEEKLY_SPEC_ID / f"{pid}__{TASK_ID}.yaml"))
        if not set(weekly) <= have:
            raise SystemExit(f"{WEEKLY_SPEC_ID} is missing {pid} origins: {sorted(set(weekly) - have)}")
    return {
        "reuse_agent_biweekly": [o for o in new if o in biweekly_agent],
        "run_agent": [o for o in new if o not in biweekly_agent and not _new_path(o, AGENT_ID).exists()],
        "done_agent": [o for o in new if o not in biweekly_agent and _new_path(o, AGENT_ID).exists()],
        "run_arima": [o for o in new if not _new_path(o, ARIMA_ID).exists()],
        "done_arima": [o for o in new if _new_path(o, ARIMA_ID).exists()],
        "reuse_weekly": weekly,
    }


def print_plan(p: dict[str, list[pd.Timestamp]]) -> None:
    def _fmt(ds: list[pd.Timestamp]) -> str:
        return ", ".join(str(d.date()) for d in ds) if ds else "-"

    print("Plan")
    print(f"  Feb-Jun 2026, both predictors, reused from {WEEKLY_SPEC_ID}: {len(p['reuse_weekly'])}")
    print(f"  Jun 2025-Jan 2026, v5.2 reused from {BIWEEKLY_2025_SPEC_ID}: {len(p['reuse_agent_biweekly'])}")
    print(f"    {_fmt(p['reuse_agent_biweekly'])}")
    print(f"  v5.2 already run here: {len(p['done_agent'])};  to run now: {len(p['run_agent'])}")
    print(f"    {_fmt(p['run_agent'])}")
    print(f"  AutoARIMA already run here: {len(p['done_arima'])};  to run now: {len(p['run_arima'])}")


# ── running ───────────────────────────────────────────────────────────────────


def probe_served_model() -> dict[str, Any]:
    """Ask the proxy which model actually answers MODEL (predictions do not record it)."""
    import litellm  # noqa: PLC0415

    response = litellm.completion(
        model=f"openai/{MODEL}",
        api_base=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        messages=[{"role": "user", "content": "Reply with OK."}],
        max_tokens=5,
        temperature=0.0,
    )
    served = {"requested": MODEL, "served": getattr(response, "model", None), "probed_at": datetime.now().isoformat()}
    print(f"Proxy model check: requested {MODEL!r}, served {served['served']!r}")
    return served


def _one_origin_spec(spec: dict[str, Any], origin: pd.Timestamp) -> Any:
    from aieng.forecasting.evaluation.backtest import BacktestSpec  # noqa: PLC0415
    from aieng.forecasting.evaluation.task import ForecastingTask  # noqa: PLC0415

    return BacktestSpec(
        task=ForecastingTask.model_validate(spec["tasks"][0]),
        start=origin.to_pydatetime(),
        end=(origin + timedelta(days=1)).to_pydatetime(),
        origin_dates=[origin.to_pydatetime()],
        stride=spec["stride"],
        warmup=spec["warmup"],
        description=spec["description"],
    )


def check_arima_reproduces(spec: dict[str, Any], data_service: Any) -> None:
    """Recompute AutoARIMA on the cached Feb-Jun origins (unsaved) and compare scores."""
    from aieng.forecasting.evaluation.backtest import backtest  # noqa: PLC0415
    from aieng.forecasting.methods.numerical.darts_arima import DartsAutoARIMAPredictor  # noqa: PLC0415

    cached = _load(STORE / WEEKLY_SPEC_ID / f"{ARIMA_ID}__{TASK_ID}.yaml")
    old = {
        (p["as_of"][:10], p["forecast_date"][:10]): s
        for p, s in zip(cached["predictions"], cached["scores"], strict=True)
    }
    predictor = DartsAutoARIMAPredictor()
    gaps = []
    for origin in sorted({pd.Timestamp(k[0]) for k in old}):
        result = backtest(predictor=predictor, spec=_one_origin_spec(spec, origin), data_service=data_service)
        for p, s in zip(result.predictions, result.scores, strict=True):
            key = (str(p.as_of)[:10], str(p.forecast_date)[:10])
            if key in old:
                gaps.append(abs(float(s) - old[key]))
    worst = max(gaps) if gaps else float("nan")
    print(f"AutoARIMA reproducibility on {len(gaps)} cached Feb-Jun forecasts: max |CRPS difference| = {worst:.4f}")
    if not gaps or worst > 0.01:
        raise SystemExit("Fresh AutoARIMA does not reproduce the cached Feb-Jun results; data differ. Stopping.")


def run_missing(spec: dict[str, Any], p: dict[str, list[pd.Timestamp]]) -> None:
    for var in ("OPENAI_BASE_URL", "OPENAI_API_KEY"):
        if not os.getenv(var):
            raise SystemExit(f"{var} is not set.")

    from aieng.forecasting.evaluation import cached_backtest  # noqa: PLC0415
    from aieng.forecasting.methods.numerical.darts_arima import DartsAutoARIMAPredictor  # noqa: PLC0415
    from energy_oil_forecasting.cfm_agent_v_5_2 import (  # noqa: PLC0415
        build_cfm_agent_config_arima_only,
        build_cfm_agent_predictor_arima_only,
    )
    from energy_oil_forecasting.data import build_wti_multivariate_service  # noqa: PLC0415

    RESULTS_DIR.mkdir(exist_ok=True)
    served = probe_served_model()
    (RESULTS_DIR / "v52_weekly_extended__served_model.json").write_text(json.dumps(served, indent=2))

    data_service = build_wti_multivariate_service()
    check_arima_reproduces(spec, data_service)

    arima = DartsAutoARIMAPredictor()
    for i, origin in enumerate(p["run_arima"], 1):
        cached_backtest(
            arima, _one_origin_spec(spec, origin), f"{NEW_SPEC_ID}/{origin.date()}", data_service, store_dir=STORE
        )
        print(f"  AutoARIMA {i}/{len(p['run_arima'])} {origin.date()} done", flush=True)

    if p["run_agent"]:
        agent = build_cfm_agent_predictor_arima_only(build_cfm_agent_config_arima_only())  # weekly lite settings
        if agent.predictor_id != AGENT_ID:
            raise SystemExit(f"Unexpected predictor_id {agent.predictor_id!r}")
        for i, origin in enumerate(p["run_agent"], 1):
            started = datetime.now()
            result = cached_backtest(
                agent, _one_origin_spec(spec, origin), f"{NEW_SPEC_ID}/{origin.date()}", data_service, store_dir=STORE
            )
            took = (datetime.now() - started).total_seconds() / 60
            print(
                f"  v5.2 {i}/{len(p['run_agent'])} {origin.date()} done in {took:.1f} min ({len(result.predictions)} forecasts)",
                flush=True,
            )


# ── analysis ──────────────────────────────────────────────────────────────────


def assemble(origins: list[pd.Timestamp]) -> pd.DataFrame:
    """Pair v5.2 and AutoARIMA on identical (origin, forecast date) across all three caches."""
    new = {o for o in origins if period_of(o) == "Jun 2025-Jan 2026"}
    weekly = {o for o in origins if period_of(o) == "Feb-Jun 2026"}
    frames = {AGENT_ID: [], ARIMA_ID: []}
    for pid in (AGENT_ID, ARIMA_ID):
        raw = _load(STORE / WEEKLY_SPEC_ID / f"{pid}__{TASK_ID}.yaml")
        frames[pid] += _rows(raw, weekly, WEEKLY_SPEC_ID)
        for origin in sorted(new):
            raw = _load(_new_path(origin, pid))
            if raw:
                frames[pid] += _rows(raw, {origin}, NEW_SPEC_ID)
    biweekly = _load(STORE / BIWEEKLY_2025_SPEC_ID / f"{AGENT_ID}__{TASK_ID}.yaml")
    if biweekly:
        frames[AGENT_ID] += _rows(biweekly, new, BIWEEKLY_2025_SPEC_ID)

    agent = pd.DataFrame(frames[AGENT_ID])
    arima = pd.DataFrame(frames[ARIMA_ID])
    for name, df in (("v5.2", agent), ("AutoARIMA", arima)):
        dup = df.duplicated(["origin", "forecast_date"])
        if dup.any():
            raise SystemExit(f"{name}: the same forecast appears in two caches: {df[dup].head()}")
    pairs = agent.merge(arima, on=["origin", "forecast_date"], suffixes=("_agent", "_arima"))
    pairs = pairs.rename(columns={"horizon_agent": "horizon", "source_agent": "agent_source"})
    pairs["diff"] = pairs["crps_agent"] - pairs["crps_arima"]
    pairs["period"] = pairs["origin"].map(period_of)
    missing = sorted(set(origins) - set(pairs["origin"]))
    if missing:
        print(f"WARNING: {len(missing)} origins have no paired forecast yet: {[str(d.date()) for d in missing]}")
    return pairs


def _paired(per_origin: pd.Series) -> dict[str, Any]:
    d = per_origin.dropna()
    n = len(d)
    return {
        "n_origins": n,
        "mean_diff": d.mean() if n else np.nan,
        "se_diff": d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan,
        "v52_wins": int((d < 0).sum()),
        "autoarima_wins": int((d > 0).sum()),
        "ties": int((d == 0).sum()),
    }


def summarise(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [*PERIODS, "All dates"]
    for period in groups:
        sub_p = pairs if period == "All dates" else pairs[pairs["period"] == period]
        for h in HORIZONS:
            sub = sub_p[sub_p["horizon"] == h]
            rows.append(
                {
                    "period": period,
                    "horizon": f"{h}-day",
                    "role": "primary" if h == PRIMARY_HORIZON else "secondary (overlapping forecasts; SE too small)",
                    "crps_v52_arima_only": sub["crps_agent"].mean(),
                    "crps_autoarima": sub["crps_arima"].mean(),
                    **_paired(sub.set_index("origin")["diff"]),
                }
            )
    return pd.DataFrame(rows)


def per_origin_table(pairs: pd.DataFrame) -> pd.DataFrame:
    wide = pairs.pivot_table(index="origin", columns="horizon", values=["crps_agent", "crps_arima", "diff"])
    wide.columns = [
        f"{name.replace('crps_agent', 'crps_v52').replace('crps_arima', 'crps_autoarima')}_h{h}"
        for name, h in wide.columns
    ]
    info = pairs.groupby("origin").agg(
        period=("period", "first"),
        v52_source=("agent_source", "first"),
        v52_disposition=("disposition_agent", "first"),
        v52_issued=("issued_at_agent", "first"),
    )
    out = info.join(wide).reset_index()
    out["winner_h5"] = np.where(out.get("diff_h5") < 0, "v5.2", np.where(out.get("diff_h5") > 0, "AutoARIMA", "tie"))
    return out


def report(summary: pd.DataFrame, per_origin: pd.DataFrame) -> None:
    pd.set_option("display.width", 200)
    show = summary.drop(columns="role").copy()
    print("\nPaired diff = v5.2 ARIMA-only lite CRPS minus AutoARIMA CRPS (negative = v5.2 better).")
    print(
        f"Primary result: {PRIMARY_HORIZON}-day. 10- and 21-day forecasts overlap week to week, so their SEs are too small.\n"
    )
    for period in show["period"].unique():
        print(f"--- {period} ---")
        print(show[show["period"] == period].drop(columns="period").round(3).to_string(index=False))
        print()

    origins = pd.to_datetime(per_origin["origin"])
    before = int((origins <= MODEL_KNOWLEDGE_CUTOFF).sum())
    print(f"Model knowledge cutoff ({MODEL}): {MODEL_KNOWLEDGE_CUTOFF_NOTE}")
    print(f"Origins on or before it: {before} of {len(origins)}")
    print("\nv5.2 results by source and structured-output outcome:")
    print(per_origin.groupby(["period", "v52_source", "v52_disposition"]).size().to_string())


# ── source publication dates ─────────────────────────────────────────────────


def _sources_by_origin(origins: list[pd.Timestamp]) -> dict[pd.Timestamp, list[dict[str, Any]]]:
    """Return the agent's research-packet sources per origin, from the caches the analysis uses."""
    raws = [_load(STORE / WEEKLY_SPEC_ID / f"{AGENT_ID}__{TASK_ID}.yaml")]
    raws += [_load(STORE / BIWEEKLY_2025_SPEC_ID / f"{AGENT_ID}__{TASK_ID}.yaml")]
    raws += [_load(_new_path(o, AGENT_ID)) for o in origins]
    wanted = set(origins)
    out: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    for raw in raws:
        for pred in (raw or {}).get("predictions", []):
            origin = pd.Timestamp(pred["as_of"]).normalize()
            if origin in wanted and origin not in out:  # every horizon shares one packet
                packet = (pred.get("metadata") or {}).get("active_research_packet") or {}
                out[origin] = packet.get("sources", [])
    return out


def audit_dates(origins: list[pd.Timestamp]) -> None:
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    from energy_oil_forecasting.cfm_agent_v_5_2 import DEFAULT_SETTINGS  # noqa: PLC0415
    from energy_oil_forecasting.cfm_agent_v_5_2.schemas import ActiveSource  # noqa: PLC0415
    from energy_oil_forecasting.cfm_agent_v_5_2.source_validator import _inspect_source  # noqa: PLC0415

    by_origin = _sources_by_origin(origins)
    unique = {}
    for sources in by_origin.values():
        for src in sources:
            unique.setdefault(src.get("resolved_url") or src.get("grounding_url"), src)
    print(
        f"Checking publication dates: {sum(map(len, by_origin.values()))} sources at {len(by_origin)} origins, "
        f"{len(unique)} distinct pages"
    )

    def _inspect(src: dict[str, Any]) -> Any:
        # A far-future cutoff: only the extracted date is used; eligibility is computed per origin below.
        return _inspect_source(ActiveSource.model_validate(src), "9999-12-31", DEFAULT_SETTINGS)

    with ThreadPoolExecutor(max_workers=8) as pool:
        records = dict(zip(unique, pool.map(_inspect, unique.values()), strict=True))

    rows = []
    for origin, sources in sorted(by_origin.items()):
        for src in sources:
            rec = records[src.get("resolved_url") or src.get("grounding_url")]
            if not rec.resolved_url:
                status = "unresolved"
            elif not rec.publication_date:
                status = "date unknown"
            elif pd.Timestamp(rec.publication_date) < origin:
                status = "before origin"
            else:
                status = "ON/AFTER ORIGIN"
            rows.append(
                {
                    "origin": origin,
                    "period": period_of(origin),
                    "source_id": src.get("source_id"),
                    "url": rec.resolved_url or src.get("grounding_url"),
                    "domain": src.get("resolved_domain"),
                    "publication_date": rec.publication_date,
                    "date_from": rec.publication_date_provenance,
                    "status": status,
                    "live_page": rec.potentially_mutable,
                    "fetch_warnings": "; ".join(rec.extraction_warnings)[:300],
                }
            )
    df = pd.DataFrame(rows)
    missing = sorted(set(origins) - set(by_origin))
    if missing:
        print(f"WARNING: no agent forecast yet for {len(missing)} origins: {[str(d.date()) for d in missing]}")

    print("\nSources by publication date, per period:")
    table = df.pivot_table(index="period", columns="status", values="url", aggfunc="count", fill_value=0)
    table.loc["All dates"] = table.sum()
    print(table.to_string())
    late = df[df["status"] == "ON/AFTER ORIGIN"]
    print(
        f"\nOrigins with at least one source dated on/after the origin: {late['origin'].nunique()} of {len(by_origin)}"
    )
    if len(late):
        print(late[["origin", "publication_date", "date_from", "url"]].head(20).to_string(index=False))
    print(f"Sources flagged as live pages (content can change after the origin): {int(df['live_page'].sum())}")

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "v52_weekly_extended__source_dates.csv"
    df.to_csv(path, index=False, date_format="%Y-%m-%d")
    print(f"\nWrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--plan", action="store_true", help="print the plan and stop")
    parser.add_argument("--analyze-only", action="store_true", help="skip running; analyse what is on disk")
    parser.add_argument("--audit-dates", action="store_true", help="check source publication dates and stop")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv  # noqa: PLC0415

        load_dotenv(_REPO / ".env", override=False)
    except ImportError:
        pass

    spec = load_spec()
    origins = all_origins(spec)
    p = plan(origins)
    print_plan(p)
    if args.plan:
        return
    if args.audit_dates:
        audit_dates(origins)
        return

    protected = {sid: _hash_dir(STORE / sid) for sid in PROTECTED_SPEC_IDS}
    if not args.analyze_only:
        run_missing(spec, p)

    pairs = assemble(origins)
    summary = summarise(pairs)
    per_origin = per_origin_table(pairs)
    for sid, before in protected.items():
        if _hash_dir(STORE / sid) != before:
            raise SystemExit(f"{sid} changed during this run; it must not be touched.")
    print(f"\nExisting caches unchanged (hash-checked): {', '.join(PROTECTED_SPEC_IDS)}")

    report(summary, per_origin)
    RESULTS_DIR.mkdir(exist_ok=True)
    summary_path = RESULTS_DIR / "v52_weekly_extended__summary.csv"
    origin_path = RESULTS_DIR / "v52_weekly_extended__by_origin.csv"
    summary.to_csv(summary_path, index=False, float_format="%.4f")
    per_origin.to_csv(origin_path, index=False, float_format="%.4f", date_format="%Y-%m-%d")
    print(f"\nWrote {summary_path}\nWrote {origin_path}")


def _hash_dir(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return {
        str(f.relative_to(path)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(path.rglob("*"))
        if f.is_file()
    }


if __name__ == "__main__":
    main()
