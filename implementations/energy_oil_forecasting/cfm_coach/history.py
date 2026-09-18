"""Ensemble-only history: v5.2's numerical suite re-run at past cutoffs.

The width scale (omega) is the one calibration lever that does not need live runs to
fit. When the evidence policy grants the LLM nothing -- about half of all live
horizons -- the published forecast *is* the numerical ensemble, to the cent. And the
ensemble uses no LLM and no web search, so re-running it at a past cutoff cannot be
contaminated by hindsight the way re-running the agent is (R7 is about research
*selection*; there is no research here). That makes years of history admissible for
*choosing* a width, while live runs remain the only thing allowed to *judge* it.

**Identity with live runs.** Each origin calls v5.2's own
``AuthoritativeSuiteTool._run_model_suite`` -- the exact method a live run calls,
with the package's own settings, covariate panel, SHA-256-derived RNG seed and
ensemble weights. Nothing is reimplemented, so the history is the live suite
evaluated at an earlier date rather than a lookalike of it. (LightGBM's multithreaded
training is not bit-deterministic even live -- IMPROVEMENTS.md A2 -- so neither is
this; the effect is cents and irrelevant to a width fit.)

**Cutoff honesty.** The data service's context is scoped to each origin exactly as a
live run's is: every close is released the next business day.

**A frozen price cache.** Workers read a private copy of ``data/yfinance`` taken once
at the start. A parallel run must never write the shared cache the daily job reads,
and a fit should not change underneath itself because a parquet refreshed mid-run.

**Resumable.** One JSON per origin, written atomically. Re-running skips origins that
already have a file, so an interrupted overnight run continues where it stopped.

No LLM, no network. Usage (from the repo root)::

    uv run python -m energy_oil_forecasting.cfm_coach.history              # full default range
    uv run python -m energy_oil_forecasting.cfm_coach.history --limit=8    # timing sample
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from energy_oil_forecasting.cfm_coach.config import PACKAGE_ROOT


HISTORY_SCHEMA_VERSION = 1
HISTORY_DIR = PACKAGE_ROOT / "history_v52_ensemble"
PRICE_CACHE_DIRNAME = "_price_cache"
SOURCE_PRICE_CACHE = Path("data/yfinance")

#: First Monday after every covariate exists. USL, which the curve-shape covariate
#: needs, starts 2007-12-06; an origin before LightGBM has enough aligned history
#: fails that model, and a degraded ensemble is excluded from the fit rather than
#: silently pooled with complete ones.
DEFAULT_START = "2008-01-07"
#: Last Monday whose 21-business-day target had printed when the price cache was
#: frozen (data through 2026-09-11), and before the first live v5.2 cutoff
#: (2026-08-19) -- so every live origin is out-of-sample for anything fitted here.
DEFAULT_END = "2026-08-10"

HORIZONS = (5, 10, 21)
TARGET_SERIES_ID = "wti_crude_oil_price"

_SERVICE: Any = None


def origins(start: str = DEFAULT_START, end: str = DEFAULT_END) -> list[str]:
    """Weekly Monday cutoffs in ``[start, end]``, as ISO dates."""
    return [stamp.date().isoformat() for stamp in pd.date_range(start, end, freq="W-MON")]


def path_for(out_dir: Path, cutoff: str) -> Path:
    return out_dir / f"{cutoff}.json"


def _service(cache_dir: str) -> Any:
    """One data service per worker process, built from the frozen cache and never the network.

    ``YFinanceDailyAdapter._cache_covers_range`` rejects any cache that starts after the
    requested ``2004-01-01``. Brent (2007), USO (2006) and USL (2007) always do, so every
    service build re-downloads those tickers and rewrites their parquet -- the side effect
    HANDOFF.md relies on to keep the daily job's prices fresh. Here it would mean N
    workers fetching and rewriting the same files concurrently, and a history whose
    inputs changed mid-run. So, inside this process only, a non-empty cache is
    authoritative. The shared library file is not modified.
    """
    global _SERVICE  # noqa: PLW0603 - per-process memo; each spawned worker builds its own
    if _SERVICE is None:
        from aieng.forecasting.data.adapters.yfinance import YFinanceDailyAdapter  # noqa: PLC0415
        from energy_oil_forecasting.data import build_wti_multivariate_service  # noqa: PLC0415

        YFinanceDailyAdapter._cache_covers_range = lambda self, frame: not frame.empty  # noqa: ARG005, SLF001
        _SERVICE = build_wti_multivariate_service(cache_dir=Path(cache_dir))
    return _SERVICE


def _quantiles_json(quantiles: dict[float, float]) -> dict[str, float]:
    return {str(float(level)): float(value) for level, value in sorted(quantiles.items())}


def run_origin(cutoff: str, cache_dir: str, out_dir: str) -> dict[str, Any]:
    """Run v5.2's authoritative numerical suite at one cutoff and write the result."""
    from energy_oil_forecasting.cfm_agent_v_5_2.config import CfmV52Settings  # noqa: PLC0415
    from energy_oil_forecasting.cfm_agent_v_5_2.tools.market_data import AuthoritativeSuiteTool  # noqa: PLC0415
    from energy_oil_forecasting.cfm_coach.run_store import agent_package_fingerprint  # noqa: PLC0415
    from energy_oil_forecasting.cfm_coach.targets import V52  # noqa: PLC0415
    from energy_oil_forecasting.data import DEFAULT_WTI_COVARIATE_SERIES_IDS  # noqa: PLC0415

    service = _service(cache_dir)
    settings = CfmV52Settings()
    covariates = [name for name in DEFAULT_WTI_COVARIATE_SERIES_IDS if name in service.series_ids]
    payload: dict[str, Any] = {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "cutoff": cutoff,
        "provenance": "ensemble_only",
        "agent_id": V52.agent_id,
        "package_fingerprint": agent_package_fingerprint(V52),
        "settings": settings.model_dump(mode="json"),
        "covariate_series_ids": covariates,
        "horizons": list(HORIZONS),
        "written_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
    }

    started = time.monotonic()
    try:
        # A fresh tool per origin: the suite's predictors are stateful Darts wrappers,
        # and nothing fitted at one cutoff may carry into the next.
        tool = AuthoritativeSuiteTool(service, settings=settings, covariate_series_ids=covariates)
        suite = tool._run_model_suite(  # noqa: SLF001 - the exact method a live run calls; see module docstring
            context=service.context(as_of=datetime.strptime(cutoff, "%Y-%m-%d")),
            target_series_id=TARGET_SERIES_ID,
            horizons=list(HORIZONS),
            frequency="B",
            cutoff_date=cutoff,
        )
    except Exception as exc:  # noqa: BLE001 - recorded, so the run continues and the origin is visibly lost
        payload |= {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    else:
        horizons: dict[str, Any] = {}
        for index, horizon in enumerate(HORIZONS):
            ensemble = suite.ensemble.forecasts[index] if suite.ensemble else None
            horizons[str(horizon)] = {
                "forecast_date": ensemble.forecast_date if ensemble else None,
                "ensemble": _quantiles_json(ensemble.quantiles) if ensemble else None,
                "components": {
                    name: _quantiles_json(result.forecasts[index].quantiles)
                    for name, result in suite.models.items()
                    if result.status == "ok" and result.forecasts
                },
            }
        payload |= {
            "status": "ok" if suite.ensemble else "error",
            "rng_seed": suite.rng_seed,
            "successful_models": list(suite.successful_models),
            "failed_models": list(suite.failed_models),
            "model_disagreement_std": {str(key): value for key, value in suite.model_disagreement_std.items()},
            "diagnostics": suite.diagnostics.model_dump(mode="json"),
            "forecasts": horizons,
        }
    payload["elapsed_seconds"] = round(time.monotonic() - started, 2)

    target = path_for(Path(out_dir), cutoff)
    scratch = target.with_suffix(".json.tmp")
    scratch.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    scratch.replace(target)
    return {key: payload.get(key) for key in ("cutoff", "status", "failed_models", "elapsed_seconds", "error")}


def freeze_price_cache(out_dir: Path, source: Path = SOURCE_PRICE_CACHE) -> Path:
    """Copy the price cache once; later runs reuse the copy so the whole history shares one snapshot."""
    frozen = out_dir / PRICE_CACHE_DIRNAME
    if frozen.exists():
        return frozen
    if not source.is_dir():
        raise SystemExit(f"price cache {source} not found -- run from the repo root")
    shutil.copytree(source, frozen)
    return frozen


def load_history(out_dir: Path = HISTORY_DIR) -> list[dict[str, Any]]:
    """Every origin written so far, in cutoff order."""
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(out_dir.glob("????-??-??.json"))]


def main(argv: list[str] | None = None) -> None:  # noqa: PLR0915 - arguments, manifest and pool loop read top to bottom
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--out", type=Path, default=HISTORY_DIR)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--threads", type=int, default=2, help="OpenMP threads per worker (LightGBM)")
    parser.add_argument("--limit", type=int, default=None, help="run at most N pending origins, spread evenly")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument(
        "--deadline",
        default=None,
        help="local ISO time (e.g. 2026-09-14T07:00) after which queued origins are cancelled; re-run to resume",
    )
    args = parser.parse_args(argv)

    # Inherited by spawned workers, and read by LightGBM's OpenMP runtime when it loads.
    # Without it every worker takes every core and N workers run slower than one.
    os.environ["OMP_NUM_THREADS"] = str(args.threads)

    args.out.mkdir(parents=True, exist_ok=True)
    cache_dir = freeze_price_cache(args.out)
    # Build once in the parent so any cache touch-up happens here, serially, not in N workers at once.
    _service(str(cache_dir))

    wanted = origins(args.start, args.end)
    pending = []
    for cutoff in wanted:
        path = path_for(args.out, cutoff)
        if path.exists() and (
            not args.retry_errors or json.loads(path.read_text(encoding="utf-8")).get("status") == "ok"
        ):
            continue
        pending.append(cutoff)
    already_done = len(wanted) - len(pending)
    if args.limit is not None and len(pending) > args.limit:
        step = len(pending) / args.limit
        pending = [pending[int(i * step)] for i in range(args.limit)]
    else:
        # Alternate weeks first, then the weeks between. A run stopped by --deadline then
        # holds a fortnightly sample of the whole 2008-2026 span -- training and holdout
        # years alike -- rather than a complete early history and no recent years at all.
        pending = pending[::2] + pending[1::2]

    manifest = {
        "start": args.start,
        "end": args.end,
        "origins": len(wanted),
        "price_cache": str(cache_dir),
        "price_data_through": str(pd.read_parquet(cache_dir / "cl_f_adj_close_1d.parquet")["timestamp"].max().date()),
        "workers": args.workers,
        "threads_per_worker": args.threads,
        "updated_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"{len(wanted)} origin(s) in range, {already_done} done, {len(pending)} to run now", flush=True)
    print(f"workers={args.workers} threads/worker={args.threads} out={args.out}", flush=True)
    if not pending:
        return

    # A hard stop, because the weekday 08:00 job runs the same LightGBM suite 13 times under a
    # 15-minute watchdog: competing with this for CPU could time those runs out, and a missed
    # live day cannot be recovered. Checked as each origin finishes, so it can overrun by at
    # most one origin's runtime; queued work is cancelled and a re-run resumes it.
    deadline = datetime.fromisoformat(args.deadline) if args.deadline else None
    if deadline is not None:
        print(f"deadline {deadline:%Y-%m-%d %H:%M} local", flush=True)

    started = time.monotonic()
    finished = 0
    stopping = False
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_origin, cutoff, str(cache_dir), str(args.out)): cutoff for cutoff in pending}
        for future in as_completed(futures):
            if future.cancelled():
                continue
            finished += 1
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - a crashed worker loses one origin, not the run
                result = {"cutoff": futures[future], "status": "crashed", "error": repr(exc)}
            elapsed = time.monotonic() - started
            eta = elapsed / finished * (len(pending) - finished)
            note = f" failed={result['failed_models']}" if result.get("failed_models") else ""
            note += f" error={result['error']}" if result.get("error") else ""
            print(
                f"[{finished}/{len(pending)}] {result['cutoff']} {result['status']} "
                f"{result.get('elapsed_seconds')}s{note}  elapsed {elapsed / 60:.1f}m  eta {eta / 3600:.1f}h",
                flush=True,
            )
            if deadline is not None and not stopping and datetime.now() >= deadline:
                stopping = True
                cancelled = sum(1 for item in futures if item.cancel())
                print(
                    f"deadline reached: cancelled {cancelled} queued origin(s); running ones will finish. "
                    "Re-run the same command to resume.",
                    flush=True,
                )


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["DEFAULT_END", "DEFAULT_START", "HISTORY_DIR", "load_history", "origins", "run_origin"]
