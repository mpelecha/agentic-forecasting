"""Re-run the ensemble-only history under a settings patch, into a sibling directory.

Bucket-A levers -- ``lightgbm_lags``, ``kalman_dim_x``, ``ensemble_weights`` as the
suite would actually apply them (the RNG seed includes the weights) -- cannot be
priced by replaying stored quantiles. They need the numerical suite re-run at
every origin, which is hours of CPU. This CLI does that the way `history.py`
does, with one difference: the settings the suite runs under come from
``--set field=value`` patches. Output lands in ``history_v52_ensemble__<variant>/``
and is priced like omega afterwards.

Code-level changes (LightGBM on returns, a random-walk member) are not settings.
They need a package copy: pass ``--package`` with the importable module of a
drafted challenger (``review/challenger.py`` writes one) and the suite is taken
from there. Never run from the weekly job.

Usage (from the repo root)::

    uv run python -m energy_oil_forecasting.cfm_coach.review.history_variant --variant lags42 --set lightgbm_lags=42 --limit 8
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from energy_oil_forecasting.cfm_coach.history import (
    DEFAULT_END,
    DEFAULT_START,
    HISTORY_DIR,
    HISTORY_SCHEMA_VERSION,
    HORIZONS,
    TARGET_SERIES_ID,
    _quantiles_json,
    _service,
    freeze_price_cache,
    origins,
    path_for,
)


def parse_patch(items: list[str]) -> dict[str, Any]:
    patch: dict[str, Any] = {}
    for item in items:
        name, _, raw = item.partition("=")
        if not name or not raw:
            raise SystemExit(f"--set expects field=value, got {item!r}")
        try:
            patch[name] = json.loads(raw)
        except json.JSONDecodeError:
            patch[name] = raw
    return patch


def run_origin_variant(
    cutoff: str, cache_dir: str, out_dir: str, patch: dict[str, Any], package: str
) -> dict[str, Any]:
    """`history.run_origin` with the settings patched and the suite taken from `package`."""
    config = importlib.import_module(f"{package}.config")
    market_data = importlib.import_module(f"{package}.tools.market_data")
    from energy_oil_forecasting.data import DEFAULT_WTI_COVARIATE_SERIES_IDS  # noqa: PLC0415

    service = _service(cache_dir)
    settings = config.CfmV52Settings(**patch)
    covariates = [name for name in DEFAULT_WTI_COVARIATE_SERIES_IDS if name in service.series_ids]
    payload: dict[str, Any] = {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "cutoff": cutoff,
        "provenance": "ensemble_only",
        "agent_id": getattr(config, "AGENT_NAME", package),
        "package": package,
        "settings_patch": patch,
        "settings": settings.model_dump(mode="json"),
        "covariate_series_ids": covariates,
        "horizons": list(HORIZONS),
        "written_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
    }
    started = time.monotonic()
    try:
        tool = market_data.AuthoritativeSuiteTool(service, settings=settings, covariate_series_ids=covariates)
        suite = tool._run_model_suite(  # noqa: SLF001 - the exact method a live run calls
            context=service.context(as_of=datetime.strptime(cutoff, "%Y-%m-%d")),
            target_series_id=TARGET_SERIES_ID,
            horizons=list(HORIZONS),
            frequency="B",
            cutoff_date=cutoff,
        )
    except Exception as exc:  # noqa: BLE001 - recorded, so the run continues
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
            "model_disagreement_std": {str(k): v for k, v in suite.model_disagreement_std.items()},
            "diagnostics": suite.diagnostics.model_dump(mode="json"),
            "forecasts": horizons,
        }
    payload["elapsed_seconds"] = round(time.monotonic() - started, 2)
    target = path_for(Path(out_dir), cutoff)
    scratch = target.with_suffix(".json.tmp")
    scratch.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    scratch.replace(target)
    return {key: payload.get(key) for key in ("cutoff", "status", "elapsed_seconds", "error")}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--variant", required=True, help="suffix for history_v52_ensemble__<variant>/")
    parser.add_argument("--set", action="append", default=[], help="field=value (JSON value); repeatable")
    parser.add_argument("--package", default="energy_oil_forecasting.cfm_agent_v_5_2")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    patch = parse_patch(args.set)
    out_dir = HISTORY_DIR.parent / f"{HISTORY_DIR.name}__{args.variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = freeze_price_cache(HISTORY_DIR)  # share the base history's frozen prices
    pending = [c for c in origins(args.start, args.end) if not path_for(out_dir, c).exists()]
    if args.limit is not None:
        step = max(1, len(pending) // args.limit)
        pending = pending[::step][: args.limit]
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "variant": args.variant,
                "package": args.package,
                "settings_patch": patch,
                "start": args.start,
                "end": args.end,
                "pending": len(pending),
                "updated_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"{args.variant}: {len(pending)} origin(s) pending -> {out_dir}")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_origin_variant, c, str(cache_dir), str(out_dir), patch, args.package): c for c in pending
        }
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            print(
                f"[{done}/{len(pending)}] {result['cutoff']} {result['status']} {result['elapsed_seconds']}s {result.get('error') or ''}"
            )


if __name__ == "__main__":
    main()


__all__ = ["parse_patch", "run_origin_variant"]
