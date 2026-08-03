r"""Multi-origin score collection for the CFM calibration study.

Runs the v2.2 event scorer over many historical origin dates and records, for
each run, the fixed-width feature row alongside what the price actually did
afterwards. The result is a regression-ready table that answers the question
nothing in this project has answered yet: **do the agent's scores predict
anything?**

The script is deliberately boring. It calls the scorer, looks up outcomes, and
appends rows. It fits no model and draws no conclusion — analysis is a separate
step, so that collecting data and interpreting it stay independent.

Cost
----
Every origin is a live LLM run with web search, so a 40-origin sweep is 40
agent calls (more when validation retries fire). Start with ``--limit 3`` to
confirm the wiring before committing to a full sweep.

Resumability
------------
Rows are flushed after every run and the script skips origin/sample pairs
already present in the output. A crashed or cancelled sweep can be restarted
with the same command and will continue where it stopped.

Cutoff discipline
-----------------
Two different views of the price series are used, and keeping them apart is the
whole point:

- The **base price** at an origin comes from the cutoff-limited series, so it is
  what was actually knowable when the forecast was made.
- The **outcome price** comes from the full series, which is legitimate because
  outcomes are only ever read after the fact, never passed to the agent.

Examples
--------
Cheap wiring check, three origins::

    uv run python implementations/energy_oil_forecasting/cfm_calibration_study.py --limit 3

Full sweep, weekly origins over three years::

    uv run python implementations/energy_oil_forecasting/cfm_calibration_study.py \\
        --start 2023-01-01 --end 2026-02-01 --stride 5

Two samples per origin, to measure run-to-run variance::

    uv run python implementations/energy_oil_forecasting/cfm_calibration_study.py --samples 2
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from energy_oil_forecasting.cfm_agent_v_2_2 import CfmEventScorer, build_cfm_agent_v22_config
from energy_oil_forecasting.data import WTI_SERIES_ID, build_wti_service


logger = logging.getLogger("cfm_calibration_study")

# Outcomes are read after the fact, so a far-future cutoff is the honest way to
# ask the service for everything it has.
_ALL_DATA = datetime(2100, 1, 1)

DEFAULT_HORIZONS = (5, 10, 21)


def find_repository_root(start: Path) -> Path:
    """Locate the agentic-forecasting repository root."""
    candidate = start.resolve()
    while True:
        if (candidate / "implementations" / "energy_oil_forecasting").is_dir() and (
            candidate / "aieng-forecasting" / "pyproject.toml"
        ).is_file():
            return candidate
        if candidate.parent == candidate:
            raise FileNotFoundError("Could not locate the repository root.")
        candidate = candidate.parent


def build_origins(
    prices: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    stride: int,
    max_horizon: int,
) -> list[pd.Timestamp]:
    """Return strided business-day origins that have enough forward data.

    An origin is only usable when the series extends far enough past it to
    observe the longest horizon; otherwise the run would cost a live agent call
    and produce no target to regress against.
    """
    dates = pd.DatetimeIndex(sorted(prices["timestamp"].unique()))
    usable = dates[(dates >= start) & (dates <= end)]
    if len(dates) == 0:
        return []

    last_available = dates[-1]
    origins: list[pd.Timestamp] = []
    for date in usable[::stride]:
        forward = dates[dates > date]
        if len(forward) < max_horizon:
            continue
        if forward[max_horizon - 1] > last_available:
            continue
        origins.append(date)
    return origins


def outcome_returns(
    prices: pd.DataFrame,
    *,
    origin: pd.Timestamp,
    base_price: float,
    horizons: tuple[int, ...],
) -> dict[str, float]:
    """Return realized log returns from ``base_price`` at each horizon.

    The log return is the target the calibration regression predicts. Against a
    naive baseline the expected return is zero, so the realized return *is* the
    residual — the simplest honest first question. Swapping in a quant baseline
    later means subtracting that model's expected return from these values.
    """
    dates = pd.DatetimeIndex(sorted(prices["timestamp"].unique()))
    forward = dates[dates > origin]
    by_date = prices.drop_duplicates("timestamp").set_index("timestamp")["value"]

    outcomes: dict[str, float] = {}
    for horizon in horizons:
        if len(forward) < horizon:
            outcomes[f"actual_return_h{horizon}"] = float("nan")
            outcomes[f"actual_price_h{horizon}"] = float("nan")
            continue
        target_date = forward[horizon - 1]
        target_price = float(by_date.loc[target_date])
        outcomes[f"actual_price_h{horizon}"] = target_price
        outcomes[f"actual_return_h{horizon}"] = (
            float(np.log(target_price / base_price)) if base_price > 0 else float("nan")
        )
    return outcomes


def load_completed(path: Path) -> set[tuple[str, int]]:
    """Return the (origin, sample) pairs already recorded, for resumability."""
    if not path.exists():
        return set()
    done: set[tuple[str, int]] = set()
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            text = raw.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                logger.warning("Skipping unparsable line in %s", path)
                continue
            if record.get("status") == "ok":
                done.add((record["as_of"], int(record.get("sample", 0))))
    return done


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Rewrite the flat table, unioning keys so every row has every column.

    Feature keys are stable by construction, but a failed run contributes no
    features at all; unioning keeps the table rectangular either way.
    """
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Return every recorded JSONL row, flattened for the CSV table."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            text = raw.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                continue
            row: dict[str, Any] = {
                "as_of": record.get("as_of"),
                "sample": record.get("sample"),
                "status": record.get("status"),
                "attempts": record.get("attempts"),
                "base_price": record.get("base_price"),
                "langfuse_trace_id": record.get("langfuse_trace_id"),
                "error": record.get("error", ""),
            }
            row.update(record.get("features") or {})
            row.update(record.get("outcomes") or {})
            rows.append(row)
    return rows


def score_one_origin(
    scorer: CfmEventScorer,
    *,
    service: Any,
    prices: pd.DataFrame,
    origin: pd.Timestamp,
    sample: int,
    horizons: tuple[int, ...],
    prefix: str,
) -> dict[str, Any] | None:
    """Score one origin and return its record, or ``None`` when unusable.

    A failed agent run is recorded rather than raised: one bad origin must not
    end a sweep that has already spent real money on the origins before it.
    """
    as_of_date = origin.to_pydatetime()
    max_horizon = max(horizons)

    cutoff_history = service.get_series(WTI_SERIES_ID, as_of=as_of_date)
    if cutoff_history.empty:
        print(f"{prefix}: skipped, no history at cutoff")
        return None

    base_price = float(cutoff_history.iloc[-1]["value"])
    outcomes = outcome_returns(prices, origin=origin, base_price=base_price, horizons=horizons)
    record: dict[str, Any] = {
        "as_of": str(origin.date()),
        "sample": sample,
        "base_price": base_price,
        "outcomes": outcomes,
    }

    try:
        result = scorer.score(service.context(as_of_date))
    except Exception as error:  # noqa: BLE001 — one bad origin must not end the sweep.
        record.update({"status": "error", "error": f"{type(error).__name__}: {error}"})
        print(f"{prefix}: FAILED — {type(error).__name__}: {error}")
        return record

    features = result.output.calibration_features()
    record.update(
        {
            "status": "ok",
            "attempts": result.attempts,
            "langfuse_trace_id": result.langfuse_trace_id,
            "features": features,
            "score_card": result.output.model_dump(mode="json"),
        }
    )
    print(
        f"{prefix}: ok  attempts={result.attempts}  "
        f"geo={features['score_geopolitical']:+.0f}  "
        f"expected={features['expected_scenario_impact']:+.2f}  "
        f"actual_h{max_horizon}={outcomes[f'actual_return_h{max_horizon}']:+.4f}"
    )
    return record


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect CFM event scores across many origins for calibration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start", default="2023-01-01", help="First candidate origin date.")
    parser.add_argument("--end", default="2026-02-01", help="Last candidate origin date.")
    parser.add_argument("--stride", type=int, default=5, help="Business days between origins.")
    parser.add_argument("--samples", type=int, default=1, help="Scorer runs per origin.")
    parser.add_argument("--limit", type=int, default=0, help="Cap the origin count; 0 means no cap.")
    parser.add_argument(
        "--horizons",
        default=",".join(str(h) for h in DEFAULT_HORIZONS),
        help="Comma-separated business-day horizons for outcome returns.",
    )
    parser.add_argument("--model", default="gemini-3.1-flash-lite-preview", help="Agent model.")
    parser.add_argument(
        "--out",
        default="data/calibration",
        help="Output directory, relative to the repository root unless absolute.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-run origins already present in the output instead of skipping them.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    repository_root = find_repository_root(Path.cwd())
    load_dotenv(repository_root / ".env", override=False)

    horizons = tuple(int(part) for part in args.horizons.split(",") if part.strip())
    max_horizon = max(horizons)

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = repository_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "score_runs.jsonl"
    csv_path = out_dir / "calibration_table.csv"

    service = build_wti_service(cache_dir=repository_root / "data" / "yfinance")
    prices = service.get_series(WTI_SERIES_ID, as_of=_ALL_DATA).copy()
    prices["timestamp"] = pd.to_datetime(prices["timestamp"])
    prices = prices.sort_values("timestamp").reset_index(drop=True)

    origins = build_origins(
        prices,
        start=pd.Timestamp(args.start),
        end=pd.Timestamp(args.end),
        stride=args.stride,
        max_horizon=max_horizon,
    )
    if args.limit > 0:
        origins = origins[: args.limit]

    completed = set() if args.no_resume else load_completed(jsonl_path)
    planned = [
        (origin, sample)
        for origin in origins
        for sample in range(args.samples)
        if (str(origin.date()), sample) not in completed
    ]

    print(f"Origins in range:      {len(origins)}")
    print(f"Samples per origin:    {args.samples}")
    print(f"Already recorded:      {len(completed)}")
    print(f"Runs to execute:       {len(planned)}")
    print(f"Output:                {out_dir}")
    if not planned:
        print("\nNothing to do. Use --no-resume to re-run recorded origins.")
        return 0
    print("\nEach run is a live agent call with web search. Ctrl-C is safe; progress is saved.\n")

    scorer = CfmEventScorer(build_cfm_agent_v22_config(model=args.model))

    succeeded = 0
    failed = 0
    for index, (origin, sample) in enumerate(planned, start=1):
        prefix = f"[{index}/{len(planned)}] {origin.date()} sample {sample}"
        record = score_one_origin(
            scorer,
            service=service,
            prices=prices,
            origin=origin,
            sample=sample,
            horizons=horizons,
            prefix=prefix,
        )
        if record is None:
            continue
        if record["status"] == "ok":
            succeeded += 1
        else:
            failed += 1

        with jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()

    write_csv(load_rows(jsonl_path), csv_path)

    print(f"\nSucceeded: {succeeded}   Failed: {failed}")
    print(f"Full records: {jsonl_path}")
    print(f"Flat table:   {csv_path}")
    print(
        "\nNext: regress the feature columns against actual_return_h*. "
        "Against a naive baseline the realized return is the residual; to use a "
        "quant baseline instead, subtract its expected return from those columns first."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
