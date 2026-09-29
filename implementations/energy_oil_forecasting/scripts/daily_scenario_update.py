"""Run scenario_schema_anchored_enhanced live, once, and log the result.

Not a backtest. Each invocation is a single forecast at ``as_of=now()``,
using whatever price history and live news are available right now — meant
to be invoked once per day (cron, a scheduled CI job, a Claude Code Remote
Routine, whatever the runner is) so the model's estimate stays current.

Because ``scenario_schema_anchored_enhanced``'s own memory read filters on
``as_of < today`` (see ``scenario_schema_anchored_enhanced/memory.py``),
running this more than once on the same calendar day is safe — the second
run still only sees memory from strictly earlier days — but wasteful, since
each run is a real LLM call with real token cost. Guard against that
yourself in whatever scheduler you use (skip if today's log entry already
exists) if that matters to you; this script does not check.

Every run appends one record to a running JSONL log at
``data/live_forecasts/<predictor_id>/<task_id>.jsonl`` (gitignored, same
convention as ``data/predictions/`` and ``data/scenario_memory/``), holding
the full forecast (point_forecast/quantiles per horizon) plus the
scenarios/factors metadata, so the log itself can be charted or reviewed
later without needing to rerun anything.

Usage::

    uv run python scripts/daily_scenario_update.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import energy_oil_forecasting
import numpy as np
import yaml
from aieng.forecasting.evaluation import MultiTargetBacktestSpec
from energy_oil_forecasting.data import build_wti_service
from energy_oil_forecasting.paths import DATA_DIR
from energy_oil_forecasting.scenario_schema_anchored_enhanced import (
    build_wti_news_scenario_schema_anchored_enhanced_config,
    build_wti_scenario_schema_anchored_enhanced_predictor,
)

LIVE_FORECAST_DIR = DATA_DIR / "live_forecasts"

MAX_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 10.0


def _load_task():
    """Load the WTI forecasting task from the project's own backtest spec.

    Reuses ``specs/energy_oil_backtest.yaml`` as the single source of truth
    for ``task_id``, ``target_series_id``, ``horizons``, and ``frequency``,
    the same way NB04c/NB04d load their tasks — rather than hand-duplicating
    those fields here and risking drift.
    """
    spec_dir = Path(energy_oil_forecasting.__file__).parent / "specs"
    with open(spec_dir / "energy_oil_backtest.yaml") as f:
        spec = MultiTargetBacktestSpec.model_validate(yaml.safe_load(f))
    return spec.tasks[0]


def _log_path(predictor_id: str, task_id: str) -> Path:
    directory = LIVE_FORECAST_DIR / predictor_id.replace("/", "_")
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{task_id}.jsonl"


def main() -> None:
    task = _load_task()
    data_service = build_wti_service()
    context = data_service.context(datetime.now())

    config = build_wti_news_scenario_schema_anchored_enhanced_config()
    predictor = build_wti_scenario_schema_anchored_enhanced_predictor(config)

    print(f"Running {predictor.predictor_id} live, as_of={str(context.as_of)[:10]}")

    last_error: Exception | None = None
    predictions = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            predictions = predictor.predict(task, context)
            break
        except Exception as exc:  # noqa: BLE001 — daily job: log and retry, don't crash the scheduler on one bad call
            last_error = exc
            print(f"  attempt {attempt}/{MAX_ATTEMPTS} failed: {exc}", file=sys.stderr)
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_DELAY_SECONDS)

    if predictions is None:
        print(f"FAILED after {MAX_ATTEMPTS} attempts: {last_error}", file=sys.stderr)
        sys.exit(1)

    log_path = _log_path(predictor.predictor_id, task.task_id)
    issued_at = datetime.now().isoformat()

    print("━" * 72)
    print(f"{predictor.predictor_id}  as_of={str(context.as_of)[:10]}")
    print("━" * 72)
    for pred in predictions:
        payload = pred.payload
        q = {float(k): float(v) for k, v in payload.quantiles.items()}
        horizon = int(np.busday_count(context.as_of.date(), pred.forecast_date.date()))
        print(
            f"  h={horizon:>2}d  {pred.forecast_date.date()}  point=${payload.point_forecast:6.2f}"
            f"  p10=${q.get(0.1, float('nan')):6.2f}  p90=${q.get(0.9, float('nan')):6.2f}"
        )
        record = {
            "issued_at": issued_at,
            "as_of": str(context.as_of)[:10],
            "forecast_date": str(pred.forecast_date.date()),
            "horizon": horizon,
            "point_forecast": payload.point_forecast,
            "quantiles": q,
            "metadata": pred.metadata,
        }
        with log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    scenarios = predictions[0].metadata.get("scenarios", [])
    if scenarios:
        print()
        print("  Scenarios:")
        for scenario in scenarios:
            print(
                f"    {scenario['name']:28} p={scenario['probability']:.2f}"
                f"  [{scenario['price_low']:.2f}, {scenario['price_high']:.2f}]"
                f"{'  (tail)' if scenario.get('is_tail_case') else ''}"
            )

    print()
    print(f"Appended {len(predictions)} record(s) to {log_path}")


if __name__ == "__main__":
    main()
