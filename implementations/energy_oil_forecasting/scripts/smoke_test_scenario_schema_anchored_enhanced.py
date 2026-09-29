"""Smoke test: scenario_schema_anchored_enhanced across a real sequential backtest.

Not a performance backtest (three origins says nothing about CRPS/coverage)
and not another live single-call test (already done manually). This exists
to answer the one question neither of those covers before NB04b spends real
token budget on this predictor at scale: does the *actual* backtest harness
path work correctly across multiple sequential origins — memory read/write
in the right order, the anchoring arithmetic, the continuation fields — and
roughly how long does one origin take, so the cost of a full NB04b run can
be estimated.

Runs 3 weekly origins in August 2026 (recent enough to resolve cleanly
against current data, distinctly in the past so this is a genuine historical
replay rather than "live now" repeated) through ``cached_multi_backtest`` —
the same harness function NB04b itself uses — via a spec_id/predictor name
distinct from the live daily-run one, so this smoke test's memory file never
touches or reads from your `daily_scenario_update.py` history.

Usage::

    uv run python scripts/smoke_test_scenario_schema_anchored_enhanced.py
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import energy_oil_forecasting
import yaml
from aieng.forecasting.evaluation import MultiTargetBacktestSpec, cached_multi_backtest
from energy_oil_forecasting.data import build_wti_service
from energy_oil_forecasting.scenario_schema_anchored_enhanced import (
    build_wti_news_scenario_schema_anchored_enhanced_config,
    build_wti_scenario_schema_anchored_enhanced_predictor,
)

SPEC_ID = "energy_oil_smoketest_scenario_schema_anchored_enhanced"
HORIZONS = [5, 10, 21]

# Three weekly origins, distinctly in the past (resolves cleanly against
# current data even at h=21) but recent enough that the LLM's live search
# still returns real, dated news rather than reaching for training-era
# background knowledge.
START = "2026-08-07"
END = "2026-08-21"
STRIDE = 5  # business days = weekly


def _load_task():
    spec_dir = Path(energy_oil_forecasting.__file__).parent / "specs"
    with open(spec_dir / "energy_oil_backtest.yaml") as f:
        spec = MultiTargetBacktestSpec.model_validate(yaml.safe_load(f))
    return spec.tasks[0]


def main() -> None:
    task = _load_task()
    task.horizons = list(HORIZONS)
    data_service = build_wti_service()

    spec = MultiTargetBacktestSpec(
        spec_id=SPEC_ID,
        tasks=[task],
        start=START,
        end=END,
        stride=STRIDE,
        warmup=250,
        description="Smoke test for scenario_schema_anchored_enhanced -- mechanism, not performance.",
    )

    config = build_wti_news_scenario_schema_anchored_enhanced_config()
    # Distinct name so predictor_id -- and therefore the memory file and the
    # prediction cache -- never overlaps with the live daily-run predictor.
    # See ScenarioSchemaAnchoredEnhancedPredictor's memory docstring: memory
    # is scoped per predictor_id, exactly for this reason.
    config = config.model_copy(update={"name": config.name + "_smoketest"})
    predictor = build_wti_scenario_schema_anchored_enhanced_predictor(config)

    origins = spec.origins() if hasattr(spec, "origins") else None
    print(f"Predictor:  {predictor.predictor_id}")
    print(f"Window:     {START} -> {END}  (stride={STRIDE}, {len(origins) if origins else '?'} candidate origins)")
    print(f"Horizons:   {HORIZONS}")
    print()

    started = time.time()
    results = cached_multi_backtest(predictor, spec, data_service, max_retries=2, retry_delay=5.0)
    elapsed = time.time() - started

    result = results[task.task_id]
    n_origins = len({pred.as_of for pred in result.predictions})
    print(f"Ran in {elapsed:.1f}s total, ~{elapsed / max(n_origins, 1):.1f}s/origin "
          f"({n_origins} origin(s) scored, {result.skipped_origins} skipped)")
    print(f"Cached at data/predictions/{SPEC_ID}/{predictor.predictor_id}__{task.task_id}.yaml")
    print()

    # One entry per origin (metadata is duplicated across every horizon's
    # Prediction for that origin -- see ScenarioSchemaAnchoredPredictor.predict
    # -- so the longest-horizon entry per as_of is enough to report on).
    by_origin: dict = {}
    for pred in result.predictions:
        current = by_origin.get(pred.as_of)
        if current is None or pred.forecast_date > current.forecast_date:
            by_origin[pred.as_of] = pred

    seen_names: set[str] = set()
    for as_of in sorted(by_origin):
        pred = by_origin[as_of]
        print("━" * 72)
        print(f"as_of={str(as_of)[:10]}  point(h={HORIZONS[-1]})=${pred.payload.point_forecast:.2f}")
        print(f"  prior_frameworks_used: {pred.metadata.get('prior_frameworks_used')}")
        unverified = pred.metadata.get("unverified_continuations")
        if unverified:
            print(f"  ⚠ unverified_continuations: {unverified}")
        for scenario in pred.metadata.get("scenarios", []):
            flag = ""
            if scenario["continues_prior_scenario"]:
                flag = (
                    "  [continues: " + scenario["continued_from"] + "]"
                    if scenario["continued_from"] in seen_names
                    else "  [claims continuation, but name not seen before -- check unverified_continuations]"
                )
            else:
                flag = "  [new]"
            print(f"    {scenario['name']:32} p={scenario['probability']:.2f}{flag}")
            seen_names.add(scenario["name"])

    print()
    print("━" * 72)
    print("What to check before deciding on NB04b:")
    print("  1. Did every origin score without an error/retry above?")
    print("  2. Do origins after the first show [continues: ...] rather than all [new]?")
    print("  3. Any ⚠ unverified_continuations lines -- and if so, is that a real hallucination")
    print("     or just the LLM paraphrasing a prior name instead of reusing it exactly?")
    print(f"  4. Extrapolate cost: ~{elapsed / max(n_origins, 1):.1f}s/origin x however many origins")
    print("     NB04b's actual grid would use.")


if __name__ == "__main__":
    main()
