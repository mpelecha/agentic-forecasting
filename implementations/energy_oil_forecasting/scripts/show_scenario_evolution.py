"""Print how scenario_schema_anchored_enhanced's scenarios evolved across a cached backtest.

Read-only: loads an already-computed ``BacktestResult`` from
``data/predictions/`` and prints one line per origin showing the longest-
horizon point forecast and, for every scenario, whether it continued a prior
one (and which) or was genuinely new. No LLM calls, no network — safe and
free to run repeatedly.

Usage::

    uv run python scripts/show_scenario_evolution.py
    uv run python scripts/show_scenario_evolution.py --spec-id energy_oil_eval_10yr_quarterly
    uv run python scripts/show_scenario_evolution.py --predictor-id agent_predictor_wti_analyst_news_scenario_schema_anchored_enhanced_logret_gemini-3.1-flash-lite-preview_continuous
"""

from __future__ import annotations

import argparse

from aieng.forecasting.evaluation.artifacts import load_backtest_result

DEFAULT_SPEC_ID = "energy_oil_backtest_10yr_quarterly"
DEFAULT_PREDICTOR_ID = (
    "agent_predictor_wti_analyst_news_scenario_schema_anchored_enhanced_gemini-3.1-flash-lite-preview_continuous"
)
DEFAULT_TASK_ID = "wti_oil_price_forecast"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-id", default=DEFAULT_SPEC_ID)
    parser.add_argument("--predictor-id", default=DEFAULT_PREDICTOR_ID)
    parser.add_argument("--task-id", default=DEFAULT_TASK_ID)
    args = parser.parse_args()

    predictor_id = f"{args.predictor_id}__{args.task_id}"
    result = load_backtest_result(args.spec_id, predictor_id)
    if result is None:
        print(f"No cached result at data/predictions/{args.spec_id}/{predictor_id}.yaml")
        return

    # One entry per origin (metadata is duplicated across every horizon's
    # Prediction for that origin, same as the smoke test).
    by_origin: dict = {}
    for pred in result.predictions:
        current = by_origin.get(pred.as_of)
        if current is None or pred.forecast_date > current.forecast_date:
            by_origin[pred.as_of] = pred

    print(f"{args.spec_id} / {args.predictor_id}")
    print(f"{len(by_origin)} origins, {result.spec.start} -> {result.spec.end}")
    print()

    seen_names: set[str] = set()
    total_unverified = 0
    for as_of in sorted(by_origin):
        pred = by_origin[as_of]
        scenarios = pred.metadata.get("scenarios", [])
        unverified = pred.metadata.get("unverified_continuations")
        line = f"{str(as_of)[:10]}  point=${pred.payload.point_forecast:7.2f}  "
        parts = []
        for scenario in scenarios:
            if scenario.get("continues_prior_scenario"):
                tag = "=" if scenario["continued_from"] in seen_names else "?"
            else:
                tag = "+"
            parts.append(f"{tag}{scenario['name']}")
            seen_names.add(scenario["name"])
        line += "  ".join(parts)
        if unverified:
            line += f"   ⚠ unverified: {unverified}"
            total_unverified += len(unverified)
        print(line)

    print()
    print("Legend: + new scenario   = verified continuation   ? claimed continuation, name not previously seen")
    print()
    print(f"{len(seen_names)} distinct scenario names appeared across {len(by_origin)} origins.")
    print(f"{total_unverified} unverified continuation claim(s) total.")


if __name__ == "__main__":
    main()
