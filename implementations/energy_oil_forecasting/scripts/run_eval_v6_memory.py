"""Score CFM v6.0 and the scenario-memory agent on the 2026 out-of-sample window.

The window is ``specs/energy_oil_eval.yaml``: 18 weekly origins from
2026-02-02 to 2026-06-01, horizons 5, 10 and 21 business days — the same
protocol behind the NB04 "Results: On CRPS" slide. The new agents run through
``cached_multi_backtest`` exactly like every other predictor, so their mean
CRPS sits on the same scale as the cached baselines printed alongside them.

    # free: re-score whatever is already cached, never call an LLM
    uv run python implementations/energy_oil_forecasting/scripts/run_eval_v6_memory.py --score-only

    # paid: run the agents (needs OPENAI_BASE_URL / OPENAI_API_KEY, and
    # E2B_API_KEY for v6.0's code execution); completed tasks are cached
    uv run python implementations/energy_oil_forecasting/scripts/run_eval_v6_memory.py
    uv run python implementations/energy_oil_forecasting/scripts/run_eval_v6_memory.py --advanced

Working directory does not matter: every path is derived from this file.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd
import yaml


warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve()
PACKAGE_DIR = _HERE.parents[1]  # implementations/energy_oil_forecasting
sys.path.insert(0, str(PACKAGE_DIR.parent))  # <repo>/implementations

from aieng.forecasting.evaluation import MultiTargetBacktestSpec, cached_multi_backtest  # noqa: E402
from aieng.forecasting.evaluation.artifacts import load_multi_backtest_results  # noqa: E402
from aieng.forecasting.methods import LastValuePredictor  # noqa: E402
from aieng.forecasting.methods.numerical.darts_arima import DartsAutoARIMAPredictor  # noqa: E402
from aieng.forecasting.models import ADVANCED_MODEL, LITE_MODEL  # noqa: E402
from energy_oil_forecasting.cfm_agent_v_6_0 import build_cfm_agent_config, build_cfm_agent_predictor  # noqa: E402
from energy_oil_forecasting.data import build_wti_multivariate_service  # noqa: E402
from energy_oil_forecasting.wti_scenario_memory_agent import build_agent, memory  # noqa: E402


SPEC_PATH = PACKAGE_DIR / "specs" / "energy_oil_eval.yaml"
STORE = PACKAGE_DIR / "data" / "predictions"
# Free numerical baselines recomputed on every resolved origin. Kept apart from
# STORE so the slide's cached files are never overwritten.
FULL_STORE = PACKAGE_DIR / "data" / "predictions_eval_full"
# The memory agent's store is module-global and keyed only by predictor_id, so
# a run on this window would otherwise read scenarios written by any earlier
# experiment with the same id. Give this window its own memory.
MEMORY_DIR = PACKAGE_DIR / "data" / "scenario_memory" / "energy_oil_eval"

# The slide's agent rows, loaded by id and never recomputed (they cover all 18
# origins). The slide's naive and AutoARIMA caches were written on 2026-05-22
# and hold only the 22 forecasts resolved by then (origins 2026-02-02 to
# 2026-03-23); they define the "slide window" column, and the baselines are
# recomputed on every origin into FULL_STORE.
CACHED_AGENTS = {
    "Context agent, lite": f"agent_predictor_wti_analyst_news_{LITE_MODEL}_continuous",
    "Context agent, advanced": f"agent_predictor_wti_analyst_news_{ADVANCED_MODEL}_continuous",
}
SLIDE_REFERENCE = "last_value_naive"


def _v6(data_service, model):
    return build_cfm_agent_predictor(build_cfm_agent_config(data_service=data_service, model=model))


def _memory_agent(model):
    memory.MEMORY_DIR = MEMORY_DIR
    return build_agent(model)


def _scored(results) -> dict[tuple[str, str], float]:
    """Map (as_of, forecast_date) to the CRPS the harness recorded for it."""
    out = {}
    for result in results.values():
        for pred, score in zip(result.predictions, result.scores, strict=True):
            out[(str(pred.as_of)[:10], str(pred.forecast_date)[:10])] = score
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--score-only", action="store_true", help="load cached results only; never call an LLM")
    parser.add_argument("--advanced", action="store_true", help=f"run the new agents on {ADVANCED_MODEL} too")
    args = parser.parse_args()

    data_service = build_wti_multivariate_service(cache_dir=PACKAGE_DIR / "data" / "yfinance")
    spec = MultiTargetBacktestSpec.model_validate(yaml.safe_load(SPEC_PATH.read_text()))

    scored: dict[str, dict[tuple[str, str], float]] = {}
    for label, predictor in (("Naive last value", LastValuePredictor()), ("AutoARIMA", DartsAutoARIMAPredictor())):
        scored[label] = _scored(cached_multi_backtest(predictor, spec, data_service, store_dir=FULL_STORE))
    for label, predictor_id in CACHED_AGENTS.items():
        results = load_multi_backtest_results(spec, predictor_id, store_dir=STORE)
        if results:
            scored[label] = _scored(results)
        else:
            print(f"  {label}: no cached results for {predictor_id}")

    for model in [LITE_MODEL, ADVANCED_MODEL] if args.advanced else [LITE_MODEL]:
        tag = "lite" if model == LITE_MODEL else "advanced"
        for label, factory in (
            (f"CFM v6.0, {tag}", lambda m=model: _v6(data_service, m)),
            (f"Scenario memory agent, {tag}", lambda m=model: _memory_agent(m)),
        ):
            predictor = factory()
            if args.score_only:
                results = load_multi_backtest_results(spec, predictor.predictor_id, store_dir=STORE)
            else:
                print(f"Running {label} ({predictor.predictor_id}) ...", flush=True)
                results = cached_multi_backtest(predictor, spec, data_service, store_dir=STORE)
            if results:
                scored[label] = _scored(results)
            else:
                print(f"  {label}: no cached results for {predictor.predictor_id}")

    slide_keys = set(_scored(load_multi_backtest_results(spec, SLIDE_REFERENCE, store_dir=STORE) or {}))
    rows = []
    for label, by_key in scored.items():
        on_slide = [v for k, v in by_key.items() if k in slide_keys]
        rows.append(
            {
                "Predictor": label,
                "CRPS, slide window": sum(on_slide) / len(on_slide) if on_slide else float("nan"),
                "n": len(on_slide),
                "CRPS, all resolved": sum(by_key.values()) / len(by_key),
                "n all": len(by_key),
                "Origins": len({k[0] for k in by_key}),
            }
        )

    board = pd.DataFrame(rows).set_index("Predictor").sort_values("CRPS, slide window")
    print(f"\nOut of sample {spec.start:%Y-%m-%d} to {spec.end:%Y-%m-%d}, horizons {spec.tasks[0].horizons}.")
    print(f"Slide window = the {len(slide_keys)} forecasts the slide's cached baselines cover.")
    print(board.round(2).to_string())


if __name__ == "__main__":
    main()
