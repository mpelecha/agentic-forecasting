"""Score CFM v6.0 and the scenario-memory agent on the 2026 out-of-sample window.

The window is ``specs/energy_oil_eval.yaml``: 18 weekly origins from
2026-02-02 to 2026-06-01, horizons 5, 10 and 21 business days, the NB04
protocol. Every predictor runs through ``cached_multi_backtest``, so all rows
share one scale.

The cache is keyed by spec_id alone and never checks the window. Until
2026-07-01 this spec had 8 origins (to 2026-03-23), and files written then
still load silently. This script compares each cached file's window with the
spec; a stale numerical file is recomputed (free), a stale LLM file is
reported and left alone.

    # free: numerical rows, plus whatever LLM/agent rows are already cached
    uv run python implementations/energy_oil_forecasting/scripts/run_eval_v6_memory.py --score-only

    # paid: also run CFM v6.0 and the memory agent (needs OPENAI_BASE_URL,
    # OPENAI_API_KEY, and E2B_API_KEY for v6.0's code execution)
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
from aieng.forecasting.methods.numerical.darts_regression import DartsLightGBMPredictor  # noqa: E402
from aieng.forecasting.models import ADVANCED_MODEL, LITE_MODEL  # noqa: E402
from energy_oil_forecasting.cfm_agent_v_6_0 import build_cfm_agent_config, build_cfm_agent_predictor  # noqa: E402
from energy_oil_forecasting.data import DEFAULT_WTI_COVARIATE_SERIES_IDS, build_wti_multivariate_service  # noqa: E402
from energy_oil_forecasting.prophet_baseline import ProphetPredictor  # noqa: E402
from energy_oil_forecasting.wti_scenario_memory_agent import build_agent, memory  # noqa: E402


SPEC_PATH = PACKAGE_DIR / "specs" / "energy_oil_eval.yaml"
STORE = PACKAGE_DIR / "data" / "predictions"
# The memory agent's store is module-global and keyed only by predictor_id, so
# a run on this window would otherwise read scenarios written by any earlier
# experiment with the same id. Give this window its own memory.
MEMORY_DIR = PACKAGE_DIR / "data" / "scenario_memory" / "energy_oil_eval"

# NB04's LightGBM settings, shared by both variants.
LAGS = 21
NUM_SAMPLES_LGBM = 200
LGBM_KWARGS = {"num_threads": 1, "n_jobs": 1, "verbosity": -1}

# NB04 rows that cost tokens: loaded from the cache if present, never run here.
CACHED_LLM_ROWS = {
    "Context agent, lite": f"agent_predictor_wti_analyst_news_{LITE_MODEL}_continuous",
    "Context agent, advanced": f"agent_predictor_wti_analyst_news_{ADVANCED_MODEL}_continuous",
    "LLM process, grid (lite)": f"llmp_quantile_grid[{LITE_MODEL}]",
    "LLM process, grid (advanced)": f"llmp_quantile_grid[{ADVANCED_MODEL}]",
}


def _numerical(covariates: list[str]) -> dict[str, object]:
    return {
        "Naive last value": LastValuePredictor(),
        "AutoARIMA": DartsAutoARIMAPredictor(),
        "Prophet": ProphetPredictor(),
        "LightGBM": DartsLightGBMPredictor(lags=LAGS, num_samples=NUM_SAMPLES_LGBM, lgbm_kwargs=LGBM_KWARGS),
        "LightGBM + covariates": DartsLightGBMPredictor(
            lags=LAGS,
            lags_past_covariates=LAGS,
            covariate_series_ids=covariates,
            num_samples=NUM_SAMPLES_LGBM,
            lgbm_kwargs=LGBM_KWARGS,
        ),
    }


def _v6(data_service, model):
    return build_cfm_agent_predictor(build_cfm_agent_config(data_service=data_service, model=model))


def _memory_agent(model):
    memory.MEMORY_DIR = MEMORY_DIR
    return build_agent(model)


def _is_stale(results, spec) -> bool:
    """Return True when a cached file was written for a different window of this spec_id."""
    return any((r.spec.start, r.spec.end) != (spec.start, spec.end) for r in results.values())


def _row(label, results, spec) -> dict[str, object]:
    preds = [p for r in results.values() for p in r.predictions]
    scores = [x for r in results.values() for x in r.scores]
    return {
        "Predictor": label,
        "Mean CRPS": sum(scores) / len(scores),
        "Forecasts": len(scores),
        "Origins": len({p.as_of for p in preds}),
        "Window": "ok" if not _is_stale(results, spec) else "STALE",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--score-only", action="store_true", help="never call an LLM; numerical rows still run")
    parser.add_argument("--advanced", action="store_true", help=f"run the new agents on {ADVANCED_MODEL} too")
    args = parser.parse_args()

    data_service = build_wti_multivariate_service(cache_dir=PACKAGE_DIR / "data" / "yfinance")
    covariates = [c for c in DEFAULT_WTI_COVARIATE_SERIES_IDS if c in set(data_service.series_ids)]
    spec = MultiTargetBacktestSpec.model_validate(yaml.safe_load(SPEC_PATH.read_text()))

    rows = []
    for label, predictor in _numerical(covariates).items():
        cached = load_multi_backtest_results(spec, predictor.predictor_id, store_dir=STORE)
        refresh = bool(cached) and _is_stale(cached, spec)
        if refresh:
            print(f"  {label}: cached file covers an older window, recomputing")
        results = cached_multi_backtest(predictor, spec, data_service, store_dir=STORE, force_refresh=refresh)
        rows.append(_row(label, results, spec))

    for label, predictor_id in CACHED_LLM_ROWS.items():
        results = load_multi_backtest_results(spec, predictor_id, store_dir=STORE)
        if results:
            rows.append(_row(label, results, spec))
        else:
            print(f"  {label}: not cached ({predictor_id})")

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
                rows.append(_row(label, results, spec))
            else:
                print(f"  {label}: not cached ({predictor.predictor_id})")

    board = pd.DataFrame(rows).set_index("Predictor").sort_values("Mean CRPS")
    print(f"\nOut of sample {spec.start:%Y-%m-%d} to {spec.end:%Y-%m-%d}, horizons {spec.tasks[0].horizons}.")
    print(board.round(2).to_string())


if __name__ == "__main__":
    main()
