"""Run AutoARIMA on the *agent* origin grid and compare it to the cached agents.

Why this exists
---------------
The token question — is the scenario-schema agent worth its cost over a free
numerical method? — needs both on one grid. The agents are already cached on
the grid anchored at 2026-07-23 (backtest 2014-04-14 -> 2024-07-23). AutoARIMA
is not: the only AutoARIMA run we have came off the local machine's grid
(2014-04-21 -> 2024-08-06), which is a different experiment.

This script rebuilds the agent grid exactly, computes AutoARIMA on it, and
scores every predictor cached under that spec_id together. Nothing else
recomputes: cached_multi_backtest loads the agents from disk, so this costs no
tokens. Working directory does not matter - all paths are absolute.

    python implementations/energy_oil_forecasting/scripts/run_autoarima_agent_grid.py

Note the cache is keyed by spec_id alone and never validates the window
(aieng/forecasting/evaluation/artifacts.py:349), so the grid assert below is
the only thing standing between you and a silent cross-grid comparison.
"""

from __future__ import annotations

import glob
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))  # <repo>/implementations

import energy_oil_forecasting  # noqa: E402
from aieng.forecasting.evaluation import (  # noqa: E402
    MultiTargetBacktestSpec,
    cached_multi_backtest,
)
from aieng.forecasting.evaluation.backtest import BacktestResult  # noqa: E402
from aieng.forecasting.methods.numerical.darts_arima import (  # noqa: E402
    DartsAutoARIMAPredictor,
)
from aieng.forecasting.methods.numerical.error_correction_regression import (  # noqa: E402
    ErrorCorrectionRegressionPredictor,
)
from energy_oil_forecasting.analysis import (  # noqa: E402
    per_horizon_crps,
    predictions_to_frame,
)
from energy_oil_forecasting.data import (  # noqa: E402
    DEFAULT_WTI_COVARIATE_SERIES_IDS,
    WTI_SERIES_ID,
    build_wti_multivariate_service,
)

# The anchor that reproduces the grid the cached agents were computed on.
ANCHOR_END = pd.Timestamp("2026-07-23")
EXPECTED_GRID = ("2014-04-14", "2024-07-23")

# Absolute, so the working directory cannot change where results are read or
# written. The library's DEFAULT_STORE_DIR is the *relative* Path
# ("data/predictions"), which resolves correctly only when the process runs
# from implementations/energy_oil_forecasting/ (as the notebook does). Running
# from the repo root instead silently wrote a completed backtest to
# <repo>/data/predictions/ while this script read from the package directory —
# no error, no warning, just a missing row. So STORE is derived from this
# file's own location and passed explicitly to every cache call below.
STORE = _HERE.parents[1] / "data" / "predictions"


def build_specs(data_service):
    """Rebuild the notebook's backtest/eval specs from ANCHOR_END."""
    spec_dir = Path(energy_oil_forecasting.__file__).parent / "specs"
    with open(spec_dir / "energy_oil_backtest.yaml") as f:
        backtest_spec = MultiTargetBacktestSpec.model_validate(yaml.safe_load(f))
    with open(spec_dir / "energy_oil_eval.yaml") as f:
        eval_spec = MultiTargetBacktestSpec.model_validate(yaml.safe_load(f))

    full_df = data_service.get_series(WTI_SERIES_ID, as_of=datetime.now()).sort_values("timestamp")
    data_start = full_df["timestamp"].min()
    data_end = ANCHOR_END

    holdout_start = data_end - pd.DateOffset(years=2)
    backtest_start = data_start + (holdout_start - data_start) / 2

    backtest_spec.start = backtest_start.strftime("%Y-%m-%d")
    backtest_spec.end = holdout_start.strftime("%Y-%m-%d")
    backtest_spec.stride = 63
    backtest_spec.tasks[0].horizons = [5, 10, 21, 63]
    backtest_spec.spec_id = "energy_oil_backtest_10yr_quarterly"

    eval_spec.start = holdout_start.strftime("%Y-%m-%d")
    eval_spec.end = (data_end - pd.tseries.offsets.BDay(63)).strftime("%Y-%m-%d")
    eval_spec.stride = 63
    eval_spec.tasks[0].horizons = [5, 10, 21, 63]
    eval_spec.spec_id = "energy_oil_eval_10yr_quarterly"

    if (backtest_spec.start, backtest_spec.end) != EXPECTED_GRID:
        raise SystemExit(
            f"Grid drifted to {backtest_spec.start} -> {backtest_spec.end}, expected "
            f"{EXPECTED_GRID[0]} -> {EXPECTED_GRID[1]}. The cached agents are on the "
            "expected grid; scoring against a different one compares different "
            "experiments. Check that the WTI series still starts where it did."
        )
    return backtest_spec, eval_spec


def load_cached(spec_id: str) -> dict[str, dict[str, BacktestResult]]:
    """Load every predictor already cached under a spec_id, keyed by predictor_id."""
    out: dict[str, dict[str, BacktestResult]] = {}
    for path in sorted(glob.glob(str(STORE / spec_id / "*.yaml"))):
        if "__eval_run" in os.path.basename(path):
            continue  # eval-run artefacts, not backtest results
        with open(path) as f:
            br = BacktestResult.model_validate(yaml.safe_load(f))
        out[br.predictor_id] = {br.spec.task.task_id: br}
    return out


def report(frame: pd.DataFrame, label: str) -> None:
    """Per-horizon CRPS plus a paired comparison against every other method."""
    print()
    print("=" * 96)
    print(f"{label}: MEAN CRPS BY PREDICTOR x HORIZON (lower is better)")
    print("=" * 96)
    print(per_horizon_crps(frame).round(2).to_string())

    grid = frame.pivot_table(index=["as_of", "horizon"], columns="predictor", values="crps")
    if "darts_autoarima" not in grid.columns:
        print("\n(no AutoARIMA column — nothing to compare)")
        return

    print()
    print("=" * 96)
    print(f"{label}: PAIRED vs AutoARIMA — same origin AND horizon. Negative = beats AutoARIMA.")
    print("=" * 96)
    print(f"{'predictor':52s} {'n':>4s} {'dCRPS':>8s} {'95% CI':>18s} {'win%':>6s} {'p':>8s}")
    for col in sorted(grid.columns):
        if col == "darts_autoarima":
            continue
        d = (grid[col] - grid["darts_autoarima"]).dropna()
        if len(d) < 3:
            print(f"{col:52s} {len(d):4d}   (too few overlapping points to test)")
            continue
        lo, hi = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
        p = stats.wilcoxon(d).pvalue if d.abs().sum() > 0 else float("nan")
        print(
            f"{col:52s} {len(d):4d} {d.mean():+8.3f} [{lo:+7.2f},{hi:+7.2f}] "
            f"{100 * (d < 0).mean():5.0f}% {p:8.4f}"
        )

    # Coverage is the axis that decided the h=63 ranking last time, so show it.
    print()
    print(f"{label}: 80% interval coverage by horizon (nominal 80)")
    print(
        frame.pivot_table(index="predictor", columns="horizon", values="inside80", aggfunc="mean")
        .mul(100)
        .round(1)
        .to_string()
    )


def build_numerical_predictors(data_service) -> list:
    """The token-free methods to place on the agent grid.

    Only the *base* ECM is here. Its configuration is fully recoverable from
    the local-run cache metadata (the seven standard yfinance covariates,
    use_log_levels=False, covariate_diff_path="zero"), and every one of those
    series is available from build_wti_multivariate_service().

    The four "expanded" ECM variants are deliberately absent. They need nine
    further series (OVX, EIA crude stocks, refinery utilization, the financial
    stress index, INDPRO, durable goods, the crack spread, and two Treasury
    series) that this data.py does not define. Their configs also cannot be
    fully recovered: ecm_regression_expanded and ecm_regression_expanded_
    levelonly record byte-identical metadata - same covariates, same flags -
    so whatever separates them lives in code that was never committed.
    Reproducing them needs the local machine's data.py plus its data/eia/ and
    data/fred/ caches.

    That is a smaller loss than it sounds: on the local grid the expanded
    variants scored 3.83-3.85 against the base ECM's 3.89 overall, a gap well
    inside the noise. The base is a fair stand-in for the family.
    """
    covariates = [c for c in DEFAULT_WTI_COVARIATE_SERIES_IDS if c in set(data_service.series_ids)]
    missing = sorted(set(DEFAULT_WTI_COVARIATE_SERIES_IDS) - set(covariates))
    if missing:
        print(f"  note: {len(missing)} covariate(s) unavailable, ECM will run without them: {missing}")
    return [
        DartsAutoARIMAPredictor(),
        ErrorCorrectionRegressionPredictor(covariate_series_ids=covariates),
    ]


def main() -> None:
    data_service = build_wti_multivariate_service()
    backtest_spec, eval_spec = build_specs(data_service)
    print(f"Backtest grid: {backtest_spec.start} -> {backtest_spec.end}  (stride {backtest_spec.stride})")
    print(f"Eval grid:     {eval_spec.start} -> {eval_spec.end}  (stride {eval_spec.stride})")

    predictors = build_numerical_predictors(data_service)
    print(f"Numerical predictors to place on the grid: {[p.predictor_id for p in predictors]}")

    for spec, label in ((backtest_spec, "BACKTEST"), (eval_spec, "EVAL")):
        cached_before = set(load_cached(spec.spec_id))
        print(f"\nAlready cached under {spec.spec_id}: {sorted(cached_before) or '(none)'}")

        for predictor in predictors:
            pid = predictor.predictor_id
            if pid in cached_before:
                print(f"  {pid}: already cached, loading")
                continue
            print(f"  {pid}: computing (agents load from cache — no tokens spent)...")
            cached_multi_backtest(
                predictor,
                spec,
                data_service,
                store_dir=STORE,  # never rely on cwd — see the STORE comment above
                force_refresh=False,
            )

        results = load_cached(spec.spec_id)
        for predictor in predictors:
            if predictor.predictor_id not in results:
                print(
                    f"  WARNING: {predictor.predictor_id} is still missing from "
                    f"{STORE / spec.spec_id}. It neither loaded nor computed — check for a "
                    f"stray copy under {Path.cwd() / 'data' / 'predictions' / spec.spec_id}."
                )
        frame = predictions_to_frame(results, data_service)
        if frame.empty:
            print(f"{label}: no scoreable predictions (horizons may not have resolved yet).")
            continue
        report(frame, label)


if __name__ == "__main__":
    main()
