"""Generate 04c_dense_numerical_calibration.ipynb.

Kept as a generator rather than hand-edited JSON so the notebook can be
regenerated cleanly; notebooks are unpleasant to review as diffs and this one
exists to answer a single question, so it is worth being able to rebuild it.
"""

from __future__ import annotations

import json
from pathlib import Path


def md(text: str) -> dict:
    return {"cell_type": "markdown", "id": f"md{abs(hash(text)) % 10**8}", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": f"cd{abs(hash(text)) % 10**8}",
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


CELLS = [
    md('''# WTI Dense Numerical Grid — Does Calibration Survive More Origins? (Notebook 4c)

`stride=21` instead of `63`, numerical predictors only. Roughly **three times
the origins** of NB04b over the same span, at zero token cost.

It exists to answer two questions that every other result in this project is
currently blocked on:

1. **Does conformal calibration land on its nominal 80% when it has enough
   history?** On the quarterly grid it overshoots to 86–92%, and the cause is
   arithmetic rather than mysterious: the conformal cut is taken at
   `ceil((n+1)(1-α))/n`, which is 0.824 when `n=34` and 0.875 when `n=16`. Only
   more calibration points can close that gap. If ~100 points lands it near 80%,
   conformal is worth building into the pipeline; if not, the method needs more
   data than this problem provides and the tokens are better spent elsewhere.

2. **Are the numerical coverage numbers real?** AutoARIMA at 82.6% and ECM at
   86.8% come from 43 origins. Several conclusions in this project have not
   survived a second look at a wider window — including two of my own — so
   these deserve one before they are quoted.

Both questions are predictor-agnostic, which is why this notebook can answer
them without a single LLM call. If conformal works here, spending tokens to run
the agents on a dense grid becomes justified. That decision is the point.

## Why CRPS, and why MAE is only a diagnostic here

CRPS **generalises** MAE: for a point-mass forecast — every quantile equal, as
Naive emits — CRPS reduces exactly to MAE. Reporting the two as peers implies a
choice that does not exist.

MAE also cannot see this project's dominant failure mode. It scores the centre
and is blind to the interval, so a model with an excellent point forecast and
52% coverage looks good under it. Sorting a leaderboard on MAE would put the
worst-calibrated model near the top.

MAE keeps exactly one use, and it appears below as a diagnostic column, never
as the headline: it isolates the centre, so "MAE fine, coverage bad" localises
the fault to the band rather than the aim.'''),
    md("---\n## 1. Setup"),
    code('''import sys
import warnings
from datetime import datetime
from pathlib import Path

import energy_oil_forecasting
import numpy as np
import pandas as pd
import yaml
from aieng.forecasting.evaluation import MultiTargetBacktestSpec, cached_multi_backtest, describe_spec
from energy_oil_forecasting.data import (
    DEFAULT_WTI_COVARIATE_SERIES_IDS,
    WTI_SERIES_ID,
    build_wti_multivariate_service,
)

warnings.filterwarnings("ignore")

# scripts/ holds the calibration code this notebook is built around.
sys.path.insert(0, str(Path.cwd() / "scripts"))
from conformal import calibrate_series  # noqa: E402
from labels import label  # noqa: E402

# 2 origins instead of ~129, for a fast end-to-end check.
SMOKE_TEST = False

HORIZONS = [5, 10, 21, 63]
STRIDE = 21  # the whole point of this notebook; NB04b uses 63

data_service = build_wti_multivariate_service()
_available = set(data_service.series_ids)
COVARIATES = [c for c in DEFAULT_WTI_COVARIATE_SERIES_IDS if c in _available]
_missing = [c for c in DEFAULT_WTI_COVARIATE_SERIES_IDS if c not in _available]
if _missing:
    print(f"!! {len(_missing)} covariate(s) failed to build and were dropped: {_missing}")

spec_dir = Path(energy_oil_forecasting.__file__).parent / "specs"
with open(spec_dir / "energy_oil_backtest.yaml") as f:
    backtest_spec = MultiTargetBacktestSpec.model_validate(yaml.safe_load(f))
with open(spec_dir / "energy_oil_eval.yaml") as f:
    eval_spec = MultiTargetBacktestSpec.model_validate(yaml.safe_load(f))

# ── ANCHOR_END: the origin grid must not drift with the calendar ─────────────
# Same discipline as NB04b, and for the same reason. Deriving the grid from
# "today" makes it a function of when the cell ran, and the prediction cache is
# keyed on spec_id alone -- it cannot tell that the window moved underneath it,
# so two runs a fortnight apart land in one folder and are later compared as
# though they were one experiment. The 10yr-local cache has three such grids in
# it, sharing zero origins.
ANCHOR_END = pd.Timestamp("2026-07-23")
_full = data_service.get_series(WTI_SERIES_ID, as_of=datetime.now()).sort_values("timestamp")
data_start = _full["timestamp"].min()
data_end = _full["timestamp"].max() if SMOKE_TEST else ANCHOR_END

holdout_start = data_end - pd.DateOffset(years=2)
backtest_start = data_start + (holdout_start - data_start) / 2

_suffix = "_smoke" if SMOKE_TEST else ""
if SMOKE_TEST:
    backtest_spec.start = (data_end - pd.DateOffset(years=5)).strftime("%Y-%m-%d")
    backtest_spec.end = (data_end - pd.DateOffset(years=5) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    eval_spec.start = (data_end - pd.DateOffset(months=1)).strftime("%Y-%m-%d")
    eval_spec.end = data_end.strftime("%Y-%m-%d")
    backtest_spec.stride = eval_spec.stride = 5
else:
    backtest_spec.start = backtest_start.strftime("%Y-%m-%d")
    backtest_spec.end = holdout_start.strftime("%Y-%m-%d")
    eval_spec.start = holdout_start.strftime("%Y-%m-%d")
    eval_spec.end = (data_end - pd.tseries.offsets.BDay(max(HORIZONS))).strftime("%Y-%m-%d")
    backtest_spec.stride = eval_spec.stride = STRIDE

for _spec, _name in ((backtest_spec, "backtest"), (eval_spec, "eval")):
    _spec.tasks[0].horizons = list(HORIZONS)
    # "_dense" keeps this off NB04b's quarterly results entirely. Same window,
    # different stride, so sharing a spec_id would silently mix the two.
    _spec.spec_id = f"energy_oil_{_name}_10yr_dense{_suffix}"

if not SMOKE_TEST:
    assert (backtest_spec.start, backtest_spec.end) == ("2014-04-14", "2024-07-23"), (
        f"Backtest grid drifted to {backtest_spec.start} → {backtest_spec.end}; expected the NB04b window."
    )

_origins = len(pd.bdate_range(backtest_spec.start, backtest_spec.end)) // STRIDE
print(f"{'⚡ SMOKE' if SMOKE_TEST else '📊 FULL'}  stride={backtest_spec.stride}  horizons={HORIZONS}")
print(f"backtest {backtest_spec.start} → {backtest_spec.end}   (~{_origins} origins, vs 43 at stride 63)")
print(f"eval     {eval_spec.start} → {eval_spec.end}")
print(f"covariates for ECM ({len(COVARIATES)}): {', '.join(COVARIATES)}")'''),
    md('''---
## 2. Predictors — numerical only

No agents, so this notebook costs nothing to run and can be repeated freely.

The line-up is deliberately small. NB04b's registry records the finding that
settled the ECM family: neither a wider covariate panel nor a corrected
cointegration specification improved accuracy, so this carries the base ECM and
its log-target counterpart as calibration reference points and leaves the other
four variants behind. Adding more numerical variants is not what this grid is
for.'''),
    code('''from dataclasses import dataclass
from typing import Callable

from aieng.forecasting.methods import LastValuePredictor
from aieng.forecasting.methods.numerical.darts_arima import DartsAutoARIMAPredictor
from aieng.forecasting.methods.numerical.darts_classical import DartsKalmanForecasterPredictor
from aieng.forecasting.methods.numerical.error_correction_regression import (
    ErrorCorrectionRegressionPredictor,
)


@dataclass
class PredictorEntry:
    name: str
    factory: Callable[[], object]
    enabled: bool = True


REGISTRY = [
    # The floor. Also the one predictor whose CRPS *is* its MAE, since every
    # quantile is equal -- which is why its coverage reads 0% and why that is
    # not a calibration result.
    PredictorEntry("Naive", LastValuePredictor),
    # Worst-calibrated numerical method on the quarterly grid at 28.7%. Kept
    # because conformal improved it more than anything else (30.4% -> 86.7%,
    # CRPS -0.458), so it is the strongest single test of whether conformal
    # still works with more history.
    PredictorEntry("Kalman", DartsKalmanForecasterPredictor),
    # The reference. Best-calibrated token-free method on the quarterly grid.
    PredictorEntry("AutoARIMA", DartsAutoARIMAPredictor),
    PredictorEntry("AutoARIMA (log ret)", lambda: DartsAutoARIMAPredictor(log_returns=True)),
    PredictorEntry("ECM (level)", lambda: ErrorCorrectionRegressionPredictor(covariate_series_ids=COVARIATES)),
    PredictorEntry(
        "ECM (log target)",
        lambda: ErrorCorrectionRegressionPredictor(covariate_series_ids=COVARIATES, log_target=True),
    ),
]

PREDICTORS = {e.name: e.factory() for e in REGISTRY if e.enabled}
print(f"{len(PREDICTORS)} predictors, 0 LLM calls:")
for _n in PREDICTORS:
    print(f"  {_n}")'''),
    md('''---
## 3. Run

Three times the origins of NB04b, so expect roughly three times its runtime —
AutoARIMA refits at every origin. Results cache to
`data/predictions/energy_oil_*_10yr_dense/`, so an interrupted run resumes
where it stopped rather than starting over.'''),
    code('''import time

def run(spec, label_text):
    out = {}
    for i, (name, predictor) in enumerate(PREDICTORS.items()):
        if i:
            time.sleep(1)
        started = time.time()
        out[name] = cached_multi_backtest(predictor, spec, data_service, max_retries=2, retry_delay=5.0)
        print(f"  {name:22} {time.time() - started:6.1f}s")
    print(f"{label_text} complete.\\n")
    return out

print(f"Backtest ({backtest_spec.start} → {backtest_spec.end})")
backtest_results = run(backtest_spec, "Backtest")
print(f"Eval ({eval_spec.start} → {eval_spec.end})")
eval_results = run(eval_spec, "Eval")'''),
    md('''---
## 4. Results

One tidy frame per scored prediction, then everything reads from it.

`crps` is the headline throughout. `abs_error` is present, but as a diagnostic
for separating a bad centre from a bad band — never as a ranking.'''),
    code('''import properscoring as ps


def to_frame(results_by_predictor, service):
    """One row per scored prediction: crps, coverage, width, and the centre's error."""
    actual_df = service.get_series(WTI_SERIES_ID, as_of=datetime.now())
    actual_by_date = {pd.Timestamp(r["timestamp"]).normalize(): float(r["value"]) for _, r in actual_df.iterrows()}

    rows = []
    for name, task_results in results_by_predictor.items():
        for result in task_results.values():
            for pred in result.predictions:
                payload = pred.payload
                quantiles = getattr(payload, "quantiles", None)
                if not quantiles:
                    continue
                forecast_date = pd.Timestamp(pred.forecast_date).normalize()
                actual = actual_by_date.get(forecast_date)
                if actual is None:
                    continue
                q = {float(k): float(v) for k, v in quantiles.items()}
                origin = pd.Timestamp(pred.as_of)
                rows.append(
                    {
                        "predictor": name,
                        "origin": origin.normalize(),
                        "resolves": forecast_date,
                        "horizon": int(np.busday_count(origin.date(), forecast_date.date())),
                        "actual": actual,
                        "quantiles": q,
                        "crps": float(ps.crps_ensemble(actual, np.array(sorted(q.values())))),
                        "inside80": float(q[0.1] <= actual <= q[0.9]),
                        "width80": q[0.9] - q[0.1],
                        "abs_error": abs(payload.point_forecast - actual),
                    }
                )
    return pd.DataFrame(rows)


backtest_frame = to_frame(backtest_results, data_service)
eval_frame = to_frame(eval_results, data_service)
print(f"backtest: {len(backtest_frame)} scored predictions, "
      f"{backtest_frame['origin'].nunique()} origins")
print(f"eval:     {len(eval_frame)} scored predictions, {eval_frame['origin'].nunique()} origins")
print(f"\\ncalibration points per predictor x horizon: "
      f"{int(backtest_frame.groupby(['predictor', 'horizon']).size().median())} "
      f"(quarterly grid had ~34)")'''),
    code('''def leaderboard(frame, title):
    """CRPS-ranked, with a standard error so a lead can be judged against noise."""
    grouped = frame.groupby("predictor")
    board = pd.DataFrame(
        {
            "CRPS": grouped["crps"].mean(),
            "se": grouped["crps"].std(ddof=1) / np.sqrt(grouped["crps"].count()),
            "Cov80": 100 * grouped["inside80"].mean(),
            "Width80": grouped["width80"].mean(),
            "MAE": grouped["abs_error"].mean(),
            "n": grouped["crps"].count().astype(int),
        }
    ).sort_values("CRPS")
    print("━" * 78)
    print(title)
    print("━" * 78)
    print(board.round(3).to_string())
    best = board.index[0]
    gap = board["CRPS"].iloc[1] - board["CRPS"].iloc[0]
    noise = board["se"].iloc[:2].max()
    verdict = "clears" if gap > noise else "does NOT clear"
    print(f"\\n{best} leads by {gap:.3f} CRPS; that {verdict} the standard error ({noise:.3f}).")
    return board


backtest_board = leaderboard(backtest_frame, f"BACKTEST {backtest_spec.start} → {backtest_spec.end}")
print()
eval_board = leaderboard(eval_frame, f"EVAL {eval_spec.start} → {eval_spec.end}")'''),
    md('''---
## 5. Calibration — question 2

Coverage per predictor × horizon, against a nominal 80%.

Pooling horizons hides the effect: band width grows with horizon while the miss
rate need not, so a model can sit at 70% at h=10 and 88% at h=63 and report a
reassuring 81% overall. That is not hypothetical — it is what AutoARIMA (log
returns) did on the quarterly grid.

The comparison that matters is against the quarterly numbers below. Where they
disagree, the quarterly number was a 43-origin artifact.'''),
    code('''QUARTERLY_COV80 = {  # NB04b, 43 origins — what these numbers are being tested against
    "Naive": 0.0, "Kalman": 28.7, "AutoARIMA": 82.6,
    "AutoARIMA (log ret)": 81.4, "ECM (level)": 86.8, "ECM (log target)": 83.8,
}

def calibration_table(frame, title):
    pivot = 100 * frame.pivot_table(index="predictor", columns="horizon", values="inside80", aggfunc="mean")
    pivot.columns = [f"h={h}d" for h in pivot.columns]
    pivot["All"] = 100 * frame.groupby("predictor")["inside80"].mean()
    pivot["quarterly"] = [QUARTERLY_COV80.get(p, float("nan")) for p in pivot.index]
    pivot["shift"] = pivot["All"] - pivot["quarterly"]
    print("━" * 78)
    print(f"{title}  (nominal 80%)")
    print("━" * 78)
    print(pivot.round(1).sort_values("All", ascending=False).to_string())
    moved = pivot["shift"].abs().max()
    print(f"\\nlargest move vs the quarterly grid: {moved:.1f} points.")
    print("A large move means the quarterly figure was a small-sample artifact,")
    print("not a property of the model.")
    return pivot


_ = calibration_table(backtest_frame, "COVERAGE BY PREDICTOR × HORIZON — backtest")
print()
_ = calibration_table(eval_frame, "COVERAGE BY PREDICTOR × HORIZON — eval")'''),
    md('''---
## 6. Conformal calibration — question 1, the one this notebook exists for

Conformalized quantile regression shifts each quantile pair by a constant read
off the predictor's own past conformity scores. On the quarterly grid it moved
the mean distance from nominal across 13 predictors from 20.2 points to 8.5 —
but it overshot, landing at 86–92% rather than 80%.

The overshoot is arithmetic, not a defect. The cut is taken at
`ceil((n+1)(1-α))/n`, which is 0.824 at `n=34`. Shrinking the calibration window
made it *worse* (tested: 8.5 → 9.5 points at `n=16`, where the level is 0.875),
which is the signature of this cause and rules out the alternatives.

So the prediction under test is specific: **with roughly three times the
calibration points, the built-in conservatism should fall by about two thirds
and coverage should land materially closer to 80%.** The cell prints the level
being cut at, so the prediction is checkable rather than merely plausible.

Calibration uses only forecasts that had already resolved at the origin being
corrected, so this is what running conformal live would have produced.'''),
    code('''def conformal_report(frame, title):
    rows = []
    for (name, horizon), group in frame.groupby(["predictor", "horizon"]):
        records = group[["origin", "resolves", "quantiles", "actual"]].to_dict("records")
        calibrated, _ = calibrate_series(records, window=None)
        for record in calibrated:
            if record["conformal"] == record["quantiles"]:
                continue  # no resolved history yet; nothing was applied
            values = np.array(sorted(record["conformal"].values()))
            rows.append(
                {
                    "predictor": name,
                    "horizon": horizon,
                    "base_crps": float(ps.crps_ensemble(record["actual"], np.array(sorted(record["quantiles"].values())))),
                    "conf_crps": float(ps.crps_ensemble(record["actual"], values)),
                    "base_in": float(record["quantiles"][0.1] <= record["actual"] <= record["quantiles"][0.9]),
                    "conf_in": float(record["conformal"][0.1] <= record["actual"] <= record["conformal"][0.9]),
                }
            )

    out = pd.DataFrame(rows)
    grouped = out.groupby("predictor")
    table = pd.DataFrame(
        {
            "CRPS base": grouped["base_crps"].mean(),
            "CRPS conf": grouped["conf_crps"].mean(),
            "Cov base": 100 * grouped["base_in"].mean(),
            "Cov conf": 100 * grouped["conf_in"].mean(),
            "n": grouped["base_crps"].count().astype(int),
        }
    ).sort_values("CRPS conf")

    print("━" * 78)
    print(f"{title}  (nominal 80%)")
    print("━" * 78)
    print(table.round(2).to_string())

    typical = int(out.groupby(["predictor", "horizon"]).size().median())
    level = np.ceil((typical + 1) * 0.8) / typical
    before = (table["Cov base"] - 80).abs().mean()
    after = (table["Cov conf"] - 80).abs().mean()
    print(f"\\ncalibration sets ~{typical} points -> the 80% band is cut at the "
          f"{100 * level:.1f}th percentile")
    print(f"built-in conservatism: {100 * (level - 0.8):.1f} points "
          f"(quarterly grid: 2.4 points at n=34)")
    print(f"mean distance from nominal: {before:.1f} -> {after:.1f} points "
          f"(quarterly grid: 20.2 -> 8.5)")
    if after < 4.0:
        print("\\n=> Conformal lands near nominal with this much history. Worth building")
        print("   into the pipeline, and worth spending tokens to test on the agents.")
    else:
        print("\\n=> Still short of nominal. Either more origins are needed than this grid")
        print("   provides, or something other than the finite-sample correction is at")
        print("   work -- check whether the residual gap tracks 100*(level-0.8) above.")
    return table


_ = conformal_report(backtest_frame, "CONFORMAL — backtest")'''),
    md('''---
## 7. What this settles

Read the three numbers in order:

1. **Conformal's mean distance from nominal** (Section 6). Below ~4 points, the
   overshoot was a sample-size problem and conformal is worth productionising as
   a `Predictor` wrapper — and worth spending tokens to test on the LLM agents,
   which are the badly-calibrated ones and stand to gain most. Still near 8
   points, and the method needs more data than this problem provides.

2. **The coverage shift vs the quarterly grid** (Section 5). Small moves mean
   the 43-origin numbers were sound and can be quoted. Large moves mean they
   were artifacts — and so, by extension, are the agent coverage figures from
   the same grid.

3. **Whether any CRPS lead clears its standard error** (Section 4). On the
   quarterly grid nothing did except against Naive and Kalman. With three times
   the origins the bars shrink by about √3; if the numerical methods are still
   inseparable, then they are genuinely inseparable and the remaining headroom
   is in calibration and in the agent layer, not in another regression.

What this notebook cannot answer: anything about the agents. Their coverage
problem is the project's largest, and it needs a token-spending run on this same
dense grid. The point of doing this first is to find out whether that run is
worth paying for.'''),
]


def main() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = Path(__file__).resolve().parent.parent / "04c_dense_numerical_calibration.ipynb"
    out.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
