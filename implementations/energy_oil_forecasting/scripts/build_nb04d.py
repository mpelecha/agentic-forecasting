"""Generate 04d_numerical_timesfm3.ipynb.

Kept as a generator rather than hand-edited JSON, same reasoning as
build_nb04c.py: notebooks are unpleasant to review as diffs and this one
exists to answer a single question, so it should be trivial to rebuild.
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
    md('''# WTI Numerical Leaderboard vs. TimesFM 3.0 (Notebook 4d)

Every numerical predictor this project has tested (Naive, Kalman, AutoARIMA
level and log-return, ECM level and log-target) plus one new entry: Google's
TimesFM 3.0, a pretrained zero-shot foundation model. No LLM agents, no
fitting for TimesFM3 -- the checkpoint loads once and every origin is a
forward pass over whatever history `context` exposes at that point.

**Backtest: 2023-01 -> 2024-12, weekly origins.**
**Eval: 2025-01 -> most recent resolvable origin, weekly origins.**

This is a fixed calendar split rather than a rolling "last N years" window
(contrast NB04c's dense grid), chosen so the eval period cleanly post-dates
TimesFM 3.0's disclosed pretraining-data cutoffs (Wikipedia Pageviews: Nov
2023; Google Trends: EoY 2022) -- see the caveat below and in
`TimesFM3Predictor`'s docstring.

## Read the backtest and eval windows differently for TimesFM3

Every other predictor here is fit at each origin from data `context` exposes,
so backtest and eval are both genuinely out-of-sample for them by
construction. TimesFM 3.0 is pretrained once, and its training corpus's exact
per-series cutoff is not fully disclosed -- only two of its named components
(Wikipedia Pageviews, Google Trends) have a published cutoff, both in
2022-2023. **Treat TimesFM3's 2023-2024 backtest numbers as a weaker signal**
than the other six predictors': if WTI crude price history was in its
pretraining set, "zero-shot forecast" on those origins is not really
out-of-sample. The 2025-2026 eval window is the more trustworthy read for
TimesFM3, precisely because it sits after the only cutoffs Google has
disclosed.

## Requires the `timesfm` extra

`TimesFM3Predictor` needs `pip install timesfm[torch]` (or
`uv sync --extra timesfm` from `aieng-forecasting/`) and downloads the
`google/timesfm-3.0-pytorch` checkpoint from Hugging Face on first use. Its
pretrained weights are licensed `timesfm-non-commercial-license-v1.0`, not
Apache-2.0 -- fine here, not for production use. See
`TimesFM3Predictor`'s docstring for both caveats in full.'''),
    md("---\n## 1. Setup"),
    code('''import warnings
from datetime import datetime
from pathlib import Path

import energy_oil_forecasting
import numpy as np
import pandas as pd
import yaml
from aieng.forecasting.evaluation import MultiTargetBacktestSpec, cached_multi_backtest
from energy_oil_forecasting.data import (
    DEFAULT_WTI_COVARIATE_SERIES_IDS,
    EXPANDED_WTI_COVARIATE_SERIES_IDS,
    LEVEL_VALUED_WTI_COVARIATE_SERIES_IDS,
    WTI_SERIES_ID,
    build_wti_multivariate_service,
)

warnings.filterwarnings("ignore")

# 2 origins instead of the full grid, for a fast end-to-end check.
SMOKE_TEST = False

HORIZONS = [5, 10, 21]
STRIDE = 5  # weekly, matching NB04/NB04b -- this is a leaderboard notebook, not a dense-grid one

data_service = build_wti_multivariate_service()
_available = set(data_service.series_ids)
COVARIATES = [c for c in DEFAULT_WTI_COVARIATE_SERIES_IDS if c in _available]
COVARIATES_EXPANDED = [c for c in EXPANDED_WTI_COVARIATE_SERIES_IDS if c in _available]
LEVEL_VALUED_EXPANDED = [c for c in LEVEL_VALUED_WTI_COVARIATE_SERIES_IDS if c in COVARIATES_EXPANDED]
_missing = [c for c in EXPANDED_WTI_COVARIATE_SERIES_IDS if c not in _available]
if _missing:
    print(f"!! {len(_missing)} covariate(s) failed to build and were dropped: {_missing}")

spec_dir = Path(energy_oil_forecasting.__file__).parent / "specs"
with open(spec_dir / "energy_oil_backtest.yaml") as f:
    backtest_spec = MultiTargetBacktestSpec.model_validate(yaml.safe_load(f))
with open(spec_dir / "energy_oil_eval.yaml") as f:
    eval_spec = MultiTargetBacktestSpec.model_validate(yaml.safe_load(f))

# ── ANCHOR: fixed calendar windows, not derived from "today" ─────────────────
# Same discipline as NB04c and for the same reason -- the prediction cache is
# keyed on spec_id alone, so a window that drifts with the calendar silently
# mixes runs from different dates into one cache folder. The backtest window
# is fully fixed. The eval window's START is fixed at 2025-01-01; its END is
# the latest origin whose longest horizon (21 business days) actually
# resolves against cached data, so the notebook works the day it's run
# without needing a hand-updated date -- but every run after the first
# should see the SAME end date turn into MORE resolved origins, not a moved
# window, unless the WTI cache itself was rebuilt with a later cutoff.
BACKTEST_START = "2023-01-02"
BACKTEST_END = "2024-12-31"
EVAL_START = "2025-01-01"

_full = data_service.get_series(WTI_SERIES_ID, as_of=datetime.now()).sort_values("timestamp")
data_end = _full["timestamp"].max()
eval_end = (pd.Timestamp(data_end) - pd.tseries.offsets.BDay(max(HORIZONS))).strftime("%Y-%m-%d")

_suffix = "_smoke" if SMOKE_TEST else ""
if SMOKE_TEST:
    backtest_spec.start = BACKTEST_START
    backtest_spec.end = "2023-02-01"
    eval_spec.start = EVAL_START
    eval_spec.end = "2025-02-01"
    backtest_spec.stride = eval_spec.stride = 5
else:
    backtest_spec.start, backtest_spec.end = BACKTEST_START, BACKTEST_END
    eval_spec.start, eval_spec.end = EVAL_START, eval_end
    backtest_spec.stride = eval_spec.stride = STRIDE

for _spec, _name in ((backtest_spec, "backtest"), (eval_spec, "eval")):
    _spec.tasks[0].horizons = list(HORIZONS)
    _spec.spec_id = f"energy_oil_{_name}_2023_timesfm3{_suffix}"

if not SMOKE_TEST:
    assert pd.Timestamp(eval_spec.end).year in (2025, 2026), (
        f"Eval window resolved to {eval_spec.end}, outside 2025-2026 -- check the WTI price cache is current."
    )

print(f"{'⚡ SMOKE' if SMOKE_TEST else '📊 FULL'}  stride={backtest_spec.stride}  horizons={HORIZONS}")
print(f"backtest {backtest_spec.start} -> {backtest_spec.end}")
print(f"eval     {eval_spec.start} -> {eval_spec.end}  (resolved against latest cached data {data_end.date()})")
print(f"covariates for ECM / TimesFM3 (cov)   base={len(COVARIATES)}  expanded={len(COVARIATES_EXPANDED)}")'''),
    md('''---
## 2. Predictors — numerical only, plus TimesFM 3.0

Six predictors carried over unchanged from NB04c (same classes, same
`predictor_id`s, so their cached results are directly comparable across
notebooks), plus two new TimesFM 3.0 entries: univariate zero-shot, and with
the same expanded covariate panel ECM uses (`TimesFM3Predictor` holds future
covariates flat the same way ECM does -- returns at 0, levels at their last
observed value, using `LEVEL_VALUED_WTI_COVARIATE_SERIES_IDS` to tell them
apart).

TimesFM3 forecasts price level only, no log-return variant -- see
`TimesFM3Predictor`'s docstring for why that variant is deliberately not
implemented yet (its quantile-only output cannot be cumulated across horizon
steps as safely as AutoARIMA's sample paths can).'''),
    code('''from dataclasses import dataclass
from typing import Callable

from aieng.forecasting.methods import LastValuePredictor
from aieng.forecasting.methods.numerical.darts_arima import DartsAutoARIMAPredictor
from aieng.forecasting.methods.numerical.darts_classical import DartsKalmanForecasterPredictor
from aieng.forecasting.methods.numerical.error_correction_regression import (
    ErrorCorrectionRegressionPredictor,
)
from aieng.forecasting.methods.numerical.timesfm3 import TimesFM3Predictor


@dataclass
class PredictorEntry:
    name: str
    factory: Callable[[], object]
    enabled: bool = True


REGISTRY = [
    PredictorEntry("Naive", LastValuePredictor),
    PredictorEntry("Kalman", DartsKalmanForecasterPredictor),
    PredictorEntry("AutoARIMA", DartsAutoARIMAPredictor),
    PredictorEntry("AutoARIMA (log ret)", lambda: DartsAutoARIMAPredictor(log_returns=True)),
    PredictorEntry("ECM (level)", lambda: ErrorCorrectionRegressionPredictor(covariate_series_ids=COVARIATES)),
    PredictorEntry(
        "ECM (log target)",
        lambda: ErrorCorrectionRegressionPredictor(covariate_series_ids=COVARIATES, log_target=True),
    ),
    PredictorEntry("TimesFM3", TimesFM3Predictor),
    PredictorEntry(
        "TimesFM3 (cov)",
        lambda: TimesFM3Predictor(
            covariate_series_ids=COVARIATES_EXPANDED,
            level_valued_covariate_series_ids=LEVEL_VALUED_EXPANDED,
        ),
    ),
]

PREDICTORS = {e.name: e.factory() for e in REGISTRY if e.enabled}
print(f"{len(PREDICTORS)} predictors, 0 LLM calls:")
for _n in PREDICTORS:
    print(f"  {_n}")'''),
    md('''---
## 3. Run

TimesFM3 loads its checkpoint once (first call) and does not refit per
origin, so it should be one of the faster entries here despite being the
newest; AutoARIMA refits at every origin and is usually the slowest. Results
cache to `data/predictions/energy_oil_*_2023_timesfm3/`, so an interrupted
run resumes where it stopped rather than starting over.'''),
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

print(f"Backtest ({backtest_spec.start} -> {backtest_spec.end})")
backtest_results = run(backtest_spec, "Backtest")
print(f"Eval ({eval_spec.start} -> {eval_spec.end})")
eval_results = run(eval_spec, "Eval")'''),
    md('''---
## 4. Results

One tidy frame per scored prediction, then everything reads from it. `crps`
is the headline throughout; `abs_error` is a diagnostic for separating a bad
centre from a bad band, never a ranking.

TimesFM3's quantile dict only has the 9 levels 0.1-0.9 (no 0.05/0.95 --
the model does not expose those tails), so its `crps_ensemble` approximation
uses 9 points where the other predictors use 11. The 80% coverage check below
(`q[0.1] <= actual <= q[0.9]`) is unaffected either way.'''),
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
print(f"backtest: {len(backtest_frame)} scored predictions, {backtest_frame['origin'].nunique()} origins")
print(f"eval:     {len(eval_frame)} scored predictions, {eval_frame['origin'].nunique()} origins")'''),
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


backtest_board = leaderboard(backtest_frame, f"BACKTEST {backtest_spec.start} -> {backtest_spec.end}  (weaker signal for TimesFM3 -- see caveat above)")
print()
eval_board = leaderboard(eval_frame, f"EVAL {eval_spec.start} -> {eval_spec.end}  (post-dates TimesFM3's disclosed pretraining cutoffs)")'''),
    md('''---
## 5. Calibration

Coverage per predictor x horizon, against a nominal 80%. Pooling horizons
hides the effect: band width grows with horizon while the miss rate need
not, so a model can sit well below nominal at one horizon and above it at
another and still report a reassuring pooled number.'''),
    code('''def calibration_table(frame, title):
    pivot = 100 * frame.pivot_table(index="predictor", columns="horizon", values="inside80", aggfunc="mean")
    pivot.columns = [f"h={h}d" for h in pivot.columns]
    pivot["All"] = 100 * frame.groupby("predictor")["inside80"].mean()
    print("━" * 78)
    print(f"{title}  (nominal 80%)")
    print("━" * 78)
    print(pivot.round(1).sort_values("All", ascending=False).to_string())
    return pivot


_ = calibration_table(backtest_frame, "COVERAGE BY PREDICTOR x HORIZON — backtest")
print()
_ = calibration_table(eval_frame, "COVERAGE BY PREDICTOR x HORIZON — eval")'''),
    md('''---
## 6. What this settles

Read the eval-window numbers (Section 4/5, 2025-2026) as the primary result
for TimesFM3: they post-date its only two disclosed pretraining cutoffs
(Wikipedia Pageviews Nov 2023, Google Trends EoY 2022), so a genuinely
zero-shot forecast is the more defensible read there than on the 2023-2024
backtest window. Where TimesFM3 lands relative to the other six predictors on
CRPS and 80% coverage in that window is the headline; the backtest window is
useful context but should not be quoted as an independent confirmation of the
same finding.

Two things this notebook does not answer: whether TimesFM3's covariate mode
(`TimesFM3 (cov)`) does any better than univariate here -- every other
covariate experiment in this project found the panel did not help, so a
repeat null result would be the fourth independent confirmation, not a new
finding -- and whether GiftEvalPretrain (TimesFM3's largest disclosed
pretraining component) contains WTI or other crude-oil price series at all.
That second question is the one that would most change how much to trust
this notebook's TimesFM3 numbers, and it is not answerable from here.'''),
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
    out = Path(__file__).resolve().parent.parent / "04d_numerical_timesfm3.ipynb"
    out.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
