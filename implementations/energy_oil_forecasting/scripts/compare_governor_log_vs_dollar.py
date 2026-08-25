"""Paired offline comparison: dollar-delta governor vs log-return governor.

The governor is pure Python post-processing -- the LLM never sees the
percentile table (delta-governed computes it in ``prepare()``, before the
prompt is built and without putting it in the payload). So for a *fixed*
cached LLM response -- same rank, same uncertainty action, same evidence
tier -- the two governors produce different final numbers deterministically.

That makes this a properly paired experiment: identical LLM judgment on both
sides, only the governor differs. No LLM calls, no cost, and none of the
run-to-run search/sampling variance that a side-by-side leaderboard run
would carry.

Self-check: recomputing with ``log_returns=False`` must reproduce the cached
``final_quantiles`` exactly. If it does not, the reconstruction below is
wrong and the comparison is meaningless -- the script says so and stops.

Run from the implementations/energy_oil_forecasting directory:

    python scripts/compare_governor_log_vs_dollar.py
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from aieng.forecasting.evaluation.backtest import _crps_for_prediction
from aieng.forecasting.evaluation.prediction import ContinuousForecast, Prediction
from aieng.forecasting.evaluation.task import ForecastingTask
from energy_oil_forecasting.cfm_agent_v_5_2.config import DEFAULT_SETTINGS
from energy_oil_forecasting.cfm_agent_v_5_2.schemas import ModelHorizonForecast
from energy_oil_forecasting.cfm_agent_v_5_2_2_delta_governed.forecast_engine import (
    PythonForecastEngineDeltaGoverned,
)
from energy_oil_forecasting.cfm_agent_v_5_2_2_delta_governed.schemas import (
    PolicyDecisionDeltaGoverned,
)
from energy_oil_forecasting.data import WTI_SERIES_ID, build_wti_multivariate_service
from energy_oil_forecasting.price_deltas import compute_horizon_delta_percentiles


CACHE = Path(
    "data/predictions/energy_oil_eval_biweekly/"
    "agent_predictor_cfm_agent_v_5_2_2_delta_governed_gemini-3.1-flash-lite-preview_"
    "continuous__wti_oil_price_forecast.yaml"
)
HORIZONS = [5, 10, 21]


def _float_keys(quantiles: dict) -> dict[float, float]:
    return {float(k): float(v) for k, v in quantiles.items()}


def _score(quantiles: dict[float, float], point: float, actual: float, template: dict) -> tuple[float, float]:
    """Return (crps, inside_80) for one forecast, using the pipeline's own CRPS."""
    prediction = Prediction(
        predictor_id="recompute",
        task_id=template["task_id"],
        issued_at=datetime.now(tz=timezone.utc).replace(tzinfo=None),
        as_of=pd.Timestamp(template["as_of"]).to_pydatetime(),
        forecast_date=pd.Timestamp(template["forecast_date"]).to_pydatetime(),
        payload=ContinuousForecast(point_forecast=point, quantiles=quantiles),
    )
    crps = _crps_for_prediction(prediction, actual)
    inside = float(quantiles[0.1] <= actual <= quantiles[0.9])
    return crps, inside


def main() -> None:
    data = yaml.safe_load(CACHE.read_text())
    predictions = data["predictions"]

    service = build_wti_multivariate_service()
    resolved_as_of = datetime.now(tz=timezone.utc).replace(tzinfo=None)
    actual_df = service.get_series(WTI_SERIES_ID, as_of=resolved_as_of)
    actuals = {pd.Timestamp(r["timestamp"]).normalize(): float(r["value"]) for _, r in actual_df.iterrows()}

    engine = PythonForecastEngineDeltaGoverned(DEFAULT_SETTINGS)
    task_template = ForecastingTask(
        task_id="wti_oil_price_forecast",
        target_series_id=WTI_SERIES_ID,
        horizons=HORIZONS,
        frequency="B",
        description="WTI price forecast",
    )

    # Percentile tables are per-origin; compute each table once.
    tables: dict[str, dict[str, dict]] = {}
    for origin in sorted({str(p["as_of"])[:10] for p in predictions}):
        context = service.context(as_of=pd.Timestamp(origin))
        tables[origin] = {
            "dollar": compute_horizon_delta_percentiles(context, WTI_SERIES_ID, HORIZONS, log_returns=False),
            "log": compute_horizon_delta_percentiles(context, WTI_SERIES_ID, HORIZONS),
        }

    rows = []
    mismatches = 0
    for pred in predictions:
        meta = pred.get("metadata", {})
        ft = meta.get("forecast_transformation")
        policy = meta.get("policy_decision")
        ensemble_raw = meta.get("unadjusted_ensemble")
        assessment = meta.get("llm_context_assessment", {})
        diagnostics = meta.get("market_diagnostics", {})
        if not (ft and policy and ensemble_raw):
            continue

        origin = str(pred["as_of"])[:10]
        horizon = int(ft["horizon"])
        forecast_date = pd.Timestamp(pred["forecast_date"]).normalize()
        actual = actuals.get(forecast_date)
        if actual is None:
            continue

        ensemble = ModelHorizonForecast(
            horizon=horizon,
            forecast_date=str(forecast_date.date()),
            point_forecast=float(ensemble_raw["point_forecast"]),
            quantiles=_float_keys(ensemble_raw["quantiles"]),
        )
        decision = PolicyDecisionDeltaGoverned.model_validate(policy)
        novelty = assessment.get("incremental_novelty", "indeterminate")
        latest_price = diagnostics.get("latest_value")

        out = {}
        for label in ("dollar", "log"):
            engine._delta_percentiles = tables[origin][label]  # noqa: SLF001
            transformed = engine.transform(ensemble, decision, novelty, latest_price)
            out[label] = transformed

        # --- self-check: the dollar path must reproduce what is cached ---
        cached_final = _float_keys(ft["final_quantiles"])
        recomputed = out["dollar"].final_quantiles
        if any(abs(recomputed[q] - cached_final[q]) > 1e-6 for q in cached_final):
            mismatches += 1

        crps_old, in_old = _score(out["dollar"].final_quantiles, out["dollar"].final_point_forecast, actual, pred)
        crps_new, in_new = _score(out["log"].final_quantiles, out["log"].final_point_forecast, actual, pred)

        rows.append(
            {
                "origin": origin,
                "horizon": horizon,
                "rank": decision.center_action,
                "unc": decision.uncertainty_action,
                "actual": actual,
                "point_old": out["dollar"].final_point_forecast,
                "point_new": out["log"].final_point_forecast,
                "width_old": out["dollar"].final_quantiles[0.9] - out["dollar"].final_quantiles[0.1],
                "width_new": out["log"].final_quantiles[0.9] - out["log"].final_quantiles[0.1],
                "crps_old": crps_old,
                "crps_new": crps_new,
                "in_old": in_old,
                "in_new": in_new,
            }
        )

    df = pd.DataFrame(rows)
    print(f"scored {len(df)} predictions from {df['origin'].nunique()} origins\n")

    if mismatches:
        print(f"!! SELF-CHECK FAILED: {mismatches}/{len(df)} dollar-path recomputes do not match the cache.")
        print("   The reconstruction is wrong; the comparison below is NOT trustworthy.\n")
    else:
        print("self-check OK: dollar-path recompute reproduces the cached quantiles exactly.\n")

    print("=" * 78)
    print("PAIRED COMPARISON — same LLM responses, governor swapped")
    print("=" * 78)
    print(f"{'':16} {'DOLLAR (cached)':>18} {'LOG (new)':>18} {'change':>12}")
    for label, old_col, new_col, fmt in [
        ("mean CRPS", "crps_old", "crps_new", "{:.4f}"),
        ("80% coverage %", "in_old", "in_new", "{:.1f}"),
        ("mean width", "width_old", "width_new", "{:.2f}"),
    ]:
        old = df[old_col].mean() * (100 if "coverage" in label else 1)
        new = df[new_col].mean() * (100 if "coverage" in label else 1)
        print(f"{label:16} {fmt.format(old):>18} {fmt.format(new):>18} {new - old:>+12.4f}")

    print("\nby horizon:")
    for horizon in HORIZONS:
        sub = df[df["horizon"] == horizon]
        print(
            f"  h={horizon:>2}d  CRPS {sub['crps_old'].mean():7.4f} -> {sub['crps_new'].mean():7.4f}"
            f"   coverage {100 * sub['in_old'].mean():5.1f}% -> {100 * sub['in_new'].mean():5.1f}%"
            f"   width {sub['width_old'].mean():6.2f} -> {sub['width_new'].mean():6.2f}"
        )

    moved = df[df["rank"] != 0]
    print(f"\npredictions where the LLM actually took a position (rank != 0): {len(moved)}/{len(df)}")
    if len(moved):
        print(
            f"  CRPS {moved['crps_old'].mean():.4f} -> {moved['crps_new'].mean():.4f}"
            f"   coverage {100 * moved['in_old'].mean():.1f}% -> {100 * moved['in_new'].mean():.1f}%"
        )
    neutral = df[df["rank"] == 0]
    if len(neutral):
        print(f"neutral (rank == 0): {len(neutral)}")
        print(
            f"  CRPS {neutral['crps_old'].mean():.4f} -> {neutral['crps_new'].mean():.4f}"
            f"   coverage {100 * neutral['in_old'].mean():.1f}% -> {100 * neutral['in_new'].mean():.1f}%"
        )


if __name__ == "__main__":
    main()
