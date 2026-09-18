"""Fit the width scale (omega) on ensemble-only history; judge it on live runs; record proposals.

    history (ensemble_only) -> needed stretch per forecast -> omega per horizon (train)
        -> validate on held-out years -> ComparisonPolicy on each stream's live corpus
        -> proposals/<stream>/<id>.json   (recorded, never applied)

**What is fitted.** One multiplier per horizon that stretches P10..P90 around P50:
``q' = P50 + omega * (q - P50)`` -- exactly `CalibrationLayer.apply` with a centre
gain of 1. The centre does not move.

**How.** For every historical forecast, the *needed stretch* is how much wider the
range had to be for the realized price to land inside it::

    needed = (actual - P50) / (P90 - P50)   if actual >= P50
             (P50 - actual) / (P50 - P10)   otherwise

The omega at which 80% of forecasts would have covered is the 80th percentile of the
needed stretches. That is the proposed value, because the defect being fixed is
coverage (R1). The pinball-optimal omega over a 0.50-3.00 grid is computed beside it
as a cross-check: pinball scores every quantile level, so a large gap between the two
says the *shape* of the distribution is off, not just its width.

**Honesty about samples.** Weekly origins at h=10 and h=21 overlap, so the interval on
omega comes from a moving-block bootstrap (block = the number of weekly origins a
horizon spans), not from resampling rows as if they were independent.

**Who decides.** History *chooses* omega; live runs *judge* it. The candidate declares
``fitted_through`` = the end of the training window, so every live origin is holdout
and ``ComparisonPolicy``'s provenance rule (live_forward only) applies unchanged --
no config change is needed. A passed verdict is still only a proposal: nothing here
writes a calibration version.

No LLM, no network. Usage (from the repo root, after ``cfm_coach.history`` has run)::

    uv run python -m energy_oil_forecasting.cfm_coach.fit_width
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from energy_oil_forecasting.cfm_coach.history import HISTORY_DIR, PRICE_CACHE_DIRNAME, load_history
from energy_oil_forecasting.cfm_coach.outcomes import MAX_STALENESS_DAYS


TRAIN_THROUGH = date(2021, 12, 31)
COVERAGE_TARGET = 0.80
OMEGA_GRID = np.round(np.arange(0.50, 3.0001, 0.05), 2)
BOOTSTRAP_RESAMPLES = 2_000
BOOTSTRAP_SEED = 20260913
#: Origins whose outcome windows run through WTI's negative-price episode, reported as
#: a sensitivity check rather than silently dropped or silently kept.
COVID_WINDOW = (date(2020, 2, 15), date(2020, 6, 30))
#: The shared proposal is rounded to a readable step; the unrounded fit is recorded too.
OMEGA_ROUND = 0.05


# ---------------------------------------------------------------- data ------


def load_prices(history_dir: Path = HISTORY_DIR) -> pd.Series:
    """Realized WTI closes from the history run's own frozen cache, indexed by date."""
    frame = pd.read_parquet(history_dir / PRICE_CACHE_DIRNAME / "cl_f_adj_close_1d.parquet")
    series = pd.Series(
        pd.to_numeric(frame["value"], errors="coerce").to_numpy(),
        index=pd.to_datetime(frame["timestamp"]).dt.date,
    ).dropna()
    return series[~series.index.duplicated(keep="last")].sort_index()


def resolve_actual(prices: pd.Series, target: date) -> float | None:
    """Price on or just before `target`, under the same rules `OutcomeResolver` applies live."""
    if prices.empty or prices.index[-1] < target:
        return None
    on_or_before = prices[prices.index <= target]
    if on_or_before.empty or (target - on_or_before.index[-1]).days > MAX_STALENESS_DAYS:
        return None
    return float(on_or_before.iloc[-1])


def build_rows(history: list[dict[str, Any]], prices: pd.Series) -> tuple[pd.DataFrame, dict[str, int]]:
    """One row per (origin, horizon) with a complete ensemble and a resolved outcome."""
    counts = {"origins": len(history), "errored": 0, "degraded": 0, "unresolved_horizons": 0}
    rows: list[dict[str, Any]] = []
    for payload in history:
        if payload.get("status") != "ok":
            counts["errored"] += 1
            continue
        if payload.get("failed_models"):
            # A two-model ensemble is a different forecaster (IMPROVEMENTS.md A3).
            counts["degraded"] += 1
            continue
        diagnostics = payload.get("diagnostics") or {}
        vix = (diagnostics.get("covariate_latest_values") or {}).get("vix_level_l1b")
        for horizon_key, forecast in payload["forecasts"].items():
            quantiles = {float(level): float(value) for level, value in (forecast.get("ensemble") or {}).items()}
            target = forecast.get("forecast_date")
            actual = resolve_actual(prices, date.fromisoformat(target)) if target and quantiles else None
            if actual is None:
                counts["unresolved_horizons"] += 1
                continue
            rows.append(
                {
                    "cutoff": date.fromisoformat(payload["cutoff"]),
                    "horizon": int(horizon_key),
                    "actual": actual,
                    "p10": quantiles[0.1],
                    "p50": quantiles[0.5],
                    "p90": quantiles[0.9],
                    "last_value": diagnostics.get("latest_value"),
                    "vix": vix,
                    "quantiles": quantiles,
                }
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["horizon", "cutoff"]).reset_index(drop=True)
    return frame, counts


# ------------------------------------------------------------- the maths ----


def needed_stretch(rows: pd.DataFrame) -> np.ndarray:
    """Multiplier on each forecast's own half-width that would just have covered the outcome."""
    above = rows["actual"].to_numpy() >= rows["p50"].to_numpy()
    upper = (rows["p90"] - rows["p50"]).to_numpy()
    lower = (rows["p50"] - rows["p10"]).to_numpy()
    distance = np.abs(rows["actual"].to_numpy() - rows["p50"].to_numpy())
    half_width = np.where(above, upper, lower)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(half_width > 0, distance / half_width, np.where(distance > 0, np.inf, 0.0))


def coverage_at(rows: pd.DataFrame, omega: float) -> float:
    """80% coverage if every range were scaled by `omega` around its P50."""
    return float(np.mean(needed_stretch(rows) <= omega))


def _quantile_matrix(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    levels = np.array(sorted(rows["quantiles"].iloc[0]), dtype=float)
    matrix = np.array([[row[level] for level in levels] for row in rows["quantiles"]], dtype=float)
    return levels, matrix


def pinball_curve(rows: pd.DataFrame, grid: np.ndarray = OMEGA_GRID) -> np.ndarray:
    """Mean pinball loss over the quantile grid, for each candidate omega."""
    levels, matrix = _quantile_matrix(rows)
    centre = rows["p50"].to_numpy()[:, None]
    actual = rows["actual"].to_numpy()[:, None, None]
    scaled = centre[:, None, :] + grid[None, :, None] * (matrix[:, None, :] - centre[:, None, :])
    loss = (levels[None, None, :] - (actual < scaled)) * (actual - scaled)
    return loss.mean(axis=(0, 2))


def omega_for_coverage(stretch: np.ndarray, target: float = COVERAGE_TARGET) -> float:
    """Smallest scale at which `target` of the forecasts cover. Infinite stretches count as misses."""
    finite = np.sort(stretch)
    index = max(0, math.ceil(target * len(finite)) - 1)
    return float(finite[index])


def block_bootstrap_omega(
    stretch: np.ndarray,
    *,
    block: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    target: float = COVERAGE_TARGET,
) -> tuple[float, float]:
    """90% interval on the coverage omega, resampling contiguous blocks of origins.

    Rows are in cutoff order. Consecutive weekly origins share most of an h=21 outcome
    window, so resampling single rows would treat near-duplicates as independent and
    report a falsely tight interval.
    """
    n = len(stretch)
    block = max(1, min(block, n))
    starts = n - block + 1
    blocks_needed = math.ceil(n / block)
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples)
    for index in range(resamples):
        picks = rng.integers(0, starts, size=blocks_needed)
        sample = np.concatenate([stretch[start : start + block] for start in picks])[:n]
        estimates[index] = omega_for_coverage(sample, target)
    low, high = np.quantile(estimates, [0.05, 0.95])
    return float(low), float(high)


def round_omega(value: float, step: float = OMEGA_ROUND) -> float:
    return round(round(value / step) * step, 2)


def fit_horizon(rows: pd.DataFrame, horizon: int) -> dict[str, Any]:
    """Fit on the training window, validate on everything after it, and describe both."""
    train = rows[rows["cutoff"] <= TRAIN_THROUGH]
    test = rows[rows["cutoff"] > TRAIN_THROUGH]
    stretch = needed_stretch(train)
    omega = omega_for_coverage(stretch)
    curve = pinball_curve(train)
    omega_pinball = float(OMEGA_GRID[int(np.argmin(curve))])
    ci_low, ci_high = block_bootstrap_omega(stretch, block=math.ceil(horizon / 5) + 1)
    proposed = round_omega(omega)

    def evaluate(frame: pd.DataFrame) -> dict[str, Any]:
        if frame.empty:
            return {"n": 0}
        base_curve = pinball_curve(frame, np.array([1.0, proposed, omega_pinball]))
        return {
            "n": len(frame),
            "origins": frame["cutoff"].nunique(),
            "from": str(frame["cutoff"].min()),
            "through": str(frame["cutoff"].max()),
            "coverage_at_1": coverage_at(frame, 1.0),
            "coverage_at_proposed": coverage_at(frame, proposed),
            "coverage_at_pinball_omega": coverage_at(frame, omega_pinball),
            "pinball_at_1": float(base_curve[0]),
            "pinball_at_proposed": float(base_curve[1]),
            "pinball_at_pinball_omega": float(base_curve[2]),
            "mean_width_at_1": float((frame["p90"] - frame["p10"]).mean()),
            "mean_width_at_proposed": float(proposed * (frame["p90"] - frame["p10"]).mean()),
            "omega_needed_for_80": omega_for_coverage(needed_stretch(frame)),
        }

    no_covid = train[~train["cutoff"].between(*COVID_WINDOW)]
    return {
        "horizon": horizon,
        "omega_coverage_raw": omega,
        "omega_proposed": proposed,
        "omega_coverage_ci90": [ci_low, ci_high],
        "omega_pinball": omega_pinball,
        "omega_coverage_excluding_covid_window": omega_for_coverage(needed_stretch(no_covid)),
        "omega_full_history": omega_for_coverage(needed_stretch(rows)),
        "train": evaluate(train),
        "holdout": evaluate(test),
        "by_vix_quintile": vix_quintiles(train, rows, proposed),
    }


def vix_quintiles(train: pd.DataFrame, rows: pd.DataFrame, proposed: float) -> list[dict[str, Any]]:
    """How far one constant omega is from what each volatility regime needed. Diagnostic only."""
    usable = rows.dropna(subset=["vix"])
    if usable.empty or train["vix"].dropna().empty:
        return []
    edges = np.quantile(train["vix"].dropna(), [0.2, 0.4, 0.6, 0.8])
    labels = np.digitize(usable["vix"].to_numpy(), edges)
    result = []
    bounds = [-np.inf, *edges, np.inf]
    for quintile in range(5):
        frame = usable[labels == quintile]
        if frame.empty:
            continue
        result.append(
            {
                "quintile": quintile + 1,
                "vix_from": None if not np.isfinite(bounds[quintile]) else float(bounds[quintile]),
                "vix_to": None if not np.isfinite(bounds[quintile + 1]) else float(bounds[quintile + 1]),
                "n": len(frame),
                "omega_needed_for_80": omega_for_coverage(needed_stretch(frame)),
                "coverage_at_1": coverage_at(frame, 1.0),
                "coverage_at_proposed": coverage_at(frame, proposed),
            }
        )
    return result


# ------------------------------------------------------------ live gate -----


def judge_on_streams(width_scale: dict[int, float], *, candidate_id: str, rationale: str) -> list[dict[str, Any]]:
    """Run the candidate through `ComparisonPolicy` on every stream's own live corpus."""
    from energy_oil_forecasting.cfm_coach.ledger import CalibrationLedger  # noqa: PLC0415
    from energy_oil_forecasting.cfm_coach.outcomes import OutcomeResolver  # noqa: PLC0415
    from energy_oil_forecasting.cfm_coach.policy.comparison_policy import ComparisonPolicy  # noqa: PLC0415
    from energy_oil_forecasting.cfm_coach.replay import ReplayEngine  # noqa: PLC0415
    from energy_oil_forecasting.cfm_coach.run_store import RunRecordStore  # noqa: PLC0415
    from energy_oil_forecasting.cfm_coach.schemas import CalibrationLayer, Candidate  # noqa: PLC0415
    from energy_oil_forecasting.cfm_coach.streams import STREAMS  # noqa: PLC0415

    results = []
    for stream in STREAMS:
        settings = stream.coach_settings
        records = RunRecordStore(settings).load_all()
        resolution = OutcomeResolver(settings).resolve_all(records)
        incumbent = CalibrationLedger(settings).current(date.today())
        candidate = Candidate(
            candidate_id=candidate_id,
            parent_version=incumbent.version,
            layer=CalibrationLayer(width_scale=width_scale),
            fitted_through=TRAIN_THROUGH,
            rationale=rationale,
        )
        replay = ReplayEngine(settings, target=stream.target)
        replay.verify_all(records, strict=True)
        policy = ComparisonPolicy(settings, replay=replay)
        verdict = policy.evaluate(candidate, incumbent, records=records, resolution=resolution)

        # The gate judges the fitted value, but what a human would apply is the value
        # shrunk toward the incumbent -- which the gate never scores. Judge it as well, so
        # the recorded proposal says whether the value it recommends applying is itself
        # an improvement rather than assuming it inherits the fitted value's verdict.
        shrunk_candidate = Candidate(
            candidate_id=f"{candidate_id}__shrunk",
            parent_version=incumbent.version,
            layer=verdict.shrunk_layer or CalibrationLayer(),
            fitted_through=TRAIN_THROUGH,
            rationale=f"{rationale} Shrunk {verdict.shrinkage_factor:.0%} of the way from the incumbent.",
        )
        shrunk_verdict = policy.evaluate(shrunk_candidate, incumbent, records=records, resolution=resolution)

        results.append(
            {
                "stream": stream,
                "incumbent": incumbent,
                "candidate": candidate,
                "verdict": verdict,
                "shrunk_candidate": shrunk_candidate,
                "shrunk_verdict": shrunk_verdict,
                "live_by_horizon": live_by_horizon(records, resolution, replay, settings, incumbent, candidate),
                "live_by_horizon_shrunk": live_by_horizon(
                    records, resolution, replay, settings, incumbent, shrunk_candidate
                ),
            }
        )
    return results


def live_by_horizon(  # noqa: PLR0913 - the pieces of one stream's replay, passed explicitly rather than bundled
    records: list[Any],
    resolution: Any,
    replay: Any,
    settings: Any,
    incumbent: Any,
    candidate: Any,
) -> dict[str, dict[str, float]]:
    """Coverage, pinball and width per horizon, as published vs. under `candidate`, on live_forward rows.

    The detail a verdict aggregates away: a pooled pass can hide a horizon that got worse.
    """
    from energy_oil_forecasting.cfm_coach.scoring import ForecastScorer  # noqa: PLC0415

    by_id = {record.run_id: record for record in records}
    version = candidate.as_version("candidate", effective_from=incumbent.effective_from)
    scorer = ForecastScorer()
    detail: dict[int, dict[str, list[float]]] = {}
    for resolved in resolution.resolved:
        record = by_id[resolved.run_id]
        if record.provenance not in settings.fitting_eligible_provenance:
            continue
        before = scorer.score_recorded(record, resolved)
        after = scorer.score_replayed(
            replay.replay(record, version, horizons=(resolved.horizon,))[0], resolved, variant="candidate"
        )
        bucket = detail.setdefault(
            resolved.horizon, {"cov0": [], "cov1": [], "pin0": [], "pin1": [], "w0": [], "w1": []}
        )
        bucket["cov0"].append(before.covered_80)
        bucket["cov1"].append(after.covered_80)
        bucket["pin0"].append(before.pinball)
        bucket["pin1"].append(after.pinball)
        bucket["w0"].append(before.interval_width)
        bucket["w1"].append(after.interval_width)
    return {
        str(horizon): {
            "n": len(values["cov0"]),
            "coverage_before": float(np.mean(values["cov0"])),
            "coverage_after": float(np.mean(values["cov1"])),
            "pinball_before": float(np.mean(values["pin0"])),
            "pinball_after": float(np.mean(values["pin1"])),
            "mean_width_before": float(np.mean(values["w0"])),
            "mean_width_after": float(np.mean(values["w1"])),
        }
        for horizon, values in sorted(detail.items())
    }


# -------------------------------------------------------------- output ------


def _pct(value: float | None) -> str:
    return "--" if value is None else f"{value:.0%}"


def render_summary(fit: dict[str, Any], judged: list[dict[str, Any]]) -> str:
    lines = [
        f"# Width-scale (omega) proposal -- {fit['proposal_id']}",
        "",
        "**Status: recorded, not applied.** No calibration file was written.",
        "",
        f"History: {fit['history']['origins']} weekly origins {fit['history']['from']} -> {fit['history']['through']}, "
        f"{fit['history']['errored']} errored, {fit['history']['degraded']} degraded (a model failed; excluded). "
        f"Trained through {TRAIN_THROUGH}, validated after it. Prices through {fit['history']['price_data_through']}.",
        "",
        "## Fit on history (ensemble only, no LLM)",
        "",
        "| h | proposed omega | 90% CI | pinball-optimal | excl. Feb-Jun 2020 | coverage train @1 -> @omega | coverage holdout @1 -> @omega | holdout needed |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for item in fit["horizons"]:
        train, hold = item["train"], item["holdout"]
        lines.append(
            f"| {item['horizon']} | **{item['omega_proposed']:.2f}** | {item['omega_coverage_ci90'][0]:.2f}-{item['omega_coverage_ci90'][1]:.2f} "
            f"| {item['omega_pinball']:.2f} | {item['omega_coverage_excluding_covid_window']:.2f} "
            f"| {_pct(train['coverage_at_1'])} -> {_pct(train['coverage_at_proposed'])} "
            f"| {_pct(hold.get('coverage_at_1'))} -> {_pct(hold.get('coverage_at_proposed'))} "
            f"| {hold.get('omega_needed_for_80', float('nan')):.2f} |"
        )
    lines += ["", "### One omega across volatility regimes (all history, diagnostic)", ""]
    lines += ["| h | VIX quintile | n | needed omega | coverage @1 | coverage @proposed |", "|---|---|---|---|---|---|"]
    for item in fit["horizons"]:
        for bucket in item["by_vix_quintile"]:
            lines.append(
                f"| {item['horizon']} | Q{bucket['quintile']} | {bucket['n']} | {bucket['omega_needed_for_80']:.2f} "
                f"| {_pct(bucket['coverage_at_1'])} | {_pct(bucket['coverage_at_proposed'])} |"
            )
    lines += [
        "",
        "## Judged on live runs, per stream (ComparisonPolicy, pinball)",
        "",
        "Two values per stream: the fitted omega, and the value shrunk halfway toward 1.0 that a",
        "human would actually apply. The gate's own verdict covers only the first, so both are judged.",
        "",
    ]
    for item in judged:
        stream = item["stream"]
        fitted, shrunk = item["verdict"], item["shrunk_verdict"]
        lines += [
            f"### {stream.stream_id} ({stream.target.agent_id}, {stream.model})",
            "",
            "| value judged | width_scale | verdict | holdout origins | pinball | effect | p | coverage 80 |",
            "|---|---|---|---|---|---|---|---|",
            _verdict_row("fitted", item["candidate"].layer.width_scale, fitted),
            _verdict_row("shrunk (would be applied)", item["shrunk_candidate"].layer.width_scale, shrunk),
            "",
        ]
        for label, key in (("fitted", "live_by_horizon"), ("shrunk", "live_by_horizon_shrunk")):
            lines += [
                f"Live detail by horizon, {label}:",
                "",
                "| h | n | coverage before -> after | pinball before -> after | width before -> after |",
                "|---|---|---|---|---|",
            ]
            for horizon, row in item[key].items():
                lines.append(
                    f"| {horizon} | {row['n']} | {_pct(row['coverage_before'])} -> {_pct(row['coverage_after'])} "
                    f"| {row['pinball_before']:.3f} -> {row['pinball_after']:.3f} "
                    f"| {row['mean_width_before']:.2f} -> {row['mean_width_after']:.2f} |"
                )
            lines.append("")
        lines += ["Gate conditions for the shrunk value:", "", *[f"- {c.describe()}" for c in shrunk.conditions], ""]
    return "\n".join(lines) + "\n"


def _verdict_row(label: str, width_scale: dict[int, float], verdict: Any) -> str:
    scale = ", ".join(f"h{horizon}={value:.2f}" for horizon, value in sorted(width_scale.items()))
    head = f"| {label} | {scale} | {'PASSED' if verdict.passed else 'REJECTED'} | {verdict.n_holdout_origins} "
    if verdict.mean_paired_delta is None:
        return head + "| -- | -- | -- | -- |"
    return (
        head + f"| {verdict.incumbent_score:.3f} -> {verdict.candidate_score:.3f} | {verdict.effect_size_pct:+.1f}% "
        f"| {verdict.p_value:.3f} | {_pct(verdict.coverage_incumbent)} -> {_pct(verdict.coverage_candidate)} |"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--history", type=Path, default=HISTORY_DIR)
    parser.add_argument(
        "--out", type=Path, default=None, help="proposals directory (default: CoachSettings.proposals_dir)"
    )
    parser.add_argument("--no-gate", action="store_true", help="fit on history only; skip live judging and proposals")
    args = parser.parse_args(argv)

    from energy_oil_forecasting.cfm_coach.config import DEFAULT_SETTINGS  # noqa: PLC0415

    history = load_history(args.history)
    prices = load_prices(args.history)
    rows, counts = build_rows(history, prices)
    if rows.empty:
        raise SystemExit(f"no usable history rows in {args.history}; run cfm_coach.history first")

    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    proposal_id = f"width_scale_history_{stamp}"
    # int(): pandas hands back numpy int64, which neither json nor a CalibrationLayer key should carry.
    horizons = [
        fit_horizon(rows[rows["horizon"] == horizon], int(horizon)) for horizon in sorted(rows["horizon"].unique())
    ]
    fit = {
        "proposal_id": proposal_id,
        "created_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "method": "omega = 80th percentile of needed stretch on train origins; pinball-optimal shown as cross-check",
        "train_through": str(TRAIN_THROUGH),
        "history": {
            **counts,
            "rows": len(rows),
            "from": str(rows["cutoff"].min()),
            "through": str(rows["cutoff"].max()),
            "price_data_through": str(prices.index[-1]),
            "package_fingerprints": sorted({item.get("package_fingerprint") for item in history}),
        },
        "horizons": horizons,
    }
    width_scale = {item["horizon"]: item["omega_proposed"] for item in horizons}

    out = args.out or DEFAULT_SETTINGS.proposals_dir
    out.mkdir(parents=True, exist_ok=True)
    if args.no_gate:
        (out / f"{proposal_id}__history_fit.json").write_text(json.dumps(fit, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"width_scale": width_scale}, indent=2))
        return

    rationale = (
        f"Width scale fitted on {counts['origins']} weekly ensemble-only origins "
        f"({fit['history']['from']}..{TRAIN_THROUGH} train), omega = 80th percentile of needed stretch. "
        "History chose it; live_forward runs after fitted_through judge it."
    )
    judged = judge_on_streams(width_scale, candidate_id=proposal_id, rationale=rationale)

    (out / f"{proposal_id}__history_fit.json").write_text(json.dumps(fit, indent=2) + "\n", encoding="utf-8")
    for item in judged:
        stream, verdict = item["stream"], item["verdict"]
        stream_dir = out / stream.stream_id
        stream_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "proposal_id": proposal_id,
            "status": "recorded_not_applied",
            "stream_id": stream.stream_id,
            "agent_id": stream.target.agent_id,
            "agent_model": stream.model,
            "incumbent_version": item["incumbent"].version,
            "gate_passed": verdict.passed,
            "shrunk_gate_passed": item["shrunk_verdict"].passed,
            "proposed_width_scale": width_scale,
            "shrunk_width_scale_if_approved": verdict.shrunk_layer.width_scale if verdict.shrunk_layer else None,
            "candidate": item["candidate"].model_dump(mode="json"),
            "verdict": verdict.model_dump(mode="json"),
            "shrunk_candidate": item["shrunk_candidate"].model_dump(mode="json"),
            "shrunk_verdict": item["shrunk_verdict"].model_dump(mode="json"),
            "live_by_horizon": item["live_by_horizon"],
            "live_by_horizon_shrunk": item["live_by_horizon_shrunk"],
            "history_fit": f"../{proposal_id}__history_fit.json",
            "how_to_apply": (
                "Not applied. A human who approves writes a new calibration version in this stream's "
                "ledger (CalibrationLedger.save) with layer.width_scale = shrunk_width_scale_if_approved, "
                "parent_version = incumbent_version, source_proposal_id = proposal_id."
            ),
        }
        (stream_dir / f"{proposal_id}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    summary = render_summary(fit, judged)
    (out / f"{proposal_id}.md").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Recorded: {out / (proposal_id + '.md')}")


if __name__ == "__main__":
    main()


__all__ = [
    "TRAIN_THROUGH",
    "block_bootstrap_omega",
    "build_rows",
    "coverage_at",
    "fit_horizon",
    "needed_stretch",
    "omega_for_coverage",
    "pinball_curve",
    "resolve_actual",
]
