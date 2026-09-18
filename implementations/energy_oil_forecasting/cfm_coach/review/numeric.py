"""The numeric track: fit on history, price on live rows, judge with the gate, emit a candidate version.

Three levers the coach can express without touching the agent package:

- ``layer.width_scale`` (omega) -- ingested from `fit_width`'s proposals, never re-fitted here;
- ``layer.rw_anchor`` -- new: pull the distribution toward the last close; fitted on the
  ensemble-only history by minimising pinball on the training window;
- ``settings_overlay.ensemble_weights`` -- simplex weights fitted on stored component
  quantiles, priced by offline recombination (fidelity *approximate*: live LightGBM
  samples change with the RNG seed, which the weights are part of).

Policy/engine constants (`settings_overlay`) are priced by exact replay and judged
by the gate, but not fitted here: the grid is coarse and the value must be argued.

Every candidate is judged twice -- the fitted value and the shrunk value a human
would actually apply -- and the emitted candidate JSON carries the *shrunk*
value, with int-typed settings rounded so `to_agent_settings` accepts it. The
gate itself is unchanged; `effective_n` and `underpowered` are review-side labels.
"""

from __future__ import annotations

import fnmatch
import json
import math
import types
import typing
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from energy_oil_forecasting.cfm_coach.baselines import RandomWalkBaseline
from energy_oil_forecasting.cfm_coach.config import CoachSettings
from energy_oil_forecasting.cfm_coach.fit_width import (
    TRAIN_THROUGH,
    build_rows,
    load_prices,
)
from energy_oil_forecasting.cfm_coach.history import HISTORY_DIR, load_history
from energy_oil_forecasting.cfm_coach.ledger import CalibrationLedger
from energy_oil_forecasting.cfm_coach.policy.comparison_policy import ComparisonPolicy
from energy_oil_forecasting.cfm_coach.replay import ReplayEngine
from energy_oil_forecasting.cfm_coach.report import effective_n
from energy_oil_forecasting.cfm_coach.review.coached import coach_ledger, ensure_baseline
from energy_oil_forecasting.cfm_coach.review.collect import AGENT, StreamCorpus
from energy_oil_forecasting.cfm_coach.review.settings import DEFAULT_REVIEW_SETTINGS, ReviewSettings
from energy_oil_forecasting.cfm_coach.review.stats import CounterfactualEngine, Pricing, recombine
from energy_oil_forecasting.cfm_coach.schemas import CalibrationLayer, CalibrationVersion, Candidate, ComparisonVerdict
from energy_oil_forecasting.cfm_coach.scoring import pinball_loss
from energy_oil_forecasting.cfm_coach.streams import RunStream
from pydantic import BaseModel, ConfigDict, Field


ANCHOR_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
WEIGHT_STEP = 0.1
BLOCK_BOOTSTRAP_RESAMPLES = 500
BOOTSTRAP_SEED = 20260913


# ------------------------------------------------------------- history ----


def history_rows(history_dir: Path = HISTORY_DIR) -> tuple[pd.DataFrame, dict[str, int], pd.Series]:
    """`fit_width.build_rows` over the ensemble-only history, plus its frozen prices."""
    history = load_history(history_dir)
    prices = load_prices(history_dir)
    rows, counts = build_rows(history, prices)
    return rows, counts, prices


def component_rows(history: list[dict[str, Any]], prices: pd.Series) -> pd.DataFrame:
    """One row per (origin, horizon) with every component's quantiles and the outcome."""
    from energy_oil_forecasting.cfm_coach.fit_width import resolve_actual  # noqa: PLC0415

    rows = []
    for payload in history:
        if payload.get("status") != "ok" or payload.get("failed_models"):
            continue
        diagnostics = payload.get("diagnostics") or {}
        for horizon_key, forecast in payload["forecasts"].items():
            components = {
                name: {float(level): float(value) for level, value in q.items()}
                for name, q in (forecast.get("components") or {}).items()
            }
            target = forecast.get("forecast_date")
            actual = resolve_actual(prices, date.fromisoformat(target)) if target and components else None
            if actual is None:
                continue
            rows.append(
                {
                    "cutoff": date.fromisoformat(payload["cutoff"]),
                    "horizon": int(horizon_key),
                    "actual": actual,
                    "last_value": diagnostics.get("latest_value"),
                    "components": components,
                    "weights": dict((payload.get("settings") or {}).get("ensemble_weights") or {}),
                }
            )
    frame = pd.DataFrame(rows)
    return frame.sort_values(["horizon", "cutoff"]).reset_index(drop=True) if not frame.empty else frame


# ------------------------------------------------------------ rw anchor ----


def anchor_pinball_curve(rows: pd.DataFrame, grid: np.ndarray = ANCHOR_GRID) -> np.ndarray:
    """Mean pinball for each anchor: every quantile translated by ``a * (last_value - p50)``."""
    levels = np.array(sorted(rows["quantiles"].iloc[0]), dtype=float)
    matrix = np.array([[q[level] for level in levels] for q in rows["quantiles"]], dtype=float)
    shift = (rows["last_value"].to_numpy(dtype=float) - rows["p50"].to_numpy(dtype=float))[:, None, None]
    actual = rows["actual"].to_numpy(dtype=float)[:, None, None]
    shifted = matrix[:, None, :] + grid[None, :, None] * shift
    loss = (levels[None, None, :] - (actual < shifted)) * (actual - shifted)
    return loss.mean(axis=(0, 2))


def _block_bootstrap_argmin(
    rows: pd.DataFrame,
    curve_fn,
    grid: np.ndarray,
    *,
    block: int,
    resamples: int = BLOCK_BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    n = len(rows)
    block = max(1, min(block, n))
    starts = n - block + 1
    needed = math.ceil(n / block)
    rng = np.random.default_rng(seed)
    picks_out = np.empty(resamples)
    for i in range(resamples):
        picks = rng.integers(0, starts, size=needed)
        sample = pd.concat([rows.iloc[s : s + block] for s in picks]).iloc[:n]
        picks_out[i] = grid[int(np.argmin(curve_fn(sample)))]
    low, high = np.quantile(picks_out, [0.05, 0.95])
    return float(low), float(high)


def fit_rw_anchor(
    rows: pd.DataFrame, horizon: int, *, train_through: date = TRAIN_THROUGH, grid: np.ndarray = ANCHOR_GRID
) -> dict[str, Any]:
    """The anchor that minimises training pinball, with a block-bootstrap interval and a holdout check."""
    frame = rows[(rows["horizon"] == horizon) & rows["last_value"].notna()]
    train = frame[frame["cutoff"] <= train_through]
    test = frame[frame["cutoff"] > train_through]
    if train.empty:
        raise ValueError(f"no training rows for horizon {horizon} through {train_through}")
    curve = anchor_pinball_curve(train, grid)
    best = float(grid[int(np.argmin(curve))])
    ci_low, ci_high = _block_bootstrap_argmin(
        train, lambda f: anchor_pinball_curve(f, grid), grid, block=math.ceil(horizon / 5) + 1
    )

    def evaluate(f: pd.DataFrame) -> dict[str, Any]:
        if f.empty:
            return {"n": 0}
        c = anchor_pinball_curve(f, np.array([0.0, best, 1.0]))
        return {
            "n": len(f),
            "origins": int(f["cutoff"].nunique()),
            "from": str(f["cutoff"].min()),
            "through": str(f["cutoff"].max()),
            "pinball_at_0": float(c[0]),
            "pinball_at_best": float(c[1]),
            "pinball_at_1": float(c[2]),
            "gain_pct_vs_0": float((c[0] - c[1]) / c[0] * 100) if c[0] else 0.0,
        }

    return {
        "lever": "layer.rw_anchor",
        "horizon": horizon,
        "anchor": best,
        "anchor_ci90": [ci_low, ci_high],
        "curve": {str(a): float(v) for a, v in zip(grid, curve, strict=True)},
        "train": evaluate(train),
        "holdout": evaluate(test),
    }


# ------------------------------------------------------ ensemble weights ----


def simplex_grid(names: list[str], step: float = WEIGHT_STEP) -> list[dict[str, float]]:
    """Every non-negative weight vector on the `step` lattice summing to one."""
    steps = int(round(1 / step))
    out: list[dict[str, float]] = []

    def rec(prefix: list[int], remaining: int, k: int) -> None:
        if k == 1:
            out.append({n: v * step for n, v in zip(names, [*prefix, remaining], strict=True)})
            return
        for v in range(remaining + 1):
            rec([*prefix, v], remaining - v, k - 1)

    rec([], steps, len(names))
    return [{k: round(v, 3) for k, v in w.items()} for w in out]


def weights_pinball(rows: pd.DataFrame, weights: dict[str, float], *, prices: pd.Series | None = None) -> float:
    losses = []
    for row in rows.itertuples():
        components = dict(row.components)
        if "random_walk" in weights and weights["random_walk"] > 0:
            if prices is None or row.last_value is None:
                continue
            history = prices[prices.index < row.cutoff]
            rw = RandomWalkBaseline(history).quantiles(
                history,
                last_value=float(row.last_value),
                horizon=int(row.horizon),
                levels=sorted(next(iter(components.values()))),
            )
            if rw is None:
                continue
            components["random_walk"] = rw
        try:
            pooled = recombine(components, weights)
        except ValueError:
            continue
        losses.append(pinball_loss(pooled, float(row.actual)))
    return float(np.mean(losses)) if losses else float("nan")


def fit_ensemble_weights(
    rows: pd.DataFrame,
    horizon: int,
    *,
    train_through: date = TRAIN_THROUGH,
    with_rw: bool = False,
    prices: pd.Series | None = None,
    shrink_to_equal: float = 0.5,
) -> dict[str, Any]:
    """Grid-search simplex weights minimising training pinball; report equal weights beside them."""
    frame = rows[rows["horizon"] == horizon]
    train = frame[frame["cutoff"] <= train_through]
    test = frame[frame["cutoff"] > train_through]
    if train.empty:
        raise ValueError(f"no training rows for horizon {horizon} through {train_through}")
    names = sorted(next(iter(train["components"])).keys())
    if with_rw:
        names.append("random_walk")
    equal = {n: 1.0 / len(names) for n in names}
    scored = [(weights_pinball(train, w, prices=prices), w) for w in simplex_grid(names)]
    scored = [(p, w) for p, w in scored if not math.isnan(p)]
    best_pinball, best = min(scored, key=lambda t: t[0])
    shrunk = {n: equal[n] + shrink_to_equal * (best[n] - equal[n]) for n in names}
    return {
        "lever": "settings_overlay.ensemble_weights",
        "horizon": horizon,
        "members": names,
        "with_rw": with_rw,
        "best": best,
        "shrunk_toward_equal": shrunk,
        "equal": equal,
        "train": {
            "n": len(train),
            "origins": int(train["cutoff"].nunique()),
            "pinball_equal": weights_pinball(train, equal, prices=prices),
            "pinball_best": best_pinball,
            "pinball_shrunk": weights_pinball(train, shrunk, prices=prices),
        },
        "holdout": (
            {
                "n": len(test),
                "origins": int(test["cutoff"].nunique()),
                "pinball_equal": weights_pinball(test, equal, prices=prices),
                "pinball_best": weights_pinball(test, best, prices=prices),
                "pinball_shrunk": weights_pinball(test, shrunk, prices=prices),
            }
            if not test.empty
            else {"n": 0}
        ),
    }


# ------------------------------------------------------- fit_width ingest ----


def ingest_fit_width(proposals_dir: Path, stream_id: str) -> dict[str, Any] | None:
    """The latest `fit_width` proposal recorded for `stream_id`, if any."""
    stream_dir = proposals_dir / stream_id
    if not stream_dir.exists():
        return None
    paths = sorted(stream_dir.glob("width_scale_history_*.json"))
    if not paths:
        return None
    return json.loads(paths[-1].read_text(encoding="utf-8"))


# ------------------------------------------------------------ candidates ----


def denied(name: str, denylist: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in denylist)


def _is_int_field(annotation: Any) -> bool:
    if annotation is int:
        return True
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        return any(_is_int_field(a) for a in typing.get_args(annotation) if a is not type(None))
    return False


def round_overlay(overlay: dict[str, Any], settings_cls: type) -> dict[str, Any]:
    """Round int-typed fields after shrinkage so the overlay still validates (2 -> 3 shrinks to 2.5)."""
    out: dict[str, Any] = {}
    for name, value in overlay.items():
        field = settings_cls.model_fields.get(name)
        if field is not None and _is_int_field(field.annotation) and isinstance(value, float):
            out[name] = int(round(value))
        else:
            out[name] = value
    return out


def next_version_name(ledger: CalibrationLedger) -> str:
    versions = ledger.all_versions()
    highest = max(
        (int(v.version[1:]) for v in versions if v.version.startswith("v") and v.version[1:].isdigit()), default=0
    )
    return f"v{highest + 1:03d}"


def next_business_day(day: date) -> date:
    nxt = day + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt


class NumericVerdict(BaseModel):
    """One candidate, judged on one stream's live rows, priced against the random walk."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stream: str
    candidate_id: str
    lever: str
    fidelity: str
    incumbent_version: str
    fitted: ComparisonVerdict
    shrunk: ComparisonVerdict
    pricing_fitted: Pricing
    pricing_shrunk: Pricing
    effective_n: float
    underpowered: bool
    rw_dominance_after_shrunk: bool
    promotable: bool
    candidate_file: str | None = None
    fit: dict[str, Any] = Field(default_factory=dict)


def judge_candidate(
    corpus: StreamCorpus,
    stream: RunStream,
    candidate: Candidate,
    *,
    lever: str,
    fit: dict[str, Any] | None = None,
    settings: ReviewSettings = DEFAULT_REVIEW_SETTINGS,
    emit: bool = True,
) -> NumericVerdict:
    """Gate (unchanged) + pricing + review-side labels; optionally write the shrunk candidate JSON."""
    for name in candidate.settings_overlay:
        if denied(name, settings.overlay_denylist):
            raise ValueError(
                f"settings field {name!r} is on the numeric-track denylist: it changes the run, not the arithmetic"
            )
    ledger = coach_ledger(stream, settings)
    ensure_baseline(ledger)
    incumbent = ledger.current(date.today())
    coach_settings = CoachSettings(target_agent=stream.target.agent_id, calibration_dir=ledger.directory)
    replay = ReplayEngine(coach_settings, target=stream.target)
    policy = ComparisonPolicy(coach_settings, replay=replay)
    fitted = policy.evaluate(candidate, incumbent, records=corpus.records, resolution=corpus.resolution)

    shrunk_overlay = round_overlay(fitted.shrunk_settings_overlay, stream.target.settings_cls)
    shrunk_candidate = Candidate(
        candidate_id=f"{candidate.candidate_id}__shrunk",
        parent_version=incumbent.version,
        settings_overlay=shrunk_overlay,
        layer=fitted.shrunk_layer or CalibrationLayer(),
        fitted_through=candidate.fitted_through,
        rationale=f"{candidate.rationale} Shrunk {fitted.shrinkage_factor:.0%} of the way from the incumbent.",
    )
    shrunk = policy.evaluate(shrunk_candidate, incumbent, records=corpus.records, resolution=corpus.resolution)

    engine = CounterfactualEngine(corpus, settings, coach_settings)
    fidelity = "exact"
    if "ensemble_weights" in candidate.settings_overlay:
        fidelity = "approximate_ensemble_recombination"
        pricing_fitted = engine.ensemble_reweight(candidate.settings_overlay["ensemble_weights"])
        pricing_shrunk = engine.ensemble_reweight(shrunk_overlay["ensemble_weights"])
    else:
        pricing_fitted = engine.calibration(layer=candidate.layer, settings_overlay=candidate.settings_overlay)
        pricing_shrunk = engine.calibration(layer=shrunk_candidate.layer, settings_overlay=shrunk_overlay)

    agent_cards = [c for (_, _, v), c in corpus.scores.items() if v == AGENT]
    n_eff = effective_n(agent_cards)
    pooled = pricing_shrunk.pooled
    dominance = bool(pooled and pooled.rw_dominance_after)
    verdict = NumericVerdict(
        stream=stream.stream_id,
        candidate_id=candidate.candidate_id,
        lever=lever,
        fidelity=fidelity,
        incumbent_version=incumbent.version,
        fitted=fitted,
        shrunk=shrunk,
        pricing_fitted=pricing_fitted,
        pricing_shrunk=pricing_shrunk,
        effective_n=n_eff,
        underpowered=n_eff < settings.min_effective_n_powered,
        rw_dominance_after_shrunk=dominance,
        promotable=bool(shrunk.passed and dominance),
        fit=fit or {},
    )
    if emit:
        path = emit_candidate(stream, shrunk_candidate, incumbent, settings, ledger=ledger)
        verdict = verdict.model_copy(update={"candidate_file": str(path)})
    return verdict


def emit_candidate(
    stream: RunStream,
    candidate: Candidate,
    incumbent: CalibrationVersion,
    settings: ReviewSettings = DEFAULT_REVIEW_SETTINGS,
    *,
    ledger: CalibrationLedger | None = None,
) -> Path:
    """Write a ready-to-save `CalibrationVersion` JSON under ``review/candidates/<stream>/``. Never into a ledger."""
    ledger = ledger or coach_ledger(stream, settings)
    version = candidate.as_version(next_version_name(ledger), effective_from=next_business_day(date.today()))
    # Prove it applies before anyone is offered it.
    ledger.to_agent_settings(version, base=stream.base_settings, target=stream.target)
    out_dir = settings.candidates_dir / stream.stream_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{candidate.candidate_id}__{version.version}.json"
    path.write_text(json.dumps(version.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def describe_verdict(v: NumericVerdict) -> str:
    p = v.pricing_shrunk.pooled
    head = (
        f"{v.stream} {v.candidate_id} [{v.lever}, {v.fidelity}]: gate fitted {'PASS' if v.fitted.passed else 'FAIL'} / shrunk "
        f"{'PASS' if v.shrunk.passed else 'FAIL'}; effective_n {v.effective_n:.1f}{' (underpowered)' if v.underpowered else ''}; "
        f"promotable {v.promotable}"
    )
    if p is None:
        return head
    rows = [
        head,
        f"  pooled: pinball {p.pinball_before:.3f} -> {p.pinball_after:.3f} (rw {p.pinball_rw:.3f}, gain vs rw {p.gain_vs_rw_pct:+.1f}%), "
        f"coverage {p.coverage_before:.0%} -> {p.coverage_after:.0%}, beats rw after: {p.rw_dominance_after}",
    ]
    for h in v.pricing_shrunk.by_horizon:
        rows.append(
            f"  h{h.horizon}: n={h.n} cutoffs={h.distinct_cutoffs} pinball {h.pinball_before:.3f} -> {h.pinball_after:.3f} rw {h.pinball_rw:.3f}; coverage {h.coverage_before:.0%} -> {h.coverage_after:.0%}"
        )
    return "\n".join(rows)


__all__ = [
    "ANCHOR_GRID",
    "NumericVerdict",
    "anchor_pinball_curve",
    "component_rows",
    "denied",
    "describe_verdict",
    "emit_candidate",
    "fit_ensemble_weights",
    "fit_rw_anchor",
    "history_rows",
    "ingest_fit_width",
    "judge_candidate",
    "next_business_day",
    "next_version_name",
    "round_overlay",
    "simplex_grid",
    "weights_pinball",
]
