"""Stage C: code-only tables over the corpus -- strata, episodes, counterfactuals.

Everything here is arithmetic the LLM is never asked to do. Synthesis calls these
through tools and quotes the numbers; the numbers themselves come from here.

- **strata** -- one row per (run, horizon) with features and scores; grouped by
  any feature into counts, *distinct cutoffs*, `effective_n`, and the agent's gap
  to the random walk. Totals reconcile with the scoreboard by construction.
- **episodes** -- the price path cut into same-direction moves by a zig-zag with
  a threshold taken from the random walk's own h10 spread, so a two-week rally
  is one episode however many daily cutoffs sit inside it.
- **independent windows** -- the support unit for outcome-dependent evidence:
  cutoffs at least ``h`` business days apart share no outcome window.
- **counterfactuals** -- named, priced operations on stored records: what
  pinball (vs the random walk), coverage and direction would have been under a
  different constant, weight, anchor, or action. The one pricing currency both
  proposal tracks use.
"""

from __future__ import annotations

import fnmatch
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from energy_oil_forecasting.cfm_coach.baselines import RandomWalkBaseline
from energy_oil_forecasting.cfm_coach.config import CoachSettings
from energy_oil_forecasting.cfm_coach.replay import ReplayEngine
from energy_oil_forecasting.cfm_coach.report import AGENT, ENSEMBLE, RANDOM_WALK, effective_n, hit_rate_needed
from energy_oil_forecasting.cfm_coach.review.collect import StreamCorpus
from energy_oil_forecasting.cfm_coach.review.settings import DEFAULT_REVIEW_SETTINGS, ReviewSettings
from energy_oil_forecasting.cfm_coach.schemas import (
    CalibrationLayer,
    CalibrationVersion,
    HorizonRecord,
    Quantiles,
    RunRecord,
    ScoreCard,
)
from energy_oil_forecasting.cfm_coach.scoring import FLAT_EPS, ForecastScorer
from energy_oil_forecasting.cfm_coach.targets import target_for
from pydantic import BaseModel, ConfigDict


# ------------------------------------------------------------------ strata ----

STRATA_FEATURES = (
    "vol_tercile",
    "trend_sign",
    "granted_center",
    "granted_uncertainty",
    "tier",
    "eligible",
    "novelty",
    "confidence_bin",
    "conflict",
    "zeroed",
    "acted",
    "component_bias_sign",
    "lgbm_below_last_close",
    "width_vs_needed_bin",
    "disagreement_bin",
    "vix_quintile",
)


def _tercile(value: float | None, edges: tuple[float, float]) -> str | None:
    if value is None or not math.isfinite(value):
        return None
    return "low" if value < edges[0] else "high" if value >= edges[1] else "mid"


def _bin(value: float | None, edges: list[float], labels: list[str]) -> str | None:
    if value is None or not math.isfinite(value):
        return None
    return labels[int(np.searchsorted(edges, value, side="right"))]


def strata_frame(corpus: StreamCorpus, *, vix_edges: list[float] | None = None) -> pd.DataFrame:
    """One row per resolved (run, horizon): features on the left, scores on the right."""
    rows: list[dict[str, Any]] = []
    vols = [r.diagnostics.get("realized_volatility_21b") for r in corpus.records]
    vols = [float(v) for v in vols if v is not None]
    vol_edges = tuple(np.quantile(vols, [1 / 3, 2 / 3])) if len(vols) >= 3 else (0.0, float("inf"))
    for record in corpus.records:
        proposed = {int(a["horizon"]): a for a in record.assessment.get("horizon_actions", [])}
        diag = record.diagnostics
        vix = (diag.get("covariate_latest_values") or {}).get("vix_level_l1b")
        for item in record.forecasts:
            agent = corpus.score(record.run_id, item.horizon, AGENT)
            ens = corpus.score(record.run_id, item.horizon, ENSEMBLE)
            rw = corpus.score(record.run_id, item.horizon, RANDOM_WALK)
            if agent is None or ens is None or rw is None:
                continue
            decision = item.policy_decision or {}
            action = proposed.get(item.horizon, {})
            last = float(diag["latest_value"]) if diag.get("latest_value") is not None else None
            overlay = item.final_point_forecast - item.ensemble_quantiles[0.5]
            lgbm = item.component_quantiles.get("lightgbm", {}).get(0.5)
            dis = record.audit_signals.get("model_disagreement_std") or {}
            dis_value = dis.get(str(item.horizon)) if isinstance(dis, dict) else None
            width = agent.interval_width
            from energy_oil_forecasting.cfm_coach.review.cards import omega_needed  # noqa: PLC0415

            needed = omega_needed(dict(item.final_quantiles), agent.realized_value)
            rows.append(
                {
                    "stream": corpus.stream_id,
                    "run_id": record.run_id,
                    "cutoff": record.cutoff,
                    "horizon": item.horizon,
                    "provenance": record.provenance,
                    # features
                    "vol_tercile": _tercile(diag.get("realized_volatility_21b"), vol_edges),
                    "trend_sign": None
                    if diag.get("return_21b") is None
                    else ("up" if float(diag["return_21b"]) > 0 else "down"),
                    "granted_center": str(decision.get("center_action", "--")),
                    "granted_uncertainty": str(decision.get("uncertainty_action", "--")),
                    "proposed_center": str(action.get("center_action", "--")),
                    "tier": str(decision.get("evidence_tier", "none")),
                    "eligible": bool(decision.get("eligible", False)),
                    "novelty": str(record.assessment.get("incremental_novelty", "--")),
                    "confidence_bin": _bin(record.assessment.get("confidence"), [0.5, 0.8], ["low", "mid", "high"]),
                    "conflict": bool(record.assessment.get("material_evidence_conflict", False)),
                    "zeroed": action.get("center_action") not in (None, "no_change")
                    and decision.get("center_action") == "no_change",
                    "acted": abs(overlay) > FLAT_EPS
                    or float(item.forecast_transformation.get("uncertainty_multiplier", 1.0) or 1.0) != 1.0,
                    "component_bias_sign": None
                    if last is None
                    else ("down" if item.ensemble_quantiles[0.5] < last else "up"),
                    "lgbm_below_last_close": None if (lgbm is None or last is None) else bool(lgbm < last),
                    "width_vs_needed_bin": _bin(needed, [1.0, 1.5], ["covered", "near", "far"]),
                    "disagreement_bin": _bin(
                        None if dis_value is None or width <= 0 else float(dis_value) / width,
                        [0.1, 0.25],
                        ["low", "mid", "high"],
                    ),
                    "vix_quintile": _bin(None if vix is None else float(vix), vix_edges, ["q1", "q2", "q3", "q4", "q5"])
                    if vix_edges
                    else None,
                    # scores
                    "pinball_agent": agent.pinball,
                    "pinball_ensemble": ens.pinball,
                    "pinball_rw": rw.pinball,
                    "gap_vs_rw": agent.pinball - rw.pinball,
                    "base_term": ens.pinball - rw.pinball,
                    "overlay_term": agent.pinball - ens.pinball,
                    "covered_agent": float(agent.covered_80),
                    "covered_rw": float(rw.covered_80),
                    "overlay_usd": overlay,
                    "direction_call": agent.direction_call,
                    "direction_outcome": agent.direction_outcome,
                    "direction_hit": agent.direction_hit,
                }
            )
    return pd.DataFrame(rows)


def strata_table(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Aggregate a strata frame; every row carries distinct cutoffs and `effective_n`."""
    if frame.empty:
        return pd.DataFrame()
    out = []
    for keys, group in frame.groupby(by, dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        called = group[group["direction_hit"].notna()]
        n_eff = sum(len(set(g["cutoff"])) / h for h, g in group.groupby("horizon"))
        out.append(
            {
                **dict(zip(by, keys, strict=True)),
                "rows": len(group),
                "distinct_cutoffs": group["cutoff"].nunique(),
                "effective_n": float(n_eff),
                "pinball_agent": float(group["pinball_agent"].mean()),
                "pinball_rw": float(group["pinball_rw"].mean()),
                "gap_vs_rw": float(group["gap_vs_rw"].mean()),
                "base_term": float(group["base_term"].mean()),
                "overlay_term": float(group["overlay_term"].mean()),
                "coverage_agent": float(group["covered_agent"].mean()),
                "coverage_rw": float(group["covered_rw"].mean()),
                "direction_calls": int(len(called)),
                "hit_rate": float(called["direction_hit"].mean()) if len(called) else None,
                "always_up_rate": float((called["direction_outcome"] == 1).mean()) if len(called) else None,
            }
        )
    return pd.DataFrame(out)


# --------------------------------------------------------------- episodes ----


class Episode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    episode_id: str
    start: date
    end: date
    direction: int
    start_price: float
    end_price: float
    threshold: float

    @property
    def magnitude(self) -> float:
        return self.end_price - self.start_price

    def covers(self, day: date) -> bool:
        return self.start <= day <= self.end


def episode_threshold(prices: pd.Series, at: date, *, horizon: int = 10, window: int = 520) -> float | None:
    """The random walk's h-step P90 half-width at `at`: a move smaller than this is noise, not an episode."""
    history = prices[prices.index < at]
    if history.empty:
        return None
    baseline = RandomWalkBaseline(history, window=window)
    quantiles = baseline.quantiles(history, last_value=float(history.iloc[-1]), horizon=horizon, levels=[0.5, 0.9])
    if quantiles is None:
        return None
    return float(quantiles[0.9] - quantiles[0.5])


def find_episodes(prices: pd.Series, *, start: date, end: date | None = None, horizon: int = 10) -> list[Episode]:
    """Zig-zag the price path: an episode runs from one extreme to the next reversal larger than the threshold."""
    series = prices[(prices.index >= start)]
    if end is not None:
        series = series[series.index <= end]
    if len(series) < 2:
        return []
    threshold = episode_threshold(prices, start, horizon=horizon)
    if threshold is None or threshold <= 0:
        threshold = float(series.std() or 1.0)

    episodes: list[Episode] = []
    dates = list(series.index)
    values = series.to_numpy(dtype=float)
    anchor_i = 0
    direction = 0
    extreme_i = 0
    for i in range(1, len(values)):
        move = values[i] - values[anchor_i]
        if direction == 0:
            if abs(move) >= threshold:
                direction = 1 if move > 0 else -1
                extreme_i = i
            continue
        if (values[i] - values[extreme_i]) * direction > 0:
            extreme_i = i
            continue
        if (values[extreme_i] - values[i]) * direction >= threshold:
            episodes.append(_episode(dates, values, anchor_i, extreme_i, direction, threshold, len(episodes)))
            anchor_i, extreme_i, direction = extreme_i, i, -direction
    if direction != 0:
        episodes.append(_episode(dates, values, anchor_i, extreme_i, direction, threshold, len(episodes)))
    return episodes


def _episode(
    dates: list[date], values: np.ndarray, a: int, b: int, direction: int, threshold: float, n: int
) -> Episode:
    return Episode(
        episode_id=f"E{n + 1:03d}_{dates[a].isoformat()}_{'up' if direction > 0 else 'down'}",
        start=dates[a],
        end=dates[b],
        direction=direction,
        start_price=float(values[a]),
        end_price=float(values[b]),
        threshold=threshold,
    )


def episode_for(episodes: list[Episode], *, cutoff: date, forecast_date: date) -> Episode | None:
    """The episode an outcome window belongs to: the one covering most of ``[cutoff, forecast_date]``."""
    best, best_overlap = None, 0
    for ep in episodes:
        lo, hi = max(ep.start, cutoff), min(ep.end, forecast_date)
        overlap = (hi - lo).days + 1 if hi >= lo else 0
        if overlap > best_overlap:
            best, best_overlap = ep, overlap
    return best


def independent_windows(cutoffs: list[date], horizon: int) -> int:
    """Largest set of cutoffs pairwise >= `horizon` business days apart (greedy on sorted dates is optimal)."""
    picked: list[date] = []
    for cutoff in sorted(set(cutoffs)):
        if not picked or np.busday_count(picked[-1], cutoff) >= horizon:
            picked.append(cutoff)
    return len(picked)


# ------------------------------------------------------- counterfactuals ----


class HorizonPricing(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    horizon: int
    n: int
    distinct_cutoffs: int
    pinball_before: float
    pinball_after: float
    pinball_rw: float
    gain_pct: float
    gain_vs_rw_pct: float
    rw_dominance_before: bool
    rw_dominance_after: bool
    coverage_before: float
    coverage_after: float
    hit_rate_before: float | None
    hit_rate_after: float | None
    always_up_rate: float | None
    hit_rate_needed: float | None


class Pricing(BaseModel):
    """What one counterfactual would have done, per horizon and pooled, on live_forward resolved rows."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    op: str
    params: dict[str, Any]
    fidelity: str
    stream: str
    by_horizon: list[HorizonPricing]
    pooled: HorizonPricing | None
    #: (run_id, horizon) -> pinball delta before - after; positive is an improvement.
    deltas: dict[str, float]


def _summarise(h: int, before: list[ScoreCard], after: list[ScoreCard], rw: list[ScoreCard]) -> HorizonPricing:
    pb = float(np.mean([c.pinball for c in before]))
    pa = float(np.mean([c.pinball for c in after]))
    pr = float(np.mean([c.pinball for c in rw]))
    called_b = [c for c in before if c.direction_hit is not None]
    called_a = [c for c in after if c.direction_hit is not None]
    n_eff = effective_n(before)
    ups = [c for c in called_a if c.direction_outcome == 1]
    always_up = len(ups) / len(called_a) if called_a else None
    return HorizonPricing(
        horizon=h,
        n=len(before),
        distinct_cutoffs=len({c.cutoff for c in before}),
        pinball_before=pb,
        pinball_after=pa,
        pinball_rw=pr,
        gain_pct=(pb - pa) / pb * 100 if pb else 0.0,
        gain_vs_rw_pct=(pr - pa) / pr * 100 if pr else 0.0,
        rw_dominance_before=pb < pr,
        rw_dominance_after=pa < pr,
        coverage_before=float(np.mean([c.covered_80 for c in before])),
        coverage_after=float(np.mean([c.covered_80 for c in after])),
        hit_rate_before=(sum(bool(c.direction_hit) for c in called_b) / len(called_b)) if called_b else None,
        hit_rate_after=(sum(bool(c.direction_hit) for c in called_a) / len(called_a)) if called_a else None,
        always_up_rate=always_up,
        hit_rate_needed=hit_rate_needed(n_eff, always_up) if always_up is not None else None,
    )


#: Settings fields the exact replay cannot express: they shape the numerical base
#: (component models and their combination), which the replay reads back from the
#: stored ensemble quantiles instead of recomputing.
REPLAY_BLIND_FIELDS: frozenset[str] = frozenset(
    {"model_num_samples", "lightgbm_lags", "lightgbm_covariate_lags", "kalman_dim_x", "ensemble_weights"}
)


@dataclass
class CounterfactualEngine:
    """Price a change on a stream's live_forward resolved rows. No LLM, no network."""

    corpus: StreamCorpus
    review_settings: ReviewSettings = DEFAULT_REVIEW_SETTINGS
    coach_settings: CoachSettings | None = None
    scorer: ForecastScorer = field(default_factory=ForecastScorer)

    def __post_init__(self) -> None:
        self.target = target_for(self.corpus.agent_id)
        self.coach_settings = self.coach_settings or CoachSettings(target_agent=self.corpus.agent_id)
        self.replay = ReplayEngine(self.coach_settings, target=self.target)
        self._identity = CalibrationVersion(version="identity", effective_from=date(2004, 1, 1))

    # -- rows -------------------------------------------------------------

    def _rows(self, horizons: tuple[int, ...] | None = None) -> list[tuple[RunRecord, int]]:
        out = []
        for record in self.corpus.records:
            if record.provenance not in self.coach_settings.fitting_eligible_provenance:
                continue
            for h in record.horizons:
                if horizons is not None and h not in horizons:
                    continue
                if (
                    self.corpus.resolved(record.run_id, h) is not None
                    and self.corpus.score(record.run_id, h, RANDOM_WALK) is not None
                ):
                    out.append((record, h))
        return out

    def _last(self, record: RunRecord) -> float | None:
        v = record.diagnostics.get("latest_value")
        return float(v) if v is not None else None

    def _score_quantiles(self, record: RunRecord, h: int, point: float, quantiles: Quantiles) -> ScoreCard:
        return self.scorer.score_quantiles(
            quantiles,
            point_forecast=point,
            resolved=self.corpus.resolved(record.run_id, h),
            cutoff_variant="counterfactual",
            calibration_version="counterfactual",
            last_value=self._last(record),
        )

    def _price(self, op: str, params: dict[str, Any], fidelity: str, after_fn, horizons=None) -> Pricing:
        by_h: dict[int, tuple[list, list, list]] = {}
        deltas: dict[str, float] = {}
        for record, h in self._rows(horizons):
            before = self.corpus.score(record.run_id, h, AGENT)
            rw = self.corpus.score(record.run_id, h, RANDOM_WALK)
            after = after_fn(record, h)
            b, a, r = by_h.setdefault(h, ([], [], []))
            b.append(before), a.append(after), r.append(rw)
            deltas[f"{record.run_id}|{h}"] = before.pinball - after.pinball
        rows = [_summarise(h, *by_h[h]) for h in sorted(by_h)]
        pooled = None
        if by_h:
            allb = [c for b, _, _ in by_h.values() for c in b]
            alla = [c for _, a, _ in by_h.values() for c in a]
            allr = [c for _, _, r in by_h.values() for c in r]
            pooled = _summarise(0, allb, alla, allr)
        return Pricing(
            op=op,
            params=params,
            fidelity=fidelity,
            stream=self.corpus.stream_id,
            by_horizon=rows,
            pooled=pooled,
            deltas=deltas,
        )

    # -- operations ---------------------------------------------------------

    def overlay_zero(self, horizons=None) -> Pricing:
        return self._price("overlay_zero", {}, "exact", lambda r, h: self.corpus.score(r.run_id, h, ENSEMBLE), horizons)

    def rw_centre_keep_width(self, a: float, horizons=None) -> Pricing:
        def after(record: RunRecord, h: int) -> ScoreCard:
            item = record.horizon(h)
            last = self._last(record)
            delta = a * (last - item.final_point_forecast)
            q = {level: v + delta for level, v in item.final_quantiles.items()}
            return self._score_quantiles(record, h, item.final_point_forecast + delta, q)

        return self._price("rw_centre_keep_width", {"a": a}, "exact", after, horizons)

    def calibration(
        self, *, layer: CalibrationLayer | None = None, settings_overlay: dict[str, Any] | None = None, horizons=None
    ) -> Pricing:
        for name in settings_overlay or {}:
            if name in REPLAY_BLIND_FIELDS:
                raise ValueError(
                    f"{name!r} is not reproduced by exact replay (the replay re-runs the policy on the stored "
                    "ensemble); price ensemble weights with `ensemble_reweight`, and model hyperparameters "
                    "through a history_variant brief"
                )
            if any(fnmatch.fnmatch(name, pattern) for pattern in self.review_settings.overlay_denylist):
                raise ValueError(f"{name!r} is on the overlay denylist and cannot be tuned by the coach")
        version = CalibrationVersion(
            version="counterfactual",
            effective_from=date(2004, 1, 1),
            layer=layer or CalibrationLayer(),
            settings_overlay=settings_overlay or {},
        )

        def after(record: RunRecord, h: int) -> ScoreCard:
            replayed = self.replay.replay(record, version, horizons=(h,))[0]
            return self.scorer.score_replayed(
                replayed,
                self.corpus.resolved(record.run_id, h),
                variant="counterfactual",
                last_value=self._last(record),
            )

        params = {"layer": (layer or CalibrationLayer()).model_dump(), "settings_overlay": settings_overlay or {}}
        return self._price("calibration", params, "exact", after, horizons)

    def ensemble_reweight(self, weights: dict[str, float], *, with_rw: bool = False, horizons=None) -> Pricing:
        def after(record: RunRecord, h: int) -> ScoreCard:
            patched = reweight_record(record, weights, baseline=self.corpus.baseline if with_rw else None)
            replayed = self.replay.replay(patched, self._identity, horizons=(h,))[0]
            return self.scorer.score_replayed(
                replayed,
                self.corpus.resolved(record.run_id, h),
                variant="counterfactual",
                last_value=self._last(record),
            )

        return self._price(
            "ensemble_reweight",
            {"weights": weights, "with_rw": with_rw},
            "approximate_ensemble_recombination",
            after,
            horizons,
        )

    def action_override(self, patch: dict[str, Any], horizons=None) -> Pricing:
        """Replay with a patched assessment (e.g. every horizon `no_change`, or a fixed novelty)."""

        def after(record: RunRecord, h: int) -> ScoreCard:
            assessment = {**record.assessment, **patch}
            if "horizon_actions_patch" in patch:
                assessment.pop("horizon_actions_patch")
                assessment["horizon_actions"] = [
                    {**a, **patch["horizon_actions_patch"]} if int(a["horizon"]) == h else a
                    for a in record.assessment["horizon_actions"]
                ]
            patched = record.model_copy(update={"assessment": assessment})
            replayed = self.replay.replay(patched, self._identity, horizons=(h,))[0]
            return self.scorer.score_replayed(
                replayed,
                self.corpus.resolved(record.run_id, h),
                variant="counterfactual",
                last_value=self._last(record),
            )

        return self._price("action_override", {"patch": patch}, "counterfactual_action", after, horizons)


# ------------------------------------------------------------ recombiner ----


def recombine(components: dict[str, Quantiles], weights: dict[str, float]) -> Quantiles:
    """`CfmEnsemblePredictor.combine` on stored component quantiles: weighted level-by-level pool, cummax."""
    active = [name for name in components if float(weights.get(name, 0.0)) > 0]
    if not active:
        raise ValueError("no positively weighted component present")
    total = sum(float(weights[name]) for name in active)
    levels = set.intersection(*(set(components[name]) for name in active))
    pooled = {
        level: sum(float(weights[name]) / total * components[name][level] for name in active)
        for level in sorted(levels)
    }
    previous = float("-inf")
    for level in sorted(pooled):
        pooled[level] = max(previous, pooled[level])
        previous = pooled[level]
    return pooled


def reweight_record(
    record: RunRecord, weights: dict[str, float], *, baseline: RandomWalkBaseline | None = None
) -> RunRecord:
    """A copy of `record` whose ensemble is recombined from its stored components under `weights`.

    ``random_walk`` may appear in `weights` only when a baseline is given; it is
    built from the record's own visible history. The final quantiles are left as
    recorded (they are recomputed by replay), so the copy is not replay-faithful
    and must never be stored as a record.
    """
    forecasts: list[HorizonRecord] = []
    for item in record.forecasts:
        components = {name: dict(q) for name, q in item.component_quantiles.items()}
        if "random_walk" in weights and float(weights["random_walk"]) > 0:
            if baseline is None:
                raise ValueError("a random_walk weight needs a RandomWalkBaseline")
            built = baseline.for_record(record, item.horizon)
            if built is None:
                raise ValueError(f"random-walk quantiles unavailable for {record.run_id} h={item.horizon}")
            components["random_walk"] = dict(built[1])
        pooled = recombine(components, weights)
        forecasts.append(item.model_copy(update={"ensemble_quantiles": pooled, "ensemble_point_forecast": pooled[0.5]}))
    return record.model_copy(update={"forecasts": forecasts})


__all__ = [
    "STRATA_FEATURES",
    "CounterfactualEngine",
    "Episode",
    "HorizonPricing",
    "Pricing",
    "episode_for",
    "episode_threshold",
    "find_episodes",
    "independent_windows",
    "recombine",
    "reweight_record",
    "strata_frame",
    "strata_table",
]
