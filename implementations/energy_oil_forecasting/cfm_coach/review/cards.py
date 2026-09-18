"""Stage B: one case card per (stream, cutoff). Deterministic, cached, bounded.

A card is what triage reads. It has three parts, all built by code:

- **representative run** -- the median-overlay run's claims, actions, rationale,
  queries, summaries and policy decision, every free-text field sanitized;
- **dispersion** -- how the runs at this cutoff disagreed with each other (the
  lite stream's ten draws are a natural experiment on LLM judgment);
- **numerical base** -- per resolved horizon: components, ensemble, random walk,
  the agent, and the *loss decomposition*: how much of the agent's gap to the
  random walk is the ensemble and how much is the overlay. Pinball is additive,
  so ``gap_vs_rw == base_term + overlay_term`` to floating-point precision.

Cards are cached by ``(run_ids, resolved horizons, builder_version)`` -- never by
``data_through``, which advances daily and would rebuild every card every week.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from energy_oil_forecasting.cfm_coach.diagnostics import state_from_diagnostics
from energy_oil_forecasting.cfm_coach.report import AGENT, ENSEMBLE, RANDOM_WALK
from energy_oil_forecasting.cfm_coach.review.collect import StreamCorpus
from energy_oil_forecasting.cfm_coach.review.settings import DEFAULT_REVIEW_SETTINGS, ReviewSettings
from energy_oil_forecasting.cfm_coach.review.text import estimate_tokens, sanitize
from energy_oil_forecasting.cfm_coach.schemas import Quantiles, RunRecord, ScoreCard
from energy_oil_forecasting.cfm_coach.scoring import pinball_by_level, pinball_loss
from pydantic import BaseModel, ConfigDict, Field


#: Bump when the card's content changes shape; every card rebuilds.
BUILDER_VERSION = 1
#: Prompt budget per card. Measured by `card_tokens` and asserted in tests.
CARD_TOKEN_BUDGET = 3_500

_MAX_CLAIMS = 8
_MAX_SUMMARIES = 4
_CLAIM_CAP = 280
_RATIONALE_CAP = 400
_SUMMARY_CAP = 260


# ------------------------------------------------------------------ models ----


class VariantBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    p10: float
    p50: float
    p90: float
    #: ``p50 - last_close``; the sign the component leans.
    bias: float
    pinball: float | None = None
    covered: float | None = None
    direction_call: int | None = None


class Decomposition(BaseModel):
    """``pinball(agent) - pinball(rw) = base_term + overlay_term``, each split into centre and width."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gap_vs_rw: float
    base_term: float
    overlay_term: float
    base_centre: float
    base_width: float
    overlay_centre: float
    overlay_width: float
    #: ``|base| / (|base| + |overlay|)``; None when both are zero.
    base_share: float | None
    overlay_share: float | None
    beats_rw: bool


class HorizonBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    horizon: int
    n_runs: int
    realized: float
    realized_move: float
    direction_outcome: int
    rw: VariantBase
    ensemble: VariantBase
    agent: VariantBase
    components: dict[str, VariantBase]
    lightgbm_p50_spread: float | None
    lgbm_below_last_close: bool | None
    width_agent: float
    width_rw: float
    #: Multiplier on the agent's own half-width that would just have covered the outcome.
    omega_needed: float | None
    overlay_usd_mean: float
    overlay_usd_max: float
    disagreement_std: float | None
    disagreement_over_width: float | None
    #: Agent vs random walk at the tails and the centre; where the loss sits.
    pinball_levels_agent: dict[str, float]
    pinball_levels_rw: dict[str, float]
    decomposition: Decomposition


class DispersionRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    horizon: int
    n_runs: int
    proposed_center: dict[str, int]
    granted_center: dict[str, int]
    proposed_uncertainty: dict[str, int]
    granted_uncertainty: dict[str, int]
    tiers: dict[str, int]
    #: Runs whose proposed centre move was zeroed by the policy (A7's signature).
    zeroed: int
    novelty: dict[str, int]
    overlay_min: float
    overlay_max: float
    published_p50_spread: float
    ensemble_p50_spread: float
    lightgbm_p50_spread: float | None
    #: Only when resolved: mean pinball by granted centre action, and the best one.
    pinball_by_granted_center: dict[str, float] = Field(default_factory=dict)
    best_granted_center: str | None = None
    pinball_min: float | None = None
    pinball_max: float | None = None


class RepresentativeAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    horizon: int
    proposed_center: str
    proposed_uncertainty: str
    granted_center: str
    granted_uncertainty: str
    tier: str
    eligible: bool
    reasons: list[str]
    cited_claim_ids: list[str]
    persistence_profile: str | None
    rationale: str
    action_fraction: float | None
    novelty_multiplier: float | None
    uncertainty_multiplier: float | None
    applied_center_adjustment: float | None


class RepresentativeClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
    claim_type: str
    statement: str
    material: bool
    n_sources: int
    n_summaries: int


class RepresentativeSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    summary_id: str
    area: str
    status: str
    verifier_confidence: float | None
    text: str


class RepresentativeRun(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    issued_at: str
    physical_status: str
    novelty: str
    confidence: float | None
    conflict: bool
    claims: list[RepresentativeClaim]
    claims_omitted: int
    queries: list[str]
    summaries: list[RepresentativeSummary]
    actions: list[RepresentativeAction]
    overall_rationale: str
    research_summary: str
    warnings: list[str]


class CaseCard(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    builder_version: int
    card_version: int
    card_hash: str
    stream: str
    agent_id: str
    model: str
    cutoff: date
    run_ids: list[str]
    resolved_horizons: list[int]
    data_through: date | None
    last_close: float | None
    latest_observation_date: str | None
    regime: dict[str, float | None]
    successful_models: list[str]
    failed_models: list[str]
    representative: RepresentativeRun
    dispersion: list[DispersionRow]
    base: list[HorizonBase]
    episode_id: str | None = None

    @property
    def key(self) -> tuple[str, date]:
        return (self.stream, self.cutoff)


# --------------------------------------------------------------- helpers ----


def card_hash(run_ids: list[str], resolved_horizons: list[int], builder_version: int = BUILDER_VERSION) -> str:
    payload = json.dumps(
        {"runs": sorted(run_ids), "h": sorted(resolved_horizons), "b": builder_version}, sort_keys=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _shift(quantiles: Quantiles, delta: float) -> Quantiles:
    return {level: value + delta for level, value in quantiles.items()}


def _hist(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _spread(values: list[float]) -> float:
    return float(max(values) - min(values)) if values else 0.0


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _opt_mean(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return float(statistics.fmean(clean)) if clean else None


def _overlay(record: RunRecord, horizon: int) -> float:
    item = record.horizon(horizon)
    return item.final_point_forecast - item.ensemble_quantiles[0.5]


def _proposed(record: RunRecord) -> dict[int, dict[str, Any]]:
    return {int(a["horizon"]): a for a in record.assessment.get("horizon_actions", [])}


def omega_needed(quantiles: Quantiles, actual: float) -> float | None:
    """`fit_width.needed_stretch` for one forecast: the stretch that would just have covered."""
    p10, p50, p90 = quantiles[0.1], quantiles[0.5], quantiles[0.9]
    half = (p90 - p50) if actual >= p50 else (p50 - p10)
    if half <= 0:
        return None
    return float(abs(actual - p50) / half)


# ----------------------------------------------------------- the pieces ----


def representative_run(runs: list[RunRecord]) -> RunRecord:
    """The run at the median of max-|overlay| -- neither the boldest nor the most neutral."""
    ranked = sorted(runs, key=lambda r: (max(abs(_overlay(r, h)) for h in r.horizons), r.run_id))
    return ranked[len(ranked) // 2]


def build_representative(record: RunRecord) -> RepresentativeRun:
    assessment = record.assessment
    packet = record.research_packet
    proposed = _proposed(record)
    accepted_ids = {
        s.get("verified_summary_id") for s in packet.get("verified_summaries", []) if s.get("status") == "accepted"
    }

    claims_raw = assessment.get("evidence_claims", []) or []
    claims = [
        RepresentativeClaim(
            claim_id=str(c.get("claim_id", "")),
            claim_type=str(c.get("claim_type", "?")),
            statement=sanitize(c.get("statement"), cap=_CLAIM_CAP),
            material=bool(c.get("material_to_forecast", False)),
            n_sources=len(c.get("supporting_source_ids", []) or []),
            n_summaries=sum(1 for s in (c.get("supporting_summary_ids") or []) if s in accepted_ids),
        )
        for c in claims_raw[:_MAX_CLAIMS]
    ]
    summaries = [
        RepresentativeSummary(
            summary_id=str(s.get("verified_summary_id", ""))[-12:],
            area=str(s.get("query_area", "?")),
            status=str(s.get("status", "?")),
            verifier_confidence=s.get("verifier_confidence"),
            text=sanitize(s.get("cleaned_summary"), cap=_SUMMARY_CAP),
        )
        for s in (packet.get("verified_summaries", []) or [])[:_MAX_SUMMARIES]
    ]
    actions = []
    for item in record.forecasts:
        decision = item.policy_decision or {}
        action = proposed.get(item.horizon, {})
        transformation = item.forecast_transformation or {}
        actions.append(
            RepresentativeAction(
                horizon=item.horizon,
                proposed_center=str(action.get("center_action", "--")),
                proposed_uncertainty=str(action.get("uncertainty_action", "--")),
                granted_center=str(decision.get("center_action", "--")),
                granted_uncertainty=str(decision.get("uncertainty_action", "--")),
                tier=str(decision.get("evidence_tier", "none")),
                eligible=bool(decision.get("eligible", False)),
                reasons=[sanitize(r, cap=200) for r in (decision.get("eligibility_reasons") or [])][:3],
                cited_claim_ids=[str(c) for c in (action.get("cited_claim_ids") or [])],
                persistence_profile=action.get("persistence_profile"),
                rationale=sanitize(action.get("rationale"), cap=_RATIONALE_CAP),
                action_fraction=transformation.get("action_fraction"),
                novelty_multiplier=transformation.get("novelty_multiplier"),
                uncertainty_multiplier=transformation.get("uncertainty_multiplier"),
                applied_center_adjustment=transformation.get("applied_center_adjustment"),
            )
        )
    return RepresentativeRun(
        run_id=record.run_id,
        issued_at=record.issued_at.isoformat(timespec="minutes"),
        physical_status=str(assessment.get("physical_status", "--")),
        novelty=str(assessment.get("incremental_novelty", "--")),
        confidence=assessment.get("confidence"),
        conflict=bool(assessment.get("material_evidence_conflict", False)),
        claims=claims,
        claims_omitted=max(0, len(claims_raw) - _MAX_CLAIMS),
        queries=[sanitize(q.get("query"), cap=160) for q in (packet.get("queries") or [])],
        summaries=summaries,
        actions=actions,
        overall_rationale=sanitize(assessment.get("overall_rationale"), cap=500),
        research_summary=sanitize(assessment.get("research_summary"), cap=400),
        warnings=[sanitize(w, cap=160) for w in (assessment.get("warnings") or [])][:4],
    )


def build_dispersion(corpus: StreamCorpus, runs: list[RunRecord], horizon: int) -> DispersionRow:
    proposed = [_proposed(r).get(horizon, {}) for r in runs]
    decisions = [r.horizon(horizon).policy_decision or {} for r in runs]
    overlays = [_overlay(r, horizon) for r in runs]
    zeroed = sum(
        1
        for p, d in zip(proposed, decisions, strict=True)
        if p.get("center_action") not in (None, "no_change") and d.get("center_action") == "no_change"
    )
    lgbm = [r.horizon(horizon).component_quantiles.get("lightgbm", {}).get(0.5) for r in runs]
    lgbm_clean = [v for v in lgbm if v is not None]

    by_action: dict[str, list[float]] = {}
    pinballs: list[float] = []
    for r, d in zip(runs, decisions, strict=True):
        card = corpus.score(r.run_id, horizon, AGENT)
        if card is None:
            continue
        pinballs.append(card.pinball)
        by_action.setdefault(str(d.get("center_action", "--")), []).append(card.pinball)
    by_action_mean = {k: _mean(v) for k, v in sorted(by_action.items())}
    return DispersionRow(
        horizon=horizon,
        n_runs=len(runs),
        proposed_center=_hist([str(p.get("center_action", "--")) for p in proposed]),
        granted_center=_hist([str(d.get("center_action", "--")) for d in decisions]),
        proposed_uncertainty=_hist([str(p.get("uncertainty_action", "--")) for p in proposed]),
        granted_uncertainty=_hist([str(d.get("uncertainty_action", "--")) for d in decisions]),
        tiers=_hist([str(d.get("evidence_tier", "none")) for d in decisions]),
        zeroed=zeroed,
        novelty=_hist([str(r.assessment.get("incremental_novelty", "--")) for r in runs]),
        overlay_min=float(min(overlays)),
        overlay_max=float(max(overlays)),
        published_p50_spread=_spread([r.horizon(horizon).final_point_forecast for r in runs]),
        ensemble_p50_spread=_spread([r.horizon(horizon).ensemble_quantiles[0.5] for r in runs]),
        lightgbm_p50_spread=_spread(lgbm_clean) if lgbm_clean else None,
        pinball_by_granted_center=by_action_mean,
        best_granted_center=min(by_action_mean, key=by_action_mean.get) if by_action_mean else None,
        pinball_min=min(pinballs) if pinballs else None,
        pinball_max=max(pinballs) if pinballs else None,
    )


def decompose(
    *, agent: ScoreCard, ensemble: ScoreCard, rw: ScoreCard, agent_q: Quantiles, ens_q: Quantiles, last_close: float
) -> Decomposition:
    """Split the agent's gap to the random walk into base/overlay and centre/width parts. Exactly additive."""
    actual = agent.realized_value
    gap = agent.pinball - rw.pinball
    base_term = ensemble.pinball - rw.pinball
    overlay_term = agent.pinball - ensemble.pinball
    # Centre part of the base: what shifting the ensemble's median onto the last close would have saved.
    base_centre = ensemble.pinball - pinball_loss(_shift(ens_q, last_close - ens_q[0.5]), actual)
    base_width = base_term - base_centre
    # Centre part of the overlay: what undoing the centre move (keeping the width change) would have saved.
    overlay_centre = agent.pinball - pinball_loss(_shift(agent_q, ens_q[0.5] - agent_q[0.5]), actual)
    overlay_width = overlay_term - overlay_centre
    total = abs(base_term) + abs(overlay_term)
    return Decomposition(
        gap_vs_rw=gap,
        base_term=base_term,
        overlay_term=overlay_term,
        base_centre=base_centre,
        base_width=base_width,
        overlay_centre=overlay_centre,
        overlay_width=overlay_width,
        base_share=abs(base_term) / total if total > 0 else None,
        overlay_share=abs(overlay_term) / total if total > 0 else None,
        beats_rw=gap < 0,
    )


def _variant_base(cards: list[ScoreCard], last_close: float) -> VariantBase:
    return VariantBase(
        p10=_mean([c.p10 for c in cards]),
        p50=_mean([c.p50 for c in cards]),
        p90=_mean([c.p90 for c in cards]),
        bias=_mean([c.p50 for c in cards]) - last_close,
        pinball=_mean([c.pinball for c in cards]),
        covered=_mean([float(c.covered_80) for c in cards]),
        direction_call=Counter(c.direction_call for c in cards).most_common(1)[0][0],
    )


def _levels(cards: list[ScoreCard], quantiles_of: Any) -> dict[str, float]:
    """Mean pinball at 0.1 / 0.5 / 0.9 across runs."""
    acc: dict[float, list[float]] = {0.1: [], 0.5: [], 0.9: []}
    for card in cards:
        losses = pinball_by_level(quantiles_of(card), card.realized_value)
        for level in acc:
            if level in losses:
                acc[level].append(losses[level])
    return {str(level): _mean(values) for level, values in acc.items() if values}


def build_horizon_base(corpus: StreamCorpus, runs: list[RunRecord], horizon: int) -> HorizonBase | None:
    by_run = {r.run_id: r for r in runs}
    agent_cards = [c for r in runs if (c := corpus.score(r.run_id, horizon, AGENT)) is not None]
    ens_cards = [c for r in runs if (c := corpus.score(r.run_id, horizon, ENSEMBLE)) is not None]
    rw_cards = [c for r in runs if (c := corpus.score(r.run_id, horizon, RANDOM_WALK)) is not None]
    if not agent_cards or not ens_cards or not rw_cards or len(agent_cards) != len(rw_cards):
        return None
    last_close = float(runs[0].diagnostics["latest_value"])
    realized = agent_cards[0].realized_value

    decomps = []
    for a in agent_cards:
        e = corpus.score(a.run_id, horizon, ENSEMBLE)
        w = corpus.score(a.run_id, horizon, RANDOM_WALK)
        record = by_run[a.run_id].horizon(horizon)
        decomps.append(
            decompose(
                agent=a,
                ensemble=e,
                rw=w,
                agent_q=dict(record.final_quantiles),
                ens_q=dict(record.ensemble_quantiles),
                last_close=last_close,
            )
        )
    mean_decomp = Decomposition(
        gap_vs_rw=_mean([d.gap_vs_rw for d in decomps]),
        base_term=_mean([d.base_term for d in decomps]),
        overlay_term=_mean([d.overlay_term for d in decomps]),
        base_centre=_mean([d.base_centre for d in decomps]),
        base_width=_mean([d.base_width for d in decomps]),
        overlay_centre=_mean([d.overlay_centre for d in decomps]),
        overlay_width=_mean([d.overlay_width for d in decomps]),
        base_share=_opt_mean([d.base_share for d in decomps]),
        overlay_share=_opt_mean([d.overlay_share for d in decomps]),
        beats_rw=_mean([d.gap_vs_rw for d in decomps]) < 0,
    )

    components: dict[str, VariantBase] = {}
    names = sorted({name for r in runs for name in r.horizon(horizon).component_quantiles})
    for name in names:
        qs = [
            r.horizon(horizon).component_quantiles[name] for r in runs if name in r.horizon(horizon).component_quantiles
        ]
        p50 = _mean([q[0.5] for q in qs])
        components[name] = VariantBase(
            p10=_mean([q[0.1] for q in qs]), p50=p50, p90=_mean([q[0.9] for q in qs]), bias=p50 - last_close
        )
    lgbm = [r.horizon(horizon).component_quantiles.get("lightgbm", {}).get(0.5) for r in runs]
    lgbm_clean = [v for v in lgbm if v is not None]

    disagreement = _opt_mean(
        [
            (r.audit_signals.get("model_disagreement_std") or {}).get(str(horizon))
            if isinstance(r.audit_signals.get("model_disagreement_std"), dict)
            else None
            for r in runs
        ]
    )
    width_agent = _mean([c.interval_width for c in agent_cards])
    omegas = [
        omega_needed(dict(by_run[c.run_id].horizon(horizon).final_quantiles), c.realized_value) for c in agent_cards
    ]
    return HorizonBase(
        horizon=horizon,
        n_runs=len(runs),
        realized=realized,
        realized_move=realized - last_close,
        direction_outcome=agent_cards[0].direction_outcome or 0,
        rw=_variant_base(rw_cards, last_close),
        ensemble=_variant_base(ens_cards, last_close),
        agent=_variant_base(agent_cards, last_close),
        components=components,
        lightgbm_p50_spread=_spread(lgbm_clean) if lgbm_clean else None,
        lgbm_below_last_close=(all(v < last_close for v in lgbm_clean) if lgbm_clean else None),
        width_agent=width_agent,
        width_rw=_mean([c.interval_width for c in rw_cards]),
        omega_needed=_opt_mean(omegas),
        overlay_usd_mean=_mean([_overlay(r, horizon) for r in runs]),
        overlay_usd_max=max((abs(_overlay(r, horizon)) for r in runs), default=0.0),
        disagreement_std=disagreement,
        disagreement_over_width=(disagreement / width_agent if disagreement is not None and width_agent > 0 else None),
        pinball_levels_agent=_levels(agent_cards, lambda c: dict(by_run[c.run_id].horizon(horizon).final_quantiles)),
        pinball_levels_rw=_levels(rw_cards, lambda c: _rw_quantiles(corpus, by_run[c.run_id], horizon)),
        decomposition=mean_decomp,
    )


def _rw_quantiles(corpus: StreamCorpus, record: RunRecord, horizon: int) -> Quantiles:
    built = corpus.baseline.for_record(record, horizon) if corpus.baseline is not None else None
    if built is None:
        raise RuntimeError(f"random-walk quantiles missing for {record.run_id} h={horizon}")
    return built[1]


# ---------------------------------------------------------------- build ----


def build_card(corpus: StreamCorpus, cutoff: date, settings: ReviewSettings = DEFAULT_REVIEW_SETTINGS) -> CaseCard:
    runs = corpus.runs_by_cutoff[cutoff]
    resolved = sorted(corpus.resolved_horizons(cutoff))
    first, second = set(settings.triage_after_horizons), set(settings.rescore_at_horizons)
    #: 0 = not ready for triage, 1 = triage horizons resolved, 2 = re-score horizons resolved too.
    version = 2 if (first | second) <= set(resolved) else 1 if first <= set(resolved) else 0
    rep = representative_run(runs)
    horizons = sorted({h for r in runs for h in r.horizons})
    diagnostics = rep.diagnostics
    successful = sorted({m for r in runs for m in (r.audit_signals.get("successful_models") or [])})
    failed = sorted({m for r in runs for m in (r.audit_signals.get("failed_models") or [])})
    base = [b for h in resolved if (b := build_horizon_base(corpus, runs, h)) is not None]
    return CaseCard(
        builder_version=BUILDER_VERSION,
        card_version=version,
        card_hash=card_hash([r.run_id for r in runs], resolved),
        stream=corpus.stream_id,
        agent_id=corpus.agent_id,
        model=corpus.model,
        cutoff=cutoff,
        run_ids=sorted(r.run_id for r in runs),
        resolved_horizons=resolved,
        data_through=corpus.data_through,
        last_close=float(diagnostics["latest_value"]) if diagnostics.get("latest_value") is not None else None,
        latest_observation_date=str(diagnostics.get("latest_observation_date"))
        if diagnostics.get("latest_observation_date")
        else None,
        regime=state_from_diagnostics(diagnostics),
        successful_models=successful,
        failed_models=failed,
        representative=build_representative(rep),
        dispersion=[build_dispersion(corpus, runs, h) for h in horizons],
        base=base,
    )


class CardStore:
    """``cards/<stream>/<cutoff>.json``; a card is reused when its hash matches."""

    def __init__(self, settings: ReviewSettings = DEFAULT_REVIEW_SETTINGS):
        self.settings = settings

    def path_for(self, stream: str, cutoff: date) -> Path:
        return self.settings.cards_dir / stream / f"{cutoff.isoformat()}.json"

    def load(self, stream: str, cutoff: date) -> CaseCard | None:
        path = self.path_for(stream, cutoff)
        if not path.exists():
            return None
        return CaseCard.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, card: CaseCard) -> Path:
        path = self.path_for(card.stream, card.cutoff)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(card.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return path

    def load_or_build(self, corpus: StreamCorpus, cutoff: date) -> tuple[CaseCard, bool]:
        """Return (card, rebuilt)."""
        runs = corpus.runs_by_cutoff[cutoff]
        wanted = card_hash([r.run_id for r in runs], sorted(corpus.resolved_horizons(cutoff)))
        cached = self.load(corpus.stream_id, cutoff)
        if cached is not None and cached.card_hash == wanted and cached.builder_version == BUILDER_VERSION:
            return cached, False
        card = build_card(corpus, cutoff, self.settings)
        self.save(card)
        return card, True


# --------------------------------------------------------------- render ----


def _f(value: float | None, digits: int = 2) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def _pct(value: float | None) -> str:
    return "--" if value is None else f"{value:.0%}"


def render_card(card: CaseCard, *, horizons: tuple[int, ...] | None = None) -> str:
    """The card as prompt text. Everything free-text is inside untrusted tags."""
    rep = card.representative
    lines = [
        f"# Case {card.stream} cutoff {card.cutoff} (card v{card.card_version}, hash {card.card_hash})",
        f"agent {card.agent_id} on {card.model}; {len(card.run_ids)} runs; resolved horizons {card.resolved_horizons}; "
        f"last close {_f(card.last_close)} on {card.latest_observation_date}; models ok {card.successful_models} failed {card.failed_models}",
        "regime: " + ", ".join(f"{k}={_f(v, 3)}" for k, v in card.regime.items()),
        "",
        "## Numerical base and outcome (per resolved horizon; means over runs)",
    ]
    for b in card.base:
        if horizons is not None and b.horizon not in horizons:
            continue
        d = b.decomposition
        lines += [
            f"h{b.horizon}: realized {_f(b.realized)} (move {b.realized_move:+.2f}, direction {b.direction_outcome:+d})",
            f"  rw       p10/p50/p90 {_f(b.rw.p10)}/{_f(b.rw.p50)}/{_f(b.rw.p90)} pinball {_f(b.rw.pinball, 3)} covered {_pct(b.rw.covered)} width {_f(b.width_rw)}",
            f"  ensemble p10/p50/p90 {_f(b.ensemble.p10)}/{_f(b.ensemble.p50)}/{_f(b.ensemble.p90)} bias {b.ensemble.bias:+.2f} pinball {_f(b.ensemble.pinball, 3)} covered {_pct(b.ensemble.covered)} call {b.ensemble.direction_call}",
            f"  agent    p10/p50/p90 {_f(b.agent.p10)}/{_f(b.agent.p50)}/{_f(b.agent.p90)} bias {b.agent.bias:+.2f} pinball {_f(b.agent.pinball, 3)} covered {_pct(b.agent.covered)} call {b.agent.direction_call} width {_f(b.width_agent)} omega_needed {_f(b.omega_needed)}",
            "  components p50 bias: "
            + ", ".join(f"{k} {v.bias:+.2f}" for k, v in b.components.items())
            + f"; lightgbm spread {_f(b.lightgbm_p50_spread)}; lgbm below last close {b.lgbm_below_last_close}; disagreement/width {_f(b.disagreement_over_width, 3)}",
            f"  overlay $ mean {b.overlay_usd_mean:+.2f} max {_f(b.overlay_usd_max)}; pinball by level agent {b.pinball_levels_agent} rw {b.pinball_levels_rw}",
            f"  decomposition: gap_vs_rw {d.gap_vs_rw:+.3f} = base {d.base_term:+.3f} (centre {d.base_centre:+.3f}, width {d.base_width:+.3f}) + overlay {d.overlay_term:+.3f} (centre {d.overlay_centre:+.3f}, width {d.overlay_width:+.3f}); base_share {_pct(d.base_share)} overlay_share {_pct(d.overlay_share)} beats_rw {d.beats_rw}",
        ]
    lines += ["", "## Same-cutoff dispersion across runs"]
    for row in card.dispersion:
        lines += [
            f"h{row.horizon} n={row.n_runs}: proposed centre {row.proposed_center} -> granted {row.granted_center} (zeroed {row.zeroed}); "
            f"proposed unc {row.proposed_uncertainty} -> granted {row.granted_uncertainty}; tiers {row.tiers}; novelty {row.novelty}",
            f"  overlay [{row.overlay_min:+.2f}, {row.overlay_max:+.2f}]; p50 spread published {_f(row.published_p50_spread)} ensemble {_f(row.ensemble_p50_spread)} lightgbm {_f(row.lightgbm_p50_spread)}"
            + (
                f"; pinball by granted centre {{{', '.join(f'{k}: {v:.3f}' for k, v in row.pinball_by_granted_center.items())}}} best {row.best_granted_center} range [{_f(row.pinball_min, 3)}, {_f(row.pinball_max, 3)}]"
                if row.pinball_by_granted_center
                else ""
            ),
        ]
    lines += [
        "",
        f"## Representative run {rep.run_id} (issued {rep.issued_at})",
        f"physical_status {rep.physical_status}; novelty {rep.novelty}; confidence {rep.confidence}; conflict {rep.conflict}",
        "queries: " + " | ".join(f"<untrusted query>{q}</untrusted query>" for q in rep.queries),
        "claims:",
    ]
    for c in rep.claims:
        lines.append(
            f"  {c.claim_id} [{c.claim_type}; material {c.material}; sources {c.n_sources}; accepted summaries {c.n_summaries}] <untrusted claim>{c.statement}</untrusted claim>"
        )
    if rep.claims_omitted:
        lines.append(f"  (+{rep.claims_omitted} more claims omitted)")
    lines.append("verified summaries:")
    for s in rep.summaries:
        lines.append(
            f"  {s.summary_id} [{s.area}; {s.status}; verifier {s.verifier_confidence}] <untrusted summary>{s.text}</untrusted summary>"
        )
    lines.append("actions:")
    for a in rep.actions:
        lines += [
            f"  h{a.horizon}: proposed {a.proposed_center}/{a.proposed_uncertainty} -> granted {a.granted_center}/{a.granted_uncertainty}; tier {a.tier}; eligible {a.eligible}; cites {a.cited_claim_ids}; persistence {a.persistence_profile}; "
            f"fraction {a.action_fraction} novelty_mult {a.novelty_multiplier} unc_mult {a.uncertainty_multiplier} applied {_f(a.applied_center_adjustment)}",
        ]
        if a.reasons:
            lines.append("    policy reasons: " + " / ".join(a.reasons))
        lines.append(f"    <untrusted rationale>{a.rationale}</untrusted rationale>")
    lines += [
        f"overall: <untrusted rationale>{rep.overall_rationale}</untrusted rationale>",
        f"research summary: <untrusted summary>{rep.research_summary}</untrusted summary>",
    ]
    if rep.warnings:
        lines.append("warnings: " + " / ".join(f"<untrusted warning>{w}</untrusted warning>" for w in rep.warnings))
    return "\n".join(lines)


def card_tokens(card: CaseCard) -> int:
    return estimate_tokens(render_card(card))


# ------------------------------------------------------------------ cli ----


def main() -> None:
    from energy_oil_forecasting.cfm_coach.review.collect import collect_stream, describe  # noqa: PLC0415
    from energy_oil_forecasting.cfm_coach.streams import stream_for  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Build (or rebuild) case cards for one stream.")
    parser.add_argument("--stream", required=True)
    parser.add_argument("--cutoff", default=None, help="one cutoff (YYYY-MM-DD); default all")
    parser.add_argument("--print", action="store_true", help="print the rendered card(s)")
    args = parser.parse_args()

    corpus = collect_stream(stream_for(args.stream))
    print(describe(corpus))
    store = CardStore()
    cutoffs = [date.fromisoformat(args.cutoff)] if args.cutoff else corpus.cutoffs
    for cutoff in cutoffs:
        card, rebuilt = store.load_or_build(corpus, cutoff)
        tokens = card_tokens(card)
        flag = "OVER BUDGET" if tokens > CARD_TOKEN_BUDGET else "ok"
        print(
            f"{cutoff} v{card.card_version} resolved={card.resolved_horizons} runs={len(card.run_ids)} tokens={tokens} {flag} {'rebuilt' if rebuilt else 'cached'}"
        )
        if args.print:
            print(render_card(card))
            print()


if __name__ == "__main__":
    main()


__all__ = [
    "BUILDER_VERSION",
    "CARD_TOKEN_BUDGET",
    "CardStore",
    "CaseCard",
    "Decomposition",
    "DispersionRow",
    "HorizonBase",
    "RepresentativeRun",
    "build_card",
    "card_hash",
    "card_tokens",
    "decompose",
    "omega_needed",
    "render_card",
    "representative_run",
]
