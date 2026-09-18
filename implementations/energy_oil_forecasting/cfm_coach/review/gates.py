"""Evidence, computed by code: support units, thresholds, seed statistics, and (later) the critic.

Two support units, chosen by whether a tag can only be judged once the price is
known (decision 12 of the review):

- **outcome-dependent** tags -- support = independent windows (supporting
  cutoffs pairwise >= h business days apart, max over horizons) spanning >= 2
  episodes; direction/bias codes need one episode of each sign;
- **process** tags -- support = distinct cutoffs, plus the same-cutoff
  replication rate (what fraction of the runs at a supporting cutoff show it).

Numeric/policy hypotheses are judged by the gate and the pricing in
`review.numeric`; this module only labels them.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from energy_oil_forecasting.cfm_coach.review.cards import CaseCard
from energy_oil_forecasting.cfm_coach.review.memory import Annotation, Hypothesis, Taxonomy
from energy_oil_forecasting.cfm_coach.review.settings import DEFAULT_REVIEW_SETTINGS, ReviewSettings
from energy_oil_forecasting.cfm_coach.review.state import ReviewKey
from energy_oil_forecasting.cfm_coach.review.stats import Episode, episode_for, independent_windows
from pydantic import BaseModel, ConfigDict, Field


DIRECTIONAL_CODES = (
    "JUDGMENT.DIRECTION_WRONG",
    "BASE.LGBM_LEVEL_BIAS",
    "BASE.ENSEMBLE_LOSES_TO_RW",
    "POLICY.RIGHT_ACTION_WRONG_SIZE",
)


class EvidenceSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hypothesis_id: str
    unit: str  # "independent_windows" | "distinct_cutoffs"
    supporting_keys: int
    contradicting_keys: int
    distinct_cutoffs: int
    independent_windows: int
    episodes: int
    episode_signs: list[int]
    streams: list[str]
    support_ratio: float | None
    replication_rate: float | None
    pointers_complete: bool
    promotable: bool
    reasons: list[str]


def _keys(items: list[str]) -> list[ReviewKey]:
    return [ReviewKey.parse(k) for k in items]


def evidence_for(
    hypothesis: Hypothesis,
    *,
    annotations: dict[ReviewKey, Annotation],
    episodes_by_stream: dict[str, list[Episode]],
    forecast_dates: dict[ReviewKey, date],
    runs_per_cutoff: dict[tuple[str, date], int],
    taxonomy: Taxonomy,
    settings: ReviewSettings = DEFAULT_REVIEW_SETTINGS,
) -> EvidenceSummary:
    """Fold a hypothesis's supporting/contradicting keys into support units and a promotability verdict."""
    codes = [taxonomy.resolve(c) for c in hypothesis.codes]
    outcome = any(c is not None and c.outcome_dependent for c in codes)
    support = _keys(hypothesis.supporting)
    contra = _keys(hypothesis.contradicting)
    reasons: list[str] = []
    if hypothesis.track == "numeric":
        # Numeric hypotheses are not promoted on annotations: a candidate is drafted, priced and judged by the gate.
        return EvidenceSummary(
            hypothesis_id=hypothesis.hypothesis_id,
            unit="gate",
            supporting_keys=len(support),
            contradicting_keys=len(contra),
            distinct_cutoffs=len({k.cutoff for k in support}),
            independent_windows=0,
            episodes=0,
            episode_signs=[],
            streams=sorted({k.stream for k in support}),
            support_ratio=None,
            replication_rate=None,
            pointers_complete=True,
            promotable=False,
            reasons=[
                "numeric track: draft a candidate (layer / settings_overlay / ensemble_weights); the M1 gate and RW dominance decide"
            ],
        )

    cutoffs = sorted({k.cutoff for k in support})
    streams = sorted({k.stream for k in support})
    windows = max(
        (independent_windows([k.cutoff for k in support if k.horizon == h], h) for h in {k.horizon for k in support}),
        default=0,
    )

    seen_eps: dict[str, Episode] = {}
    for k in support:
        ep = episode_for(
            episodes_by_stream.get(k.stream, []), cutoff=k.cutoff, forecast_date=forecast_dates.get(k, k.cutoff)
        )
        if ep is not None:
            seen_eps[ep.episode_id] = ep
    signs = sorted({ep.direction for ep in seen_eps.values()})

    total = len(support) + len(contra)
    ratio = len(support) / total if total else None

    # Replication: at each supporting cutoff, how many of the runs carried a tag for one of the codes?
    reps = []
    for k in support:
        ann = annotations.get(k)
        n_runs = runs_per_cutoff.get((k.stream, k.cutoff), 1)
        if ann is None:
            continue
        hits = sum(1 for t in ann.tags if t.code in hypothesis.codes)
        reps.append(min(1.0, hits / n_runs) if n_runs else 0.0)
    replication = float(np.mean(reps)) if reps else None

    pointers_ok = all(
        any(t.code in hypothesis.codes and t.pointer for t in annotations[k].tags) for k in support if k in annotations
    ) and bool(support)

    if outcome:
        unit = "independent_windows"
        if windows < settings.min_independent_windows:
            reasons.append(f"{windows} independent window(s) < {settings.min_independent_windows}")
        if len(seen_eps) < settings.min_episodes:
            reasons.append(f"{len(seen_eps)} episode(s) < {settings.min_episodes}")
        if any(c in DIRECTIONAL_CODES for c in hypothesis.codes) and len(signs) < 2:
            reasons.append("directional/bias code needs one episode of each sign")
    else:
        unit = "distinct_cutoffs"
        if len(cutoffs) < settings.min_process_cutoffs:
            reasons.append(f"{len(cutoffs)} distinct cutoff(s) < {settings.min_process_cutoffs}")
    if ratio is not None and ratio < settings.min_support_ratio:
        reasons.append(f"support ratio {ratio:.2f} < {settings.min_support_ratio}")
    if not pointers_ok:
        reasons.append("a supporting case lacks a pointer")
    if hypothesis.post_cutoff:
        reasons.append("rests on post-cutoff (hindsight) material")

    return EvidenceSummary(
        hypothesis_id=hypothesis.hypothesis_id,
        unit=unit,
        supporting_keys=len(support),
        contradicting_keys=len(contra),
        distinct_cutoffs=len(cutoffs),
        independent_windows=windows,
        episodes=len(seen_eps),
        episode_signs=signs,
        streams=streams,
        support_ratio=ratio,
        replication_rate=replication,
        pointers_complete=pointers_ok,
        promotable=not reasons and hypothesis.track in ("llm", "policy", "data", "code"),
        reasons=reasons,
    )


def refuted(summary: EvidenceSummary, settings: ReviewSettings = DEFAULT_REVIEW_SETTINGS) -> bool:
    """Contradiction ratio above the threshold over enough cutoffs."""
    total = summary.supporting_keys + summary.contradicting_keys
    if total == 0 or summary.distinct_cutoffs + summary.contradicting_keys < settings.min_process_cutoffs:
        return False
    return summary.contradicting_keys / total > settings.contradiction_ratio


# ------------------------------------------------------------ seed statistics ----


def seed_statistics(frame: pd.DataFrame, cards: list[CaseCard]) -> dict[str, Any]:
    """Code-computed values the numeric/policy seeds cite (`seeds.yaml: evidence_stat`)."""
    out: dict[str, Any] = {}
    if not frame.empty:
        out["lgbm_below_rate"] = (
            float(frame["lgbm_below_last_close"].dropna().astype(float).mean())
            if frame["lgbm_below_last_close"].notna().any()
            else None
        )
        out["failed_models_rate"] = None
        out["zeroed_rate"] = float(frame["zeroed"].astype(float).mean())
        out["coverage_agent"] = float(frame["covered_agent"].mean())
        out["no_call_rate"] = float((frame["direction_call"].fillna(0) == 0).mean())
        out["base_share_mean"] = float(
            (
                frame["base_term"].abs() / (frame["base_term"].abs() + frame["overlay_term"].abs()).replace(0, np.nan)
            ).mean()
        )
        by_vol = frame.groupby("vol_tercile")["width_vs_needed_bin"].apply(lambda s: float((s == "far").mean()))
        out["omega_needed_by_vol"] = {str(k): v for k, v in by_vol.items()}
    if cards:
        spreads = [
            row.lightgbm_p50_spread
            for c in cards
            for row in c.dispersion
            if row.lightgbm_p50_spread is not None and row.n_runs > 1
        ]
        out["lgbm_spread_mean"] = float(np.mean(spreads)) if spreads else None
        p50 = [row.published_p50_spread for c in cards for row in c.dispersion if row.n_runs > 1]
        out["p50_spread_mean"] = float(np.mean(p50)) if p50 else None
        splits = [len(row.novelty) > 1 for c in cards for row in c.dispersion if row.n_runs > 1 and row.horizon == 5]
        out["novelty_split_rate"] = float(np.mean(splits)) if splits else None
        blocked, asked = 0, 0
        for c in cards:
            for row in c.dispersion:
                wider = sum(v for k, v in row.proposed_uncertainty.items() if "wider" in k)
                granted_wider = sum(v for k, v in row.granted_uncertainty.items() if "wider" in k)
                asked += wider
                blocked += max(0, wider - granted_wider)
        out["widen_blocked_rate"] = blocked / asked if asked else None
        failed = [bool(c.failed_models) for c in cards]
        out["failed_models_rate"] = float(np.mean(failed)) if failed else None
    return out


__all__ = ["DIRECTIONAL_CODES", "EvidenceSummary", "evidence_for", "refuted", "seed_statistics"]


# ------------------------------------------------------------------ gates ----

import json as _json  # noqa: E402
from typing import Literal as _Literal  # noqa: E402

from energy_oil_forecasting.cfm_coach.review.collect import StreamCorpus  # noqa: E402
from energy_oil_forecasting.cfm_coach.review.llm import LlmClient, LlmFailureError, inline_refs  # noqa: E402
from energy_oil_forecasting.cfm_coach.review.memory import Proposal, ProposalStore  # noqa: E402
from energy_oil_forecasting.cfm_coach.review.stats import CounterfactualEngine  # noqa: E402
from energy_oil_forecasting.cfm_coach.review.triage import PROMPTS_DIR  # noqa: E402
from pydantic import ValidationError  # noqa: E402


LLM_LEVERS = ("skill_text", "persona", "prompt_instruction")
NUMERIC_LEVERS = ("layer", "settings_overlay", "ensemble_weights")


class CriticOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: _Literal["pass", "downgrade", "block"]
    hindsight: str = ""
    mechanism: str = ""
    confound: str = ""
    blind_spot: str = ""


class GateOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hypothesis_id: str
    title: str
    track: str
    lever_kind: str
    fingerprint: str
    decision: _Literal["pass", "downgrade", "block"]
    reasons: list[str]
    rank_score: float | None = None
    pricing: dict[str, Any] | None = None
    numeric: dict[str, Any] | None = None
    critic: dict[str, Any] | None = None
    base_share_mean: float | None = None
    streams_supporting: list[str] = Field(default_factory=list)


def _cases_base_share(keys: list[str], cards_by_key: dict[tuple[str, date], CaseCard]) -> float | None:
    shares = []
    for text in keys:
        k = ReviewKey.parse(text)
        card = cards_by_key.get((k.stream, k.cutoff))
        if card is None:
            continue
        base = next((b for b in card.base if b.horizon == k.horizon), None)
        if base is not None and base.decomposition.base_share is not None:
            shares.append(base.decomposition.base_share)
    return float(np.mean(shares)) if shares else None


def _cross_stream(
    keys: list[str], annotations: dict[ReviewKey, Annotation], codes: list[str], corpora: dict[str, StreamCorpus]
) -> tuple[list[str], bool]:
    """(streams with support, model_specific): support in one stream only while the other stream was annotated at the same cutoffs without the codes."""
    supporting = sorted({ReviewKey.parse(k).stream for k in keys})
    if len(supporting) != 1 or len(corpora) < 2:
        return supporting, False
    only = supporting[0]
    cutoffs = {ReviewKey.parse(k).cutoff for k in keys}
    for other in (s for s in corpora if s != only):
        annotated = [a for k, a in annotations.items() if k.stream == other and k.cutoff in cutoffs]
        if annotated and not any(t.code in codes for a in annotated for t in a.tags):
            return supporting, True
    return supporting, False


def _new_cutoffs_since(corpora: dict[str, StreamCorpus], since: date | None) -> int:
    if since is None:
        return 0
    return len({c for corpus in corpora.values() for c in corpus.cutoffs if c > since})


async def _critic(
    client: LlmClient,
    *,
    title: str,
    statement: str,
    lever: dict[str, Any],
    evidence: dict[str, Any],
    pricing: dict[str, Any] | None,
) -> dict[str, Any]:
    rubric = (PROMPTS_DIR / "critic.md").read_text(encoding="utf-8")
    body = _json.dumps(
        {"title": title, "claim": statement, "lever": lever, "evidence": evidence, "pricing": pricing},
        indent=1,
        default=str,
    )[:12_000]
    try:
        response = await client.complete(
            stage="critic",
            messages=[{"role": "system", "content": rubric}, {"role": "user", "content": body}],
            response_schema=inline_refs(CriticOut.model_json_schema()),
            schema_name="CriticOut",
        )
        return CriticOut.model_validate_json(response.content or "{}").model_dump()
    except (LlmFailureError, ValidationError, ValueError) as exc:
        return {"verdict": "downgrade", "error": f"critic failed: {str(exc)[:200]}"}


LAYER_PARAMS = ("centre_gain", "width_scale", "rw_anchor")
_SCALAR_KEYS = ("a", "value", "omega", "g", "gain", "scale", "anchor")


def _per_horizon(value: Any, horizons: tuple[int, ...]) -> dict[int, float]:
    """Read a layer parameter written as a scalar, `{"a": x}`, or `{"5": x, "10": y}`."""
    if isinstance(value, dict) and value and all(str(k).lstrip("h").isdigit() for k in value):
        return {int(str(k).lstrip("h")): float(v) for k, v in value.items()}
    if isinstance(value, dict):
        scalar = next((v for k, v in value.items() if k in _SCALAR_KEYS and isinstance(v, (int, float))), None)
        if scalar is None:
            raise ValueError(f"cannot read a layer value from {value!r}")
        return dict.fromkeys(horizons, float(scalar))
    return dict.fromkeys(horizons, float(value))


def _layer_from_draft(draft: Any, layer_cls: Any) -> Any:
    """A layer draft either nests parameters in `value` or names one in `field_name` with a scalar `value`."""
    horizons = tuple(int(h) for h in draft.horizons) or (5, 10, 21)
    params = {k: _per_horizon(v, horizons) for k, v in dict(draft.value or {}).items() if k in LAYER_PARAMS}
    if not params and draft.field_name in LAYER_PARAMS:
        params[draft.field_name] = _per_horizon(draft.value, horizons)
    if not params:
        raise ValueError(f"layer draft names no layer parameter {LAYER_PARAMS}; got field {draft.field_name!r}")
    return layer_cls(**params)


def missing_files(draft: Any) -> list[str]:
    """Files a draft edits or points at that do not exist under the agent package, its parent, or the repo."""
    from energy_oil_forecasting.cfm_coach.targets import V52  # noqa: PLC0415

    root = Path(V52.package_root)
    named = [e.file for e in draft.text_edits] + ([draft.file_line.split(":")[0]] if draft.file_line else [])
    missing = []
    for rel in named:
        if not rel or rel.startswith("<"):
            continue
        rel = rel.strip()
        if not any((base / rel).exists() for base in (root, root.parent, root.parents[2])):
            missing.append(rel)
    return missing


def _candidate_for(draft: Any, hypothesis_id: str, stream_id: str) -> Any:
    from energy_oil_forecasting.cfm_coach.schemas import CalibrationLayer, Candidate  # noqa: PLC0415

    cid = f"{hypothesis_id}__{stream_id}"
    if draft.lever_kind == "layer":
        layer = _layer_from_draft(draft, CalibrationLayer)
        return Candidate(
            candidate_id=cid, parent_version="v001", layer=layer, fitted_through=None, rationale=draft.statement[:200]
        )
    if draft.lever_kind == "ensemble_weights":
        return Candidate(
            candidate_id=cid,
            parent_version="v001",
            settings_overlay={"ensemble_weights": draft.value},
            fitted_through=None,
            rationale=draft.statement[:200],
        )
    if not draft.field_name or draft.value.get("value") is None:
        raise ValueError('a settings draft needs `field_name` and `value: {"value": ...}`')
    return Candidate(
        candidate_id=cid,
        parent_version="v001",
        settings_overlay={draft.field_name: draft.value.get("value")},
        fitted_through=None,
        rationale=draft.statement[:200],
    )


async def run_gates(  # noqa: PLR0912, PLR0915 - the gates read top to bottom
    drafts: list[Any],
    *,
    hypotheses: dict[str, Hypothesis],
    evidence: dict[str, EvidenceSummary],
    proposals: ProposalStore,
    corpora: dict[str, StreamCorpus],
    annotations: dict[ReviewKey, Annotation],
    cards_by_key: dict[tuple[str, date], CaseCard],
    client: LlmClient | None,
    settings: ReviewSettings = DEFAULT_REVIEW_SETTINGS,
) -> list[GateOutcome]:
    """Deterministic gates first; the critic only on what passes them."""
    from energy_oil_forecasting.cfm_coach.review.numeric import judge_candidate  # noqa: PLC0415
    from energy_oil_forecasting.cfm_coach.streams import stream_for  # noqa: PLC0415

    outcomes: list[GateOutcome] = []
    for draft in drafts:
        h = hypotheses.get(draft.hypothesis_id)
        if h is None:
            continue
        reasons: list[str] = []
        decision = "pass"
        lever = {
            "kind": draft.lever_kind,
            "file_line": draft.file_line,
            "field": draft.field_name,
            "value": draft.value,
            "text_edits": [e.model_dump() for e in draft.text_edits],
        }
        fingerprint = Proposal.make_fingerprint(
            {
                "kind": draft.lever_kind,
                "file": draft.file_line or draft.field_name or (draft.text_edits[0].file if draft.text_edits else ""),
            },
            h.codes,
        )
        ev = evidence.get(h.hypothesis_id)
        streams_supporting, model_specific = _cross_stream(h.supporting, annotations, h.codes, corpora)
        base_share = _cases_base_share(h.supporting, cards_by_key)

        rejected = proposals.rejected().get(fingerprint)
        blocked, why = proposals.blocked(
            fingerprint,
            new_cutoffs_since_rejection=_new_cutoffs_since(corpora, rejected.decision.on if rejected else None),
        )
        if blocked:
            reasons.append(why)
            decision = "block"
        if draft.lever_kind in LLM_LEVERS and base_share is not None and base_share >= settings.base_dominated_share:
            reasons.append(
                f"supporting cases are {base_share:.0%} base-attributable; an LLM lever does not address the loss"
            )
            decision = "downgrade" if decision != "block" else decision
        if missing := missing_files(draft):
            reasons.append(f"names file(s) that do not exist: {missing}")
            decision = "downgrade" if decision != "block" else decision
        if model_specific and draft.streams and len(draft.streams) > 1:
            reasons.append(
                f"support only in {streams_supporting}; the other stream was annotated at the same cutoffs without these codes (model_specific)"
            )
            decision = "downgrade" if decision != "block" else decision

        pricing_dump = None
        numeric_dump = None
        rank = None
        if draft.lever_kind in NUMERIC_LEVERS:
            verdicts = []
            for stream_id in draft.streams or list(corpora):
                corpus = corpora.get(stream_id)
                if corpus is None:
                    continue
                try:
                    verdict = judge_candidate(
                        corpus,
                        stream_for(stream_id),
                        _candidate_for(draft, h.hypothesis_id, stream_id),
                        lever=f"{draft.lever_kind}.{draft.field_name or ','.join(draft.value)}",
                        settings=settings,
                        emit=True,
                    )
                except (ValueError, KeyError) as exc:
                    reasons.append(f"{stream_id}: {type(exc).__name__}: {str(exc)[:160]}")
                    decision = "block"
                    continue
                verdicts.append(verdict)
            if verdicts:
                numeric_dump = {
                    v.stream: {
                        "gate_fitted": v.fitted.passed,
                        "gate_shrunk": v.shrunk.passed,
                        "effective_n": v.effective_n,
                        "underpowered": v.underpowered,
                        "rw_dominance_after_shrunk": v.rw_dominance_after_shrunk,
                        "promotable": v.promotable,
                        "candidate_file": v.candidate_file,
                        "pooled": v.pricing_shrunk.pooled.model_dump() if v.pricing_shrunk.pooled else None,
                        "failed_conditions": [c.name for c in v.shrunk.failed_conditions],
                    }
                    for v in verdicts
                }
                gains = [v.pricing_shrunk.pooled.gain_vs_rw_pct for v in verdicts if v.pricing_shrunk.pooled]
                rank = float(np.mean(gains)) if gains else None
                if not any(v.promotable for v in verdicts):
                    reasons.append(
                        "gate: "
                        + "; ".join(
                            f"{v.stream} failed {[c.name for c in v.shrunk.failed_conditions]}{' (underpowered)' if v.underpowered else ''}"
                            for v in verdicts
                        )
                    )
                    decision = "downgrade" if decision != "block" else decision
        else:
            if h.status != "promoted":
                reasons.append(
                    f"hypothesis {h.hypothesis_id} is {h.status}: "
                    + "; ".join(ev.reasons if ev else ["no evidence summary"])
                )
                decision = "downgrade" if decision != "block" else decision
            if draft.pricing_op:
                for stream_id in draft.streams or list(corpora):
                    corpus = corpora.get(stream_id)
                    if corpus is None:
                        continue
                    engine = CounterfactualEngine(corpus, settings)
                    try:
                        if draft.pricing_op == "no_change":
                            pricing = engine.action_override(
                                {
                                    "horizon_actions_patch": {
                                        "center_action": "no_change",
                                        "uncertainty_action": "unchanged",
                                    }
                                }
                            )
                        elif draft.pricing_op == "action":
                            pricing = engine.action_override({"horizon_actions_patch": draft.pricing_params})
                        elif draft.pricing_op == "settings":
                            pricing = engine.calibration(
                                settings_overlay={draft.pricing_params["field"]: draft.pricing_params["value"]}
                            )
                        else:
                            continue
                    except Exception as exc:  # noqa: BLE001 - reported, never raised out of the gates
                        reasons.append(f"pricing failed: {type(exc).__name__}")
                        continue
                    pricing_dump = pricing.model_dump(exclude={"deltas"})
                    if pricing.pooled:
                        rank = pricing.pooled.gain_vs_rw_pct

        critic_dump = None
        if decision == "pass" and client is not None:
            critic_dump = await _critic(
                client,
                title=draft.title,
                statement=draft.statement,
                lever=lever,
                evidence=(ev.model_dump() if ev else {}),
                pricing=pricing_dump or numeric_dump,
            )
            if critic_dump.get("verdict") in ("downgrade", "block"):
                decision = critic_dump["verdict"]
                reasons.append(
                    "critic: "
                    + (critic_dump.get("hindsight") or critic_dump.get("confound") or critic_dump.get("error") or "")[
                        :300
                    ]
                )

        outcomes.append(
            GateOutcome(
                hypothesis_id=h.hypothesis_id,
                title=draft.title,
                track=h.track,
                lever_kind=draft.lever_kind,
                fingerprint=fingerprint,
                decision=decision,
                reasons=reasons,
                rank_score=rank,
                pricing=pricing_dump,
                numeric=numeric_dump,
                critic=critic_dump,
                base_share_mean=base_share,
                streams_supporting=streams_supporting,
            )
        )
    return outcomes


def _default_test_plan(kind: str) -> str:
    if kind in NUMERIC_LEVERS:
        return "Accept the candidate JSON into the coach ledger; the coached forecast applies it from the next business day; re-judge with ComparisonPolicy after 12 more live cutoffs."
    if kind in LLM_LEVERS:
        return "Register the drafted challenger stream; compare with pairing.py after 12 shared cutoffs."
    return "Implement per the brief in a Claude Code session; add a package copy and a challenger stream; pairing.py after 12 shared cutoffs."


def to_proposal(
    draft: Any,
    outcome: GateOutcome,
    hypothesis: Hypothesis,
    evidence: EvidenceSummary | None,
    *,
    proposal_id: str,
    week: str,
) -> Proposal:
    lever = {
        "kind": draft.lever_kind,
        "file_line": draft.file_line,
        "field": draft.field_name,
        "value": draft.value,
        "text_edits": [e.model_dump() for e in draft.text_edits],
        "fidelity": (outcome.pricing or {}).get("fidelity")
        if outcome.pricing
        else ("exact" if outcome.numeric else "not_priced"),
    }
    artifacts = {}
    if outcome.numeric:
        artifacts["candidate_files"] = _json.dumps({s: d.get("candidate_file") for s, d in outcome.numeric.items()})
    return Proposal(
        id=proposal_id,
        track=hypothesis.track,
        title=draft.title[:160],
        statement=draft.statement[:1200],
        mechanism=draft.mechanism[:800],
        codes=hypothesis.codes,
        seed_source=hypothesis.seed_source,
        hypothesis_id=hypothesis.hypothesis_id,
        stream_scope={
            "streams": draft.streams or outcome.streams_supporting,
            "horizons": draft.horizons,
            "predicate": hypothesis.scope,
        },
        lever=lever,
        evidence={
            **(evidence.model_dump() if evidence else {}),
            "base_share_mean": outcome.base_share_mean,
            "gate_reasons": outcome.reasons,
        },
        effect_vs_rw={"pricing": outcome.pricing, "numeric": outcome.numeric},
        rank_score=outcome.rank_score,
        test_plan=(draft.test_plan[:800] or _default_test_plan(draft.lever_kind)),
        artifacts=artifacts,
        fingerprint=outcome.fingerprint,
        created_week=week,
    )


__all__ += ["LLM_LEVERS", "NUMERIC_LEVERS", "CriticOut", "GateOutcome", "run_gates", "to_proposal"]  # noqa: PLE0605
