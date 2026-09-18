"""The coach's own report card: does it see what it claims to see?

- **blind re-triage** -- a sample of already-triaged keys is triaged again with
  the outcome hidden (no numerical-base block, no resolved-horizon pinball). Tags
  that appear only when the outcome is visible are hindsight-shaped.
- **tag reliability** -- per code, agreement between the sighted and blind
  passes (Jaccard over codes per key).
- **predictive precision** -- for each promoted hypothesis, of the in-scope keys
  resolved after promotion, how many carried one of its codes.

Sampled at `blind_retriage_fraction`; reported from the third review week.
"""

from __future__ import annotations

import json
import random
from datetime import date
from typing import Any

from energy_oil_forecasting.cfm_coach.review.cards import CaseCard
from energy_oil_forecasting.cfm_coach.review.llm import LlmClient
from energy_oil_forecasting.cfm_coach.review.memory import Annotation, AnnotationStore, Hypothesis, Taxonomy
from energy_oil_forecasting.cfm_coach.review.settings import DEFAULT_REVIEW_SETTINGS, ReviewSettings
from energy_oil_forecasting.cfm_coach.review.state import ReviewKey
from energy_oil_forecasting.cfm_coach.review.triage import triage_card


def blind_card(card: CaseCard) -> CaseCard:
    """The card with every outcome-bearing field removed."""
    dispersion = [
        row.model_copy(
            update={
                "pinball_by_granted_center": {},
                "best_granted_center": None,
                "pinball_min": None,
                "pinball_max": None,
            }
        )
        for row in card.dispersion
    ]
    return card.model_copy(update={"base": [], "dispersion": dispersion, "resolved_horizons": []})


async def blind_retriage(
    client: LlmClient,
    *,
    cards: dict[tuple[str, date], CaseCard],
    store: AnnotationStore,
    taxonomy: Taxonomy,
    week: str,
    settings: ReviewSettings = DEFAULT_REVIEW_SETTINGS,
    seed: int = 20260913,
) -> list[Annotation]:
    """Re-triage a sample of triaged keys with outcomes hidden; annotations are stored under stage `blind_retriage`."""
    rng = random.Random(seed + hash(week) % 1000)
    written: list[Annotation] = []
    for stream in {k[0] for k in cards}:
        sighted = store.current(stream, stage="triage")
        blind = store.current(stream, stage="blind_retriage")
        pool = [k for k in sighted if k not in blind and (k.stream, k.cutoff) in cards]
        n = max(1, round(len(pool) * settings.blind_retriage_fraction)) if pool else 0
        for key in rng.sample(pool, min(n, len(pool))):
            card = blind_card(cards[(key.stream, key.cutoff)])
            result = await triage_card(
                client,
                card,
                [key],
                taxonomy=taxonomy,
                week=week,
                stage="blind_retriage",
                week_header="(outcome hidden)",
            )
            if result.failure:
                continue
            for ann in result.annotations:
                store.append(ann)
                written.append(ann)
    return written


def tag_reliability(store: AnnotationStore, streams: list[str]) -> dict[str, Any]:
    """Agreement between sighted and blind triage on the same keys."""
    per_code: dict[str, dict[str, int]] = {}
    jaccards = []
    hindsight_only = 0
    pairs = 0
    for stream in streams:
        sighted = store.current(stream, stage="triage")
        blind = store.current(stream, stage="blind_retriage")
        for key, b in blind.items():
            s = sighted.get(key)
            if s is None:
                continue
            pairs += 1
            a_codes = {t.code for t in s.tags}
            b_codes = {t.code for t in b.tags}
            union = a_codes | b_codes
            jaccards.append(len(a_codes & b_codes) / len(union) if union else 1.0)
            hindsight_only += len(a_codes - b_codes)
            for code in union:
                row = per_code.setdefault(code, {"sighted": 0, "blind": 0, "both": 0})
                row["sighted"] += code in a_codes
                row["blind"] += code in b_codes
                row["both"] += code in a_codes and code in b_codes
    return {
        "pairs": pairs,
        "mean_jaccard": (sum(jaccards) / len(jaccards)) if jaccards else None,
        "hindsight_only_tags": hindsight_only,
        "per_code": per_code,
    }


def predictive_precision(
    hypotheses: dict[str, Hypothesis], annotations: dict[ReviewKey, Annotation], *, weeks_order: list[str]
) -> dict[str, Any]:
    """Of keys annotated after a hypothesis was promoted and in its scope, the share carrying one of its codes."""
    out: dict[str, Any] = {}
    for hid, h in hypotheses.items():
        if h.status != "promoted" or not h.updated_week:
            continue
        promoted_at = h.updated_week
        later = [
            a
            for a in annotations.values()
            if a.week > promoted_at
            and (not h.scope.get("streams") or a.stream in h.scope["streams"])
            and (not h.scope.get("horizons") or a.horizon in h.scope["horizons"])
        ]
        if not later:
            out[hid] = {"keys_after_promotion": 0, "precision": None}
            continue
        hits = sum(1 for a in later if any(t.code in h.codes for t in a.tags))
        out[hid] = {"keys_after_promotion": len(later), "precision": hits / len(later)}
    return out


def write_metrics(settings: ReviewSettings, week: str, payload: dict[str, Any]) -> None:
    path = settings.data_dir / "coach_metrics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"week": week, **payload}, default=str) + "\n")


__all__ = ["blind_card", "blind_retriage", "predictive_precision", "tag_reliability", "write_metrics"]
