"""Stage D: one structured-output call per case card; annotations per newly resolved key.

The model sees the rubric, the taxonomy, a short week header, and the card, in
that order (stable to volatile). It never sees seeds, hypotheses or the proposal
register, which keeps post-ship measurement blind.

Code does the trusting: every tag's code is resolved through the taxonomy,
every pointer is checked against the card, a table-cell pointer's cited value is
checked for sign and magnitude, and anything that fails is *dropped and
recorded* -- never repaired, never guessed. A schema-invalid response gets one
correction call; a second failure is a recorded stage failure.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

from energy_oil_forecasting.cfm_coach.review.cards import CardStore, CaseCard, render_card
from energy_oil_forecasting.cfm_coach.review.collect import StreamCorpus
from energy_oil_forecasting.cfm_coach.review.llm import LlmClient, LlmFailureError, LlmResponse
from energy_oil_forecasting.cfm_coach.review.memory import Annotation, AnnotationStore, Tag, Taxonomy, annotation_id
from energy_oil_forecasting.cfm_coach.review.settings import DEFAULT_REVIEW_SETTINGS, ReviewSettings
from energy_oil_forecasting.cfm_coach.review.state import ReviewKey, ReviewState
from pydantic import BaseModel, ConfigDict, Field, ValidationError


PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
Knowability = Literal["in_packet_ignored", "knowable_missed", "precursors_knowable", "unforeseeable", "undetermined"]


# ----------------------------------------------------------------- schema ----


class TagOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    severity: int = Field(ge=1, le=3)
    confidence: float = Field(ge=0.0, le=1.0)
    pointer: str
    outcome_dependent: bool
    horizon: int | None = None
    value: float | None = None
    note: str = ""


class TriageOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tags: list[TagOut]
    went_right: list[str] = Field(default_factory=list)
    new_pattern_notes: list[str] = Field(default_factory=list)
    knowability: Knowability | None = None
    summary: str = ""


def triage_schema() -> dict[str, Any]:
    return TriageOut.model_json_schema()


# ----------------------------------------------------------------- prompt ----


def load_rubric(name: str = "triage") -> str:
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")


def prompt_hash(*parts: str) -> str:
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:12]


def build_messages(
    card: CaseCard,
    taxonomy: Taxonomy,
    *,
    horizons: tuple[int, ...],
    week_header: str = "",
    prior: dict[int, Annotation] | None = None,
    rubric: str | None = None,
) -> tuple[list[dict[str, str]], str]:
    """(messages, prompt_hash). Stable parts first so any proxy-side cache can hit."""
    rubric = rubric or load_rubric()
    stable = f"{rubric}\n\n{taxonomy.render()}"
    volatile = [f"Newly resolved horizons to tag: {list(horizons)}."]
    if week_header:
        volatile.append(week_header)
    if prior:
        volatile.append(
            "Read-only: earlier annotations for this cutoff's other horizons (do not repeat them; they are context):"
        )
        for h, ann in sorted(prior.items()):
            volatile.append(
                f"  h{h}: "
                + "; ".join(f"{t.code}@{t.pointer}" for t in ann.tags)
                + (f" | knowability {ann.knowability}" if ann.knowability else "")
            )
    volatile.append(render_card(card))
    messages = [
        {"role": "system", "content": stable},
        {"role": "user", "content": "\n\n".join(volatile)},
    ]
    return messages, prompt_hash(stable, "triage_v1")


# -------------------------------------------------------------- validation ----

_POLICY_FIELD = re.compile(r"^h(?P<h>\d+)\.(?P<field>[a-z_]+)$")
_TABLE_CELL = re.compile(r"^base\.h(?P<h>\d+)\.(?P<path>[a-z0-9_.]+)$")
_QUERY = re.compile(r"^q(?P<i>\d+)$")


def _walk(obj: Any, path: str) -> Any:
    for part in path.split("."):
        if isinstance(obj, dict):
            obj = obj.get(part)
        else:
            obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def resolve_pointer(card: CaseCard, pointer: str, kind: str) -> tuple[bool, Any]:
    """Whether `pointer` names something on the card of the right kind; returns (ok, cell value or None)."""
    rep = card.representative
    if kind == "none":
        return True, None
    if kind == "claim_id":
        return pointer in {c.claim_id for c in rep.claims}, None
    if kind == "summary_id":
        return pointer.removeprefix("summary_") in {s.summary_id for s in rep.summaries}, None
    if kind == "query_index":
        m = _QUERY.match(pointer)
        return bool(m and int(m.group("i")) < len(rep.queries)), None
    if kind == "policy_field":
        m = _POLICY_FIELD.match(pointer)
        if not m:
            return False, None
        action = next((a for a in rep.actions if a.horizon == int(m.group("h"))), None)
        return bool(action is not None and hasattr(action, m.group("field"))), None
    if kind == "table_cell":
        m = _TABLE_CELL.match(pointer)
        if not m:
            return False, None
        base = next((b for b in card.base if b.horizon == int(m.group("h"))), None)
        if base is None:
            return False, None
        path = m.group("path")
        value = _walk(base, path)
        if value is None and not path.startswith("components."):
            # The card renders component rows as "lightgbm -1.71", so a pointer of
            # `base.h5.lightgbm.bias` is the natural spelling of `components.lightgbm.bias`.
            value = _walk(base, f"components.{path}")
        return value is not None, value
    return False, None


def value_matches(cited: float | None, actual: Any) -> bool:
    """A cited table value must agree in sign and be within 50% (or $0.50) of the cell.

    Shares and coverage are stored as fractions but rendered as percentages, so a
    fraction cell also accepts the cited value read as a percentage.
    """
    if cited is None or not isinstance(actual, (int, float)) or isinstance(actual, bool):
        return True
    actual = float(actual)
    if 0.0 <= actual <= 1.0:
        # Fraction cells (shares, coverage) get a tight tolerance: the $0.50 slack that
        # suits price-denominated cells would accept almost any fraction.
        return _close(cited, actual, floor=0.05, rel=0.25) or (
            abs(cited) > 1.0 and _close(cited / 100.0, actual, floor=0.05, rel=0.25)
        )
    return _close(cited, actual, floor=0.5, rel=0.5)


def _close(cited: float, actual: float, *, floor: float, rel: float) -> bool:
    if (cited > 0) != (actual > 0) and abs(actual) > 1e-9 and abs(cited) > 1e-9:
        return False
    return abs(cited - actual) <= max(floor, rel * abs(actual))


@dataclass
class Validated:
    tags: list[Tag] = field(default_factory=list)
    tags_by_horizon: dict[int | None, list[Tag]] = field(default_factory=dict)
    dropped: list[dict[str, Any]] = field(default_factory=list)


def validate(out: TriageOut, card: CaseCard, taxonomy: Taxonomy) -> Validated:
    result = Validated()
    for raw in out.tags:
        code = taxonomy.resolve(raw.code)
        if code is None:
            result.dropped.append({"code": raw.code, "pointer": raw.pointer, "why": "unknown code"})
            continue
        ok, cell = resolve_pointer(card, raw.pointer, code.pointer_kind)
        if not ok:
            result.dropped.append(
                {"code": raw.code, "pointer": raw.pointer, "why": f"pointer not on card ({code.pointer_kind})"}
            )
            continue
        if code.pointer_kind == "table_cell" and not value_matches(raw.value, cell):
            result.dropped.append(
                {"code": raw.code, "pointer": raw.pointer, "why": f"cited value {raw.value} does not match cell {cell}"}
            )
            continue
        tag = Tag(
            code=code.code,
            track=code.track,
            severity=raw.severity,
            confidence=raw.confidence,
            pointer=raw.pointer.removeprefix("summary_") if code.pointer_kind == "summary_id" else raw.pointer,
            outcome_dependent=code.outcome_dependent,
            note=raw.note[:300],
        )
        result.tags.append(tag)
        result.tags_by_horizon.setdefault(raw.horizon, []).append(tag)
    return result


# -------------------------------------------------------------- the call ----


class TriageFailureError(RuntimeError):
    pass


async def call_with_correction(
    client: LlmClient, messages: list[dict[str, str]], *, stage: str = "triage"
) -> tuple[TriageOut, LlmResponse]:
    """Parse; on a schema error ask once for a corrected response; then fail. Never fabricate."""
    response = await client.complete(
        stage=stage, messages=messages, response_schema=triage_schema(), schema_name="TriageOut"
    )
    try:
        if not response.content:
            raise ValueError("empty content")
        return TriageOut.model_validate_json(response.content), response
    except (ValidationError, ValueError, json.JSONDecodeError) as first:
        correction = [
            *messages,
            {"role": "assistant", "content": response.content or ""},
            {
                "role": "user",
                "content": f"Your response did not match the schema: {str(first)[:600]}. Reply again with only the JSON object, corrected. Do not add tags.",
            },
        ]
        response = await client.complete(
            stage=stage, messages=correction, response_schema=triage_schema(), schema_name="TriageOut"
        )
        try:
            if not response.content:
                raise ValueError("empty content")
            return TriageOut.model_validate_json(response.content), response
        except (ValidationError, ValueError, json.JSONDecodeError) as second:
            raise TriageFailureError(f"schema-invalid after one correction: {str(second)[:300]}") from second


@dataclass
class TriageResult:
    card: CaseCard
    keys: list[ReviewKey]
    annotations: list[Annotation]
    dropped: list[dict[str, Any]]
    failure: str | None = None
    raw: dict[str, Any] | None = None


async def triage_card(
    client: LlmClient,
    card: CaseCard,
    keys: list[ReviewKey],
    *,
    taxonomy: Taxonomy,
    week: str,
    prior: dict[int, Annotation] | None = None,
    week_header: str = "",
    stage: Literal["triage", "blind_retriage"] = "triage",
    supersedes: dict[ReviewKey, str] | None = None,
) -> TriageResult:
    horizons = tuple(sorted(k.horizon for k in keys))
    messages, phash = build_messages(card, taxonomy, horizons=horizons, week_header=week_header, prior=prior)
    try:
        out, response = await call_with_correction(client, messages, stage="triage")
    except (TriageFailureError, LlmFailureError) as exc:
        return TriageResult(card=card, keys=keys, annotations=[], dropped=[], failure=str(exc))
    checked = validate(out, card, taxonomy)
    annotations = []
    for key in keys:
        tags = [*checked.tags_by_horizon.get(None, []), *checked.tags_by_horizon.get(key.horizon, [])]
        annotations.append(
            Annotation(
                annotation_id=annotation_id(key, card.card_hash, phash, stage),
                stream=key.stream,
                cutoff=key.cutoff,
                horizon=key.horizon,
                card_hash=card.card_hash,
                card_version=card.card_version,
                prompt_hash=phash,
                stage=stage,
                week=week,
                model=response.usage.model,
                tags=tags,
                went_right=[c for c in out.went_right if taxonomy.resolve(c)],
                new_pattern_notes=[n[:300] for n in out.new_pattern_notes][:5],
                knowability=out.knowability,
                dropped=checked.dropped,
                supersedes=(supersedes or {}).get(key),
            )
        )
    return TriageResult(card=card, keys=keys, annotations=annotations, dropped=checked.dropped, raw=response.raw)


async def run_triage(
    client: LlmClient,
    *,
    corpus: StreamCorpus,
    cards: dict[date, CaseCard],
    state: ReviewState,
    store: AnnotationStore,
    taxonomy: Taxonomy,
    week: str,
    settings: ReviewSettings = DEFAULT_REVIEW_SETTINGS,
    week_header: str = "",
) -> list[TriageResult]:
    """Triage every card whose newly resolved keys are still `resolved`; advance them to `triaged`."""
    current = store.current(corpus.stream_id, stage="triage")
    jobs = []
    for cutoff, card in sorted(cards.items()):
        if card.card_version < 1:
            continue
        keys = [k for k in state.keys_in("resolved") if k.stream == corpus.stream_id and k.cutoff == cutoff]
        if not keys:
            continue
        prior = {
            k.horizon: a
            for k, a in current.items()
            if k.cutoff == cutoff and k.horizon not in {x.horizon for x in keys}
        }
        jobs.append(triage_card(client, card, keys, taxonomy=taxonomy, week=week, prior=prior, week_header=week_header))
    results = await asyncio.gather(*jobs)
    for result in results:
        if result.failure:
            continue
        for ann in result.annotations:
            store.append(ann)
            state.advance(
                ann.key,
                "triaged",
                annotation_id=ann.annotation_id,
                card_hash=ann.card_hash,
                card_version=ann.card_version,
            )
    return list(results)


# ------------------------------------------------------------ revalidate ----


def revalidate_week(
    settings: ReviewSettings,
    week: str,
    *,
    taxonomy: Taxonomy,
    streams: Sequence[str] | None = None,
) -> dict[str, tuple[int, int]]:
    """Re-run pointer validation on a week's stored raw triage responses and rewrite the annotations.

    The model's raw output is kept under ``weeks/<week>/raw/<stream>/triage_<cutoff>.json``,
    so a change to the validation rules (a new pointer alias, a tolerance) can be applied
    to every card already triaged without another LLM call. Annotation ids do not change:
    they hash the key, card and prompt, none of which validation touches. Returns
    ``{stream: (tags_before, tags_after)}``.
    """
    cards = CardStore(settings)
    raw_root = settings.weeks_dir / week / "raw"
    store = AnnotationStore(settings)
    out: dict[str, tuple[int, int]] = {}
    for stream in streams or sorted(p.name for p in raw_root.iterdir() if p.is_dir()):
        raws = {
            p.stem.split("_", 1)[1]: json.loads(p.read_text(encoding="utf-8"))
            for p in (raw_root / stream).glob("triage_*.json")
        }
        path = store.path_for(stream)
        if not raws or not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        before = after = 0
        for row in rows:
            if row.get("stage") != "triage" or row["cutoff"] not in raws or not raws[row["cutoff"]].get("content"):
                continue
            card = cards.load(stream, date.fromisoformat(row["cutoff"]))
            if card is None or card.card_hash != row["card_hash"]:
                continue
            checked = validate(TriageOut.model_validate_json(raws[row["cutoff"]]["content"]), card, taxonomy)
            tags = [*checked.tags_by_horizon.get(None, []), *checked.tags_by_horizon.get(int(row["horizon"]), [])]
            before += len(row["tags"])
            after += len(tags)
            row["tags"] = [t.model_dump(mode="json") for t in tags]
            row["dropped"] = checked.dropped
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        out[stream] = (before, after)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Triage utilities (no LLM calls).")
    parser.add_argument("--revalidate", action="store_true", help="re-run pointer validation on stored raw responses")
    parser.add_argument("--week", required=True, help="ISO week id, e.g. 2026-W37")
    parser.add_argument("--stream", action="append", default=None)
    args = parser.parse_args(argv)
    if not args.revalidate:
        parser.error("nothing to do; pass --revalidate")
    settings = ReviewSettings()
    for stream, (before, after) in revalidate_week(
        settings, args.week, taxonomy=Taxonomy.load(), streams=args.stream
    ).items():
        print(f"{stream}: tags {before} -> {after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PROMPTS_DIR",
    "TagOut",
    "TriageFailureError",
    "TriageOut",
    "TriageResult",
    "build_messages",
    "call_with_correction",
    "load_rubric",
    "resolve_pointer",
    "revalidate_week",
    "run_triage",
    "triage_card",
    "triage_schema",
    "validate",
    "value_matches",
]
