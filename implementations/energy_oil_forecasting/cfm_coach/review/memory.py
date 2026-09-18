"""The coach's memory: annotations, hypotheses, proposals, taxonomy, playbook, seeds.

Four stores, all files, all append-only where the LLM writes and human-edited
only where a human decides:

- **annotations** ``annotations/<stream>.jsonl`` -- what triage and deep dives said
  about one (stream, cutoff, horizon), one line per annotation, ``supersedes`` for
  re-triage. Ids are content hashes, so re-running a week writes nothing new.
- **hypotheses** ``hypotheses.jsonl`` -- an event log folded into state in memory.
  Event sourcing at this size buys nothing; a fold is thirty lines and auditable.
- **proposals** ``cfm_coach/proposals/P-NNNN.yaml`` -- the one proposal store, shared
  with `fit_width`. Only ``decision`` is human-edited; the file is schema-validated
  on load and an invalid one fails loudly rather than being skipped.
- **taxonomy** / **playbook** / **seeds** -- YAML and markdown read at the start of
  every review. Static; edited by people.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from energy_oil_forecasting.cfm_coach.review.settings import DEFAULT_REVIEW_SETTINGS, ReviewSettings
from energy_oil_forecasting.cfm_coach.review.state import ReviewKey
from pydantic import BaseModel, ConfigDict, Field, ValidationError


REVIEW_PACKAGE_DIR = Path(__file__).resolve().parent
TAXONOMY_PATH = REVIEW_PACKAGE_DIR / "taxonomy.yaml"
SEEDS_PATH = REVIEW_PACKAGE_DIR / "seeds.yaml"
PLAYBOOK_PATH = REVIEW_PACKAGE_DIR / "playbook.md"

Track = Literal["numeric", "policy", "llm", "code", "data", "none"]
TRACKS: tuple[str, ...] = ("numeric", "policy", "llm", "code", "data", "none")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def content_id(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ---------------------------------------------------------------- taxonomy ----


class Code(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    track: Track
    outcome_dependent: bool
    description: str
    pointer_kind: Literal["table_cell", "policy_field", "claim_id", "summary_id", "query_index", "none"]


class Taxonomy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int
    codes: list[Code]
    aliases: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: Path = TAXONOMY_PATH) -> Taxonomy:
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    @property
    def by_code(self) -> dict[str, Code]:
        return {c.code: c for c in self.codes}

    def resolve(self, code: str) -> Code | None:
        """A code or any alias of it; None when the taxonomy does not know it."""
        seen: set[str] = set()
        while code in self.aliases and code not in seen:
            seen.add(code)
            code = self.aliases[code]
        return self.by_code.get(code)

    def render(self) -> str:
        lines = [f"## Taxonomy v{self.version} (use only these codes; anything else goes in new_pattern_notes)"]
        for c in self.codes:
            lines.append(
                f"- {c.code} [{c.track}; {'outcome' if c.outcome_dependent else 'process'}; pointer: {c.pointer_kind}] {c.description}"
            )
        return "\n".join(lines)


# ------------------------------------------------------------- annotations ----


class Tag(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    track: Track
    severity: int = Field(ge=1, le=3)
    confidence: float = Field(ge=0.0, le=1.0)
    pointer: str
    outcome_dependent: bool
    note: str = ""


class Annotation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    annotation_id: str
    stream: str
    cutoff: date
    horizon: int
    card_hash: str
    card_version: int
    prompt_hash: str
    stage: Literal["triage", "deep_dive", "blind_retriage", "seed"]
    week: str
    model: str
    tags: list[Tag]
    went_right: list[str] = Field(default_factory=list)
    new_pattern_notes: list[str] = Field(default_factory=list)
    knowability: str | None = None
    #: pointer-less or fabricated tags that code dropped, kept for the self-metrics
    dropped: list[dict[str, Any]] = Field(default_factory=list)
    created_at: str = Field(default_factory=_now)
    supersedes: str | None = None

    @property
    def key(self) -> ReviewKey:
        return ReviewKey(stream=self.stream, cutoff=self.cutoff, horizon=self.horizon)


def annotation_id(key: ReviewKey, card_hash: str, prompt_hash: str, stage: str) -> str:
    return content_id(str(key), card_hash, prompt_hash, stage)


class AnnotationStore:
    """One JSONL per stream. Append only; the latest non-superseded annotation per key wins."""

    def __init__(self, settings: ReviewSettings = DEFAULT_REVIEW_SETTINGS):
        self.settings = settings

    def path_for(self, stream: str) -> Path:
        return self.settings.annotations_dir / f"{stream}.jsonl"

    def load(self, stream: str) -> list[Annotation]:
        path = self.path_for(stream)
        if not path.exists():
            return []
        return [
            Annotation.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def ids(self, stream: str) -> set[str]:
        return {a.annotation_id for a in self.load(stream)}

    def append(self, annotation: Annotation) -> bool:
        """Write unless an annotation with this id already exists. Returns whether it wrote."""
        if annotation.annotation_id in self.ids(annotation.stream):
            return False
        path = self.path_for(annotation.stream)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(annotation.model_dump_json() + "\n")
        return True

    def current(self, stream: str, *, stage: str | None = None) -> dict[ReviewKey, Annotation]:
        """Latest annotation per key, honouring `supersedes`."""
        superseded = {a.supersedes for a in self.load(stream) if a.supersedes}
        out: dict[ReviewKey, Annotation] = {}
        for a in self.load(stream):
            if a.annotation_id in superseded or (stage is not None and a.stage != stage):
                continue
            out[a.key] = a
        return out


# -------------------------------------------------------------- hypotheses ----

HypothesisStatus = Literal["candidate", "promoted", "stale", "retired", "refuted", "merged"]


class Hypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hypothesis_id: str
    statement: str
    mechanism: str
    track: Track
    lever: dict[str, Any]
    codes: list[str]
    scope: dict[str, Any]
    prediction: str
    seed_source: str | None = None
    post_cutoff: bool = False
    status: HypothesisStatus = "candidate"
    supporting: list[str] = Field(default_factory=list)  # ReviewKey strings
    contradicting: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    reviews_seen: int = 0
    reviews_without_new_support: int = 0
    created_week: str | None = None
    updated_week: str | None = None
    merged_into: str | None = None


class HypothesisEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event: Literal["created", "supported", "contradicted", "evidence", "status", "merged"]
    hypothesis_id: str
    week: str
    at: str = Field(default_factory=_now)
    payload: dict[str, Any] = Field(default_factory=dict)


class HypothesisStore:
    """`hypotheses.jsonl` folded into `Hypothesis` state. Append events; never rewrite."""

    def __init__(self, settings: ReviewSettings = DEFAULT_REVIEW_SETTINGS):
        self.settings = settings
        self.path = settings.hypotheses_path

    def events(self) -> list[HypothesisEvent]:
        if not self.path.exists():
            return []
        return [
            HypothesisEvent.model_validate_json(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def append(self, event: HypothesisEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json() + "\n")

    def state(self) -> dict[str, Hypothesis]:
        out: dict[str, Hypothesis] = {}
        for ev in self.events():
            if ev.event == "created":
                out[ev.hypothesis_id] = Hypothesis(
                    hypothesis_id=ev.hypothesis_id, created_week=ev.week, updated_week=ev.week, **ev.payload
                )
                continue
            h = out.get(ev.hypothesis_id)
            if h is None:
                raise ValueError(f"event {ev.event!r} for unknown hypothesis {ev.hypothesis_id!r}")
            update: dict[str, Any] = {"updated_week": ev.week}
            if ev.event == "supported":
                update["supporting"] = sorted(set(h.supporting) | set(ev.payload.get("keys", [])))
            elif ev.event == "contradicted":
                update["contradicting"] = sorted(set(h.contradicting) | set(ev.payload.get("keys", [])))
            elif ev.event == "evidence":
                update["evidence"] = {**h.evidence, **ev.payload}
                update["reviews_seen"] = h.reviews_seen + 1
                update["reviews_without_new_support"] = (
                    0 if ev.payload.get("new_support") else h.reviews_without_new_support + 1
                )
            elif ev.event == "status":
                update["status"] = ev.payload["status"]
            elif ev.event == "merged":
                update["status"] = "merged"
                update["merged_into"] = ev.payload["into"]
            out[ev.hypothesis_id] = h.model_copy(update=update)
        return out

    def create(self, hypothesis: Hypothesis, *, week: str) -> None:
        if hypothesis.hypothesis_id in self.state():
            raise ValueError(f"hypothesis {hypothesis.hypothesis_id!r} already exists; append events instead")
        payload = hypothesis.model_dump(exclude={"hypothesis_id", "created_week", "updated_week"})
        self.append(
            HypothesisEvent(event="created", hypothesis_id=hypothesis.hypothesis_id, week=week, payload=payload)
        )

    def next_id(self) -> str:
        n = sum(1 for h in self.state() if h.startswith("H-"))
        return f"H-{n + 1:04d}"

    def render_active(self, *, max_chars: int = 24_000) -> str:
        """Compact projection for synthesis: active hypotheses in full, retired ones as one line."""
        lines = ["## Hypotheses"]
        for h in sorted(self.state().values(), key=lambda x: x.hypothesis_id):
            if h.status in ("retired", "refuted", "merged"):
                lines.append(f"- {h.hypothesis_id} [{h.status}] {h.statement[:100]}")
                continue
            lines.append(
                f"- {h.hypothesis_id} [{h.status}; {h.track}; codes {h.codes}; seed {h.seed_source or '-'}] {h.statement}\n"
                f"    mechanism: {h.mechanism}\n    lever: {h.lever}\n    scope: {h.scope}\n    prediction: {h.prediction}\n"
                f"    support {len(h.supporting)} / contradict {len(h.contradicting)}; evidence {h.evidence}"
            )
        text = "\n".join(lines)
        return text[:max_chars]


# --------------------------------------------------------------- proposals ----

DecisionStatus = Literal["open", "accepted", "rejected", "deferred"]


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: DecisionStatus = "open"
    by: str | None = None
    on: date | None = None
    reason: str | None = None
    applied_as: dict[str, Any] = Field(default_factory=dict)
    post_ship: dict[str, Any] = Field(default_factory=dict)


class Proposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    track: Track
    title: str
    statement: str
    mechanism: str
    codes: list[str]
    seed_source: str | None = None
    hypothesis_id: str | None = None
    stream_scope: dict[str, Any]
    lever: dict[str, Any]
    evidence: dict[str, Any] = Field(default_factory=dict)
    effect_vs_rw: dict[str, Any] = Field(default_factory=dict)
    rank_score: float | None = None
    test_plan: str
    artifacts: dict[str, str] = Field(default_factory=dict)
    fingerprint: str
    created_week: str
    decision: Decision = Field(default_factory=Decision)

    @staticmethod
    def make_fingerprint(lever: dict[str, Any], codes: list[str]) -> str:
        target = lever.get("file") or lever.get("file_line") or lever.get("field") or lever.get("kind")
        return content_id(lever.get("kind"), target, sorted(codes))


class ProposalStore:
    """``proposals/P-NNNN.yaml`` + ``index.jsonl`` + ``briefs/``. Invalid YAML fails loudly."""

    def __init__(self, settings: ReviewSettings = DEFAULT_REVIEW_SETTINGS):
        self.settings = settings
        self.directory = settings.proposals_dir

    @property
    def briefs_dir(self) -> Path:
        return self.directory / "briefs"

    def path_for(self, proposal_id: str) -> Path:
        return self.directory / f"{proposal_id}.yaml"

    def load_all(self) -> list[Proposal]:
        if not self.directory.exists():
            return []
        out = []
        for path in sorted(self.directory.glob("P-*.yaml")):
            try:
                out.append(Proposal.model_validate(yaml.safe_load(path.read_text(encoding="utf-8"))))
            except (ValidationError, yaml.YAMLError) as exc:
                raise ValueError(f"proposal {path.name} is invalid: {exc}") from exc
        return out

    def next_id(self) -> str:
        existing = [int(p.stem[2:]) for p in self.directory.glob("P-*.yaml")] if self.directory.exists() else []
        return f"P-{max(existing, default=0) + 1:04d}"

    def save(self, proposal: Proposal) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path_for(proposal.id)
        if path.exists():
            raise FileExistsError(f"{path.name} exists; a proposal is written once and only its decision is edited")
        path.write_text(
            yaml.safe_dump(proposal.model_dump(mode="json"), sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        with (self.directory / "index.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "id": proposal.id,
                        "track": proposal.track,
                        "title": proposal.title,
                        "fingerprint": proposal.fingerprint,
                        "week": proposal.created_week,
                    }
                )
                + "\n"
            )
        return path

    def rejected(self) -> dict[str, Proposal]:
        return {p.fingerprint: p for p in self.load_all() if p.decision.status == "rejected"}

    def blocked(self, fingerprint: str, *, new_cutoffs_since_rejection: int) -> tuple[bool, str]:
        """A rejected fingerprint may be re-proposed only with enough new distinct cutoffs after the rejection."""
        prior = self.rejected().get(fingerprint)
        if prior is None:
            return False, ""
        needed = self.settings.reproposal_new_cutoffs
        if new_cutoffs_since_rejection >= needed:
            return (
                False,
                f"re-proposal allowed: {new_cutoffs_since_rejection} new cutoffs since {prior.id} was rejected",
            )
        return (
            True,
            f"blocked: {prior.id} was rejected ({prior.decision.reason or 'no reason'}); {new_cutoffs_since_rejection}/{needed} new cutoffs since",
        )

    def render_register(self) -> str:
        lines = ["## Proposal register (open and rejected)"]
        for p in self.load_all():
            if p.decision.status in ("open", "rejected", "deferred"):
                lines.append(
                    f"- {p.id} [{p.track}; {p.decision.status}] {p.title} -- lever {p.lever.get('kind')} {p.lever.get('file') or p.lever.get('field') or ''}"
                    + (f" -- reason: {p.decision.reason}" if p.decision.reason else "")
                )
        return "\n".join(lines)

    def write_brief(self, brief_id: str, text: str) -> Path:
        self.briefs_dir.mkdir(parents=True, exist_ok=True)
        path = self.briefs_dir / f"{brief_id}.md"
        path.write_text(text, encoding="utf-8")
        return path

    def next_brief_id(self) -> str:
        existing = [int(p.stem[2:]) for p in self.briefs_dir.glob("B-*.md")] if self.briefs_dir.exists() else []
        return f"B-{max(existing, default=0) + 1:04d}"


# -------------------------------------------------------------------- seeds ----


class Seed(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    seed_source: str
    track: Track
    statement: str
    mechanism: str
    lever: dict[str, Any]
    codes: list[str]
    scope: dict[str, Any]
    prediction: str
    evidence_stat: str = "none"


def load_seeds(path: Path = SEEDS_PATH) -> list[Seed]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [Seed.model_validate(item) for item in payload["seeds"]]


def seed_hypotheses(
    store: HypothesisStore, seeds: list[Seed], taxonomy: Taxonomy, *, week: str, evidence: dict[str, Any] | None = None
) -> list[str]:
    """Create one hypothesis per seed not yet present. Codes must exist in the taxonomy."""
    existing = store.state()
    created = []
    for seed in seeds:
        unknown = [c for c in seed.codes if taxonomy.resolve(c) is None]
        if unknown:
            raise ValueError(f"seed {seed.id} names unknown taxonomy code(s) {unknown}")
        if seed.id in existing:
            continue
        stat = (evidence or {}).get(seed.evidence_stat)
        store.create(
            Hypothesis(
                hypothesis_id=seed.id,
                statement=seed.statement,
                mechanism=seed.mechanism,
                track=seed.track,
                lever=seed.lever,
                codes=seed.codes,
                scope=seed.scope,
                prediction=seed.prediction,
                seed_source=seed.seed_source,
                evidence={"seed_stat": seed.evidence_stat, "seed_value": stat} if stat is not None else {},
            ),
            week=week,
        )
        created.append(seed.id)
    return created


def load_playbook(path: Path = PLAYBOOK_PATH, *, max_chars: int = 6_000) -> str:
    return path.read_text(encoding="utf-8")[:max_chars] if path.exists() else ""


__all__ = [
    "PLAYBOOK_PATH",
    "SEEDS_PATH",
    "TAXONOMY_PATH",
    "TRACKS",
    "Annotation",
    "AnnotationStore",
    "Code",
    "Decision",
    "Hypothesis",
    "HypothesisEvent",
    "HypothesisStore",
    "Proposal",
    "ProposalStore",
    "Seed",
    "Tag",
    "Taxonomy",
    "annotation_id",
    "content_id",
    "load_playbook",
    "load_seeds",
    "seed_hypotheses",
]
