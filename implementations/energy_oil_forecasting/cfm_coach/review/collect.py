"""Stage A: load a stream's records, resolve them, score the three reference variants.

Nothing here is new arithmetic. Records come from `RunRecordStore`, outcomes from
`OutcomeResolver`, scores from `ForecastScorer` -- the same three the scoreboard
uses -- so a number on a case card is the number `report.py` would print.

The one thing added is *readiness*: which (stream, cutoff, horizon) keys have
resolved since the last review, and which cutoffs are ready for a card at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from energy_oil_forecasting.cfm_coach.baselines import RandomWalkBaseline
from energy_oil_forecasting.cfm_coach.outcomes import OutcomeResolver, ResolutionReport
from energy_oil_forecasting.cfm_coach.report import AGENT, ENSEMBLE, RANDOM_WALK
from energy_oil_forecasting.cfm_coach.review.settings import DEFAULT_REVIEW_SETTINGS, ReviewSettings
from energy_oil_forecasting.cfm_coach.review.state import ReviewKey, ReviewState
from energy_oil_forecasting.cfm_coach.run_store import RunRecordStore
from energy_oil_forecasting.cfm_coach.schemas import ResolvedForecast, RunRecord, ScoreCard
from energy_oil_forecasting.cfm_coach.scoring import ForecastScorer
from energy_oil_forecasting.cfm_coach.streams import SCHEDULED_STREAMS, RunStream


COACHED = "coached"
REFERENCE_VARIANTS = (RANDOM_WALK, ENSEMBLE, AGENT)


@dataclass
class StreamCorpus:
    """One stream, loaded, resolved and scored. The input every later stage reads."""

    stream_id: str
    agent_id: str
    model: str
    records: list[RunRecord]
    resolution: ResolutionReport
    #: Every reference-variant card, keyed by (run_id, horizon, variant).
    scores: dict[tuple[str, int, str], ScoreCard]
    baseline: RandomWalkBaseline | None
    data_through: date | None
    runs_by_cutoff: dict[date, list[RunRecord]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        by_cutoff: dict[date, list[RunRecord]] = {}
        for record in self.records:
            by_cutoff.setdefault(record.cutoff, []).append(record)
        self.runs_by_cutoff = dict(sorted(by_cutoff.items()))
        self._resolved: dict[tuple[str, int], ResolvedForecast] = {
            (item.run_id, item.horizon): item for item in self.resolution.resolved
        }

    def resolved(self, run_id: str, horizon: int) -> ResolvedForecast | None:
        return self._resolved.get((run_id, horizon))

    def score(self, run_id: str, horizon: int, variant: str) -> ScoreCard | None:
        return self.scores.get((run_id, horizon, variant))

    def resolved_horizons(self, cutoff: date) -> set[int]:
        """Horizons resolved for *every* run at `cutoff`.

        Runs at one cutoff share forecast dates, so this is all-or-nothing in
        practice; requiring all of them keeps a partially resolved cutoff (one
        record refused as stale) from being read as ready.
        """
        runs = self.runs_by_cutoff.get(cutoff, [])
        if not runs:
            return set()
        horizons: set[int] | None = None
        for record in runs:
            mine = {h for h in record.horizons if self.resolved(record.run_id, h) is not None}
            horizons = mine if horizons is None else horizons & mine
        return horizons or set()

    @property
    def cutoffs(self) -> list[date]:
        return list(self.runs_by_cutoff)


def score_reference_variants(
    records: list[RunRecord],
    resolution: ResolutionReport,
    baseline: RandomWalkBaseline | None,
    *,
    scorer: ForecastScorer | None = None,
) -> dict[tuple[str, int, str], ScoreCard]:
    """random_walk / ensemble / agent on every resolved horizon -- what `score_corpus` does, minus `current`."""
    scorer = scorer or ForecastScorer()
    by_id = {record.run_id: record for record in records}
    out: dict[tuple[str, int, str], ScoreCard] = {}
    for resolved in resolution.resolved:
        record = by_id[resolved.run_id]
        if baseline is not None:
            rw = scorer.score_random_walk(record, resolved, baseline, variant=RANDOM_WALK)
            if rw is not None:
                out[(resolved.run_id, resolved.horizon, RANDOM_WALK)] = rw
        out[(resolved.run_id, resolved.horizon, ENSEMBLE)] = scorer.score_ensemble(record, resolved, variant=ENSEMBLE)
        out[(resolved.run_id, resolved.horizon, AGENT)] = scorer.score_recorded(record, resolved, variant=AGENT)
    return out


def score_coached(
    records: list[RunRecord],
    resolution: ResolutionReport,
    coached: dict[str, RunRecord],
    *,
    scorer: ForecastScorer | None = None,
) -> dict[tuple[str, int, str], ScoreCard]:
    """The `coached` variant: each source run's coached sibling, scored on the source's resolved horizons."""
    scorer = scorer or ForecastScorer()
    out: dict[tuple[str, int, str], ScoreCard] = {}
    for resolved in resolution.resolved:
        sibling = coached.get(resolved.run_id)
        if sibling is None:
            continue
        out[(resolved.run_id, resolved.horizon, COACHED)] = scorer.score_recorded(sibling, resolved, variant=COACHED)
    return out


def collect_stream(
    stream: RunStream,
    *,
    records: list[RunRecord] | None = None,
    resolver: OutcomeResolver | None = None,
    review_settings: ReviewSettings | None = None,
) -> StreamCorpus:
    """Load, resolve and score one stream. `records`/`resolver` are injectable for tests.

    When `review_settings` is given and a coached corpus exists for the stream,
    the coached sibling of every resolved run is scored as the ``coached`` variant.
    """
    settings = stream.coach_settings
    if records is None:
        records = RunRecordStore(settings).load_all()
    resolver = resolver or OutcomeResolver(settings)
    resolution = resolver.resolve_all(records)
    baseline = RandomWalkBaseline(resolver.price_series()) if records else None
    scores = score_reference_variants(records, resolution, baseline)
    if review_settings is not None:
        from energy_oil_forecasting.cfm_coach.review.coached import load_coached  # noqa: PLC0415

        coached = load_coached(stream, review_settings)
        if coached:
            scores.update(score_coached(records, resolution, coached))
    return StreamCorpus(
        stream_id=stream.stream_id,
        agent_id=stream.target.agent_id,
        model=stream.model,
        records=records,
        resolution=resolution,
        scores=scores,
        baseline=baseline,
        data_through=resolution.data_through,
    )


def collect_all(
    streams: tuple[RunStream, ...] = SCHEDULED_STREAMS,
    *,
    resolver: OutcomeResolver | None = None,
    review_settings: ReviewSettings | None = None,
) -> dict[str, StreamCorpus]:
    return {
        stream.stream_id: collect_stream(stream, resolver=resolver, review_settings=review_settings)
        for stream in streams
    }


# -- readiness ----------------------------------------------------------------


@dataclass(frozen=True)
class Readiness:
    """Which cutoffs can be carded, and at which card version, given what has resolved."""

    #: cutoff -> card version (1 once `triage_after_horizons` resolved, 2 once `rescore_at_horizons` did too)
    versions: dict[date, int]
    #: keys that resolved and have not been triaged
    new_keys: list[ReviewKey]

    def cutoffs_at(self, version: int) -> list[date]:
        return sorted(cutoff for cutoff, v in self.versions.items() if v >= version)


def readiness(
    corpus: StreamCorpus, state: ReviewState, settings: ReviewSettings = DEFAULT_REVIEW_SETTINGS
) -> Readiness:
    """Mark newly resolved keys in `state` and say which cutoffs are ready for a card.

    Only advances keys ``pending -> resolved``; triage advances them further.
    """
    versions: dict[date, int] = {}
    new_keys: list[ReviewKey] = []
    first = set(settings.triage_after_horizons)
    second = set(settings.rescore_at_horizons)
    for cutoff in corpus.cutoffs:
        resolved = corpus.resolved_horizons(cutoff)
        for horizon in sorted(resolved):
            key = ReviewKey(stream=corpus.stream_id, cutoff=cutoff, horizon=horizon)
            item = state.get(key)
            if item.status == "pending":
                state.advance(key, "resolved")
            if item.status == "resolved":
                new_keys.append(key)
        if first <= resolved:
            versions[cutoff] = 2 if second <= resolved else 1
    return Readiness(versions=versions, new_keys=sorted(new_keys))


def describe(corpus: StreamCorpus) -> str:
    n_res = len(corpus.resolution.resolved)
    return (
        f"{corpus.stream_id}: {len(corpus.records)} records over {len(corpus.cutoffs)} cutoffs; "
        f"{n_res} resolved horizons, {len(corpus.resolution.pending)} pending; data through {corpus.data_through}"
    )


__all__ = [
    "COACHED",
    "REFERENCE_VARIANTS",
    "Readiness",
    "StreamCorpus",
    "collect_all",
    "collect_stream",
    "describe",
    "readiness",
    "score_coached",
    "score_reference_variants",
]
