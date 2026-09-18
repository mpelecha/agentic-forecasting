"""The scoreboard: floors vs frozen baseline vs current calibration, over time.

Four variants are scored on the *same* resolved horizons, always:

===============  ==========================================================
``random_walk``  Today's price: the last visible close as P50, WTI's own
                 recent h-step moves as the spread. The floor that is
                 genuinely hard to beat on a near-driftless series.
``ensemble``     The numerical suite with no LLM overlay at all. What the
                 overlay is applied to -- and not itself the floor.
``agent``        What was actually published -- the frozen baseline. Because
                 ``v001`` is a true no-op, this is also the agent as shipped.
``current``      Every record replayed under the calibration in force now.
                 Identical to ``agent`` until a calibration is approved, which
                 is the point: the gap between the two *is* what the coach did.
===============  ==========================================================

Scoring all four on identical days is not a presentational choice. At this sample
size an unpaired before/after comparison is not evidence (R10), and the cheapest way
to guarantee pairing is to never compute an unpaired number in the first place.

**This module reports; it does not gate.** It will happily print that a variant looks
better. Whether that difference survives an origin-blocked bootstrap, a holdout split
and an effect-size floor is `policy/comparison_policy.py`'s decision, and only that
module may return ``passed=True``.

Three things it is written to keep visible rather than average away:

- **Provenance.** Only ``live_forward`` records are fitting evidence (R7). Replayed
  history is scored and displayed, and it is banded separately, because the last time
  these two were mixed the result was a "+10.9% overlay improvement" that came
  entirely from two replays and meant nothing.
- **Effective sample size.** Overlapping horizons mean 60 runs at h=21 carry roughly
  3 independent observations. The header says so rather than letting ``n=60`` imply
  a precision that is not there, and verdicts that need more than that are withheld.
- **The direction base rate.** WTI rises over 5-21 business days about 53% of the
  time, so "always up" scores ~53% with no skill at all. Every hit rate is printed
  beside what always-up scored on the same horizons, and beside the hit rate the
  effective sample would need before it could be called better than that.

Usage::

    uv run python -m energy_oil_forecasting.cfm_coach.report
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import pandas as pd
from energy_oil_forecasting.cfm_coach.baselines import RandomWalkBaseline
from energy_oil_forecasting.cfm_coach.config import DEFAULT_SETTINGS, CoachSettings
from energy_oil_forecasting.cfm_coach.ledger import CalibrationLedger
from energy_oil_forecasting.cfm_coach.outcomes import OutcomeResolver, ResolutionReport
from energy_oil_forecasting.cfm_coach.replay import FidelityReport, ReplayEngine
from energy_oil_forecasting.cfm_coach.run_store import RunRecordStore
from energy_oil_forecasting.cfm_coach.schemas import RunRecord, ScoreCard
from energy_oil_forecasting.cfm_coach.scoring import (
    COVERAGE_TARGET,
    FLAT_EPS,
    ForecastScorer,
    ScoreSummary,
)
from energy_oil_forecasting.cfm_coach.streams import DEFAULT_STREAM, stream_for
from energy_oil_forecasting.cfm_coach.targets import DEFAULT_TARGET, AgentTarget


ENSEMBLE = "ensemble"
AGENT = "agent"
CURRENT = "current"
RANDOM_WALK = "random_walk"

VARIANT_ORDER = (RANDOM_WALK, ENSEMBLE, AGENT, CURRENT)

#: A cumulative mean over one or two cutoffs is the mean drawn twice. Withhold the
#: curve until there is enough of a sequence for its shape to mean anything.
_MIN_CUTOFFS_FOR_CURVE = 3

#: Below this many effective independent observations the R1 line reports the
#: coverage it measured but withholds the verdict. 12 resolved horizons from two
#: replayed March cutoffs once printed "intervals too narrow" on ~0.7 observations,
#: and the project steered on it.
MIN_EFFECTIVE_N_FOR_R1 = 10.0

#: One-sided significance level for "hit rate beats always-up", matching the gate.
DIRECTION_ALPHA = 0.10


#: Horizons overlap, so consecutive daily runs at h=21 are near-duplicates of each
#: other. Dividing scored rows by the horizon gives the order of magnitude of
#: genuinely independent observations -- the number that governs what can be concluded.
def effective_n(cards: list[ScoreCard]) -> float:
    """Rough count of independent observations behind a set of scored horizons."""
    if not cards:
        return 0.0
    by_horizon: dict[int, set[date]] = {}
    for card in cards:
        by_horizon.setdefault(card.horizon, set()).add(card.cutoff)
    return sum(len(cutoffs) / horizon for horizon, cutoffs in by_horizon.items())


def binomial_upper_tail(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return float(sum(math.comb(n, i) * p**i * (1.0 - p) ** (n - i) for i in range(k, n + 1)))


def hit_rate_needed(n_effective: float, base_rate: float, alpha: float = DIRECTION_ALPHA) -> float | None:
    """Smallest hit rate that beats `base_rate` at one-sided `alpha` on `n_effective` observations.

    Uses the *effective* count, floored, because a binomial test on overlapping
    horizons treats near-duplicates as independent and manufactures significance.
    None means no hit rate, not even 100%, would be significant at this sample size --
    which is the honest answer for the first months of a daily corpus.
    """
    n = int(math.floor(n_effective))
    if n < 1:
        return None
    for k in range(n + 1):
        if binomial_upper_tail(k, n, base_rate) < alpha:
            return k / n
    return None


@dataclass
class CycleReport:
    """Everything one scoring pass established, ready to print or persist."""

    generated_at: datetime
    calibration_version: str
    records: int
    fitting_eligible_records: int
    resolution: ResolutionReport
    fidelity: list[FidelityReport]
    cards: list[ScoreCard]
    #: Scored rows restricted to `live_forward` provenance -- the only fitting evidence.
    fitting_cards: list[ScoreCard] = field(default_factory=list)

    @property
    def summaries(self) -> list[ScoreSummary]:
        return ForecastScorer.summarize_all(self.cards)

    @property
    def fitting_summaries(self) -> list[ScoreSummary]:
        return ForecastScorer.summarize_all(self.fitting_cards) if self.fitting_cards else []

    @property
    def drifted(self) -> list[FidelityReport]:
        return [report for report in self.fidelity if not report.faithful]


class CoachReporter:
    """Builds the scoreboard from the corpus on disk."""

    reporter_id = "cfm_coach_reporter_v1"

    def __init__(
        self,
        settings: CoachSettings = DEFAULT_SETTINGS,
        *,
        target: AgentTarget = DEFAULT_TARGET,
        store: RunRecordStore | None = None,
        resolver: OutcomeResolver | None = None,
        replay: ReplayEngine | None = None,
        scorer: ForecastScorer | None = None,
        baseline: RandomWalkBaseline | None = None,
    ):
        self.s = settings
        self.target = target
        self.store = store or RunRecordStore(settings)
        self.resolver = resolver or OutcomeResolver(settings)
        # `target` must match the corpus `settings.runs_dir` holds: `ReplayEngine`
        # refuses a record from another agent rather than replaying it through the
        # wrong engine. One reporter reads one stream.
        self.replay = replay or ReplayEngine(settings, target=target)
        self.scorer = scorer or ForecastScorer()
        self.ledger = CalibrationLedger(settings)
        self._baseline = baseline

    @classmethod
    def for_stream(cls, stream: Any, **kwargs: Any) -> "CoachReporter":
        """Build a reporter over one `RunStream`'s corpus, ledger and agent."""
        return cls(stream.coach_settings, target=stream.target, **kwargs)

    @property
    def baseline(self) -> RandomWalkBaseline:
        """Built lazily from the resolver's own series, so the floor and the score share one set of prices."""
        if self._baseline is None:
            self._baseline = RandomWalkBaseline(self.resolver.price_series())
        return self._baseline

    def score_corpus(
        self,
        records: list[RunRecord],
        resolution: ResolutionReport,
        *,
        as_of: date | None = None,
    ) -> list[ScoreCard]:
        """Score every variant on every resolved horizon."""
        by_id = {record.run_id: record for record in records}
        current = self.ledger.current(as_of or date.today())
        cards: list[ScoreCard] = []

        for resolved in resolution.resolved:
            record = by_id[resolved.run_id]
            random_walk = self.scorer.score_random_walk(record, resolved, self.baseline, variant=RANDOM_WALK)
            if random_walk is not None:
                cards.append(random_walk)
            cards.append(self.scorer.score_ensemble(record, resolved, variant=ENSEMBLE))
            cards.append(self.scorer.score_recorded(record, resolved, variant=AGENT))
            replayed = self.replay.replay(record, current, horizons=(resolved.horizon,))
            if replayed:
                latest = record.diagnostics.get("latest_value")
                cards.append(
                    self.scorer.score_replayed(
                        replayed[0],
                        resolved,
                        variant=CURRENT,
                        last_value=float(latest) if latest is not None else None,
                    )
                )
        return cards

    def build(self, records: list[RunRecord] | None = None, *, strict_fidelity: bool = False) -> CycleReport:
        """Resolve, fidelity-check, and score the corpus.

        ``strict_fidelity`` is off by default so the scoreboard still prints when a
        record has drifted -- the drift is reported at the top instead. A *fitting*
        cycle must set it, because fitting on a corpus the coach can no longer
        reproduce is how you get a confident wrong answer.
        """
        records = records if records is not None else self.store.load_all()
        resolution = self.resolver.resolve_all(records)
        fidelity = self.replay.verify_all(records, strict=strict_fidelity)
        cards = self.score_corpus(records, resolution)

        eligible = {record.run_id for record in records if record.provenance in self.s.fitting_eligible_provenance}
        return CycleReport(
            generated_at=datetime.now(),
            calibration_version=self.ledger.current(date.today()).version,
            records=len(records),
            fitting_eligible_records=len(eligible),
            resolution=resolution,
            fidelity=fidelity,
            cards=cards,
            fitting_cards=[card for card in cards if card.run_id in eligible],
        )

    # -- tables ---------------------------------------------------------------

    @staticmethod
    def to_frame(cards: list[ScoreCard]) -> pd.DataFrame:
        return pd.DataFrame([card.model_dump() for card in cards])

    def by_horizon(self, cards: list[ScoreCard]) -> pd.DataFrame:
        """Variant x horizon mean pinball, plus coverage -- where R1 actually lives.

        Split by horizon because the calibration hypothesis on the table is
        horizon-shaped: width should scale with volatility, and how much depends on
        how far ahead you are looking.
        """
        frame = self.to_frame(cards)
        if frame.empty:
            return frame
        table = frame.pivot_table(
            index="variant",
            columns="horizon",
            values=["pinball", "covered_80", "interval_width"],
            aggfunc="mean",
        )
        return table.sort_index(axis=1)

    def learning_curve(self, cards: list[ScoreCard]) -> pd.DataFrame:
        """Cumulative mean pinball per variant, in cutoff order.

        The question this answers is "is the system getting better as it learns?",
        and the honest answer for a long while will be "too early to say" -- the
        curve is here so that becomes visible rather than assumed.
        """
        frame = self.to_frame(cards)
        if frame.empty:
            return frame
        frame = frame.sort_values(["cutoff", "run_id", "horizon"])
        frame["cumulative_pinball"] = (
            frame.groupby("variant")["pinball"].expanding().mean().reset_index(level=0, drop=True)
        )
        frame["cumulative_coverage"] = (
            frame.groupby("variant")["covered_80"].expanding().mean().reset_index(level=0, drop=True)
        )
        return frame.pivot_table(
            index="cutoff",
            columns="variant",
            values=["cumulative_pinball", "cumulative_coverage"],
            aggfunc="last",
        )

    # -- rendering ------------------------------------------------------------

    @staticmethod
    def render_direction(cards: list[ScoreCard]) -> list[str]:
        """One row per variant: hit rate, what always-up scored on the same calls, and the bar to clear."""
        lines = [
            f"  did P50 sit on the side of the last close the price moved to? (within ${FLAT_EPS:.2f} = no call)",
            "",
            f"  {'variant':<13}{'calls':>6}{'no-call':>9}{'hit rate':>10}{'always-up':>11}"
            f"{'eff. n':>8}  {'needed for p<' + format(DIRECTION_ALPHA, '.2f'):<18}verdict",
        ]
        grouped = ForecastScorer.by_variant(cards)
        for variant in [name for name in VARIANT_ORDER if name in grouped]:
            group = grouped[variant]
            summary = ForecastScorer.summarize(group, variant=variant)
            if summary.direction_calls == 0:
                moved = [card for card in group if card.direction_outcome]
                base = sum(1 for card in moved if card.direction_outcome == 1) / len(moved) if moved else None
                note = (
                    f"never calls direction; base rate on these horizons: {base:.1%} up"
                    if base is not None
                    else "no directional calls"
                )
                lines.append(f"  {variant:<13}{0:>6}{summary.direction_no_calls:>9}{'--':>10}{'':>11}{'':>8}  {note}")
                continue

            called = [card for card in group if card.direction_hit is not None]
            n_eff = effective_n(called)
            hit, up = summary.direction_hit_rate, summary.always_up_rate
            needed = hit_rate_needed(n_eff, up)
            if needed is None:
                bar, verdict = "unreachable", "too few independent observations to tell"
            elif hit >= needed:
                bar, verdict = f"{needed:.1%}", "beats always-up"
            else:
                bar, verdict = f"{needed:.1%}", "indistinguishable from always-up"
            lines.append(
                f"  {variant:<13}{summary.direction_calls:>6}{summary.direction_no_calls:>9}"
                f"{hit:>10.1%}{up:>11.1%}{n_eff:>8.1f}  {bar:<18}{verdict}"
            )
        return lines

    @staticmethod
    def render_r1(cards: list[ScoreCard]) -> str | None:
        """Render the coverage verdict, withheld when the sample cannot support one."""
        agent = [card for card in cards if card.variant == AGENT]
        if not agent:
            return None
        achieved = ForecastScorer.summarize(agent, variant=AGENT).coverage_80
        n_eff = effective_n(agent)
        if n_eff < MIN_EFFECTIVE_N_FOR_R1:
            return (
                f"R1 (80% coverage): verdict withheld -- agent covered {achieved:.0%} on n={len(agent)} horizon(s),"
                f" but that is ~{n_eff:.1f} effective independent observations; a verdict needs"
                f" {MIN_EFFECTIVE_N_FOR_R1:.0f}."
            )
        return (
            f"R1 (80% coverage): {achieved:.0%} against a {COVERAGE_TARGET:.0%} target"
            f"  -- {'intervals too narrow' if achieved < COVERAGE_TARGET else 'on target'}"
            f" on n={len(agent)} resolved horizon(s), ~{n_eff:.1f} effective observations."
        )

    def render(self, report: CycleReport) -> str:  # noqa: PLR0912
        lines: list[str] = [
            "=" * 92,
            f"CFM Coach scoreboard   {report.generated_at:%Y-%m-%d %H:%M}   calibration in force: {report.calibration_version}",
            "=" * 92,
            "",
            f"Corpus:     {report.records} record(s), {report.fitting_eligible_records} fitting-eligible (live_forward)",
            f"Resolved:   {len(report.resolution.resolved)} horizon(s), {len(report.resolution.pending)} pending"
            f"   (price data through {report.resolution.data_through})",
        ]

        if report.drifted:
            lines += ["", "!! FIDELITY DRIFT -- these records no longer replay to their recorded output:"]
            lines += [f"     {item.describe()}" for item in report.drifted]
            lines += ["   Any fit built on this corpus is invalid until resolved."]
        else:
            lines.append(f"Fidelity:   all {len(report.fidelity)} record(s) replay exactly")

        if not report.cards:
            lines += ["", "Nothing has resolved yet -- no scores to report.", ""]
            lines += [f"  pending: {item.describe()}" for item in report.resolution.pending]
            return "\n".join(lines)

        lines += [
            "",
            "Variants:   random_walk = today's price, recent spread (the floor)  ·  ensemble = models, no LLM",
            "            agent = as published  ·  current = replayed under the calibration in force",
        ]

        lines += ["", "-" * 92, "ALL RESOLVED HORIZONS (includes replayed history -- not fitting evidence)", "-" * 92]
        lines.append(f"  effective independent observations: ~{effective_n(report.cards):.1f}")
        lines += [f"  {summary.describe()}" for summary in sorted(report.summaries, key=lambda s: s.pinball)]

        lines += ["", "-" * 92, "FITTING EVIDENCE ONLY (live_forward)", "-" * 92]
        if report.fitting_summaries:
            lines.append(f"  effective independent observations: ~{effective_n(report.fitting_cards):.1f}")
            lines += [
                f"  {summary.describe()}" for summary in sorted(report.fitting_summaries, key=lambda s: s.pinball)
            ]
        else:
            lines.append(
                "  No live_forward horizon has resolved yet. Every number above comes from runs"
                "\n  re-executed at a past cutoff against today's web, so none of it is evidence"
                "\n  about how the agent performs live."
            )

        direction_cards = report.fitting_cards or report.cards
        band = "fitting evidence only" if report.fitting_cards else "all resolved -- no live_forward yet"
        lines += ["", "-" * 92, f"DIRECTION ({band})", "-" * 92, *self.render_direction(direction_cards)]

        table = self.by_horizon(report.cards)
        if not table.empty:
            lines += ["", "-" * 92, "BY HORIZON (all resolved)", "-" * 92, table.round(3).to_string()]

        # Below three distinct cutoffs a cumulative mean is just the mean redrawn,
        # so the curve is withheld rather than dressed up as a trend.
        cutoffs = {card.cutoff for card in report.cards}
        if len(cutoffs) >= _MIN_CUTOFFS_FOR_CURVE:
            curve = self.learning_curve(report.cards)
            if not curve.empty:
                lines += [
                    "",
                    "-" * 92,
                    "LEARNING CURVE (cumulative, in cutoff order)",
                    "-" * 92,
                    curve.round(3).to_string(),
                ]
        else:
            lines += [
                "",
                f"Learning curve withheld: {len(cutoffs)} distinct cutoff(s), needs {_MIN_CUTOFFS_FOR_CURVE}.",
            ]

        r1 = self.render_r1(report.fitting_cards or report.cards)
        if r1:
            lines += ["", r1]

        if report.resolution.pending:
            lines += ["", f"Pending ({len(report.resolution.pending)}):"]
            lines += [f"  {item.describe()}" for item in report.resolution.pending]

        return "\n".join(lines)


def main() -> None:
    stream = DEFAULT_STREAM
    for arg in sys.argv[1:]:
        if arg.startswith("--stream="):
            stream = stream_for(arg.split("=", 1)[1])
    reporter = CoachReporter.for_stream(stream)
    print(f"stream: {stream.stream_id}  ({stream.target.agent_id}, {stream.model})\n")
    print(reporter.render(reporter.build()))


if __name__ == "__main__":
    main()


__all__ = [
    "AGENT",
    "CURRENT",
    "DIRECTION_ALPHA",
    "ENSEMBLE",
    "MIN_EFFECTIVE_N_FOR_R1",
    "RANDOM_WALK",
    "CoachReporter",
    "CycleReport",
    "binomial_upper_tail",
    "effective_n",
    "hit_rate_needed",
]
