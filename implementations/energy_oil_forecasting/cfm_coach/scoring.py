"""Metrics for quantile forecasts: pinball, CRPS, 80% interval coverage, and direction.

**The pinball loss did not exist in this repo.** The framework scores with CRPS via
``properscoring.crps_ensemble``, which treats the quantile *values* as if they were
ensemble members. That is a reasonable summary and the wrong instrument for M1:
it is insensitive to which part of the distribution moved, and the coach's whole
job is deciding whether to change the centre or the width. Pinball loss decomposes
by quantile level, so "P90 is too tight" and "P50 is biased low" are different
numbers rather than one blended one.

Four metrics, each answering a different question:

===============  ==========================================================
``pinball``      Primary. Mean pinball loss over the grid -- a proper scoring
                 rule for quantile forecasts, decomposable by level.
``crps``         Quantile-integrated CRPS, for a single number per forecast.
``covered_80``   Did the realized value land inside P10-P90? This is R1's
                 target -- currently 0 of 6 -- and it is a *rate*, so it needs
                 accumulated runs before it says anything.
``direction``    Did P50 sit on the side of the last close the price moved to?
                 Moves and calls within `FLAT_EPS` are "no call", never a miss.
                 Also a rate, and one with a ~53% "always up" base rate, so it
                 is reported beside that base rate and never alone.
===============  ==========================================================

``crps_ensemble`` is also recorded, computed exactly as
``aieng.forecasting.evaluation.backtest._crps_for_prediction`` does, so coach numbers
remain comparable with the repo's existing leaderboard rather than forming a private
scale nothing else can be read against.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import properscoring as ps
from energy_oil_forecasting.cfm_coach.replay import ReplayedHorizon
from energy_oil_forecasting.cfm_coach.schemas import (
    HorizonRecord,
    Quantiles,
    ResolvedForecast,
    RunRecord,
    ScoreCard,
)


if TYPE_CHECKING:
    from energy_oil_forecasting.cfm_coach.baselines import RandomWalkBaseline


#: The 80% central interval R1 is written against.
COVERAGE_LOWER = 0.1
COVERAGE_UPPER = 0.9
COVERAGE_TARGET = 0.8

#: Below this many dollars a centre call, or a realized move, is "no call" rather than a
#: direction. A $0.02 nudge is not a directional opinion. Must match `FLAT_EPS` in
#: `make_corpus_page.py`, so the scoreboard and the corpus page agree on every hit.
FLAT_EPS = 0.05


def direction_of(delta: float) -> int:
    """+1, -1, or 0 when ``|delta|`` is within `FLAT_EPS`."""
    if abs(delta) < FLAT_EPS:
        return 0
    return 1 if delta > 0 else -1


def _sorted_levels(quantiles: Quantiles) -> list[float]:
    return sorted(float(level) for level in quantiles)


def pinball_by_level(quantiles: Quantiles, actual: float) -> dict[float, float]:
    """Pinball loss at each quantile level.

    ``QL_tau(q, y) = (tau - 1{y < q}) * (y - q)`` -- under-forecasting is penalised
    by ``tau`` and over-forecasting by ``1 - tau``, which is what makes the loss
    minimised only by the true ``tau``-quantile. Kept per-level because the
    aggregate hides the diagnosis: a calibration that fixes the centre and one that
    fixes the tails can post the same mean.
    """
    return {
        level: (level - (1.0 if actual < quantiles[level] else 0.0)) * (actual - quantiles[level])
        for level in _sorted_levels(quantiles)
    }


def pinball_loss(quantiles: Quantiles, actual: float) -> float:
    """Mean pinball loss across the quantile grid -- the primary M1 metric."""
    losses = pinball_by_level(quantiles, actual)
    if not losses:
        raise ValueError("cannot score an empty quantile set")
    return float(np.mean(list(losses.values())))


def crps_from_quantiles(quantiles: Quantiles, actual: float) -> float:
    """CRPS by integrating the pinball loss over quantile level.

    ``CRPS = 2 * integral of QL_tau dtau``. The grid is non-uniform
    (``STANDARD_QUANTILES`` steps 0.05 at the tails and 0.10 in the body), so this
    integrates by trapezoid rather than averaging -- a plain mean would silently
    over-weight the tails.

    *Honest boundary:* the grid spans [0.05, 0.95], so the outer 10% of probability
    mass is not represented and this is a lower bound on true CRPS. It is a
    consistent lower bound across variants scored on the same grid, which is all a
    paired comparison needs, but it should not be read as an absolute CRPS.
    """
    losses = pinball_by_level(quantiles, actual)
    levels = np.array(list(losses), dtype=float)
    if levels.size < 2:
        raise ValueError("integrated CRPS needs at least two quantile levels")
    return float(2.0 * np.trapezoid(np.array(list(losses.values()), dtype=float), levels))


def crps_ensemble(quantiles: Quantiles, actual: float) -> float:
    """CRPS treating the quantile values as ensemble members.

    Reproduces the framework's scorer exactly, so a coach score can be compared
    against numbers already recorded elsewhere in the repo. Prefer
    :func:`crps_from_quantiles` when comparing calibrations.
    """
    ensemble = np.array(sorted(float(value) for value in quantiles.values()), dtype=float)
    return float(ps.crps_ensemble(actual, ensemble))


def interval_width(quantiles: Quantiles, *, lower: float = COVERAGE_LOWER, upper: float = COVERAGE_UPPER) -> float:
    return float(quantiles[upper] - quantiles[lower])


def interval_covers(
    quantiles: Quantiles,
    actual: float,
    *,
    lower: float = COVERAGE_LOWER,
    upper: float = COVERAGE_UPPER,
) -> bool:
    """Whether the realized value fell inside the central interval, endpoints included."""
    return bool(quantiles[lower] <= actual <= quantiles[upper])


def _latest_value(record: RunRecord) -> float | None:
    value = record.diagnostics.get("latest_value")
    return float(value) if value is not None else None


@dataclass(frozen=True)
class ScoreSummary:
    """Aggregate scores for one variant over a set of scored horizons."""

    variant: str
    n: int
    pinball: float
    crps: float
    crps_ensemble: float
    mae: float
    coverage_80: float
    mean_interval_width: float
    #: Horizons where both the call and the realized move were outside `FLAT_EPS`.
    direction_calls: int = 0
    direction_hits: int = 0
    #: Horizons with a known reference price but a flat call or a flat move.
    direction_no_calls: int = 0
    #: Of `direction_calls`, how many moved up -- what "always predict up" would have
    #: scored on exactly the same horizons. The benchmark a hit rate is read against.
    direction_calls_up: int = 0

    @property
    def coverage_gap(self) -> float:
        """Signed distance from the 80% target. Negative means intervals are too narrow."""
        return self.coverage_80 - COVERAGE_TARGET

    @property
    def direction_hit_rate(self) -> float | None:
        return self.direction_hits / self.direction_calls if self.direction_calls else None

    @property
    def always_up_rate(self) -> float | None:
        return self.direction_calls_up / self.direction_calls if self.direction_calls else None

    def describe(self) -> str:
        return (
            f"{self.variant:<24} n={self.n:<4} pinball={self.pinball:7.4f}  crps={self.crps:7.4f}  "
            f"mae={self.mae:7.4f}  cover80={self.coverage_80:6.1%} ({self.coverage_gap:+.1%})  "
            f"width={self.mean_interval_width:6.2f}"
        )


class ForecastScorer:
    """Turns forecasts plus realized values into `ScoreCard`s."""

    scorer_id = "cfm_coach_forecast_scorer_v1"

    def score_quantiles(
        self,
        quantiles: Quantiles,
        *,
        point_forecast: float,
        resolved: ResolvedForecast,
        cutoff_variant: str,
        calibration_version: str,
        last_value: float | None = None,
    ) -> ScoreCard:
        actual = resolved.realized_value
        return ScoreCard(
            run_id=resolved.run_id,
            horizon=resolved.horizon,
            cutoff=resolved.cutoff,
            forecast_date=resolved.forecast_date,
            variant=cutoff_variant,
            calibration_version=calibration_version,
            realized_value=actual,
            point_forecast=point_forecast,
            p10=quantiles[COVERAGE_LOWER],
            p50=quantiles[0.5],
            p90=quantiles[COVERAGE_UPPER],
            pinball=pinball_loss(quantiles, actual),
            crps=crps_from_quantiles(quantiles, actual),
            crps_ensemble=crps_ensemble(quantiles, actual),
            absolute_error=abs(point_forecast - actual),
            interval_width=interval_width(quantiles),
            covered_80=interval_covers(quantiles, actual),
            last_value=last_value,
            direction_call=None if last_value is None else direction_of(quantiles[0.5] - last_value),
            direction_outcome=None if last_value is None else direction_of(actual - last_value),
        )

    def score_replayed(
        self,
        replayed: ReplayedHorizon,
        resolved: ResolvedForecast,
        *,
        variant: str,
        last_value: float | None = None,
    ) -> ScoreCard:
        """Score one horizon re-derived under a candidate calibration."""
        return self.score_quantiles(
            replayed.quantiles,
            point_forecast=replayed.point_forecast,
            resolved=resolved,
            cutoff_variant=variant,
            calibration_version=replayed.calibration_version,
            last_value=last_value,
        )

    def score_recorded(self, record: RunRecord, resolved: ResolvedForecast, *, variant: str = "agent") -> ScoreCard:
        """Score what the agent actually published -- the frozen baseline."""
        horizon_record: HorizonRecord = record.horizon(resolved.horizon)
        return self.score_quantiles(
            horizon_record.final_quantiles,
            point_forecast=horizon_record.final_point_forecast,
            resolved=resolved,
            cutoff_variant=variant,
            calibration_version=record.calibration_version,
            last_value=_latest_value(record),
        )

    def score_ensemble(self, record: RunRecord, resolved: ResolvedForecast, *, variant: str = "ensemble") -> ScoreCard:
        """Score the unadjusted numerical ensemble -- what the LLM overlay is applied to.

        This is the comparison that answers "does the LLM overlay help at all?",
        which nothing in the repo has ever measured on live forward runs. It is not
        the floor, though: see `score_random_walk`.
        """
        horizon_record = record.horizon(resolved.horizon)
        return self.score_quantiles(
            horizon_record.ensemble_quantiles,
            point_forecast=horizon_record.ensemble_point_forecast,
            resolved=resolved,
            cutoff_variant=variant,
            calibration_version="none",
            last_value=_latest_value(record),
        )

    def score_random_walk(
        self,
        record: RunRecord,
        resolved: ResolvedForecast,
        baseline: RandomWalkBaseline,
        *,
        variant: str = "random_walk",
    ) -> ScoreCard | None:
        """Score "today's price" on the same horizon, or None when the baseline cannot be built.

        The floor that is actually hard to beat on WTI. Its P50 is the last close, so it
        never makes a directional call; its direction benchmark is the always-up rate.
        """
        built = baseline.for_record(record, resolved.horizon)
        if built is None:
            return None
        point, quantiles = built
        return self.score_quantiles(
            quantiles,
            point_forecast=point,
            resolved=resolved,
            cutoff_variant=variant,
            calibration_version="none",
            last_value=point,
        )

    @staticmethod
    def summarize(cards: Sequence[ScoreCard], *, variant: str | None = None) -> ScoreSummary:
        """Aggregate one variant's cards.

        Means only -- no standard error here, because these observations are *not*
        independent: horizons overlap, so 60 runs at h=21 carry roughly 3 independent
        observations. Uncertainty is the comparison policy's job, and it gets it from
        an origin-blocked bootstrap rather than from a naive n.
        """
        if not cards:
            raise ValueError("cannot summarize zero score cards")
        names = {card.variant for card in cards}
        if variant is None and len(names) > 1:
            raise ValueError(f"cards mix variants {sorted(names)}; summarize one at a time")
        called = [card for card in cards if card.direction_hit is not None]
        return ScoreSummary(
            variant=variant or names.pop(),
            n=len(cards),
            pinball=float(np.mean([card.pinball for card in cards])),
            crps=float(np.mean([card.crps for card in cards])),
            crps_ensemble=float(np.mean([card.crps_ensemble for card in cards])),
            mae=float(np.mean([card.absolute_error for card in cards])),
            coverage_80=float(np.mean([card.covered_80 for card in cards])),
            mean_interval_width=float(np.mean([card.interval_width for card in cards])),
            direction_calls=len(called),
            direction_hits=sum(1 for card in called if card.direction_hit),
            direction_no_calls=sum(
                1 for card in cards if card.direction_call is not None and card.direction_hit is None
            ),
            direction_calls_up=sum(1 for card in called if card.direction_outcome == 1),
        )

    @staticmethod
    def by_variant(cards: Iterable[ScoreCard]) -> dict[str, list[ScoreCard]]:
        grouped: dict[str, list[ScoreCard]] = {}
        for card in cards:
            grouped.setdefault(card.variant, []).append(card)
        return grouped

    @classmethod
    def summarize_all(cls, cards: Iterable[ScoreCard]) -> list[ScoreSummary]:
        return [cls.summarize(group, variant=name) for name, group in cls.by_variant(cards).items()]


__all__ = [
    "COVERAGE_LOWER",
    "COVERAGE_TARGET",
    "COVERAGE_UPPER",
    "FLAT_EPS",
    "ForecastScorer",
    "ScoreSummary",
    "crps_ensemble",
    "crps_from_quantiles",
    "direction_of",
    "interval_covers",
    "interval_width",
    "pinball_by_level",
    "pinball_loss",
]
