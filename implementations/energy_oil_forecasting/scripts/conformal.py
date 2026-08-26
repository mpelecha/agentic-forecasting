"""Conformalized quantile regression: make any predictor's bands hit their nominal rate.

The width floor in ``width_recalibration.py`` rescales a band toward the spread
of historical price moves. It works, but it is a heuristic with no guarantee
attached, it needs an external percentile table, and it can only ever widen.

Conformal calibration is the principled version. For a quantile pair
``(q_lo, q_hi)`` with nominal coverage ``1 - alpha``, the conformity score of a
past forecast is how far outside the band the outcome fell::

    E = max(q_lo - y, y - q_hi)

negative when the outcome landed inside. Taking the ``(1 - alpha)`` empirical
quantile of those scores over a calibration set and shifting both edges by it::

    q_lo' = q_lo - delta,   q_hi' = q_hi + delta

gives an interval whose coverage converges on the nominal rate — under
exchangeability, with a finite-sample guarantee. ``delta`` is negative when the
model is too wide, so this narrows an over-covering predictor as readily as it
widens an under-covering one. That is the reason for choosing CQR over plain
split conformal: it corrects the model's own quantiles rather than replacing
them, so whatever adaptive shape the model has survives.

**Leakage is the whole difficulty in a time series.** A forecast issued at
origin ``s`` for horizon ``h`` is not scoreable until ``s + h`` business days
have passed, so calibrating the forecast issued at ``t`` may only use origins
with ``s + h <= t``. Calibrating on all past origins regardless — the obvious
implementation — quietly uses outcomes that had not happened yet, and inflates
measured coverage. :func:`calibrate_series` enforces the constraint and reports
how many predictions it had to leave uncalibrated for want of history.

Exchangeability does not hold strictly for a series with volatility clustering,
so the guarantee is approximate here. The rolling window is the concession to
that: recent errors describe the current regime better than a ten-year average
does.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


#: Quantile pairs to calibrate, each with its own conformity score. Calibrating
#: only the 80% band would leave the rest of the grid uncorrected and make CRPS
#: incomparable, since CRPS reads the whole grid.
QUANTILE_PAIRS: tuple[tuple[float, float], ...] = ((0.05, 0.95), (0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.4, 0.6))

#: Minimum calibration points before a conformal correction is attempted. Below
#: this the empirical quantile is too noisy to be worth applying, and the
#: uncorrected band is returned instead.
MIN_CALIBRATION = 8


@dataclass
class CalibrationReport:
    """What the calibration actually managed to do, so it can be checked."""

    calibrated: int = 0
    uncalibrated: int = 0
    deltas: list[float] = field(default_factory=list)
    calibration_sizes: list[int] = field(default_factory=list)


def conformal_delta(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample-valid quantile of the conformity scores.

    Uses the ``ceil((n + 1)(1 - alpha)) / n`` level rather than plain
    ``1 - alpha``: the correction for the fact that the calibration set is
    finite and the test point is exchangeable with it, not drawn after it.
    When that level exceeds 1 (too few points for the requested coverage) the
    largest observed score is the best available answer.
    """
    n = len(scores)
    level = np.ceil((n + 1) * (1.0 - alpha)) / n
    if level > 1.0:
        return float(np.max(scores))
    return float(np.quantile(scores, level, method="higher"))


def apply_conformal(
    quantiles: dict[float, float],
    calibration: list[dict[float, float]],
    actuals: list[float],
) -> dict[float, float]:
    """Return ``quantiles`` with each pair widened or narrowed to its nominal rate.

    Parameters
    ----------
    quantiles
        The prediction being corrected, keyed by quantile level.
    calibration
        Past predictions' quantile dicts, all scoreable before this origin.
    actuals
        Realised values for ``calibration``, positionally aligned.

    Returns
    -------
    dict[float, float]
        Corrected quantiles, re-sorted so the grid stays non-decreasing. The
        median is never moved — conformal calibration is about the interval,
        and shifting the centre would change the point forecast, which is a
        different claim than the one being made here.
    """
    out = dict(quantiles)
    for lo, hi in QUANTILE_PAIRS:
        if lo not in quantiles or hi not in quantiles:
            continue
        scores = np.array(
            [max(past[lo] - actual, actual - past[hi]) for past, actual in zip(calibration, actuals, strict=True)],
            dtype=float,
        )
        delta = conformal_delta(scores, alpha=1.0 - (hi - lo))
        out[lo] = quantiles[lo] - delta
        out[hi] = quantiles[hi] + delta

    # Pairs are calibrated independently, so a narrow pair can in principle
    # overtake a wider one. Reassigning the sorted values back onto the sorted
    # levels restores monotonicity without moving any mass.
    levels = sorted(out)
    values = sorted(out[level] for level in levels)
    return dict(zip(levels, values, strict=True))


def calibrate_series(
    records: list[dict],
    *,
    window: int | None = None,
    horizon_business_days: bool = True,
) -> tuple[list[dict], CalibrationReport]:
    """Conformally calibrate one predictor's predictions at one horizon.

    ``records`` must each carry ``origin`` (``pd.Timestamp``), ``resolves``
    (``pd.Timestamp``, when the outcome becomes observable), ``quantiles`` and
    ``actual``. They are processed in origin order; each prediction is
    calibrated only on records that had already resolved at its own origin.

    Parameters
    ----------
    window
        Keep only the most recent ``window`` eligible records. ``None`` uses
        every one of them. A window is the concession to volatility
        clustering: exchangeability across a decade of oil prices is a fiction,
        and recent residuals describe the current regime better.
    horizon_business_days
        Unused placeholder kept so callers can be explicit that ``resolves``
        is already in the right calendar; the eligibility test is a plain
        timestamp comparison.

    Returns
    -------
    tuple[list[dict], CalibrationReport]
        Records with a ``conformal`` key added (the corrected quantiles, or the
        originals where there was too little history), and the report.
    """
    _ = horizon_business_days
    ordered = sorted(records, key=lambda r: r["origin"])
    report = CalibrationReport()
    out: list[dict] = []

    for record in ordered:
        # Only forecasts whose outcome was observable strictly before this
        # origin. This is the leak guard: a 63-day forecast issued last quarter
        # has not resolved yet and must not calibrate today's band.
        eligible = [other for other in ordered if other["resolves"] <= record["origin"]]
        if window is not None:
            eligible = eligible[-window:]

        enriched = dict(record)
        if len(eligible) >= MIN_CALIBRATION:
            enriched["conformal"] = apply_conformal(
                record["quantiles"],
                [other["quantiles"] for other in eligible],
                [other["actual"] for other in eligible],
            )
            report.calibrated += 1
            report.calibration_sizes.append(len(eligible))
            width_before = record["quantiles"][0.9] - record["quantiles"][0.1]
            width_after = enriched["conformal"][0.9] - enriched["conformal"][0.1]
            report.deltas.append((width_after - width_before) / 2.0)
        else:
            enriched["conformal"] = dict(record["quantiles"])
            report.uncalibrated += 1
        out.append(enriched)

    return out, report
