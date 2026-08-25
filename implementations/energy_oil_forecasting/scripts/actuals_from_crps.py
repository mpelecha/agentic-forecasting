"""Recover the observed price at each forecast date from the cached CRPS scores.

Scoring needs actuals, and actuals live in the gitignored price cache — so any
machine without that cache (a fresh clone, a sandbox, CI) could not re-analyse a
completed backtest even though the result YAMLs were sitting right there.

They can. ``crps_ensemble(y, x)`` for a fixed quantile vector ``x`` is

    g(y) - C,   g(y) = mean_i |x_i - y|,   C = 0.5 * mean_ij |x_i - x_j|

``g`` is convex and piecewise linear in ``y``, so a stored score pins the actual
down to exactly two candidates — one either side of the median. Every predictor
scored at the same forecast date gives its own pair from its own quantiles, and
only the true actual appears in all of them. With nine or ten predictors in a
window the intersection is unique and agrees to ~1e-13.

The agreement across predictors is the proof, and
:func:`recover_actuals` reports it: predictors that share nothing but the
observation they were scored against cannot coincidentally land on the same
value to machine precision.

Not usable when fewer than two predictors are scored at a date (one predictor
leaves the two-candidate ambiguity unresolved); those dates are returned in the
``unresolved`` list rather than guessed at.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml


# Two candidates from different predictors are the same observation if they
# agree to within this many dollars. Real agreement is ~1e-13; distinct
# candidates from a single predictor are dollars apart, so anything in between
# separates them cleanly.
_AGREEMENT_TOL = 0.01


@dataclass
class Recovery:
    """Recovered actuals plus the evidence that they are right."""

    actuals: dict[str, float] = field(default_factory=dict)
    unresolved: list[tuple[str, str]] = field(default_factory=list)
    worst_disagreement: float = 0.0
    sources_per_date: list[int] = field(default_factory=list)


def _candidates(quantile_values: np.ndarray, score: float) -> list[float]:
    """Return the ``y`` values whose CRPS against these quantiles equals ``score``."""
    x = np.sort(np.asarray(quantile_values, dtype=float))
    spread = 0.5 * float(np.abs(x[:, None] - x[None, :]).mean())
    target = score + spread

    def g(y: float) -> float:
        return float(np.abs(x - y).mean())

    median = float(np.median(x))
    if g(median) > target:
        # Score below the achievable minimum — inconsistent, so claim nothing.
        return []

    reach = target + 1000.0
    out: list[float] = []
    for lo, hi in ((float(x[0]) - reach, median), (median, float(x[-1]) + reach)):
        if (g(lo) - target) * (g(hi) - target) > 0:
            continue
        for _ in range(300):  # bisection to well below float noise
            mid = 0.5 * (lo + hi)
            if (g(lo) - target) * (g(mid) - target) <= 0:
                hi = mid
            else:
                lo = mid
        out.append(0.5 * (lo + hi))
    return out


def recover_actuals(result_dir: Path) -> Recovery:
    """Recover the observed value at every scored forecast date under ``result_dir``.

    Parameters
    ----------
    result_dir
        Directory of backtest result YAMLs (one per predictor), each holding
        ``predictions`` and a positionally aligned ``scores`` list.

    Returns
    -------
    Recovery
        ``actuals`` keyed by ``YYYY-MM-DD``, plus the dates that could not be
        resolved and the worst cross-predictor disagreement observed.
    """
    per_date: dict[str, list[tuple[np.ndarray, float]]] = defaultdict(list)
    for path in sorted(result_dir.glob("*.yaml")):
        doc = yaml.safe_load(path.read_text())
        scores = doc.get("scores") or []
        for pred, score in zip(doc.get("predictions", []), scores, strict=False):
            quantiles = (pred.get("payload") or {}).get("quantiles")
            if not quantiles or score is None:
                continue
            values = np.array(sorted(float(v) for v in quantiles.values()))
            per_date[str(pred["forecast_date"])[:10]].append((values, float(score)))

    out = Recovery()
    for date, items in sorted(per_date.items()):
        pairs = [c for c in (_candidates(values, score) for values, score in items) if c]
        out.sources_per_date.append(len(pairs))
        if len(pairs) < 2:
            out.unresolved.append((date, f"only {len(pairs)} scored predictor(s)"))
            continue
        for candidate in pairs[0]:
            errors = [min(abs(candidate - other) for other in group) for group in pairs[1:]]
            if all(e < _AGREEMENT_TOL for e in errors):
                out.actuals[date] = candidate
                out.worst_disagreement = max(out.worst_disagreement, max(errors))
                break
        else:
            out.unresolved.append((date, "no candidate shared by every predictor"))
    return out


def print_recovery_report(recovery: Recovery) -> None:
    """Print the self-check so a caller can see whether to trust the actuals."""
    n_ok = len(recovery.actuals)
    n_total = n_ok + len(recovery.unresolved)
    sources = recovery.sources_per_date
    print(f"actuals recovered from cached CRPS: {n_ok}/{n_total} forecast dates")
    if sources:
        print(f"  scored predictors per date: min {min(sources)}, max {max(sources)}")
    print(f"  worst cross-predictor disagreement: ${recovery.worst_disagreement:.2e}")
    for date, reason in recovery.unresolved[:10]:
        print(f"  unresolved {date}: {reason}")
    if len(recovery.unresolved) > 10:
        print(f"  ... and {len(recovery.unresolved) - 10} more")
