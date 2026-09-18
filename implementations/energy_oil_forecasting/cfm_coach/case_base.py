"""M3a -- the market case base: how the market looked, and what happened next.

~4,700 business days since 2007-12, each described by the same state vector v5.0
computes at every cutoff (`diagnostics.STATE_FEATURES`) and paired with the move WTI
actually made over the next 5, 10 and 21 business days.

This exists to answer two questions about a live forecast, both *before* any of its
horizons resolve:

- **Is today unprecedented?** (R5) If today's state has unusually few close analogues
  in 4,700 days, the system has no track record to speak from and should say so.
- **Does the interval look plausible?** If comparable days moved a median of $16 and
  the ensemble is offering a $7 P10-P90, the interval is almost certainly too narrow.
  That is available on day one, where measuring coverage takes ~40 runs.

Two deliberate boundaries:

- **Nothing here is an LLM.** Standardise, euclidean distance, argsort. The model
  never sees the corpus and, in stage 1, nothing here is reachable from a prompt
  builder -- which is what keeps `prompt_version` stable while M1 builds a baseline.
- **Dispersion is a plausibility check, not a measured error distribution.** It can
  say an interval looks wrong; only accumulated coverage says by how much. It informs
  the trust tier and must not silently set the calibration layer's width scale.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from aieng.forecasting.data import DataService
from energy_oil_forecasting.cfm_coach.config import DEFAULT_SETTINGS, CoachSettings
from energy_oil_forecasting.cfm_coach.diagnostics import (
    STATE_FEATURES,
    forward_moves,
    historical_states,
)


_CHUNK = 512


class OutOfDistributionVerdict:
    """Why a state was or was not judged unprecedented."""

    def __init__(
        self,
        *,
        is_out_of_distribution: bool,
        neighbour_distance: float,
        threshold: float,
        distance_percentile: float,
        most_extreme_feature: str,
        most_extreme_percentile: float,
    ):
        self.is_out_of_distribution = is_out_of_distribution
        self.neighbour_distance = neighbour_distance
        self.threshold = threshold
        self.distance_percentile = distance_percentile
        self.most_extreme_feature = most_extreme_feature
        self.most_extreme_percentile = most_extreme_percentile

    def as_dict(self) -> dict[str, float | str | bool]:
        return {
            "is_out_of_distribution": self.is_out_of_distribution,
            "neighbour_distance": self.neighbour_distance,
            "threshold": self.threshold,
            "distance_percentile": self.distance_percentile,
            "most_extreme_feature": self.most_extreme_feature,
            "most_extreme_percentile": self.most_extreme_percentile,
        }


class MarketCaseBase:
    case_base_id = "cfm_coach_market_case_base_v1"

    def __init__(self, frame: pd.DataFrame, *, horizons: tuple[int, ...], settings: CoachSettings = DEFAULT_SETTINGS):
        self.frame = frame
        self.horizons = horizons
        self.s = settings

        matrix = frame[list(STATE_FEATURES)].to_numpy(dtype=float)
        # Robust scaling: oil vol and jump z-scores have fat tails, so centring on
        # the mean and scaling by the standard deviation would let a handful of
        # crisis days dominate every distance.
        self._centre = np.median(matrix, axis=0)
        spread = np.percentile(matrix, 75, axis=0) - np.percentile(matrix, 25, axis=0)
        fallback = matrix.std(axis=0, ddof=0)
        self._scale = np.where(spread > 0, spread, np.where(fallback > 0, fallback, 1.0))
        self._standardised = (matrix - self._centre) / self._scale
        self._threshold: float | None = None
        self._reference: np.ndarray | None = None

    # -- construction ---------------------------------------------------------

    @classmethod
    def build(
        cls,
        service: DataService,
        *,
        horizons: tuple[int, ...] = (5, 10, 21),
        settings: CoachSettings = DEFAULT_SETTINGS,
        start: str | None = None,
    ) -> MarketCaseBase:
        states = historical_states(service, start=start)
        moves = forward_moves(service, horizons=horizons)
        frame = states.join(moves, how="left")

        # A case is only usable as an analogue once its outcome is known. Recording
        # when that happened lets historical evaluation of the trust report stay
        # cutoff-honest (R3) instead of quietly consulting the future.
        index = pd.Series(frame.index, index=frame.index)
        longest = max(horizons)
        resolved = index.shift(-longest)
        frame["outcome_known_by"] = resolved
        return cls(frame, horizons=horizons, settings=settings)

    @classmethod
    def load(cls, path: Path, *, horizons: tuple[int, ...] = (5, 10, 21), settings: CoachSettings = DEFAULT_SETTINGS):
        return cls(pd.read_parquet(path), horizons=horizons, settings=settings)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.frame.to_parquet(path)
        return path

    def __len__(self) -> int:
        return len(self.frame)

    # -- queries --------------------------------------------------------------

    def _vector(self, state: Mapping[str, float | None]) -> np.ndarray:
        missing = [name for name in STATE_FEATURES if state.get(name) is None]
        if missing:
            raise ValueError(f"state is missing required features: {missing}")
        raw = np.array([float(state[name]) for name in STATE_FEATURES])
        return (raw - self._centre) / self._scale

    def _usable_mask(self, as_of: date | None) -> np.ndarray:
        """Cases with a known outcome, and -- if `as_of` is given -- known *by then*."""
        move_columns = [f"move_{horizon}b" for horizon in self.horizons]
        mask = self.frame[move_columns].notna().all(axis=1).to_numpy(copy=True)
        if as_of is not None:
            known = self.frame["outcome_known_by"]
            mask = mask & (known.notna() & (known <= as_of)).to_numpy()
        return mask

    def neighbours(self, state: Mapping[str, float | None], *, k: int | None = None, as_of: date | None = None):
        """Return the `k` most similar historical days, with what the price did next."""
        k = k or self.s.analogue_k
        mask = self._usable_mask(as_of)
        if not mask.any():
            return self.frame.iloc[:0].assign(distance=[])

        distances = np.linalg.norm(self._standardised[mask] - self._vector(state), axis=1)
        candidates = self.frame[mask].copy()
        candidates["distance"] = distances
        return candidates.nsmallest(min(k, len(candidates)), "distance")

    def _knn_distance(self, vector: np.ndarray, k: int) -> float:
        distances = np.linalg.norm(self._standardised - vector, axis=1)
        return float(np.partition(distances, k)[k])

    def _reference_distances(self, k: int) -> np.ndarray:
        """Each historical day's own distance to its k-th nearest neighbour.

        This is the yardstick: "unprecedented" means today is further from its
        neighbours than a typical day is from its own, not that some feature is
        merely large. It catches unusual *combinations*, which per-feature
        thresholds cannot.
        """
        if self._reference is None:
            out = np.empty(len(self._standardised))
            for start in range(0, len(self._standardised), _CHUNK):
                block = self._standardised[start : start + _CHUNK]
                distances = np.linalg.norm(block[:, None, :] - self._standardised[None, :, :], axis=2)
                # Column 0 is the row's distance to itself; the k-th neighbour is at k.
                out[start : start + len(block)] = np.partition(distances, k, axis=1)[:, k]
            self._reference = out
        return self._reference

    def percentile_of(self, feature: str, value: float) -> float:
        column = self.frame[feature].to_numpy(dtype=float)
        return float((column <= value).mean() * 100.0)

    def assess_novelty(self, state: Mapping[str, float | None], *, k: int | None = None) -> OutOfDistributionVerdict:
        """Judge whether today has a usable precedent (R5). Annotates; never suppresses."""
        k = k or self.s.analogue_k
        vector = self._vector(state)
        reference = self._reference_distances(k)
        if self._threshold is None:
            self._threshold = float(np.percentile(reference, self.s.ood_percentile))

        distance = self._knn_distance(vector, k)
        deviations = np.abs(vector)
        index = int(np.argmax(deviations))
        feature = STATE_FEATURES[index]

        return OutOfDistributionVerdict(
            is_out_of_distribution=distance > self._threshold,
            neighbour_distance=distance,
            threshold=self._threshold,
            distance_percentile=float((reference <= distance).mean() * 100.0),
            most_extreme_feature=feature,
            most_extreme_percentile=self.percentile_of(feature, float(state[feature])),
        )

    def move_dispersion(
        self, state: Mapping[str, float | None], horizon: int, *, k: int | None = None, as_of: date | None = None
    ) -> dict[str, float]:
        """How far comparable days moved over `horizon` -- the interval plausibility check."""
        rows = self.neighbours(state, k=k, as_of=as_of)
        moves = rows[f"move_{horizon}b"].dropna().to_numpy(dtype=float)
        if moves.size == 0:
            return {}
        return {
            "n": float(moves.size),
            "median_abs_move": float(np.median(np.abs(moves))),
            "max_abs_move": float(np.max(np.abs(moves))),
            "p10": float(np.percentile(moves, 10)),
            "p90": float(np.percentile(moves, 90)),
            "implied_width": float(np.percentile(moves, 90) - np.percentile(moves, 10)),
        }


__all__ = ["MarketCaseBase", "OutOfDistributionVerdict"]
