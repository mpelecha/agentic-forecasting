"""Shared return-path machinery (specification Sections 1.2, 1.3, 1.6).

Holds the failure exception, block sampling, compounding with the simulated
floor, quantile reading, validation, and :class:`ReturnPathComponent`, the base
class the three models share. The base class runs the normative order of
operations for one component: simulate, compound, floor, quantiles, validate,
point = P50. Paths never leave this module except as quantiles.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime, timezone

import numpy as np
from aieng.forecasting.data import ForecastContext
from aieng.forecasting.evaluation.prediction import (
    STANDARD_QUANTILES,
    ContinuousForecast,
    Prediction,
)
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask
from energy_oil_forecasting.cfm_agent_v_6_0.config import CfmV60Settings
from energy_oil_forecasting.cfm_agent_v_6_0.history import PriceHistory
from energy_oil_forecasting.cfm_agent_v_6_0.schemas import (
    ModelTrainingData,
    is_failure_code,
)


#: The eleven quantile levels, from the shared library (structural constant).
QUANTILE_LEVELS: tuple[float, ...] = tuple(float(level) for level in STANDARD_QUANTILES)
MEDIAN_LEVEL = 0.5  # structural: the median quantile level


class ComponentFailure(Exception):  # noqa: N818 - name fixed by the specification
    """A component failed a check; ``code`` enters the identity, ``message`` the audit only."""

    def __init__(self, code: str, message: str) -> None:
        if not is_failure_code(code):
            raise ValueError(f"not a failure code: {code!r}")
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def library_error(exc: BaseException) -> ComponentFailure:
    """Wrap any other exception as ``library_error:<ExceptionClass>``."""
    return ComponentFailure(f"library_error:{type(exc).__name__}", str(exc))


def sample_block_starts(rng: np.random.Generator, n_blocks: int, n_returns: int, block: int) -> np.ndarray:
    """Sample ``n_blocks`` start indices uniformly with replacement from ``0 .. n_returns - block``."""
    if block < 1 or n_returns < block:
        raise ValueError("block length must be positive and at most the number of returns")
    return rng.integers(0, n_returns - block + 1, size=n_blocks)


def overlapping_block_paths(
    rng: np.random.Generator,
    values: np.ndarray,
    *,
    n_paths: int,
    path_length: int,
    block: int,
) -> np.ndarray:
    """Build ``(n_paths, path_length)`` paths from overlapping blocks of ``values``.

    Every eligible overlapping block (start indices ``0 .. n - block``) is a
    candidate; ``ceil(path_length / block)`` blocks are sampled uniformly with
    replacement per path, concatenated, and truncated to ``path_length``.
    """
    values = np.asarray(values, dtype=float)
    n_returns = len(values)
    blocks_per_path = math.ceil(path_length / block)
    starts = sample_block_starts(rng, n_paths * blocks_per_path, n_returns, block).reshape(n_paths, blocks_per_path)
    offsets = np.arange(block)
    index = (starts[:, :, None] + offsets[None, None, :]).reshape(n_paths, blocks_per_path * block)
    return values[index[:, :path_length]]


def compound(last_price: float, returns: np.ndarray, *, floor: float) -> tuple[np.ndarray, int]:
    """Return simulated prices ``last_price * cumprod(1 + returns)`` floored, and the activation count."""
    prices = float(last_price) * np.cumprod(1.0 + np.asarray(returns, dtype=float), axis=1)
    with np.errstate(invalid="ignore"):
        activations = int(np.sum(prices < floor))
    return np.maximum(prices, floor), activations


def quantiles_at(prices: np.ndarray, horizon: int, levels: Sequence[float] = QUANTILE_LEVELS) -> dict[float, float]:
    """Linear-interpolation quantiles of the simulated prices at column ``horizon - 1``."""
    values = np.quantile(prices[:, horizon - 1], list(levels), method="linear")
    return {float(level): float(value) for level, value in zip(levels, values)}


def validate_paths(
    returns: np.ndarray,
    prices: np.ndarray,
    *,
    n_paths: int,
    path_length: int,
    floor: float,
) -> None:
    """Raise :class:`ComponentFailure` on a shape, finiteness, or floor violation."""
    expected = (n_paths, path_length)
    if returns.shape != expected or prices.shape != expected:
        raise ComponentFailure(
            "shape_mismatch",
            f"expected paths of shape {expected}; got returns {returns.shape} and prices {prices.shape}",
        )
    if not np.isfinite(returns).all() or not np.isfinite(prices).all():
        raise ComponentFailure("non_finite_output", "a simulated return or price is not finite")
    if bool(np.any(prices < floor)):
        raise ComponentFailure("floor_violation", f"a simulated price lies below the simulated floor {floor}")


def validate_quantiles(quantiles: dict[float, float], levels: Sequence[float] = QUANTILE_LEVELS) -> None:
    """Raise :class:`ComponentFailure` unless all levels are present, finite, and non-decreasing."""
    missing = [level for level in levels if level not in quantiles]
    if missing:
        raise ComponentFailure("quantile_crossing", f"quantile levels missing: {missing}")
    ordered = [quantiles[level] for level in levels]
    if not all(math.isfinite(value) for value in ordered):
        raise ComponentFailure("non_finite_output", "a component quantile is not finite")
    if any(later < earlier for earlier, later in zip(ordered, ordered[1:])):
        raise ComponentFailure("quantile_crossing", "component quantiles decrease in the level")


def lag_matrix(returns: np.ndarray, lags: Sequence[int], rows: np.ndarray) -> np.ndarray:
    """Return ``returns[row - lag]`` for every training row and lag, shape ``(len(rows), len(lags))``.

    ``rows`` are indices into ``returns`` of the targets; every row must be at
    least ``max(lags)`` so that all lagged values are observed.
    """
    rows = np.asarray(rows, dtype=int)
    return np.column_stack([returns[rows - lag] for lag in lags]) if len(rows) else np.empty((0, len(lags)))


def recursive_lag_simulation(
    observed: np.ndarray,
    lags: Sequence[int],
    innovations: np.ndarray,
    conditional_mean,
) -> np.ndarray:
    """Simulate returns step by step, each step's lag vector drawn from observed and simulated returns.

    ``innovations`` has shape ``(n_paths, path_length)``; ``conditional_mean``
    maps a ``(n_paths, len(lags))`` lag matrix for one step to ``(n_paths,)``
    conditional returns. The simulated return is the conditional return plus the
    sampled innovation.
    """
    n_paths, path_length = innovations.shape
    n_observed = len(observed)
    simulated = np.empty((n_paths, path_length), dtype=float)
    for step in range(path_length):
        lagged = np.empty((n_paths, len(lags)), dtype=float)
        for column, lag in enumerate(lags):
            index = n_observed + step - lag
            lagged[:, column] = observed[index] if index < n_observed else simulated[:, index - n_observed]
        simulated[:, step] = np.asarray(conditional_mean(lagged), dtype=float) + innovations[:, step]
    return simulated


class ReturnPathComponent(Predictor):
    """Base class for the three return-path models.

    Subclasses implement :meth:`_simulate_returns`, which returns a
    ``(model_num_paths, simulation_length_business_days)`` array of simple
    returns drawn from the component generator, and fill ``self.training_data``
    before returning or raising.
    """

    component_name: str = ""

    def __init__(
        self,
        history: PriceHistory,
        settings: CfmV60Settings,
        rng: np.random.Generator,
        component_seed: int,
    ) -> None:
        self.history = history
        self.settings = settings
        self.rng = rng
        self.component_seed = int(component_seed)
        self.training_data: ModelTrainingData | None = None
        self.simulated_floor_activations: int = 0
        self.quantiles_by_horizon: dict[int, dict[float, float]] = {}

    @property
    def predictor_id(self) -> str:
        """Stable component identifier."""
        return f"cfm_v60_{self.component_name}_return"

    # -- hooks -------------------------------------------------------------

    def _simulate_returns(self) -> np.ndarray:
        raise NotImplementedError

    def _base_training_data(self) -> dict[str, object]:
        history = self.history
        return {
            "target_observations": int(len(history.prices)),
            "first_target_date": history.first_date,
            "last_target_date": history.last_date,
            "returns_available": int(len(history.returns)),
        }

    # -- Predictor protocol ------------------------------------------------

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Simulate, compound, floor, read quantiles, validate, and label one Prediction per horizon."""
        settings = self.settings
        n_paths = settings.model_num_paths
        path_length = settings.simulation_length_business_days
        horizons = [int(h) for h in task.horizons]
        if any(h < 1 or h > path_length for h in horizons):
            raise ValueError(f"task horizons must lie in 1..{path_length}; got {horizons}")
        try:
            returns = np.asarray(self._simulate_returns(), dtype=float)
            prices, activations = compound(
                self.history.last_price,
                returns,
                floor=settings.simulated_price_floor,
            )
        except ComponentFailure:
            raise
        except Exception as exc:
            raise library_error(exc) from exc
        self.simulated_floor_activations = activations
        validate_paths(
            returns,
            prices,
            n_paths=n_paths,
            path_length=path_length,
            floor=settings.simulated_price_floor,
        )
        predictions: list[Prediction] = []
        issued_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        self.quantiles_by_horizon = {}
        for horizon in horizons:
            quantiles = quantiles_at(prices, horizon)
            validate_quantiles(quantiles)
            self.quantiles_by_horizon[horizon] = quantiles
            point = quantiles[MEDIAN_LEVEL]
            predictions.append(
                Prediction(
                    predictor_id=self.predictor_id,
                    task_id=task.task_id,
                    issued_at=issued_at,
                    as_of=context.as_of,
                    forecast_date=self.history.forecast_date(horizon),
                    payload=ContinuousForecast(point_forecast=point, quantiles=quantiles),
                    metadata={
                        "component": self.component_name,
                        "component_seed": self.component_seed,
                        "horizon": horizon,
                        "num_paths": n_paths,
                        "path_length": path_length,
                        "simulated_price_floor": settings.simulated_price_floor,
                        "simulated_floor_activations": activations,
                        "block_length_business_days": settings.block_length_business_days,
                    },
                )
            )
        return predictions


__all__ = [
    "MEDIAN_LEVEL",
    "QUANTILE_LEVELS",
    "ComponentFailure",
    "ReturnPathComponent",
    "compound",
    "lag_matrix",
    "library_error",
    "overlapping_block_paths",
    "quantiles_at",
    "recursive_lag_simulation",
    "sample_block_starts",
    "validate_paths",
    "validate_quantiles",
]
