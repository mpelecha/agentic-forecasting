"""Kalman-filter factory for the CFM deterministic model suite."""

from __future__ import annotations

from aieng.forecasting.methods.numerical import DartsKalmanForecasterPredictor


def build_kalman_predictor(
    *,
    num_samples: int = 200,
    dim_x: int = 2,
) -> DartsKalmanForecasterPredictor:
    """Return the repository's probabilistic linear-Gaussian state-space model."""
    return DartsKalmanForecasterPredictor(num_samples=num_samples, dim_x=dim_x)


__all__ = ["build_kalman_predictor"]
