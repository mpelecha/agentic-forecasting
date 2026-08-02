"""Kalman factory for CFM v3.3 using the unchanged shared predictor."""

from aieng.forecasting.methods.numerical import DartsKalmanForecasterPredictor


def build_kalman_predictor(*, num_samples: int = 1_000, dim_x: int = 2) -> DartsKalmanForecasterPredictor:
    return DartsKalmanForecasterPredictor(num_samples=num_samples, dim_x=dim_x)


__all__ = ["build_kalman_predictor"]
