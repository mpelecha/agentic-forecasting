"""AutoARIMA factory for CFM v5.2 using the unchanged shared predictor."""

from aieng.forecasting.methods.numerical import DartsAutoARIMAPredictor


def build_arima_predictor(*, num_samples: int = 1_000) -> DartsAutoARIMAPredictor:
    return DartsAutoARIMAPredictor(num_samples=num_samples)


__all__ = ["build_arima_predictor"]
