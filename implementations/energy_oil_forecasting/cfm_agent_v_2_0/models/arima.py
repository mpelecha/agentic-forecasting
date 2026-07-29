"""AutoARIMA factory for the CFM deterministic model suite."""

from __future__ import annotations

from aieng.forecasting.methods.numerical import DartsAutoARIMAPredictor


def build_arima_predictor(*, num_samples: int = 200) -> DartsAutoARIMAPredictor:
    """Return the repository's probabilistic univariate AutoARIMA predictor."""
    return DartsAutoARIMAPredictor(num_samples=num_samples)


__all__ = ["build_arima_predictor"]
