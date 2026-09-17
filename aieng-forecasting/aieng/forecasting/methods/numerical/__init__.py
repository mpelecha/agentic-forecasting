"""Numerical forecasting predictor implementations.

These predictors wrap classical or machine-learning time-series models behind
the shared :class:`~aieng.forecasting.evaluation.predictor.Predictor`
interface.
"""

from .darts_arima import DartsAutoARIMAPredictor
from .darts_classical import DartsExponentialSmoothingPredictor, DartsKalmanForecasterPredictor
from .darts_regression import DartsLightGBMPredictor, DartsLinearRegressionPredictor
from .direct_regression import DirectRegressionPredictor, ResidualCalibration
from .error_correction_regression import ErrorCorrectionRegressionPredictor
from .timesfm3 import TimesFM3Predictor


__all__ = [
    "DartsAutoARIMAPredictor",
    "DartsExponentialSmoothingPredictor",
    "DartsKalmanForecasterPredictor",
    "DartsLightGBMPredictor",
    "DartsLinearRegressionPredictor",
    "DirectRegressionPredictor",
    "ErrorCorrectionRegressionPredictor",
    "ResidualCalibration",
    "TimesFM3Predictor",
]
