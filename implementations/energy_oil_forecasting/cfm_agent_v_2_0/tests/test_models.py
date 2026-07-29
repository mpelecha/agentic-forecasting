"""Numerical model factory tests."""

from energy_oil_forecasting.cfm_agent_v_2_0.models import (
    build_arima_predictor,
    build_kalman_predictor,
    build_lightgbm_predictor,
)


def test_model_factories_return_expected_predictor_families() -> None:
    assert build_arima_predictor(num_samples=50).predictor_id == "darts_autoarima"
    assert build_kalman_predictor(num_samples=50).predictor_id == "darts_kalman"
    assert build_lightgbm_predictor(num_samples=50).predictor_id == "darts_lightgbm"
