"""Sparse AR Return (specification Section 1.4)."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.features import StaticFrameAdapter
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES
from energy_oil_forecasting.cfm_agent_v_6_0.config import CfmV60Settings
from energy_oil_forecasting.cfm_agent_v_6_0.history import PriceHistory
from energy_oil_forecasting.cfm_agent_v_6_0.models.paths import (
    ComponentFailure,
    lag_matrix,
    recursive_lag_simulation,
)
from energy_oil_forecasting.cfm_agent_v_6_0.models.sparse_ar import SparseARReturn, ridge_fit
from energy_oil_forecasting.data import WTI_SERIES_ID


CUTOFF = datetime(2026, 9, 14)


def _history(service, settings):
    return PriceHistory.from_context(service.context(as_of=CUTOFF), WTI_SERIES_ID, [], settings)


def _ar1_service(n: int = 1500, phi: float = 0.6, intercept: float = 0.001, seed: int = 4) -> DataService:
    rng = np.random.default_rng(seed)
    returns = np.zeros(n)
    for index in range(1, n):
        returns[index] = intercept + phi * returns[index - 1] + rng.normal(0.0, 0.01)
    prices = 50.0 * np.cumprod(1.0 + returns)
    dates = pd.bdate_range(end="2026-09-14", periods=n)
    service = DataService()
    service.register(
        WTI_SERIES_ID,
        StaticFrameAdapter(pd.DataFrame({"timestamp": dates, "value": prices, "released_at": dates})),
        SeriesMetadata(series_id=WTI_SERIES_ID, description="ar1", source="t", units="USD/bbl", frequency="B"),
    )
    return service


def test_ridge_equals_closed_form_solution_on_a_known_ar1(synthetic_task):
    settings = CfmV60Settings()
    service = _ar1_service()
    history = _history(service, settings)
    model = SparseARReturn(history, settings, np.random.default_rng(1), 1)
    model.predict(synthetic_task, service.context(as_of=CUTOFF))
    audit = model.training_data
    assert audit is not None
    lags = settings.return_lags
    returns = history.returns
    rows = np.arange(max(lags), len(returns))
    design = np.column_stack([np.ones(len(rows)), lag_matrix(returns, lags, rows)])
    penalty = np.eye(design.shape[1]) * settings.ar_ridge_penalty
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ returns[rows])
    assert audit.fitted_parameters["intercept"] == pytest.approx(beta[0], abs=1e-12)
    for index, lag in enumerate(lags):
        assert audit.fitted_parameters[f"coef_lag_{lag}"] == pytest.approx(beta[index + 1], abs=1e-12)
    residuals = returns[rows] - design @ beta
    assert audit.residual_std == pytest.approx(float(np.std(residuals, ddof=0)))
    assert audit.fitted_parameters["coef_lag_1"] == pytest.approx(0.6, abs=0.05)
    assert audit.return_lags == lags
    assert audit.eligible_blocks == len(rows) - settings.block_length_business_days + 1
    assert audit.fitted_parameters["training_rows"] == len(rows)


def test_ridge_fit_without_intercept_and_penalty_direction():
    rng = np.random.default_rng(2)
    features = rng.normal(size=(200, 3))
    target = features @ np.array([1.0, -2.0, 0.5]) + rng.normal(0.0, 0.01, size=200)
    unpenalised = ridge_fit(features, target, penalty=0.0, fit_intercept=False)
    np.testing.assert_allclose(unpenalised, np.linalg.lstsq(features, target, rcond=None)[0])
    heavy = ridge_fit(features, target, penalty=1e6, fit_intercept=False)
    assert np.all(np.abs(heavy) < np.abs(unpenalised))
    with_intercept = ridge_fit(features, target + 3.0, penalty=1e6, fit_intercept=True)
    assert with_intercept[0] == pytest.approx(3.0, abs=0.05)


def test_recursive_simulation_builds_lags_from_observed_and_simulated_returns():
    observed = np.arange(10, dtype=float)
    innovations = np.zeros((2, 4))
    seen = []

    def conditional_mean(lagged):
        seen.append(lagged.copy())
        return np.full(lagged.shape[0], 100.0)

    simulated = recursive_lag_simulation(observed, [1, 3], innovations, conditional_mean)
    np.testing.assert_array_equal(simulated, np.full((2, 4), 100.0))
    np.testing.assert_array_equal(seen[0], [[9.0, 7.0], [9.0, 7.0]])
    np.testing.assert_array_equal(seen[1], [[100.0, 8.0], [100.0, 8.0]])
    np.testing.assert_array_equal(seen[2], [[100.0, 9.0], [100.0, 9.0]])
    np.testing.assert_array_equal(seen[3], [[100.0, 100.0], [100.0, 100.0]])


def test_sparse_ar_predicts_and_is_reproducible(synthetic_service, synthetic_task, synthetic_context):
    settings = CfmV60Settings()
    history = _history(synthetic_service, settings)
    first = SparseARReturn(history, settings, np.random.default_rng(9), 9).predict(synthetic_task, synthetic_context)
    second = SparseARReturn(history, settings, np.random.default_rng(9), 9).predict(synthetic_task, synthetic_context)
    assert [p.payload.quantiles for p in first] == [p.payload.quantiles for p in second]
    for prediction in first:
        assert list(prediction.payload.quantiles) == [float(level) for level in STANDARD_QUANTILES]
        assert prediction.payload.point_forecast == prediction.payload.quantiles[0.5]
        assert prediction.predictor_id == "cfm_v60_sparse_ar_return"


def test_sparse_ar_insufficient_history(synthetic_service_factory, synthetic_task):
    settings = CfmV60Settings()
    minimum = max(settings.return_lags) + settings.block_length_business_days + settings.min_training_rows_margin
    service = synthetic_service_factory(n_observations=minimum)  # minimum - 1 returns
    history = _history(service, settings)
    model = SparseARReturn(history, settings, np.random.default_rng(1), 1)
    with pytest.raises(ComponentFailure) as info:
        model.predict(synthetic_task, service.context(as_of=CUTOFF))
    assert info.value.code == "insufficient_history"
    assert model.training_data is not None
    assert model.training_data.minimum_returns_required == minimum
    enough = synthetic_service_factory(n_observations=minimum + 1)
    predictions = SparseARReturn(_history(enough, settings), settings, np.random.default_rng(1), 1).predict(
        synthetic_task, enough.context(as_of=CUTOFF)
    )
    assert len(predictions) == 3
