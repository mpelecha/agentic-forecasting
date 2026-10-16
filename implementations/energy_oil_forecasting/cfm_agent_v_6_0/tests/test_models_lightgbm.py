"""Lightweight LightGBM Return (specification Section 1.5)."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES
from energy_oil_forecasting.cfm_agent_v_6_0.config import CfmV60Settings
from energy_oil_forecasting.cfm_agent_v_6_0.history import PriceHistory
from energy_oil_forecasting.cfm_agent_v_6_0.models import lightgbm_return
from energy_oil_forecasting.cfm_agent_v_6_0.models.lightgbm_return import LightGBMReturn, is_level_covariate
from energy_oil_forecasting.cfm_agent_v_6_0.models.paths import ComponentFailure
from energy_oil_forecasting.data import WTI_SERIES_ID


CUTOFF = datetime(2026, 9, 14)


def _history(service, settings, covariate_ids=None):
    ids = settings.covariate_series_ids if covariate_ids is None else covariate_ids
    return PriceHistory.from_context(service.context(as_of=CUTOFF), WTI_SERIES_ID, ids, settings)


class RecordingRegressor:
    """Stand-in for ``LGBMRegressor`` that records the fit and predict inputs and predicts zero."""

    instances: list[RecordingRegressor] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fit_features = None
        self.fit_target = None
        self.predict_calls: list[np.ndarray] = []
        RecordingRegressor.instances.append(self)

    def fit(self, features, target):
        self.fit_features = features.copy()
        self.fit_target = np.asarray(target).copy()
        return self

    def predict(self, features):
        self.predict_calls.append(features.to_numpy(dtype=float).copy())
        return np.zeros(len(features))


@pytest.fixture
def recording_regressor(monkeypatch):
    RecordingRegressor.instances = []
    monkeypatch.setattr(lightgbm_return, "LGBMRegressor", RecordingRegressor)
    return RecordingRegressor


def test_level_covariate_rule():
    assert is_level_covariate("oil_curve_contango_l1b")
    assert is_level_covariate("vix_level_l1b")
    assert not is_level_covariate("brent_log_ret_1b_l1b")
    assert not is_level_covariate("gold_log_ret_1b_l1b")


def test_feature_construction_and_hyperparameters(
    recording_regressor, synthetic_service, synthetic_task, synthetic_context
):
    settings = CfmV60Settings()
    history = _history(synthetic_service, settings)
    model = LightGBMReturn(history, settings, np.random.default_rng(3), 3)
    model.predict(synthetic_task, synthetic_context)
    regressor = recording_regressor.instances[0]
    assert regressor.kwargs == {
        "objective": "regression",
        "n_estimators": settings.lightgbm_n_estimators,
        "learning_rate": settings.lightgbm_learning_rate,
        "num_leaves": settings.lightgbm_num_leaves,
        "max_depth": settings.lightgbm_max_depth,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 1,
        "random_state": 3,
        "verbose": -1,
    }
    lags = settings.return_lags
    ids = settings.covariate_series_ids
    features = regressor.fit_features
    assert list(features.columns) == [f"wti_return_lag_{lag}" for lag in lags] + ids
    returns = history.returns
    max_lag = max(lags)
    expected_rows = len(returns) - max_lag
    assert features.shape == (expected_rows, len(lags) + len(ids))
    values = features.to_numpy()
    # Training row k targets returns[k]; feature j is returns[k - lag_j]; covariates are the
    # aligned values on the date of returns[k], which is dates[k + 1].
    first_row = max_lag
    np.testing.assert_array_equal(regressor.fit_target, returns[first_row:])
    for column, lag in enumerate(lags):
        np.testing.assert_array_equal(values[:, column], returns[first_row - lag : len(returns) - lag])
    aligned = history.aligned.loc[history.dates[first_row + 1 :]]
    np.testing.assert_array_equal(values[:, len(lags) :], aligned[ids].to_numpy())
    assert model.training_data is not None
    assert model.training_data.covariates == ids
    assert model.training_data.rows_remaining == 3000
    assert model.training_data.rows_dropped == 0
    assert model.training_data.fitted_parameters["training_rows"] == expected_rows


def test_future_covariate_rule_holds_levels_and_zeroes_returns(
    recording_regressor, synthetic_service, synthetic_task, synthetic_context
):
    settings = CfmV60Settings()
    history = _history(synthetic_service, settings)
    model = LightGBMReturn(history, settings, np.random.default_rng(3), 3)
    model.predict(synthetic_task, synthetic_context)
    regressor = recording_regressor.instances[0]
    # One in-sample predict for the residuals, then one per simulation step.
    assert len(regressor.predict_calls) == 1 + settings.simulation_length_business_days
    lags = settings.return_lags
    ids = settings.covariate_series_ids
    latest = history.covariate_latest_values()
    for step_features in regressor.predict_calls[1:]:
        assert step_features.shape == (settings.model_num_paths, len(lags) + len(ids))
        for column, series_id in enumerate(ids):
            expected = latest[series_id] if is_level_covariate(series_id) else 0.0
            assert np.all(step_features[:, len(lags) + column] == expected)
    # The first simulated step's lag-1 feature is the latest observed return on every path.
    assert np.all(regressor.predict_calls[1][:, 0] == history.returns[-1])
    # With zero conditional mean the simulated returns are the residual blocks, i.e. the
    # in-sample residuals, so later lag-1 features are simulated values that vary across paths.
    assert len(np.unique(regressor.predict_calls[2][:, 0])) > 1


def test_missing_covariate_fails_with_covariate_missing(synthetic_service_factory, synthetic_task):
    settings = CfmV60Settings()
    present = [series_id for series_id in settings.covariate_series_ids if series_id != "gold_log_ret_1b_l1b"]
    service = synthetic_service_factory(covariate_ids=present)
    history = _history(service, settings)
    model = LightGBMReturn(history, settings, np.random.default_rng(1), 1)
    with pytest.raises(ComponentFailure) as info:
        model.predict(synthetic_task, service.context(as_of=CUTOFF))
    assert info.value.code == "covariate_missing"
    assert "gold_log_ret_1b_l1b" in info.value.message
    assert model.training_data is not None
    assert model.training_data.covariates == settings.covariate_series_ids


def test_gaps_are_filled_and_dropped_and_audited(synthetic_service_factory, synthetic_task):
    settings = CfmV60Settings()
    limit = settings.covariate_forward_fill_limit
    service = synthetic_service_factory(
        covariate_gaps={"brent_log_ret_1b_l1b": [1500, 1501], "vix_level_l1b": list(range(2000, 2000 + limit + 4))}
    )
    history = _history(service, settings)
    model = LightGBMReturn(history, settings, np.random.default_rng(1), 1)
    predictions = model.predict(synthetic_task, service.context(as_of=CUTOFF))
    assert len(predictions) == 3
    audit = model.training_data
    assert audit is not None
    assert audit.rows_before_alignment == 3000
    assert audit.covariate_fill_counts["brent_log_ret_1b_l1b"] == 2
    assert audit.covariate_fill_counts["vix_level_l1b"] == limit
    assert audit.rows_dropped == 4
    assert audit.rows_remaining == 3000 - 4
    assert audit.covariate_observations["brent_log_ret_1b_l1b"] == 2998
    assert audit.fitted_parameters["training_rows"] == len(history.returns) - max(settings.return_lags) - 4


def test_short_aligned_frame_fails_with_insufficient_aligned_rows(synthetic_service_factory, synthetic_task):
    settings = CfmV60Settings()
    minimum = max(settings.return_lags) + settings.block_length_business_days + settings.min_training_rows_margin
    # Enough WTI history for the AR models, but a covariate gap wide enough to drop rows below the minimum.
    n = minimum + 40
    gap = list(range(0, 41 + settings.covariate_forward_fill_limit))
    service = synthetic_service_factory(n_observations=n, covariate_gaps={"natgas_log_ret_1b_l1b": gap})
    history = _history(service, settings)
    assert len(history.returns) >= minimum
    model = LightGBMReturn(history, settings, np.random.default_rng(1), 1)
    with pytest.raises(ComponentFailure) as info:
        model.predict(synthetic_task, service.context(as_of=CUTOFF))
    assert info.value.code == "insufficient_aligned_rows"
    assert model.training_data is not None
    assert model.training_data.rows_remaining < minimum
    assert model.training_data.rows_dropped > 0


def test_real_lightgbm_predicts_eleven_quantiles_and_is_reproducible(
    synthetic_service, synthetic_task, synthetic_context
):
    settings = CfmV60Settings()
    history = _history(synthetic_service, settings)
    first = LightGBMReturn(history, settings, np.random.default_rng(8), 8).predict(synthetic_task, synthetic_context)
    second = LightGBMReturn(history, settings, np.random.default_rng(8), 8).predict(synthetic_task, synthetic_context)
    assert [p.payload.quantiles for p in first] == [p.payload.quantiles for p in second]
    for prediction in first:
        assert list(prediction.payload.quantiles) == [float(level) for level in STANDARD_QUANTILES]
        assert prediction.payload.point_forecast == prediction.payload.quantiles[0.5]
        assert prediction.predictor_id == "cfm_v60_lightgbm_return"
