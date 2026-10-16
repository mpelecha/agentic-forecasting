"""Historical Bootstrap Return (specification Section 1.3) and the shared path machinery."""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pytest
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES, ContinuousForecast
from energy_oil_forecasting.cfm_agent_v_6_0.config import CfmV60Settings
from energy_oil_forecasting.cfm_agent_v_6_0.history import PriceHistory
from energy_oil_forecasting.cfm_agent_v_6_0.models.bootstrap import HistoricalBootstrapReturn
from energy_oil_forecasting.cfm_agent_v_6_0.models.paths import (
    ComponentFailure,
    compound,
    overlapping_block_paths,
    quantiles_at,
    validate_paths,
    validate_quantiles,
)
from energy_oil_forecasting.data import WTI_SERIES_ID


CUTOFF = datetime(2026, 9, 14)


def _history(service, settings):
    return PriceHistory.from_context(service.context(as_of=CUTOFF), WTI_SERIES_ID, [], settings)


def test_overlapping_block_paths_use_contiguous_history_blocks():
    values = np.arange(100, dtype=float)
    rng = np.random.default_rng(1)
    paths = overlapping_block_paths(rng, values, n_paths=50, path_length=21, block=5)
    assert paths.shape == (50, 21)
    for path in paths:
        for start in range(0, 20, 5):
            block = path[start : start + 5]
            assert np.array_equal(block, np.arange(block[0], block[0] + 5))
            assert block[0] <= 95


def test_overlapping_block_paths_are_generator_deterministic():
    values = np.random.default_rng(0).normal(size=300)
    first = overlapping_block_paths(np.random.default_rng(7), values, n_paths=100, path_length=21, block=5)
    second = overlapping_block_paths(np.random.default_rng(7), values, n_paths=100, path_length=21, block=5)
    assert np.array_equal(first, second)


def test_compound_and_floor():
    returns = np.array([[0.1, -0.5, -1.0], [0.0, 0.0, 0.0]])
    prices, activations = compound(10.0, returns, floor=1.0)
    np.testing.assert_allclose(prices[0], [11.0, 5.5, 1.0])
    np.testing.assert_allclose(prices[1], [10.0, 10.0, 10.0])
    assert activations == 1


def test_quantiles_at_uses_linear_numpy_quantiles():
    prices = np.random.default_rng(3).uniform(20, 80, size=(1000, 21))
    quantiles = quantiles_at(prices, 5)
    assert list(quantiles) == [float(level) for level in STANDARD_QUANTILES]
    for level, value in quantiles.items():
        assert value == float(np.quantile(prices[:, 4], level, method="linear"))


def test_validate_paths_failure_codes():
    good = np.zeros((10, 3))
    validate_paths(good, np.ones((10, 3)), n_paths=10, path_length=3, floor=1.0)
    with pytest.raises(ComponentFailure) as info:
        validate_paths(good, np.ones((9, 3)), n_paths=10, path_length=3, floor=1.0)
    assert info.value.code == "shape_mismatch"
    bad = np.ones((10, 3))
    bad[0, 0] = np.nan
    with pytest.raises(ComponentFailure) as info:
        validate_paths(good, bad, n_paths=10, path_length=3, floor=1.0)
    assert info.value.code == "non_finite_output"
    with pytest.raises(ComponentFailure) as info:
        validate_paths(good, np.full((10, 3), 0.5), n_paths=10, path_length=3, floor=1.0)
    assert info.value.code == "floor_violation"


def test_validate_quantiles_failure_codes():
    levels = [float(level) for level in STANDARD_QUANTILES]
    validate_quantiles(dict.fromkeys(levels, 5.0))
    with pytest.raises(ComponentFailure) as info:
        validate_quantiles({level: float(index) for index, level in enumerate(levels[:-1])})
    assert info.value.code == "quantile_crossing"
    crossing = {level: float(index) for index, level in enumerate(levels)}
    crossing[0.5] = -1.0
    with pytest.raises(ComponentFailure) as info:
        validate_quantiles(crossing)
    assert info.value.code == "quantile_crossing"
    with pytest.raises(ComponentFailure) as info:
        validate_quantiles(dict.fromkeys(levels, math.inf))
    assert info.value.code == "non_finite_output"


def test_component_failure_requires_a_vocabulary_code():
    assert ComponentFailure("library_error:ValueError", "x").code == "library_error:ValueError"
    with pytest.raises(ValueError):
        ComponentFailure("some_other_code", "x")


def test_bootstrap_predicts_eleven_quantiles_per_horizon(synthetic_service, synthetic_task, synthetic_context):
    settings = CfmV60Settings()
    history = _history(synthetic_service, settings)
    model = HistoricalBootstrapReturn(history, settings, np.random.default_rng(11), 11)
    predictions = model.predict(synthetic_task, synthetic_context)
    assert [prediction.forecast_date for prediction in predictions] == [
        history.forecast_date(h) for h in synthetic_task.horizons
    ]
    for prediction, horizon in zip(predictions, synthetic_task.horizons):
        payload = prediction.payload
        assert isinstance(payload, ContinuousForecast)
        assert list(payload.quantiles) == [float(level) for level in STANDARD_QUANTILES]
        assert payload.point_forecast == payload.quantiles[0.5]
        assert prediction.metadata["horizon"] == horizon
        assert prediction.metadata["component_seed"] == 11
        assert prediction.predictor_id == "cfm_v60_bootstrap_return"
    audit = model.training_data
    assert audit is not None
    assert audit.returns_available == len(history.returns)
    assert audit.eligible_blocks == len(history.returns) - settings.block_length_business_days + 1
    assert audit.block_length == settings.block_length_business_days
    assert audit.minimum_returns_required == settings.block_length_business_days + settings.min_training_rows_margin
    assert model.simulated_floor_activations == 0


def test_bootstrap_is_reproducible_for_the_same_generator_seed(synthetic_service, synthetic_task, synthetic_context):
    settings = CfmV60Settings()
    history = _history(synthetic_service, settings)
    first = HistoricalBootstrapReturn(history, settings, np.random.default_rng(5), 5).predict(
        synthetic_task, synthetic_context
    )
    second = HistoricalBootstrapReturn(history, settings, np.random.default_rng(5), 5).predict(
        synthetic_task, synthetic_context
    )
    assert [p.payload.quantiles for p in first] == [p.payload.quantiles for p in second]
    third = HistoricalBootstrapReturn(history, settings, np.random.default_rng(6), 6).predict(
        synthetic_task, synthetic_context
    )
    assert [p.payload.quantiles for p in first] != [p.payload.quantiles for p in third]


def test_bootstrap_insufficient_history(synthetic_service_factory, synthetic_task):
    settings = CfmV60Settings()
    minimum = settings.block_length_business_days + settings.min_training_rows_margin
    service = synthetic_service_factory(n_observations=minimum)  # minimum - 1 returns
    history = _history(service, settings)
    model = HistoricalBootstrapReturn(history, settings, np.random.default_rng(1), 1)
    with pytest.raises(ComponentFailure) as info:
        model.predict(synthetic_task, service.context(as_of=CUTOFF))
    assert info.value.code == "insufficient_history"
    assert model.training_data is not None
    assert model.training_data.returns_available == minimum - 1
    enough = synthetic_service_factory(n_observations=minimum + 1)
    predictions = HistoricalBootstrapReturn(_history(enough, settings), settings, np.random.default_rng(1), 1).predict(
        synthetic_task, enough.context(as_of=CUTOFF)
    )
    assert len(predictions) == 3


def test_simulated_floor_activations_are_counted(synthetic_service, synthetic_task, synthetic_context):
    settings = CfmV60Settings(simulated_price_floor=25.0, final_price_floor=25.0)
    history = _history(synthetic_service, settings)
    model = HistoricalBootstrapReturn(history, settings, np.random.default_rng(2), 2)
    predictions = model.predict(synthetic_task, synthetic_context)
    assert model.simulated_floor_activations > 0
    assert all(min(p.payload.quantiles.values()) >= 25.0 for p in predictions)
