"""Historical forecast transformations (specification Section 3)."""

from __future__ import annotations

from datetime import datetime

import pytest
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES
from energy_oil_forecasting.cfm_agent_v_6_0.config import (
    CENTER_PERCENTILE_ACTIONS,
    NOVELTY_CLASSES,
    UNCERTAINTY_RUNGS,
    CfmV60Settings,
)
from energy_oil_forecasting.cfm_agent_v_6_0.forecast_engine import PythonForecastEngine
from energy_oil_forecasting.cfm_agent_v_6_0.history import PriceHistory
from energy_oil_forecasting.cfm_agent_v_6_0.schemas import (
    ForecastTransformation,
    HistoricalReturnProfile,
    ModelHorizonForecast,
    PolicyDecision,
)
from energy_oil_forecasting.data import WTI_SERIES_ID


CUTOFF = datetime(2026, 9, 14)
LEVELS = [float(level) for level in STANDARD_QUANTILES]
LATEST_PRICE = 82.0


def _ensemble(center: float = 80.0, spread: float = 20.0, horizon: int = 21) -> ModelHorizonForecast:
    quantiles = {level: center + spread * (level - 0.5) for level in LEVELS}
    return ModelHorizonForecast(
        horizon=horizon, forecast_date="2026-10-13", point_forecast=quantiles[0.5], quantiles=quantiles
    )


def _decision(center="no_change", uncertainty="unchanged", eligible=True, horizon=21) -> PolicyDecision:
    return PolicyDecision(
        policy_id="test",
        horizon=horizon,
        eligible=eligible,
        center_action=center,
        uncertainty_action=uncertainty,
    )


@pytest.fixture
def settings():
    return CfmV60Settings()


@pytest.fixture
def engine(settings):
    return PythonForecastEngine(settings)


@pytest.fixture
def profile(synthetic_service, settings) -> HistoricalReturnProfile:
    context = synthetic_service.context(as_of=CUTOFF)
    history = PriceHistory.from_context(context, WTI_SERIES_ID, [], settings)
    profile = history.profile(21)
    assert profile.eligible
    # The synthetic history's percentiles straddle a median near zero: down percentiles are
    # negative and up percentiles positive, so the directional guard stays inactive.
    assert all(profile.return_percentiles[settings.center_percentiles[a]] < 0 for a in CENTER_PERCENTILE_ACTIONS[:3])
    assert all(profile.return_percentiles[settings.center_percentiles[a]] > 0 for a in CENTER_PERCENTILE_ACTIONS[3:])
    return profile


def test_engine_identity(engine):
    assert engine.engine_id == "python_forecast_engine_v60"
    assert ForecastTransformation.model_fields["engine_id"].default == engine.engine_id


@pytest.mark.parametrize("action", CENTER_PERCENTILE_ACTIONS)
def test_center_percentile_mapping(engine, settings, profile, action):
    ensemble = _ensemble()
    result = engine.transform(
        ensemble, _decision(center=action), "likely_new_relative_to_model_data", LATEST_PRICE, profile
    )
    percentile = settings.center_percentiles[action]
    selected = profile.return_percentiles[percentile]
    assert result.neutral is False
    assert result.selected_center_percentile == percentile
    assert result.selected_center_value == selected
    assert result.center_sign_guard_applied is False
    assert result.raw_center_adjustment == ensemble.point_forecast * selected
    assert result.novelty_multiplier == 1.0
    assert result.novelty_adjusted_amount == result.raw_center_adjustment
    assert result.applied_center_adjustment == result.raw_center_adjustment
    assert result.adjusted_p50 == ensemble.point_forecast + result.raw_center_adjustment
    assert result.uncertainty_multiplier == 1.0
    for level in LEVELS:
        assert result.final_quantiles[level] == pytest.approx(
            ensemble.quantiles[level] + result.applied_center_adjustment
        )
    assert result.final_point_forecast == result.final_quantiles[0.5]
    assert result.historical_profile == profile


@pytest.mark.parametrize("novelty", NOVELTY_CLASSES)
def test_novelty_multipliers(engine, settings, profile, novelty):
    result = engine.transform(_ensemble(), _decision(center="moderate_up"), novelty, LATEST_PRICE, profile)
    assert result.novelty == novelty
    assert result.novelty_multiplier == settings.novelty_multipliers[novelty]
    assert result.novelty_adjusted_amount == result.raw_center_adjustment * settings.novelty_multipliers[novelty]
    assert result.applied_center_adjustment == result.novelty_adjusted_amount
    assert result.adjusted_p50 == 80.0 + result.novelty_adjusted_amount


def test_unknown_novelty_raises(engine, profile):
    with pytest.raises(ValueError, match="novelty"):
        engine.transform(_ensemble(), _decision(center="small_up"), "brand_new", LATEST_PRICE, profile)


def test_directional_sign_guard(engine, settings, profile):
    flipped = profile.model_copy(
        update={
            "return_percentiles": {
                **profile.return_percentiles,
                settings.center_percentiles["small_down"]: 0.012,
                settings.center_percentiles["small_up"]: -0.008,
            }
        }
    )
    down = engine.transform(_ensemble(), _decision(center="small_down"), "indeterminate", LATEST_PRICE, flipped)
    assert down.center_sign_guard_applied is True
    assert down.selected_center_value == 0.012
    assert down.raw_center_adjustment == 0.0
    assert down.applied_center_adjustment == 0.0
    assert down.adjusted_p50 == 80.0
    assert any("directional guard" in warning for warning in down.warnings)
    assert down.neutral is False
    up = engine.transform(_ensemble(), _decision(center="small_up"), "indeterminate", LATEST_PRICE, flipped)
    assert up.center_sign_guard_applied is True and up.applied_center_adjustment == 0.0
    normal = engine.transform(_ensemble(), _decision(center="moderate_down"), "indeterminate", LATEST_PRICE, flipped)
    assert normal.center_sign_guard_applied is False and normal.applied_center_adjustment < 0


def test_emergency_cap(profile):
    settings = CfmV60Settings(emergency_center_cap_usd=0.25)
    engine = PythonForecastEngine(settings)
    result = engine.transform(
        _ensemble(), _decision(center="large_up"), "likely_new_relative_to_model_data", LATEST_PRICE, profile
    )
    assert result.emergency_cap == 0.25
    assert result.novelty_adjusted_amount > 0.25
    assert result.applied_center_adjustment == 0.25
    assert result.adjusted_p50 == 80.25
    assert any("emergency cap" in warning for warning in result.warnings)
    down = engine.transform(
        _ensemble(), _decision(center="large_down"), "likely_new_relative_to_model_data", LATEST_PRICE, profile
    )
    assert down.applied_center_adjustment == -0.25
    fraction = CfmV60Settings(emergency_center_cap_usd=100.0, emergency_center_cap_price_fraction=0.01)
    priced = PythonForecastEngine(fraction).transform(
        _ensemble(), _decision(center="large_up"), "likely_new_relative_to_model_data", 50.0, profile
    )
    assert priced.emergency_cap == 0.5
    unpriced = PythonForecastEngine(fraction).transform(
        _ensemble(), _decision(center="large_up"), "likely_new_relative_to_model_data", None, profile
    )
    assert unpriced.emergency_cap == 100.0
    assert unpriced.latest_price is None


def test_uncertainty_ladder_is_monotone_and_exactly_one_at_unchanged(engine, settings, profile):
    ensemble = _ensemble()
    multipliers = {}
    for rung in UNCERTAINTY_RUNGS:
        result = engine.transform(ensemble, _decision(uncertainty=rung), "indeterminate", LATEST_PRICE, profile)
        coverage = settings.uncertainty_coverage[rung]
        assert result.uncertainty_coverage_target == coverage
        assert result.anchor_coverage == settings.uncertainty_anchor_coverage
        if rung == "unchanged":
            assert result.neutral is True
            assert result.uncertainty_multiplier == 1.0
            assert result.final_quantiles == ensemble.quantiles
        else:
            assert result.neutral is False
            assert result.anchor_band == profile.absolute_deviation_bands[settings.uncertainty_anchor_coverage]
            assert result.target_band == profile.absolute_deviation_bands[coverage]
            assert result.uncertainty_multiplier == result.target_band / result.anchor_band
            assert result.uncertainty_clamp_applied is False
            for level in LEVELS:
                expected = 80.0 + result.uncertainty_multiplier * (ensemble.quantiles[level] - 80.0)
                assert result.final_quantiles[level] == pytest.approx(expected)
            assert result.adjusted_p50 == 80.0
        multipliers[rung] = result.uncertainty_multiplier
    ordered = [multipliers[rung] for rung in UNCERTAINTY_RUNGS]
    assert ordered == sorted(ordered)
    assert all(multipliers[rung] < 1.0 for rung in UNCERTAINTY_RUNGS[:3])
    assert all(multipliers[rung] > 1.0 for rung in UNCERTAINTY_RUNGS[4:])


def test_uncertainty_tie_is_clamped_with_a_warning(engine, settings, profile):
    anchor = settings.uncertainty_anchor_coverage
    tied = profile.model_copy(
        update={
            "absolute_deviation_bands": {
                **profile.absolute_deviation_bands,
                settings.uncertainty_coverage["small_wider"]: profile.absolute_deviation_bands[anchor],
            }
        }
    )
    result = engine.transform(_ensemble(), _decision(uncertainty="small_wider"), "indeterminate", LATEST_PRICE, tied)
    assert result.uncertainty_multiplier == 1.0
    assert result.uncertainty_clamp_applied is True
    assert any("tie" in warning for warning in result.warnings)


def test_ladder_violation_is_a_bug_not_a_forecast(engine, settings, profile):
    broken = profile.model_copy(
        update={
            "absolute_deviation_bands": {
                **profile.absolute_deviation_bands,
                settings.uncertainty_coverage["small_wider"]: 0.5 * profile.absolute_deviation_bands[80],
            }
        }
    )
    with pytest.raises(ValueError, match="ladder violated"):
        engine.transform(_ensemble(), _decision(uncertainty="small_wider"), "indeterminate", LATEST_PRICE, broken)


def test_center_and_uncertainty_compose_around_the_adjusted_p50(engine, settings, profile):
    ensemble = _ensemble()
    result = engine.transform(
        ensemble,
        _decision(center="moderate_up", uncertainty="moderately_wider"),
        "possibly_partly_reflected",
        LATEST_PRICE,
        profile,
    )
    for level in LEVELS:
        expected = result.adjusted_p50 + result.uncertainty_multiplier * (ensemble.quantiles[level] - 80.0)
        assert result.pre_floor_quantiles[level] == pytest.approx(expected)
    assert result.final_quantiles == result.pre_floor_quantiles
    assert result.floor_applied is False
    assert result.final_point_forecast == result.adjusted_p50


def test_ineligible_profile_forces_neutral_with_reason(engine, profile):
    short = profile.model_copy(update={"eligible": False, "n_eligible": 120, "ineligibility_reason": "120 below 500"})
    result = engine.transform(
        _ensemble(),
        _decision(center="large_up", uncertainty="substantially_wider"),
        "indeterminate",
        LATEST_PRICE,
        short,
    )
    assert result.neutral is True
    assert result.neutral_reason is not None and "120 below 500" in result.neutral_reason
    assert result.center_action == "large_up" and result.uncertainty_action == "substantially_wider"
    assert result.applied_center_adjustment == 0.0
    assert result.uncertainty_multiplier == 1.0
    assert result.final_quantiles == _ensemble().quantiles
    assert result.historical_profile == short
    none = engine.transform(_ensemble(), _decision(center="large_up"), "indeterminate", LATEST_PRICE, None)
    assert none.neutral is True and "no historical return profile" in (none.neutral_reason or "")


def test_zero_width_baseline_forces_neutral(engine, profile):
    flat = _ensemble(spread=0.0)
    result = engine.transform(
        flat, _decision(center="large_up", uncertainty="small_wider"), "indeterminate", LATEST_PRICE, profile
    )
    assert result.neutral is True and "width is zero" in (result.neutral_reason or "")
    assert result.final_quantiles == flat.quantiles


def test_exact_neutral_is_a_literal_copy_and_takes_precedence_over_the_floor(engine, profile):
    quantiles = {0.1: 15.423411805148268, 0.5: 87.96894132007836, 0.9: 224.4901508722186}
    ensemble = ModelHorizonForecast(
        horizon=5, forecast_date="2026-09-21", point_forecast=quantiles[0.5], quantiles=quantiles
    )
    result = engine.transform(ensemble, _decision(), "likely_reflected_in_model_data", LATEST_PRICE, profile)
    assert result.neutral is True
    assert result.final_quantiles == quantiles
    assert result.pre_floor_quantiles == quantiles
    assert result.floor_applied is False
    assert result.final_point_forecast == quantiles[0.5]
    assert result.warnings == []
    at_floor = _ensemble(center=engine.s.final_price_floor, spread=0.0)
    exact = engine.transform(at_floor, _decision(), "indeterminate", LATEST_PRICE, profile)
    assert exact.final_quantiles == at_floor.quantiles and exact.floor_applied is False
    below = {0.1: 0.25, 0.5: 0.5, 0.9: 0.75}
    broken = ModelHorizonForecast(horizon=5, forecast_date="2026-09-21", point_forecast=0.5, quantiles=below)
    with pytest.raises(ValueError, match="neutral transformation did not exactly reproduce"):
        engine.transform(broken, _decision(), "indeterminate", LATEST_PRICE, profile)


def test_final_floor_and_ordering(engine, settings, profile):
    low = _ensemble(center=2.0, spread=3.0)  # P05 = 0.65, below the $1.00 final floor
    assert min(low.quantiles.values()) < settings.final_price_floor
    result = engine.transform(low, _decision(uncertainty="substantially_wider"), "indeterminate", 2.0, profile)
    assert result.floor_applied is True
    assert min(result.final_quantiles.values()) == settings.final_price_floor
    assert min(result.pre_floor_quantiles.values()) < settings.final_price_floor
    ordered = [result.final_quantiles[level] for level in LEVELS]
    assert ordered == sorted(ordered)
    assert result.final_point_forecast == result.final_quantiles[0.5]
    assert any("final floor" in warning for warning in result.warnings)


def test_ineligible_decision_zeroes_the_centre(engine, profile):
    result = engine.transform(
        _ensemble(), _decision(center="large_up", eligible=False), "indeterminate", LATEST_PRICE, profile
    )
    assert result.applied_center_adjustment == 0.0
    assert result.adjusted_p50 == 80.0
    assert any("ineligible" in warning for warning in result.warnings)


def test_preconditions(engine, profile):
    with pytest.raises(ValueError, match="missing"):
        engine.transform(
            ModelHorizonForecast(horizon=5, forecast_date="2026-09-21", point_forecast=80.0, quantiles={0.5: 80.0}),
            _decision(),
            "indeterminate",
            LATEST_PRICE,
            profile,
        )
    with pytest.raises(ValueError, match="finite"):
        engine.transform(_ensemble(), _decision(), "indeterminate", float("nan"), profile)


def test_replay_from_the_record_alone_reproduces_final_quantiles(settings, profile):
    engine = PythonForecastEngine(settings)
    original = engine.transform(
        _ensemble(center=77.5, spread=18.0),
        _decision(center="moderate_down", uncertainty="small_wider"),
        "possibly_partly_reflected",
        LATEST_PRICE,
        profile,
    )
    record = ForecastTransformation.model_validate_json(original.model_dump_json())
    replay = PythonForecastEngine(CfmV60Settings()).transform(
        ModelHorizonForecast(
            horizon=record.horizon,
            forecast_date="2026-10-13",
            point_forecast=record.original_point_forecast,
            quantiles=record.original_quantiles,
        ),
        _decision(center=record.center_action, uncertainty=record.uncertainty_action),
        record.novelty,
        record.latest_price,
        record.historical_profile,
    )
    assert replay.final_quantiles == record.final_quantiles
    assert replay.final_point_forecast == record.final_point_forecast
    assert replay.adjusted_p50 == record.adjusted_p50
    assert replay.uncertainty_multiplier == record.uncertainty_multiplier
    assert replay.model_dump() == record.model_dump()
