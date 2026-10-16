"""Deterministic diagnostics (specification Section 2)."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.features import StaticFrameAdapter
from energy_oil_forecasting.cfm_agent_v_6_0.config import CfmV60Settings
from energy_oil_forecasting.cfm_agent_v_6_0.diagnostics import (
    CONTANGO_SERIES_ID,
    VIX_SERIES_ID,
    compute_market_diagnostics,
    curve_state_from,
)
from energy_oil_forecasting.cfm_agent_v_6_0.history import PriceHistory
from energy_oil_forecasting.data import WTI_SERIES_ID


CUTOFF = datetime(2026, 9, 14)


def _diagnostics(service, settings=None, as_of=CUTOFF, covariate_ids=None, horizons=(5, 10, 21)):
    settings = settings or CfmV60Settings()
    context = service.context(as_of=as_of)
    ids = settings.covariate_series_ids if covariate_ids is None else covariate_ids
    history = PriceHistory.from_context(context, WTI_SERIES_ID, ids, settings)
    return history, compute_market_diagnostics(history, as_of=as_of, settings=settings, horizons=horizons)


def _register(service: DataService, series_id: str, dates: pd.DatetimeIndex, values) -> None:
    service.register(
        series_id,
        StaticFrameAdapter(pd.DataFrame({"timestamp": dates, "value": values, "released_at": dates})),
        SeriesMetadata(series_id=series_id, description=series_id, source="t", units="u", frequency="B"),
    )


def test_window_formulas_match_pandas(synthetic_service):
    settings = CfmV60Settings()
    history, diagnostics = _diagnostics(synthetic_service, settings)
    values = pd.Series(history.prices)
    returns = values.pct_change().dropna().reset_index(drop=True)
    for n in settings.diagnostic_return_windows:
        assert diagnostics.cumulative_return[n] == pytest.approx(values.iloc[-1] / values.iloc[-1 - n] - 1)
    for n in settings.diagnostic_volatility_windows:
        expected_vol = returns.tail(n).std(ddof=0) * np.sqrt(settings.trading_days_per_year)
        assert diagnostics.realized_volatility[n] == pytest.approx(expected_vol)
        rolling = returns.rolling(n).std(ddof=0).dropna()
        expected_z = (rolling.iloc[-1] - rolling.mean()) / rolling.std(ddof=0)
        assert diagnostics.volatility_zscore[n] == pytest.approx(expected_z)
        assert diagnostics.drawdown[n] == pytest.approx(values.iloc[-1] / values.tail(n).max() - 1)
    for n in settings.diagnostic_jump_windows:
        recent = returns.tail(n)
        assert diagnostics.jump_zscore[n] == pytest.approx((returns.iloc[-1] - recent.mean()) / recent.std(ddof=0))
    assert set(diagnostics.cumulative_return) == set(settings.diagnostic_return_windows)
    assert set(diagnostics.jump_zscore) == set(settings.diagnostic_jump_windows)
    assert diagnostics.latest_observation_date == "2026-09-14"
    assert diagnostics.latest_value == history.prices[-1]
    assert diagnostics.calendar_gap_days == 0
    assert diagnostics.vix_level == history.covariate_latest_values()[VIX_SERIES_ID]
    assert set(diagnostics.covariate_latest_values) == set(settings.covariate_series_ids)
    assert set(diagnostics.historical_profiles) == {5, 10, 21}
    assert diagnostics.historical_profiles[21] == history.profile(21)
    assert diagnostics.model_disagreement_std == {}
    assert diagnostics.data_quality.observation_counts[WTI_SERIES_ID] == 3000
    assert diagnostics.data_quality.rows_remaining == 3000
    assert diagnostics.data_quality.cutoff_date == "2026-09-14"


def test_short_history_yields_none_not_error(synthetic_service_factory):
    settings = CfmV60Settings()
    _, diagnostics = _diagnostics(synthetic_service_factory(n_observations=12), settings)
    assert diagnostics.cumulative_return[1] is not None
    assert diagnostics.cumulative_return[5] is not None
    assert diagnostics.cumulative_return[21] is None
    assert diagnostics.realized_volatility[5] is not None
    assert diagnostics.realized_volatility[10] is not None
    assert diagnostics.realized_volatility[21] is None
    assert diagnostics.volatility_zscore[21] is None
    assert diagnostics.drawdown[10] is not None and diagnostics.drawdown[63] is None
    assert diagnostics.jump_zscore[10] is not None and diagnostics.jump_zscore[21] is None
    assert not diagnostics.historical_profiles[21].eligible
    _, tiny = _diagnostics(synthetic_service_factory(n_observations=1), settings)
    assert all(value is None for value in tiny.cumulative_return.values())
    assert all(value is None for value in tiny.realized_volatility.values())
    assert all(value is None for value in tiny.jump_zscore.values())
    assert tiny.latest_value is not None


def test_cutoff_safety_uses_only_rows_on_or_before_the_cutoff(synthetic_service_factory):
    settings = CfmV60Settings()
    later = synthetic_service_factory(n_observations=3020, end="2026-10-12")
    _, at_cutoff = _diagnostics(later, settings, as_of=CUTOFF)
    assert at_cutoff.latest_observation_date == "2026-09-14"
    assert at_cutoff.calendar_gap_days == 0
    assert at_cutoff.data_quality.observation_counts[WTI_SERIES_ID] == 3000
    assert at_cutoff.data_quality.last_dates[WTI_SERIES_ID] == "2026-09-14"
    assert all(date <= "2026-09-14" for date in at_cutoff.data_quality.last_dates.values())
    _, weekend = _diagnostics(later, settings, as_of=datetime(2026, 9, 13))
    assert weekend.latest_observation_date == "2026-09-11"
    assert weekend.calendar_gap_days == 2


def test_curve_state_labels_and_flat_threshold():
    settings = CfmV60Settings()
    threshold = settings.curve_flat_threshold
    assert curve_state_from(0.0, threshold) == "flat"
    assert curve_state_from(threshold / 2, threshold) == "flat"
    assert curve_state_from(-threshold / 2, threshold) == "flat"
    assert curve_state_from(threshold, threshold) == "contango"
    assert curve_state_from(-threshold, threshold) == "backwardation"
    assert curve_state_from(0.05, threshold) == "contango"
    assert curve_state_from(-0.05, threshold) == "backwardation"
    assert curve_state_from(None, threshold) is None
    assert curve_state_from(float("nan"), threshold) is None


@pytest.mark.parametrize(
    ("last_contango", "expected"),
    [(0.0004, "flat"), (-0.0004, "flat"), (0.02, "contango"), (-0.02, "backwardation")],
)
def test_curve_state_from_the_latest_served_contango_value(last_contango, expected):
    settings = CfmV60Settings()
    dates = pd.bdate_range(end="2026-09-14", periods=100)
    service = DataService()
    _register(service, WTI_SERIES_ID, dates, np.linspace(60.0, 70.0, 100))
    contango = np.full(100, 0.03)
    contango[-1] = last_contango
    _register(service, CONTANGO_SERIES_ID, dates, contango)
    _register(service, VIX_SERIES_ID, dates, np.full(100, 22.5))
    _, diagnostics = _diagnostics(service, settings, covariate_ids=[CONTANGO_SERIES_ID, VIX_SERIES_ID])
    assert diagnostics.curve_state == expected
    assert diagnostics.vix_level == 22.5
    assert diagnostics.covariate_latest_values[CONTANGO_SERIES_ID] == last_contango


def test_missing_level_covariates_leave_curve_and_vix_none(synthetic_service_factory):
    settings = CfmV60Settings()
    present = [s for s in settings.covariate_series_ids if s not in (CONTANGO_SERIES_ID, VIX_SERIES_ID)]
    _, diagnostics = _diagnostics(synthetic_service_factory(covariate_ids=present), settings)
    assert diagnostics.curve_state is None
    assert diagnostics.vix_level is None
    assert diagnostics.data_quality.missing_covariates == [CONTANGO_SERIES_ID, VIX_SERIES_ID]


def test_diagnostics_are_deterministic_and_serialisable(synthetic_service_factory):
    settings = CfmV60Settings()
    _, first = _diagnostics(synthetic_service_factory(), settings)
    _, second = _diagnostics(synthetic_service_factory(), settings)
    assert first == second
    assert first.model_dump_json() == second.model_dump_json()
