"""History and data preparation (specification Sections 1.2, 1.5, 3.4)."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.features import StaticFrameAdapter
from energy_oil_forecasting.cfm_agent_v_6_0.config import CfmV60Settings
from energy_oil_forecasting.cfm_agent_v_6_0.history import PriceHistory, forecast_date_for
from energy_oil_forecasting.data import WTI_SERIES_ID


CUTOFF = datetime(2026, 9, 14)


def _history(service, settings=None, covariate_ids=None):
    settings = settings or CfmV60Settings()
    context = service.context(as_of=CUTOFF)
    ids = settings.covariate_series_ids if covariate_ids is None else covariate_ids
    return PriceHistory.from_context(context, WTI_SERIES_ID, ids, settings)


def _register_wti(values: list[float], end: str = "2026-09-14") -> DataService:
    dates = pd.bdate_range(end=end, periods=len(values))
    service = DataService()
    service.register(
        WTI_SERIES_ID,
        StaticFrameAdapter(pd.DataFrame({"timestamp": dates, "value": values, "released_at": dates})),
        SeriesMetadata(series_id=WTI_SERIES_ID, description="t", source="t", units="USD/bbl", frequency="B"),
    )
    return service


def test_floor_is_applied_before_returns():
    service = _register_wti([50.0, 40.0, -37.63, 10.0, 12.0] + [12.0] * 20)
    history = _history(service, covariate_ids=[])
    settings = history.settings
    assert history.input_floor_activation_count == 1
    assert history.input_floor_activation_dates == [str(history.dates[2].date())]
    assert history.prices[2] == settings.input_price_floor
    assert history.returns[1] == pytest.approx(settings.input_price_floor / 40.0 - 1)
    assert history.returns[2] == pytest.approx(10.0 / settings.input_price_floor - 1)
    assert history.returns[0] == pytest.approx(40.0 / 50.0 - 1)


def test_returns_are_simple_and_unclipped(synthetic_service):
    history = _history(synthetic_service)
    expected = history.prices[1:] / history.prices[:-1] - 1
    np.testing.assert_array_equal(history.returns, expected)
    assert len(history.returns) == len(history.prices) - 1


def test_window_keeps_the_trailing_rows(synthetic_service):
    settings = CfmV60Settings(history_window_business_days=400)
    full = _history(synthetic_service)
    windowed = _history(synthetic_service, settings)
    assert len(windowed.prices) == 400
    assert windowed.window.rows == 400
    assert windowed.window.setting_business_days == 400
    np.testing.assert_array_equal(windowed.prices, full.prices[-400:])
    assert windowed.window.end_date == full.window.end_date
    assert windowed.total_target_observations == 3000
    assert windowed.data_fingerprint != full.data_fingerprint


def test_horizon_returns_are_all_overlapping(synthetic_service):
    history = _history(synthetic_service)
    for h in (5, 10, 21):
        returns = history.horizon_returns(h)
        assert len(returns) == len(history.prices) - h
        np.testing.assert_allclose(returns, history.prices[h:] / history.prices[:-h] - 1)


def test_profile_percentiles_and_bands_equal_numpy(synthetic_service):
    history = _history(synthetic_service)
    settings = history.settings
    profile = history.profile(21)
    returns = history.horizon_returns(21)
    assert profile.eligible and profile.n_eligible == len(returns)
    assert profile.median == float(np.median(returns))
    assert profile.mean == float(np.mean(returns))
    assert set(profile.return_percentiles) == set(settings.center_percentiles.values())
    for percentile, value in profile.return_percentiles.items():
        assert value == float(np.percentile(returns, percentile))
    expected_coverages = set(settings.uncertainty_coverage.values()) | {settings.uncertainty_anchor_coverage}
    assert set(profile.absolute_deviation_bands) == expected_coverages
    deviations = np.abs(returns - np.median(returns))
    for coverage, band in profile.absolute_deviation_bands.items():
        assert band == float(np.percentile(deviations, coverage))
    assert profile.window_start == history.window.start_date
    assert profile.window_end == history.window.end_date
    assert profile.conditioning == "none"


def test_profile_ineligible_below_minimum(synthetic_service_factory):
    settings = CfmV60Settings()
    service = synthetic_service_factory(n_observations=settings.min_historical_returns + 10)
    history = _history(service, settings)
    profile = history.profile(21)
    assert profile.n_eligible == settings.min_historical_returns + 10 - 21
    assert not profile.eligible
    assert profile.ineligibility_reason and "below the minimum" in profile.ineligibility_reason
    assert profile.median is not None
    assert history.profile(5).eligible


def test_profile_with_no_returns():
    history = _history(_register_wti([10.0, 11.0]), covariate_ids=[])
    profile = history.profile(5)
    assert profile.n_eligible == 0 and not profile.eligible
    assert profile.median is None and profile.return_percentiles == {}


def test_fingerprint_is_stable_and_reflects_served_data(synthetic_service_factory):
    settings = CfmV60Settings()
    first = _history(synthetic_service_factory(), settings)
    second = _history(synthetic_service_factory(), settings)
    assert first.data_fingerprint == second.data_fingerprint
    gapped_service = synthetic_service_factory(covariate_gaps={"brent_log_ret_1b_l1b": [2990, 2991]})
    gapped = _history(gapped_service, settings)
    assert gapped.covariate_fill_counts["brent_log_ret_1b_l1b"] == 2
    assert gapped.data_fingerprint != first.data_fingerprint
    payload = gapped.fingerprint_payload()
    assert len(payload["series"]["brent_log_ret_1b_l1b"]) == 2998
    assert len(payload["series"][WTI_SERIES_ID]) == 3000
    assert payload["cutoff"] == "2026-09-14" and payload["window"] == 0
    assert set(payload["series"]) == {WTI_SERIES_ID, *settings.covariate_series_ids}


def test_fingerprint_uses_floored_prices():
    raw = [50.0, 40.0, -37.63, 10.0, 12.0]
    history = _history(_register_wti(raw), covariate_ids=[])
    values = [value for _, value in history.fingerprint_payload()["series"][WTI_SERIES_ID]]
    assert values == [50.0, 40.0, history.settings.input_price_floor, 10.0, 12.0]


def test_missing_covariate_is_recorded_not_raised(synthetic_service_factory):
    settings = CfmV60Settings()
    present = [series_id for series_id in settings.covariate_series_ids if series_id != "vix_level_l1b"]
    history = _history(synthetic_service_factory(covariate_ids=present), settings)
    assert history.missing_covariates == ["vix_level_l1b"]
    assert history.fingerprint_payload()["series"]["vix_level_l1b"] is None
    assert "vix_level_l1b" not in history.aligned.columns
    assert history.covariate_latest_values().get("vix_level_l1b") is None


def test_alignment_fill_and_drop_are_audited(synthetic_service_factory):
    settings = CfmV60Settings()
    limit = settings.covariate_forward_fill_limit
    long_gap = list(range(1000, 1000 + limit + 3))
    service = synthetic_service_factory(covariate_gaps={"gold_log_ret_1b_l1b": long_gap, "vix_level_l1b": [5]})
    history = _history(service, settings)
    assert history.rows_before_alignment == 3000
    assert history.covariate_fill_counts["gold_log_ret_1b_l1b"] == limit
    assert history.covariate_fill_counts["vix_level_l1b"] == 1
    assert history.rows_dropped == 3
    assert history.rows_remaining == 3000 - 3
    assert list(history.aligned.columns) == list(settings.covariate_series_ids)
    assert not history.aligned.isna().any().any()


def test_forecast_dates_are_cutoff_plus_business_days():
    assert forecast_date_for("2026-09-14", 5) == datetime(2026, 9, 21)
    assert forecast_date_for("2026-09-11", 1) == datetime(2026, 9, 14)
    assert forecast_date_for("2026-09-12", 1) == datetime(2026, 9, 14)


def test_conditioning_path_runs(synthetic_service):
    settings = CfmV60Settings(historical_conditioning="volatility_quintile")
    history = _history(synthetic_service, settings)
    unconditioned = _history(synthetic_service).profile(10)
    profile = history.profile(10)
    assert profile.conditioning == "volatility_quintile"
    assert profile.conditioning_bin is not None
    assert 0 <= profile.conditioning_bin < settings.conditioning_quantile_bins
    assert 0 < profile.n_eligible < unconditioned.n_eligible
    assert set(profile.return_percentiles) == set(unconditioned.return_percentiles)
    assert set(profile.absolute_deviation_bands) == set(unconditioned.absolute_deviation_bands)
