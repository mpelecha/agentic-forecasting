"""Leak-safety and shape tests for the expanded WTI covariate panel.

Every test here runs on synthetic frames — no network, no API keys — so the
point-in-time discipline of the new FRED/EIA covariates is verifiable offline.
"""

from __future__ import annotations

import pandas as pd
import pytest
from aieng.forecasting.data.features import business_daily_expand_from_releases
from energy_oil_forecasting.data import (
    _DURABLE_GOODS_RELEASE_BDAYS_AFTER_MONTH_END,
    _INDPRO_RELEASE_BDAYS_AFTER_MONTH_END,
    DEFAULT_WTI_COVARIATE_SERIES_IDS,
    EXPANDED_WTI_COVARIATE_SERIES_IDS,
    SERIES_ID_CRACK_SPREAD,
    SERIES_ID_CRUDE_STOCKS_EX_SPR,
    SERIES_ID_DURABLE_GOODS_ORDERS,
    SERIES_ID_INDPRO,
    SERIES_ID_OVX_LEVEL,
    SERIES_ID_REFINERY_UTILIZATION,
    _build_release_expanded_feature,
    _monthly_release_stamp,
    _release_to_next_business_open,
)


# ---------------------------------------------------------------------------
# Panel composition
# ---------------------------------------------------------------------------


def test_default_panel_is_unchanged_and_expanded_is_a_strict_superset() -> None:
    """The control arm's covariate list must stay frozen and ordered.

    ``04_systematic_backtest_eval.ipynb``, ``04b_quarterly_backtest_eval.ipynb``
    and the CFM agent all derive their panel from
    ``DEFAULT_WTI_COVARIATE_SERIES_IDS``.  Since ``DartsLightGBMPredictor``
    caches on ``predictor_id`` alone, mutating this list would silently change
    the control arm's inputs while it reused old cached results.
    """
    assert DEFAULT_WTI_COVARIATE_SERIES_IDS == [
        "brent_log_ret_1b_l1b",
        "natgas_log_ret_1b_l1b",
        "gasoline_log_ret_1b_l1b",
        "gold_log_ret_1b_l1b",
        "dollar_index_log_ret_1b_l1b",
        "oil_curve_contango_l1b",
        "vix_level_l1b",
    ]
    assert EXPANDED_WTI_COVARIATE_SERIES_IDS[: len(DEFAULT_WTI_COVARIATE_SERIES_IDS)] == (
        DEFAULT_WTI_COVARIATE_SERIES_IDS
    )
    assert EXPANDED_WTI_COVARIATE_SERIES_IDS[len(DEFAULT_WTI_COVARIATE_SERIES_IDS) :] == [
        SERIES_ID_OVX_LEVEL,
        SERIES_ID_CRUDE_STOCKS_EX_SPR,
        SERIES_ID_REFINERY_UTILIZATION,
        "fin_stress_index_wl",
        SERIES_ID_INDPRO,
        SERIES_ID_DURABLE_GOODS_ORDERS,
        SERIES_ID_CRACK_SPREAD,
    ]
    assert len(set(EXPANDED_WTI_COVARIATE_SERIES_IDS)) == len(EXPANDED_WTI_COVARIATE_SERIES_IDS)


# ---------------------------------------------------------------------------
# Intraday release stamps vs. the midnight expansion grid
# ---------------------------------------------------------------------------


def _weekly_eia_frame() -> pd.DataFrame:
    """Two weekly prints stamped Wednesday 10:30 ET, as :class:`EIAAdapter` emits."""
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-02", "2026-01-09"]),
            "value": [420000.0, 418500.0],
            "released_at": pd.to_datetime(["2026-01-07 15:30", "2026-01-14 15:30"]),
        }
    )


def test_intraday_release_stamps_would_silently_empty_the_expansion() -> None:
    """Pin the failure mode ``_release_to_next_business_open`` exists to prevent.

    ``business_daily_expand_from_releases`` reindexes on a midnight-stamped
    ``bdate_range``; a 10:30 stamp matches no label and is dropped, producing an
    empty covariate rather than an error.
    """
    raw = business_daily_expand_from_releases(_weekly_eia_frame(), start="2026-01-01", end="2026-01-23")
    assert raw.empty, "Expected the intraday stamp to be dropped by the midnight reindex."


def test_release_normalization_makes_weekly_prints_visible_the_next_morning() -> None:
    """A Wednesday 10:30 ET print first appears on Thursday — never same-day."""
    normalized = _release_to_next_business_open(_weekly_eia_frame())
    assert list(normalized["released_at"]) == [
        pd.Timestamp("2026-01-08"),  # Thu after the Wed 2026-01-07 release
        pd.Timestamp("2026-01-15"),
    ]

    feature = _build_release_expanded_feature(normalized, start="2026-01-01", end="2026-01-23")
    assert not feature.empty

    # The 2026-01-02 print is worth 420000. After the builder's own 1-business-day
    # feature lag it can first appear on the business day following 2026-01-08.
    first_visible = feature.loc[feature["value"] == 420000.0, "timestamp"].min()
    assert first_visible == pd.Timestamp("2026-01-09")


def test_expanded_feature_rows_are_never_released_before_their_timestamp() -> None:
    """Every expanded row is self-consistent for the cutoff enforcer."""
    feature = _build_release_expanded_feature(
        _release_to_next_business_open(_weekly_eia_frame()),
        start="2026-01-01",
        end="2026-01-23",
    )
    assert (feature["released_at"] <= feature["timestamp"]).all()


def test_expansion_end_extends_the_covariate_to_the_target_edge() -> None:
    """``end`` must carry the last released value to the target's last session.

    Darts intersects covariate spans, so a covariate that stops at its last
    release would truncate the whole panel and trigger
    ``past_covariates are not long enough``.
    """
    feature = _build_release_expanded_feature(
        _release_to_next_business_open(_weekly_eia_frame()),
        start="2026-01-01",
        end="2026-02-27",
    )
    assert feature["timestamp"].max() >= pd.Timestamp("2026-02-26")
    # Nothing new was invented past the last release — it is a forward-fill.
    assert feature["value"].iloc[-1] == 418500.0


# ---------------------------------------------------------------------------
# Monthly publication lags, pinned to published release calendars
# ---------------------------------------------------------------------------


# Federal Reserve G.17: reference month -> actual 2026 release date.
_G17_2026 = {
    "2025-12-01": "2026-01-16",
    "2026-01-01": "2026-02-18",
    "2026-02-01": "2026-03-16",
    "2026-03-01": "2026-04-16",
    "2026-04-01": "2026-05-15",
    "2026-05-01": "2026-06-15",
    "2026-06-01": "2026-07-17",
    "2026-07-01": "2026-08-18",
    "2026-08-01": "2026-09-18",
    "2026-09-01": "2026-10-16",
    "2026-10-01": "2026-11-17",
    "2026-11-01": "2026-12-16",
}

# Census M3 advance durable goods: reference month -> actual release date.
# February 2026 was rescheduled from 2026-03-25 to 2026-04-07.
_M3_ADVANCE_KNOWN = {
    "2026-02-01": "2026-04-07",
    "2026-06-01": "2026-07-27",
}


@pytest.mark.parametrize(("reference_month", "actual_release"), sorted(_G17_2026.items()))
def test_indpro_release_stamp_never_precedes_the_g17_calendar(reference_month: str, actual_release: str) -> None:
    """INDPRO must never be visible before the G.17 release that published it."""
    frame = pd.DataFrame({"timestamp": pd.to_datetime([reference_month]), "value": [100.0], "released_at": pd.NaT})
    stamped = _monthly_release_stamp(frame, business_days_after_month_end=_INDPRO_RELEASE_BDAYS_AFTER_MONTH_END)
    assert stamped["released_at"].iloc[0] >= pd.Timestamp(actual_release)


@pytest.mark.parametrize(("reference_month", "actual_release"), sorted(_M3_ADVANCE_KNOWN.items()))
def test_durable_goods_release_stamp_survives_the_2026_reschedule(reference_month: str, actual_release: str) -> None:
    """DGORDER must stay hidden through the February 2026 schedule slip.

    The nominal ~19-business-day lag would have exposed the February print on
    2026-03-26, twelve days before its rescheduled 2026-04-07 publication —
    inside the 2026 eval window.
    """
    frame = pd.DataFrame({"timestamp": pd.to_datetime([reference_month]), "value": [100.0], "released_at": pd.NaT})
    stamped = _monthly_release_stamp(frame, business_days_after_month_end=_DURABLE_GOODS_RELEASE_BDAYS_AFTER_MONTH_END)
    assert stamped["released_at"].iloc[0] >= pd.Timestamp(actual_release)


def test_monthly_expansion_hides_the_value_until_its_release() -> None:
    """A monthly observation is invisible on the expanded grid until published."""
    frame = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2026-01-01", "2026-02-01"]), "value": [100.0, 101.0], "released_at": pd.NaT}
    )
    stamped = _monthly_release_stamp(frame, business_days_after_month_end=16)
    feature = _build_release_expanded_feature(stamped, start="2026-01-01", end="2026-05-01")

    january_release = stamped["released_at"].iloc[0]
    before_release = feature[feature["timestamp"] < january_release]
    assert not (before_release["value"] == 100.0).any(), "January value leaked before its release date."
