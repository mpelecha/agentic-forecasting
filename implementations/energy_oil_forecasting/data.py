"""Data-service setup for the WTI Crude Oil forecasting experiment.

:func:`build_wti_service` registers the continuous front-month WTI futures
close series (Yahoo Finance ticker ``CL=F``) under the canonical
:data:`WTI_SERIES_ID`.  Both the reference YAML specs under
``implementations/energy_oil_forecasting/specs/`` and the notebooks here
reference the same ``series_id`` via this module.

:func:`build_wti_multivariate_service` additionally registers a **leak-safe
covariate panel** for the covariate-bearing predictors (e.g.
:class:`~aieng.forecasting.methods.numerical.darts_regression.DartsLightGBMPredictor`
with ``covariate_series_ids=...``).  The panel is entirely sourced from Yahoo
Finance — no FRED API key required — and reuses the shared, point-in-time
feature builders in :mod:`aieng.forecasting.data.features`:

- Brent (``BZ=F``), natural gas (``NG=F``), RBOB gasoline (``RB=F``) and gold
  (``GC=F``) close-to-close log returns — the energy complex plus an inflation/
  risk hedge.
- Trade-weighted-style USD index (``DX-Y.NYB``) log return — oil is USD-priced.
- An **oil-futures-curve** contango proxy: ``log(USL / USO)`` level, where
  ``USL`` tracks a 12-month WTI strip and ``USO`` the front month, so a positive
  value is contango and a negative value backwardation — a clean term-structure
  signal with no contract-roll assembly.
- VIX (``^VIX``) level — broad risk/volatility sentiment.

Every covariate is lagged one business day and forward-filled onto a complete
business-day calendar; the :class:`DataService` cutoff then guarantees predictor
context views never include unavailable rows.

Expanded panel
--------------
:data:`EXPANDED_WTI_COVARIATE_SERIES_IDS` is a strict superset of the default
panel that adds oil-market fundamentals and macro context, sourced from FRED
and the EIA rather than Yahoo Finance:

- **OVX level** (FRED ``OVXCLS``) — crude-oil implied volatility, the
  oil-specific counterpart to VIX.
- **Weekly EIA fundamentals** — US crude stocks excluding SPR (``WCESTUS1``)
  and refinery utilisation (``WPULEUS3``), the two headline numbers from the
  Weekly Petroleum Status Report.
- **Financial stress** (FRED ``STLFSI4``) — weekly composite; note ``STLFSI3``
  was discontinued in November 2022.
- **Monthly macro** — industrial production (``INDPRO``) and durable-goods
  orders (``DGORDER``) as demand proxies, each with a publication lag measured
  against the actual release calendar rather than assumed.
- **Crack spread** — ``log(RB=F / CL=F)``, the refining margin that links
  crude to product demand.

The default panel is deliberately **frozen**: the expanded ids live in their
own constant so the existing ``LightGBM + cov`` backtest arm, the CFM agent
(:mod:`energy_oil_forecasting.cfm_agent_v_2_0.agent`) and
``04b_quarterly_backtest_eval.ipynb`` all keep byte-identical covariate lists
and cached results.  :func:`build_wti_multivariate_service` registers whatever
union of the two panels is requested, so one service feeds both arms.

Sub-daily release stamps
~~~~~~~~~~~~~~~~~~~~~~~~
:func:`~aieng.forecasting.data.features.business_daily_expand_from_releases`
reindexes on a **midnight-stamped** business-day calendar, so a release
timestamp carrying a time-of-day component (as :class:`EIAAdapter` correctly
produces) matches no index label and the series silently expands to zero rows.
:func:`_release_to_next_business_open` rolls such stamps forward to the first
business-day midnight strictly after publication, which is both grid-safe and
conservative.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.adapters.eia import EIAAdapter
from aieng.forecasting.data.adapters.fred import FREDAdapter
from aieng.forecasting.data.adapters.yfinance import YFinanceDailyAdapter
from aieng.forecasting.data.features import (
    StaticFrameAdapter,
    apply_one_business_day_feature_lag,
    business_daily_expand_from_releases,
    business_daily_ffill,
    canonical_three_col,
    drop_weekend_timestamp_rows,
    log_ratio_level_feature,
    to_level_feature_from_daily,
    to_log_return_feature,
)


def naive_utc_now() -> datetime:
    """Return current UTC time as a timezone-naive :class:`datetime`.

    :class:`~aieng.forecasting.data.service.DataService` and
    :class:`~aieng.forecasting.data.cutoff.CutoffEnforcer` require naive
    ``as_of`` values — tz-aware timestamps raise on comparison with cached
    series timestamps.
    """
    return datetime.now(tz=timezone.utc).replace(tzinfo=None)


WTI_SERIES_ID = "wti_crude_oil_price"
"""Canonical series ID for the WTI front-month futures close price."""

DEFAULT_CACHE_DIR = Path("data/yfinance")
"""Default yfinance CSV cache directory (resolved relative to CWD at call time)."""

_WTI_HISTORY_START = "2004-01-01"
"""Earliest date requested from yfinance.  Setting an explicit start ensures the
adapter fetches the full available history rather than yfinance's default 30-day
window when no cache exists."""


def build_wti_service(cache_dir: Path | None = None) -> DataService:
    """Return a :class:`DataService` with the WTI Crude Oil daily close series registered.

    Parameters
    ----------
    cache_dir : Path or None
        yfinance CSV cache directory.  Defaults to ``data/yfinance`` relative
        to the current working directory.  Notebooks typically run from their
        own directory so the adapter will transparently fetch from yfinance if
        the cache is absent or stale, then persist the result for subsequent
        runs.

    Returns
    -------
    DataService
        A data service with the WTI series registered, ready to be handed
        to :func:`~aieng.forecasting.evaluation.backtest.backtest` /
        :func:`~aieng.forecasting.evaluation.backtest.cached_multi_backtest` /
        :func:`~aieng.forecasting.evaluation.eval.evaluate`.
    """
    resolved_cache_dir: Path = cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR
    svc = DataService()
    svc.register(
        WTI_SERIES_ID,
        # field defaults to "Adj Close" — matches the cache key cl_f_adj_close_1d.parquet
        # produced by scripts/fetch_wti.py. For futures contracts like CL=F, Adj Close
        # equals Close (no dividend adjustments).
        # start is set explicitly to ensure yfinance fetches full history on a cache miss
        # rather than its default 30-day window.
        YFinanceDailyAdapter(ticker="CL=F", start=_WTI_HISTORY_START, cache_dir=resolved_cache_dir),
        SeriesMetadata(
            series_id=WTI_SERIES_ID,
            description="WTI Crude Oil continuous front-month futures adjusted close (Yahoo Finance CL=F)",
            source="yfinance",
            units="USD/bbl",
            frequency="B",
        ),
    )
    return svc


# ── Covariate panel (all Yahoo Finance) ──────────────────────────────────────
SERIES_ID_BRENT_RETURN = "brent_log_ret_1b_l1b"
SERIES_ID_NATGAS_RETURN = "natgas_log_ret_1b_l1b"
SERIES_ID_GASOLINE_RETURN = "gasoline_log_ret_1b_l1b"
SERIES_ID_GOLD_RETURN = "gold_log_ret_1b_l1b"
SERIES_ID_DOLLAR_INDEX_RETURN = "dollar_index_log_ret_1b_l1b"
SERIES_ID_OIL_CURVE_CONTANGO = "oil_curve_contango_l1b"
SERIES_ID_VIX_LEVEL = "vix_level_l1b"

#: Default covariate panel for :func:`build_wti_multivariate_service`.  Ordered
#: energy-complex first, then macro/risk.  Any series that cannot be fetched is
#: skipped (with a warning) unless ``strict_covariates=True``.
DEFAULT_WTI_COVARIATE_SERIES_IDS: list[str] = [
    SERIES_ID_BRENT_RETURN,
    SERIES_ID_NATGAS_RETURN,
    SERIES_ID_GASOLINE_RETURN,
    SERIES_ID_GOLD_RETURN,
    SERIES_ID_DOLLAR_INDEX_RETURN,
    SERIES_ID_OIL_CURVE_CONTANGO,
    SERIES_ID_VIX_LEVEL,
]

# ── Expanded panel additions (FRED + EIA fundamentals and macro) ─────────────
SERIES_ID_OVX_LEVEL = "ovx_level_l1b"
SERIES_ID_CRUDE_STOCKS_EX_SPR = "crude_stocks_ex_spr_wl"
SERIES_ID_REFINERY_UTILIZATION = "refinery_utilization_wl"
SERIES_ID_FIN_STRESS_INDEX = "fin_stress_index_wl"
SERIES_ID_INDPRO = "indpro_ml"
SERIES_ID_DURABLE_GOODS_ORDERS = "durable_goods_orders_ml"
SERIES_ID_CRACK_SPREAD = "crack_spread_l1b"

#: Covariate panel for the expanded LightGBM arm — the default panel plus
#: oil-market fundamentals and macro context.  A strict superset, so the two
#: arms differ only by the added series.  Pair with
#: ``DartsLightGBMPredictor(variant_tag="expanded")`` so the expanded run gets
#: its own ``predictor_id`` and does not collide with the default arm's cache.
EXPANDED_WTI_COVARIATE_SERIES_IDS: list[str] = [
    *DEFAULT_WTI_COVARIATE_SERIES_IDS,
    SERIES_ID_OVX_LEVEL,
    SERIES_ID_CRUDE_STOCKS_EX_SPR,
    SERIES_ID_REFINERY_UTILIZATION,
    SERIES_ID_FIN_STRESS_INDEX,
    SERIES_ID_INDPRO,
    SERIES_ID_DURABLE_GOODS_ORDERS,
    SERIES_ID_CRACK_SPREAD,
]

# Yahoo Finance tickers backing each covariate.
_BRENT_TICKER = "BZ=F"
_NATGAS_TICKER = "NG=F"
_GASOLINE_TICKER = "RB=F"
_GOLD_TICKER = "GC=F"
_DOLLAR_INDEX_TICKER = "DX-Y.NYB"
_VIX_TICKER = "^VIX"
_OIL_FRONT_ETF_TICKER = "USO"  # United States Oil Fund — front-month WTI
_OIL_12M_ETF_TICKER = "USL"  # United States 12 Month Oil Fund — 12-month strip
_WTI_FUTURES_TICKER = "CL=F"  # front-month WTI, denominator of the crack spread

# FRED series ids backing the expanded panel.
_OVX_FRED_ID = "OVXCLS"  # CBOE Crude Oil ETF Volatility Index
_FIN_STRESS_FRED_ID = "STLFSI4"  # STLFSI3 was discontinued in November 2022
_INDPRO_FRED_ID = "INDPRO"
_DURABLE_GOODS_FRED_ID = "DGORDER"

# EIA API v2 series ids and the routes that hold them.
_CRUDE_STOCKS_EX_SPR_EIA_ID = "WCESTUS1"
_CRUDE_STOCKS_EIA_ROUTE = "petroleum/stoc/wstk"
_REFINERY_UTILIZATION_EIA_ID = "WPULEUS3"
_REFINERY_UTILIZATION_EIA_ROUTE = "petroleum/pnp/wiup"

DEFAULT_FRED_CACHE_DIR = Path("data/fred")
"""Default FRED parquet cache directory (resolved relative to CWD at call time)."""

DEFAULT_EIA_CACHE_DIR = Path("data/eia")
"""Default EIA parquet cache directory (resolved relative to CWD at call time)."""

# ── Publication-lag constants ────────────────────────────────────────────────
# FRED stamps monthly series at the *month start*, so every offset below is
# measured from ``timestamp + MonthEnd(1)`` (i.e. from the end of the reference
# month).  Each was checked against the publisher's actual release calendar
# rather than assumed, and rounded so it is never optimistic.

_INDPRO_RELEASE_BDAYS_AFTER_MONTH_END = 16
"""Federal Reserve G.17 industrial production.

Checked against all twelve published 2026 G.17 release dates (2026-01-16
through 2026-12-16, each covering the prior reference month).  The mid-month
rule of thumb is *not* safe as a tight offset: 11 business days past month end
lands up to 3 days **before** the actual release, and 14 is the smallest offset
that never precedes it — with zero margin in one month.  16 clears every 2026
date by 4-7 days, buying a buffer against the schedule slips that do occur
(see :data:`_DURABLE_GOODS_RELEASE_BDAYS_AFTER_MONTH_END`).  The cost is a
covariate roughly a week staler than reality, which is the safe direction."""

_DURABLE_GOODS_RELEASE_BDAYS_AFTER_MONTH_END = 27
"""Census M3 advance durable-goods report.

The nominal schedule is ~25 calendar days past month end — the June 2026
advance landed 2026-07-27, exactly 19 business days past month end — but the
schedule is not dependable: the February 2026 advance was **rescheduled from
2026-03-25 to 2026-04-07**, a 13-day slip that falls inside the 2026 eval
window.  A 19-business-day offset would have exposed that print 12 days before
it existed.  27 business days is the smallest offset covering both the normal
schedule and the reschedule, at the cost of being ~10 days stale in ordinary
months.

The principled fix is a vintage-aware ``released_at`` from ALFRED / FRED's
``get_series_vintage_dates``, which reports each observation's true first
publication date instead of any fixed offset; :class:`FREDAdapter` notes the
same limitation.  Until then, prefer stale over early."""

_FIN_STRESS_RELEASE_BDAYS = 5
"""STLFSI4 is stamped at the week-ending Friday and published the following
Thursday/Friday; five business days covers either convention."""


def _load_yahoo_close_frame(
    ticker: str,
    *,
    cache_dir: Path,
    start: str,
) -> pd.DataFrame:
    """Fetch a daily adjusted-close ``(timestamp, value)`` frame from Yahoo Finance."""
    adapter = YFinanceDailyAdapter(ticker, field="Adj Close", start=start, cache_dir=cache_dir)
    raw = adapter.fetch()
    frame = raw[["timestamp", "value"]].copy().sort_values("timestamp").reset_index(drop=True)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame.dropna(subset=["value"]).reset_index(drop=True)


def _load_fred_frame(fred_id: str, *, cache_dir: Path) -> pd.DataFrame:
    """Fetch a FRED series as a canonical ``(timestamp, value, released_at)`` frame."""
    return canonical_three_col(FREDAdapter(fred_id, cache_dir=cache_dir).fetch())


def _load_eia_frame(series_id: str, *, route: str, cache_dir: Path) -> pd.DataFrame:
    """Fetch an EIA API v2 series as a canonical frame, preserving the release instant."""
    adapter = EIAAdapter(series_id, route=route, cache_dir=cache_dir)
    raw = adapter.fetch()
    out = raw.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["released_at"] = pd.to_datetime(out["released_at"])
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["timestamp", "value", "released_at"]).reset_index(drop=True)


def _release_to_next_business_open(frame: pd.DataFrame) -> pd.DataFrame:
    """Roll intraday ``released_at`` stamps to the next business-day midnight.

    :func:`~aieng.forecasting.data.features.business_daily_expand_from_releases`
    reindexes on a midnight-stamped ``bdate_range``.  A release stamped
    ``Wednesday 10:30`` matches no label there, is dropped by the reindex, and
    cannot be recovered by the subsequent forward-fill — the covariate silently
    expands to zero rows.  Flooring to the release day and adding one business
    day maps that print to Thursday 00:00: on the grid, and strictly after the
    information was public, so the value is never visible early.
    """
    out = frame.copy()
    out["released_at"] = pd.to_datetime(out["released_at"]).dt.floor("D") + pd.offsets.BDay(1)
    return out


def _build_daily_fred_level_feature(fred_id: str, *, cache_dir: Path) -> pd.DataFrame:
    """Daily FRED level → weekend-stripped, business-day forward-filled, lagged 1B.

    Forward-filling before the lag keeps the covariate defined on every Mon-Fri
    session, so a market holiday observed by FRED but not by NYMEX cannot end
    the covariate short of a forecast origin.
    """
    frame = _load_fred_frame(fred_id, cache_dir=cache_dir)
    frame = drop_weekend_timestamp_rows(frame)
    frame = business_daily_ffill(frame)
    return apply_one_business_day_feature_lag(to_level_feature_from_daily(frame))


def _build_release_expanded_feature(
    frame: pd.DataFrame,
    *,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Expand a release-stamped low-frequency series onto business days, lagged 1B.

    ``end`` must reach the target's last session.  With the builder's default
    (``max(released_at) + 1B``) a monthly covariate stops weeks behind the
    target; :func:`_build_past_covariates` intersects covariate spans, so a
    single short series truncates the whole panel and Darts then raises
    ``past_covariates are not long enough``.  Extending the span only repeats
    the last *released* value — each expanded row carries
    ``released_at = timestamp``, so the cutoff still hides everything the
    market had not seen.
    """
    daily = business_daily_expand_from_releases(frame, start=start, end=end)
    return apply_one_business_day_feature_lag(daily)


def _monthly_release_stamp(frame: pd.DataFrame, *, business_days_after_month_end: int) -> pd.DataFrame:
    """Stamp a month-start-indexed FRED series with its true publication date."""
    out = frame.copy()
    out["released_at"] = (
        pd.to_datetime(out["timestamp"]) + pd.offsets.MonthEnd(1) + pd.offsets.BDay(business_days_after_month_end)
    )
    return out


def build_wti_multivariate_service(  # noqa: PLR0912, PLR0915
    cache_dir: Path | None = None,
    *,
    covariate_series_ids: list[str] | None = None,
    strict_covariates: bool = False,
    start: str = _WTI_HISTORY_START,
    fred_cache_dir: Path | None = None,
    eia_cache_dir: Path | None = None,
) -> DataService:
    """Return a :class:`DataService` with the WTI target **and** a covariate panel.

    Builds on :func:`build_wti_service` (so the ``wti_crude_oil_price`` target id
    and every YAML spec keep working unchanged), then registers the leak-safe
    covariate series described in the module docstring.  Hand the result to the
    backtest harness and point a covariate-bearing predictor at the registered
    ids, e.g.::

        svc = build_wti_multivariate_service()
        covs = [c for c in DEFAULT_WTI_COVARIATE_SERIES_IDS if c in set(svc.series_ids)]
        DartsLightGBMPredictor(lags=21, lags_past_covariates=21, covariate_series_ids=covs)

    Request the expanded panel by passing its ids, and give the predictor a
    ``variant_tag`` so its cached backtest stays distinct from the default
    arm's::

        svc = build_wti_multivariate_service(
            covariate_series_ids=EXPANDED_WTI_COVARIATE_SERIES_IDS,
        )
        covs = [c for c in EXPANDED_WTI_COVARIATE_SERIES_IDS if c in set(svc.series_ids)]
        DartsLightGBMPredictor(
            lags=21,
            lags_past_covariates=21,
            covariate_series_ids=covs,
            variant_tag="expanded",
        )

    Non-covariate predictors simply ignore the extra series, so a single service
    can feed an entire leaderboard.

    Parameters
    ----------
    cache_dir : Path or None
        yfinance CSV cache directory (shared with the target).  Defaults to
        :data:`DEFAULT_CACHE_DIR`.
    covariate_series_ids : list[str] or None
        Subset of :data:`DEFAULT_WTI_COVARIATE_SERIES_IDS` or
        :data:`EXPANDED_WTI_COVARIATE_SERIES_IDS` to register.  ``None``
        registers the full default panel.
    strict_covariates : bool
        If ``True``, any covariate fetch/build failure raises.  If ``False``
        (default), unavailable covariates are skipped with a warning so the
        service still builds offline / under partial connectivity.
    start : str
        Earliest date requested from Yahoo Finance for the covariates.
    fred_cache_dir : Path or None
        FRED parquet cache directory for the expanded panel's macro series.
        Defaults to :data:`DEFAULT_FRED_CACHE_DIR`.
    eia_cache_dir : Path or None
        EIA parquet cache directory for the expanded panel's weekly petroleum
        series.  Defaults to :data:`DEFAULT_EIA_CACHE_DIR`.
    """
    resolved_cache_dir: Path = cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR
    resolved_fred_cache_dir: Path = fred_cache_dir if fred_cache_dir is not None else DEFAULT_FRED_CACHE_DIR
    resolved_eia_cache_dir: Path = eia_cache_dir if eia_cache_dir is not None else DEFAULT_EIA_CACHE_DIR
    svc = build_wti_service(cache_dir=resolved_cache_dir)

    desired = set(covariate_series_ids if covariate_series_ids is not None else DEFAULT_WTI_COVARIATE_SERIES_IDS)

    def _handle_error(series_id: str, exc: Exception) -> None:
        if strict_covariates:
            raise RuntimeError(f"Failed to build required covariate {series_id!r}.") from exc
        warnings.warn(f"Skipping unavailable covariate {series_id!r}: {exc}", stacklevel=2)

    # Release-expanded covariates must reach the target's last session, or the
    # covariate-span intersection in Darts truncates the whole panel.  Read the
    # target's end from the registered series rather than assuming "today".
    _target = svc.get_series(WTI_SERIES_ID, as_of=naive_utc_now())
    _target_end = pd.Timestamp(_target["timestamp"].max())
    expansion_end = str((_target_end + pd.offsets.BDay(1)).date())

    # ── Daily log-return covariates (energy complex + gold) ───────────────────
    _return_covariates = {
        SERIES_ID_BRENT_RETURN: (_BRENT_TICKER, "Brent crude (BZ=F) close-to-close log return, lagged 1 business day"),
        SERIES_ID_NATGAS_RETURN: (_NATGAS_TICKER, "Henry Hub natural gas (NG=F) log return, lagged 1 business day"),
        SERIES_ID_GASOLINE_RETURN: (_GASOLINE_TICKER, "RBOB gasoline (RB=F) log return, lagged 1 business day"),
        SERIES_ID_GOLD_RETURN: (_GOLD_TICKER, "Gold (GC=F) log return, lagged 1 business day"),
        SERIES_ID_DOLLAR_INDEX_RETURN: (
            _DOLLAR_INDEX_TICKER,
            "US Dollar Index (DX-Y.NYB) log return, lagged 1 business day",
        ),
    }
    for series_id, (ticker, description) in _return_covariates.items():
        if series_id not in desired:
            continue
        try:
            close = _load_yahoo_close_frame(ticker, cache_dir=resolved_cache_dir, start=start)
            feature = apply_one_business_day_feature_lag(to_log_return_feature(close))
            svc.register(
                series_id,
                StaticFrameAdapter(feature),
                SeriesMetadata(
                    series_id=series_id,
                    description=description,
                    source=f"Yahoo Finance ({ticker}), derived",
                    units="log-return",
                    frequency="B",
                    table_id=f"yahoo:{ticker}:log-return-l1b",
                ),
            )
        except (RuntimeError, ValueError, KeyError) as exc:
            _handle_error(series_id, exc)

    # ── Oil-futures-curve contango proxy: log(USL / USO) ──────────────────────
    if SERIES_ID_OIL_CURVE_CONTANGO in desired:
        try:
            usl = _load_yahoo_close_frame(_OIL_12M_ETF_TICKER, cache_dir=resolved_cache_dir, start=start)
            uso = _load_yahoo_close_frame(_OIL_FRONT_ETF_TICKER, cache_dir=resolved_cache_dir, start=start)
            curve = log_ratio_level_feature(usl, uso)
            svc.register(
                SERIES_ID_OIL_CURVE_CONTANGO,
                StaticFrameAdapter(curve),
                SeriesMetadata(
                    series_id=SERIES_ID_OIL_CURVE_CONTANGO,
                    description=(
                        "WTI futures-curve shape: log(USL/USO) level (>0 contango, <0 backwardation), "
                        "lagged 1 business day"
                    ),
                    source="Yahoo Finance (USL, USO), derived",
                    units="log-ratio",
                    frequency="B",
                    table_id="yahoo:USL-USO:log-ratio-l1b",
                ),
            )
        except (RuntimeError, ValueError, KeyError) as exc:
            _handle_error(SERIES_ID_OIL_CURVE_CONTANGO, exc)

    # ── VIX level ─────────────────────────────────────────────────────────────
    if SERIES_ID_VIX_LEVEL in desired:
        try:
            vix_close = _load_yahoo_close_frame(_VIX_TICKER, cache_dir=resolved_cache_dir, start=start)
            vix_level = apply_one_business_day_feature_lag(to_level_feature_from_daily(vix_close))
            svc.register(
                SERIES_ID_VIX_LEVEL,
                StaticFrameAdapter(vix_level),
                SeriesMetadata(
                    series_id=SERIES_ID_VIX_LEVEL,
                    description="CBOE VIX close level, lagged 1 business day",
                    source=f"Yahoo Finance ({_VIX_TICKER})",
                    units="index-level",
                    frequency="B",
                    table_id="yahoo:^VIX:close-l1b",
                ),
            )
        except (RuntimeError, ValueError, KeyError) as exc:
            _handle_error(SERIES_ID_VIX_LEVEL, exc)

    # ── OVX level (oil-specific implied volatility) ───────────────────────────
    if SERIES_ID_OVX_LEVEL in desired:
        try:
            ovx_level = _build_daily_fred_level_feature(_OVX_FRED_ID, cache_dir=resolved_fred_cache_dir)
            svc.register(
                SERIES_ID_OVX_LEVEL,
                StaticFrameAdapter(ovx_level),
                SeriesMetadata(
                    series_id=SERIES_ID_OVX_LEVEL,
                    description="CBOE Crude Oil ETF Volatility Index (OVX) close level, lagged 1 business day",
                    source=f"FRED ({_OVX_FRED_ID})",
                    units="index-level",
                    frequency="B",
                    table_id=f"fred:{_OVX_FRED_ID}:close-l1b",
                ),
            )
        except (RuntimeError, ValueError, KeyError) as exc:
            _handle_error(SERIES_ID_OVX_LEVEL, exc)

    # ── Weekly EIA petroleum fundamentals ─────────────────────────────────────
    # Both are stamped at the week-ending Friday and published in the Wednesday
    # 10:30 ET Weekly Petroleum Status Report.  The intraday release stamp is
    # rolled to the next business-day midnight so the daily expansion's
    # midnight reindex can see it (see _release_to_next_business_open).
    _eia_covariates = {
        SERIES_ID_CRUDE_STOCKS_EX_SPR: (
            _CRUDE_STOCKS_EX_SPR_EIA_ID,
            _CRUDE_STOCKS_EIA_ROUTE,
            "US ending stocks of crude oil excluding SPR, weekly (EIA WPSR), expanded to business days, lagged 1B",
            "thousand-barrels",
        ),
        SERIES_ID_REFINERY_UTILIZATION: (
            _REFINERY_UTILIZATION_EIA_ID,
            _REFINERY_UTILIZATION_EIA_ROUTE,
            "US percent utilization of refinery operable capacity, weekly (EIA WPSR), "
            "expanded to business days, lagged 1B",
            "percent",
        ),
    }
    for series_id, (eia_id, route, description, units) in _eia_covariates.items():
        if series_id not in desired:
            continue
        try:
            raw = _load_eia_frame(eia_id, route=route, cache_dir=resolved_eia_cache_dir)
            feature = _build_release_expanded_feature(
                _release_to_next_business_open(raw),
                start=start,
                end=expansion_end,
            )
            svc.register(
                series_id,
                StaticFrameAdapter(feature),
                SeriesMetadata(
                    series_id=series_id,
                    description=description,
                    source=f"EIA API v2 ({eia_id}, route {route})",
                    units=units,
                    frequency="B",
                    table_id=f"eia:{eia_id}:weekly-expanded-l1b",
                ),
            )
        except (RuntimeError, ValueError, KeyError) as exc:
            _handle_error(series_id, exc)

    # ── Weekly financial stress index ─────────────────────────────────────────
    if SERIES_ID_FIN_STRESS_INDEX in desired:
        try:
            stress = _load_fred_frame(_FIN_STRESS_FRED_ID, cache_dir=resolved_fred_cache_dir)
            stress["released_at"] = pd.to_datetime(stress["timestamp"]) + pd.offsets.BDay(_FIN_STRESS_RELEASE_BDAYS)
            stress_feature = _build_release_expanded_feature(stress, start=start, end=expansion_end)
            svc.register(
                SERIES_ID_FIN_STRESS_INDEX,
                StaticFrameAdapter(stress_feature),
                SeriesMetadata(
                    series_id=SERIES_ID_FIN_STRESS_INDEX,
                    description=(
                        "St. Louis Fed Financial Stress Index (STLFSI4), weekly, "
                        "conservative release lag + expansion to business days, lagged 1B"
                    ),
                    source=f"FRED ({_FIN_STRESS_FRED_ID})",
                    units="index-level",
                    frequency="B",
                    table_id=f"fred:{_FIN_STRESS_FRED_ID}:weekly-expanded-l1b",
                ),
            )
        except (RuntimeError, ValueError, KeyError) as exc:
            _handle_error(SERIES_ID_FIN_STRESS_INDEX, exc)

    # ── Monthly macro demand proxies ──────────────────────────────────────────
    _monthly_covariates = {
        SERIES_ID_INDPRO: (
            _INDPRO_FRED_ID,
            _INDPRO_RELEASE_BDAYS_AFTER_MONTH_END,
            "US industrial production index, monthly, G.17 release lag + expansion to business days, lagged 1B",
            "index-2017=100",
        ),
        SERIES_ID_DURABLE_GOODS_ORDERS: (
            _DURABLE_GOODS_FRED_ID,
            _DURABLE_GOODS_RELEASE_BDAYS_AFTER_MONTH_END,
            "US manufacturers' new orders for durable goods, monthly, Census M3 advance release lag + "
            "expansion to business days, lagged 1B",
            "millions-USD",
        ),
    }
    for series_id, (fred_id, release_bdays, description, units) in _monthly_covariates.items():
        if series_id not in desired:
            continue
        try:
            raw = _load_fred_frame(fred_id, cache_dir=resolved_fred_cache_dir)
            stamped = _monthly_release_stamp(raw, business_days_after_month_end=release_bdays)
            feature = _build_release_expanded_feature(stamped, start=start, end=expansion_end)
            svc.register(
                series_id,
                StaticFrameAdapter(feature),
                SeriesMetadata(
                    series_id=series_id,
                    description=description,
                    source=f"FRED ({fred_id})",
                    units=units,
                    frequency="B",
                    table_id=f"fred:{fred_id}:monthly-expanded-l1b",
                ),
            )
        except (RuntimeError, ValueError, KeyError) as exc:
            _handle_error(series_id, exc)

    # ── Crack spread: log(RBOB / WTI) ─────────────────────────────────────────
    # RBOB quotes $/gallon and WTI $/bbl, but the 42x unit conversion is a
    # constant additive offset in log space and therefore invisible to the
    # model — this is the units-corrected log crack up to that constant.
    if SERIES_ID_CRACK_SPREAD in desired:
        try:
            rbob = _load_yahoo_close_frame(_GASOLINE_TICKER, cache_dir=resolved_cache_dir, start=start)
            wti = _load_yahoo_close_frame(_WTI_FUTURES_TICKER, cache_dir=resolved_cache_dir, start=start)
            crack = log_ratio_level_feature(rbob, wti)
            svc.register(
                SERIES_ID_CRACK_SPREAD,
                StaticFrameAdapter(crack),
                SeriesMetadata(
                    series_id=SERIES_ID_CRACK_SPREAD,
                    description=(
                        "Refining margin proxy: log(RB=F/CL=F) level (higher = wider crack), lagged 1 business day"
                    ),
                    source="Yahoo Finance (RB=F, CL=F), derived",
                    units="log-ratio",
                    frequency="B",
                    table_id="yahoo:RB-CL:log-ratio-l1b",
                ),
            )
        except (RuntimeError, ValueError, KeyError) as exc:
            _handle_error(SERIES_ID_CRACK_SPREAD, exc)

    return svc


__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_EIA_CACHE_DIR",
    "DEFAULT_FRED_CACHE_DIR",
    "DEFAULT_WTI_COVARIATE_SERIES_IDS",
    "EXPANDED_WTI_COVARIATE_SERIES_IDS",
    "SERIES_ID_BRENT_RETURN",
    "SERIES_ID_CRACK_SPREAD",
    "SERIES_ID_CRUDE_STOCKS_EX_SPR",
    "SERIES_ID_DOLLAR_INDEX_RETURN",
    "SERIES_ID_DURABLE_GOODS_ORDERS",
    "SERIES_ID_FIN_STRESS_INDEX",
    "SERIES_ID_GASOLINE_RETURN",
    "SERIES_ID_GOLD_RETURN",
    "SERIES_ID_INDPRO",
    "SERIES_ID_NATGAS_RETURN",
    "SERIES_ID_OIL_CURVE_CONTANGO",
    "SERIES_ID_OVX_LEVEL",
    "SERIES_ID_REFINERY_UTILIZATION",
    "SERIES_ID_VIX_LEVEL",
    "WTI_SERIES_ID",
    "build_wti_multivariate_service",
    "build_wti_service",
    "naive_utc_now",
]
