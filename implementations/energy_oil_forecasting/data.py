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
"""

from __future__ import annotations

import os

import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.adapters.yfinance import YFinanceDailyAdapter
from aieng.forecasting.data.features import (
    StaticFrameAdapter,
    apply_one_business_day_feature_lag,
    business_daily_expand_from_releases,
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
SERIES_ID_OVX_LEVEL = "ovx_level_l1b"
SERIES_ID_CRACK_SPREAD = "crack_spread_321_l1b"
SERIES_ID_UST10Y_LEVEL = "ust10y_level_l1b"
SERIES_ID_BRENT_LEVEL = "brent_level_l1b"
SERIES_ID_COPPER_LEVEL = "copper_level_l1b"
SERIES_ID_SP500_LEVEL = "sp500_level_l1b"
SERIES_ID_DOLLAR_INDEX_LEVEL = "dollar_index_level_l1b"
SERIES_ID_CRUDE_STOCKS = "crude_stocks_ex_spr_wl"
SERIES_ID_REFINERY_UTILIZATION = "refinery_utilization_wl"

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

#: The default panel plus OVX, the 3-2-1 crack spread, and the 10-year yield.
#:
#: Deliberately a SEPARATE list rather than three more entries above. Every
#: cached numerical predictor keys on ``predictor_id``, which carries no record
#: of its covariate panel, so widening the default in place would leave
#: ``ecm_regression`` and ``darts_lightgbm_cov`` pointing at cache files
#: computed from a different set of inputs -- silently reloaded and reported as
#: the same model.
EXPANDED_WTI_COVARIATE_SERIES_IDS: list[str] = [
    *DEFAULT_WTI_COVARIATE_SERIES_IDS,
    SERIES_ID_OVX_LEVEL,
    SERIES_ID_CRACK_SPREAD,
    SERIES_ID_UST10Y_LEVEL,
    SERIES_ID_BRENT_LEVEL,
    SERIES_ID_COPPER_LEVEL,
    SERIES_ID_SP500_LEVEL,
    SERIES_ID_DOLLAR_INDEX_LEVEL,
    SERIES_ID_CRUDE_STOCKS,
    SERIES_ID_REFINERY_UTILIZATION,
]

#: The level-valued members of the expanded panel, for use as an ECM's
#: ``long_run_only_covariate_series_ids``.
#:
#: This is the textbook Engle-Granger split rather than a heuristic: levels are
#: what can cointegrate with the target, differences are what drive short-run
#: dynamics. The five here are prices or index levels that mean-revert around a
#: structural relationship with crude; the remaining five are already log
#: returns, i.e. already differenced, and belong in the short-run equation.
#:
#: Note this is a different rationale from the "level-only macro" variant in the
#: older local-run cache. There the five long-run-only series were weekly and
#: monthly releases forward-filled onto a business-day calendar, whose daily
#: differences are almost all exactly zero; excluding them from the short-run
#: equation removed a degenerate regressor. Here every series trades daily, so
#: the argument is about cointegration structure, not forward-fill artifacts.
LEVEL_VALUED_WTI_COVARIATE_SERIES_IDS: list[str] = [
    SERIES_ID_OIL_CURVE_CONTANGO,
    SERIES_ID_VIX_LEVEL,
    SERIES_ID_OVX_LEVEL,
    SERIES_ID_CRACK_SPREAD,
    SERIES_ID_UST10Y_LEVEL,
    SERIES_ID_BRENT_LEVEL,
    SERIES_ID_COPPER_LEVEL,
    SERIES_ID_SP500_LEVEL,
    SERIES_ID_DOLLAR_INDEX_LEVEL,
    SERIES_ID_CRUDE_STOCKS,
    SERIES_ID_REFINERY_UTILIZATION,
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
_OVX_TICKER = "^OVX"  # CBOE Crude Oil ETF Volatility Index — oil-specific vol
_HEATING_OIL_TICKER = "HO=F"  # ULSD/heating oil, the distillate leg of the 3-2-1 crack
_UST10Y_TICKER = "^TNX"  # CBOE 10-Year Treasury Note Yield Index
_WTI_FUTURES_TICKER = "CL=F"  # front-month WTI, the crude leg of the crack spread
_COPPER_TICKER = "HG=F"  # COMEX copper — the daily proxy for global industrial activity
_SP500_TICKER = "^GSPC"  # S&P 500 — aggregate demand and risk appetite

#: Gallons per barrel — converts the refined-product legs of the crack spread
#: ($/gal) into the crude leg's units ($/bbl).
_GALLONS_PER_BARREL = 42.0


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


#: EIA weekly petroleum series. Keys are the ids used above; values are the
#: EIA series id and a human description.
_EIA_SERIES: dict[str, tuple[str, str]] = {
    SERIES_ID_CRUDE_STOCKS: (
        "PET.WCESTUS1.W",
        "U.S. ending stocks of crude oil excluding SPR, weekly (thousand barrels)",
    ),
    SERIES_ID_REFINERY_UTILIZATION: (
        "PET.WPULEUS3.W",
        "U.S. percent utilization of refinery operable capacity, weekly",
    ),
}

#: Calendar days from an EIA weekly period-ending date to the moment the figure
#: is public. The Weekly Petroleum Status Report covers the week ending Friday
#: and is released the following Wednesday, so five days is the true lag. Seven
#: is used instead because the release slips to Thursday whenever Monday is a
#: federal holiday, and a lag that is one day too long costs nothing while one
#: that is one day too short leaks a market-moving number into the day it moved.
_EIA_RELEASE_LAG_DAYS = 7


def _load_eia_series(series_id: str, *, api_key: str, cache_dir: Path) -> pd.DataFrame:
    """Fetch one EIA v2 series as ``(timestamp, value, released_at)``.

    ``released_at`` is the period end plus :data:`_EIA_RELEASE_LAG_DAYS`, which
    is what makes this leak-safe: the value only becomes visible to a backtest
    on the day it was actually published, not on the day the week ended.
    """
    import json  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"eia_{series_id.replace('.', '_')}.csv"
    if cache_path.exists():
        frame = pd.read_csv(cache_path, parse_dates=["timestamp", "released_at"])
    else:
        url = f"https://api.eia.gov/v2/seriesid/{series_id}?api_key={api_key}"
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
            payload = json.loads(response.read())
        rows = payload.get("response", {}).get("data", [])
        if not rows:
            raise RuntimeError(f"EIA returned no data for {series_id}")
        value_key = next(k for k in rows[0] if k in {"value", "Value"})
        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime([r["period"] for r in rows]),
                "value": pd.to_numeric([r[value_key] for r in rows], errors="coerce"),
            }
        ).dropna(subset=["value"])
        frame["released_at"] = frame["timestamp"] + pd.Timedelta(days=_EIA_RELEASE_LAG_DAYS)
        frame = frame.sort_values("timestamp").reset_index(drop=True)
        frame.to_csv(cache_path, index=False)
    return frame


def build_wti_multivariate_service(
    cache_dir: Path | None = None,
    *,
    covariate_series_ids: list[str] | None = None,
    strict_covariates: bool = False,
    start: str = _WTI_HISTORY_START,
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

    Non-covariate predictors simply ignore the extra series, so a single service
    can feed an entire leaderboard.

    Parameters
    ----------
    cache_dir : Path or None
        yfinance CSV cache directory (shared with the target).  Defaults to
        :data:`DEFAULT_CACHE_DIR`.
    covariate_series_ids : list[str] or None
        Subset of :data:`DEFAULT_WTI_COVARIATE_SERIES_IDS` to register.  ``None``
        registers the full default panel.
    strict_covariates : bool
        If ``True``, any covariate fetch/build failure raises.  If ``False``
        (default), unavailable covariates are skipped with a warning so the
        service still builds offline / under partial connectivity.
    start : str
        Earliest date requested from Yahoo Finance for the covariates.
    """
    resolved_cache_dir: Path = cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR
    svc = build_wti_service(cache_dir=resolved_cache_dir)

    desired = set(covariate_series_ids if covariate_series_ids is not None else DEFAULT_WTI_COVARIATE_SERIES_IDS)

    def _handle_error(series_id: str, exc: Exception) -> None:
        if strict_covariates:
            raise RuntimeError(f"Failed to build required covariate {series_id!r}.") from exc
        warnings.warn(f"Skipping unavailable covariate {series_id!r}: {exc}", stacklevel=2)

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

    # ── OVX level ─────────────────────────────────────────────────────────────
    # Oil-specific implied volatility. VIX above is equity vol; OVX prices the
    # crude options market directly, so it carries the risk premium that VIX
    # only proxies for.
    if SERIES_ID_OVX_LEVEL in desired:
        try:
            ovx_close = _load_yahoo_close_frame(_OVX_TICKER, cache_dir=resolved_cache_dir, start=start)
            ovx_level = apply_one_business_day_feature_lag(to_level_feature_from_daily(ovx_close))
            svc.register(
                SERIES_ID_OVX_LEVEL,
                StaticFrameAdapter(ovx_level),
                SeriesMetadata(
                    series_id=SERIES_ID_OVX_LEVEL,
                    description="CBOE Crude Oil ETF Volatility Index (OVX) close level, lagged 1 business day",
                    source=f"Yahoo Finance ({_OVX_TICKER})",
                    units="index-level",
                    frequency="B",
                    table_id="yahoo:^OVX:close-l1b",
                ),
            )
        except (RuntimeError, ValueError, KeyError) as exc:
            _handle_error(SERIES_ID_OVX_LEVEL, exc)

    # ── 3-2-1 crack spread ────────────────────────────────────────────────────
    # Refining margin: (2 x gasoline + 1 x distillate - 3 x crude) / 3, the
    # standard proxy for what a refiner earns per barrel run. It is a *demand*
    # signal for crude, and a level the market mean-reverts around — which is
    # the kind of variable a long-run cointegrating relation is for.
    #
    # RB and HO quote in $/gallon while CL quotes in $/bbl, so the product legs
    # are scaled by 42 before differencing. Getting this wrong would make the
    # spread two orders of magnitude too small and it would simply be dropped
    # by the L1 penalty, silently.
    if SERIES_ID_CRACK_SPREAD in desired:
        try:
            gasoline = _load_yahoo_close_frame(_GASOLINE_TICKER, cache_dir=resolved_cache_dir, start=start)
            distillate = _load_yahoo_close_frame(_HEATING_OIL_TICKER, cache_dir=resolved_cache_dir, start=start)
            crude = _load_yahoo_close_frame(_WTI_FUTURES_TICKER, cache_dir=resolved_cache_dir, start=start)
            merged = gasoline[["timestamp", "value"]].rename(columns={"value": "rb"})
            merged = merged.merge(
                distillate[["timestamp", "value"]].rename(columns={"value": "ho"}), on="timestamp", how="inner"
            )
            merged = merged.merge(
                crude[["timestamp", "value"]].rename(columns={"value": "cl"}), on="timestamp", how="inner"
            )
            merged["value"] = (
                2.0 * merged["rb"] * _GALLONS_PER_BARREL + merged["ho"] * _GALLONS_PER_BARREL - 3.0 * merged["cl"]
            ) / 3.0
            crack = apply_one_business_day_feature_lag(
                to_level_feature_from_daily(merged[["timestamp", "value"]])
            )
            svc.register(
                SERIES_ID_CRACK_SPREAD,
                StaticFrameAdapter(crack),
                SeriesMetadata(
                    series_id=SERIES_ID_CRACK_SPREAD,
                    description=(
                        "3-2-1 crack spread (2xRB + HO - 3xCL)/3 in USD/bbl, lagged 1 business day"
                    ),
                    source=f"Yahoo Finance ({_GASOLINE_TICKER}, {_HEATING_OIL_TICKER}, {_WTI_FUTURES_TICKER})",
                    units="USD/bbl",
                    frequency="B",
                    table_id="yahoo:RB-HO-CL:crack321-l1b",
                ),
            )
        except (RuntimeError, ValueError, KeyError) as exc:
            _handle_error(SERIES_ID_CRACK_SPREAD, exc)

    # ── 10-year Treasury yield ────────────────────────────────────────────────
    # The discount-rate channel: the cost of carrying inventory, and the macro
    # variable the FRED panel was reaching for. Unlike INDPRO or the stress
    # index it is a market price, so it is never revised — the value quoted on
    # a given day is final, and a backtest reading it cannot see the future.
    if SERIES_ID_UST10Y_LEVEL in desired:
        try:
            ust10y_close = _load_yahoo_close_frame(_UST10Y_TICKER, cache_dir=resolved_cache_dir, start=start)
            ust10y_level = apply_one_business_day_feature_lag(to_level_feature_from_daily(ust10y_close))
            svc.register(
                SERIES_ID_UST10Y_LEVEL,
                StaticFrameAdapter(ust10y_level),
                SeriesMetadata(
                    series_id=SERIES_ID_UST10Y_LEVEL,
                    description="CBOE 10-Year Treasury Note Yield Index close level, lagged 1 business day",
                    source=f"Yahoo Finance ({_UST10Y_TICKER})",
                    units="index-level",
                    frequency="B",
                    table_id="yahoo:^TNX:close-l1b",
                ),
            )
        except (RuntimeError, ValueError, KeyError) as exc:
            _handle_error(SERIES_ID_UST10Y_LEVEL, exc)

    # ── Brent price level ─────────────────────────────────────────────────────
    # The cointegration anchor, and the reason this series exists separately
    # from brent_log_ret_1b_l1b above.
    #
    # An error-correction model needs covariates that are I(1) and share a
    # stochastic trend with the target. A panel of log RETURNS cannot provide
    # one: returns are stationary, and a stationary series cannot cointegrate
    # with a non-stationary price. Measured on the 2014-2024 grid, the
    # seven-series panel produced a zeroed error-correction coefficient on
    # 90.4% of origins and a median Engle-Granger p of 0.659 -- there was no
    # long-run relation to correct toward, so "ECM" was a differenced
    # regression wearing the name.
    #
    # WTI and Brent are the textbook cointegrating pair: the same commodity at
    # different delivery points, with a spread that mean-reverts around
    # transport and quality differentials. Brent's LEVEL gives the long-run
    # relation a real anchor; its return, already in the panel, drives the
    # short-run equation. Keeping both is deliberate and not redundant --
    # see LEVEL_VALUED_WTI_COVARIATE_SERIES_IDS, which routes the level to the
    # long-run step only.
    if SERIES_ID_BRENT_LEVEL in desired:
        try:
            brent_close = _load_yahoo_close_frame(_BRENT_TICKER, cache_dir=resolved_cache_dir, start=start)
            brent_level = apply_one_business_day_feature_lag(to_level_feature_from_daily(brent_close))
            svc.register(
                SERIES_ID_BRENT_LEVEL,
                StaticFrameAdapter(brent_level),
                SeriesMetadata(
                    series_id=SERIES_ID_BRENT_LEVEL,
                    description="Brent crude (BZ=F) close price level, lagged 1 business day",
                    source=f"Yahoo Finance ({_BRENT_TICKER})",
                    units="USD/bbl",
                    frequency="B",
                    table_id="yahoo:BZ=F:close-l1b",
                ),
            )
        except (RuntimeError, ValueError, KeyError) as exc:
            _handle_error(SERIES_ID_BRENT_LEVEL, exc)

    # ── Copper, S&P 500 and dollar-index LEVELS ───────────────────────────────
    # Copper is the daily stand-in for Kilian's (2009) aggregate-demand channel:
    # his own index of global real activity is monthly shipping rates, and
    # copper is the highest-frequency proxy for industrial demand that is also
    # genuinely outside the energy complex -- unlike Brent, which is nearly the
    # same asset. The S&P 500 carries aggregate demand and risk appetite; the
    # dollar index level is the cointegration-relevant form of the DXY return
    # already in the panel (Akram 2009; Chen, Rogoff & Rossi 2010).
    #
    # Levels, not returns, on purpose: only an I(1) series can cointegrate with
    # a price. Returns are stationary, which is exactly why the original panel
    # had no long-run relation to correct toward.
    for series_id, ticker, description in (
        (SERIES_ID_COPPER_LEVEL, _COPPER_TICKER, "COMEX copper (HG=F) close level, lagged 1 business day"),
        (SERIES_ID_SP500_LEVEL, _SP500_TICKER, "S&P 500 (^GSPC) close level, lagged 1 business day"),
        (
            SERIES_ID_DOLLAR_INDEX_LEVEL,
            _DOLLAR_INDEX_TICKER,
            "US Dollar Index (DX-Y.NYB) close level, lagged 1 business day",
        ),
    ):
        if series_id not in desired:
            continue
        try:
            close = _load_yahoo_close_frame(ticker, cache_dir=resolved_cache_dir, start=start)
            feature = apply_one_business_day_feature_lag(to_level_feature_from_daily(close))
            svc.register(
                series_id,
                StaticFrameAdapter(feature),
                SeriesMetadata(
                    series_id=series_id,
                    description=description,
                    source=f"Yahoo Finance ({ticker})",
                    units="index-level",
                    frequency="B",
                    table_id=f"yahoo:{ticker}:close-l1b",
                ),
            )
        except (RuntimeError, ValueError, KeyError) as exc:
            _handle_error(series_id, exc)

    # ── EIA weekly fundamentals ───────────────────────────────────────────────
    # Crude inventories are the single most-cited fundamental in the oil
    # forecasting literature (Kilian & Murphy 2014): above-ground stocks are the
    # state variable linking flow supply and demand to price, and the reason a
    # pure flow model cannot explain the level. Refinery utilization is the
    # demand-side companion.
    #
    # Requires an EIA API key in EIA_API_KEY; without one both series are simply
    # skipped, exactly as an unavailable Yahoo ticker is, so the panel still
    # builds on a machine that has no key.
    #
    # These are weekly and expanded from their RELEASE dates, so a backtest sees
    # a figure on the day it was published rather than the day the week ended.
    # Their daily differences are zero on every day but release day, which makes
    # them the textbook case for the long-run-only block -- the short-run
    # equation cannot tell a genuine zero from a forward-filled one.
    eia_key = os.environ.get("EIA_API_KEY")
    for series_id, (eia_id, description) in _EIA_SERIES.items():
        if series_id not in desired:
            continue
        if not eia_key:
            _handle_error(series_id, RuntimeError("EIA_API_KEY is not set; skipping EIA series"))
            continue
        try:
            raw = _load_eia_series(eia_id, api_key=eia_key, cache_dir=resolved_cache_dir)
            expanded = business_daily_expand_from_releases(raw, start=start, end=None)
            svc.register(
                series_id,
                StaticFrameAdapter(expanded),
                SeriesMetadata(
                    series_id=series_id,
                    description=f"{description}, visible from its release date",
                    source=f"EIA API v2 ({eia_id})",
                    units="thousand-barrels" if "STOCKS" in eia_id.upper() or "WCEST" in eia_id else "percent",
                    frequency="B",
                    table_id=f"eia:{eia_id}:release-expanded",
                ),
            )
        except (RuntimeError, ValueError, KeyError, OSError) as exc:
            _handle_error(series_id, exc)

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

    return svc


__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_WTI_COVARIATE_SERIES_IDS",
    "SERIES_ID_BRENT_RETURN",
    "SERIES_ID_DOLLAR_INDEX_RETURN",
    "SERIES_ID_GASOLINE_RETURN",
    "SERIES_ID_GOLD_RETURN",
    "SERIES_ID_NATGAS_RETURN",
    "SERIES_ID_OIL_CURVE_CONTANGO",
    "SERIES_ID_VIX_LEVEL",
    "WTI_SERIES_ID",
    "build_wti_multivariate_service",
    "build_wti_service",
    "naive_utc_now",
]
