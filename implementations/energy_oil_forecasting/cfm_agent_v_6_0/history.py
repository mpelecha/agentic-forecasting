"""Cutoff-safe price history and data preparation (specification Sections 1.2, 1.5, 3.4).

:class:`PriceHistory` reads the target and covariate series from a
:class:`~aieng.forecasting.data.context.ForecastContext`, applies the input
floor, the optional history window, computes daily simple returns, aligns the
covariates to the WTI observation dates, and computes the data fingerprint and
the per-horizon :class:`~energy_oil_forecasting.cfm_agent_v_6_0.schemas.HistoricalReturnProfile`.

Conventions
-----------
- Returns are ``P(t) / P(t-1) - 1`` on floored prices; ``returns[k]`` is the
  return realised on ``dates[k + 1]``. Returns are never clipped or demeaned.
- The input-floor activation count and dates are recorded on the windowed
  prices, which are the values every component and diagnostic consumes.
- Covariates are inner-joined on the WTI dates of the window, forward-filled up
  to ``covariate_forward_fill_limit`` consecutive rows, and rows still missing a
  value are dropped. The data fingerprint is computed before fill and drop.
- Forecast dates for horizon ``h`` are ``cutoff + h`` pandas business days (DD-19).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from aieng.forecasting.data import ForecastContext
from energy_oil_forecasting.cfm_agent_v_6_0.config import CfmV60Settings
from energy_oil_forecasting.cfm_agent_v_6_0.fingerprints import identity
from energy_oil_forecasting.cfm_agent_v_6_0.schemas import (
    HistoricalReturnProfile,
    HistoryWindow,
)


def _iso_date(value: Any) -> str:
    return str(pd.Timestamp(value).date())


def forecast_date_for(cutoff: Any, horizon: int) -> datetime:
    """Return ``cutoff + horizon`` pandas business days as a naive datetime (DD-19)."""
    return (pd.Timestamp(cutoff).normalize() + pd.offsets.BDay(int(horizon))).to_pydatetime()


def _served_frame(context: ForecastContext, series_id: str) -> pd.DataFrame:
    frame = context.get_series(series_id)
    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame["timestamp"]),
            "value": pd.to_numeric(frame["value"], errors="coerce").astype(float),
        }
    )
    return out.sort_values("timestamp").reset_index(drop=True)


@dataclass(frozen=True)
class CovariateRecord:
    """One covariate as served by the data layer, restricted to the history window."""

    series_id: str
    missing: bool
    served: pd.DataFrame | None = field(repr=False, default=None)
    observations: int = 0
    first_date: str | None = None
    last_date: str | None = None

    @property
    def latest_value(self) -> float | None:
        """The latest cutoff-safe served value, or ``None`` when absent or empty."""
        if self.served is None or self.served.empty:
            return None
        values = self.served["value"].dropna()
        return float(values.iloc[-1]) if not values.empty else None


class PriceHistory:
    """Floored, windowed WTI prices with returns, aligned covariates, and identity data."""

    def __init__(
        self,
        *,
        target_id: str,
        cutoff: pd.Timestamp,
        settings: CfmV60Settings,
        dates: pd.DatetimeIndex,
        raw_prices: np.ndarray,
        total_target_observations: int,
        first_served_date: str | None,
        covariate_ids: Sequence[str],
        covariates: dict[str, CovariateRecord],
    ) -> None:
        self.target_id = target_id
        self.cutoff = pd.Timestamp(cutoff).normalize()
        self.settings = settings
        self.dates = dates
        self.total_target_observations = total_target_observations
        self.first_served_date = first_served_date
        self.covariate_ids = list(covariate_ids)
        self.covariates = covariates
        floor = settings.input_price_floor
        below = raw_prices < floor
        self.input_floor_activation_count = int(below.sum())
        self.input_floor_activation_dates = [_iso_date(value) for value in dates[below]]
        self.prices = np.maximum(raw_prices.astype(float), floor)
        self.returns = self.prices[1:] / self.prices[:-1] - 1 if len(self.prices) > 1 else np.empty(0)
        self.window = HistoryWindow(
            setting_business_days=settings.history_window_business_days,
            start_date=_iso_date(dates[0]) if len(dates) else None,
            end_date=_iso_date(dates[-1]) if len(dates) else None,
            rows=int(len(dates)),
        )
        self.missing_covariates = [series_id for series_id in self.covariate_ids if covariates[series_id].missing]
        self._align_covariates()
        self.data_fingerprint = identity(self.fingerprint_payload(), prefix="data")

    # ----------------------------------------------------------- building --

    @classmethod
    def from_context(
        cls,
        context: ForecastContext,
        target_id: str,
        covariate_ids: Iterable[str],
        settings: CfmV60Settings,
    ) -> PriceHistory:
        """Read the target and covariates as served at the context cutoff."""
        target = _served_frame(context, target_id)
        if target.empty:
            raise ValueError(f"target series {target_id!r} has no cutoff-safe observations")
        total = len(target)
        first_served = _iso_date(target["timestamp"].iloc[0])
        window = settings.history_window_business_days
        if window > 0:
            target = target.tail(window).reset_index(drop=True)
        dates = pd.DatetimeIndex(target["timestamp"])
        raw_prices = target["value"].to_numpy(dtype=float)
        covariates: dict[str, CovariateRecord] = {}
        ids = list(covariate_ids)
        for series_id in ids:
            try:
                served = _served_frame(context, series_id)
            except KeyError:
                covariates[series_id] = CovariateRecord(series_id=series_id, missing=True)
                continue
            served = served[(served["timestamp"] >= dates[0]) & (served["timestamp"] <= dates[-1])].reset_index(
                drop=True
            )
            covariates[series_id] = CovariateRecord(
                series_id=series_id,
                missing=False,
                served=served,
                observations=int(len(served)),
                first_date=_iso_date(served["timestamp"].iloc[0]) if not served.empty else None,
                last_date=_iso_date(served["timestamp"].iloc[-1]) if not served.empty else None,
            )
        return cls(
            target_id=target_id,
            cutoff=pd.Timestamp(context.as_of),
            settings=settings,
            dates=dates,
            raw_prices=raw_prices,
            total_target_observations=total,
            first_served_date=first_served,
            covariate_ids=ids,
            covariates=covariates,
        )

    def _align_covariates(self) -> None:
        """Inner-join the present covariates on the WTI dates, forward-fill, and drop."""
        present = [series_id for series_id in self.covariate_ids if not self.covariates[series_id].missing]
        frame = pd.DataFrame(index=self.dates)
        for series_id in present:
            served = self.covariates[series_id].served
            assert served is not None
            column = served.drop_duplicates("timestamp", keep="last").set_index("timestamp")["value"]
            frame[series_id] = column.reindex(self.dates)
        self.rows_before_alignment = int(len(frame))
        limit = self.settings.covariate_forward_fill_limit
        fill_counts: dict[str, int] = {}
        for series_id in present:
            before = frame[series_id].isna().sum()
            frame[series_id] = frame[series_id].ffill(limit=limit) if limit > 0 else frame[series_id]
            fill_counts[series_id] = int(before - frame[series_id].isna().sum())
        self.covariate_fill_counts = fill_counts
        aligned = frame.dropna() if present else frame
        self.rows_dropped = int(len(frame) - len(aligned))
        self.rows_remaining = int(len(aligned))
        self.aligned = aligned

    # ------------------------------------------------------------ identity --

    def fingerprint_payload(self) -> dict[str, Any]:
        """Return the data-fingerprint payload (Section 1.7): floored target, covariates as served."""
        series: dict[str, list[list[Any]] | None] = {
            self.target_id: [[_iso_date(date), float(value)] for date, value in zip(self.dates, self.prices)]
        }
        for series_id in self.covariate_ids:
            record = self.covariates[series_id]
            if record.missing or record.served is None:
                series[series_id] = None
                continue
            series[series_id] = [
                [_iso_date(row.timestamp), None if pd.isna(row.value) else float(row.value)]
                for row in record.served.itertuples(index=False)
            ]
        return {
            "cutoff": _iso_date(self.cutoff),
            "window": self.settings.history_window_business_days,
            "series": series,
        }

    # ------------------------------------------------------------- returns --

    @property
    def last_price(self) -> float:
        """The latest cutoff-safe floored price."""
        return float(self.prices[-1])

    @property
    def first_date(self) -> str | None:
        """First date of the window."""
        return self.window.start_date

    @property
    def last_date(self) -> str | None:
        """Last date of the window (the latest cutoff-safe observation)."""
        return self.window.end_date

    def forecast_date(self, horizon: int) -> datetime:
        """Forecast date for ``horizon`` steps: cutoff plus ``horizon`` business days."""
        return forecast_date_for(self.cutoff, horizon)

    def horizon_returns(self, horizon: int) -> np.ndarray:
        """All overlapping ``P(t+h) / P(t) - 1`` on the windowed floored prices."""
        h = int(horizon)
        if h < 1:
            raise ValueError("horizon must be positive")
        if len(self.prices) <= h:
            return np.empty(0)
        return self.prices[h:] / self.prices[:-h] - 1

    def covariate_latest_values(self) -> dict[str, float]:
        """Latest served value per present covariate, in configured order."""
        values: dict[str, float] = {}
        for series_id in self.covariate_ids:
            latest = self.covariates[series_id].latest_value
            if latest is not None:
                values[series_id] = latest
        return values

    # ------------------------------------------------------------- profile --

    def _conditioned_returns(self, horizon: int) -> tuple[np.ndarray, int | None]:
        """Return the horizon returns, filtered to the current volatility bin when conditioning is on."""
        returns = self.horizon_returns(horizon)
        if self.settings.historical_conditioning == "none":
            return returns, None
        window = self.settings.conditioning_volatility_window
        bins = self.settings.conditioning_quantile_bins
        n_starts = len(returns)
        # The h-step return starting at date index i needs daily returns[i-window:i] for its
        # trailing realised volatility, so the first eligible start is i = window.
        if n_starts <= window or len(self.returns) < window:
            return np.empty(0), None
        daily = pd.Series(self.returns)
        rolling = daily.rolling(window).std(ddof=0).to_numpy()
        # rolling[k] is the volatility of returns[k-window+1:k+1], i.e. ending on date k+1;
        # the volatility known at start date i (ending on date i) is rolling[i-1].
        start_vols = rolling[window - 1 : n_starts - 1]
        eligible_returns = returns[window:]
        current_vol = float(np.std(self.returns[-window:], ddof=0))
        edges = np.quantile(start_vols, np.linspace(0, 1, bins + 1))
        interior = edges[1:-1]
        start_bins = np.digitize(start_vols, interior, right=True)
        current_bin = int(np.digitize(np.array([current_vol]), interior, right=True)[0])
        return eligible_returns[start_bins == current_bin], current_bin

    def profile(self, horizon: int) -> HistoricalReturnProfile:
        """Compute the :class:`HistoricalReturnProfile` for ``horizon`` (Section 3.4)."""
        settings = self.settings
        returns, conditioning_bin = self._conditioned_returns(horizon)
        n_eligible = int(len(returns))
        min_required = settings.min_historical_returns
        eligible = n_eligible >= min_required
        percentiles = sorted(set(settings.center_percentiles.values()))
        coverages = sorted(set(settings.uncertainty_coverage.values()) | {settings.uncertainty_anchor_coverage})
        base = {
            "horizon": int(horizon),
            "n_eligible": n_eligible,
            "min_required": min_required,
            "eligible": eligible,
            "window_start": self.window.start_date,
            "window_end": self.window.end_date,
            "conditioning": settings.historical_conditioning,
            "conditioning_bin": conditioning_bin,
        }
        if n_eligible == 0:
            return HistoricalReturnProfile(
                **base,
                ineligibility_reason=f"no eligible {horizon}-step historical returns in the window",
            )
        median = float(np.median(returns))
        deviations = np.abs(returns - median)
        return HistoricalReturnProfile(
            **base,
            median=median,
            mean=float(np.mean(returns)),
            return_percentiles={int(p): float(np.percentile(returns, p)) for p in percentiles},
            absolute_deviation_bands={int(c): float(np.percentile(deviations, c)) for c in coverages},
            ineligibility_reason=(
                None
                if eligible
                else f"{n_eligible} eligible {horizon}-step historical returns, below the minimum {min_required}"
            ),
        )

    def profiles(self, horizons: Iterable[int]) -> dict[int, HistoricalReturnProfile]:
        """Profiles keyed by horizon."""
        return {int(h): self.profile(int(h)) for h in horizons}


__all__ = ["CovariateRecord", "PriceHistory", "forecast_date_for"]
