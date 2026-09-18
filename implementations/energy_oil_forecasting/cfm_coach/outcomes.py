"""Join forecast horizons to what WTI actually did.

Scoring cannot begin until a horizon *resolves*, and at daily cadence with 5/10/21
business-day horizons most of the corpus is unresolved most of the time. So the
resolver's real job is not the lookup -- it is refusing to resolve things that have
not happened yet, and saying clearly which those are.

Two failure modes this is written against, both of which score a forecast against a
price that cannot judge it:

1. **The horizon has not elapsed.** Obvious, and guarded by requiring the series to
   extend to or past the target date before any lookup happens.
2. **The cache is stale.** Less obvious and more dangerous: if the parquet stopped
   updating, "the last observation on or before the target date" silently returns a
   price from before the forecast was even issued, and the forecast scores well for
   the wrong reason. Guarded by ``max_staleness_days``.

Non-trading target dates are legitimate and common -- v5.0 forecasts a business-day
offset, which can land on a holiday -- so resolution falls back to the last
observation on or before the target and records *which* observation was used. The
gap is visible in the `ResolvedForecast` rather than hidden inside the join.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import pandas as pd
from aieng.forecasting.data import DataService
from energy_oil_forecasting.cfm_coach.config import DEFAULT_SETTINGS, CoachSettings
from energy_oil_forecasting.cfm_coach.schemas import ResolvedForecast, RunRecord
from energy_oil_forecasting.data import build_wti_multivariate_service


#: How far back a resolution may reach for a target date that is not a trading day.
#: Comfortably covers a holiday weekend (Christmas/New Year can close 3-4 days) and
#: comfortably short of the 5-business-day horizon it would otherwise corrupt.
MAX_STALENESS_DAYS = 5

_ALL_HISTORY = datetime(2100, 1, 1)


@dataclass(frozen=True)
class PendingForecast:
    """A horizon that cannot be scored yet, and why."""

    run_id: str
    horizon: int
    cutoff: date
    forecast_date: datetime
    reason: str
    #: Business days still to elapse, when that is the reason. Negative never occurs.
    days_remaining: int | None = None

    def describe(self) -> str:
        return f"{self.run_id} h={self.horizon} due {self.forecast_date.date()}: {self.reason}"


@dataclass(frozen=True)
class ResolutionReport:
    """The outcome of resolving a corpus: what scored, what is waiting, what is broken."""

    resolved: list[ResolvedForecast]
    pending: list[PendingForecast]
    #: Latest observation in the target series -- the frontier everything resolves against.
    data_through: date | None

    @property
    def resolved_by_run(self) -> dict[str, list[ResolvedForecast]]:
        grouped: dict[str, list[ResolvedForecast]] = {}
        for item in self.resolved:
            grouped.setdefault(item.run_id, []).append(item)
        return grouped

    def describe(self) -> str:
        lines = [
            f"resolved {len(self.resolved)} horizon(s), {len(self.pending)} pending"
            f" (price data through {self.data_through})"
        ]
        lines.extend(f"  pending: {item.describe()}" for item in self.pending)
        return "\n".join(lines)


class OutcomeResolver:
    """Pairs stored run horizons with the realized WTI level at their target date."""

    resolver_id = "cfm_coach_outcome_resolver_v1"

    def __init__(
        self,
        settings: CoachSettings = DEFAULT_SETTINGS,
        *,
        service: DataService | None = None,
        max_staleness_days: int = MAX_STALENESS_DAYS,
    ):
        self.s = settings
        self._service = service
        self._series: pd.Series | None = None
        self.max_staleness_days = max_staleness_days

    def _load_series(self) -> pd.Series:
        """Realized target levels indexed by observation date, ascending.

        Read through the same `DataService` the agent forecasts with, so the value a
        forecast is scored against is the value the agent would have been shown --
        rather than a second, subtly different read of the same parquet.
        """
        if self._series is None:
            if self._service is None:
                # Built lazily rather than in __init__, so a caller that injects a
                # service (or only reads `resolver.s`) never pays to load the parquet.
                self._service = build_wti_multivariate_service()
            frame = self._service.get_series(self.s.target_series_id, as_of=_ALL_HISTORY)
            values = pd.to_numeric(frame["value"], errors="coerce")
            index = pd.to_datetime(frame["timestamp"]).dt.normalize().dt.date
            series = pd.Series(values.to_numpy(), index=index).dropna()
            self._series = series[~series.index.duplicated(keep="last")].sort_index()
        return self._series

    def price_series(self) -> pd.Series:
        """Return the realized target series this resolver scores against, indexed by observation date.

        Exposed so a baseline forecast is built from exactly the prices its score is
        judged on, rather than from a second read of the same parquet.
        """
        return self._load_series().copy()

    @property
    def data_through(self) -> date | None:
        series = self._load_series()
        return None if series.empty else series.index[-1]

    def resolve_horizon(self, record: RunRecord, horizon: int) -> ResolvedForecast | PendingForecast:
        """Resolve one horizon, or explain why it cannot be."""
        horizon_record = record.horizon(horizon)
        target = horizon_record.forecast_date.date()
        series = self._load_series()

        if series.empty:
            return PendingForecast(
                run_id=record.run_id,
                horizon=horizon,
                cutoff=record.cutoff,
                forecast_date=horizon_record.forecast_date,
                reason=f"no observations in series {self.s.target_series_id!r}",
            )

        frontier = series.index[-1]
        if frontier < target:
            return PendingForecast(
                run_id=record.run_id,
                horizon=horizon,
                cutoff=record.cutoff,
                forecast_date=horizon_record.forecast_date,
                reason=f"target date not reached; price data ends {frontier}",
                days_remaining=len(pd.bdate_range(frontier + timedelta(days=1), target)),
            )

        on_or_before = series[series.index <= target]
        if on_or_before.empty:
            return PendingForecast(
                run_id=record.run_id,
                horizon=horizon,
                cutoff=record.cutoff,
                forecast_date=horizon_record.forecast_date,
                reason=f"no observation on or before {target}",
            )

        observation_date = on_or_before.index[-1]
        staleness = (target - observation_date).days
        if staleness > self.max_staleness_days:
            return PendingForecast(
                run_id=record.run_id,
                horizon=horizon,
                cutoff=record.cutoff,
                forecast_date=horizon_record.forecast_date,
                reason=(
                    f"nearest observation {observation_date} is {staleness} days before target {target}"
                    f" (limit {self.max_staleness_days}); refusing to score against a stale price"
                ),
            )

        return ResolvedForecast(
            run_id=record.run_id,
            horizon=horizon,
            cutoff=record.cutoff,
            forecast_date=horizon_record.forecast_date,
            realized_value=float(on_or_before.iloc[-1]),
            realized_observation_date=observation_date,
            resolved_at=datetime.now(),
        )

    def resolve(self, record: RunRecord) -> ResolutionReport:
        resolved: list[ResolvedForecast] = []
        pending: list[PendingForecast] = []
        for horizon in record.horizons:
            outcome = self.resolve_horizon(record, horizon)
            (resolved if isinstance(outcome, ResolvedForecast) else pending).append(outcome)
        return ResolutionReport(resolved=resolved, pending=pending, data_through=self.data_through)

    def resolve_all(self, records: list[RunRecord]) -> ResolutionReport:
        resolved: list[ResolvedForecast] = []
        pending: list[PendingForecast] = []
        for record in records:
            report = self.resolve(record)
            resolved.extend(report.resolved)
            pending.extend(report.pending)
        return ResolutionReport(resolved=resolved, pending=pending, data_through=self.data_through)

    def to_frame(self, resolved: list[ResolvedForecast]) -> pd.DataFrame:
        """Tidy table of resolved outcomes, for reports and eyeballing."""
        return pd.DataFrame(
            [
                {
                    "run_id": item.run_id,
                    "cutoff": item.cutoff,
                    "horizon": item.horizon,
                    "forecast_date": item.forecast_date.date(),
                    "observation_date": item.realized_observation_date,
                    "realized": item.realized_value,
                }
                for item in resolved
            ]
        )


__all__ = [
    "MAX_STALENESS_DAYS",
    "OutcomeResolver",
    "PendingForecast",
    "ResolutionReport",
]
