"""Shared offline fixtures: a research packet with passages and a synthetic data service."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.features import StaticFrameAdapter
from aieng.forecasting.evaluation.task import ForecastingTask
from energy_oil_forecasting.cfm_agent_v_6_0.config import DEFAULT_SETTINGS
from energy_oil_forecasting.cfm_agent_v_6_0.schemas import (
    ActiveSource,
    ResearchPacket,
    ResearchQuery,
    SummaryPassage,
    VerifiedSummary,
)
from energy_oil_forecasting.data import WTI_SERIES_ID


SYNTHETIC_CUTOFF = "2026-09-14"
SYNTHETIC_OBSERVATIONS = 3_000
SYNTHETIC_SEED = 20_260_914


@pytest.fixture
def packet():
    sources = [
        ActiveSource(
            source_id="source_001",
            rank=1,
            grounding_url="https://redirect.test/1",
            resolved_url="https://www.eia.gov/a",
            resolved_domain="eia.gov",
            publisher="U.S. Energy Information Administration",
            source_quality="tier_1_primary",
            is_primary_or_official=True,
            resolution_status="resolved",
        ),
        ActiveSource(
            source_id="source_002",
            rank=2,
            grounding_url="https://redirect.test/2",
            resolved_url="https://www.reuters.com/a",
            resolved_domain="reuters.com",
            publisher="Reuters",
            source_quality="tier_2_independent",
            resolution_status="resolved",
        ),
        ActiveSource(
            source_id="source_003",
            rank=3,
            grounding_url="https://redirect.test/3",
            resolved_url="https://www.bloomberg.com/a",
            resolved_domain="bloomberg.com",
            publisher="Bloomberg",
            source_quality="tier_2_independent",
            resolution_status="resolved",
        ),
    ]
    summary = VerifiedSummary(
        verified_summary_id="verified_summary:sha256:test",
        query_area="physical_flows",
        query="test",
        cutoff_date="2026-07-28",
        status="accepted",
        cleaned_summary=(
            "Confirmed physical oil supply information with market relevance. "
            "Exports from the terminal were suspended for the second week running."
        ),
        passages=[
            SummaryPassage(
                passage_id="01.01",
                text="Confirmed physical oil supply information with market relevance.",
            ),
            SummaryPassage(
                passage_id="01.02",
                text="Exports from the terminal were suspended for the second week running.",
            ),
        ],
        associated_source_ids=[source.source_id for source in sources],
    )
    return ResearchPacket(
        packet_id="research_packet:sha256:test",
        cutoff_date="2026-07-28",
        queries=[ResearchQuery(area="physical_flows", query="test")],
        verified_summaries=[summary],
        sources=sources,
    )


def _frame(dates: pd.DatetimeIndex, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"timestamp": dates, "value": values.astype(float), "released_at": dates})


def _build_synthetic_service(
    *,
    n_observations: int = SYNTHETIC_OBSERVATIONS,
    seed: int = SYNTHETIC_SEED,
    end: str = SYNTHETIC_CUTOFF,
    covariate_ids: list[str] | None = None,
    start_price: float = 60.0,
    daily_drift: float = 0.0002,
    daily_volatility: float = 0.02,
    covariate_gaps: dict[str, list[int]] | None = None,
) -> DataService:
    """Deterministic geometric-Brownian WTI plus seven synthetic covariates.

    ``covariate_ids`` defaults to the seven ids in ``settings.toml``; pass a
    subset to simulate a covariate absent from the data service. ``covariate_gaps``
    maps a covariate id to row positions removed from that series, to exercise the
    forward-fill and row-drop policy.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=end, periods=n_observations)
    wti_returns = rng.normal(daily_drift, daily_volatility, size=n_observations)
    prices = start_price * np.cumprod(1.0 + wti_returns)
    service = DataService()
    service.register(
        WTI_SERIES_ID,
        StaticFrameAdapter(_frame(dates, prices)),
        SeriesMetadata(
            series_id=WTI_SERIES_ID,
            description="Synthetic WTI front-month settle",
            source="synthetic",
            units="USD/bbl",
            frequency="B",
        ),
    )
    ids = list(DEFAULT_SETTINGS.covariate_series_ids) if covariate_ids is None else list(covariate_ids)
    lagged_wti = np.concatenate([[0.0], wti_returns[:-1]])
    contango = np.zeros(n_observations)
    vix = np.zeros(n_observations)
    contango_level, vix_level = 0.0, 18.0
    for index in range(n_observations):
        contango_level = 0.95 * contango_level + rng.normal(0.0, 0.004)
        vix_level = 18.0 + 0.9 * (vix_level - 18.0) + rng.normal(0.0, 1.2)
        contango[index] = contango_level
        vix[index] = max(9.0, vix_level)
    generators = {
        "oil_curve_contango_l1b": lambda: contango,
        "vix_level_l1b": lambda: vix,
    }
    for series_id in ids:
        if series_id in generators:
            values = generators[series_id]()
            units = "log-ratio" if "contango" in series_id else "index-level"
        else:
            values = 0.5 * lagged_wti + rng.normal(0.0, 0.01, size=n_observations)
            units = "log-return"
        frame = _frame(dates, values)
        gaps = (covariate_gaps or {}).get(series_id)
        if gaps:
            frame = frame.drop(index=gaps).reset_index(drop=True)
        service.register(
            series_id,
            StaticFrameAdapter(frame),
            SeriesMetadata(
                series_id=series_id,
                description=f"Synthetic covariate {series_id}",
                source="synthetic",
                units=units,
                frequency="B",
            ),
        )
    return service


@pytest.fixture
def synthetic_service_factory():
    """Callable returning a fresh synthetic :class:`DataService`; keyword arguments as in the builder."""
    return _build_synthetic_service


@pytest.fixture
def synthetic_service():
    return _build_synthetic_service()


@pytest.fixture
def synthetic_task():
    return ForecastingTask(
        task_id="synthetic",
        target_series_id=WTI_SERIES_ID,
        horizons=[5, 10, 21],
        frequency="B",
        description="synthetic offline test",
    )


@pytest.fixture
def synthetic_context(synthetic_service):
    return synthetic_service.context(as_of=datetime.fromisoformat(SYNTHETIC_CUTOFF))
