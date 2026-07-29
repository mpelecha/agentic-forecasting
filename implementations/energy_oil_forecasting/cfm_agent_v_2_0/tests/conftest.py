"""Shared fixtures for CFM agent tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from aieng.forecasting.data import DataService, SeriesMetadata, StaticFrameAdapter
from energy_oil_forecasting.data import WTI_SERIES_ID


@pytest.fixture
def synthetic_service() -> DataService:
    """Return a cutoff-aware service with WTI and VIX-like data."""
    dates = pd.bdate_range("2020-01-01", periods=500)
    x = np.arange(len(dates), dtype=float)
    service = DataService()
    wti = pd.DataFrame(
        {
            "timestamp": dates,
            "value": 55.0 + 0.02 * x + np.sin(x / 20.0),
            "released_at": dates + pd.offsets.BDay(1),
        }
    )
    vix = pd.DataFrame(
        {
            "timestamp": dates,
            "value": 20.0 + 2.0 * np.cos(x / 15.0),
            "released_at": dates + pd.offsets.BDay(1),
        }
    )
    service.register(
        WTI_SERIES_ID,
        StaticFrameAdapter(wti),
        SeriesMetadata(
            series_id=WTI_SERIES_ID,
            description="Synthetic WTI",
            source="test",
            units="USD/bbl",
            frequency="B",
        ),
    )
    service.register(
        "vix_level_l1b",
        StaticFrameAdapter(vix),
        SeriesMetadata(
            series_id="vix_level_l1b",
            description="Synthetic VIX",
            source="test",
            units="index",
            frequency="B",
        ),
    )
    return service
