"""Shared fixtures for CFM Agent v2.1 tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from aieng.forecasting.data import DataService, SeriesMetadata, StaticFrameAdapter
from energy_oil_forecasting.data import WTI_SERIES_ID


@pytest.fixture
def synthetic_service() -> DataService:
    """Return a cutoff-aware service with a synthetic WTI-like series."""
    dates = pd.bdate_range("2020-01-01", periods=800)
    x = np.arange(len(dates), dtype=float)
    service = DataService()
    wti = pd.DataFrame(
        {
            "timestamp": dates,
            "value": 55.0 + 0.02 * x + np.sin(x / 20.0),
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
    return service
