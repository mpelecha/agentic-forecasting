"""Shared v3 fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from aieng.forecasting.data import DataService, SeriesMetadata, StaticFrameAdapter
from energy_oil_forecasting.data import WTI_SERIES_ID


@pytest.fixture
def synthetic_service() -> DataService:
    dates = pd.bdate_range("2020-01-01", periods=500)
    x = np.arange(len(dates), dtype=float)
    service = DataService()
    for series_id, values, description in [
        (WTI_SERIES_ID, 55.0 + 0.02 * x + np.sin(x / 20.0), "Synthetic WTI"),
        ("vix_level_l1b", 20.0 + 2.0 * np.cos(x / 15.0), "Synthetic VIX"),
    ]:
        frame = pd.DataFrame(
            {
                "timestamp": dates,
                "value": values,
                "released_at": dates + pd.offsets.BDay(1),
            }
        )
        service.register(
            series_id,
            StaticFrameAdapter(frame),
            SeriesMetadata(
                series_id=series_id,
                description=description,
                source="test",
                units="index" if "vix" in series_id else "USD/bbl",
                frequency="B",
            ),
        )
    return service
