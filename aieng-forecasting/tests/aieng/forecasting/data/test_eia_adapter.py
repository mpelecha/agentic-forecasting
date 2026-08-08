"""Tests for :class:`EIAAdapter` request shape, release stamps and caching (no live network)."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from aieng.forecasting.data.adapters.eia import EIAAdapter


ROUTE = "petroleum/stoc/wstk"


def _payload(rows: list[dict[str, Any]], total: int | None = None) -> dict[str, Any]:
    """Wrap records in the EIA API v2 ``response.data`` / ``response.total`` envelope."""
    return {"response": {"total": len(rows) if total is None else total, "data": rows}}


def _rows(periods: list[str], values: list[float]) -> list[dict[str, Any]]:
    return [{"period": p, "value": v, "series": "WCESTUS1"} for p, v in zip(periods, values, strict=True)]


def _get_returning(*payloads: dict[str, Any]) -> MagicMock:
    """Build a ``requests.get`` replacement returning each payload in turn."""
    responses = []
    for payload in payloads:
        resp = MagicMock()
        resp.json.return_value = payload
        resp.raise_for_status.return_value = None
        responses.append(resp)
    return MagicMock(side_effect=responses)


def test_request_targets_v2_route_with_series_facet(tmp_path: Path) -> None:
    """The adapter hits ``/v2/<route>/data/`` with the documented v2 parameters."""
    fake_get = _get_returning(_payload(_rows(["2026-01-02"], [420000.0])))

    with patch("requests.get", fake_get):
        EIAAdapter("WCESTUS1", route=ROUTE, api_key="fake-key", cache_dir=tmp_path).fetch()

    url = fake_get.call_args.args[0]
    params = dict(fake_get.call_args.kwargs["params"])

    assert url == "https://api.eia.gov/v2/petroleum/stoc/wstk/data/"
    assert params["frequency"] == "weekly"
    assert params["data[]"] == "value"
    assert params["facets[series][]"] == "WCESTUS1"
    assert params["sort[0][column]"] == "period"
    assert params["length"] == "5000"


def test_released_at_is_wednesday_1030_et_not_the_period(tmp_path: Path) -> None:
    """Weekly rows are published the Wednesday after the week-ending Friday, 10:30 ET.

    January is EST (UTC-5), July is EDT (UTC-4), so the stored naive-UTC stamps
    differ by an hour — proving the conversion is DST-aware rather than a fixed
    offset, and that ``released_at`` is never just ``timestamp``.
    """
    fake_get = _get_returning(_payload(_rows(["2026-01-02", "2026-07-03"], [420000.0, 430000.0])))

    with patch("requests.get", fake_get):
        df = EIAAdapter("WCESTUS1", route=ROUTE, api_key="fake-key", cache_dir=tmp_path).fetch()

    assert list(df["released_at"]) == [
        pd.Timestamp("2026-01-07 15:30"),  # Wed 10:30 EST
        pd.Timestamp("2026-07-08 14:30"),  # Wed 10:30 EDT
    ]
    assert (df["released_at"] > df["timestamp"]).all()


def test_pagination_walks_offset_until_total_is_reached(tmp_path: Path) -> None:
    """A series longer than one page is assembled from successive offset requests."""
    first = _payload(_rows(["2026-01-02", "2026-01-09"], [1.0, 2.0]), total=3)
    second = _payload(_rows(["2026-01-16"], [3.0]), total=3)
    fake_get = _get_returning(first, second)

    with patch("requests.get", fake_get):
        df = EIAAdapter("WCESTUS1", route=ROUTE, api_key="fake-key", cache_dir=tmp_path).fetch()

    assert len(df) == 3
    offsets = [dict(call.kwargs["params"])["offset"] for call in fake_get.call_args_list]
    assert offsets == ["0", "2"]


def test_cache_round_trip_without_api_key(tmp_path: Path) -> None:
    """First fetch writes parquet; a new adapter reads it back with no API key."""
    fake_get = _get_returning(_payload(_rows(["2026-01-02"], [420000.0])))

    with patch("requests.get", fake_get):
        df1 = EIAAdapter("WCESTUS1", route=ROUTE, api_key="fake-key", cache_dir=tmp_path).fetch()

    assert (tmp_path / "WCESTUS1.parquet").exists()

    exploding = MagicMock(side_effect=AssertionError("requests.get must not be called"))
    with patch("requests.get", exploding), patch.dict("os.environ", {}, clear=True):
        df2 = EIAAdapter("WCESTUS1", route=ROUTE, api_key=None, cache_dir=tmp_path).fetch()

    pd.testing.assert_frame_equal(df1, df2)


def test_missing_api_key_without_cache_raises(tmp_path: Path) -> None:
    """No cache file AND no API key -> ValueError with a helpful message."""
    with patch.dict("os.environ", {}, clear=True):
        adapter = EIAAdapter("WCESTUS1", route=ROUTE, api_key=None, cache_dir=tmp_path / "empty")
        with pytest.raises(ValueError, match="EIA API key not provided"):
            adapter.fetch()


def test_empty_response_raises_runtime_error(tmp_path: Path) -> None:
    """An empty ``response.data`` is a configuration error, not a silent empty series."""
    fake_get = _get_returning(_payload([]))

    with patch("requests.get", fake_get):
        adapter = EIAAdapter("BOGUS", route=ROUTE, api_key="fake-key", cache_dir=tmp_path)
        with pytest.raises(RuntimeError, match="returned no data"):
            adapter.fetch()


def test_repr_does_not_leak_api_key() -> None:
    """``repr`` shows the series, route and cache dir but never the key."""
    text = repr(EIAAdapter("WCESTUS1", route=ROUTE, api_key="super-secret", cache_dir="data/eia"))
    assert "super-secret" not in text
    assert "WCESTUS1" in text
    assert ROUTE in text
