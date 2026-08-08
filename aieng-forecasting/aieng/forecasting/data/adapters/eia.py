"""EIA (U.S. Energy Information Administration) Open Data adapter for the SeriesStore.

``EIAAdapter`` fetches a single EIA series through the `API v2
<https://www.eia.gov/opendata/documentation.php>`_ endpoint and returns it in
the canonical internal format understood by
:class:`~aieng.forecasting.data.store.SeriesStore`.

Caching
-------
When ``cache_dir`` is provided, the adapter persists each series to
``{cache_dir}/{series_id}.parquet`` on first fetch and reads from the parquet
file on all subsequent calls.  This mirrors the :class:`FREDAdapter` pattern:
populate the cache once, then notebooks and backtests read from disk with no
further network access.

**API key requirement:** EIA requires a free API key obtained from
https://www.eia.gov/opendata/.  Provide it via the ``EIA_API_KEY`` environment
variable (recommended) or the ``api_key`` constructor argument.  The key is
only needed when the local cache is empty or ``refresh=True``.

Request shape
-------------
API v2 addresses a series by *route* plus a *facet* filter, rather than by the
flat ``PET.WCESTUS1.W`` identifier used by the retired v1 API::

    GET https://api.eia.gov/v2/petroleum/stoc/wstk/data/
          ?api_key=...
          &frequency=weekly
          &data[]=value
          &facets[series][]=WCESTUS1
          &sort[0][column]=period&sort[0][direction]=asc
          &offset=0&length=5000

The JSON payload nests records under ``response.data`` (each carrying
``period`` and ``value``) and the unpaginated row count under
``response.total``; :meth:`EIAAdapter._fetch_from_api` pages through in
``_MAX_ROWS_PER_REQUEST``-row requests until every row is retrieved.

``released_at``
---------------
Unlike :class:`FREDAdapter`, which approximates ``released_at = timestamp``,
this adapter models the **true publication instant**.  The Weekly Petroleum
Status Report covers the week ending Friday and is published the following
**Wednesday at ~10:30 a.m. ET**, so a weekly observation carries
``released_at = period + 5 days at 10:30 America/New_York`` (stored tz-naive).
Backtests therefore cannot see an inventory print days before the market did.

Consumers that expand this onto a daily business calendar must first roll
``released_at`` forward to a midnight grid point — see
:func:`~aieng.forecasting.data.features.business_daily_expand_from_releases`,
which reindexes on midnight-stamped business days and would silently drop
intraday release stamps.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from aieng.forecasting.data.adapters.base import BaseAdapter


_API_ROOT = "https://api.eia.gov/v2"

_MAX_ROWS_PER_REQUEST = 5000
"""Server-side cap on ``length``; larger values are silently truncated."""

_REQUEST_TIMEOUT_SECONDS = 60

WPSR_RELEASE_WEEKDAY_OFFSET_DAYS = 5
"""Days from the Friday week-ending ``period`` to the Wednesday WPSR release."""

WPSR_RELEASE_TIME_ET = "10:30"
"""Publication time of the Weekly Petroleum Status Report, US Eastern."""

_EASTERN = "America/New_York"


class EIAAdapter(BaseAdapter):
    """Adapter that fetches a single EIA API v2 series, with optional disk cache.

    Parameters
    ----------
    series_id : str
        EIA series identifier as used by the ``series`` facet, e.g.
        ``"WCESTUS1"`` (weekly US crude stocks excluding SPR) or
        ``"WPULEUS3"`` (weekly US refinery utilisation).  Also names the
        parquet cache file.
    route : str
        API v2 route holding the series, without the ``/data`` suffix, e.g.
        ``"petroleum/stoc/wstk"`` for weekly stocks or
        ``"petroleum/pnp/wiup"`` for weekly inputs and utilisation.
    api_key : str or None
        EIA API key.  If ``None``, the value is read from the ``EIA_API_KEY``
        environment variable.  The key is only consulted when a network fetch
        is actually required (cache miss or ``refresh=True``); adapters
        pointing at a populated cache can be instantiated without a key.
    cache_dir : str, Path, or None
        Directory to read/write parquet cache files.  When ``None``, caching
        is disabled and every ``fetch()`` call hits the EIA API.  Default:
        ``"data/eia"``.
    refresh : bool
        When ``True``, force a network fetch even if a cache file exists (and
        overwrite the cache).  Default: ``False``.
    frequency : str
        EIA frequency token sent as the ``frequency`` query parameter.
        Default: ``"weekly"``.
    facet : str
        Facet name used to select the series.  Default: ``"series"``, which is
        the facet carrying legacy v1-style identifiers on the petroleum routes.

    Raises
    ------
    ValueError
        When a network fetch is required but no API key is available.

    Examples
    --------
    Populate the cache once::

        >>> adapter = EIAAdapter("WCESTUS1", route="petroleum/stoc/wstk")
        >>> df = adapter.fetch()                     # hits API, writes parquet

    Subsequent reads never touch the network::

        >>> adapter = EIAAdapter("WCESTUS1", route="petroleum/stoc/wstk")
        >>> df = adapter.fetch()                     # reads parquet
    """

    DEFAULT_CACHE_DIR = "data/eia"

    def __init__(
        self,
        series_id: str,
        route: str,
        api_key: str | None = None,
        cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
        refresh: bool = False,
        frequency: str = "weekly",
        facet: str = "series",
    ) -> None:
        self._series_id = series_id
        self._route = route.strip("/")
        self._api_key = api_key or os.environ.get("EIA_API_KEY")
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._refresh = refresh
        self._frequency = frequency
        self._facet = facet

    @property
    def series_id(self) -> str:
        """EIA series identifier."""
        return self._series_id

    @property
    def route(self) -> str:
        """API v2 route holding the series, without the ``/data`` suffix."""
        return self._route

    @property
    def cache_path(self) -> Path | None:
        """Full path to this adapter's parquet cache file, or ``None`` if disabled."""
        if self._cache_dir is None:
            return None
        return self._cache_dir / f"{self._series_id}.parquet"

    def fetch(self) -> pd.DataFrame:
        """Return the series in canonical format, using the disk cache when available.

        Flow:

        1. If ``cache_dir`` is set and the parquet file exists and
           ``refresh=False``, read and return it.
        2. Otherwise fetch from the EIA API, normalize, write to parquet (when
           caching is enabled), and return.

        Returns
        -------
        pd.DataFrame
            Columns: ``timestamp`` (datetime64[ns]), ``value`` (float64),
            ``released_at`` (datetime64[ns]).  Sorted ascending by
            ``timestamp``.  Index is a default RangeIndex.

        Raises
        ------
        ValueError
            If a network fetch is required but no API key is available.
        RuntimeError
            If the EIA API request fails or returns no data.
        """
        cache_path = self.cache_path
        if cache_path is not None and cache_path.exists() and not self._refresh:
            return self._read_cache(cache_path)

        df = self._fetch_from_api()

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path, index=False)

        return df

    def _fetch_from_api(self) -> pd.DataFrame:
        """Fetch every page of the series directly from the EIA API v2."""
        if not self._api_key:
            raise ValueError(
                "EIA API key not provided.  Set the EIA_API_KEY environment variable "
                "or pass api_key= to EIAAdapter.  (Key is only required on cache miss; "
                "populated caches can be read without one.)"
            )

        import requests  # noqa: PLC0415

        url = f"{_API_ROOT}/{self._route}/data/"
        records: list[dict[str, Any]] = []
        offset = 0

        while True:
            params = [
                ("api_key", self._api_key),
                ("frequency", self._frequency),
                ("data[]", "value"),
                (f"facets[{self._facet}][]", self._series_id),
                ("sort[0][column]", "period"),
                ("sort[0][direction]", "asc"),
                ("offset", str(offset)),
                ("length", str(_MAX_ROWS_PER_REQUEST)),
            ]
            try:
                response = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT_SECONDS)
                response.raise_for_status()
                payload = response.json()
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to fetch EIA series '{self._series_id}' from route '{self._route}': {exc}"
                ) from exc

            body = payload.get("response", {})
            page = body.get("data", [])
            records.extend(page)

            total = body.get("total")
            offset += len(page)
            if not page or total is None or offset >= int(total):
                break

        if not records:
            raise RuntimeError(
                f"EIA series '{self._series_id}' on route '{self._route}' returned no data. "
                "Check the series id, the route, and the frequency."
            )

        return self._to_canonical(records)

    @staticmethod
    def _to_canonical(records: list[dict[str, Any]]) -> pd.DataFrame:
        """Normalize raw API records into ``(timestamp, value, released_at)``.

        ``released_at`` models the Weekly Petroleum Status Report schedule: the
        week ending on the ``period`` Friday is published the following
        Wednesday at ~10:30 a.m. ET.  The instant is converted to UTC and
        stored tz-naive so it compares directly against the naive ``as_of``
        values used by :class:`~aieng.forecasting.data.cutoff.CutoffEnforcer`.
        """
        df = pd.DataFrame.from_records(records)
        df["timestamp"] = pd.to_datetime(df["period"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["timestamp", "value"])

        release_day = df["timestamp"] + pd.Timedelta(days=WPSR_RELEASE_WEEKDAY_OFFSET_DAYS)
        release_local = pd.to_datetime(
            release_day.dt.strftime(f"%Y-%m-%d {WPSR_RELEASE_TIME_ET}"),
        ).dt.tz_localize(_EASTERN, ambiguous=True, nonexistent="shift_forward")
        df["released_at"] = release_local.dt.tz_convert("UTC").dt.tz_localize(None)

        df = df.sort_values("timestamp").reset_index(drop=True)
        return df[["timestamp", "value", "released_at"]]

    @staticmethod
    def _read_cache(cache_path: Path) -> pd.DataFrame:
        """Read a cached parquet and normalize dtypes defensively."""
        df = pd.read_parquet(cache_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["released_at"] = pd.to_datetime(df["released_at"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df[["timestamp", "value", "released_at"]].reset_index(drop=True)

    def __repr__(self) -> str:
        """Return a short representation without exposing the API key."""
        cache = self._cache_dir if self._cache_dir is not None else "disabled"
        return f"EIAAdapter(series_id={self._series_id!r}, route={self._route!r}, cache_dir={cache!r})"
