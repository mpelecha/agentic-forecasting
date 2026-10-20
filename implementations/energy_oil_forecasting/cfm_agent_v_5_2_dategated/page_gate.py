"""Shared mechanics of the publication-date gate: resolve, date, and filter grounded search text.

Used by v5.2's gated research tool (``research.py``) and by the gated ``search_web``
tool (``search_web.py``) that other agents, such as the scenario-memory agent, swap
in for the shared one. Both apply exactly the same rule:

1. Read each grounding page's address from the search redirect without fetching it.
2. Date each page (``date_gate.date_source``).
3. Keep only the text segments that Google's grounding metadata attributes solely to
   pages that passed; drop segments citing a failed page, or no page.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from energy_oil_forecasting.cfm_agent_v_5_2.config import CfmV52Settings
from energy_oil_forecasting.cfm_agent_v_5_2.schemas import ActiveSource
from energy_oil_forecasting.cfm_agent_v_5_2.source_registry import normalize_domain, publisher_for_domain
from energy_oil_forecasting.cfm_agent_v_5_2.tools.url_safety import validate_public_url
from energy_oil_forecasting.cfm_agent_v_5_2_dategated.date_gate import SourceDating, date_source


GATE_ID = "publication_date_gate_v2"
# Hosts that only redirect to the real page. An error from one of these means the
# destination is unknown; an error from any other host still tells us its address.
REDIRECT_HOSTS = {"vertexaisearch.cloud.google.com"}


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None  # surface each 3xx as an HTTPError so its Location can be read


def grounding_rows(response: Any) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Return the text, grounding pages (with chunk index), and segment supports of a search response."""
    content = response.choices[0].message.content or ""
    fields = getattr(response.choices[0], "provider_specific_fields", {}) or {}
    metadata = fields.get("grounding_metadata") or {}
    rows = [
        {"url": web["uri"], "title": web.get("title"), "chunk": index}
        for index, chunk in enumerate(metadata.get("groundingChunks") or [])
        if (web := chunk.get("web") or {}).get("uri")
    ]
    return content, rows, list(metadata.get("groundingSupports") or [])


def filter_by_support(
    supports: list[dict[str, Any]], accepted_chunks: set[int]
) -> tuple[str, list[int], dict[str, int]]:
    """Keep the text segments whose every grounding page passed the date gate.

    Returns the kept text, the accepted chunk indices it cites (in order of first
    use), and counts for the audit. Text with no grounding support is dropped: it
    cannot be dated.
    """
    kept: list[str] = []
    cited: list[int] = []
    dropped = 0
    for support in supports:
        text = ((support.get("segment") or {}).get("text") or "").strip()
        chunks = [int(i) for i in support.get("groundingChunkIndices") or []]
        if not text:
            continue
        if chunks and set(chunks) <= accepted_chunks:
            if text not in kept:
                kept.append(text)
            cited += [i for i in chunks if i not in cited]
        else:
            dropped += 1
    return " ".join(kept), cited, {"segments_kept": len(kept), "segments_dropped": dropped}


def rejected_pages_note(failed: list[SourceDating]) -> str:
    """Return retry guidance naming the pages the gate rejected."""
    if not failed:
        return ""
    return "These pages were rejected and must not be used or cited:\n- " + "\n- ".join(
        f"{d.resolved_url or d.grounding_url} ({d.decision})" for d in failed
    )


@dataclass
class GateResult:
    """One search attempt after the gate: what survives, and the audit record."""

    kept_text: str
    used_rows: list[dict[str, Any]]
    datings: list[SourceDating]
    failed: list[SourceDating]
    counts: dict[str, int]


class PageGate:
    """Resolve, date, and filter grounded search results; caches per workflow (call :meth:`reset`)."""

    def __init__(self, settings: CfmV52Settings, *, allow_undated: bool = False) -> None:
        self.settings = settings
        self.allow_undated = allow_undated
        self._resolved: dict[str, ActiveSource] = {}
        self._dated: dict[tuple[str, str], SourceDating] = {}

    def reset(self) -> None:
        self._resolved = {}
        self._dated = {}

    def final_url(self, grounding_url: str) -> str | None:
        """Follow redirects by reading each Location header, never fetching the destination.

        v5.2's resolver opens the final page, so a news site that refuses automated
        requests (HTTP 403) leaves the source unresolved even though the redirect
        already named its address.
        """
        url = grounding_url
        opener = build_opener(_NoRedirect())
        for _ in range(self.settings.source_resolution_max_redirects + 1):
            validate_public_url(url)
            if urlparse(url).hostname not in REDIRECT_HOSTS:
                return url  # not a redirect service: this is the destination; do not fetch it
            request = Request(
                url, headers={"User-Agent": "CFM-Agent-v5.1/1.0", "Accept": "text/html", "Range": "bytes=0-0"}
            )
            try:
                with opener.open(request, timeout=self.settings.source_resolution_timeout_seconds):
                    return None  # the redirect service answered without redirecting
            except HTTPError as exc:
                location = exc.headers.get("Location") if exc.code in (301, 302, 303, 307, 308) else None
                if not location:
                    return None
                url = urljoin(url, location)
        return None

    def resolve(self, grounding_url: str, provider_title: str | None) -> ActiveSource:
        """Return the page's identity (cached per grounding URL)."""
        cached = self._resolved.get(grounding_url)
        if cached is None:
            cached = self._resolve_uncached(grounding_url, provider_title)
            self._resolved[grounding_url] = cached
        return cached

    def _resolve_uncached(self, grounding_url: str, provider_title: str | None) -> ActiveSource:
        warning = None
        try:
            resolved_url = self.final_url(grounding_url)
            if resolved_url is None:
                warning = "resolution_failed: redirect did not name a destination"
        except Exception as exc:  # noqa: BLE001
            resolved_url, warning = None, f"resolution_failed: {type(exc).__name__}: {exc}"
        domain = normalize_domain(urlparse(resolved_url).hostname or "") if resolved_url else ""
        identity = publisher_for_domain(domain) if domain else None
        return ActiveSource(
            source_id="unassigned",
            rank=1,
            grounding_url=grounding_url,
            provider_title=provider_title,
            resolved_url=resolved_url,
            resolved_domain=domain or None,
            publisher=identity.publisher if identity else (domain or None),
            source_quality=identity.source_tier if identity else "unclassified",
            is_primary_or_official=identity.is_primary_or_official if identity else False,
            resolution_status="resolved" if resolved_url else "unresolved",
            resolution_warning=warning,
        )

    async def date_rows(self, rows: list[dict[str, Any]], cutoff_date: str) -> list[SourceDating]:
        async def _one(row: dict[str, Any]) -> SourceDating:
            key = (str(row["url"]), cutoff_date)
            if key not in self._dated:
                source = await asyncio.to_thread(self.resolve, key[0], row.get("title"))
                self._dated[key] = await asyncio.to_thread(
                    date_source, source, cutoff_date, self.settings, allow_undated=self.allow_undated
                )
            return self._dated[key]

        return list(await asyncio.gather(*(_one(row) for row in rows)))

    async def apply(self, rows: list[dict[str, Any]], supports: list[dict[str, Any]], cutoff_date: str) -> GateResult:
        """Date every page of one search result and keep only text grounded solely in passing pages."""
        datings = await self.date_rows(rows, cutoff_date)
        accepted = {row["chunk"] for row, dating in zip(rows, datings, strict=True) if dating.accepted}
        kept_text, cited, counts = filter_by_support(supports, accepted)
        by_chunk = {row["chunk"]: row for row in rows}
        return GateResult(
            kept_text=kept_text,
            used_rows=[by_chunk[i] for i in cited if i in by_chunk],
            datings=datings,
            failed=[d for d in datings if not d.accepted],
            counts=counts,
        )
