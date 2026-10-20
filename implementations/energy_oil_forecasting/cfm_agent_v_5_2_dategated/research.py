"""v5.2 research pipeline with a deterministic publication-date gate on every search result."""

from __future__ import annotations

import asyncio
import os
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from energy_oil_forecasting.cfm_agent_v_5_2.config import CfmV52Settings
from energy_oil_forecasting.cfm_agent_v_5_2.schemas import ActiveSource, SearchVerificationAttempt
from energy_oil_forecasting.cfm_agent_v_5_2.source_registry import normalize_domain, publisher_for_domain
from energy_oil_forecasting.cfm_agent_v_5_2.tools import research_pipeline as _v52
from energy_oil_forecasting.cfm_agent_v_5_2.tools.research_pipeline import ResearchPipelineTool
from energy_oil_forecasting.cfm_agent_v_5_2.tools.url_safety import validate_public_url
from energy_oil_forecasting.cfm_agent_v_5_2_dategated.date_gate import SourceDating, date_source


GATE_ID = "publication_date_gate_v2"
# Hosts that only redirect to the real page. An error from one of these means the
# destination is unknown; an error from any other host still tells us its address.
REDIRECT_HOSTS = {"vertexaisearch.cloud.google.com"}


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None  # surface each 3xx as an HTTPError so its Location can be read


def _filter_by_support(
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


class DateGatedResearchPipelineTool(ResearchPipelineTool):
    """v5.2's research pipeline, passing on only text grounded in pages dated before the cutoff.

    Each attempt runs v5.2's grounded search. Every grounding page is dated (see
    ``date_gate``), and the summary is rebuilt from the text segments that Google's
    grounding metadata attributes only to pages that passed; segments citing a failed
    page, or no page, are dropped. v5.2's LLM cutoff verifier then checks the kept text.
    If nothing survives or the verifier rejects it, the search is retried (naming the
    rejected pages) within v5.2's ``search_verifier_max_attempts``; a query with no
    passing attempt fails closed. Tool name, signature, and description are inherited
    unchanged, so the agent's prompt and tool schema are identical to v5.2's.
    """

    def __init__(self, settings: CfmV52Settings, *, search_model: str, allow_undated: bool = False) -> None:
        super().__init__(settings, search_model=search_model)
        self.allow_undated = allow_undated
        self._resolved: dict[str, ActiveSource] = {}
        self._dated: dict[str, SourceDating] = {}
        self._gate_log: list[dict[str, Any]] = []

    def prepare_workflow(self, cutoff_date: str) -> None:
        super().prepare_workflow(cutoff_date)
        with self._lock:
            self._resolved = {}
            self._dated = {}
            self._gate_log = []

    @property
    def audit(self) -> dict[str, object]:
        audit = super().audit
        with self._lock:
            audit["date_gate"] = {
                "gate_id": GATE_ID,
                "allow_undated": self.allow_undated,
                "attempts": [dict(entry) for entry in self._gate_log],
            }
        return audit

    # ── search ────────────────────────────────────────────────────────────────

    async def _grounded_search(self, user_content: str) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
        """Run v5.2's search call; return the text, grounding pages, and segment supports."""
        import litellm  # noqa: PLC0415

        model = self.search_model if self.search_model.startswith("openai/") else f"openai/{self.search_model}"
        response = await litellm.acompletion(
            model=model,
            api_base=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a specialized web search assistant. Return concise grounded factual content and "
                        "source URLs. Do not infer bibliographic metadata."
                    ),
                },
                {"role": "user", "content": user_content},
            ],
            tools=[{"googleSearch": {}}],
            temperature=0.0,
            max_tokens=4096,
            timeout=60.0,
        )
        content = response.choices[0].message.content or ""
        fields = getattr(response.choices[0], "provider_specific_fields", {}) or {}
        metadata = fields.get("grounding_metadata") or {}
        rows = [
            {"url": web["uri"], "title": web.get("title"), "chunk": index}
            for index, chunk in enumerate(metadata.get("groundingChunks") or [])
            if (web := chunk.get("web") or {}).get("uri")
        ]
        return content, rows, list(metadata.get("groundingSupports") or [])

    # ── source resolution and dating ──────────────────────────────────────────

    def _final_url(self, grounding_url: str) -> str | None:
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

    def _resolve_source(
        self,
        source_id: str,
        rank: int,
        grounding_url: str,
        provider_title: str | None,
    ) -> ActiveSource:
        # The gate resolves every grounding page before the packet is built; reuse
        # that resolution so the packet's sources are exactly the pages that were dated.
        cached = self._resolved.get(grounding_url)
        if cached is None:
            cached = self._resolve_uncached(grounding_url, provider_title)
            self._resolved[grounding_url] = cached
        return cached.model_copy(update={"source_id": source_id, "rank": rank, "provider_title": provider_title})

    def _resolve_uncached(self, grounding_url: str, provider_title: str | None) -> ActiveSource:
        warning = None
        try:
            resolved_url = self._final_url(grounding_url)
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

    async def _date_rows(self, rows: list[dict[str, Any]], cutoff_date: str) -> list[SourceDating]:
        async def _one(row: dict[str, Any]) -> SourceDating:
            url = str(row["url"])
            if url not in self._dated:
                source = await asyncio.to_thread(self._resolve_source, "gate_check", 1, url, row.get("title"))
                self._dated[url] = await asyncio.to_thread(
                    date_source, source, cutoff_date, self.settings, allow_undated=self.allow_undated
                )
            return self._dated[url]

        return list(await asyncio.gather(*(_one(row) for row in rows)))

    # ── one query ─────────────────────────────────────────────────────────────

    async def _verified_search(
        self,
        query: str,
        cutoff_date: str,
    ) -> tuple[str | None, list[dict[str, Any]], list[SearchVerificationAttempt]]:
        attempts: list[SearchVerificationAttempt] = []
        guidance = ""
        for attempt in range(1, self.settings.search_verifier_max_attempts + 1):
            user_content = query + f"\n\nOnly include and cite information published strictly before {cutoff_date}."
            if guidance:
                user_content += f"\n\n{guidance}"
            _content, rows, supports = await self._grounded_search(user_content)

            datings = await self._date_rows(rows, cutoff_date)
            accepted = {row["chunk"] for row, dating in zip(rows, datings, strict=True) if dating.accepted}
            kept_text, cited, counts = _filter_by_support(supports, accepted)
            failed = [d for d in datings if not d.accepted]
            entry: dict[str, Any] = {
                "query": query,
                "cutoff_date": cutoff_date,
                "attempt": attempt,
                **counts,
                "pages": [d.to_dict() for d in datings],
            }
            rejected_pages = (
                "These pages were rejected and must not be used or cited:\n- "
                + "\n- ".join(f"{d.resolved_url or d.grounding_url} ({d.decision})" for d in failed)
                if failed
                else ""
            )

            if not kept_text:
                attempts.append(
                    SearchVerificationAttempt(
                        attempt=attempt, clean=False, confidence=1, flagged_claims=[], source_count=len(rows)
                    )
                )
                self._log({**entry, "outcome": "no_dated_text"})
                guidance = (
                    f"Use only pages published strictly before {cutoff_date} that state their publication date. "
                    + rejected_pages
                )
                continue

            verdict = await _v52._verify_no_leakage(
                text=kept_text,
                query=query,
                cutoff_date=cutoff_date,
                verifier_model=self.settings.search_verifier_model,
                openai_base_url=os.getenv("OPENAI_BASE_URL", ""),
                openai_api_key=os.getenv("OPENAI_API_KEY"),
            )
            by_chunk = {row["chunk"]: row for row in rows}
            used_rows = [by_chunk[i] for i in cited if i in by_chunk]
            attempts.append(
                SearchVerificationAttempt(
                    attempt=attempt,
                    clean=verdict.clean,
                    confidence=verdict.confidence,
                    flagged_claims=list(verdict.flagged_claims),
                    source_count=len(used_rows),
                )
            )
            if verdict.clean and verdict.confidence >= self.settings.search_verifier_confidence_threshold:
                self._log({**entry, "outcome": "accepted"})
                return verdict.filtered_text, used_rows, attempts
            self._log({**entry, "outcome": "verifier_rejected"})
            guidance = (
                f"The previous result may contain information published on or after {cutoff_date}. "
                "Do not repeat or rely on these claims:\n- "
                + "\n- ".join(verdict.flagged_claims or ["unspecified; be more conservative"])
                + (f"\n\n{rejected_pages}" if rejected_pages else "")
            )
        return None, [], attempts

    def _log(self, entry: dict[str, Any]) -> None:
        with self._lock:
            self._gate_log.append(entry)
