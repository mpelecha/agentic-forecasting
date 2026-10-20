"""v5.2 research pipeline with a deterministic publication-date gate on every search result."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from energy_oil_forecasting.cfm_agent_v_5_2.config import CfmV52Settings
from energy_oil_forecasting.cfm_agent_v_5_2.schemas import ActiveSource, SearchVerificationAttempt
from energy_oil_forecasting.cfm_agent_v_5_2.tools import research_pipeline as _v52
from energy_oil_forecasting.cfm_agent_v_5_2.tools.research_pipeline import ResearchPipelineTool
from energy_oil_forecasting.cfm_agent_v_5_2_dategated.date_gate import SourceDating, date_source


GATE_ID = "publication_date_gate_v1"


class DateGatedResearchPipelineTool(ResearchPipelineTool):
    """v5.2's research pipeline, accepting a search result only if every grounding page predates the cutoff.

    Each attempt runs v5.2's search and LLM cutoff verifier unchanged. A result the
    verifier accepts is then dated page by page (see ``date_gate``); if any page fails,
    the whole result is discarded and the search is retried with those pages named as
    off-limits. The attempt budget is v5.2's ``search_verifier_max_attempts``; a query
    with no passing attempt fails closed, as it already does when the verifier rejects
    every attempt. Tool name, signature, and description are inherited unchanged, so
    the agent's prompt and tool schema are identical to v5.2's.
    """

    def __init__(self, settings: CfmV52Settings, *, search_model: str, allow_undated: bool = False) -> None:
        super().__init__(settings, search_model=search_model)
        self.allow_undated = allow_undated
        self._resolved: dict[str, ActiveSource] = {}
        self._gate_log: list[dict[str, Any]] = []

    def prepare_workflow(self, cutoff_date: str) -> None:
        super().prepare_workflow(cutoff_date)
        with self._lock:
            self._resolved = {}
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
            cached = super()._resolve_source(source_id, rank, grounding_url, provider_title)
            self._resolved[grounding_url] = cached
        return cached.model_copy(update={"source_id": source_id, "rank": rank, "provider_title": provider_title})

    async def _date_rows(self, rows: list[dict[str, str | None]], cutoff_date: str) -> list[SourceDating]:
        async def _one(row: dict[str, str | None]) -> SourceDating:
            source = await asyncio.to_thread(self._resolve_source, "gate_check", 1, str(row["url"]), row.get("title"))
            return await asyncio.to_thread(
                date_source, source, cutoff_date, self.settings, allow_undated=self.allow_undated
            )

        return list(await asyncio.gather(*(_one(row) for row in rows)))

    async def _verified_search(
        self,
        query: str,
        cutoff_date: str,
    ) -> tuple[str | None, list[dict[str, str | None]], list[SearchVerificationAttempt]]:
        attempts: list[SearchVerificationAttempt] = []
        guidance = ""
        for attempt in range(1, self.settings.search_verifier_max_attempts + 1):
            user_content = query + f"\n\nOnly include and cite information published strictly before {cutoff_date}."
            if guidance:
                user_content += f"\n\n{guidance}"
            content, rows = await self._do_search(user_content)
            verdict = await _v52._verify_no_leakage(
                text=content,
                query=query,
                cutoff_date=cutoff_date,
                verifier_model=self.settings.search_verifier_model,
                openai_base_url=os.getenv("OPENAI_BASE_URL", ""),
                openai_api_key=os.getenv("OPENAI_API_KEY"),
            )
            attempts.append(
                SearchVerificationAttempt(
                    attempt=attempt,
                    clean=verdict.clean,
                    confidence=verdict.confidence,
                    flagged_claims=list(verdict.flagged_claims),
                    source_count=len(rows),
                )
            )
            entry: dict[str, Any] = {"query": query, "cutoff_date": cutoff_date, "attempt": attempt}
            if not (verdict.clean and verdict.confidence >= self.settings.search_verifier_confidence_threshold):
                self._log({**entry, "outcome": "verifier_rejected", "pages": []})
                guidance = (
                    f"The previous result may contain information published on or after {cutoff_date}. "
                    "Do not repeat or rely on these claims:\n- "
                    + "\n- ".join(verdict.flagged_claims or ["unspecified; be more conservative"])
                )
                continue

            datings = await self._date_rows(rows, cutoff_date)
            failed = [d for d in datings if not d.accepted]
            if datings and not failed:
                self._log({**entry, "outcome": "accepted", "pages": [d.to_dict() for d in datings]})
                return verdict.filtered_text, rows, attempts
            self._log(
                {
                    **entry,
                    "outcome": "no_sources" if not datings else "date_rejected",
                    "pages": [d.to_dict() for d in datings],
                }
            )
            guidance = (
                f"Use only pages published strictly before {cutoff_date} that state their publication date. "
                "These pages were rejected and must not be used or cited:\n- "
                + "\n- ".join(f"{d.resolved_url or d.grounding_url} ({d.decision})" for d in failed)
                if failed
                else f"Cite only dated pages published strictly before {cutoff_date}."
            )
        return None, [], attempts

    def _log(self, entry: dict[str, Any]) -> None:
        with self._lock:
            self._gate_log.append(entry)
