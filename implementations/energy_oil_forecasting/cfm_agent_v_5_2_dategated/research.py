"""v5.2 research pipeline with a deterministic publication-date gate on every search result."""

from __future__ import annotations

import os
from typing import Any

from energy_oil_forecasting.cfm_agent_v_5_2.config import CfmV52Settings
from energy_oil_forecasting.cfm_agent_v_5_2.schemas import ActiveSource, SearchVerificationAttempt
from energy_oil_forecasting.cfm_agent_v_5_2.tools import research_pipeline as _v52
from energy_oil_forecasting.cfm_agent_v_5_2.tools.research_pipeline import ResearchPipelineTool
from energy_oil_forecasting.cfm_agent_v_5_2_dategated.page_gate import (
    GATE_ID,
    PageGate,
    grounding_rows,
    rejected_pages_note,
)


__all__ = ["GATE_ID", "DateGatedResearchPipelineTool"]


class DateGatedResearchPipelineTool(ResearchPipelineTool):
    """v5.2's research pipeline, passing on only text grounded in pages dated before the cutoff.

    Each attempt runs v5.2's grounded search. Every grounding page is dated (see
    ``date_gate``), and the summary is rebuilt from the text segments that Google's
    grounding metadata attributes only to pages that passed; segments citing a failed
    page, or no page, are dropped (``page_gate``). v5.2's LLM cutoff verifier then
    checks the kept text. If nothing survives or the verifier rejects it, the search is
    retried (naming the rejected pages) within v5.2's ``search_verifier_max_attempts``;
    a query with no passing attempt fails closed. Tool name, signature, and description
    are inherited unchanged, so the agent's prompt and tool schema are identical to v5.2's.
    """

    def __init__(self, settings: CfmV52Settings, *, search_model: str, allow_undated: bool = False) -> None:
        super().__init__(settings, search_model=search_model)
        self.allow_undated = allow_undated
        self.gate = PageGate(settings, allow_undated=allow_undated)
        self._gate_log: list[dict[str, Any]] = []

    def prepare_workflow(self, cutoff_date: str) -> None:
        super().prepare_workflow(cutoff_date)
        with self._lock:
            self.gate.reset()
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
        return grounding_rows(response)

    def _resolve_source(
        self,
        source_id: str,
        rank: int,
        grounding_url: str,
        provider_title: str | None,
    ) -> ActiveSource:
        # Reuse the gate's resolution so the packet's sources are exactly the pages that were dated.
        resolved = self.gate.resolve(grounding_url, provider_title)
        return resolved.model_copy(update={"source_id": source_id, "rank": rank, "provider_title": provider_title})

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
            result = await self.gate.apply(rows, supports, cutoff_date)
            entry: dict[str, Any] = {
                "query": query,
                "cutoff_date": cutoff_date,
                "attempt": attempt,
                **result.counts,
                "pages": [d.to_dict() for d in result.datings],
            }
            rejected = rejected_pages_note(result.failed)

            if not result.kept_text:
                attempts.append(
                    SearchVerificationAttempt(
                        attempt=attempt, clean=False, confidence=1, flagged_claims=[], source_count=len(rows)
                    )
                )
                self._log({**entry, "outcome": "no_dated_text"})
                guidance = (
                    f"Use only pages published strictly before {cutoff_date} that state their publication date. "
                    + rejected
                )
                continue

            verdict = await _v52._verify_no_leakage(
                text=result.kept_text,
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
                    source_count=len(result.used_rows),
                )
            )
            if verdict.clean and verdict.confidence >= self.settings.search_verifier_confidence_threshold:
                self._log({**entry, "outcome": "accepted"})
                return verdict.filtered_text, result.used_rows, attempts
            self._log({**entry, "outcome": "verifier_rejected"})
            guidance = (
                f"The previous result may contain information published on or after {cutoff_date}. "
                "Do not repeat or rely on these claims:\n- "
                + "\n- ".join(verdict.flagged_claims or ["unspecified; be more conservative"])
                + (f"\n\n{rejected}" if rejected else "")
            )
        return None, [], attempts

    def _log(self, entry: dict[str, Any]) -> None:
        with self._lock:
            self._gate_log.append(entry)
