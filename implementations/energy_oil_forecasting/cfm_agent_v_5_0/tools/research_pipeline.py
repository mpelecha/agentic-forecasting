"""Main-verifier-cleaned active research pipeline for CFM Agent v5.0."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from datetime import datetime
from urllib.parse import urlparse
from urllib.request import Request, build_opener

from aieng.forecasting.methods.agentic.agent_factory import (
    AS_OF_STATE_KEY,
    _verify_no_leakage,
)
from energy_oil_forecasting.cfm_agent_v_5_0.config import CfmV50Settings
from energy_oil_forecasting.cfm_agent_v_5_0.fingerprints import stable_fingerprint
from energy_oil_forecasting.cfm_agent_v_5_0.schemas import (
    ActiveSource,
    QueryVerificationAudit,
    ResearchPacket,
    ResearchQuery,
    SearchVerificationAttempt,
    SourceAuditBundle,
    VerificationAuditBundle,
    VerifiedSummary,
)
from energy_oil_forecasting.cfm_agent_v_5_0.source_registry import (
    normalize_domain,
    publisher_for_domain,
)
from energy_oil_forecasting.cfm_agent_v_5_0.source_validator import run_source_audit
from energy_oil_forecasting.cfm_agent_v_5_0.tools.url_safety import (
    SafeRedirectHandler,
    validate_public_url,
)
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext


logger = logging.getLogger(__name__)
_SEARCH_FAILED = "[SEARCH_VERIFICATION_FAILED]"


class ResearchPipelineTool:
    """Return only cleaned summaries and resolved source identities to the LLM."""

    AREAS = ("underlying_event", "physical_flows", "supply_response", "market_reaction")

    def __init__(self, settings: CfmV50Settings, *, search_model: str) -> None:
        self.settings = settings
        self.search_model = search_model
        self._last_result: ResearchPacket | None = None
        self._last_source_audit = SourceAuditBundle(audit_enabled=settings.audit_enabled, executed=False)
        self._last_verification_audit = VerificationAuditBundle()
        self._lock = threading.Lock()
        self._expected_cutoff: str | None = None
        self._attempt_count = 0
        self._successful_execution_count = 0
        self._rejected_duplicate_count = 0
        self._in_progress = False
        self._attempt_errors: list[str] = []

    def prepare_workflow(self, cutoff_date: str) -> None:
        """Reset and bind the one-shot research tool to the current workflow."""
        datetime.strptime(cutoff_date, "%Y-%m-%d")
        with self._lock:
            self._expected_cutoff = cutoff_date
            self._last_result = None
            self._last_source_audit = SourceAuditBundle(audit_enabled=self.settings.audit_enabled, executed=False)
            self._last_verification_audit = VerificationAuditBundle()
            self._attempt_count = 0
            self._successful_execution_count = 0
            self._rejected_duplicate_count = 0
            self._in_progress = False
            self._attempt_errors = []

    @property
    def audit(self) -> dict[str, object]:
        with self._lock:
            return {
                "tool_name": "run_research_pipeline",
                "expected_cutoff": self._expected_cutoff,
                "attempt_count": self._attempt_count,
                "successful_execution_count": self._successful_execution_count,
                "rejected_duplicate_count": self._rejected_duplicate_count,
                "in_progress": self._in_progress,
                "attempt_errors": list(self._attempt_errors),
            }

    @property
    def last_result(self) -> ResearchPacket | None:
        return self._last_result.model_copy(deep=True) if self._last_result else None

    @property
    def last_source_audit(self) -> SourceAuditBundle:
        return self._last_source_audit.model_copy(deep=True)

    @property
    def last_verification_audit(self) -> VerificationAuditBundle:
        return self._last_verification_audit.model_copy(deep=True)

    def as_function_tool(self) -> FunctionTool:
        return FunctionTool(func=self.run_research_pipeline)

    async def run_research_pipeline(
        self,
        cutoff_date: str,
        underlying_event_query: str,
        physical_flows_query: str,
        supply_response_query: str,
        market_reaction_query: str,
        tool_context: ToolContext | None = None,
    ) -> str:
        """Run exactly one cutoff-bound four-query pipeline."""
        harness_cutoff = tool_context.state.get(AS_OF_STATE_KEY) if tool_context is not None else None
        authoritative_cutoff = str(harness_cutoff or self._expected_cutoff or cutoff_date)
        with self._lock:
            self._attempt_count += 1
            if self._successful_execution_count > 0 or self._in_progress:
                self._rejected_duplicate_count += 1
                message = "research pipeline already executed or is executing for this workflow"
                self._attempt_errors.append(message)
                return json.dumps({"status": "error", "error": message})
            if self._expected_cutoff is not None and authoritative_cutoff != self._expected_cutoff:
                message = "research cutoff does not match the harness-controlled workflow cutoff"
                self._attempt_errors.append(message)
                return json.dumps({"status": "error", "error": message})
            self._in_progress = True
        try:
            result = await self._run_research_pipeline_impl(
                authoritative_cutoff,
                underlying_event_query,
                physical_flows_query,
                supply_response_query,
                market_reaction_query,
                tool_context,
            )
            payload = json.loads(result)
            with self._lock:
                if isinstance(payload, dict) and payload.get("status") == "error":
                    self._attempt_errors.append(str(payload.get("error", "research pipeline error")))
                elif self._last_result is not None:
                    self._successful_execution_count += 1
            return result
        except Exception as exc:
            with self._lock:
                self._attempt_errors.append(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            with self._lock:
                self._in_progress = False

    async def _run_research_pipeline_impl(
        self,
        cutoff_date: str,
        underlying_event_query: str,
        physical_flows_query: str,
        supply_response_query: str,
        market_reaction_query: str,
        tool_context: ToolContext | None = None,
    ) -> str:
        """Run four cutoff-verified searches and return active summary evidence."""
        queries = {
            "underlying_event": underlying_event_query,
            "physical_flows": physical_flows_query,
            "supply_response": supply_response_query,
            "market_reaction": market_reaction_query,
        }
        harness_cutoff = tool_context.state.get(AS_OF_STATE_KEY) if tool_context is not None else None
        if harness_cutoff and cutoff_date and harness_cutoff != cutoff_date:
            logger.warning(
                "run_research_pipeline cutoff %r disagrees with harness cutoff %r; using harness cutoff",
                cutoff_date,
                harness_cutoff,
            )
        cutoff_date = str(harness_cutoff or cutoff_date)
        datetime.strptime(cutoff_date, "%Y-%m-%d")
        if set(queries) != set(self.AREAS):
            return json.dumps(
                {
                    "status": "error",
                    "error": f"queries must have exactly {list(self.AREAS)}",
                }
            )

        query_rows: list[ResearchQuery] = []
        summaries: list[VerifiedSummary] = []
        sources: list[ActiveSource] = []
        source_id_by_url: dict[str, str] = {}
        warnings: list[str] = []
        verification_audits: list[QueryVerificationAudit] = []

        for area in self.AREAS:
            query = str(queries[area]).strip()
            if not query:
                return json.dumps({"status": "error", "error": f"empty query for {area}"})
            query_rows.append(ResearchQuery(area=area, query=query))
            cleaned_text, rows, attempts = await self._verified_search(query, cutoff_date)
            summary_id = stable_fingerprint(
                {
                    "area": area,
                    "query": query,
                    "cutoff": cutoff_date,
                    "cleaned": cleaned_text,
                },
                prefix="verified_summary",
            )
            associated_ids: list[str] = []
            if cleaned_text is None:
                warnings.append(f"{area}: {_SEARCH_FAILED}")
                summaries.append(
                    VerifiedSummary(
                        verified_summary_id=summary_id,
                        query_area=area,
                        query=query,
                        cutoff_date=cutoff_date,
                        status="failed_closed",
                        verifier_confidence=attempts[-1].confidence if attempts else None,
                        verifier_attempt_count=len(attempts),
                    )
                )
                verification_audits.append(
                    QueryVerificationAudit(
                        verified_summary_id=summary_id,
                        query_area=area,
                        query=query,
                        cutoff_date=cutoff_date,
                        status="failed_closed",
                        attempts=attempts,
                        removed_flagged_claims=list(
                            dict.fromkeys(claim for attempt in attempts for claim in attempt.flagged_claims)
                        ),
                    )
                )
                continue

            for rank, row in enumerate(rows[: self.settings.research_max_results_per_query], 1):
                grounding_url = str(row["url"])
                source_id = source_id_by_url.get(grounding_url)
                if source_id is None:
                    source_id = f"source_{len(source_id_by_url) + 1:03d}"
                    source_id_by_url[grounding_url] = source_id
                    source = await asyncio.to_thread(
                        self._resolve_source,
                        source_id,
                        rank,
                        grounding_url,
                        row.get("title"),
                    )
                    sources.append(source)
                associated_ids.append(source_id)

            summaries.append(
                VerifiedSummary(
                    verified_summary_id=summary_id,
                    query_area=area,
                    query=query,
                    cutoff_date=cutoff_date,
                    status="accepted",
                    cleaned_summary=cleaned_text,
                    associated_source_ids=list(dict.fromkeys(associated_ids)),
                    verifier_confidence=attempts[-1].confidence,
                    verifier_attempt_count=len(attempts),
                )
            )
            verification_audits.append(
                QueryVerificationAudit(
                    verified_summary_id=summary_id,
                    query_area=area,
                    query=query,
                    cutoff_date=cutoff_date,
                    status="accepted",
                    attempts=attempts,
                    removed_flagged_claims=list(
                        dict.fromkeys(claim for attempt in attempts for claim in attempt.flagged_claims)
                    ),
                )
            )

        packet = ResearchPacket(
            packet_id=stable_fingerprint(
                {
                    "cutoff": cutoff_date,
                    "queries": [row.model_dump() for row in query_rows],
                    "summaries": [summary.model_dump() for summary in summaries],
                    "sources": [source.model_dump() for source in sources],
                },
                prefix="research_packet",
            ),
            cutoff_date=cutoff_date,
            queries=query_rows,
            verified_summaries=summaries,
            sources=sources,
            warnings=warnings,
        )
        self._last_result = packet
        self._last_verification_audit = VerificationAuditBundle(queries=verification_audits)
        self._last_source_audit = await asyncio.to_thread(run_source_audit, sources, cutoff_date, self.settings)
        return packet.model_dump_json(indent=2)

    async def _do_search(self, user_content: str) -> tuple[str, list[dict[str, str | None]]]:
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
        rows: list[dict[str, str | None]] = []
        for chunk in metadata.get("groundingChunks", []) or []:
            web = chunk.get("web") or {}
            if web.get("uri"):
                rows.append({"url": web["uri"], "title": web.get("title")})
        return content, rows

    async def _verified_search(
        self,
        query: str,
        cutoff_date: str,
    ) -> tuple[str | None, list[dict[str, str | None]], list[SearchVerificationAttempt]]:
        attempts: list[SearchVerificationAttempt] = []
        negative_guidance = ""
        for attempt in range(1, self.settings.search_verifier_max_attempts + 1):
            user_content = query + f"\n\nOnly include and cite information published strictly before {cutoff_date}."
            if negative_guidance:
                user_content += f"\n\n{negative_guidance}"
            content, rows = await self._do_search(user_content)
            verdict = await _verify_no_leakage(
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
            if verdict.clean and verdict.confidence >= self.settings.search_verifier_confidence_threshold:
                return verdict.filtered_text, rows, attempts
            negative_guidance = (
                f"The previous result may contain information published on or after {cutoff_date}. "
                "Do not repeat or rely on these claims:\n- "
                + "\n- ".join(verdict.flagged_claims or ["unspecified; be more conservative"])
            )
        return None, [], attempts

    def _resolve_source(
        self,
        source_id: str,
        rank: int,
        grounding_url: str,
        provider_title: str | None,
    ) -> ActiveSource:
        resolved_url = None
        warning = None
        try:
            validate_public_url(grounding_url)
            request = Request(
                grounding_url,
                headers={
                    "User-Agent": "CFM-Agent-v5.0/1.0",
                    "Accept": "text/html,application/xhtml+xml",
                    "Range": "bytes=0-0",
                },
            )
            with build_opener(SafeRedirectHandler(self.settings.source_resolution_max_redirects)).open(
                request,
                timeout=self.settings.source_resolution_timeout_seconds,
            ) as response:
                resolved_url = response.geturl()
                validate_public_url(resolved_url)
        except Exception as exc:  # noqa: BLE001
            warning = f"resolution_failed: {type(exc).__name__}: {exc}"

        domain = normalize_domain(urlparse(resolved_url).hostname or "") if resolved_url else ""
        identity = publisher_for_domain(domain) if domain else None
        publisher = identity.publisher if identity else (domain or None)
        quality = identity.source_tier if identity else "unclassified"
        primary = identity.is_primary_or_official if identity else False
        return ActiveSource(
            source_id=source_id,
            rank=rank,
            grounding_url=grounding_url,
            provider_title=provider_title,
            resolved_url=resolved_url,
            resolved_domain=domain or None,
            publisher=publisher,
            source_quality=quality,
            is_primary_or_official=primary,
            resolution_status="resolved" if resolved_url else "unresolved",
            resolution_warning=warning,
        )
