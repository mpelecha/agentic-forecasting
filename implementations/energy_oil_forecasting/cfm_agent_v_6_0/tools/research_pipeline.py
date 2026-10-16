"""Main-verifier-cleaned active research pipeline for CFM Agent v6.0 (specification Section 4.1).

The pipeline runs exactly four neutral research categories per workflow. Each
category's grounded search result is verified as strictly before the cutoff by an
independent verifier (fail-closed per category), the cleaned summary is split into
passages, and every grounding URL is mechanically resolved to a publisher
identity. Only cleaned summaries, their passages, and resolved source identities
reach the decision LLM; destination-page text never enters the packet.

The search and cutoff-verifier calls take their model and generation parameters
from settings through :mod:`llm_roles`, and the exact keyword arguments handed
to ``litellm`` are recorded for the audit.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from datetime import datetime
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, build_opener

from aieng.forecasting.methods.agentic.agent_factory import (
    _LEAKAGE_VERIFIER_INSTRUCTION,
    AS_OF_STATE_KEY,
    _build_leakage_verification_schema,
    _LeakageVerification,
)
from aieng.forecasting.methods.llm_processes._client import (
    make_json_schema_response_format,
    strip_markdown_fence,
)
from energy_oil_forecasting.cfm_agent_v_6_0.config import CfmV60Settings
from energy_oil_forecasting.cfm_agent_v_6_0.fingerprints import identity
from energy_oil_forecasting.cfm_agent_v_6_0.llm_roles import (
    completion_kwargs,
    record_effective_parameters,
)
from energy_oil_forecasting.cfm_agent_v_6_0.passages import split_summary
from energy_oil_forecasting.cfm_agent_v_6_0.schemas import (
    ActiveSource,
    QueryVerificationAudit,
    ResearchPacket,
    ResearchQuery,
    SearchVerificationAttempt,
    SourceAuditBundle,
    VerificationAuditBundle,
    VerifiedSummary,
)
from energy_oil_forecasting.cfm_agent_v_6_0.source_registry import (
    normalize_domain,
    publisher_for_domain,
)
from energy_oil_forecasting.cfm_agent_v_6_0.source_validator import run_source_audit
from energy_oil_forecasting.cfm_agent_v_6_0.tools.url_safety import (
    SafeRedirectHandler,
    validate_public_url,
)
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from pydantic import ValidationError


logger = logging.getLogger(__name__)
_SEARCH_FAILED = "[SEARCH_VERIFICATION_FAILED]"
_SEARCH_INSTRUCTION = (
    "You are a specialized web search assistant. Return concise grounded factual content and "
    "source URLs. Do not infer bibliographic metadata."
)
LeakageVerdict = _LeakageVerification


class ResearchPipelineTool:
    """Return only cleaned summaries, their passages, and resolved source identities to the LLM."""

    AREAS = ("underlying_event", "physical_flows", "supply_response", "market_reaction")

    def __init__(self, settings: CfmV60Settings) -> None:
        self.settings = settings
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

    @property
    def search_model(self) -> str:
        """The configured search-role model (``settings.search_model``)."""
        return self.settings.search_model

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

    async def _run_research_pipeline_impl(  # noqa: PLR0914
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
            return json.dumps({"status": "error", "error": f"queries must have exactly {list(self.AREAS)}"})

        query_rows: list[ResearchQuery] = []
        summaries: list[VerifiedSummary] = []
        sources: list[ActiveSource] = []
        source_id_by_url: dict[str, str] = {}
        warnings: list[str] = []
        verification_audits: list[QueryVerificationAudit] = []

        for summary_index, area in enumerate(self.AREAS, 1):
            query = str(queries[area]).strip()
            if not query:
                return json.dumps({"status": "error", "error": f"empty query for {area}"})
            query_rows.append(ResearchQuery(area=area, query=query))
            cleaned_text, rows, attempts = await self._verified_search(query, cutoff_date)
            removed_claims = list(dict.fromkeys(claim for attempt in attempts for claim in attempt.flagged_claims))
            if cleaned_text is None:
                summary_id = self._summary_identity(area, query, cutoff_date, None, [])
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
                        removed_flagged_claims=removed_claims,
                    )
                )
                continue

            associated_ids: list[str] = []
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

            passages = split_summary(
                cleaned_text,
                summary_index=summary_index,
                min_chars=self.settings.passage_min_chars,
                abbreviations=self.settings.passage_abbreviations,
            )
            summary_id = self._summary_identity(
                area,
                query,
                cutoff_date,
                cleaned_text,
                [passage.model_dump() for passage in passages],
            )
            summaries.append(
                VerifiedSummary(
                    verified_summary_id=summary_id,
                    query_area=area,
                    query=query,
                    cutoff_date=cutoff_date,
                    status="accepted",
                    cleaned_summary=cleaned_text,
                    passages=passages,
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
                    removed_flagged_claims=removed_claims,
                )
            )

        packet = ResearchPacket(
            packet_id=identity(
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

    @staticmethod
    def _summary_identity(
        area: str,
        query: str,
        cutoff_date: str,
        cleaned: str | None,
        passages: list[dict[str, Any]],
    ) -> str:
        """Return the summary id, which binds the cleaned text and its passages (Section 4.1)."""
        return identity(
            {"area": area, "query": query, "cutoff": cutoff_date, "cleaned": cleaned, "passages": passages},
            prefix="verified_summary",
        )

    async def _do_search(self, user_content: str) -> tuple[str, list[dict[str, str | None]]]:
        """One grounded search call under the ``search`` role parameters."""
        import litellm  # noqa: PLC0415

        kwargs = completion_kwargs(self.settings, "search", with_tools=True)
        kwargs["messages"] = [
            {"role": "system", "content": _SEARCH_INSTRUCTION},
            {"role": "user", "content": user_content},
        ]
        kwargs["tools"] = [{"googleSearch": {}}]
        response = await litellm.acompletion(**kwargs)
        record_effective_parameters("search", kwargs, response_model=getattr(response, "model", None))
        content = response.choices[0].message.content or ""
        fields = getattr(response.choices[0], "provider_specific_fields", {}) or {}
        metadata = fields.get("grounding_metadata") or {}
        rows: list[dict[str, str | None]] = []
        for chunk in metadata.get("groundingChunks", []) or []:
            web = chunk.get("web") or {}
            if web.get("uri"):
                rows.append({"url": web["uri"], "title": web.get("title")})
        return content, rows

    async def _verify_no_leakage(self, *, text: str, query: str, cutoff_date: str) -> LeakageVerdict:
        """Judge a search result for post-cutoff claims under the ``cutoff_verifier`` role parameters.

        A malformed verifier response never raises: it is treated as a non-clean
        verdict with the lowest confidence, so it consumes an attempt like any
        other rejection.
        """
        import litellm  # noqa: PLC0415

        kwargs = completion_kwargs(self.settings, "cutoff_verifier")
        kwargs["messages"] = [
            {"role": "system", "content": _LEAKAGE_VERIFIER_INSTRUCTION},
            {
                "role": "user",
                "content": f"Original query: {query}\nCutoff date: {cutoff_date}\n\nSearch result to verify:\n{text}",
            },
        ]
        kwargs["response_format"] = make_json_schema_response_format(
            "LeakageVerification", _build_leakage_verification_schema()
        )
        response = await litellm.acompletion(**kwargs)
        record_effective_parameters("cutoff_verifier", kwargs, response_model=getattr(response, "model", None))
        raw = response.choices[0].message.content or "{}"
        try:
            return LeakageVerdict.model_validate(json.loads(strip_markdown_fence(raw)))
        except (json.JSONDecodeError, ValidationError):
            logger.warning("cutoff verifier returned unparseable output; treating as non-clean")
            return LeakageVerdict(
                flagged_claims=["verifier response could not be parsed"],
                filtered_text=text,
                confidence=1,
                clean=False,
            )

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
            verdict = await self._verify_no_leakage(text=content, query=query, cutoff_date=cutoff_date)
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
                    "User-Agent": "CFM-Agent-v6.0/1.0",
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
        registered = publisher_for_domain(domain) if domain else None
        publisher = registered.publisher if registered else (domain or None)
        quality = registered.source_tier if registered else "unclassified"
        primary = registered.is_primary_or_official if registered else False
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


__all__ = ["LeakageVerdict", "ResearchPipelineTool"]
