"""A date-gated drop-in for the shared ``search_web`` tool in ``aieng.forecasting.methods.agentic``.

The shared tool asks the search model for pre-cutoff information and has an LLM
verifier strip post-cutoff claims; nothing checks page dates. This builds a tool
with the same name, signature, and description (so the calling agent's prompt and
tool schema are unchanged) whose every attempt also passes the publication-date gate
(``page_gate``): only text that the grounding metadata attributes solely to pages
dated strictly before the cutoff is kept, and the LLM verifier then checks that.
Everything else follows the shared tool: the harness-controlled cutoff wins over the
LLM's argument, failed attempts retry with feedback, and exhausting the attempts
returns the same ``[SEARCH_VERIFICATION_FAILED]`` sentinel.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from aieng.forecasting.methods.agentic import agent_factory as _af
from aieng.forecasting.methods.agentic.agent_factory import AS_OF_STATE_KEY, ContextRetrievalConfig
from energy_oil_forecasting.cfm_agent_v_5_2_dategated.page_gate import (
    GATE_ID,
    PageGate,
    grounding_rows,
    rejected_pages_note,
)
from google.adk.tools.tool_context import ToolContext


logger = logging.getLogger(__name__)


def build_date_gated_search_web(
    config: ContextRetrievalConfig,
    *,
    gate: PageGate,
    log: list[dict[str, Any]],
    openai_base_url: str,
    openai_api_key: str | None,
) -> Callable[..., Any]:
    """Return a gated ``search_web`` coroutine function; each attempt is appended to ``log``."""

    def _format_result(content: str, sources: list[str]) -> str:
        if sources:
            content += "\n\nSources:\n" + "\n".join(sources[:5])
        return content

    async def _grounded_search(user_content: str) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
        import litellm  # noqa: PLC0415

        model = config.search_model if config.search_model.startswith("openai/") else f"openai/{config.search_model}"
        response = await litellm.acompletion(
            model=model,
            api_base=openai_base_url,
            api_key=openai_api_key,
            messages=[
                {"role": "system", "content": config.instruction},
                {"role": "user", "content": user_content},
            ],
            tools=[{"googleSearch": {}}],
            max_tokens=config.max_output_tokens or 4096,
            temperature=config.temperature or 0.0,
            timeout=60.0,
        )
        return grounding_rows(response)

    async def search_web(query: str, cutoff_date: str | None = None, tool_context: ToolContext | None = None) -> str:
        harness_as_of = tool_context.state.get(AS_OF_STATE_KEY) if tool_context is not None else None
        if harness_as_of and cutoff_date and harness_as_of != cutoff_date:
            logger.warning(
                "search_web: cutoff_date=%r disagrees with harness as_of=%r; using the harness value.",
                cutoff_date,
                harness_as_of,
            )
        effective_cutoff = harness_as_of or cutoff_date
        if not (effective_cutoff and config.enforce_cutoff):
            content, rows, _ = await _grounded_search(query)
            return _format_result(content, [str(row["url"]) for row in rows])

        guidance = ""
        for attempt in range(1, config.verifier_max_attempts + 1):
            user_content = (
                query + f"\n\nOnly include and cite information published strictly before {effective_cutoff}."
            )
            if guidance:
                user_content += f"\n\n{guidance}"
            _content, rows, supports = await _grounded_search(user_content)
            result = await gate.apply(rows, supports, effective_cutoff)
            entry: dict[str, Any] = {
                "query": query,
                "cutoff_date": effective_cutoff,
                "attempt": attempt,
                **result.counts,
                "pages": [d.to_dict() for d in result.datings],
            }
            rejected = rejected_pages_note(result.failed)
            if not result.kept_text:
                log.append({**entry, "outcome": "no_dated_text"})
                guidance = (
                    f"Use only pages published strictly before {effective_cutoff} that state their publication "
                    "date. " + rejected
                )
                continue

            verdict = await _af._verify_no_leakage(
                text=result.kept_text,
                query=query,
                cutoff_date=effective_cutoff,
                verifier_model=config.verifier_model,
                openai_base_url=openai_base_url,
                openai_api_key=openai_api_key,
            )
            if verdict.clean and verdict.confidence >= config.verifier_confidence_threshold:
                log.append({**entry, "outcome": "accepted"})
                sources = [
                    (gate.resolve(str(row["url"]), row.get("title")).resolved_url or str(row["url"]))
                    for row in result.used_rows
                ]
                return _format_result(verdict.filtered_text, sources)
            log.append({**entry, "outcome": "verifier_rejected"})
            guidance = (
                f"Your previous search result may have included information published on or after "
                f"{effective_cutoff}. Do not repeat or rely on these claims:\n- "
                + "\n- ".join(verdict.flagged_claims or ["(unspecified — be more conservative)"])
                + (f"\n\n{rejected}" if rejected else "")
            )

        logger.error("search_web exhausted %d attempts without a dated, verified result.", config.verifier_max_attempts)
        return (
            f"[SEARCH_VERIFICATION_FAILED] Could not verify search results as free of information "
            f"published on or after {effective_cutoff} after {config.verifier_max_attempts} attempts. "
            "Treat this as no verified news context being available for this query."
        )

    # The agent sees the shared tool's name and description, so its prompt is unchanged.
    original = _af._build_search_tool(config, openai_base_url=openai_base_url, openai_api_key=openai_api_key)
    search_web.__doc__ = original.__doc__
    search_web.__name__ = original.__name__
    search_web.__qualname__ = original.__qualname__
    return search_web


__all__ = ["GATE_ID", "build_date_gated_search_web"]
