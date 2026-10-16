"""Research pipeline: cleaned summaries only, passages, one-shot execution, and role parameters."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from energy_oil_forecasting.cfm_agent_v_6_0 import llm_roles
from energy_oil_forecasting.cfm_agent_v_6_0.config import CfmV60Settings
from energy_oil_forecasting.cfm_agent_v_6_0.schemas import ActiveSource, SummaryPassage
from energy_oil_forecasting.cfm_agent_v_6_0.tools import research_pipeline
from energy_oil_forecasting.cfm_agent_v_6_0.tools.research_pipeline import (
    ResearchPipelineTool,
)


CLEANED = "Cleaned cutoff-safe summary with enough length to stand alone. Second cleaned sentence follows here for the record."


class Verdict:
    clean = True
    confidence = 10
    flagged_claims = ["removed post-cutoff claim"]
    filtered_text = CLEANED


class FailedVerdict:
    clean = False
    confidence = 3
    flagged_claims = ["leaked claim"]
    filtered_text = "leaked"


def fake_resolve(source_id, rank, grounding_url, provider_title):
    return ActiveSource(
        source_id=source_id,
        rank=rank,
        grounding_url=grounding_url,
        provider_title=provider_title,
        resolved_url="https://www.eia.gov/a",
        resolved_domain="eia.gov",
        publisher="U.S. Energy Information Administration",
        source_quality="tier_1_primary",
        is_primary_or_official=True,
        resolution_status="resolved",
    )


def stubbed_tool(monkeypatch, verdicts=None) -> ResearchPipelineTool:
    """Build a tool whose search, verifier, and resolver never touch the network."""
    tool = ResearchPipelineTool(CfmV60Settings())
    calls = {"verify": 0}

    async def fake_search(_):
        return "Original content", [{"url": "https://redirect.test/a", "title": "Provider"}]

    async def fake_verify(**_):
        index = calls["verify"]
        calls["verify"] += 1
        if verdicts is None:
            return Verdict()
        return verdicts[min(index, len(verdicts) - 1)]()

    monkeypatch.setattr(tool, "_do_search", fake_search)
    monkeypatch.setattr(tool, "_verify_no_leakage", fake_verify)
    monkeypatch.setattr(tool, "_resolve_source", fake_resolve)
    return tool


async def run(tool: ResearchPipelineTool, cutoff: str = "2026-07-28", suffix: str = "") -> dict:
    queries = {area: f"query {area}{suffix}" for area in tool.AREAS}
    return json.loads(
        await tool.run_research_pipeline(
            cutoff,
            queries["underlying_event"],
            queries["physical_flows"],
            queries["supply_response"],
            queries["market_reaction"],
        )
    )


def test_tool_is_constructed_from_settings_only():
    settings = CfmV60Settings()
    tool = ResearchPipelineTool(settings)
    assert tool.search_model == settings.search_model
    with pytest.raises(TypeError):
        ResearchPipelineTool(settings, search_model="test")


@pytest.mark.asyncio
async def test_pipeline_returns_only_cleaned_summaries(monkeypatch):
    tool = stubbed_tool(monkeypatch)
    packet = await run(tool)
    assert len(packet["verified_summaries"]) == 4
    assert all(s["cleaned_summary"] == CLEANED for s in packet["verified_summaries"])
    assert packet["destination_page_content_included"] is False
    assert "passages" not in packet
    assert "removed post-cutoff claim" not in json.dumps(packet)
    assert "Original content" not in json.dumps(packet)
    assert tool.last_verification_audit.queries[0].removed_flagged_claims == ["removed post-cutoff claim"]
    assert tool.last_source_audit.executed is False
    assert tool.last_source_audit.validator_id == "source_validator_v60_audit_only"


@pytest.mark.asyncio
async def test_accepted_summaries_carry_passages_indexed_by_packet_position(monkeypatch):
    tool = stubbed_tool(monkeypatch)
    packet = await run(tool)
    for index, summary in enumerate(packet["verified_summaries"], 1):
        assert summary["status"] == "accepted"
        assert [p["passage_id"] for p in summary["passages"]] == [f"{index:02d}.01", f"{index:02d}.02"]
        assert " ".join(p["text"] for p in summary["passages"]) == CLEANED
    typed = tool.last_result
    assert all(isinstance(p, SummaryPassage) for s in typed.verified_summaries for p in s.passages)


@pytest.mark.asyncio
async def test_failed_closed_summary_has_no_passages_and_keeps_its_position(monkeypatch):
    settings = CfmV60Settings()
    failing = [FailedVerdict] * settings.search_verifier_max_attempts + [Verdict]
    tool = stubbed_tool(monkeypatch, verdicts=failing)
    packet = await run(tool)
    first, second = packet["verified_summaries"][:2]
    assert first["status"] == "failed_closed"
    assert first["passages"] == []
    assert first["cleaned_summary"] == ""
    assert first["verifier_attempt_count"] == settings.search_verifier_max_attempts
    assert second["status"] == "accepted"
    assert [p["passage_id"] for p in second["passages"]] == ["02.01", "02.02"]
    assert packet["warnings"] == ["underlying_event: [SEARCH_VERIFICATION_FAILED]"]
    assert "leaked" not in json.dumps(packet)


@pytest.mark.asyncio
async def test_passages_are_part_of_summary_and_packet_identity(monkeypatch):
    baseline = await run(stubbed_tool(monkeypatch))
    original_split = research_pipeline.split_summary

    def shifted_split(text, *, summary_index, min_chars, abbreviations):
        passages = original_split(text, summary_index=summary_index, min_chars=min_chars, abbreviations=abbreviations)
        return [passages[0].model_copy(update={"text": passages[0].text + " (altered)"}), *passages[1:]]

    monkeypatch.setattr(research_pipeline, "split_summary", shifted_split)
    altered = await run(stubbed_tool(monkeypatch))
    assert altered["verified_summaries"][0]["cleaned_summary"] == baseline["verified_summaries"][0]["cleaned_summary"]
    assert (
        altered["verified_summaries"][0]["verified_summary_id"]
        != baseline["verified_summaries"][0]["verified_summary_id"]
    )
    assert altered["packet_id"] != baseline["packet_id"]


def test_summary_identity_is_stable_and_passage_sensitive():
    passages = [{"passage_id": "01.01", "text": "one"}]
    same = ResearchPipelineTool._summary_identity("area", "q", "2026-07-28", "text", passages)
    again = ResearchPipelineTool._summary_identity("area", "q", "2026-07-28", "text", passages)
    changed = ResearchPipelineTool._summary_identity(
        "area", "q", "2026-07-28", "text", [{"passage_id": "01.01", "text": "two"}]
    )
    assert same == again
    assert same.startswith("verified_summary:sha256:")
    assert same != changed


@pytest.mark.asyncio
async def test_pipeline_is_one_shot_and_resets_between_workflows(monkeypatch):
    tool = stubbed_tool(monkeypatch)
    tool.prepare_workflow("2026-07-28")
    first = await run(tool)
    first_id = first["packet_id"]
    duplicate = await run(tool)
    assert duplicate["status"] == "error"
    assert tool.last_result.packet_id == first_id
    assert tool.audit["successful_execution_count"] == 1
    assert tool.audit["rejected_duplicate_count"] == 1

    tool.prepare_workflow("2026-07-29")
    assert tool.last_result is None
    assert tool.audit["attempt_count"] == 0
    second = await run(tool, cutoff="2026-07-29", suffix=" 2")
    assert second["packet_id"] != first_id
    assert tool.audit["successful_execution_count"] == 1


@pytest.mark.asyncio
async def test_workflow_cutoff_overrides_the_cutoff_argument(monkeypatch):
    tool = stubbed_tool(monkeypatch)
    tool.prepare_workflow("2026-07-28")
    result = await run(tool, cutoff="2026-07-29")
    assert result["cutoff_date"] == "2026-07-28"
    assert all(summary["cutoff_date"] == "2026-07-28" for summary in result["verified_summaries"])
    assert tool.audit["successful_execution_count"] == 1
    assert tool.audit["attempt_errors"] == []


@pytest.mark.asyncio
async def test_harness_cutoff_that_disagrees_with_the_workflow_is_rejected(monkeypatch):
    tool = stubbed_tool(monkeypatch)
    tool.prepare_workflow("2026-07-28")
    context = SimpleNamespace(state={research_pipeline.AS_OF_STATE_KEY: "2026-07-29"})
    result = json.loads(await tool.run_research_pipeline("2026-07-28", "u", "p", "s", "m", tool_context=context))
    assert result["status"] == "error"
    assert tool.last_result is None
    assert tool.audit["successful_execution_count"] == 0
    assert tool.audit["attempt_errors"] == ["research cutoff does not match the harness-controlled workflow cutoff"]


@pytest.mark.asyncio
async def test_pipeline_is_one_shot_and_resets(monkeypatch):
    tool = stubbed_tool(monkeypatch)
    tool.prepare_workflow("2026-07-28")
    first = await run(tool)
    second = await run(tool)
    assert "packet_id" in first
    assert second["status"] == "error"
    assert tool.audit["successful_execution_count"] == 1
    assert tool.audit["rejected_duplicate_count"] == 1
    tool.prepare_workflow("2026-07-29")
    assert tool.last_result is None
    assert tool.audit["attempt_count"] == 0


# ------------------------------------------------------------ role parameters


def fake_response(content: str, *, model: str = "gemini-3.1-flash-lite-preview", grounding: bool = False):
    fields = {}
    if grounding:
        fields = {
            "grounding_metadata": {
                "groundingChunks": [
                    {"web": {"uri": "https://redirect.test/a", "title": "Provider"}},
                    {"web": {"uri": "https://redirect.test/b"}},
                    {"web": {}},
                ]
            }
        }
    choice = SimpleNamespace(message=SimpleNamespace(content=content), provider_specific_fields=fields)
    return SimpleNamespace(model=model, choices=[choice])


@pytest.mark.asyncio
async def test_search_call_uses_the_search_role_parameters(monkeypatch):
    settings = CfmV60Settings()
    captured: dict = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return fake_response("grounded content", grounding=True)

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)
    monkeypatch.setenv(llm_roles.OPENAI_BASE_URL_ENV, "https://proxy.test/v1")
    monkeypatch.setenv(llm_roles.OPENAI_API_KEY_ENV, "secret-key")
    llm_roles.RECORDER.reset()

    content, rows = await ResearchPipelineTool(settings)._do_search("query text")
    assert content == "grounded content"
    assert rows == [
        {"url": "https://redirect.test/a", "title": "Provider"},
        {"url": "https://redirect.test/b", "title": None},
    ]
    assert captured["model"] == "openai/gemini-3.1-flash-lite-preview"
    assert captured["temperature"] == 0.1
    assert captured["max_tokens"] == 4096
    assert captured["timeout"] == 60
    assert captured["top_p"] == 1.0
    assert captured["n"] == 1
    assert captured["frequency_penalty"] == 0.0
    assert captured["presence_penalty"] == 0.0
    assert captured["parallel_tool_calls"] is False
    assert captured["drop_params"] is True
    assert captured["api_base"] == "https://proxy.test/v1"
    assert captured["api_key"] == "secret-key"
    assert captured["tools"] == [{"googleSearch": {}}]
    assert captured["messages"][-1] == {"role": "user", "content": "query text"}

    records = llm_roles.RECORDER.snapshot()["search"]
    assert len(records) == 1
    recorded = records[0]["parameters"]
    assert recorded["model"] == "openai/gemini-3.1-flash-lite-preview"
    assert recorded["temperature"] == 0.1
    assert recorded["max_tokens"] == 4096
    assert recorded["timeout"] == 60
    assert recorded["top_p"] == 1.0
    assert recorded["n"] == 1
    assert recorded["api_key"] == llm_roles.REDACTED
    assert recorded["tool_names"] == ["googleSearch"]
    assert "messages" not in recorded
    assert records[0]["response_model"] == "gemini-3.1-flash-lite-preview"


@pytest.mark.asyncio
async def test_cutoff_verifier_call_uses_the_cutoff_verifier_role_parameters(monkeypatch):
    settings = CfmV60Settings()
    captured: dict = {}
    verdict = {"flagged_claims": ["late claim"], "filtered_text": "kept text", "confidence": 9, "clean": True}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return fake_response(json.dumps(verdict), model="gemini-3.5-flash")

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)
    llm_roles.RECORDER.reset()

    result = await ResearchPipelineTool(settings)._verify_no_leakage(
        text="search result", query="q", cutoff_date="2026-07-28"
    )
    assert (result.clean, result.confidence, result.filtered_text, result.flagged_claims) == (
        True,
        9,
        "kept text",
        ["late claim"],
    )
    assert captured["model"] == "openai/gemini-3.5-flash"
    assert captured["temperature"] == 0.0
    assert captured["max_tokens"] == 4096
    assert captured["timeout"] == 60
    assert captured["top_p"] == 1.0
    assert captured["n"] == 1
    assert "parallel_tool_calls" not in captured
    assert captured["response_format"]["type"] == "json_schema"
    assert "2026-07-28" in captured["messages"][-1]["content"]
    assert "search result" in captured["messages"][-1]["content"]
    records = llm_roles.RECORDER.snapshot()["cutoff_verifier"]
    assert records[0]["parameters"]["model"] == "openai/gemini-3.5-flash"
    assert records[0]["parameters"]["temperature"] == 0.0
    assert records[0]["parameters"]["response_format_type"] == "json_schema"
    assert records[0]["response_model"] == "gemini-3.5-flash"


@pytest.mark.asyncio
async def test_unparseable_verifier_output_is_a_non_clean_verdict(monkeypatch):
    async def fake_acompletion(**_):
        return fake_response("not json", model="gemini-3.5-flash")

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)
    result = await ResearchPipelineTool(CfmV60Settings())._verify_no_leakage(
        text="search result", query="q", cutoff_date="2026-07-28"
    )
    assert result.clean is False
    assert result.confidence == 1
    assert result.filtered_text == "search result"


@pytest.mark.asyncio
async def test_verified_search_retries_up_to_the_settings_cap_with_negative_guidance(monkeypatch):
    settings = CfmV60Settings(search_verifier_max_attempts=2)
    tool = ResearchPipelineTool(settings)
    prompts: list[str] = []

    async def fake_search(user_content):
        prompts.append(user_content)
        return "content", [{"url": "https://redirect.test/a", "title": None}]

    async def fake_verify(**_):
        return FailedVerdict()

    monkeypatch.setattr(tool, "_do_search", fake_search)
    monkeypatch.setattr(tool, "_verify_no_leakage", fake_verify)
    cleaned, rows, attempts = await tool._verified_search("q", "2026-07-28")
    assert cleaned is None
    assert rows == []
    assert [attempt.attempt for attempt in attempts] == [1, 2]
    assert len(prompts) == 2
    assert "leaked claim" in prompts[1]
    assert "leaked claim" not in prompts[0]


@pytest.mark.asyncio
async def test_verified_search_rejects_confidence_below_the_settings_threshold(monkeypatch):
    settings = CfmV60Settings(search_verifier_max_attempts=1)
    tool = ResearchPipelineTool(settings)

    class LowConfidence(Verdict):
        confidence = settings.search_verifier_confidence_threshold - 1

    async def fake_search(_):
        return "content", []

    async def fake_verify(**_):
        return LowConfidence()

    monkeypatch.setattr(tool, "_do_search", fake_search)
    monkeypatch.setattr(tool, "_verify_no_leakage", fake_verify)
    cleaned, _, attempts = await tool._verified_search("q", "2026-07-28")
    assert cleaned is None
    assert attempts[0].clean is True
    assert attempts[0].confidence == settings.search_verifier_confidence_threshold - 1
