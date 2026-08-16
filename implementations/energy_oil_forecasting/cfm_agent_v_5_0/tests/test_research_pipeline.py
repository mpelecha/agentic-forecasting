import json

import pytest
from energy_oil_forecasting.cfm_agent_v_5_0.config import CfmV50Settings
from energy_oil_forecasting.cfm_agent_v_5_0.schemas import ActiveSource
from energy_oil_forecasting.cfm_agent_v_5_0.tools.research_pipeline import (
    ResearchPipelineTool,
)


class Verdict:
    clean = True
    confidence = 10
    flagged_claims = ["removed post-cutoff claim"]
    filtered_text = "Cleaned cutoff-safe summary."


@pytest.mark.asyncio
async def test_pipeline_returns_only_cleaned_summaries(monkeypatch):
    tool = ResearchPipelineTool(CfmV50Settings(), search_model="test")

    async def fake_search(_):
        return "Original content", [{"url": "https://redirect.test/a", "title": "Provider"}]

    async def fake_verify(**_):
        return Verdict()

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

    monkeypatch.setattr(tool, "_do_search", fake_search)
    monkeypatch.setattr(
        "energy_oil_forecasting.cfm_agent_v_5_0.tools.research_pipeline._verify_no_leakage",
        fake_verify,
    )
    monkeypatch.setattr(tool, "_resolve_source", fake_resolve)
    queries = {area: f"query {area}" for area in tool.AREAS}
    packet = json.loads(
        await tool.run_research_pipeline(
            "2026-07-28",
            queries["underlying_event"],
            queries["physical_flows"],
            queries["supply_response"],
            queries["market_reaction"],
        )
    )
    assert len(packet["verified_summaries"]) == 4
    assert all(s["cleaned_summary"] == "Cleaned cutoff-safe summary." for s in packet["verified_summaries"])
    assert packet["destination_page_content_included"] is False
    assert "passages" not in packet
    assert "removed post-cutoff claim" not in json.dumps(packet)
    assert tool.last_verification_audit.queries[0].removed_flagged_claims == ["removed post-cutoff claim"]
    assert tool.last_source_audit.executed is False


@pytest.mark.asyncio
async def test_pipeline_is_one_shot_and_resets_between_workflows(monkeypatch):
    tool = ResearchPipelineTool(CfmV50Settings(), search_model="test")

    async def fake_search(_):
        return "Original content", [{"url": "https://redirect.test/a", "title": "Provider"}]

    async def fake_verify(**_):
        return Verdict()

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

    monkeypatch.setattr(tool, "_do_search", fake_search)
    monkeypatch.setattr(
        "energy_oil_forecasting.cfm_agent_v_5_0.tools.research_pipeline._verify_no_leakage",
        fake_verify,
    )
    monkeypatch.setattr(tool, "_resolve_source", fake_resolve)

    tool.prepare_workflow("2026-07-28")
    first = json.loads(await tool.run_research_pipeline("2026-07-28", "u", "p", "s", "m"))
    first_id = first["packet_id"]
    duplicate = json.loads(await tool.run_research_pipeline("2026-07-28", "u", "p", "s", "m"))
    assert duplicate["status"] == "error"
    assert tool.last_result.packet_id == first_id
    assert tool.audit["successful_execution_count"] == 1
    assert tool.audit["rejected_duplicate_count"] == 1

    tool.prepare_workflow("2026-07-29")
    assert tool.last_result is None
    assert tool.audit["attempt_count"] == 0
    second = json.loads(await tool.run_research_pipeline("2026-07-29", "u2", "p2", "s2", "m2"))
    assert second["packet_id"] != first_id
    assert tool.audit["successful_execution_count"] == 1


@pytest.mark.asyncio
async def test_pipeline_is_one_shot_and_resets(monkeypatch):
    tool = ResearchPipelineTool(CfmV50Settings(), search_model="test")

    async def fake_search(_):
        return "Original content", [{"url": "https://redirect.test/a", "title": "Provider"}]

    async def fake_verify(**_):
        return Verdict()

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

    monkeypatch.setattr(tool, "_do_search", fake_search)
    monkeypatch.setattr(
        "energy_oil_forecasting.cfm_agent_v_5_0.tools.research_pipeline._verify_no_leakage",
        fake_verify,
    )
    monkeypatch.setattr(tool, "_resolve_source", fake_resolve)
    tool.prepare_workflow("2026-07-28")
    args = ("2026-07-28", "u", "p", "s", "m")
    first = json.loads(await tool.run_research_pipeline(*args))
    second = json.loads(await tool.run_research_pipeline(*args))
    assert "packet_id" in first
    assert second["status"] == "error"
    assert tool.audit["successful_execution_count"] == 1
    assert tool.audit["rejected_duplicate_count"] == 1
    tool.prepare_workflow("2026-07-29")
    assert tool.last_result is None
    assert tool.audit["attempt_count"] == 0
