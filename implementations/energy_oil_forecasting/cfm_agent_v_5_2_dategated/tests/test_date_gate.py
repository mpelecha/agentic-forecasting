import json
from datetime import date
from types import SimpleNamespace

import pytest
from energy_oil_forecasting.cfm_agent_v_5_2.config import CfmV52Settings
from energy_oil_forecasting.cfm_agent_v_5_2.schemas import ActiveSource
from energy_oil_forecasting.cfm_agent_v_5_2_dategated import date_gate
from energy_oil_forecasting.cfm_agent_v_5_2_dategated.date_gate import date_source, url_dates
from energy_oil_forecasting.cfm_agent_v_5_2_dategated.research import DateGatedResearchPipelineTool


CUTOFF = "2025-07-07"


def _source(url: str | None, grounding: str = "https://redirect.test/x") -> ActiveSource:
    return ActiveSource(
        source_id="source_001",
        rank=1,
        grounding_url=grounding,
        resolved_url=url,
        resolution_status="resolved" if url else "unresolved",
    )


def _page(publication_date=None, mutable=False):
    return lambda *_: SimpleNamespace(
        publication_date=publication_date,
        publication_date_provenance="html_metadata" if publication_date else "missing",
        potentially_mutable=mutable,
        extraction_warnings=[],
    )


def test_url_dates_reads_the_formats_seen_in_v52_sources():
    assert url_dates("https://boereport.com/2025/07/11/july-overview/") == [date(2025, 7, 11)]
    assert url_dates("https://www.forex.com/x/crude-oil-forecast-technical-tuesday-june-24-2025/") == [
        date(2025, 6, 24)
    ]
    assert url_dates("https://www150.statcan.gc.ca/n1/daily-quotidien/260820/dq260820a-eng.htm") == [date(2026, 8, 20)]
    assert url_dates("https://example.com/2025/13/40/not-a-date") == []


@pytest.mark.parametrize(
    ("url", "page", "allow_undated", "decision"),
    [
        # A date in the URL on or after the cutoff rejects, even when the page claims an earlier date.
        ("https://x.test/2025/07/07/a", _page("2025-07-01"), False, date_gate.REJECT_LATE),
        ("https://x.test/a", _page("2026-08-20"), False, date_gate.REJECT_LATE),
        ("https://x.test/2025/07/01/a", _page(mutable=True), False, date_gate.REJECT_LIVE),
        ("https://x.test/a", _page(), False, date_gate.REJECT_UNDATED),
        ("https://x.test/a", _page(), True, date_gate.ACCEPTED),
        ("https://x.test/2025/07/03/a", _page(), False, date_gate.ACCEPTED),
        (None, _page("2025-01-01"), False, date_gate.REJECT_UNRESOLVED),
    ],
)
def test_date_source_needs_an_earlier_date_and_no_later_one(url, page, allow_undated, decision):
    verdict = date_source(_source(url), CUTOFF, CfmV52Settings(), allow_undated=allow_undated, inspect=page)
    assert verdict.decision == decision


class _Verdict:
    clean = True
    confidence = 10
    flagged_claims: list[str] = []
    filtered_text = "Cleaned summary."


def _gated_tool(monkeypatch, searches, page_dates):
    """Build a gated tool whose search returns ``searches`` in turn and whose pages carry ``page_dates``."""
    tool = DateGatedResearchPipelineTool(CfmV52Settings(), search_model="test")
    prompts: list[str] = []

    async def fake_search(user_content):
        prompts.append(user_content)
        return "content", [{"url": u, "title": None} for u in searches[min(len(prompts), len(searches)) - 1]]

    async def fake_verify(**_):
        return _Verdict()

    def fake_resolve(source_id, rank, grounding_url, provider_title):
        return _source(grounding_url.replace("redirect", "site"), grounding_url).model_copy(
            update={"source_id": source_id, "rank": rank}
        )

    monkeypatch.setattr(tool, "_do_search", fake_search)
    monkeypatch.setattr(
        "energy_oil_forecasting.cfm_agent_v_5_2.tools.research_pipeline._verify_no_leakage", fake_verify
    )
    monkeypatch.setattr(
        "energy_oil_forecasting.cfm_agent_v_5_2.tools.research_pipeline.ResearchPipelineTool._resolve_source",
        lambda self, *a: fake_resolve(*a),
    )
    monkeypatch.setattr(
        date_gate,
        "_inspect_source",
        lambda source, *_: _page(page_dates.get(source.resolved_url))(),
    )
    return tool, prompts


async def _run(tool):
    tool.prepare_workflow(CUTOFF)
    return json.loads(await tool.run_research_pipeline(CUTOFF, "q1", "q2", "q3", "q4"))


@pytest.mark.asyncio
async def test_a_late_page_discards_the_whole_result_and_retries_without_it(monkeypatch):
    early, late = "https://site.test/early", "https://site.test/late"
    tool, prompts = _gated_tool(
        monkeypatch,
        searches=[["https://redirect.test/early", "https://redirect.test/late"], ["https://redirect.test/early"]],
        page_dates={early: "2025-07-01", late: "2026-08-20"},
    )
    packet = await _run(tool)

    assert all(s["status"] == "accepted" for s in packet["verified_summaries"])
    assert {s["resolved_url"] for s in packet["sources"]} == {early}  # the late page never reaches the agent
    assert late in prompts[1] and "must not be used" in prompts[1]
    outcomes = [a["outcome"] for a in tool.audit["date_gate"]["attempts"]]
    assert outcomes[:2] == ["date_rejected", "accepted"]


@pytest.mark.asyncio
async def test_a_query_with_no_provably_early_result_fails_closed(monkeypatch):
    tool, _ = _gated_tool(
        monkeypatch,
        searches=[["https://redirect.test/undated"]],
        page_dates={},  # every page undated
    )
    packet = await _run(tool)

    assert all(s["status"] == "failed_closed" for s in packet["verified_summaries"])
    assert packet["sources"] == []
    attempts_per_query = CfmV52Settings().search_verifier_max_attempts
    assert len(tool.audit["date_gate"]["attempts"]) == 4 * attempts_per_query
