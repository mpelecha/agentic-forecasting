import json
from datetime import date
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest
from energy_oil_forecasting.cfm_agent_v_5_2.config import CfmV52Settings
from energy_oil_forecasting.cfm_agent_v_5_2.schemas import ActiveSource
from energy_oil_forecasting.cfm_agent_v_5_2_dategated import date_gate, page_gate
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

    def __init__(self, text: str) -> None:
        self.filtered_text = text


EARLY, LATE = "https://site.test/early", "https://site.test/late"
SEGMENTS = [
    {"segment": {"text": "Early fact."}, "groundingChunkIndices": [0]},
    {"segment": {"text": "Late fact."}, "groundingChunkIndices": [1]},
    {"segment": {"text": "Mixed fact."}, "groundingChunkIndices": [0, 1]},
    {"segment": {"text": "Ungrounded fact."}, "groundingChunkIndices": []},
]


def _gated_tool(monkeypatch, page_dates):
    """Build a gated tool whose search grounds ``SEGMENTS`` on an early and a late page."""
    tool = DateGatedResearchPipelineTool(CfmV52Settings(), search_model="test")
    verified: list[str] = []

    async def fake_search(_user_content):
        rows = [
            {"url": "https://redirect.test/early", "title": None, "chunk": 0},
            {"url": "https://redirect.test/late", "title": None, "chunk": 1},
        ]
        return "Early fact. Late fact. Mixed fact. Ungrounded fact.", rows, SEGMENTS

    async def fake_verify(*, text, **_):
        verified.append(text)
        return _Verdict(text)

    monkeypatch.setattr(tool, "_grounded_search", fake_search)
    monkeypatch.setattr(
        "energy_oil_forecasting.cfm_agent_v_5_2.tools.research_pipeline._verify_no_leakage", fake_verify
    )
    monkeypatch.setattr(
        tool.gate, "_resolve_uncached", lambda url, title: _source(url.replace("redirect", "site"), url)
    )
    monkeypatch.setattr(date_gate, "_inspect_source", lambda source, *_: _page(page_dates.get(source.resolved_url))())
    return tool, verified


async def _run(tool):
    tool.prepare_workflow(CUTOFF)
    return json.loads(await tool.run_research_pipeline(CUTOFF, "q1", "q2", "q3", "q4"))


@pytest.mark.asyncio
async def test_only_text_grounded_solely_in_early_pages_reaches_the_agent(monkeypatch):
    tool, verified = _gated_tool(monkeypatch, page_dates={EARLY: "2025-07-01", LATE: "2026-08-20"})
    packet = await _run(tool)

    assert all(s["status"] == "accepted" for s in packet["verified_summaries"])
    # Segments citing the late page (alone or with the early one) and ungrounded text are dropped,
    # before the LLM verifier sees anything.
    assert verified == ["Early fact."] * 4
    assert {s["cleaned_summary"] for s in packet["verified_summaries"]} == {"Early fact."}
    assert {s["resolved_url"] for s in packet["sources"]} == {EARLY}
    attempt = tool.audit["date_gate"]["attempts"][0]
    assert (attempt["outcome"], attempt["segments_kept"], attempt["segments_dropped"]) == ("accepted", 1, 3)


@pytest.mark.asyncio
async def test_a_query_with_no_provably_early_text_fails_closed(monkeypatch):
    tool, verified = _gated_tool(monkeypatch, page_dates={})  # every page undated
    packet = await _run(tool)

    assert all(s["status"] == "failed_closed" for s in packet["verified_summaries"])
    assert packet["sources"] == []
    assert verified == []
    attempts_per_query = CfmV52Settings().search_verifier_max_attempts
    outcomes = [a["outcome"] for a in tool.audit["date_gate"]["attempts"]]
    assert outcomes == ["no_dated_text"] * (4 * attempts_per_query)


def test_redirect_is_read_without_fetching_the_destination(monkeypatch):
    fetched: list[str] = []

    class _Opener:
        def open(self, request, timeout):
            fetched.append(request.full_url)
            raise HTTPError(request.full_url, 302, "Found", {"Location": "https://news.test/2025/07/01/a"}, None)

    monkeypatch.setattr(page_gate, "build_opener", lambda *_: _Opener())
    monkeypatch.setattr(page_gate, "validate_public_url", lambda url: None)
    tool = DateGatedResearchPipelineTool(CfmV52Settings(), search_model="test")
    grounding = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/abc"

    source = tool._resolve_source("source_001", 1, grounding, "news.test")

    assert source.resolved_url == "https://news.test/2025/07/01/a"
    assert fetched == [grounding]  # the news site itself, which may refuse bots, is never requested
