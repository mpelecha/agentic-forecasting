import pytest
from energy_oil_forecasting.cfm_agent_v_5_0.schemas import (
    ActiveSource,
    ResearchPacket,
    ResearchQuery,
    VerifiedSummary,
)


@pytest.fixture
def packet():
    sources = [
        ActiveSource(
            source_id="source_001",
            rank=1,
            grounding_url="https://redirect.test/1",
            resolved_url="https://www.eia.gov/a",
            resolved_domain="eia.gov",
            publisher="U.S. Energy Information Administration",
            source_quality="tier_1_primary",
            is_primary_or_official=True,
            resolution_status="resolved",
        ),
        ActiveSource(
            source_id="source_002",
            rank=2,
            grounding_url="https://redirect.test/2",
            resolved_url="https://www.reuters.com/a",
            resolved_domain="reuters.com",
            publisher="Reuters",
            source_quality="tier_2_independent",
            resolution_status="resolved",
        ),
        ActiveSource(
            source_id="source_003",
            rank=3,
            grounding_url="https://redirect.test/3",
            resolved_url="https://www.bloomberg.com/a",
            resolved_domain="bloomberg.com",
            publisher="Bloomberg",
            source_quality="tier_2_independent",
            resolution_status="resolved",
        ),
    ]
    summary = VerifiedSummary(
        verified_summary_id="verified_summary:sha256:test",
        query_area="physical_flows",
        query="test",
        cutoff_date="2026-07-28",
        status="accepted",
        cleaned_summary="Confirmed physical oil supply information with market relevance.",
        associated_source_ids=[source.source_id for source in sources],
    )
    return ResearchPacket(
        packet_id="research_packet:sha256:test",
        cutoff_date="2026-07-28",
        queries=[ResearchQuery(area="physical_flows", query="test")],
        verified_summaries=[summary],
        sources=sources,
    )
