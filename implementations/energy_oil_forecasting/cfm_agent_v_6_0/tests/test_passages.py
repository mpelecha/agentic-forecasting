"""Passage extraction edge cases (specification Section 4.1, DD-13)."""

from __future__ import annotations

import pytest
from energy_oil_forecasting.cfm_agent_v_6_0.config import DEFAULT_SETTINGS
from energy_oil_forecasting.cfm_agent_v_6_0.passages import (
    passage_ids_for,
    passage_texts_for,
    split_summary,
)
from energy_oil_forecasting.cfm_agent_v_6_0.schemas import SummaryPassage, VerifiedSummary


MIN_CHARS = DEFAULT_SETTINGS.passage_min_chars
ABBREVIATIONS = DEFAULT_SETTINGS.passage_abbreviations


def split(text: str, *, summary_index: int = 1, min_chars: int = MIN_CHARS, abbreviations=None) -> list[str]:
    passages = split_summary(
        text,
        summary_index=summary_index,
        min_chars=min_chars,
        abbreviations=ABBREVIATIONS if abbreviations is None else abbreviations,
    )
    return [passage.text for passage in passages]


def test_decimal_price_does_not_split():
    text = "WTI settled at $102.58 on Tuesday after a volatile session. Brent closed at $105.10 in London trading."
    assert split(text) == [
        "WTI settled at $102.58 on Tuesday after a volatile session.",
        "Brent closed at $105.10 in London trading.",
    ]


def test_abbreviations_are_protected_case_sensitively():
    text = (
        "U.S. Exports fell sharply during the week according to the agency. "
        "Exxon Mobil Corp. Inc. Reported steady output for the quarter overall. "
        "Deliveries resumed on Sept. 12 after the outage was repaired fully."
    )
    assert split(text) == [
        "U.S. Exports fell sharply during the week according to the agency.",
        "Exxon Mobil Corp. Inc. Reported steady output for the quarter overall.",
        "Deliveries resumed on Sept. 12 after the outage was repaired fully.",
    ]
    lowered = "Shipments via u.s. Ports were rerouted after the storm passed through the gulf."
    assert split(lowered, abbreviations=["U.S."], min_chars=1) == [
        "Shipments via u.s.",
        "Ports were rerouted after the storm passed through the gulf.",
    ]


def test_bullet_markers_are_removed_from_each_line():
    text = (
        "- Exports from the terminal were suspended for the second week running.\n"
        "* Tanker loadings at the port fell by roughly half compared with last month.\n"
        "• Refinery runs in the region were reduced to protect crude inventories.\n"
        "1. Officials expect the suspension to remain in place through the month.\n"
        "2) Traders reported firm bids for prompt cargoes across the regional market."
    )
    assert split(text) == [
        "Exports from the terminal were suspended for the second week running.",
        "Tanker loadings at the port fell by roughly half compared with last month.",
        "Refinery runs in the region were reduced to protect crude inventories.",
        "Officials expect the suspension to remain in place through the month.",
        "Traders reported firm bids for prompt cargoes across the regional market.",
    ]


def test_ellipsis_is_a_single_terminator_run():
    text = "Prices rose sharply through the morning session... Then they fell back after the announcement was made."
    assert split(text) == [
        "Prices rose sharply through the morning session...",
        "Then they fell back after the announcement was made.",
    ]


def test_closing_quotes_and_brackets_stay_with_the_sentence():
    text = (
        'The minister said "flows have stopped for the time being." Officials confirmed the statement later on. '
        "The company (listed in New York) confirmed the outage lasted several days. (Its shares fell sharply afterwards in New York trading.) "
        "Analysts expect a further update within the coming days."
    )
    assert split(text) == [
        'The minister said "flows have stopped for the time being."',
        "Officials confirmed the statement later on.",
        "The company (listed in New York) confirmed the outage lasted several days.",
        "(Its shares fell sharply afterwards in New York trading.)",
        "Analysts expect a further update within the coming days.",
    ]


def test_no_split_when_lowercase_follows_the_terminator():
    text = "The agency cut its forecast by 0.2 mb/d. however traders shrugged off the revision entirely on the day."
    assert split(text) == [text]


def test_short_fragment_merges_forward_and_short_tail_merges_backward():
    text = (
        "Short one. Then a much longer sentence follows here to absorb the short fragment. "
        "A final long sentence precedes a very short tail fragment at the end. Tail."
    )
    assert split(text) == [
        "Short one. Then a much longer sentence follows here to absorb the short fragment.",
        "A final long sentence precedes a very short tail fragment at the end. Tail.",
    ]


def test_consecutive_short_fragments_keep_merging_forward():
    text = "One. Two. Three. Then a much longer sentence follows here to absorb all the fragments."
    assert split(text) == [text]


def test_single_sentence_summary_yields_one_passage():
    assert split("Exports were suspended for the second week running at the terminal.") == [
        "Exports were suspended for the second week running at the terminal."
    ]


def test_summary_shorter_than_minimum_yields_exactly_one_passage():
    assert split("Short.") == ["Short."]


def test_empty_and_whitespace_only_summaries_yield_no_passages():
    assert split("") == []
    assert split(" \n\t\r\n ") == []
    assert split("- \n* ") == []


def test_normalisation_of_line_endings_and_whitespace():
    text = "First  sentence\twith   gaps here to be long enough.\r\nSecond line\rThird line with more text here."
    assert split(text, min_chars=1) == [
        "First sentence with gaps here to be long enough.",
        "Second line",
        "Third line with more text here.",
    ]


def test_passage_ids_are_zero_padded_and_one_based():
    text = "First sentence of the summary is long enough to stand alone. Second sentence is also long enough here."
    first = split_summary(text, summary_index=1, min_chars=MIN_CHARS, abbreviations=ABBREVIATIONS)
    twelfth = split_summary(text, summary_index=12, min_chars=MIN_CHARS, abbreviations=ABBREVIATIONS)
    assert [passage.passage_id for passage in first] == ["01.01", "01.02"]
    assert [passage.passage_id for passage in twelfth] == ["12.01", "12.02"]
    assert all(isinstance(passage, SummaryPassage) for passage in first)


def test_summary_index_must_be_one_based():
    with pytest.raises(ValueError):
        split_summary("x", summary_index=0, min_chars=MIN_CHARS, abbreviations=ABBREVIATIONS)


def test_split_is_deterministic():
    text = (
        "- WTI settled at $102.58 on Tuesday, up 3.1% on the session. U.S. exports fell sharply.\n"
        "* Prices rose... Then fell again after the announcement. Short. Then a longer sentence follows here."
    )
    first = split_summary(text, summary_index=3, min_chars=MIN_CHARS, abbreviations=ABBREVIATIONS)
    second = split_summary(text, summary_index=3, min_chars=MIN_CHARS, abbreviations=ABBREVIATIONS)
    assert first == second
    assert [passage.model_dump() for passage in first] == [passage.model_dump() for passage in second]


def test_packet_fixture_passages_match_the_splitter(packet):
    summary = packet.verified_summaries[0]
    recomputed = split_summary(
        summary.cleaned_summary, summary_index=1, min_chars=MIN_CHARS, abbreviations=ABBREVIATIONS
    )
    assert recomputed == summary.passages


def test_passage_lookup_helpers(packet):
    summary = packet.verified_summaries[0]
    assert passage_ids_for(summary) == {"01.01", "01.02"}
    texts = passage_texts_for([summary], ["01.02", "99.01"])
    assert texts == {"01.02": summary.passages[1].text}
    empty = VerifiedSummary(
        verified_summary_id="verified_summary:sha256:none",
        query_area="market_reaction",
        query="q",
        cutoff_date="2026-07-28",
        status="failed_closed",
    )
    assert passage_ids_for(empty) == set()
