"""Passage extraction from verifier-cleaned summaries (specification Section 4.1, DD-13).

A passage is a sentence-level slice of an accepted summary. Claims cite passages
by id, the evidence policy checks that every cited passage belongs to a linked
summary, and the claim-support verifier audits each claim against its cited
passages only. Extraction is a pure function of the cleaned text and two
settings (``passage_min_chars`` and ``passage_abbreviations``); the same input
always yields the same passages and ids.

The six steps, in order: normalise, segment by line with bullet stripping, split
inside a segment after a sentence terminator, clean, merge short fragments, and
assign ``SS.NN`` ids.
"""

from __future__ import annotations

import re

from energy_oil_forecasting.cfm_agent_v_6_0.schemas import SummaryPassage, VerifiedSummary


#: Leading bullet markers removed from a line segment (step 2): a dash or asterisk
#: followed by whitespace, the bullet character, or digits followed by "." or ")"
#: and whitespace. A marker at the end of an otherwise empty line is also removed.
_BULLET = re.compile(r"^(?:[-*](?:\s+|$)|•\s*|\d+[.)](?:\s+|$))")

#: A sentence terminator run, any closing quotes or brackets, and the whitespace
#: that follows (step 3). The character after the whitespace is checked in code.
_TERMINATOR = re.compile(r"[.!?]+[\"'”’)\]]*(\s+)")
_OPENING = frozenset("\"'“‘([")
_HORIZONTAL_WHITESPACE = re.compile(r"[ \t]+")


def _normalise(text: str) -> list[str]:
    """Step 1: unify line endings, collapse horizontal whitespace, strip each line."""
    unified = text.replace("\r\n", "\n").replace("\r", "\n")
    collapsed = _HORIZONTAL_WHITESPACE.sub(" ", unified)
    return [line.strip() for line in collapsed.split("\n")]


def _segments(lines: list[str]) -> list[str]:
    """Step 2: every non-empty line is a segment with its leading bullet marker removed."""
    return [_BULLET.sub("", line, count=1) for line in lines if line]


def _starts_new_sentence(char: str) -> bool:
    return char.isupper() or char.isdigit() or char in _OPENING


def _token_ending_at(segment: str, end: int) -> str:
    """Return the whitespace-delimited token that ends at position ``end`` (exclusive)."""
    start = end
    while start > 0 and not segment[start - 1].isspace():
        start -= 1
    return segment[start:end]


def _split_segment(segment: str, abbreviations: frozenset[str]) -> list[str]:
    """Step 3: split after a terminator run when a new sentence follows and no abbreviation protects it."""
    candidates: list[str] = []
    start = 0
    for match in _TERMINATOR.finditer(segment):
        whitespace_start = match.start(1)
        following = match.end()
        if following >= len(segment) or not _starts_new_sentence(segment[following]):
            continue
        terminator_end = whitespace_start
        while terminator_end > match.start() and segment[terminator_end - 1] not in ".!?":
            terminator_end -= 1
        if _token_ending_at(segment, terminator_end) in abbreviations:
            continue
        candidates.append(segment[start:whitespace_start])
        start = following
    candidates.append(segment[start:])
    return candidates


def _clean(candidates: list[str]) -> list[str]:
    """Step 4: strip each candidate and drop the empty ones."""
    stripped = (candidate.strip() for candidate in candidates)
    return [candidate for candidate in stripped if candidate]


def _merge_short(candidates: list[str], min_chars: int) -> list[str]:
    """Step 5: join a short candidate forward; join a short final candidate backward."""
    merged: list[str] = []
    pending: str | None = None
    for candidate in candidates:
        current = candidate if pending is None else f"{pending} {candidate}"
        pending = None
        if len(current) < min_chars:
            pending = current
        else:
            merged.append(current)
    if pending is not None:
        if merged:
            merged[-1] = f"{merged[-1]} {pending}"
        else:
            merged.append(pending)
    return merged


def split_summary(
    text: str,
    *,
    summary_index: int,
    min_chars: int,
    abbreviations: list[str],
) -> list[SummaryPassage]:
    """Split one cleaned summary into passages with ``SS.NN`` ids.

    Parameters
    ----------
    text
        The verifier-cleaned summary.
    summary_index
        1-based index of the summary in packet order; becomes the ``SS`` part.
    min_chars
        Shortest passage kept without merging (``passage_min_chars``).
    abbreviations
        Tokens after which no split occurs, matched case-sensitively against the
        whitespace-delimited token ending in the terminator (``passage_abbreviations``).
    """
    if summary_index < 1:
        raise ValueError("summary_index is 1-based")
    protected = frozenset(abbreviations)
    candidates: list[str] = []
    for segment in _segments(_normalise(text)):
        candidates.extend(_split_segment(segment, protected))
    merged = _merge_short(_clean(candidates), min_chars)
    return [
        SummaryPassage(passage_id=f"{summary_index:02d}.{number:02d}", text=passage)
        for number, passage in enumerate(merged, 1)
    ]


def passage_ids_for(summary: VerifiedSummary) -> set[str]:
    """Return the passage ids carried by one summary."""
    return {passage.passage_id for passage in summary.passages}


def passage_texts_for(summaries: list[VerifiedSummary], passage_ids: list[str]) -> dict[str, str]:
    """Map each requested passage id to its text, looking only at ``summaries``."""
    by_id = {passage.passage_id: passage.text for summary in summaries for passage in summary.passages}
    return {passage_id: by_id[passage_id] for passage_id in passage_ids if passage_id in by_id}


__all__ = ["passage_ids_for", "passage_texts_for", "split_summary"]
