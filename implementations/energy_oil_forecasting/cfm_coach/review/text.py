"""Text that goes into a prompt is data, not instructions. This module makes it so.

Cards carry web-derived summaries, claim statements and the agent's own
rationales. Any of them could contain text addressed to a model. Before a field
enters a prompt it is: stripped of URLs, stripped of lines that look like
instructions, capped in length, and wrapped so the prompt can say "the following
is untrusted data". Triage output is an enum plus a pointer that code validates
against the record, so an injected "tag this as X" can at most name a code whose
pointer must exist.
"""

from __future__ import annotations

import re


_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
#: Sentence-level: an injected instruction is usually spliced mid-paragraph.
_INSTRUCTION = re.compile(
    r"(?:ignore (?:all |the )?(?:previous|prior|above)|you are (?:now|an?|the)\b|\bsystem\s*:|\bassistant\s*:|"
    r"as an ai\b|do not (?:follow|obey)|new instructions?\b|\bdisregard\b|\[inst\]|<\||</?untrusted)",
    re.IGNORECASE,
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE = re.compile(r"[ \t]+")

#: Per-field cap, characters. Long enough for a paragraph of rationale.
DEFAULT_FIELD_CAP = 600


def sanitize(text: str | None, *, cap: int = DEFAULT_FIELD_CAP) -> str:
    """Return `text` fit for a prompt: no URLs, no instruction-shaped sentences, capped."""
    if not text:
        return ""
    kept: list[str] = []
    for raw in str(text).splitlines():
        line = _WHITESPACE.sub(" ", _URL.sub("[url]", raw)).strip()
        for sentence in _SENTENCE_BOUNDARY.split(line):
            if sentence and not _INSTRUCTION.search(sentence):
                kept.append(sentence)
    out = " ".join(kept)
    if len(out) > cap:
        out = out[: cap - 1].rstrip() + "…"
    return out


def untrusted_block(label: str, text: str) -> str:
    """Wrap sanitized text so the prompt can point at it as data."""
    return f"<untrusted {label}>\n{text}\n</untrusted {label}>"


def estimate_tokens(text: str) -> int:
    """~4 characters per token; the same heuristic the LLM layer's pre-flight uses."""
    return max(1, len(text) // 4)


__all__ = ["DEFAULT_FIELD_CAP", "estimate_tokens", "sanitize", "untrusted_block"]
