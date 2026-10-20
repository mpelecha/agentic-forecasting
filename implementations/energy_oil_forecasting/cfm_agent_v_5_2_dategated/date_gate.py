"""Decide whether a web source is provably published strictly before the cutoff.

A search result reaches the agent as summary text written from its grounding
pages, so a page published after the cutoff can leak into the summary even if
the page itself is never shown. The date gate judges every grounding page; the
research tool then keeps only the sentences supported solely by pages that pass.

A page passes only with positive evidence of an earlier date and no evidence of
a later one:

* every date found — in the URL (``/2025/10/14/``, ``...-june-24-2025``,
  StatCan's ``daily-quotidien/YYMMDD``) or on the page (HTML metadata, JSON-LD,
  visible "Published" text, read with v5.2's own audit extractor) — is strictly
  before the cutoff;
* at least one such date exists (an undated page fails unless ``allow_undated``);
* the page is not a live page whose content changes over time (quote pages,
  "latest" reports), which shows today's content whatever its date says;
* the grounding redirect resolves to a public URL.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Callable

from energy_oil_forecasting.cfm_agent_v_5_2.config import CfmV52Settings
from energy_oil_forecasting.cfm_agent_v_5_2.schemas import ActiveSource
from energy_oil_forecasting.cfm_agent_v_5_2.source_validator import _inspect_source


_MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
_MONTH_NUM = {m: i + 1 for i, m in enumerate(_MONTHS.split("|"))}

ACCEPTED = "before_cutoff"
REJECT_LATE = "on_or_after_cutoff"
REJECT_UNDATED = "undated"
REJECT_LIVE = "live_page"
REJECT_UNRESOLVED = "unresolved"


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def url_dates(url: str | None) -> list[date]:
    """Return the dates written in ``url``."""
    if not url:
        return []
    u = url.lower()
    found = [_safe_date(int(y), int(m), int(d)) for y, m, d in re.findall(r"(20\d\d)[/-](\d\d)[/-](\d\d)", u)]
    found += [
        _safe_date(int(y), _MONTH_NUM[m], int(d)) for m, d, y in re.findall(rf"({_MONTHS})-(\d{{1,2}})-(20\d\d)", u)
    ]
    found += [
        _safe_date(2000 + int(s[:2]), int(s[2:4]), int(s[4:])) for s in re.findall(r"daily-quotidien/(\d{6})/", u)
    ]
    return [d for d in found if d is not None]


@dataclass(frozen=True)
class SourceDating:
    """The gate's verdict on one grounding page, kept in the run's audit."""

    grounding_url: str
    resolved_url: str | None
    decision: str
    url_dates: list[str] = field(default_factory=list)
    page_date: str | None = None
    page_date_from: str = "missing"
    note: str = ""

    @property
    def accepted(self) -> bool:
        return self.decision == ACCEPTED

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "accepted": self.accepted}


def date_source(
    source: ActiveSource,
    cutoff_date: str,
    settings: CfmV52Settings,
    *,
    allow_undated: bool = False,
    inspect: Callable[[ActiveSource, str, CfmV52Settings], Any] | None = None,
) -> SourceDating:
    """Judge one resolved grounding page against ``cutoff_date`` (YYYY-MM-DD).

    ``inspect`` fetches and reads the page; it defaults to v5.2's audit extractor.
    """
    inspect = inspect or _inspect_source
    cutoff = date.fromisoformat(cutoff_date)
    if not source.resolved_url:
        return SourceDating(source.grounding_url, None, REJECT_UNRESOLVED, note=source.resolution_warning or "")

    in_url = url_dates(source.resolved_url)
    record = inspect(source, cutoff_date, settings)
    page = date.fromisoformat(record.publication_date) if record.publication_date else None
    known = [*in_url, *([page] if page else [])]

    def verdict(decision: str, note: str = "") -> SourceDating:
        return SourceDating(
            grounding_url=source.grounding_url,
            resolved_url=source.resolved_url,
            decision=decision,
            url_dates=[d.isoformat() for d in in_url],
            page_date=page.isoformat() if page else None,
            page_date_from=record.publication_date_provenance,
            note=note or "; ".join(record.extraction_warnings)[:300],
        )

    late = [d for d in known if d >= cutoff]
    if late:
        return verdict(REJECT_LATE, f"dated {max(late).isoformat()}, cutoff {cutoff_date}")
    if record.potentially_mutable:
        return verdict(REJECT_LIVE, "live page: content can change after the cutoff")
    if not known and not allow_undated:
        return verdict(REJECT_UNDATED, "no publication date in the URL or on the page")
    return verdict(ACCEPTED)
