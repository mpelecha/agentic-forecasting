"""Decide whether a web source is provably published strictly before the cutoff.

A search result reaches the agent as summary text written from its grounding
pages, so a page published after the cutoff can leak into the summary even if
the page itself is never shown. The date gate judges every grounding page; the
research tool then keeps only the sentences supported solely by pages that pass.

A page passes only with positive evidence of an earlier date and no evidence of
a later one:

* every date found is strictly before the cutoff. Dates come from the URL
  (``/2025/10/14/``, ``...-june-24-2025``, StatCan's ``daily-quotidien/YYMMDD``,
  S&P Global's ``/061025-`` (MMDDYY), file names like ``17JUNE2025``), from the page
  (HTML metadata, JSON-LD, visible "Published" text, read with v5.2's own audit
  extractor), and for PDFs from the file's own creation and modification dates;
* at least one date counts as positive evidence (an undated page fails unless
  ``allow_undated``). Two kinds of evidence can only reject, never pass: a month
  without a day in the URL (``may-2025``, ``/2025/05/``, ``may25.pdf``), because it
  may name the period a later review covers; and a PDF date less than
  ``PDF_MARGIN_DAYS`` before the cutoff, because reports are often created days
  before they are released;
* the page is not a live page whose content changes over time (quote pages,
  "latest" reports), which shows today's content whatever its date says;
* the grounding redirect resolves to a public URL.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from typing import Any, Callable
from urllib.parse import urlparse
from urllib.request import Request, build_opener

from energy_oil_forecasting.cfm_agent_v_5_2.config import CfmV52Settings
from energy_oil_forecasting.cfm_agent_v_5_2.schemas import ActiveSource
from energy_oil_forecasting.cfm_agent_v_5_2.source_validator import _inspect_source
from energy_oil_forecasting.cfm_agent_v_5_2.tools.url_safety import SafeRedirectHandler, validate_public_url


_MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
_MONTH_NUM = {m: i + 1 for i, m in enumerate(_MONTHS.split("|"))}
_MON3 = "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
_MON3_NUM = {m: i + 1 for i, m in enumerate(_MON3.split("|"))}
_ANY_MONTH = f"{_MONTHS}|{_MON3}"
PDF_MARGIN_DAYS = 7
PDF_MAX_BYTES = 20_000_000


def _month_num(name: str) -> int:
    return _MONTH_NUM.get(name) or _MON3_NUM[name[:3]]


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
    # S&P Global article slugs: /061025-crude-price-... is MMDDYY. Restricted to that site,
    # where the format is known: elsewhere a six-digit ID could be misread as an early date.
    if "spglobal.com" in (urlparse(u).hostname or ""):
        found += [
            _safe_date(2000 + int(y), int(m), int(d))
            for m, d, y in re.findall(r"/(\d{2})(\d{2})(\d{2})-", u)
            if 20 <= int(y) <= 39
        ]
    # File names such as -17JUNE2025_OilMarketReport.
    found += [
        _safe_date(int(y), _month_num(m), int(d))
        for d, m, y in re.findall(rf"(?<![0-9])(\d{{1,2}})[-_ ]?({_ANY_MONTH})[-_ ]?(20\d\d)(?![0-9])", u)
    ]
    return [d for d in found if d is not None]


def url_months(url: str | None) -> list[date]:
    """Return the first day of each month named in ``url`` without a day (``may-2025``, ``/2025/05/``)."""
    if not url:
        return []
    u = url.lower()
    found = [_safe_date(int(y), _month_num(m), 1) for m, y in re.findall(rf"(?<![a-z])({_ANY_MONTH})[-_ ]?(20\d\d)", u)]
    found += [_safe_date(int(y), int(m), 1) for y, m in re.findall(r"/(20\d\d)/(\d\d)(?:/|$)", u)]
    found += [_safe_date(2000 + int(y), _MON3_NUM[m], 1) for m, y in re.findall(rf"/({_MON3})(\d{{2}})\.pdf$", u)]
    return [d for d in found if d is not None]


def pdf_dates(url: str, settings: CfmV52Settings) -> list[date]:
    """Return the creation and modification dates stored in a PDF (Info dictionary or XMP)."""
    validate_public_url(url)
    request = Request(url, headers={"User-Agent": "CFM-Agent-v5.1-Audit/1.0", "Accept": "application/pdf"})
    opener = build_opener(SafeRedirectHandler(settings.source_resolution_max_redirects))
    with opener.open(request, timeout=settings.source_resolution_timeout_seconds) as response:
        data = response.read(PDF_MAX_BYTES)
    found = re.findall(rb"/(?:CreationDate|ModDate)\s*\(D:(\d{4})(\d{2})(\d{2})", data)
    found += re.findall(rb"<xmp:(?:CreateDate|ModifyDate)>(\d{4})-(\d{2})-(\d{2})", data)
    dates = [_safe_date(int(y), int(m), int(d)) for y, m, d in found]
    return [d for d in dates if d is not None]


@dataclass(frozen=True)
class SourceDating:
    """The gate's verdict on one grounding page, kept in the run's audit."""

    grounding_url: str
    resolved_url: str | None
    decision: str
    url_dates: list[str] = field(default_factory=list)
    url_months: list[str] = field(default_factory=list)
    pdf_dates: list[str] = field(default_factory=list)
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
    read_pdf: Callable[[str, CfmV52Settings], list[date]] | None = None,
) -> SourceDating:
    """Judge one resolved grounding page against ``cutoff_date`` (YYYY-MM-DD).

    ``inspect`` fetches and reads the page; it defaults to v5.2's audit extractor.
    ``read_pdf`` reads a PDF's own dates; it defaults to :func:`pdf_dates`.
    """
    inspect = inspect or _inspect_source
    read_pdf = read_pdf or pdf_dates
    cutoff = date.fromisoformat(cutoff_date)
    if not source.resolved_url:
        return SourceDating(source.grounding_url, None, REJECT_UNRESOLVED, note=source.resolution_warning or "")

    in_url = url_dates(source.resolved_url)
    months = url_months(source.resolved_url)
    record = inspect(source, cutoff_date, settings)
    page = date.fromisoformat(record.publication_date) if record.publication_date else None
    in_pdf: list[date] = []
    pdf_note = ""
    if urlparse(source.resolved_url).path.lower().endswith(".pdf"):
        try:
            in_pdf = read_pdf(source.resolved_url, settings)
        except Exception as exc:  # noqa: BLE001
            pdf_note = f"pdf_read_failed: {type(exc).__name__}: {exc}"
    pdf_latest = max(in_pdf) if in_pdf else None
    # Positive evidence: exact dates, and a PDF date safely before the cutoff.
    known = [*in_url, *([page] if page else [])]
    if pdf_latest and pdf_latest <= cutoff - timedelta(days=PDF_MARGIN_DAYS):
        known.append(pdf_latest)

    def verdict(decision: str, note: str = "") -> SourceDating:
        return SourceDating(
            grounding_url=source.grounding_url,
            resolved_url=source.resolved_url,
            decision=decision,
            url_dates=[d.isoformat() for d in in_url],
            url_months=[d.strftime("%Y-%m") for d in months],
            pdf_dates=[d.isoformat() for d in in_pdf],
            page_date=page.isoformat() if page else None,
            page_date_from=record.publication_date_provenance,
            note=note or "; ".join([*record.extraction_warnings, *([pdf_note] if pdf_note else [])])[:300],
        )

    # Negative evidence: any exact or PDF date on/after the cutoff, or a month starting on/after it.
    late = [d for d in [*in_url, *([page] if page else []), *in_pdf, *months] if d >= cutoff]
    if late:
        return verdict(REJECT_LATE, f"dated {max(late).isoformat()}, cutoff {cutoff_date}")
    if record.potentially_mutable:
        return verdict(REJECT_LIVE, "live page: content can change after the cutoff")
    if not known and not allow_undated:
        reason = "no publication date in the URL or on the page"
        if months or in_pdf:
            reason = "only month-level or recent PDF dates, which cannot prove an earlier publication"
        return verdict(REJECT_UNDATED, reason)
    return verdict(ACCEPTED)
