"""Optional audit-only destination-page inspection for CFM Agent v5.0."""

from __future__ import annotations

import json
import re
from datetime import datetime
from html import unescape
from html.parser import HTMLParser
from urllib.request import Request, build_opener

from energy_oil_forecasting.cfm_agent_v_5_0.config import CfmV50Settings
from energy_oil_forecasting.cfm_agent_v_5_0.schemas import (
    ActiveSource,
    AuditPassage,
    AuditSourceRecord,
    SourceAuditBundle,
)
from energy_oil_forecasting.cfm_agent_v_5_0.tools.url_safety import (
    SafeRedirectHandler,
    validate_public_url,
)


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.text: list[str] = []
        self.title: list[str] = []
        self.in_title = False
        self.meta: dict[str, str] = {}
        self.jsonld: list[str] = []
        self._jsonld = False
        self._buffer: list[str] = []

    def handle_starttag(self, tag, attrs):  # noqa: ANN001, ANN201
        values = {key.lower(): value for key, value in attrs if value is not None}
        tag = tag.lower()
        if tag == "title":
            self.in_title = True
        elif tag == "meta":
            key = (values.get("property") or values.get("name") or "").lower()
            value = values.get("content", "")
            if key and value:
                self.meta[key] = value
        elif tag == "script" and values.get("type", "").lower() == "application/ld+json":
            self._jsonld = True
            self._buffer = []

    def handle_endtag(self, tag):  # noqa: ANN001, ANN201
        tag = tag.lower()
        if tag == "title":
            self.in_title = False
        elif tag == "script" and self._jsonld:
            self.jsonld.append("".join(self._buffer))
            self._jsonld = False

    def handle_data(self, data):  # noqa: ANN001, ANN201
        if self._jsonld:
            self._buffer.append(data)
        elif self.in_title:
            self.title.append(data)
        elif data.strip():
            self.text.append(data.strip())


def _date_only(value: object) -> str | None:
    if not value:
        return None
    match = re.search(r"\d{4}-\d{2}-\d{2}", str(value))
    return match.group(0) if match else None


def _jsonld_date(blobs: list[str]) -> str | None:
    for blob in blobs:
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            continue
        stack = list(data if isinstance(data, list) else [data])
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                date = _date_only(value.get("datePublished"))
                if date:
                    return date
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
    return None


def _visible_date(text: str) -> str | None:
    patterns = (
        r"(?:date\s+published|published|publication\s+date|posted)\s*[:|\-]?\s*"
        r"([A-Za-z]+\s+\d{1,2},\s+\d{4})",
        r"(?:date\s+published|published|publication\s+date|posted)\s*[:|\-]?\s*"
        r"(\d{4}-\d{2}-\d{2})",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(match.group(1), fmt).date().isoformat()
            except ValueError:
                continue
    return None


def _extract_passages(source_id: str, text: str, settings: CfmV50Settings) -> list[AuditPassage]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chosen = [sentence for sentence in sentences if 80 <= len(sentence) <= settings.research_passage_chars][
        : settings.research_max_passages_per_source
    ]
    return [
        AuditPassage(
            passage_id=f"{source_id}_audit_passage_{index:02d}",
            source_id=source_id,
            text=sentence,
            extraction_method="visible_page_text",
        )
        for index, sentence in enumerate(chosen, 1)
    ]


def _inspect_source(source: ActiveSource, cutoff_date: str, settings: CfmV50Settings) -> AuditSourceRecord:
    warnings: list[str] = []
    if not source.resolved_url:
        return AuditSourceRecord(
            source_id=source.source_id,
            source_quality=source.source_quality,
            forecast_eligibility="unresolved",
            extraction_warnings=["resolved URL unavailable"],
        )
    html = ""
    try:
        validate_public_url(source.resolved_url)
        request = Request(
            source.resolved_url,
            headers={
                "User-Agent": "CFM-Agent-v5.0-Audit/1.0",
                "Accept": "text/html,application/xhtml+xml",
            },
        )
        with build_opener(SafeRedirectHandler(settings.source_resolution_max_redirects)).open(
            request,
            timeout=settings.source_resolution_timeout_seconds,
        ) as response:
            html = response.read(2_000_000).decode(response.headers.get_content_charset() or "utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"audit_fetch_failed: {type(exc).__name__}: {exc}")

    parser = _TextParser()
    if html:
        try:
            parser.feed(html)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"html_parse_failed: {type(exc).__name__}: {exc}")
    title = parser.meta.get("og:title") or "".join(parser.title).strip() or source.provider_title
    publisher_metadata = parser.meta.get("article:publisher") or parser.meta.get("og:site_name")
    publication_date = None
    provenance = "missing"
    for key in (
        "article:published_time",
        "date",
        "datepublished",
        "publishdate",
        "og:published_time",
    ):
        publication_date = _date_only(parser.meta.get(key))
        if publication_date:
            provenance = "html_metadata"
            break
    if not publication_date:
        publication_date = _jsonld_date(parser.jsonld)
        if publication_date:
            provenance = "json_ld"
    visible_text = unescape(re.sub(r"\s+", " ", " ".join(parser.text))).strip()
    if not publication_date:
        publication_date = _visible_date(visible_text[:20_000])
        if publication_date:
            provenance = "visible_page_text"
    passages = _extract_passages(source.source_id, visible_text, settings)
    if not passages:
        warnings.append("no_page_passages_extracted")

    eligibility = "date_unknown"
    if publication_date:
        source_date = datetime.strptime(publication_date, "%Y-%m-%d").date()
        cutoff = datetime.strptime(cutoff_date, "%Y-%m-%d").date()
        eligibility = "eligible" if source_date < cutoff else "ineligible"
    mutable = bool(source.resolved_url and source.resolved_url.rstrip("/").endswith(("outlooks/steo", "latest")))
    return AuditSourceRecord(
        source_id=source.source_id,
        resolved_url=source.resolved_url,
        title=title,
        publisher_metadata=publisher_metadata,
        publication_date=publication_date,
        publication_date_provenance=provenance,
        passages=passages,
        extraction_warnings=warnings,
        forecast_eligibility=eligibility,
        source_quality=source.source_quality,
        potentially_mutable=mutable,
    )


def run_source_audit(sources: list[ActiveSource], cutoff_date: str, settings: CfmV50Settings) -> SourceAuditBundle:
    """Fetch and inspect destination pages only when the audit switch is enabled."""
    if not settings.audit_enabled:
        return SourceAuditBundle(audit_enabled=False, executed=False)
    records = [_inspect_source(source, cutoff_date, settings) for source in sources]
    return SourceAuditBundle(audit_enabled=True, executed=True, records=records)


__all__ = ["run_source_audit"]
