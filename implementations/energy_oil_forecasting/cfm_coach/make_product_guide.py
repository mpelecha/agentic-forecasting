"""Generate the CFM Coach product guide as a PDF.

Written for the desk: macro portfolio managers who will read the coach's reports and
sign off on its recommendations. Plain language, finance framing, figures labelled in
words. Shares the palette of ``make_teammate_guide.py`` so the two read as one set.

Every number is stamped ``AS_OF`` and was read from the running code on that date:
``report.py`` and ``review/weeks/2026-W37/report.md`` for the scoreboard, ``config.py``
for the approval thresholds, the run directories for corpus sizes. Regenerate after the
corpus grows and re-check each figure when you do.

Usage (from the repo root)::

    uv run --with reportlab python -m energy_oil_forecasting.cfm_coach.make_product_guide
"""

from __future__ import annotations

from pathlib import Path

from reportlab.graphics.shapes import Circle, Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


AS_OF = "13 September 2026"
OUTPUT = Path(__file__).resolve().parent / "CFM_Coach_Product_Guide.pdf"
PAGES = 6

NAVY = colors.HexColor("#1e2a38")
DEEP = colors.HexColor("#11223e")
TEAL = colors.HexColor("#2a8b82")
GOLD = colors.HexColor("#c49a45")
BLUE = colors.HexColor("#2e67a8")
RED = colors.HexColor("#b5482f")
RULE = colors.HexColor("#d4dde6")
WASH = colors.HexColor("#f1f5f8")
CREAM = colors.HexColor("#f8f1e4")
MUTED = colors.HexColor("#606f83")
PALE_TEAL = colors.HexColor("#dff0ee")
PALE_GOLD = colors.HexColor("#f6ead2")
PALE_BLUE = colors.HexColor("#dde8f5")
GREY = colors.HexColor("#9aa7b6")

PAGE_W, PAGE_H = LETTER
MARGIN = 0.8 * inch
CONTENT_W = PAGE_W - 2 * MARGIN


# -- styles ------------------------------------------------------------------

S = {
    "kicker": ParagraphStyle("kicker", fontName="Helvetica-Bold", fontSize=10, textColor=TEAL, leading=13, spaceAfter=6),
    "cover_title": ParagraphStyle("cover_title", fontName="Helvetica-Bold", fontSize=30, textColor=NAVY, leading=34, spaceAfter=8),
    "cover_sub": ParagraphStyle("cover_sub", fontName="Helvetica", fontSize=11, textColor=MUTED, leading=16, spaceAfter=12),
    "section": ParagraphStyle("section", fontName="Helvetica-Bold", fontSize=9, textColor=TEAL, leading=11, spaceAfter=3),
    "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=17, textColor=NAVY, leading=21, spaceAfter=8),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=11, textColor=DEEP, leading=14, spaceBefore=9, spaceAfter=4),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9.4, textColor=NAVY, leading=13.6, spaceAfter=6, alignment=TA_LEFT),
    "bullet": ParagraphStyle("bullet", fontName="Helvetica", fontSize=9.4, textColor=NAVY, leading=13.4, leftIndent=13, bulletIndent=3, spaceAfter=3),
    "callout": ParagraphStyle("callout", fontName="Helvetica", fontSize=9.2, textColor=DEEP, leading=13.2, spaceAfter=0),
    "th": ParagraphStyle("th", fontName="Helvetica-Bold", fontSize=8.3, textColor=colors.white, leading=10.8),
    "td": ParagraphStyle("td", fontName="Helvetica", fontSize=8.3, textColor=NAVY, leading=11.2),
    "td_b": ParagraphStyle("td_b", fontName="Helvetica-Bold", fontSize=8.3, textColor=NAVY, leading=11.2),
    "caption": ParagraphStyle("caption", fontName="Helvetica-Oblique", fontSize=8, textColor=MUTED, leading=10.8, spaceBefore=3, spaceAfter=8),
    "gloss_t": ParagraphStyle("gloss_t", fontName="Helvetica-Bold", fontSize=8.3, textColor=DEEP, leading=11),
    "gloss_d": ParagraphStyle("gloss_d", fontName="Helvetica", fontSize=8.3, textColor=NAVY, leading=11),
}


# -- helpers -----------------------------------------------------------------


def hex_of(color: colors.Color) -> str:
    return "#" + color.hexval()[2:]


def para(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def bullets(items: list[str]) -> list[Paragraph]:
    return [Paragraph(item, S["bullet"], bulletText="•") for item in items]


def section(number: str, name: str, headline: str) -> list:
    bar = Table([[Paragraph(f"{number} | {name.upper()}", S["section"])]], colWidths=[CONTENT_W])
    bar.setStyle(
        TableStyle(
            [
                ("LINEBELOW", (0, 0), (-1, -1), 1.4, TEAL),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    return [bar, Spacer(1, 6), para(headline, "h1")]


def table(rows: list[list[str]], widths: list[float], *, bold_first_col: bool = False) -> Table:
    data = [[Paragraph(cell, S["th"]) for cell in rows[0]]]
    for row in rows[1:]:
        first = S["td_b"] if bold_first_col else S["td"]
        data.append([Paragraph(row[0], first), *[Paragraph(cell, S["td"]) for cell in row[1:]]])
    made = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.5, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]
    for index in range(1, len(data)):
        if index % 2 == 0:
            commands.append(("BACKGROUND", (0, index), (-1, index), WASH))
    made.setStyle(TableStyle(commands))
    return made


def callout(title: str, text: str, *, tone: str = "teal") -> Table:
    accent = {"teal": TEAL, "gold": GOLD, "blue": BLUE}[tone]
    background = {"teal": WASH, "gold": CREAM, "blue": WASH}[tone]
    inner = [
        Paragraph(f'<font color="{hex_of(accent)}"><b>{title}</b></font>', S["callout"]),
        Spacer(1, 2),
        Paragraph(text, S["callout"]),
    ]
    box = Table([[inner]], colWidths=[CONTENT_W])
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("LINEBEFORE", (0, 0), (0, -1), 2.5, accent),
                ("BOX", (0, 0), (-1, -1), 0.5, RULE),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    return box


def caption(text: str) -> Paragraph:
    return para(text, "caption")


# -- drawing primitives -------------------------------------------------------


def node(
    d: Drawing,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    lines: list[str] | None = None,
    *,
    fill=WASH,
    stroke=TEAL,
    title_size: float = 7.8,
    body_size: float = 6.8,
) -> None:
    d.add(Rect(x, y, w, h, fillColor=fill, strokeColor=stroke, strokeWidth=0.9, rx=3, ry=3))
    lines = lines or []
    total = title_size + 1.5 + len(lines) * (body_size + 2.4)
    top = y + h / 2 + total / 2 - title_size + 1
    d.add(String(x + w / 2, top, title, fontName="Helvetica-Bold", fontSize=title_size, fillColor=NAVY, textAnchor="middle"))
    cursor = top - title_size - 1.5
    for line in lines:
        d.add(String(x + w / 2, cursor, line, fontName="Helvetica", fontSize=body_size, fillColor=NAVY, textAnchor="middle"))
        cursor -= body_size + 2.4


def arrow(d: Drawing, x1: float, y1: float, x2: float, y2: float, *, color=MUTED, width: float = 0.9, head: float = 4.5) -> None:
    d.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=width))
    dx, dy = x2 - x1, y2 - y1
    length = (dx * dx + dy * dy) ** 0.5 or 1.0
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    bx, by = x2 - ux * head, y2 - uy * head
    d.add(
        Polygon(
            [x2, y2, bx + px * head * 0.55, by + py * head * 0.55, bx - px * head * 0.55, by - py * head * 0.55],
            fillColor=color,
            strokeColor=color,
            strokeWidth=0.3,
        )
    )


def label(d: Drawing, x: float, y: float, text: str, *, size: float = 6.6, color=MUTED, anchor: str = "middle", bold: bool = False) -> None:
    d.add(String(x, y, text, fontName="Helvetica-Bold" if bold else "Helvetica", fontSize=size, fillColor=color, textAnchor=anchor))


def band(d: Drawing, x_lo: float, x_hi: float, x_mid: float, y: float, *, color, thickness: float = 6) -> None:
    """One forecast range drawn as a horizontal bar with a tick at its central estimate."""
    d.add(Rect(x_lo, y - thickness / 2, x_hi - x_lo, thickness, fillColor=color, strokeColor=None))
    d.add(Rect(x_mid - 1.2, y - thickness / 2 - 3, 2.4, thickness + 6, fillColor=NAVY, strokeColor=None))


# -- figures -----------------------------------------------------------------


def figure_cycle() -> Drawing:
    """Figure 1: the loop from a published call to a signed-off adjustment."""
    width, height = CONTENT_W, 150
    d = Drawing(width, height)
    top_y, bot_y = height - 60, 14
    bw, bh, gap = 100, 46, (width - 4 * 100) / 3
    steps_top = [
        ("1. A call is made", ["The forecaster publishes a", "range for oil at 5, 10 and", "21 business days ahead"], colors.white, GREY),
        ("2. The coach files it", ["Every call is stored with", "everything needed to", "re-derive it later"], PALE_TEAL, TEAL),
        ("3. The market decides", ["When the target date", "arrives, the actual close", "is attached to the call"], PALE_TEAL, TEAL),
        ("4. It is scored", ["Against a random walk,", "the benchmark: was the", "range right, and how wrong?"], PALE_TEAL, TEAL),
    ]
    steps_bot = [
        ("8. A second forecast", ["The coached range is", "published beside the", "original, every day"], PALE_BLUE, BLUE),
        ("7. The team decides", ["Each recommendation is", "accepted, rejected or", "deferred, with a reason"], CREAM, GOLD),
        ("6. Recommendations", ["Ranked by how much of the", "shortfall to the benchmark", "they would recover"], PALE_GOLD, GOLD),
        ("5. Weekly review", ["Where did the shortfall", "come from, and which", "mistakes keep repeating?"], PALE_GOLD, GOLD),
    ]
    for i, (title, lines, fill, stroke) in enumerate(steps_top):
        x = i * (bw + gap)
        node(d, x, top_y, bw, bh, title, lines, fill=fill, stroke=stroke)
        if i < 3:
            arrow(d, x + bw, top_y + bh / 2, x + bw + gap, top_y + bh / 2)
    for i, (title, lines, fill, stroke) in enumerate(steps_bot):
        x = i * (bw + gap)
        node(d, x, bot_y, bw, bh, title, lines, fill=fill, stroke=stroke)
        if i > 0:
            arrow(d, x, bot_y + bh / 2, x - gap, bot_y + bh / 2)
    arrow(d, 3 * (bw + gap) + bw / 2, top_y, 3 * (bw + gap) + bw / 2, bot_y + bh)
    label(d, width / 2, height - 8, "The forecaster itself is never edited. Adjustments live in a separate ledger owned by the coach.", size=6.8, color=TEAL, bold=True)
    return d


def figure_band() -> Drawing:
    """Figure 2: the anatomy of one call, and the three ways the coach can adjust it."""
    width, height = CONTENT_W, 150
    d = Drawing(width, height)
    left_w = 200
    # Left: anatomy
    d.add(Rect(0, 0, left_w, height, fillColor=colors.white, strokeColor=RULE, strokeWidth=0.6, rx=3, ry=3))
    label(d, left_w / 2, height - 13, "What a call looks like", size=7.6, color=NAVY, bold=True)
    axis_y = 30
    d.add(Line(14, axis_y, left_w - 14, axis_y, strokeColor=RULE, strokeWidth=0.8))
    label(d, 14, axis_y - 10, "lower price", size=5.8, anchor="start")
    label(d, left_w - 14, axis_y - 10, "higher price", size=5.8, anchor="end")
    spot = 92
    d.add(Line(spot, axis_y - 2, spot, height - 24, strokeColor=GOLD, strokeWidth=0.9, strokeDashArray=[2, 2]))
    label(d, spot, axis_y - 20, "today's price", size=5.8, color=GOLD)
    y_models = 100
    band(d, spot - 22, spot + 30, spot + 2, y_models, color=GREY)
    label(d, spot + 34, y_models - 2, "statistical models: central", size=5.8, anchor="start")
    label(d, spot + 34, y_models - 9, "estimate and an 80% range", size=5.8, anchor="start")
    y_final = 62
    band(d, spot - 18, spot + 34, spot + 8, y_final, color=TEAL)
    label(d, spot + 38, y_final - 2, "published call: the models", size=5.8, anchor="start")
    label(d, spot + 38, y_final - 9, "plus a judgment overlay", size=5.8, anchor="start")
    arrow(d, spot + 2, y_models - 6, spot + 8, y_final + 7, color=BLUE, width=0.7, head=3.5)
    label(d, spot - 20, 82, "overlay: the news-", size=5.6, color=BLUE, anchor="end")
    label(d, spot - 20, 75, "reading step nudges", size=5.6, color=BLUE, anchor="end")
    label(d, spot - 20, 68, "the centre", size=5.6, color=BLUE, anchor="end")

    # Right: three adjustments
    rx = left_w + 12
    rw = width - rx
    d.add(Rect(rx, 0, rw, height, fillColor=colors.white, strokeColor=RULE, strokeWidth=0.6, rx=3, ry=3))
    label(d, rx + rw / 2, height - 13, "Three adjustments the coach can make to a call", size=7.6, color=NAVY, bold=True)
    col_w = (rw - 24) / 3
    panels = [
        ("Widen the range", "when the 80% range covers", "far fewer than 80% of outcomes", dict(lo=-30, hi=30, mid=0)),
        ("Trust the overlay less", "when the judgment nudge", "adds noise rather than signal", dict(lo=-20, hi=32, mid=2)),
        ("Lean toward today's price", "when the central estimate", "sits below spot too often", dict(lo=-8, hi=44, mid=18)),
    ]
    for i, (title, l1, l2, after) in enumerate(panels):
        cx = rx + 12 + i * col_w + col_w / 2
        label(d, cx, height - 30, title, size=6.8, color=NAVY, bold=True)
        label(d, cx, height - 40, l1, size=5.6)
        label(d, cx, height - 48, l2, size=5.6)
        base = cx - 6
        d.add(Line(base + 10, 28, base + 10, 92, strokeColor=GOLD, strokeWidth=0.8, strokeDashArray=[2, 2]))
        label(d, base + 10, 20, "today's price", size=5.4, color=GOLD)
        band(d, base - 18, base + 34, base + 8, 78, color=GREY)
        label(d, cx - col_w / 2 + 4, 76, "before", size=5.4, anchor="start")
        band(d, base + after["lo"], base + after["hi"], base + after["mid"] + 8, 46, color=TEAL)
        label(d, cx - col_w / 2 + 4, 44, "after", size=5.4, color=TEAL, anchor="start")
    return d


def figure_scoring() -> Drawing:
    """Figure 3: what a score rewards, shown on one resolved call."""
    width, height = CONTENT_W, 96
    d = Drawing(width, height)
    axis_y = 22
    d.add(Line(20, axis_y, 300, axis_y, strokeColor=RULE, strokeWidth=0.8))
    spot = 110
    d.add(Line(spot, axis_y - 2, spot, height - 14, strokeColor=GOLD, strokeWidth=0.9, strokeDashArray=[2, 2]))
    label(d, spot, axis_y - 11, "price when the call was made", size=5.8, color=GOLD)
    actual = 190
    d.add(Circle(actual, axis_y, 3.2, fillColor=RED, strokeColor=None))
    label(d, actual, axis_y - 11, "what oil actually did", size=5.8, color=RED)
    y_call = 66
    band(d, spot - 20, spot + 44, spot + 6, y_call, color=TEAL)
    label(d, spot - 24, y_call - 2, "the call", size=6, color=TEAL, anchor="end")
    y_bench = 44
    band(d, spot - 46, spot + 46, spot, y_bench, color=GOLD)
    label(d, spot - 50, y_bench - 2, "benchmark", size=6, color=GOLD, anchor="end")
    d.add(Line(actual, axis_y + 4, actual, y_call + 4, strokeColor=RED, strokeWidth=0.6, strokeDashArray=[1.5, 1.5]))

    tx = 320
    items = [
        ("Coverage", "Did the outcome land inside the 80% range? Here the call missed and the benchmark caught it."),
        ("Score (pinball loss)", "How far the outcome sat from the range, penalised across the whole range, not only the centre. Lower is better."),
        ("Direction", "Was the centre on the same side of the starting price as the outcome? Reported against the base rate, not on its own."),
    ]
    y = height - 12
    for title, text in items:
        label(d, tx, y, title, size=6.8, color=NAVY, anchor="start", bold=True)
        words = text.split()
        line, lines = "", []
        for word in words:
            if len(line) + len(word) > 52:
                lines.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        lines.append(line)
        for k, part in enumerate(lines):
            label(d, tx, y - 9 - k * 8, part, size=5.9, color=NAVY, anchor="start")
        y -= 12 + len(lines) * 8 + 4
    return d


def figure_approval() -> Drawing:
    """Figure 4: what a proposed adjustment must pass before a person sees it."""
    width, height = CONTENT_W, 96
    d = Drawing(width, height)
    steps = [
        ("Clean data", ["Live, forward-dated calls", "only. One version of the", "forecaster, one model."], PALE_TEAL, TEAL),
        ("Enough of it", ["At least 12 resolved dates,", "6 of them never used to", "choose the adjustment."], PALE_TEAL, TEAL),
        ("A fair test", ["Scored on the same days as", "the current setting. One", "change at a time."], PALE_BLUE, BLUE),
        ("Worth doing", ["At least 3% better, and", "robust to resampling at", "90% confidence."], PALE_GOLD, GOLD),
        ("Sign-off", ["Applied at half strength,", "and only after a person", "accepts it."], CREAM, GOLD),
    ]
    n = len(steps)
    gap = 10
    bw = (width - gap * (n - 1)) / n
    for i, (title, lines, fill, stroke) in enumerate(steps):
        x = i * (bw + gap)
        node(d, x, 22, bw, 54, title, lines, fill=fill, stroke=stroke, body_size=6.3)
        if i < n - 1:
            arrow(d, x + bw, 49, x + bw + gap, 49)
    label(d, width / 2, 8, "A candidate that fails any step is held back with the reason printed, and comes up again as more dates resolve.", size=6.4)
    return d


def figure_weekly() -> Drawing:
    """Figure 5: the weekly review, in the order the report is written."""
    width, height = CONTENT_W, 92
    d = Drawing(width, height)
    steps = [
        ("Collect", ["The calls that resolved", "since last week"], WASH, TEAL),
        ("Attribute", ["Split the shortfall to the", "benchmark: models vs", "judgment overlay"], WASH, TEAL),
        ("Diagnose", ["Tag recurring mistakes,", "each with a reference", "into the original call"], PALE_GOLD, GOLD),
        ("Recommend", ["Draft changes, estimate", "their effect, run the", "approval checks"], PALE_GOLD, GOLD),
        ("Report", ["One page: scope, attribution,", "ranked recommendations,", "watchlist, cost"], PALE_BLUE, BLUE),
    ]
    n = len(steps)
    gap = 10
    bw = (width - gap * (n - 1)) / n
    for i, (title, lines, fill, stroke) in enumerate(steps):
        x = i * (bw + gap)
        node(d, x, 22, bw, 52, title, lines, fill=fill, stroke=stroke, body_size=6.3)
        if i < n - 1:
            arrow(d, x + bw, 48, x + bw + gap, 48)
    label(d, width / 2, 8, "Only the diagnose and recommend steps use a language model. Every number in the report is computed by code, not written by the model.", size=6.4)
    return d


def figure_scoreboard() -> Drawing:
    """Figure 6: the forecaster against its benchmark, and where the shortfall sits."""
    width, height = CONTENT_W, 120
    d = Drawing(width, height)
    rows = [
        ("Standard model, 5 days", 2.415, 2.055, 0.79),
        ("Standard model, 10 days", 3.889, 3.413, 0.79),
        ("Premium model, 5 days", 2.521, 2.056, 0.92),
        ("Premium model, 10 days", 4.044, 3.433, 0.95),
    ]
    left = 118
    scale = 46.0
    top = height - 16
    row_h = 24
    label(d, left, height - 5, "average score on live calls (lower is better)", size=6.4, anchor="start")
    for i, (name, agent, rw, base_share) in enumerate(rows):
        y = top - i * row_h - 8
        label(d, left - 6, y + 2, name, size=6.6, color=NAVY, anchor="end", bold=True)
        d.add(Rect(left, y + 5, rw * scale, 6, fillColor=GOLD, strokeColor=None))
        label(d, left + rw * scale + 3, y + 6, f"benchmark {rw:.2f}", size=5.9, color=GOLD, anchor="start")
        gap = agent - rw
        d.add(Rect(left, y - 4, rw * scale, 6, fillColor=NAVY, strokeColor=None))
        d.add(Rect(left + rw * scale, y - 4, gap * base_share * scale, 6, fillColor=RED, strokeColor=None))
        d.add(Rect(left + (rw + gap * base_share) * scale, y - 4, gap * (1 - base_share) * scale, 6, fillColor=TEAL, strokeColor=None))
        label(d, left + agent * scale + 3, y - 3, f"forecaster {agent:.2f}, {(agent / rw - 1) * 100:.0f}% behind", size=5.9, color=NAVY, anchor="start")
    ly = 8
    for offset, color, text in [(0, GOLD, "benchmark"), (58, NAVY, "forecaster, up to the benchmark"), (200, RED, "shortfall from the statistical models"), (352, TEAL, "shortfall from the judgment overlay")]:
        d.add(Rect(left + offset, ly, 8, 5, fillColor=color, strokeColor=None))
        label(d, left + offset + 11, ly, text, size=5.9, anchor="start")
    return d


# -- page furniture ----------------------------------------------------------


def draw_page(canvas, doc):  # noqa: ANN001
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, PAGE_H - 0.22 * inch, PAGE_W, 0.22 * inch, stroke=0, fill=1)
    canvas.setFillColor(TEAL)
    canvas.rect(0, PAGE_H - 0.22 * inch, 2.0 * inch, 0.22 * inch, stroke=0, fill=1)
    canvas.setFont("Helvetica", 7.4)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, PAGE_H - 0.48 * inch, "CFM Coach  |  Product guide")
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 0.48 * inch, f"Page {doc.page} of {PAGES}")
    canvas.setFillColor(RULE)
    canvas.rect(MARGIN, 0.5 * inch, CONTENT_W, 0.5, stroke=0, fill=1)
    canvas.setFont("Helvetica", 6.8)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, 0.35 * inch, f"Figures as of {AS_OF}. Regenerate with make_product_guide.py after the corpus grows.")
    canvas.restoreState()


def build_document() -> BaseDocTemplate:
    doc = BaseDocTemplate(
        str(OUTPUT),
        pagesize=LETTER,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=0.62 * inch,
        bottomMargin=0.62 * inch,
        title="CFM Coach - Product guide",
        author="CFM Coach",
        subject="What the coach does, how it scores and adjusts the oil forecast, how to use it, and what is still open",
    )
    body_frame = Frame(MARGIN, 0.62 * inch, CONTENT_W, PAGE_H - 1.24 * inch, id="body", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([PageTemplate(id="body", frames=[body_frame], onPage=draw_page)])
    return doc


# -- content -----------------------------------------------------------------


def page_what() -> list:
    facts = [
        ["The forecaster", "An automated system that reads the news, runs three statistical models, and publishes a range for WTI crude at 5, 10 and 21 business days ahead. Two versions run every weekday, one on a standard language model and one on a premium model."],
        ["The coach", "A performance and model-governance layer beside it. It keeps the track record, scores every call once the outcome is known, works out where the shortfall comes from, and recommends adjustments."],
        ["The benchmark", "A random walk: today's price as the central estimate, with a range sized to recent volatility. The forecaster earns its keep only if it beats this."],
        ["Governance", "The forecaster is never edited. Every adjustment is a recommendation until a person accepts it, and accepted adjustments are published as a second, coached range beside the original."],
    ]
    return [
        para("PRODUCT GUIDE", "kicker"),
        para("CFM Coach", "cover_title"),
        para(
            "Think of a desk with a systematic forecaster and a performance analyst sitting next to it. The forecaster makes calls on the "
            "oil price. The analyst keeps the book, scores each call against a benchmark once the market has spoken, tells the team where "
            "the money is being left on the table, and proposes adjustments that go through sign-off before they take effect. The coach is that analyst.",
            "cover_sub",
        ),
        table([["", "In one line"], *facts], [1.15 * inch, CONTENT_W - 1.15 * inch], bold_first_col=True),
        Spacer(1, 12),
        *section("1", "The cycle", "From a published call to a signed-off adjustment"),
        para(
            "The coach runs in two rhythms. Every weekday it files the new calls, attaches outcomes to the calls that have matured, "
            "and updates the scoreboard. Every week it reviews what resolved, attributes the shortfall, and writes recommendations for the team."
        ),
        figure_cycle(),
        caption("Figure 1. Steps 1 to 4 run automatically each weekday and use no language model. Steps 5 and 6 run weekly on a small budget. Step 7 is the team's."),
        callout(
            "The one number that matters",
            "Score against the random walk at each horizon, on live calls only. A recommendation that does not move that number, or "
            "coverage toward 80%, or direction above the base rate, is a hypothesis for the watchlist rather than a recommendation.",
            tone="gold",
        ),
    ]


def page_adjust() -> list:
    return [
        *section("2", "The call", "What the coach can and cannot adjust"),
        para(
            "Each call is built in two layers. Three statistical models produce a central estimate and an 80% range. A news-reading step then "
            "proposes a nudge to the centre and a change to the width, which an evidence check either grants, trims, or refuses. The coach "
            "does not touch either layer directly. It records the call, and can then re-derive it under different settings, exactly and without "
            "any language model, because the news-reading step only ever outputs categories such as \"moderately up\" or \"substantially wider\" while code turns those into numbers."
        ),
        figure_band(),
        caption("Figure 2. Left: the anatomy of one call. Right: the three adjustments, each shown before and after. All three are set per horizon and can be combined, but only one is tested at a time."),
        para("How each adjustment is chosen", "h2"),
        table(
            [
                ["Adjustment", "The problem it addresses", "Where the evidence comes from"],
                ["Widen the range", "The 80% range has covered roughly a third of outcomes. Too narrow a range understates risk.", "Eighteen years of the models' own history, re-run at past dates with no news input, so there is no look-ahead. Live calls then confirm or reject."],
                ["Lean toward today's price", "The central estimate has sat below spot on every recent call, which costs in a rally.", "The same history, choosing the lean that would have scored best. Live calls judge it."],
                ["Trust the overlay less or more", "The judgment nudge is small and its size varies between runs made minutes apart.", "Live calls only, and not yet: the coach is first measuring how noisy the nudge is, so a setting is not fitted to noise."],
                ["Engine settings", "Constants inside the evidence check, such as how many sources a claim needs before it counts.", "Priced by re-deriving stored calls; the value is argued in the recommendation rather than fitted."],
                ["Wording of the forecaster's instructions", "A recurring judgment error that traces back to how the forecaster is told to work.", "Not testable offline. An accepted change becomes a challenger version run in parallel and compared on the same dates."],
            ],
            [1.45 * inch, 2.35 * inch, CONTENT_W - 3.8 * inch],
            bold_first_col=True,
        ),
        Spacer(1, 6),
        callout(
            "Why the adjustments stay out of the forecaster",
            "The forecaster ships with a manifest that pins every file. Editing it would break that provenance and make old calls incomparable "
            "with new ones. So the coach keeps its settings in a ledger of its own, dated so that any past call can be re-derived under the "
            "settings that were in force at the time, and publishes the adjusted range as a separate daily series.",
            tone="teal",
        ),
    ]


def page_scoring() -> list:
    return [
        *section("3", "Scoring", "How a call is judged once the market has spoken"),
        para(
            "A call matures when its target date arrives. The coach attaches the actual close, then scores the published range, the "
            "statistical models alone, the benchmark, and the coached range on the same day. Because all four are scored on identical days, a "
            "difference between them is a difference in method rather than in luck about which days were scored."
        ),
        figure_scoring(),
        caption("Figure 3. One resolved call. The three measures answer different questions, and the report shows all three rather than a single blended number."),
        para("Three rules that keep the scoring honest", "h2"),
        *bullets(
            [
                "<b>Only live calls count.</b> A call re-run for a past date searches today's news, and the choice of what to read is coloured by what turned out to matter. Such calls are stored and shown, but never used to fit an adjustment.",
                "<b>Overlapping calls are not independent.</b> Twenty daily calls on a 21-day horizon look at almost the same window. The report prints the number of genuinely independent observations beside every result, and it is small today: about three and a half per horizon.",
                "<b>Direction is reported against the base rate.</b> Oil rose in most of the windows in the current sample, so a high hit rate can be reproduced by always saying up. The report shows both, and says when the sample is too small to tell skill from luck.",
            ]
        ),
        Spacer(1, 4),
        *section("4", "Approval", "What an adjustment must pass before it reaches the team"),
        para(
            "Every candidate adjustment goes through the same five checks, in this order. The data checks come first so that a statistical "
            "test is never run on a sample that mixes versions of the forecaster or includes look-ahead."
        ),
        figure_approval(),
        caption("Figure 4. The approval checks. Half strength means the applied value sits midway between the current setting and the fitted one, because a value fitted on a small sample tends to overshoot."),
    ]


def page_weekly() -> list:
    return [
        *section("5", "The weekly review", "Where the shortfall comes from, and what to do about it"),
        para(
            "The scoreboard says how far behind the benchmark the forecaster is. The weekly review says why, and what would close the gap. "
            "It reads every call that matured during the week, splits the shortfall between the statistical models and the judgment overlay, "
            "tags the mistakes that keep recurring, and drafts recommendations ranked by the share of the shortfall each would recover."
        ),
        figure_weekly(),
        caption("Figure 5. The review in the order the report is written. It runs on a budget of one to five dollars a week and stops itself at the limit."),
        para("What a recommendation contains", "h2"),
        para(
            "Each one states the mistake it addresses, the mechanism behind it, the change proposed, the estimated effect against the benchmark "
            "at each horizon, the evidence (how many independent dates, how many market episodes of each sign), and the file that would be applied. "
            "Three kinds exist, and each is delivered differently."
        ),
        table(
            [
                ["Kind", "Example", "What accepting it does"],
                ["A setting", "Lean the central estimate 40% of the way toward today's price at the 10-day horizon.", "The coach ledger gains a new dated version and the coached range changes from the next business day."],
                ["A wording change", "Tell the forecaster to cite each claim's source once, so that a bookkeeping slip stops zeroing its own view.", "A challenger copy of the forecaster is drafted with the change applied. Someone registers it, and it runs in parallel until enough shared dates exist to compare."],
                ["An engineering change", "Have one of the statistical models forecast returns rather than price levels.", "A written brief. Implemented by an engineer, then watched as a version change on the scoreboard."],
            ],
            [1.2 * inch, 2.7 * inch, CONTENT_W - 3.9 * inch],
            bold_first_col=True,
        ),
        Spacer(1, 6),
        para("How evidence is counted", "h2"),
        para(
            "A mistake that depends on knowing the outcome, such as calling the wrong direction, needs three independent windows across two "
            "market episodes of opposite sign before it can become a recommendation. A mistake visible in the call itself, such as the forecaster "
            "refusing its own proposed move on a citation technicality, needs four distinct dates. The review model reasons about the cases, but every "
            "count comes from code, and any claim it makes must point at a specific line in the original call. A recommendation the team rejects is remembered and not raised again until three new dates exist."
        ),
    ]


def page_usage() -> list:
    return [
        *section("6", "Day to day", "What a teammate actually does"),
        para(
            "Most of the coach runs itself. A scheduled job on one machine files the new calls each weekday morning; a second job publishes the "
            "coached range at noon; the review runs weekly. Three things involve a person."
        ),
        table(
            [
                ["When", "What", "How"],
                ["Any time", "Read the scoreboard: does everything still re-derive exactly, which calls have matured, and how each version compares with the benchmark.", "Run the report command below. Offline, two seconds, no cost."],
                ["Weekly", "Read the review. The first line says what can and cannot be concluded this week; then attribution, ranked recommendations, the watchlist, and anything held back with the reason.", "Open the week's report.md, or the HTML page published from it."],
                ["When a recommendation lands", "Decide. Set the status to accepted, rejected or deferred in the recommendation's file, with your name, the date and a reason. Reasons are kept and shape future reviews.", "Then run the accept command. It writes to the coach's ledger only, never to the live forecaster."],
            ],
            [1.2 * inch, 3.2 * inch, CONTENT_W - 4.4 * inch],
            bold_first_col=True,
        ),
        Spacer(1, 8),
        para("Commands", "h2"),
        para("All from the repository root. Each line is prefixed with <font face='Courier'>uv run python -m energy_oil_forecasting.cfm_coach.</font>"),
        table(
            [
                ["Purpose", "Command", "Uses a language model"],
                ["Scoreboard", "report --stream=v52_lite", "no"],
                ["This week's review, read-only pass", "review.run_weekly --no-llm", "no"],
                ["This week's review, full", "review.run_weekly", "yes, budgeted"],
                ["Record a decision", "review.accept --proposal P-0003 --stream v52_lite --by &lt;you&gt;", "no"],
                ["Compare a challenger with the original", "review.pairing --champion v52_advanced --challenger &lt;name&gt;", "no"],
                ["One extra live call by hand", "run_daily --stream=v52_lite", "yes, about four minutes"],
            ],
            [1.9 * inch, CONTENT_W - 3.3 * inch, 1.4 * inch],
            bold_first_col=True,
        ),
        Spacer(1, 8),
        callout(
            "Two habits worth keeping",
            "A weekday with no live call is a gap in the track record that cannot be filled later, so the machine running the morning job should stay "
            "awake and on power. And re-running a past date is safe to do for curiosity: the result is labelled as a re-run and kept out of every fit automatically.",
            tone="gold",
        ),
    ]


def page_status() -> list:
    return [
        *section("7", "Where things stand", "The scoreboard today and what is still open"),
        Spacer(1, 4),
        figure_scoreboard(),
        caption(
            "Figure 6. Live calls with outcomes through 11 September 2026: 18 dates per version, all within one rally, about three and a half independent "
            "observations per horizon. The 80% range covered 31 to 38% of outcomes; the benchmark's covered 45 to 63%."
        ),
        para(
            "Two readings follow from the chart. Most of the shortfall sits in the statistical models, whose range is too narrow and whose centre "
            "has leaned below spot through the rally. The judgment overlay is small and has helped slightly. So the first adjustments worth "
            "making are the two numeric ones in section 2, and the first candidate, a lean toward today's price, is already drafted. It is on hold "
            "only because the sample is too small for the approval checks, which is the process working as intended."
        ),
        para("Built and in use", "h2"),
        *bullets(
            [
                "Daily filing, outcome attachment and scoring for both versions of the forecaster, with the retired earlier version still scored. 288 calls on file, every one re-derives exactly.",
                "The approval checks, the coach's ledger, and the coached range published daily beside the original (currently identical, since nothing has been accepted yet).",
                "The weekly review end to end, with a first live run on 13 September 2026 that cost 89 cents and produced the watchlist and the first drafted candidate.",
            ]
        ),
        para("Open items", "h2"),
        *bullets(
            [
                "<b>More history.</b> The re-run of the statistical models at past dates, which the range and lean adjustments are fitted on, is 215 of 955 dates complete. The first fitted candidates follow within days of it finishing.",
                "<b>Time.</b> Judgment-related recommendations need two market episodes of opposite sign; with one rally on file, that is roughly eight to ten more weeks. A challenger comparison needs twelve shared dates plus the 21-day lag.",
                "<b>A confidence signal.</b> The design calls for a per-call flag saying when to trust the forecast less. A first version exists but did not separate good calls from bad ones, so it is parked until it can be tested on the longer history.",
                "<b>Automation gaps.</b> The noon and weekly jobs are written but not yet installed on the machine, and the team page is published by hand.",
                "<b>Same family of judge and judged.</b> The review runs on the same premium model as one version of the forecaster. Code-computed evidence and blind re-checks limit the risk; they do not remove it.",
            ]
        ),
        Spacer(1, 6),
        para("Glossary", "h2"),
        glossary(),
    ]


def glossary() -> Table:
    terms = [
        ("Call", "One published forecast: a central estimate and an 80% range at each of the three horizons."),
        ("Horizon", "Business days ahead: 5, 10 or 21."),
        ("Benchmark", "The random walk: today's price, with a range sized to recent volatility."),
        ("Coverage", "How often the 80% range contained the outcome. Target 80%."),
        ("Score", "Pinball loss: a penalty for distance between the outcome and the whole range. Lower is better."),
        ("Live call", "Made on the day, for the future. The only calls used to fit adjustments."),
        ("Re-run", "A call made after the fact for a past date. Shown, never fitted on."),
        ("Overlay", "The nudge the news-reading step adds to the statistical models' estimate."),
        ("Coach ledger", "The dated list of accepted adjustments. Separate from the forecaster."),
        ("Coached range", "Each day's calls re-derived under the ledger and published alongside."),
        ("Version", "One forecaster build on one language model. Versions are never pooled."),
        ("Challenger", "A copy of the forecaster with a wording change, run in parallel for comparison."),
        ("Independent observation", "Calls whose outcome windows do not overlap. The honest sample size."),
        ("Episode", "A sustained move in one direction. Evidence needs episodes of both signs."),
    ]
    half = (len(terms) + 1) // 2

    def column(items: list[tuple[str, str]]) -> Table:
        rows = [[Paragraph(t, S["gloss_t"]), Paragraph(d, S["gloss_d"])] for t, d in items]
        made = Table(rows, colWidths=[1.0 * inch, CONTENT_W / 2 - 1.0 * inch - 8], hAlign="LEFT")
        made.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
                    ("TOPPADDING", (0, 0), (-1, -1), 2.4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.4),
                    ("LEFTPADDING", (0, 0), (-1, -1), 2),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        return made

    two = Table([[column(terms[:half]), column(terms[half:])]], colWidths=[CONTENT_W / 2, CONTENT_W / 2], hAlign="LEFT")
    two.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 8)]))
    return two


def main() -> None:
    doc = build_document()
    story: list = []
    for index, page in enumerate((page_what, page_adjust, page_scoring, page_weekly, page_usage, page_status)):
        if index:
            story.append(PageBreak())
        story.extend(page())
    doc.build(story)
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
