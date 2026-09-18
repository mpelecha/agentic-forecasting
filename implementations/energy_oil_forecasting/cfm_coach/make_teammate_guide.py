"""Generate the CFM Coach teammate guide as a PDF.

A sibling document to ``CFM_Agent_v5_0_Teammate_Guide_Final.pdf``: same page size,
same palette, same section grammar, so the two read as one set. Kept as a script
rather than a hand-made file so the guide can be regenerated when the corpus grows
and the numbers move.

Every figure below is stamped ``AS_OF`` and was verified against the running code on
that date -- ``report.py`` for the scoreboard, ``config.py`` for the thresholds.
When regenerating, re-check them; a stale number in a shared document is worse than
no number.

Usage::

    uv run --with reportlab python -m energy_oil_forecasting.cfm_coach.make_teammate_guide
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


AS_OF = "14 August 2026"
OUTPUT = Path(__file__).resolve().parent / "CFM_Coach_Teammate_Guide.pdf"

# Palette lifted from the v5.0 guide so the two documents are visually one family.
NAVY = colors.HexColor("#1e2a38")
DEEP = colors.HexColor("#11223e")
TEAL = colors.HexColor("#2a8b82")
GOLD = colors.HexColor("#c49a45")
BLUE = colors.HexColor("#2e67a8")
RULE = colors.HexColor("#d4dde6")
WASH = colors.HexColor("#f1f5f8")
CREAM = colors.HexColor("#f8f1e4")
MUTED = colors.HexColor("#606f83")

PAGE_W, PAGE_H = LETTER
MARGIN = 0.8 * inch
CONTENT_W = PAGE_W - 2 * MARGIN


# -- styles ------------------------------------------------------------------

S = {
    "kicker": ParagraphStyle(
        "kicker", fontName="Helvetica-Bold", fontSize=11, textColor=TEAL, leading=14, spaceAfter=10
    ),
    "cover_title": ParagraphStyle(
        "cover_title", fontName="Helvetica-Bold", fontSize=40, textColor=NAVY, leading=44, spaceAfter=14
    ),
    "cover_sub": ParagraphStyle(
        "cover_sub", fontName="Helvetica", fontSize=11.5, textColor=MUTED, leading=17, spaceAfter=26
    ),
    "section": ParagraphStyle(
        "section", fontName="Helvetica-Bold", fontSize=9.5, textColor=TEAL, leading=12, spaceAfter=4
    ),
    "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=19, textColor=NAVY, leading=23, spaceAfter=11),
    "h2": ParagraphStyle(
        "h2", fontName="Helvetica-Bold", fontSize=11.5, textColor=DEEP, leading=15, spaceBefore=11, spaceAfter=5
    ),
    "body": ParagraphStyle(
        "body", fontName="Helvetica", fontSize=9.4, textColor=NAVY, leading=13.6, spaceAfter=7, alignment=TA_LEFT
    ),
    "bullet": ParagraphStyle(
        "bullet",
        fontName="Helvetica",
        fontSize=9.4,
        textColor=NAVY,
        leading=13.4,
        leftIndent=13,
        bulletIndent=3,
        spaceAfter=3.5,
    ),
    "callout": ParagraphStyle(
        "callout", fontName="Helvetica", fontSize=9.4, textColor=DEEP, leading=13.6, spaceAfter=0
    ),
    "th": ParagraphStyle("th", fontName="Helvetica-Bold", fontSize=8.4, textColor=colors.white, leading=11),
    "td": ParagraphStyle("td", fontName="Helvetica", fontSize=8.4, textColor=NAVY, leading=11.4),
    "td_b": ParagraphStyle("td_b", fontName="Helvetica-Bold", fontSize=8.4, textColor=NAVY, leading=11.4),
    "box_h": ParagraphStyle("box_h", fontName="Helvetica-Bold", fontSize=9.6, textColor=TEAL, leading=12, spaceAfter=3),
    "box_b": ParagraphStyle("box_b", fontName="Helvetica", fontSize=8.8, textColor=NAVY, leading=12),
    "footer": ParagraphStyle("footer", fontName="Helvetica", fontSize=7.6, textColor=MUTED),
}


# -- helpers -----------------------------------------------------------------


def hex_of(color: colors.Color) -> str:
    """`#rrggbb` for use inside Paragraph markup -- `hexval()` returns `0x...`."""
    return "#" + color.hexval()[2:]


def para(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def bullets(items: list[str]) -> list[Paragraph]:
    return [Paragraph(item, S["bullet"], bulletText="●") for item in items]


def keep(*flowables) -> KeepTogether:
    """Bind a heading to what it introduces, so a table never strands its title.

    Only for blocks comfortably shorter than a page; taller tables are left to split
    with their header row repeating, which reads better than a forced overflow.
    """
    return KeepTogether(list(flowables))


def section(number: str, name: str, headline: str) -> list:
    """Build a numbered section opener in the v5.0 guide's grammar."""
    bar = Table([[Paragraph(f"{number} | {name.upper()}", S["section"])]], colWidths=[CONTENT_W])
    bar.setStyle(
        TableStyle(
            [
                ("LINEBELOW", (0, 0), (-1, -1), 1.6, TEAL),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    return [bar, Spacer(1, 9), para(headline, "h1")]


def table(rows: list[list[str]], widths: list[float], *, bold_first_col: bool = False) -> Table:
    """Header-shaded, zebra-striped table matching the v5.0 guide."""
    data = [[Paragraph(cell, S["th"]) for cell in rows[0]]]
    for row in rows[1:]:
        style = "td_b" if bold_first_col else "td"
        data.append([Paragraph(row[0], S[style]), *[Paragraph(cell, S["td"]) for cell in row[1:]]])

    made = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.5, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]
    for index in range(1, len(data)):
        if index % 2 == 0:
            commands.append(("BACKGROUND", (0, index), (-1, index), WASH))
    made.setStyle(TableStyle(commands))
    return made


def callout(title: str, text: str, *, tone: str = "teal") -> Table:
    """Build a boxed aside -- used for boundaries, warnings and the bottom line."""
    accent = {"teal": TEAL, "gold": GOLD, "blue": BLUE}[tone]
    background = {"teal": WASH, "gold": CREAM, "blue": WASH}[tone]
    inner = [
        Paragraph(f'<font color="{hex_of(accent)}"><b>{title}</b></font>', S["callout"]),
        Spacer(1, 3),
        Paragraph(text, S["callout"]),
    ]
    box = Table([[inner]], colWidths=[CONTENT_W])
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("LINEBEFORE", (0, 0), (0, -1), 2.5, accent),
                ("BOX", (0, 0), (-1, -1), 0.5, RULE),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    return box


def stage_strip(stages: list[tuple[str, str, str]]) -> Table:
    """Build the numbered pipeline strip, as on page 3 of the v5.0 guide."""
    cells = []
    for number, title, detail in stages:
        cells.append(
            [
                Paragraph(f'<font color="{hex_of(TEAL)}" size="13"><b>{number}</b></font>', S["box_b"]),
                Paragraph(f"<b>{title}</b>", S["box_b"]),
                Paragraph(detail, S["box_b"]),
            ]
        )
    # One row, each cell a stack of flowables (number, title, detail).
    width = CONTENT_W / len(stages)
    made = Table([cells], colWidths=[width] * len(stages), hAlign="LEFT")
    made.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), WASH),
                ("BOX", (0, 0), (-1, -1), 0.5, RULE),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, RULE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return made


# -- page furniture ----------------------------------------------------------


def draw_cover(canvas, doc):  # noqa: ANN001, ARG001
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, PAGE_H - 0.32 * inch, PAGE_W, 0.32 * inch, stroke=0, fill=1)
    canvas.setFillColor(TEAL)
    canvas.rect(0, PAGE_H - 0.32 * inch, 2.4 * inch, 0.32 * inch, stroke=0, fill=1)
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, PAGE_W, 0.16 * inch, stroke=0, fill=1)
    canvas.restoreState()


def draw_page(canvas, doc):  # noqa: ANN001
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, PAGE_H - 0.62 * inch, PAGE_W - MARGIN, PAGE_H - 0.62 * inch)
    canvas.setFont("Helvetica", 7.6)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, PAGE_H - 0.55 * inch, "CFM Coach  |  Teammate Guide")
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 0.55 * inch, str(doc.page))
    canvas.setFillColor(RULE)
    canvas.rect(MARGIN, 0.52 * inch, CONTENT_W, 0.5, stroke=0, fill=1)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7)
    canvas.drawString(MARGIN, 0.36 * inch, f"Figures as of {AS_OF}. Regenerate with make_teammate_guide.py.")
    canvas.restoreState()


def build_document() -> BaseDocTemplate:
    doc = BaseDocTemplate(
        str(OUTPUT),
        pagesize=LETTER,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=0.72 * inch,
        title="CFM Coach - Teammate Guide",
        author="CFM Coach",
        subject="Goals, architecture, calibration harness, findings, and prioritized next steps",
    )
    cover_frame = Frame(MARGIN, 0.9 * inch, CONTENT_W, PAGE_H - 2.0 * inch, id="cover")
    body_frame = Frame(MARGIN, 0.72 * inch, CONTENT_W, PAGE_H - 1.5 * inch, id="body")
    doc.addPageTemplates(
        [
            PageTemplate(id="cover", frames=[cover_frame], onPage=draw_cover),
            PageTemplate(id="body", frames=[body_frame], onPage=draw_page),
        ]
    )
    return doc


# -- content -----------------------------------------------------------------


def cover() -> list:
    facts = [
        (
            "Purpose",
            "A WTI forecast that can be relied on for direction <i>and</i> risk, plus an explicit "
            "signal for when to distrust it.",
        ),
        (
            "Core principle",
            "The coach measures and proposes. Every change waits for a human, and "
            "<b>cfm_agent_v_5_0 is never modified</b>.",
        ),
        (
            "How it learns",
            "M1 calibration of the constants (built), M2 memory of the agent's own track record, "
            "M3a analogue lookup over 4,700 historical days.",
        ),
        (
            "State today",
            "Recorder live and scheduled. M1 harness built and tested. One clean live record; "
            "first real evidence resolves 21 Aug 2026.",
        ),
    ]
    cells = [[Paragraph(title, S["box_h"]), Paragraph(text, S["box_b"])] for title, text in facts]
    grid = Table(
        [[cells[0], cells[1]], [cells[2], cells[3]]],
        colWidths=[CONTENT_W / 2] * 2,
        hAlign="LEFT",
    )
    grid.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LINEABOVE", (0, 0), (-1, 0), 0.5, RULE),
                ("LINEBELOW", (0, 0), (-1, -1), 0.5, RULE),
                ("TOPPADDING", (0, 0), (-1, -1), 12),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 18),
                ("LEFTPADDING", (0, 0), (0, -1), 0),
            ]
        )
    )

    return [
        Spacer(1, 0.5 * inch),
        para("TEAMMATE GUIDE", "kicker"),
        para("CFM Coach", "cover_title"),
        para(
            "A learning loop around CFM Agent v5.0. It records every forecast the agent makes, waits "
            "for the world to resolve it, scores what happened, and proposes &mdash; never applies &mdash; "
            "calibration corrections.",
            "cover_sub",
        ),
        grid,
        Spacer(1, 0.45 * inch),
        para(
            f'<font color="{hex_of(MUTED)}" size="9">Prepared for internal team collaboration<br/>{AS_OF}</font>',
            "body",
        ),
    ]


def executive_overview() -> list:
    return [
        *section("1", "Executive overview", "What CFM Coach does"),
        para(
            "CFM Coach is a separate package that sits <i>around</i> CFM Agent v5.0. The agent produces a "
            "probabilistic WTI forecast; the coach records exactly what produced it, waits for the horizon "
            "to elapse, measures how good it was, and works out which constants should change. It is the "
            "measurement instrument the agent has never had."
        ),
        para("The short version", "h2"),
        para(
            "v5.0 is a well-controlled pipeline with no feedback. It converts the LLM's categorical "
            "judgment into numbers using a set of fixed constants &mdash; action fractions, uncertainty "
            "multipliers, tier thresholds &mdash; and nothing has ever checked whether those constants are "
            "right. The coach closes that loop, slowly and conservatively, without touching the agent."
        ),
        *bullets(
            [
                "<b>Record</b> every run with enough detail to reproduce it exactly, months later.",
                "<b>Resolve</b> each horizon against what WTI actually did.",
                "<b>Score</b> the agent against the raw ensemble and against candidate calibrations, "
                "on identical days.",
                "<b>Gate</b> any proposed change behind a locked statistical test.",
                "<b>Propose</b> the change to a human. Nothing is ever applied automatically.",
            ]
        ),
        Spacer(1, 5),
        callout(
            "The one-line boundary",
            "The coach reads v5.0's outputs and injects settings at construction time. It never edits the "
            "agent package, so v5.0's manifest, its provenance claims, and its 43-file integrity check all "
            "stay true. Verified on every change.",
        ),
        para("What the coach produces", "h2"),
        table(
            [
                ["Output", "What it contains"],
                [
                    "Run record",
                    "One JSON file per run (~90&nbsp;KB): the LLM assessment, research packet, per-model and "
                    "ensemble forecasts, policy decision, transformation audit, and the five identity fields "
                    "needed to replay it &mdash; settings, calibration version, model, package fingerprint, prompt version.",
                ],
                [
                    "Scoreboard",
                    "Pinball loss, CRPS, MAE, 80% interval coverage and mean width for three variants on "
                    "identical days: the raw ensemble, the agent as published, and the calibration in force now.",
                ],
                [
                    "Fidelity report",
                    "Proof that replaying each record under its own calibration reproduces its recorded "
                    "quantiles exactly. If it ever fails, every fit built on that corpus is invalid.",
                ],
                [
                    "Comparison verdict",
                    "Nine pass/fail conditions, a paired effect size, an origin-blocked bootstrap p-value, "
                    "and the proposed value shrunk halfway toward the incumbent.",
                ],
                [
                    "Trust report",
                    "A tier (high / medium / low / no_basis) plus one line naming the dominant driver. "
                    "Built, and deliberately <b>not</b> wired in &mdash; see section 9.",
                ],
            ],
            [1.25 * inch, CONTENT_W - 1.25 * inch],
            bold_first_col=True,
        ),
    ]


def objective() -> list:
    return [
        PageBreak(),
        *section("2", "The objective", 'Why the goal is "when to distrust it"'),
        para(
            "The instinct is to aim at accuracy. The arithmetic argues against making that the primary "
            "target. Improving accuracy means moving a <i>level</i> against a very noisy signal, and at a "
            "21-day horizon there are only about four independent observations per year &mdash; even running "
            "daily, because overlapping windows are not independent. Detecting a real improvement that way "
            "takes years."
        ),
        para(
            "Knowing <i>when to distrust</i> the forecast is a different statistical object: a <b>ranking</b>. "
            "Were the days the system flagged as low-confidence actually worse than the days it did not? "
            "Rankings converge far faster than levels &mdash; months, not years. It is also the thing an LLM "
            "forecaster can do that ARIMA cannot: report what it looked for, what it found, what conflicted, "
            "and what it could not resolve."
        ),
        # Eleven rows plus a heading, bound together: unbound, this table stranded
        # its last four rows on a page of their own.
        keep(
            para("The requirements that shape everything else", "h2"),
            table(
                [
                    ["#", "Requirement", "Status"],
                    ["R1", "Honest intervals: P10&ndash;P90 converging on 80% coverage.", "0 of 12 &mdash; open"],
                    ["R2", "A trust output, separate from the forecast.", "Built, held back"],
                    ["R3", "Cutoff-honest: no trust signal may depend on the outcome it assesses.", "Enforced"],
                    ["R4", "Trust validated by rank, or retired.", "Blocked on corpus"],
                    ["R5", "Out-of-distribution flagging. Annotates, never suppresses.", "Built, weak"],
                    ["R6", "Every run replayable.", "<b>Done and verified</b>"],
                    ["R7", "Only live forward-dated runs are fitting evidence.", "<b>Enforced by construction</b>"],
                    ["R8", "Nothing auto-applies to the agent.", "<b>Enforced</b>"],
                    ["R9", "Audit isolation preserved.", "<b>Enforced</b>"],
                    ["R10", "Improvement is a paired comparison on identical days.", "<b>Enforced by the gate</b>"],
                    ["R11", "Stage 1 is additive only &mdash; no existing file modified.", "<b>Holds</b>"],
                ],
                [0.42 * inch, CONTENT_W - 1.92 * inch, 1.5 * inch],
                bold_first_col=True,
            ),
        ),
        # Bound as one block so the table never splits across the section break.
        keep(
            para("The architectural unlock", "h2"),
            para(
                "R2 &mdash; a trust output &mdash; turns out to be far cheaper than it sounds, because v5.0 "
                "already computes a great deal of trust-bearing signal and then discards it <i>by design</i>. "
                "Audit isolation correctly forbids these from influencing the forecast &mdash; but a trust "
                "score is not the forecast. A second output that consumes exactly what the forecast is "
                "forbidden to consume respects the architecture and recovers signal that is currently "
                "thrown away."
            ),
            table(
                [
                    ["Signal v5.0 computes", "Where it lives", "Used for the forecast?"],
                    ["Evidence tier and tier level", "policy_decision", "Shapes caps, never reported"],
                    ["Material evidence conflict", "llm_context_assessment", "Disqualifies, then discarded"],
                    ["Source Validator findings", "source_validation_audit", "Audit-only"],
                    ["Claim-support entailment findings", "claim_support_findings", "Audit-only"],
                    ["Model disagreement std", "numerical_suite_audit", "Unused"],
                    ["Warnings, floor applied", "forecast_transformation", "Not surfaced"],
                    ["Market diagnostics (VIX, vol, contango)", "numerical_suite_audit", "Unused"],
                ],
                [2.5 * inch, 2.0 * inch, CONTENT_W - 4.5 * inch],
                bold_first_col=True,
            ),
            Spacer(1, 8),
            callout(
                "This is the entire source of the stage-1 trust report",
                "No case base is required for it, and nothing about it touches the forecast path. It is "
                "recovered signal, not new machinery &mdash; which is why R2 was scoped into stage 1 at all.",
            ),
        ),
    ]


def architecture() -> list:
    return [
        PageBreak(),
        *section("3", "Architecture", "Two loops, three learning mechanisms"),
        para(
            "The coach runs two workflows on very different clocks. One runs every business day and costs "
            "about four minutes. The other runs rarely, only when enough horizons have resolved, and involves "
            "no LLM at all."
        ),
        para("Workflow A &mdash; the daily record (every business day)", "h2"),
        stage_strip(
            [
                ("1", "Calibration", "Look up the version in force for today's cutoff."),
                ("2", "Inject", "Build agent settings from that version and construct v5.0."),
                ("3", "Run", "v5.0 executes unmodified: research, assessment, policy, engine."),
                ("4", "Layer", "Apply the coach's post-engine centre/width layer."),
                ("5", "Record", "Write one replayable RunRecord to the corpus."),
            ]
        ),
        Spacer(1, 8),
        para(
            "Every day produces a record whether or not anything is learned from it. <b>This loop is the "
            "corpus</b>, and it cannot be reconstructed later: re-running the agent at a past cutoff searches "
            "today's web, so a missed day is evidence permanently lost. The job is scheduled on weekdays at "
            "08:00 local."
        ),
        para("Workflow B &mdash; the reflection cycle (periodic, no LLM)", "h2"),
        stage_strip(
            [
                ("1", "Resolve", "Join due horizons to realized WTI."),
                ("2", "Verify", "Fidelity-check the corpus; stop if it drifted."),
                ("3", "Replay", "Re-derive candidate and incumbent on the same days."),
                ("4", "Score", "Pinball, CRPS, coverage."),
                ("5", "Gate", "Nine locked conditions."),
                ("6", "Propose", "Queue a shrunk change for human review."),
            ]
        ),
        Spacer(1, 8),
        para("The three learning mechanisms", "h2"),
        table(
            [
                ["", "Mechanism", "Who learns", "Changes the agent?", "State"],
                [
                    "M1",
                    "Calibrate the constants that convert judgment into quantiles",
                    "The machinery",
                    "No",
                    "<b>Built</b>",
                ],
                ["M2", "Memory of the agent's own track record", "The LLM", "Yes", "Designed"],
                ["M3a", "Analogue lookup feeding a trust report", "Nobody &mdash; it is a read", "No", "Built, held"],
                ["M3b", "Analogue injection into the prompt", "The LLM", "Yes", "Deferred"],
            ],
            [0.42 * inch, 2.35 * inch, 1.15 * inch, 1.05 * inch, CONTENT_W - 4.97 * inch],
            bold_first_col=True,
        ),
        Spacer(1, 6),
        callout(
            "Why M1 comes first",
            "M1 is the measurement instrument for M2. M2 changes what the LLM says, and nothing else can tell "
            "you whether that helped. Without a paired harness you get what this repo's earlier adaptive agent "
            "got: CRPS 9.604 &rarr; 9.116 on n=22, with no standard error and no way to know whether anything "
            "actually happened.",
            tone="blue",
        ),
    ]


def recorder() -> list:
    return [
        PageBreak(),
        *section("4", "The recorder", "How replayability is achieved without touching v5.0"),
        para(
            "A calibration corpus needs five things per run that v5.0 does not persist: the settings it ran "
            "under, the calibration version they came from, the model string, a fingerprint of the agent "
            "package, and the prompt version. The insight that makes this free is that <b>the caller already "
            "knows all five before the run starts</b>."
        ),
        para(
            "So the coach constructs the settings itself, injects them through v5.0's own "
            "<font face='Courier'>build_cfm_agent_config(settings=...)</font> entry point, takes the "
            "predictions back, and pairs them with what it already held. The agent never learns the coach "
            "exists. The package fingerprint is a hash of v5.0's shipped manifest, so if the agent's constants "
            "ever change underneath the corpus, the affected records become visibly a different system."
        ),
        para("Provenance: the field that protects every conclusion", "h2"),
        para(
            "Only runs issued on their own cutoff count as fitting evidence. A run re-executed at a past "
            "cutoff re-searches <i>today's</i> web, and even when every cited source predates the cutoff, the "
            "<i>selection</i> of what to look at is shaped by what turned out to matter. v5.0's leakage "
            "verifier inspects content, not retrieval, so it cannot catch this. The coach excludes such runs "
            "by construction rather than by discipline."
        ),
        table(
            [
                ["Provenance", "Meaning", "Fitting evidence?"],
                ["live_forward", "Issued on or after its own cutoff. Genuinely forward-looking.", "<b>Yes</b>"],
                ["replayed_live_search", "Re-executed at a past cutoff against today's web.", "No"],
                ["backfill_cached_news", "Reconstructed from cached research.", "No"],
                ["ensemble_only", "Numerical suite with no LLM stage.", "No"],
            ],
            [1.5 * inch, CONTENT_W - 2.75 * inch, 1.25 * inch],
            bold_first_col=True,
        ),
        Spacer(1, 6),
        callout(
            "This is not a hypothetical distinction",
            "Both audit files that existed before the coach was built turned out to have been issued "
            "<b>164 days after</b> their cutoff. They are recorded as replayed and excluded. Section 9 "
            "explains what that cost us.",
            tone="gold",
        ),
    ]


def replay_section() -> list:
    return [
        PageBreak(),
        *section("5", "Replay and calibration", "M1: re-deriving a forecast without an LLM"),
        para(
            "The core of M1 is the ability to take a stored run and ask: <i>what would this same judgment have "
            "produced under different constants?</i> Answering that takes microseconds, offline, with no "
            "search and no tokens."
        ),
        para(
            "It is exact rather than approximate because of one property of v5.0: <b>the prompt contains no "
            "numeric constants.</b> The LLM emits categorical judgments &mdash; "
            "<font face='Courier'>large_up</font>, <font face='Courier'>substantially_wider</font> &mdash; and "
            "Python owns every number that turns those into quantiles. The LLM's output is therefore "
            "independent of the constants by construction, so re-deciding a stored assessment under new "
            "constants is a valid counterfactual, not a guess."
        ),
        para(
            "The arithmetic is not reimplemented. The coach <i>imports</i> v5.0's own "
            "<font face='Courier'>EvidencePolicy</font> and <font face='Courier'>PythonForecastEngine</font> "
            "and feeds them reconstructed inputs, so its model of the agent cannot drift from the agent."
        ),
        para("The two levers", "h2"),
        table(
            [
                ["Lever", "Where it acts", "What it changes"],
                [
                    "Agent constants",
                    "Injected into v5.0's settings, so the policy and engine see them",
                    "Action fractions (10/20/30% of width), uncertainty multipliers, tier thresholds, novelty dampener",
                ],
                [
                    "Calibration layer",
                    "Coach-owned, applied after v5.0's engine",
                    "Per-horizon centre gain &lambda; and width scale &omega;. &lambda;=1, &omega;=1 "
                    "reproduces v5.0 bit-for-bit",
                ],
            ],
            [1.3 * inch, 2.0 * inch, CONTENT_W - 3.3 * inch],
            bold_first_col=True,
        ),
        Spacer(1, 5),
        para(
            "The layer exists because the highest-leverage lever &mdash; <i>base</i> interval width &mdash; has no "
            "in-engine control. v5.0's six uncertainty multipliers only fire when the LLM asks for a width "
            "change, and a neutral decision bypasses them entirely. Owning that lever in the coach keeps the "
            "agent unmodified."
        ),
        para("Both levers, measured on a real record", "h2"),
        para(
            "Run <font face='Courier'>2026-03-03</font> at h=21, where the agent proposed "
            "<font face='Courier'>large_up</font> and cleared Tier 3:"
        ),
        table(
            [
                ["Calibration", "P50", "Overlay", "P10&ndash;P90 width"],
                ["v001 (identity &mdash; what shipped)", "72.964", "+3.583", "15.527"],
                ["Layer width scale &omega; = 1.5", "72.964", "+3.583", "<b>23.290</b>"],
                ["Agent constant: large fraction 0.30 &rarr; 0.20", "<b>71.770</b>", "<b>+2.389</b>", "15.527"],
            ],
            [2.7 * inch, 1.0 * inch, 1.0 * inch, CONTENT_W - 4.7 * inch],
            bold_first_col=True,
        ),
        Spacer(1, 5),
        para(
            "Width scales with the centre untouched; the agent constant scales the overlay by exactly "
            "0.20/0.30. The two levers are independent, which is what lets the gate attribute a change to one "
            "hypothesis at a time."
        ),
        para("The fidelity check", "h2"),
        callout(
            "The check that keeps the coach honest",
            "Replaying a record under the calibration that actually produced it must reproduce its recorded "
            "quantiles <b>exactly</b>, or the harness raises and stops. All five records currently reproduce "
            "bit-for-bit &mdash; max absolute error <b>0.0</b>, not merely within tolerance. That includes one "
            "record whose settings were reconstructed after the fact, which independently confirms the "
            "reconstruction was correct.",
        ),
    ]


def scoring_section() -> list:
    return [
        PageBreak(),
        *section("6", "Scoring", "Three metrics, each answering a different question"),
        para(
            "The repo scored forecasts with CRPS only. CRPS is a reasonable summary and the wrong primary "
            "instrument for calibration work: it blends everything into one number, so it cannot tell you "
            "<i>which part</i> of the distribution was wrong. The coach's entire job is deciding whether to "
            "move the centre or the width, so it needs a loss that decomposes by quantile level."
        ),
        table(
            [
                ["Metric", "What it answers", "Role"],
                [
                    "Pinball loss",
                    "How wrong was each quantile, separately? Minimised only by the true quantile.",
                    "<b>Primary.</b> New to this repo &mdash; no quantile-loss function existed.",
                ],
                [
                    "CRPS",
                    "One number per forecast, for comparability with existing repo results.",
                    "Reported. Both the quantile-integrated form and the framework's ensemble approximation.",
                ],
                [
                    "80% coverage",
                    "Did the outcome land inside P10&ndash;P90? Target 80%.",
                    "R1's actual target. A rate, so it needs accumulated runs.",
                ],
            ],
            [1.15 * inch, 2.35 * inch, CONTENT_W - 3.5 * inch],
            bold_first_col=True,
        ),
        Spacer(1, 6),
        callout(
            "Effective sample size is reported, and it is brutal",
            "Horizons overlap, so 60 daily runs at h=21 carry roughly three genuinely independent "
            "observations. Every scoreboard states the effective count next to the raw one, so <i>n</i>=36 "
            "never implies a precision that is not there. Today the effective count is <b>0.7</b>.",
            tone="gold",
        ),
    ]


def gate_section() -> list:
    return [
        PageBreak(),
        *section("7", "The locked gate", "The only component allowed to say a change is an improvement"),
        para(
            "Everything else in the coach measures. This decides. The separation is deliberate: a scoreboard "
            "that also gets to declare victory will eventually be tuned until it does, and the failure is "
            "silent because every intermediate number still looks correct."
        ),
        para(
            "Nine conditions, all of which must hold. Six concern <i>what the corpus is</i>; three concern "
            "<i>what the numbers say</i>. The ordering matters &mdash; the statistical tests run last, because a "
            "p-value computed over a contaminated or pooled corpus is worse than no p-value: it launders the "
            "contamination into something that looks like evidence."
        ),
        table(
            [
                ["#", "Condition", "Why it exists"],
                ["1", "Provenance clean", "Only live_forward records. Replayed history is not evidence."],
                ["2", "Single prompt version", "A different prompt is a different agent."],
                ["3", "Single package fingerprint", "If v5.0 changed underneath, old constants do not transfer."],
                ["4", "Sufficient origins", "Counted in origins, never in (origin, horizon) rows. Needs 12."],
                ["5", "Holdout respected", "Judged only on origins after what the candidate was fitted on. Needs 6."],
                ["6", "Paired on identical days", "Both variants scored on exactly the same keys."],
                ["7", "Single tunable", "One hypothesis per cycle, or the result cannot be attributed."],
                ["8", "Effect size sufficient", "Detectable is not the same as worth applying. Floor 3%."],
                ["9", "Bootstrap significant", "One-sided, resampling origins. &alpha; = 0.10."],
            ],
            [0.35 * inch, 1.7 * inch, CONTENT_W - 2.05 * inch],
            bold_first_col=True,
        ),
        para("Why the bootstrap resamples origins, not rows", "h2"),
        para(
            "The three horizons of one run share an assessment, a research packet, a market state and "
            "overlapping outcome windows. They are close to one observation, not three. Resampling "
            "(origin, horizon) rows would treat them as independent, shrink the standard error by roughly "
            "&radic;3, and manufacture significance out of correlation. So a resample draws whole origins and "
            "carries all of their horizons with it. There is a test asserting that the naive version looks "
            "<i>more</i> significant than the blocked one on identical data &mdash; the overclaim this exists to "
            "prevent."
        ),
        para("Shrinkage, and what a pass actually means", "h2"),
        para(
            "A fitted value is a point estimate from a small, autocorrelated sample; its sampling error is "
            "large relative to the change being proposed, so applying it in full systematically over-corrects. "
            "A passed candidate is therefore shrunk halfway toward the incumbent before it is proposed &mdash; a "
            "little bias for a large reduction in the variance of what reaches a live forecast."
        ),
        callout(
            "Passing is not applying",
            "A passed verdict produces a <i>proposal</i> in a review queue, carrying the diff, the lineage, "
            "the statistics and the shrunk value. A human approves it, or it does not happen. Nothing in the "
            "coach writes a calibration version on its own.",
        ),
    ]


def reading_section() -> list:
    return [
        PageBreak(),
        *section("8", "Reading a result", "A reviewer checklist for the scoreboard"),
        para(
            "One command produces the whole picture in about two seconds, with no LLM and no network: "
            "<font face='Courier'>uv run python -m energy_oil_forecasting.cfm_coach.report</font>. "
            "Read it in this order."
        ),
        table(
            [
                ["Step", "What to check", "What good looks like"],
                [
                    "1. Fidelity",
                    "The drift banner at the top",
                    "All records replay exactly. Any drift invalidates every fit below it.",
                ],
                [
                    "2. Corpus",
                    "Records, and how many are fitting-eligible",
                    "The fitting-eligible count is the one that matters.",
                ],
                [
                    "3. Resolution",
                    "Resolved vs pending, and the price-data frontier",
                    "Pending is normal and expected; most of the corpus is unresolved most of the time.",
                ],
                [
                    "4. Fitting band",
                    "The live_forward-only section",
                    "This is the only band that is evidence. Today it is empty and says so.",
                ],
                [
                    "5. Variants",
                    "ensemble vs agent vs current",
                    "'current' equals 'agent' exactly while v001 is in force &mdash; proof the baseline is a true no-op.",
                ],
                ["6. Coverage", "cover80 against the 80% target", "R1's headline. Currently 0%."],
                [
                    "7. Effective n",
                    "The independent-observation count",
                    "Governs what may be concluded. Compare it against the raw n.",
                ],
            ],
            [0.95 * inch, 1.85 * inch, CONTENT_W - 2.8 * inch],
            bold_first_col=True,
        ),
        para("The scoreboard today", "h2"),
        table(
            [
                ["Variant", "n", "Pinball", "CRPS", "MAE", "Cover 80%", "Width"],
                ["agent (as published)", "12", "11.5434", "21.4171", "25.58", "<b>0.0%</b>", "11.08"],
                ["current (v001)", "12", "11.5434", "21.4171", "25.58", "0.0%", "11.08"],
                ["ensemble (no overlay)", "12", "12.3681", "22.8239", "26.93", "0.0%", "9.74"],
            ],
            [1.75 * inch, 0.35 * inch, 0.8 * inch, 0.8 * inch, 0.7 * inch, 0.85 * inch, CONTENT_W - 5.25 * inch],
            bold_first_col=True,
        ),
        Spacer(1, 6),
        callout(
            "Do not over-interpret this table",
            "All twelve resolved rows come from two March cutoffs that were <b>replayed</b>, not run live. "
            "The overlay appears to beat the raw ensemble by 6.7% on pinball. That is not evidence the overlay "
            "works: it is two cutoffs, re-executed against a web that already knew what happened, with roughly "
            "<b>0.7</b> effective independent observations. The report bands it separately for exactly this "
            "reason.",
            tone="gold",
        ),
    ]


def findings_section() -> list:
    return [
        PageBreak(),
        *section("9", "What we have learned so far", "Five findings, four of them uncomfortable"),
        para(
            "The coach has not yet calibrated anything &mdash; it cannot, until the corpus exists. What it has "
            "already done is measure things nobody had measured, and four of those measurements contradicted "
            "assumptions the project was built on. That is the value delivered to date, and it is worth "
            "stating plainly."
        ),
        para("1. The only prior evidence for the overlay was invalid", "h2"),
        para(
            "Both pre-existing audit files were issued 164 days after their cutoff. The '+10.9% overlay "
            "improvement on 6 of 6 horizons' measured early in this project comes entirely from those two "
            "replayed runs. It is <b>not</b> evidence that the overlay helps live. This is the finding that "
            "motivated making provenance a first-class, enforced field."
        ),
        para("2. The trust tier does not discriminate", "h2"),
        para(
            "The trust reporter returns <font face='Courier'>high</font> for all three original records &mdash; "
            "including the two whose forecasts were off by $27&ndash;$35 with zero interval coverage. That is R4's "
            "retirement condition firing before a single forecast resolved. Neither sub-check caught March 2026: "
            "the out-of-distribution flag put that day at the 87.9th percentile of neighbour distance against a "
            "99th-percentile threshold, and the five nearest historical analogues implied widths of "
            "1.5&times;/0.9&times;/1.1&times; the ensemble against realized moves of +$12/+$25/+$29."
        ),
        callout(
            "The decision taken",
            "Hold all of it. Nothing trust-related is wired into the daily path until M1 can rank it against "
            "realized error. The code stays in the tree, unused. A trust tier that does not separate outcomes "
            "is worse than none, because it is believed.",
            tone="gold",
        ),
        para("3. Citation bookkeeping &mdash; not market judgment &mdash; dominates run-to-run variance", "h2"),
        para(
            "The same cutoff was run twice, same code, same model, about fifteen hours apart. On 2026-03-03 the "
            "agent formed the <b>identical</b> view both times: same actions on all three horizons, same "
            "confidence, same novelty. The published output went from <b>+$3.58 to +$0.00</b>."
        ),
        table(
            [
                ["Cutoff", "Issued", "LLM proposed (h=21)", "Policy gave", "Why"],
                ["2026-03-03", "06:34", "large_up", "large_up, tier <b>strong</b>", "&mdash;"],
                [
                    "2026-03-03",
                    "21:30",
                    "large_up",
                    "<b>no_change</b>, tier none",
                    "Cited claims failed the source-subset contract",
                ],
                ["2026-03-02", "23:08", "moderate_up", "moderate_up, corroborated", "11 resolved publishers"],
                ["2026-03-02", "21:25", "large_up", "small_up, limited", "Only 5 resolved publishers"],
            ],
            [0.9 * inch, 0.6 * inch, 1.3 * inch, 1.45 * inch, CONTENT_W - 4.25 * inch],
            bold_first_col=True,
        ),
        Spacer(1, 6),
        para(
            "The second run wired its claim &rarr; summary &rarr; source citations in a way the evidence policy "
            "rejected. The policy behaved exactly as specified; the variance is upstream, in how claims are "
            "assembled. Three consequences, and they are large:"
        ),
        *bullets(
            [
                "Calibrating the overlay means fitting a signal that is being <b>randomly zeroed by an "
                "unrelated failure mode</b>. Any centre gain fitted across such runs is biased toward zero "
                "with inflated variance, and 40 records will not be enough if a meaningful share are nullified.",
                "The highest-leverage fix is <b>not a calibration constant</b>. It is the claim-building step.",
                "It is measurable <b>immediately</b>, with no waiting for horizons to resolve.",
            ]
        ),
        para(
            "This was invisible before, because nothing had ever compared two runs of the same cutoff. "
            "Section 11 treats it as priority two, and explains why the timing is urgent."
        ),
        para("4. The market-state signals are weaker than assumed", "h2"),
        para(
            "Measured over 4,679 historical business days, Spearman rank correlation against the absolute forward move:"
        ),
        table(
            [
                ["Signal", "|move 5b|", "|move 10b|", "|move 21b|"],
                ["vix_level_l1b", "<b>0.255</b>", "<b>0.231</b>", "<b>0.236</b>"],
                ["realized_volatility_21b", "0.175", "0.144", "0.120"],
                ["drawdown_63b", "&minus;0.174", "&minus;0.133", "&minus;0.109"],
                ["return_1b", "0.103", "0.070", "0.074"],
                ["oil_curve_contango_l1b", "&minus;0.086", "&minus;0.087", "&minus;0.097"],
                ["jump_zscore_63b", "0.045", "0.018", "0.020"],
            ],
            [2.4 * inch, 1.1 * inch, 1.1 * inch, CONTENT_W - 4.6 * inch],
            bold_first_col=True,
        ),
        Spacer(1, 6),
        *bullets(
            [
                "<b>Jump z-score has essentially no predictive power.</b> This kills the earlier hypothesis "
                "that the overlay's reach should scale with market abnormality &mdash; a hypothesis built on the "
                "2.93 jump z-score observed on 2026-03-03. A large jump today does not predict a large move next.",
                "<b>The best state signal is VIX, and it is still weak</b> (&rho; &asymp; 0.24). No state-based "
                "trust tier will discriminate strongly.",
                "<b>The usable effect is in the tail, not the median.</b> From the calmest to the most violent "
                "volatility quintile, the median 21-day absolute move rises only $4.11 &rarr; $5.40 &mdash; but the "
                "90th percentile rises $9.21 &rarr; <b>$16.07</b>.",
            ]
        ),
        Spacer(1, 4),
        callout(
            "The leading calibration hypothesis, and it is now testable end to end",
            "<b>Width should scale with volatility. The centre should not.</b> This serves R1 directly, it is "
            "supported by 4,679 days of history rather than by a handful of runs, and the harness can now "
            "measure whether it actually helps.",
        ),
        para("5. The replay is exact, and the baseline is a true no-op", "h2"),
        para(
            "The one unambiguously good finding. All five records reproduce their recorded quantiles "
            "bit-for-bit under their own calibration, and the scoreboard's 'current' column matches the "
            "'agent' column to the digit. That means the coach's model of v5.0 is faithful, and that v001 "
            "changes nothing &mdash; so any future difference is attributable to a deliberate calibration rather "
            "than to drift."
        ),
    ]


def roadmap_section() -> list:
    return [
        PageBreak(),
        *section("10", "Roadmap", "What is built, what is designed, what is deferred"),
        table(
            [
                ["Stage", "Contents", "State"],
                [
                    "1a &mdash; Recorder",
                    "Run record schema, store, ledger, daily entry point, backfill",
                    "<b>Complete and verified</b>",
                ],
                [
                    "1b &mdash; Trust report and M3a",
                    "State diagnostics, 4,700-case analogue base, trust reporter",
                    "Built; <b>deliberately not wired in</b> pending R4",
                ],
                [
                    "1c &mdash; M1 harness",
                    "Replay, scoring, outcomes, scoreboard, locked gate",
                    "<b>Built and tested</b>; idle until the corpus arrives",
                ],
                [
                    "1c &mdash; Workflow B",
                    "Candidate generation, proposal builder, review queue, cycle runner",
                    "Not built",
                ],
                [
                    "1d &mdash; M2 memory",
                    "Self-audit of novelty, confidence, persistence and claim-type calibration, rendered "
                    "into the prompt as a compact table",
                    "Designed",
                ],
                ["2 &mdash; M3b", "Analogue injection into the prompt", "Deferred"],
            ],
            [1.55 * inch, CONTENT_W - 3.35 * inch, 1.8 * inch],
            bold_first_col=True,
        ),
        para("M2: what the agent would learn about itself", "h2"),
        para(
            "The agent produces judgments, not calculations, so its self-audit is a judgment audit. All four "
            "targets reach the forecast through a <i>typed</i> channel &mdash; the LLM's categorical outputs are "
            "load-bearing inputs to a policy Python still owns &mdash; so this is not prose-to-number leakage."
        ),
        table(
            [
                ["Target", "The question it answers"],
                [
                    "Novelty calibration",
                    "Highest leverage: novelty both gates eligibility and applies the 0.5&times; dampener. "
                    "'When I called this likely new, did the price move like new information?'",
                ],
                [
                    "Confidence calibration",
                    "Do 0.9-confidence calls actually beat 0.6-confidence calls? The tiers assume so; nothing has checked.",
                ],
                [
                    "Persistence",
                    "The persistence profile is collected every horizon and used by nothing. Did 'persistent' calls hold at h=21?",
                ],
                [
                    "Claim-type efficacy",
                    "The Tier 3 gate bets that physical evidence beats market-reaction evidence. Does it?",
                ],
            ],
            [1.4 * inch, CONTENT_W - 1.4 * inch],
            bold_first_col=True,
        ),
        Spacer(1, 6),
        callout(
            "The trap that shapes the whole schedule",
            "Turning on M2 changes what the LLM says, so every earlier run came from a different agent and "
            "pooling across prompt versions when fitting is biased. The discipline: prompt version is required "
            "on every record from day one; the prompt is frozen while M1 establishes a baseline; and M2 is "
            "introduced as an explicit new agent version, evaluated by M1's paired harness. The same logic "
            "defers M3b &mdash; the cost of analogue retrieval is not the search, it is putting results into the "
            "prompt.",
            tone="blue",
        ),
    ]


def next_steps_section() -> list:
    return [
        PageBreak(),
        *section("11", "Next steps", "In priority order, highest value first"),
        para(
            "The ordering below is by <i>value per unit of cost and delay</i>, not by dependency alone. Two "
            "items are urgent for reasons of timing rather than effort: they get much more expensive the "
            "longer the corpus grows."
        ),
        table(
            [
                ["#", "Action", "Why now", "Cost", "Needs a decision?"],
                [
                    "P1",
                    "Keep the daily job running, uninterrupted",
                    "Everything is blocked on corpus, and a missed weekday is evidence that cannot be "
                    "recovered &mdash; backfilling re-searches today's web. Confirm it fires each morning.",
                    "Zero &mdash; already scheduled",
                    "No",
                ],
                [
                    "P2",
                    "Fix the claim-building instability (finding 3)",
                    "It randomly zeroes the very signal M1 exists to fit. Fixing it is a <b>prompt change</b>, "
                    "which creates a new prompt version and fragments the corpus &mdash; so it is cheapest to do "
                    "<b>now</b>, at one live record, and roughly forty times more expensive in two months.",
                    "Small; measurable within a day",
                    "<b>Yes &mdash; see below</b>",
                ],
                [
                    "P3",
                    "Run the deterministic base corpus",
                    "The ensemble alone over the 51+18 existing origins, no LLM, minutes of compute. Yields "
                    "~200 scored base-layer forecasts <i>immediately</i>, which is the fastest available "
                    "evidence on how wrong the interval widths are.",
                    "Minutes",
                    "No",
                ],
                [
                    "P4",
                    "Widen base intervals as a declared prior (v002)",
                    "Coverage is 0 of 12 and 4,679 days of history say the tails are badly under-served. "
                    "Declared with no fitted period, so all future data judges it out-of-sample and honestly.",
                    "Low",
                    "Yes &mdash; it changes live output",
                ],
                [
                    "P5",
                    "Build workflow B (proposal + cycle runner)",
                    "Needed before the first calibration cycle can run, but not before the corpus exists. "
                    "No reason to rush it, and no reason to leave it to the last week.",
                    "Moderate",
                    "No",
                ],
                [
                    "P6",
                    "Run the contamination test",
                    "Replay the 2026-08-14 cutoff in a month or two and diff the assessment against the live "
                    "original. Cheap, and it settles a design question the March comparison could not.",
                    "One run",
                    "No",
                ],
                [
                    "P7",
                    "First real calibration cycle",
                    "At roughly 40 live_forward records &mdash; about two months of daily running. The first "
                    "tunable should be interval width; the centre gain second.",
                    "&mdash;",
                    "Yes &mdash; approve or reject",
                ],
                [
                    "P8",
                    "M2 self-audit memory",
                    "Only once M1 has a baseline that can A/B it. Introducing it earlier means there is no "
                    "instrument to tell whether it helped.",
                    "Moderate",
                    "Yes &mdash; new prompt version",
                ],
                [
                    "P9",
                    "M3b analogue injection",
                    "Stage 2. Same prompt-version cost as M2, with less expected value until novelty "
                    "calibration is understood.",
                    "Moderate",
                    "Yes",
                ],
            ],
            # Wide enough that "P1" does not wrap onto two lines.
            [0.42 * inch, 1.4 * inch, CONTENT_W - 4.22 * inch, 0.95 * inch, 1.45 * inch],
            bold_first_col=True,
        ),
        keep(
            para("The one decision that needs the v5.0 owner", "h2"),
            callout(
                "P2 requires modifying CFM Agent v5.0, and the coach cannot make that call",
                "The claim-building instability lives in "
                "<font face='Courier'>cfm_agent_v_5_0/skills/claim-building/SKILL.md</font>, which is "
                "<b>inside</b> the agent package and covered by its manifest. The coach's hard rule is that "
                "it never modifies v5.0 &mdash; so this is a change only the agent's owner can authorise, and "
                "it would change the package fingerprint, making records before and after it formally a "
                "different system for fitting purposes.<br/><br/>"
                "<b>The recommendation:</b> do it deliberately and early, as an explicit version bump of the "
                "agent, while the corpus is one record. The alternative &mdash; leaving it &mdash; means M1 "
                "spends two months fitting a signal that is being randomly set to zero by something "
                "unrelated to market judgment.",
                tone="gold",
            ),
        ),
        para("What good looks like in three months", "h2"),
        *bullets(
            [
                "Roughly 60 live_forward records, of which ~40 have at least one resolved horizon.",
                "80% interval coverage measured rather than assumed &mdash; and if it is still near zero, a width "
                "calibration that has passed the gate and been approved.",
                "A definitive answer to whether the LLM overlay beats the raw ensemble on <i>live</i> runs. "
                "This has never been measured and is the single most valuable number the project can produce.",
                "The trust tier either validated against realized error, or formally retired under R4.",
                "The claim-building variance either fixed and confirmed by re-running the same cutoff twice, "
                "or quantified so calibration can account for it.",
            ]
        ),
    ]


def guidance_section() -> list:
    return [
        PageBreak(),
        *section("12", "Practical guidance", "Strengths, limitations, and shared terminology"),
        para("Key strengths", "h2"),
        *bullets(
            [
                "The agent is genuinely untouched: additive package, injected settings, verified 43-file manifest.",
                "Replay is exact rather than approximate, and that claim is re-checked on every cycle.",
                "Contaminated evidence is excluded by construction, not by remembering to.",
                "One component, and only one, may declare an improvement &mdash; behind nine conditions.",
                "The statistics respect the dependence structure: origin-blocked bootstrap, paired days, "
                "reported effective sample size.",
                "Nothing reaches a live forecast without a human approving it.",
            ]
        ),
        para("Important limitations", "h2"),
        *bullets(
            [
                "<b>The corpus is one record.</b> Everything the harness can conclude today, it concludes "
                "about replayed history, which is not evidence.",
                "The trust report does not yet discriminate and is held back for that reason.",
                "Analogue dispersion is a plausibility check, not a measured error distribution. It can say an "
                "interval looks wrong; only accumulated coverage says by how much.",
                "The quantile-integrated CRPS spans the 5th to 95th percentile, so it is a consistent lower "
                "bound rather than an absolute CRPS. Fine for paired comparison; not for quoting externally.",
                "Calibrating the machinery cannot fix a mis-specified ensemble. Improving the underlying "
                "numerical models is real work and a separate project.",
                "Run-to-run variance in the agent's citation assembly is currently unmodelled and will inflate "
                "the variance of any fit until it is addressed.",
            ]
        ),
        para("Glossary", "h2"),
        table(
            [
                ["Term", "Meaning"],
                [
                    "Origin",
                    "One forecast cutoff. The unit the bootstrap resamples, and the unit sample size is counted in.",
                ],
                [
                    "Provenance",
                    "Whether a run was issued live or re-executed at a past cutoff. Only live_forward is fitting evidence.",
                ],
                [
                    "Fidelity",
                    "Whether replaying a record under its own calibration reproduces its recorded output exactly.",
                ],
                ["Pinball loss", "Quantile loss. Minimised only by the true quantile, and decomposable by level."],
                ["Coverage", "The fraction of outcomes landing inside P10&ndash;P90. Target 80%."],
                [
                    "Calibration layer",
                    "The coach-owned post-engine transform: per-horizon centre gain and width scale.",
                ],
                ["Calibration version", "A dated, immutable JSON document of constants. v001 is a true no-op."],
                ["Holdout", "Origins after the candidate's declared fitting period. The only ones the gate judges on."],
                ["Shrinkage", "Moving a fitted value partway back toward the incumbent before proposing it."],
                [
                    "Effective n",
                    "Independent observations, after accounting for overlapping horizons. Much smaller than the row count.",
                ],
            ],
            [1.35 * inch, CONTENT_W - 1.35 * inch],
            bold_first_col=True,
        ),
        Spacer(1, 10),
        callout(
            "Bottom line",
            "CFM Agent v5.0 makes disciplined forecasts and has never been told whether they were any good. "
            "CFM Coach is the instrument that will tell it &mdash; recording every run so it can be replayed "
            "exactly, scoring against what actually happened, and holding any proposed change behind a "
            "deliberately conservative gate. It has already found four things that were being assumed "
            "incorrectly. What it cannot do is hurry: the corpus accrues one weekday at a time, and the "
            "discipline that makes the eventual answer trustworthy is the same discipline that makes it slow.",
        ),
    ]


def main() -> None:
    story: list = []
    story.extend(cover())
    story.append(NextPageTemplate("body"))
    story.append(PageBreak())
    for builder in (
        executive_overview,
        objective,
        architecture,
        recorder,
        replay_section,
        scoring_section,
        gate_section,
        reading_section,
        findings_section,
        roadmap_section,
        next_steps_section,
        guidance_section,
    ):
        story.extend(builder())

    build_document().build(story)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
