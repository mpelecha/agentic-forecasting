"""Stage H: the weekly report, deterministic markdown from what the week established.

Sections, in order: the honest scope line; the loss decomposition; the
scoreboard with the coached variant; ranked proposals with track labels;
candidates ready to accept; briefs; challengers and pairing; the hypothesis
watchlist; held-back items with gate reasons; post-ship verdicts; coach
self-metrics; cost. Markdown is the source of truth; the HTML is a plain
wrapper for the team page, published by hand from a Claude Code session.
"""

from __future__ import annotations

import html
import json
from typing import Any

import pandas as pd
from energy_oil_forecasting.cfm_coach.report import AGENT, ENSEMBLE, RANDOM_WALK, effective_n, hit_rate_needed
from energy_oil_forecasting.cfm_coach.review.collect import COACHED, StreamCorpus
from energy_oil_forecasting.cfm_coach.review.memory import Hypothesis, Proposal
from energy_oil_forecasting.cfm_coach.review.stats import strata_table
from energy_oil_forecasting.cfm_coach.scoring import ForecastScorer


def _pct(v: float | None) -> str:
    return "--" if v is None else f"{v:.0%}"


def _f(v: float | None, d: int = 3) -> str:
    return "--" if v is None else f"{v:.{d}f}"


def scope_line(corpora: dict[str, StreamCorpus], episodes: dict[str, list[Any]]) -> str:
    parts = []
    for sid, corpus in corpora.items():
        agent_cards = [c for (_, _, v), c in corpus.scores.items() if v == AGENT]
        n_eff = effective_n(agent_cards)
        eps = episodes.get(sid, [])
        signs = sorted({e.direction for e in eps})
        parts.append(
            f"{sid}: {len(corpus.cutoffs)} cutoffs, {len(agent_cards)} resolved rows, effective_n {n_eff:.1f}, {len(eps)} episode(s) with signs {signs}"
        )
    return (
        "**Read this first.** "
        + "; ".join(parts)
        + ". Outcome-dependent hypotheses need 3 independent windows across 2 episodes of both signs; "
        "numeric gate verdicts are underpowered below effective_n 10. Everything below is labelled accordingly."
    )


def decomposition_table(frames: dict[str, pd.DataFrame]) -> list[str]:
    lines = [
        "| stream | h | rows | cutoffs | eff. n | agent pinball | rw | gap | base term | overlay term | base share |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for sid, frame in frames.items():
        if frame is None or frame.empty:
            continue
        live = frame[frame["provenance"] == "live_forward"]
        table = strata_table(live, ["horizon"])
        for row in table.itertuples():
            total = abs(row.base_term) + abs(row.overlay_term)
            share = abs(row.base_term) / total if total else None
            lines.append(
                f"| {sid} | {row.horizon} | {row.rows} | {row.distinct_cutoffs} | {row.effective_n:.1f} | {row.pinball_agent:.3f} | {row.pinball_rw:.3f} | {row.gap_vs_rw:+.3f} | {row.base_term:+.3f} | {row.overlay_term:+.3f} | {_pct(share)} |"
            )
    return lines


def scoreboard(corpora: dict[str, StreamCorpus]) -> list[str]:
    lines = [
        "| stream | h | variant | n | pinball | cover80 | width | hit rate | always-up | needed |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for sid, corpus in corpora.items():
        by_h: dict[int, dict[str, list]] = {}
        live = {r.run_id for r in corpus.records if r.provenance == "live_forward"}
        for (run_id, h, v), card in corpus.scores.items():
            if run_id in live:
                by_h.setdefault(h, {}).setdefault(v, []).append(card)
        for h in sorted(by_h):
            for v in (RANDOM_WALK, ENSEMBLE, AGENT, COACHED):
                cards = by_h[h].get(v)
                if not cards:
                    continue
                s = ForecastScorer.summarize(cards, variant=v)
                needed = hit_rate_needed(effective_n(cards), s.always_up_rate) if s.always_up_rate is not None else None
                lines.append(
                    f"| {sid} | {h} | {v} | {s.n} | {s.pinball:.3f} | {_pct(s.coverage_80)} | {s.mean_interval_width:.2f} | {_pct(s.direction_hit_rate)} | {_pct(s.always_up_rate)} | {_pct(needed) if needed is not None else 'unreachable'} |"
                )
    return lines


def proposals_section(proposals: list[Proposal], outcomes: list[Any]) -> list[str]:
    by_h = {o.hypothesis_id: o for o in outcomes}
    ranked = sorted(
        [p for p in proposals if p.decision.status == "open"],
        key=lambda p: (p.rank_score is None, -(p.rank_score or 0)),
    )
    if not ranked:
        return ["_No proposal passed the gates this week. See the watchlist and held-back items._"]
    lines = []
    for i, p in enumerate(ranked, start=1):
        o = by_h.get(p.hypothesis_id or "")
        label = {
            "numeric": "numeric (M1 gate)",
            "policy": "policy (replay + gate)",
            "llm": "LLM judgment (two-unit bar)",
            "code": "code brief",
            "data": "data",
        }.get(p.track, p.track)
        lines += [
            f"### {i}. {p.id} — {p.title}",
            "",
            f"track **{label}**; lever `{p.lever.get('kind')}` {p.lever.get('file_line') or p.lever.get('field') or ''}; fidelity {p.lever.get('fidelity')}; rank score {_f(p.rank_score, 2)} (gain vs rw %, pooled h5+h10)",
            "",
            p.statement,
            "",
            f"Mechanism: {p.mechanism}",
            "",
            f"Evidence: unit {p.evidence.get('unit')}, windows {p.evidence.get('independent_windows')}, cutoffs {p.evidence.get('distinct_cutoffs')}, episodes {p.evidence.get('episodes')}, support ratio {_f(p.evidence.get('support_ratio'), 2)}, base share {_pct(p.evidence.get('base_share_mean'))}",
            "",
        ]
        if p.effect_vs_rw.get("numeric"):
            for sid, d in p.effect_vs_rw["numeric"].items():
                pooled = d.get("pooled") or {}
                lines.append(
                    f"- {sid}: gate shrunk {'PASS' if d.get('gate_shrunk') else 'FAIL'} {d.get('failed_conditions')}; effective_n {_f(d.get('effective_n'), 1)}{' (underpowered)' if d.get('underpowered') else ''}; pinball {_f(pooled.get('pinball_before'))} → {_f(pooled.get('pinball_after'))} (rw {_f(pooled.get('pinball_rw'))}); beats rw after: {pooled.get('rw_dominance_after')}; candidate `{d.get('candidate_file')}`"
                )
        if p.effect_vs_rw.get("pricing"):
            pooled = (p.effect_vs_rw["pricing"] or {}).get("pooled") or {}
            lines.append(
                f"- priced by {p.effect_vs_rw['pricing'].get('op')}: pinball {_f(pooled.get('pinball_before'))} → {_f(pooled.get('pinball_after'))} (rw {_f(pooled.get('pinball_rw'))})"
            )
        lines += ["", f"Test plan: {p.test_plan}", ""]
        if o and o.critic:
            lines += [
                f"Critic: {o.critic.get('verdict')} — {o.critic.get('hindsight') or o.critic.get('mechanism') or ''}",
                "",
            ]
        lines += [
            f"Decide in `proposals/{p.id}.yaml` (`decision.status`: accepted | rejected | deferred, with a reason).",
            "",
        ]
    return lines


def watchlist(hypotheses: dict[str, Hypothesis], evidence: dict[str, Any]) -> list[str]:
    lines = [
        "| id | status | track | codes | support | unit | windows | cutoffs | episodes | what would promote it |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for hid, h in sorted(hypotheses.items()):
        if h.status in ("retired", "refuted", "merged"):
            continue
        ev = evidence.get(hid)
        why = "; ".join(ev.reasons) if ev and ev.reasons else ("promoted" if h.status == "promoted" else "-")
        seed = f" (seed {h.seed_source})" if h.seed_source else ""
        lines.append(
            f"| {hid} | {h.status} | {h.track} | {', '.join(h.codes)}{seed} | {len(h.supporting)}/{len(h.contradicting)} | {ev.unit if ev else '-'} | {ev.independent_windows if ev else '-'} | {ev.distinct_cutoffs if ev else '-'} | {ev.episodes if ev else '-'} | {why} |"
        )
    return lines


def held_back(outcomes: list[Any]) -> list[str]:
    items = [o for o in outcomes if o.decision != "pass"]
    if not items:
        return ["_Nothing held back._"]
    return [
        f"- **{o.title}** ({o.hypothesis_id}, {o.lever_kind}): {o.decision} — " + "; ".join(o.reasons) for o in items
    ]


def render_report(  # noqa: PLR0913 - the report is the sum of its inputs
    *,
    week: str,
    corpora: dict[str, StreamCorpus],
    frames: dict[str, pd.DataFrame],
    episodes: dict[str, list[Any]],
    proposals: list[Proposal],
    outcomes: list[Any],
    hypotheses: dict[str, Hypothesis],
    evidence: dict[str, Any],
    deep_dives: list[Any],
    base_dominated: list[Any],
    pairing: list[Any],
    post_ship: dict[str, Any],
    metrics: dict[str, Any],
    ledger_by_stage: dict[str, Any],
    spent_usd: float,
    budget_usd: float,
    manifest: dict[str, Any],
    failures: list[str],
) -> str:
    lines = [f"# Coach review {week}", "", scope_line(corpora, episodes), ""]
    lines += ["## Where the loss comes from (live_forward, pinball)", "", *decomposition_table(frames), ""]
    lines += ["## Scoreboard", "", *scoreboard(corpora), ""]
    lines += ["## Ranked proposals", "", *proposals_section(proposals, outcomes)]
    candidates = [p for p in proposals if p.decision.status == "open" and p.artifacts.get("candidate_files")]
    lines += ["## Candidates ready to accept", ""]
    lines += [
        f"- {p.id}: {p.artifacts['candidate_files']} → `python -m energy_oil_forecasting.cfm_coach.review.accept --proposal {p.id} --stream <stream> --by <you>`"
        for p in candidates
    ] or ["_none_"]
    lines += ["", "## Deep dives", ""]
    lines += [
        f"- {d.card.stream} {d.card.cutoff} ({d.run_id}): verdict {d.findings.verdict if d.findings else 'failed'}; turns {d.turns}; tools {d.tool_calls}"
        + (f"; failure {d.failure}" if d.failure else "")
        for d in deep_dives
    ] or ["_none_"]
    lines += [
        f"- base-dominated (no LLM call): {', '.join(f'{c.stream} {c.cutoff}' for c in base_dominated)}"
        if base_dominated
        else "- base-dominated: none",
        "",
    ]
    lines += ["## Challengers and pairing", ""]
    lines += [r.describe() for r in pairing] or [
        "_No challenger registered. Accepted text proposals draft one under review/challengers/._"
    ]
    lines += ["", "## Hypothesis watchlist", "", *watchlist(hypotheses, evidence), ""]
    lines += ["## Held back", "", *held_back(outcomes), ""]
    lines += ["## Post-ship verdicts", ""]
    lines += [f"- {k}: {json.dumps(v, default=str)[:400]}" for k, v in post_ship.items()] or ["_Nothing shipped yet._"]
    lines += ["", "## Coach self-metrics", ""]
    lines += [f"- {k}: {json.dumps(v, default=str)[:300]}" for k, v in metrics.items()] or ["_Reported from week 3._"]
    lines += [
        "",
        "## Cost",
        "",
        f"spent ${spent_usd:.3f} of ${budget_usd:.2f}; model {manifest.get('model')}; reasoning_effort supported: {manifest.get('reasoning_effort_supported', 'n/a')}",
        "",
        "```",
        json.dumps(ledger_by_stage, indent=1, default=str),
        "```",
        "",
    ]
    if failures:
        lines += ["## Failures", "", *[f"- {f}" for f in failures], ""]
    return "\n".join(lines)


def render_html(markdown_text: str, *, title: str) -> str:
    """A plain wrapper: headings and tables stay readable without a markdown library."""
    body = html.escape(markdown_text)
    return (
        f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
        "<style>body{font-family:-apple-system,Segoe UI,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;line-height:1.45}pre{white-space:pre-wrap}</style>"
        f"</head><body><pre>{body}</pre></body></html>"
    )


__all__ = [
    "decomposition_table",
    "held_back",
    "proposals_section",
    "render_html",
    "render_report",
    "scope_line",
    "scoreboard",
    "watchlist",
]
