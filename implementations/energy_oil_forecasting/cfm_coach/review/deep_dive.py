"""Stage E: a short tool loop on the few cases where the agent's judgment is implicated.

Selection is by value of information, computed by code: the largest gap to the
random walk among cards where the overlay share of that gap is at least
``overlay_implicated_share``, or a card that carries a new-pattern note.
Base-dominated cards (``base_share >= base_dominated_share``) make **zero** LLM
calls -- their loss is the numerical base's and the numeric track owns them.

Tools are read-only functions over stored records and code-computed pricing.
The two hindsight tools are the only network calls: a grounded search with no
cutoff (stored separately, tagged post-cutoff, never rendered into a proposal)
and a grounded search with the cutoff instruction plus the agent's own
independent leakage verifier (`_verify_no_leakage`, `agent_factory.py`) -- the
minimum viable hindsight the review settled on. No page fetches.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from energy_oil_forecasting.cfm_coach.review.cards import CaseCard
from energy_oil_forecasting.cfm_coach.review.collect import StreamCorpus
from energy_oil_forecasting.cfm_coach.review.llm import (
    LlmClient,
    LlmFailureError,
    complete_with_forced_submit,
    final_turn_notice,
    inline_refs,
    submit_contract,
    turn_notice,
)
from energy_oil_forecasting.cfm_coach.review.memory import Annotation, AnnotationStore, Taxonomy, annotation_id
from energy_oil_forecasting.cfm_coach.review.settings import DEFAULT_REVIEW_SETTINGS, ReviewSettings
from energy_oil_forecasting.cfm_coach.review.state import ReviewKey, ReviewState
from energy_oil_forecasting.cfm_coach.review.stats import CounterfactualEngine
from energy_oil_forecasting.cfm_coach.review.text import sanitize
from energy_oil_forecasting.cfm_coach.review.triage import PROMPTS_DIR, TagOut, prompt_hash, validate
from energy_oil_forecasting.cfm_coach.schemas import CalibrationLayer, RunRecord
from pydantic import BaseModel, ConfigDict, Field, ValidationError


Verdict = Literal["in_packet_ignored", "knowable_missed", "precursors_knowable", "unforeseeable", "undetermined"]
TOOL_OUTPUT_CAP = 3_000
SECTIONS = (
    "claims",
    "summaries",
    "sources",
    "actions",
    "policy",
    "transformation",
    "rationale",
    "diagnostics",
    "queries",
)


# ----------------------------------------------------------------- schema ----


class DeepDiveOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Verdict
    driver: str = ""
    judgment_mattered_usd: float | None = None
    tags: list[TagOut] = Field(default_factory=list)
    new_pattern_notes: list[str] = Field(default_factory=list)
    summary: str = ""


class HindsightRecord(BaseModel):
    """Post-cutoff material. Lives in ``hindsight/``; the proposal renderer never imports this module's store."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stream: str
    cutoff: date
    run_id: str
    driver_search: dict[str, Any] = Field(default_factory=dict)
    knowable_search: dict[str, Any] = Field(default_factory=dict)
    verdict: Verdict
    post_cutoff: bool = True


# ------------------------------------------------------------------ tools ----


def tool_declarations() -> list[dict[str, Any]]:
    def fn(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {"type": "object", "properties": properties, "required": required},
            },
        }

    return [
        fn(
            "get_record_section",
            "Read one section of the representative run's stored record (capped at 3000 characters).",
            {"section": {"type": "string", "enum": list(SECTIONS)}},
            ["section"],
        ),
        fn(
            "price_path",
            "Daily closes from a few days before the cutoff through the last resolved horizon, with the biggest-move days.",
            {"days_before": {"type": "integer"}},
            [],
        ),
        fn(
            "replay_counterfactual",
            "Price an alternative on this run's resolved horizons. op: no_change | modal_action | overlay_zero | anchor. "
            "Returns pinball before/after vs the random walk per horizon.",
            {
                "op": {"type": "string", "enum": ["no_change", "modal_action", "overlay_zero", "anchor"]},
                "a": {"type": "number"},
            },
            ["op"],
        ),
        fn(
            "hindsight_driver",
            "Grounded web search with NO cutoff: what actually drove the move. Post-cutoff material; stored separately.",
            {"question": {"type": "string"}},
            ["question"],
        ),
        fn(
            "hindsight_knowable",
            "Grounded web search restricted to before the cutoff, verified for leakage by an independent call.",
            {"question": {"type": "string"}},
            ["question"],
        ),
        fn(
            "submit_findings",
            "Finish. `findings` is a JSON string matching the schema in the system prompt: keys verdict, driver, judgment_mattered_usd, tags (each: code, severity 1-3, confidence, pointer, outcome_dependent, horizon, value, note), new_pattern_notes, summary. No other keys.",
            {"findings": {"type": "string"}},
            ["findings"],
        ),
    ]


def _cap(text: str) -> str:
    return text if len(text) <= TOOL_OUTPUT_CAP else text[: TOOL_OUTPUT_CAP - 1] + "…"


@dataclass
class CaseTools:
    """Read-only tools bound to one case. Hindsight calls go through the client's lookup stage."""

    corpus: StreamCorpus
    card: CaseCard
    record: RunRecord
    client: LlmClient
    settings: ReviewSettings = DEFAULT_REVIEW_SETTINGS
    hindsight: dict[str, Any] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def get_record_section(self, section: str) -> str:
        r = self.record
        a, p = r.assessment, r.research_packet
        if section == "claims":
            body = "\n".join(
                f"{c.get('claim_id')} [{c.get('claim_type')}; material {c.get('material_to_forecast')}; sources {c.get('supporting_source_ids')}; summaries {[s[-12:] for s in c.get('supporting_summary_ids', [])]}] <untrusted claim>{sanitize(c.get('statement'), cap=400)}</untrusted claim>"
                for c in a.get("evidence_claims", [])
            )
        elif section == "summaries":
            body = "\n".join(
                f"{s.get('verified_summary_id', '')[-12:]} [{s.get('query_area')}; {s.get('status')}; verifier {s.get('verifier_confidence')}] <untrusted summary>{sanitize(s.get('cleaned_summary'), cap=900)}</untrusted summary>"
                for s in p.get("verified_summaries", [])
            )
        elif section == "sources":
            body = "\n".join(
                f"{s.get('source_id')} {s.get('publisher')} [{s.get('source_quality')}; resolved {s.get('resolution_status')}] <untrusted title>{sanitize(s.get('provider_title'), cap=120)}</untrusted title>"
                for s in p.get("sources", [])
            )
        elif section == "queries":
            body = "\n".join(
                f"q{i} [{q.get('area')}] <untrusted query>{sanitize(q.get('query'), cap=200)}</untrusted query>"
                for i, q in enumerate(p.get("queries", []))
            )
        elif section == "actions":
            body = "\n".join(
                f"h{x.get('horizon')}: {x.get('center_action')}/{x.get('uncertainty_action')} cites {x.get('cited_claim_ids')} persistence {x.get('persistence_profile')} <untrusted rationale>{sanitize(x.get('rationale'), cap=500)}</untrusted rationale>"
                for x in a.get("horizon_actions", [])
            )
        elif section == "policy":
            body = "\n".join(
                f"h{f.horizon}: {json.dumps({k: v for k, v in (f.policy_decision or {}).items() if k != 'horizon'}, default=str)}"
                for f in r.forecasts
            )
        elif section == "transformation":
            body = "\n".join(
                f"h{f.horizon}: "
                + json.dumps(
                    {
                        k: v
                        for k, v in (f.forecast_transformation or {}).items()
                        if k
                        in (
                            "action_fraction",
                            "novelty_multiplier",
                            "uncertainty_multiplier",
                            "applied_center_adjustment",
                            "raw_center_adjustment",
                            "emergency_cap",
                            "p10_p90_width",
                        )
                    },
                    default=str,
                )
                for f in r.forecasts
            )
        elif section == "rationale":
            body = f"physical_status {a.get('physical_status')}; novelty {a.get('incremental_novelty')}; confidence {a.get('confidence')}\n<untrusted rationale>{sanitize(a.get('overall_rationale'), cap=900)}</untrusted rationale>\n<untrusted summary>{sanitize(a.get('research_summary'), cap=900)}</untrusted summary>"
        elif section == "diagnostics":
            body = json.dumps(r.diagnostics, default=str)
        else:
            body = f"unknown section {section!r}; choose one of {SECTIONS}"
        return _cap(body)

    def price_path(self, days_before: int = 5) -> str:
        prices = self.corpus.baseline.prices if self.corpus.baseline is not None else pd.Series(dtype=float)
        start = self.card.cutoff - timedelta(days=int(days_before) + 2)
        end = max((f.forecast_date.date() for f in self.record.forecasts), default=self.card.cutoff)
        window = prices[(prices.index >= start) & (prices.index <= end)]
        if window.empty:
            return "no prices in window"
        moves = window.diff().dropna()
        biggest = moves.abs().sort_values(ascending=False).head(4)
        lines = [f"{d} {v:.2f}" for d, v in window.items()]
        lines.append("biggest daily moves: " + ", ".join(f"{d} {moves[d]:+.2f}" for d in biggest.index))
        return _cap("\n".join(lines))

    def replay_counterfactual(self, op: str, a: float | None = None) -> str:
        engine = CounterfactualEngine(self.corpus, self.settings)
        horizons = tuple(self.card.resolved_horizons)
        if op == "overlay_zero":
            pricing = engine.overlay_zero(horizons)
        elif op == "no_change":
            pricing = engine.action_override(
                {"horizon_actions_patch": {"center_action": "no_change", "uncertainty_action": "unchanged"}}, horizons
            )
        elif op == "modal_action":
            rows = {row.horizon: row for row in self.card.dispersion}
            patch = {}
            for h, row in rows.items():
                if row.granted_center:
                    patch = {
                        "center_action": max(row.granted_center, key=row.granted_center.get),
                        "uncertainty_action": max(row.granted_uncertainty, key=row.granted_uncertainty.get),
                    }
            pricing = engine.action_override({"horizon_actions_patch": patch}, horizons)
        elif op == "anchor":
            value = 0.5 if a is None else float(a)
            pricing = engine.calibration(
                layer=CalibrationLayer(rw_anchor=dict.fromkeys(horizons, value)), horizons=horizons
            )
        else:
            return f"unknown op {op!r}"
        mine = {k: v for k, v in pricing.deltas.items() if k.startswith(self.record.run_id + "|")}
        lines = [
            f"op {op} (fidelity {pricing.fidelity}); this run's pinball deltas (before - after, + is better): {json.dumps(mine)}"
        ]
        for row in pricing.by_horizon:
            lines.append(
                f"h{row.horizon} all runs at scope: pinball {row.pinball_before:.3f} -> {row.pinball_after:.3f} (rw {row.pinball_rw:.3f}); coverage {row.coverage_before:.0%} -> {row.coverage_after:.0%}"
            )
        return _cap("\n".join(lines))

    async def hindsight_driver(self, question: str) -> str:
        text, rows = await self._grounded(question, cutoff=None)
        self.hindsight["driver_search"] = {"question": question, "text": text, "sources": rows, "post_cutoff": True}
        return _cap(
            f"<untrusted search post_cutoff=true>{sanitize(text, cap=2500)}</untrusted search>\nsources: {len(rows)}"
        )

    async def hindsight_knowable(self, question: str) -> str:
        cutoff = self.card.cutoff.isoformat()
        text, rows = await self._grounded(
            question + f"\n\nOnly include and cite information published strictly before {cutoff}.", cutoff=cutoff
        )
        verdict = await self._verify(text, question, cutoff)
        self.hindsight["knowable_search"] = {
            "question": question,
            "text": text,
            "sources": rows,
            "verifier": verdict,
            "post_cutoff": False,
        }
        if not verdict.get("clean") or int(verdict.get("confidence", 0)) < 8:
            return _cap(
                f"leakage verifier rejected the result (clean={verdict.get('clean')}, confidence={verdict.get('confidence')}); treat as undetermined. flagged: {verdict.get('flagged_claims')}"
            )
        return _cap(
            f"<untrusted search pre_cutoff_verified=true>{sanitize(verdict.get('filtered_text') or text, cap=2500)}</untrusted search>\nsources: {len(rows)}"
        )

    async def _grounded(self, content: str, *, cutoff: str | None) -> tuple[str, list[dict[str, Any]]]:
        messages = [
            {
                "role": "system",
                "content": "You are a specialized web search assistant. Return concise grounded factual content and source URLs. Do not infer bibliographic metadata.",
            },
            {"role": "user", "content": content},
        ]
        response = await self.client.complete(stage="lookup", messages=messages, tools=[{"googleSearch": {}}])
        self.calls.append({"kind": "grounded", "cutoff": cutoff, "usage": response.usage.as_dict()})
        return response.content or "", list(response.raw.get("grounding", []))

    async def _verify(self, text: str, query: str, cutoff: str) -> dict[str, Any]:
        from aieng.forecasting.methods.agentic.agent_factory import (  # noqa: PLC0415 - heavy import, lazy
            _LEAKAGE_VERIFIER_INSTRUCTION,
            _build_leakage_verification_schema,
        )

        messages = [
            {"role": "system", "content": _LEAKAGE_VERIFIER_INSTRUCTION},
            {
                "role": "user",
                "content": f"Original query: {query}\nCutoff date: {cutoff}\n\nSearch result to verify:\n{text}",
            },
        ]
        response = await self.client.complete(
            stage="lookup",
            messages=messages,
            response_schema=inline_refs(_build_leakage_verification_schema()),
            schema_name="LeakageVerification",
        )
        self.calls.append({"kind": "verifier", "usage": response.usage.as_dict()})
        try:
            return json.loads(response.content or "{}")
        except json.JSONDecodeError:
            return {
                "clean": False,
                "confidence": 1,
                "flagged_claims": ["verifier response could not be parsed"],
                "filtered_text": text,
            }

    async def dispatch(self, name: str, arguments: str | None) -> str:
        try:
            args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            return "arguments were not valid JSON"
        if name == "get_record_section":
            return self.get_record_section(str(args.get("section", "")))
        if name == "price_path":
            return self.price_path(int(args.get("days_before", 5)))
        if name == "replay_counterfactual":
            return self.replay_counterfactual(str(args.get("op", "")), args.get("a"))
        if name == "hindsight_driver":
            return await self.hindsight_driver(str(args.get("question", "")))
        if name == "hindsight_knowable":
            return await self.hindsight_knowable(str(args.get("question", "")))
        return f"unknown tool {name!r}"


# ------------------------------------------------------------- selection ----


def select_cases(
    cards: dict[date, CaseCard],
    annotations: dict[ReviewKey, Annotation],
    *,
    k: int,
    settings: ReviewSettings = DEFAULT_REVIEW_SETTINGS,
) -> tuple[list[CaseCard], list[CaseCard]]:
    """(chosen, base_dominated). Value of information = gap_vs_rw where the overlay is implicated, or a new-pattern note."""
    scored: list[tuple[float, CaseCard]] = []
    base_dominated: list[CaseCard] = []
    for card in cards.values():
        if not card.base:
            continue
        worst = max(card.base, key=lambda b: b.decomposition.gap_vs_rw)
        d = worst.decomposition
        notes = any(
            annotations[k].new_pattern_notes for k in annotations if k.stream == card.stream and k.cutoff == card.cutoff
        )
        if (d.base_share or 0) >= settings.base_dominated_share and not notes:
            base_dominated.append(card)
            continue
        if (d.overlay_share or 0) >= settings.overlay_implicated_share or notes:
            scored.append((d.gap_vs_rw + (1.0 if notes else 0.0), card))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [c for _, c in scored[:k]], base_dominated


# -------------------------------------------------------------- the loop ----


@dataclass
class DeepDiveResult:
    card: CaseCard
    run_id: str
    findings: DeepDiveOut | None
    annotations: list[Annotation]
    hindsight: HindsightRecord | None
    turns: int
    tool_calls: list[str]
    failure: str | None = None
    raw: list[dict[str, Any]] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)


async def deep_dive_case(
    client: LlmClient,
    corpus: StreamCorpus,
    card: CaseCard,
    *,
    taxonomy: Taxonomy,
    week: str,
    settings: ReviewSettings = DEFAULT_REVIEW_SETTINGS,
    prior: list[Annotation] | None = None,
) -> DeepDiveResult:
    record = next(r for r in corpus.records if r.run_id == card.representative.run_id)
    tools = CaseTools(corpus, card, record, client, settings)
    rubric = (PROMPTS_DIR / "deep_dive.md").read_text(encoding="utf-8")
    from energy_oil_forecasting.cfm_coach.review.cards import render_card  # noqa: PLC0415

    context = (
        [
            "Triage said: "
            + "; ".join(f"h{a.horizon}: {[t.code for t in a.tags]} knowability {a.knowability}" for a in (prior or []))
        ]
        if prior
        else []
    )
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": f"{rubric}\n\n{taxonomy.render()}\n\n{submit_contract('submit_findings', 'findings', DeepDiveOut)}",
        },
        {"role": "user", "content": "\n\n".join([*context, render_card(card, horizons=tuple(card.resolved_horizons))])},
    ]
    phash = prompt_hash(rubric, taxonomy.render(), "deep_dive_v2")
    used: list[str] = []
    findings: DeepDiveOut | None = None
    failure: str | None = None
    turns = 0
    raw: list[dict[str, Any]] = []
    max_turns = settings.deep_dive_max_turns
    budget = max_turns
    grace_used = False
    turn = 0
    while turn < budget:
        turns = turn + 1
        turn += 1
        final = turns >= max_turns
        if turns == max_turns:
            messages.append({"role": "user", "content": final_turn_notice("submit_findings")})
        try:
            response = await complete_with_forced_submit(
                client,
                stage="deep_dive",
                messages=messages,
                tools=tool_declarations(),
                submit_tool="submit_findings",
                force=final,
            )
        except LlmFailureError as exc:
            failure = str(exc)
            break
        raw.append(response.raw)
        if not response.tool_calls:
            failure = "model stopped without submit_findings"
            messages.append({"role": "assistant", "content": response.content or ""})
            break
        messages.append(
            {
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": [
                    {
                        "id": c["id"] or f"call_{i}",
                        "type": "function",
                        "function": {"name": c["name"], "arguments": c["arguments"] or "{}"},
                    }
                    for i, c in enumerate(response.tool_calls)
                ],
            }
        )
        done = False
        schema_error = False
        for i, call in enumerate(response.tool_calls):
            name = call["name"] or ""
            used.append(name)
            if name == "submit_findings":
                try:
                    payload = json.loads(call["arguments"] or "{}").get("findings", "")
                    findings = DeepDiveOut.model_validate_json(
                        payload if isinstance(payload, str) else json.dumps(payload)
                    )
                except (ValidationError, json.JSONDecodeError, AttributeError) as exc:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"] or f"call_{i}",
                            "content": f"findings did not match the schema: {str(exc)[:400]}. Call submit_findings again with a corrected JSON object.",
                        }
                    )
                    schema_error = True
                    continue
                done = True
                break
            output = await tools.dispatch(name, call["arguments"])
            messages.append({"role": "tool", "tool_call_id": call["id"] or f"call_{i}", "content": output})
        if done:
            break
        if final and schema_error and not grace_used:
            # One correction turn after a forced submission that failed the schema.
            grace_used = True
            budget += 1
            continue
        remaining = max_turns - turns
        if 0 < remaining <= 2:
            messages.append({"role": "user", "content": turn_notice(remaining, "submit_findings")})
    if findings is None and failure is None:
        failure = f"no findings after {turns} turn(s)"

    annotations: list[Annotation] = []
    hindsight: HindsightRecord | None = None
    if findings is not None:
        checked = validate(findings, card, taxonomy)
        for h in card.resolved_horizons:
            key = ReviewKey(card.stream, card.cutoff, h)
            tags = [*checked.tags_by_horizon.get(None, []), *checked.tags_by_horizon.get(h, [])]
            annotations.append(
                Annotation(
                    annotation_id=annotation_id(key, card.card_hash, phash, "deep_dive"),
                    stream=key.stream,
                    cutoff=key.cutoff,
                    horizon=h,
                    card_hash=card.card_hash,
                    card_version=card.card_version,
                    prompt_hash=phash,
                    stage="deep_dive",
                    week=week,
                    model=settings.model,
                    tags=tags,
                    new_pattern_notes=[n[:300] for n in findings.new_pattern_notes][:5],
                    knowability=findings.verdict,
                    dropped=checked.dropped,
                )
            )
        hindsight = HindsightRecord(
            stream=card.stream,
            cutoff=card.cutoff,
            run_id=record.run_id,
            verdict=findings.verdict,
            driver_search=tools.hindsight.get("driver_search", {}),
            knowable_search=tools.hindsight.get("knowable_search", {}),
        )
    return DeepDiveResult(
        card=card,
        run_id=record.run_id,
        findings=findings,
        annotations=annotations,
        hindsight=hindsight,
        turns=turns,
        tool_calls=used,
        failure=failure,
        raw=raw,
        transcript=messages,
    )


class HindsightStore:
    def __init__(self, settings: ReviewSettings = DEFAULT_REVIEW_SETTINGS):
        self.directory = settings.data_dir / "hindsight"

    def save(self, record: HindsightRecord) -> Path:
        path = self.directory / record.stream / f"{record.cutoff.isoformat()}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return path


async def run_deep_dive(
    client: LlmClient,
    *,
    corpus: StreamCorpus,
    cards: dict[date, CaseCard],
    store: AnnotationStore,
    state: ReviewState,
    taxonomy: Taxonomy,
    week: str,
    settings: ReviewSettings = DEFAULT_REVIEW_SETTINGS,
    k: int | None = None,
) -> tuple[list[DeepDiveResult], list[CaseCard]]:
    triaged = store.current(corpus.stream_id, stage="triage")
    dived = store.current(corpus.stream_id, stage="deep_dive")
    candidates = {
        c: card
        for c, card in cards.items()
        if any(k.cutoff == c for k in triaged) and not any(k.cutoff == c for k in dived)
    }
    chosen, base_dominated = select_cases(
        candidates, triaged, k=settings.deep_dive_k if k is None else k, settings=settings
    )
    results = []
    hindsight_store = HindsightStore(settings)
    for card in chosen:
        prior = [a for key, a in triaged.items() if key.cutoff == card.cutoff]
        result = await deep_dive_case(
            client, corpus, card, taxonomy=taxonomy, week=week, settings=settings, prior=prior
        )
        results.append(result)
        if result.failure:
            continue
        for ann in result.annotations:
            store.append(ann)
            if state.get(ann.key).status == "triaged":
                state.advance(ann.key, "deep_dived")
        if result.hindsight is not None:
            hindsight_store.save(result.hindsight)
    return results, base_dominated


__all__ = [
    "CaseTools",
    "DeepDiveOut",
    "DeepDiveResult",
    "HindsightRecord",
    "HindsightStore",
    "SECTIONS",
    "TOOL_OUTPUT_CAP",
    "deep_dive_case",
    "run_deep_dive",
    "select_cases",
    "tool_declarations",
]
