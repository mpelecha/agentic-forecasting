"""Stage F: from annotations to hypotheses and proposal drafts, with every count computed by code.

The model reasons; the tools count. `count_scope` and `query_annotations` take a
structured predicate and return keys, and only keys returned by a tool can end
up on a hypothesis. `price_counterfactual` and `base_table` expose the numbers a
proposal is ranked by. The loop ends with `submit_synthesis`; code then applies
the updates to the hypothesis ledger, computes evidence, and hands the drafts to
the gates. Nothing here writes a proposal file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from energy_oil_forecasting.cfm_coach.review.collect import StreamCorpus
from energy_oil_forecasting.cfm_coach.review.gates import EvidenceSummary, evidence_for
from energy_oil_forecasting.cfm_coach.review.llm import (
    LlmClient,
    LlmFailureError,
    complete_with_forced_submit,
    final_turn_notice,
    submit_contract,
    turn_notice,
)
from energy_oil_forecasting.cfm_coach.review.memory import (
    Annotation,
    Hypothesis,
    HypothesisEvent,
    HypothesisStore,
    ProposalStore,
    Taxonomy,
)
from energy_oil_forecasting.cfm_coach.review.settings import DEFAULT_REVIEW_SETTINGS, ReviewSettings
from energy_oil_forecasting.cfm_coach.review.state import ReviewKey
from energy_oil_forecasting.cfm_coach.review.stats import CounterfactualEngine, Episode, strata_table
from energy_oil_forecasting.cfm_coach.review.triage import PROMPTS_DIR, prompt_hash
from energy_oil_forecasting.cfm_coach.schemas import CalibrationLayer
from pydantic import BaseModel, ConfigDict, Field, ValidationError


# ----------------------------------------------------------------- schema ----

Action = Literal["create", "support", "contradict", "retire"]
LeverKind = Literal[
    "layer", "settings_overlay", "ensemble_weights", "skill_text", "persona", "prompt_instruction", "code", "data"
]


class HypothesisUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Action
    hypothesis_id: str | None = None
    statement: str = ""
    mechanism: str = ""
    track: Literal["numeric", "policy", "llm", "code", "data"] = "llm"
    lever: dict[str, Any] = Field(default_factory=dict)
    codes: list[str] = Field(default_factory=list)
    scope: dict[str, Any] = Field(default_factory=dict)
    prediction: str = ""
    supporting_keys: list[str] = Field(default_factory=list)
    contradicting_keys: list[str] = Field(default_factory=list)
    reason: str = ""


class TextEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: str
    find: str
    replace: str


class ProposalDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis_id: str
    title: str
    statement: str
    mechanism: str
    lever_kind: LeverKind
    file_line: str = ""
    field_name: str = ""
    value: dict[str, Any] = Field(default_factory=dict)
    text_edits: list[TextEdit] = Field(default_factory=list)
    streams: list[str] = Field(default_factory=list)
    horizons: list[int] = Field(default_factory=list)
    test_plan: str = ""
    pricing_op: str = ""
    pricing_params: dict[str, Any] = Field(default_factory=dict)


class SynthesisOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis_updates: list[HypothesisUpdate] = Field(default_factory=list)
    proposal_drafts: list[ProposalDraft] = Field(default_factory=list)
    taxonomy_notes: list[str] = Field(default_factory=list)
    playbook_notes: list[str] = Field(default_factory=list)


# ------------------------------------------------------------------ tools ----


def tool_declarations() -> list[dict[str, Any]]:
    predicate = {
        "type": "object",
        "properties": {
            "stream": {"type": "string"},
            "horizon": {"type": "integer"},
            "code": {"type": "string"},
            "track": {"type": "string"},
            "cutoff_from": {"type": "string"},
            "cutoff_to": {"type": "string"},
            "stage": {"type": "string"},
            "tier": {"type": "string"},
            "granted_center": {"type": "string"},
            "novelty": {"type": "string"},
            "zeroed": {"type": "boolean"},
        },
    }

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
            "count_scope",
            "Count resolved (run, horizon) rows in the strata frame matching a predicate; returns counts, distinct cutoffs, effective_n, pinball vs rw, coverage.",
            {"predicate": predicate},
            ["predicate"],
        ),
        fn(
            "query_annotations",
            "Keys and tags of annotations matching a predicate (code, stream, horizon, stage, cutoff range). Only keys returned here may be cited.",
            {"predicate": predicate},
            ["predicate"],
        ),
        fn("get_hypothesis", "One hypothesis in full.", {"hypothesis_id": {"type": "string"}}, ["hypothesis_id"]),
        fn(
            "base_table",
            "Per horizon: mean pinball for rw/ensemble/agent, base vs overlay share of the gap, coverage, direction hit rate.",
            {"stream": {"type": "string"}},
            ["stream"],
        ),
        fn(
            "price_counterfactual",
            'Price an operation on a stream\'s live rows. op: overlay_zero | rw_anchor | width_scale | no_change | settings | ensemble_reweight. params e.g. {"a":0.5} {"omega":1.5} {"field":"large_action_width_fraction","value":0.4} {"weights":{"arima":0.3,"kalman":0.3,"lightgbm":0.4},"with_rw":false}. Ensemble weights are not a settings field.',
            {"stream": {"type": "string"}, "op": {"type": "string"}, "params": {"type": "object"}},
            ["stream", "op"],
        ),
        fn(
            "submit_synthesis",
            "Finish. `synthesis` is a JSON string matching the schema in the system prompt: keys hypothesis_updates (each: action, hypothesis_id, statement, mechanism, track, lever, codes, scope, prediction, supporting_keys, contradicting_keys, reason), proposal_drafts, taxonomy_notes, playbook_notes. No other keys.",
            {"synthesis": {"type": "string"}},
            ["synthesis"],
        ),
    ]


def _match(pred: dict[str, Any], row: dict[str, Any]) -> bool:
    for k, v in pred.items():
        if v is None or k in ("cutoff_from", "cutoff_to", "code", "stage"):
            continue
        if str(row.get(k)) != str(v):
            return False
    cutoff = row.get("cutoff")
    if pred.get("cutoff_from") and cutoff and str(cutoff) < pred["cutoff_from"]:
        return False
    if pred.get("cutoff_to") and cutoff and str(cutoff) > pred["cutoff_to"]:
        return False
    return True


@dataclass
class SynthesisTools:
    corpora: dict[str, StreamCorpus]
    frames: dict[str, pd.DataFrame]
    annotations: dict[str, dict[ReviewKey, Annotation]]  # stream -> current annotations (all stages)
    hypotheses: HypothesisStore
    settings: ReviewSettings = DEFAULT_REVIEW_SETTINGS
    cited_keys: set[str] = field(default_factory=set)
    calls: list[str] = field(default_factory=list)

    def count_scope(self, predicate: dict[str, Any]) -> str:
        frames = [f for s, f in self.frames.items() if not predicate.get("stream") or s == predicate["stream"]]
        if not frames:
            return "no rows"
        frame = pd.concat(frames)
        mask = frame.apply(lambda r: _match(predicate, r.to_dict()), axis=1)
        sel = frame[mask]
        if sel.empty:
            return "0 rows match"
        table = strata_table(sel.assign(all="all"), ["all"]).iloc[0].to_dict()
        return json.dumps(
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in table.items() if k != "all"}, default=str
        )

    def query_annotations(self, predicate: dict[str, Any]) -> str:
        out = []
        for stream, current in self.annotations.items():
            if predicate.get("stream") and stream != predicate["stream"]:
                continue
            for key, ann in sorted(current.items()):
                if predicate.get("horizon") and key.horizon != int(predicate["horizon"]):
                    continue
                if predicate.get("stage") and ann.stage != predicate["stage"]:
                    continue
                if predicate.get("cutoff_from") and key.cutoff.isoformat() < predicate["cutoff_from"]:
                    continue
                if predicate.get("cutoff_to") and key.cutoff.isoformat() > predicate["cutoff_to"]:
                    continue
                tags = [t for t in ann.tags if not predicate.get("code") or t.code == predicate["code"]]
                if predicate.get("code") and not tags:
                    continue
                self.cited_keys.add(str(key))
                out.append(
                    {
                        "key": str(key),
                        "stage": ann.stage,
                        "tags": [f"{t.code}@{t.pointer}" for t in tags],
                        "knowability": ann.knowability,
                        "notes": ann.new_pattern_notes[:2],
                    }
                )
        if not out:
            return "no annotations match"
        text = json.dumps(out[:60])
        return text if len(text) <= 6_000 else text[:5_999] + "…"

    def get_hypothesis(self, hypothesis_id: str) -> str:
        h = self.hypotheses.state().get(hypothesis_id)
        return h.model_dump_json() if h else f"no hypothesis {hypothesis_id!r}"

    def base_table(self, stream: str) -> str:
        frame = self.frames.get(stream)
        if frame is None or frame.empty:
            return "no rows"
        live = frame[frame["provenance"] == "live_forward"]
        table = strata_table(live, ["horizon"])
        return table.round(4).to_json(orient="records")

    def price_counterfactual(self, stream: str, op: str, params: dict[str, Any] | None = None) -> str:
        corpus = self.corpora.get(stream)
        if corpus is None:
            return f"unknown stream {stream!r}"
        params = params or {}
        engine = CounterfactualEngine(corpus, self.settings)
        try:
            if op == "overlay_zero":
                pricing = engine.overlay_zero()
            elif op == "rw_anchor":
                a = float(params.get("a", 0.5))
                pricing = engine.calibration(layer=CalibrationLayer(rw_anchor=dict.fromkeys((5, 10, 21), a)))
            elif op == "width_scale":
                w = float(params.get("omega", 1.5))
                pricing = engine.calibration(layer=CalibrationLayer(width_scale=dict.fromkeys((5, 10, 21), w)))
            elif op == "no_change":
                pricing = engine.action_override(
                    {"horizon_actions_patch": {"center_action": "no_change", "uncertainty_action": "unchanged"}}
                )
            elif op == "settings":
                pricing = engine.calibration(settings_overlay={str(params["field"]): params["value"]})
            elif op == "ensemble_reweight":
                weights = {str(k): float(v) for k, v in dict(params.get("weights") or {}).items()}
                pricing = engine.ensemble_reweight(weights, with_rw=bool(params.get("with_rw", False)))
            else:
                return f"unknown op {op!r}"
        except Exception as exc:  # noqa: BLE001 - reported to the model, never raised into the loop
            return f"pricing failed: {type(exc).__name__}: {str(exc)[:200]}"
        rows = [
            {
                "horizon": r.horizon,
                "n": r.n,
                "cutoffs": r.distinct_cutoffs,
                "pinball_before": round(r.pinball_before, 4),
                "pinball_after": round(r.pinball_after, 4),
                "pinball_rw": round(r.pinball_rw, 4),
                "gain_vs_rw_pct": round(r.gain_vs_rw_pct, 2),
                "coverage_after": round(r.coverage_after, 3),
            }
            for r in pricing.by_horizon
        ]
        return json.dumps({"op": op, "params": params, "fidelity": pricing.fidelity, "by_horizon": rows})

    def dispatch(self, name: str, arguments: str | None) -> str:
        self.calls.append(name)
        try:
            args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            return "arguments were not valid JSON"
        if name == "count_scope":
            return self.count_scope(dict(args.get("predicate") or {}))
        if name == "query_annotations":
            return self.query_annotations(dict(args.get("predicate") or {}))
        if name == "get_hypothesis":
            return self.get_hypothesis(str(args.get("hypothesis_id", "")))
        if name == "base_table":
            return self.base_table(str(args.get("stream", "")))
        if name == "price_counterfactual":
            return self.price_counterfactual(str(args.get("stream", "")), str(args.get("op", "")), args.get("params"))
        return f"unknown tool {name!r}"


# ------------------------------------------------------------------ loop ----


def build_messages(
    *,
    week: str,
    taxonomy: Taxonomy,
    hypotheses: HypothesisStore,
    proposals: ProposalStore,
    week_summary: str,
    playbook: str,
) -> tuple[list[dict[str, Any]], str]:
    rubric = (PROMPTS_DIR / "synthesis.md").read_text(encoding="utf-8")
    stable = f"{rubric}\n\n{taxonomy.render()}\n\n{playbook}\n\n{submit_contract('submit_synthesis', 'synthesis', SynthesisOut)}"
    volatile = "\n\n".join([f"Week {week}.", week_summary, hypotheses.render_active(), proposals.render_register()])
    return [{"role": "system", "content": stable}, {"role": "user", "content": volatile}], prompt_hash(
        rubric, "synthesis_v2"
    )


@dataclass
class SynthesisResult:
    output: SynthesisOut | None
    tool_calls: list[str]
    cited_keys: set[str]
    turns: int
    failure: str | None = None
    raw: list[dict[str, Any]] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)


async def run_synthesis_loop(
    client: LlmClient,
    tools: SynthesisTools,
    messages: list[dict[str, Any]],
    *,
    settings: ReviewSettings = DEFAULT_REVIEW_SETTINGS,
    max_turns: int = 12,
) -> SynthesisResult:
    output: SynthesisOut | None = None
    failure: str | None = None
    turns = 0
    raw = []
    budget = max_turns
    grace_used = False
    turn = 0
    while turn < budget:
        turns = turn + 1
        turn += 1
        final = turns >= max_turns
        if turns == max_turns:
            messages.append({"role": "user", "content": final_turn_notice("submit_synthesis")})
        try:
            response = await complete_with_forced_submit(
                client,
                stage="synthesis",
                messages=messages,
                tools=tool_declarations(),
                submit_tool="submit_synthesis",
                force=final,
            )
        except LlmFailureError as exc:
            failure = str(exc)
            break
        raw.append(response.raw)
        if not response.tool_calls:
            failure = "model stopped without submit_synthesis"
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
            if call["name"] == "submit_synthesis":
                tools.calls.append("submit_synthesis")
                try:
                    payload = json.loads(call["arguments"] or "{}").get("synthesis", "")
                    output = SynthesisOut.model_validate_json(
                        payload if isinstance(payload, str) else json.dumps(payload)
                    )
                except (ValidationError, json.JSONDecodeError, AttributeError) as exc:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"] or f"call_{i}",
                            "content": f"synthesis did not match the schema: {str(exc)[:400]}. Call submit_synthesis again, corrected.",
                        }
                    )
                    schema_error = True
                    continue
                done = True
                break
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"] or f"call_{i}",
                    "content": tools.dispatch(call["name"] or "", call["arguments"]),
                }
            )
        if done:
            break
        if final and schema_error and not grace_used:
            grace_used = True
            budget += 1
            continue
        remaining = max_turns - turns
        if 0 < remaining <= 2:
            messages.append({"role": "user", "content": turn_notice(remaining, "submit_synthesis")})
    if output is None and failure is None:
        failure = f"no synthesis after {turns} turn(s)"
    return SynthesisResult(
        output=output,
        tool_calls=list(tools.calls),
        cited_keys=set(tools.cited_keys),
        turns=turns,
        failure=failure,
        raw=raw,
        transcript=list(messages),
    )


# ------------------------------------------------------------ apply ----


def _valid_keys(
    keys: list[str], cited: set[str], annotations: dict[str, dict[ReviewKey, Annotation]], codes: list[str]
) -> list[str]:
    """Only keys a tool returned, and only if that key's annotation actually carries one of the codes."""
    out = []
    for text in keys:
        if text not in cited:
            continue
        try:
            key = ReviewKey.parse(text)
        except ValueError:
            continue
        ann = annotations.get(key.stream, {}).get(key)
        if ann is not None and (not codes or any(t.code in codes for t in ann.tags)):
            out.append(text)
    return sorted(set(out))


@dataclass
class Applied:
    created: list[str] = field(default_factory=list)
    #: placeholder ids the model used for hypotheses it created -> stored ids
    id_map: dict[str, str] = field(default_factory=dict)
    updated: list[str] = field(default_factory=list)
    dropped_keys: int = 0
    evidence: dict[str, EvidenceSummary] = field(default_factory=dict)
    drafts: list[ProposalDraft] = field(default_factory=list)


def apply_synthesis(
    out: SynthesisOut,
    *,
    cited: set[str],
    annotations: dict[str, dict[ReviewKey, Annotation]],
    hypotheses: HypothesisStore,
    taxonomy: Taxonomy,
    episodes_by_stream: dict[str, list[Episode]],
    forecast_dates: dict[ReviewKey, date],
    runs_per_cutoff: dict[tuple[str, date], int],
    week: str,
    settings: ReviewSettings = DEFAULT_REVIEW_SETTINGS,
    write_events: bool = True,
) -> Applied:
    """Write hypothesis events (keys validated), recompute evidence for every touched hypothesis, keep valid drafts.

    ``write_events=False`` re-derives evidence and drafts from a stored synthesis without
    touching the ledger (the `--regate` path).
    """
    applied = Applied()
    state = hypotheses.state()
    touched: set[str] = set()

    def _emit(event: HypothesisEvent) -> None:
        if write_events:
            hypotheses.append(event)

    for update in out.hypothesis_updates:
        codes = [c for c in update.codes if taxonomy.resolve(c)]
        support = _valid_keys(update.supporting_keys, cited, annotations, codes)
        contra = _valid_keys(update.contradicting_keys, cited, annotations, codes)
        applied.dropped_keys += (
            len(update.supporting_keys) + len(update.contradicting_keys) - len(support) - len(contra)
        )
        if update.action == "create":
            if not codes or not update.statement or not update.prediction:
                continue
            existing = next((h for h in state.values() if h.statement == update.statement[:600]), None)
            if existing is not None or not write_events:
                # Same statement already in the ledger (a re-run): reuse it instead of a duplicate.
                if existing is not None:
                    if update.hypothesis_id:
                        applied.id_map[update.hypothesis_id] = existing.hypothesis_id
                    touched.add(existing.hypothesis_id)
                continue
            hid = hypotheses.next_id()
            if update.hypothesis_id:
                applied.id_map[update.hypothesis_id] = hid
            hypotheses.create(
                Hypothesis(
                    hypothesis_id=hid,
                    statement=update.statement[:600],
                    mechanism=update.mechanism[:600],
                    track=update.track,
                    lever=update.lever,
                    codes=codes,
                    scope=update.scope,
                    prediction=update.prediction[:400],
                    supporting=support,
                    contradicting=contra,
                ),
                week=week,
            )
            applied.created.append(hid)
            touched.add(hid)
            state = hypotheses.state()
            continue
        hid = update.hypothesis_id or ""
        if hid not in state:
            continue
        if update.action == "support" and support:
            _emit(HypothesisEvent(event="supported", hypothesis_id=hid, week=week, payload={"keys": support}))
        elif update.action == "contradict" and contra:
            _emit(HypothesisEvent(event="contradicted", hypothesis_id=hid, week=week, payload={"keys": contra}))
        elif update.action == "retire":
            _emit(
                HypothesisEvent(
                    event="status",
                    hypothesis_id=hid,
                    week=week,
                    payload={"status": "retired", "reason": update.reason[:300]},
                )
            )
        applied.updated.append(hid)
        touched.add(hid)

    state = hypotheses.state()
    for hid in sorted(touched | {h for h in state if state[h].status in ("candidate", "promoted", "stale")}):
        h = state[hid]
        current_all = {k: a for per in annotations.values() for k, a in per.items()}
        summary = evidence_for(
            h,
            annotations=current_all,
            episodes_by_stream=episodes_by_stream,
            forecast_dates=forecast_dates,
            runs_per_cutoff=runs_per_cutoff,
            taxonomy=taxonomy,
            settings=settings,
        )
        applied.evidence[hid] = summary
        new_support = hid in touched and any(
            u.action in ("create", "support") and (u.hypothesis_id == hid or hid in applied.created)
            for u in out.hypothesis_updates
        )
        _emit(
            HypothesisEvent(
                event="evidence",
                hypothesis_id=hid,
                week=week,
                payload={
                    "new_support": new_support,
                    "unit": summary.unit,
                    "independent_windows": summary.independent_windows,
                    "distinct_cutoffs": summary.distinct_cutoffs,
                    "episodes": summary.episodes,
                    "support_ratio": summary.support_ratio,
                    "replication_rate": summary.replication_rate,
                    "promotable": summary.promotable,
                    "reasons": summary.reasons,
                },
            )
        )
        state = hypotheses.state()
        h = state[hid]
        if h.status == "candidate" and summary.promotable:
            _emit(HypothesisEvent(event="status", hypothesis_id=hid, week=week, payload={"status": "promoted"}))
        elif h.status in ("candidate", "promoted") and h.reviews_without_new_support >= settings.stale_after_reviews:
            _emit(HypothesisEvent(event="status", hypothesis_id=hid, week=week, payload={"status": "stale"}))
        elif h.status == "stale" and h.reviews_without_new_support >= settings.retire_after_stale:
            _emit(HypothesisEvent(event="status", hypothesis_id=hid, week=week, payload={"status": "retired"}))
    final = hypotheses.state()
    drafts = []
    for d in out.proposal_drafts:
        hid = applied.id_map.get(d.hypothesis_id, d.hypothesis_id)
        if hid in final:
            drafts.append(d if hid == d.hypothesis_id else d.model_copy(update={"hypothesis_id": hid}))
    applied.drafts = drafts
    return applied


def stored_synthesis_output(week_dir: Path) -> SynthesisOut | None:
    """The synthesis the model submitted this week, read back from `raw/synthesis.json`; None if absent or invalid."""
    path = week_dir / "raw" / "synthesis.json"
    if not path.exists():
        return None
    turns = json.loads(path.read_text(encoding="utf-8"))
    payload = None
    for turn in turns:
        for call in turn.get("tool_calls") or []:
            if call.get("name") == "submit_synthesis":
                try:
                    payload = json.loads(call.get("arguments") or "{}").get("synthesis")
                except json.JSONDecodeError:
                    continue
    if payload is None:
        return None
    try:
        return SynthesisOut.model_validate_json(payload if isinstance(payload, str) else json.dumps(payload))
    except (ValidationError, json.JSONDecodeError):
        return None


__all__ = [
    "Applied",
    "HypothesisUpdate",
    "ProposalDraft",
    "SynthesisOut",
    "SynthesisResult",
    "SynthesisTools",
    "TextEdit",
    "apply_synthesis",
    "build_messages",
    "run_synthesis_loop",
    "tool_declarations",
]
