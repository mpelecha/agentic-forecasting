"""The weekly review, end to end. Every stage is a function; the CLI strings them together.

    A collect -> B cards -> C stats -> D triage -> E deep dive -> F synthesis -> G gates -> H render

``--no-llm`` runs A-C, seeds the ledger, and writes a report with zero spend --
also what a broken proxy degrades to. ``--dry-run`` prints the pre-flight and
stops before any call. ``--bootstrap`` lifts the budget to the bootstrap ceiling.
Everything a run did is in ``weeks/<YYYY-Www>/manifest.json``; state accrues in
``state.json`` so a rerun continues rather than repeats.

Usage (from the repo root)::

    uv run python -m energy_oil_forecasting.cfm_coach.review.run_weekly --no-llm
    uv run python -m energy_oil_forecasting.cfm_coach.review.run_weekly --dry-run
    uv run python -m energy_oil_forecasting.cfm_coach.review.run_weekly --budget-usd 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from energy_oil_forecasting.cfm_coach.review import metrics as coach_metrics
from energy_oil_forecasting.cfm_coach.review import ship_watch
from energy_oil_forecasting.cfm_coach.review.cards import CardStore, CaseCard, card_tokens
from energy_oil_forecasting.cfm_coach.review.collect import StreamCorpus, collect_stream, describe, readiness
from energy_oil_forecasting.cfm_coach.review.deep_dive import run_deep_dive
from energy_oil_forecasting.cfm_coach.review.gates import evidence_for, run_gates, seed_statistics, to_proposal
from energy_oil_forecasting.cfm_coach.review.llm import (
    BudgetExceededError,
    CostLedger,
    FakeClient,
    LlmClient,
    ProxyClient,
    estimate_tokens,
    load_proxy_credentials,
)
from energy_oil_forecasting.cfm_coach.review.memory import (
    AnnotationStore,
    HypothesisStore,
    ProposalStore,
    Taxonomy,
    load_playbook,
    load_seeds,
    seed_hypotheses,
)
from energy_oil_forecasting.cfm_coach.review.render import render_html, render_report
from energy_oil_forecasting.cfm_coach.review.settings import DEFAULT_REVIEW_SETTINGS, ReviewSettings
from energy_oil_forecasting.cfm_coach.review.state import ReviewKey, ReviewState
from energy_oil_forecasting.cfm_coach.review.stats import find_episodes, strata_frame
from energy_oil_forecasting.cfm_coach.review.synthesis import (
    SynthesisTools,
    apply_synthesis,
    run_synthesis_loop,
    stored_synthesis_output,
)
from energy_oil_forecasting.cfm_coach.review.synthesis import build_messages as synthesis_messages
from energy_oil_forecasting.cfm_coach.review.triage import build_messages, run_triage
from energy_oil_forecasting.cfm_coach.streams import SCHEDULED_STREAMS, RunStream, stream_for


def week_id(day: date | None = None) -> str:
    day = day or date.today()
    year, week, _ = day.isocalendar()
    return f"{year}-W{week:02d}"


@dataclass
class WeekContext:
    week: str
    settings: ReviewSettings
    state: ReviewState
    taxonomy: Taxonomy
    corpora: dict[str, StreamCorpus] = field(default_factory=dict)
    cards: dict[str, dict[date, CaseCard]] = field(default_factory=dict)
    frames: dict[str, Any] = field(default_factory=dict)
    episodes: dict[str, list[Any]] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    deep_dives: list[Any] = field(default_factory=list)
    base_dominated: list[Any] = field(default_factory=list)
    outcomes: list[Any] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    new_proposals: list[Any] = field(default_factory=list)
    post_ship: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    #: re-run the gates from the stored synthesis instead of calling the model
    regate: bool = False

    @property
    def week_dir(self) -> Path:
        return self.settings.weeks_dir / self.week

    def save_manifest(self) -> Path:
        self.week_dir.mkdir(parents=True, exist_ok=True)
        path = self.week_dir / "manifest.json"
        path.write_text(json.dumps(self.manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        return path

    @property
    def cards_by_key(self) -> dict[tuple[str, date], CaseCard]:
        return {(c.stream, c.cutoff): c for cards in self.cards.values() for c in cards.values()}

    def annotations(self, store: AnnotationStore) -> dict[str, dict[ReviewKey, Any]]:
        return {sid: store.current(sid) for sid in self.corpora}

    def forecast_dates(self) -> dict[ReviewKey, date]:
        out = {}
        for sid, corpus in self.corpora.items():
            for record in corpus.records:
                for item in record.forecasts:
                    out[ReviewKey(sid, record.cutoff, item.horizon)] = item.forecast_date.date()
        return out

    def runs_per_cutoff(self) -> dict[tuple[str, date], int]:
        return {
            (sid, c): len(runs) for sid, corpus in self.corpora.items() for c, runs in corpus.runs_by_cutoff.items()
        }


# -- stages A-C ---------------------------------------------------------------


def stage_collect(
    ctx: WeekContext, streams: tuple[RunStream, ...], *, corpora: dict[str, StreamCorpus] | None = None
) -> None:
    for stream in streams:
        corpus = corpora[stream.stream_id] if corpora else collect_stream(stream, review_settings=ctx.settings)
        ctx.corpora[stream.stream_id] = corpus
        print(describe(corpus))
        ready = readiness(corpus, ctx.state, ctx.settings)
        ctx.manifest.setdefault("inputs", {})[stream.stream_id] = {
            "records": len(corpus.records),
            "cutoffs": len(corpus.cutoffs),
            "resolved": len(corpus.resolution.resolved),
            "data_through": str(corpus.data_through),
            "ready_v1": len(ready.cutoffs_at(1)),
            "ready_v2": len(ready.cutoffs_at(2)),
            "new_keys": [str(k) for k in ready.new_keys],
        }


def stage_cards(ctx: WeekContext) -> None:
    store = CardStore(ctx.settings)
    for stream_id, corpus in ctx.corpora.items():
        ready = readiness(corpus, ctx.state, ctx.settings)
        cards: dict[date, CaseCard] = {}
        rebuilt = 0
        for cutoff in ready.cutoffs_at(1):
            card, was_rebuilt = store.load_or_build(corpus, cutoff)
            rebuilt += int(was_rebuilt)
            cards[cutoff] = card
        ctx.cards[stream_id] = cards
        sizes = [card_tokens(c) for c in cards.values()]
        ctx.manifest.setdefault("cards", {})[stream_id] = {
            "count": len(cards),
            "rebuilt": rebuilt,
            "max_tokens": max(sizes, default=0),
        }
        print(f"{stream_id}: {len(cards)} card(s) ready ({rebuilt} rebuilt), max {max(sizes, default=0)} tokens")


def stage_stats(ctx: WeekContext) -> None:
    for stream_id, corpus in ctx.corpora.items():
        ctx.frames[stream_id] = strata_frame(corpus)
        if corpus.baseline is not None and corpus.cutoffs:
            ctx.episodes[stream_id] = find_episodes(corpus.baseline.prices, start=corpus.cutoffs[0])
        ctx.manifest.setdefault("seed_statistics", {})[stream_id] = seed_statistics(
            ctx.frames[stream_id], list(ctx.cards.get(stream_id, {}).values())
        )
        ctx.manifest.setdefault("episodes", {})[stream_id] = [
            e.model_dump(mode="json") for e in ctx.episodes.get(stream_id, [])
        ]


def stage_seed(ctx: WeekContext) -> None:
    store = HypothesisStore(ctx.settings)
    pooled: dict[str, Any] = {}
    for stats in ctx.manifest.get("seed_statistics", {}).values():
        for k, v in stats.items():
            pooled.setdefault(k, v)
    created = seed_hypotheses(store, load_seeds(), ctx.taxonomy, week=ctx.week, evidence=pooled)
    ctx.manifest["seeded"] = created
    if created:
        print(f"seeded {len(created)} hypotheses from IMPROVEMENTS.md")


def week_header(ctx: WeekContext, stream_id: str) -> str:
    frame = ctx.frames.get(stream_id)
    if frame is None or frame.empty:
        return ""
    lines = [f"Week {ctx.week} base table for {stream_id} (live_forward, means):"]
    live = frame[frame["provenance"] == "live_forward"]
    for h, g in live.groupby("horizon"):
        lgbm = (
            g["lgbm_below_last_close"].dropna().astype(float).mean()
            if g["lgbm_below_last_close"].notna().any()
            else float("nan")
        )
        lines.append(
            f"  h{h}: cutoffs={g['cutoff'].nunique()} agent pinball {g['pinball_agent'].mean():.3f} vs rw {g['pinball_rw'].mean():.3f}; coverage {g['covered_agent'].mean():.0%}; zeroed {g['zeroed'].mean():.0%}; lgbm below last close {lgbm:.0%}"
        )
    return "\n".join(lines)


def preflight(ctx: WeekContext, ledger: CostLedger) -> dict[str, Any]:
    total_in = total_out = calls = 0
    for stream_id, cards in ctx.cards.items():
        pending = {k.cutoff for k in ctx.state.keys_in("resolved") if k.stream == stream_id}
        for cutoff, card in cards.items():
            if cutoff in pending:
                messages, _ = build_messages(
                    card, ctx.taxonomy, horizons=(5, 10), week_header=week_header(ctx, stream_id)
                )
                total_in += estimate_tokens(json.dumps(messages))
                total_out += 1_500
                calls += 1
    s = ctx.settings
    deep = s.deep_dive_k * s.deep_dive_max_turns * 12_000
    est = {
        "triage_calls": calls,
        "triage_prompt_tokens": total_in,
        "triage_usd": ledger.price(total_in, total_out),
        "deep_dive_prompt_tokens_bound": deep,
        "deep_dive_usd_bound": ledger.price(deep, s.deep_dive_k * s.deep_dive_max_turns * 1_500),
        "synthesis_usd_bound": ledger.price(12 * 30_000, 12 * 4_000),
        "budget_usd": ledger.budget_usd,
    }
    est["total_usd_bound"] = est["triage_usd"] + est["deep_dive_usd_bound"] + est["synthesis_usd_bound"]
    ctx.manifest["preflight"] = est
    return est


# -- stages D-G ---------------------------------------------------------------


async def stage_triage(ctx: WeekContext, client: LlmClient, store: AnnotationStore) -> None:
    for stream_id, corpus in ctx.corpora.items():
        results = await run_triage(
            client,
            corpus=corpus,
            cards=ctx.cards.get(stream_id, {}),
            state=ctx.state,
            store=store,
            taxonomy=ctx.taxonomy,
            week=ctx.week,
            settings=ctx.settings,
            week_header=week_header(ctx, stream_id),
        )
        ok = [r for r in results if not r.failure]
        ctx.manifest.setdefault("triage", {})[stream_id] = {
            "cards": len(results),
            "annotations": sum(len(r.annotations) for r in ok),
            "dropped_tags": sum(len(r.dropped) for r in ok),
            "failures": [r.failure for r in results if r.failure],
        }
        raw_dir = ctx.week_dir / "raw" / stream_id
        raw_dir.mkdir(parents=True, exist_ok=True)
        for r in results:
            if r.raw is not None:
                (raw_dir / f"triage_{r.card.cutoff}.json").write_text(
                    json.dumps(r.raw, indent=2, default=str) + "\n", encoding="utf-8"
                )
        ctx.failures.extend(f"{stream_id}: {r.failure}" for r in results if r.failure)
        print(f"{stream_id}: triaged {len(ok)}/{len(results)} card(s)")


async def stage_deep_dive(ctx: WeekContext, client: LlmClient, store: AnnotationStore) -> None:
    for stream_id, corpus in ctx.corpora.items():
        results, base_dominated = await run_deep_dive(
            client,
            corpus=corpus,
            cards=ctx.cards.get(stream_id, {}),
            store=store,
            state=ctx.state,
            taxonomy=ctx.taxonomy,
            week=ctx.week,
            settings=ctx.settings,
        )
        ctx.deep_dives.extend(results)
        ctx.base_dominated.extend(base_dominated)
        raw_dir = ctx.week_dir / "raw" / stream_id
        raw_dir.mkdir(parents=True, exist_ok=True)
        for r in results:
            (raw_dir / f"deep_dive_{r.card.cutoff}.json").write_text(
                json.dumps({"raw": r.raw, "transcript": r.transcript}, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
        ctx.manifest.setdefault("deep_dive", {})[stream_id] = {
            "cases": len(results),
            "base_dominated": len(base_dominated),
            "failures": [r.failure for r in results if r.failure],
        }
        ctx.failures.extend(f"{stream_id} deep dive {r.card.cutoff}: {r.failure}" for r in results if r.failure)
        print(f"{stream_id}: deep-dived {len(results)} case(s); {len(base_dominated)} base-dominated skipped")


async def stage_synthesis_and_gates(ctx: WeekContext, client: LlmClient | None, store: AnnotationStore) -> None:
    hyps = HypothesisStore(ctx.settings)
    props = ProposalStore(ctx.settings)
    annotations = ctx.annotations(store)
    tools = SynthesisTools(
        corpora=ctx.corpora, frames=ctx.frames, annotations=annotations, hypotheses=hyps, settings=ctx.settings
    )
    summary = "\n".join(week_header(ctx, sid) for sid in ctx.corpora)
    drafts: list[Any] = []
    if ctx.regate:
        output = stored_synthesis_output(ctx.week_dir)
        if output is None:
            ctx.failures.append("regate: no valid stored synthesis in raw/synthesis.json")
        else:
            all_keys = {str(k) for per in annotations.values() for k in per}
            applied = apply_synthesis(
                output,
                cited=all_keys,
                annotations=annotations,
                hypotheses=hyps,
                taxonomy=ctx.taxonomy,
                episodes_by_stream=ctx.episodes,
                forecast_dates=ctx.forecast_dates(),
                runs_per_cutoff=ctx.runs_per_cutoff(),
                week=ctx.week,
                settings=ctx.settings,
                write_events=False,
            )
            ctx.evidence = applied.evidence
            drafts = applied.drafts
            ctx.manifest["synthesis"] = {"regate": True, "drafts": len(drafts), "id_map": applied.id_map}
            print(f"regate: {len(drafts)} draft(s) from the stored synthesis; no events written")
    elif client is not None:
        messages, _ = synthesis_messages(
            week=ctx.week,
            taxonomy=ctx.taxonomy,
            hypotheses=hyps,
            proposals=props,
            week_summary=summary,
            playbook=load_playbook(),
        )
        result = await run_synthesis_loop(client, tools, messages, settings=ctx.settings)
        ctx.manifest["synthesis"] = {"turns": result.turns, "tool_calls": result.tool_calls, "failure": result.failure}
        (ctx.week_dir / "raw").mkdir(parents=True, exist_ok=True)
        (ctx.week_dir / "raw" / "synthesis.json").write_text(
            json.dumps(result.raw, indent=2, default=str) + "\n", encoding="utf-8"
        )
        (ctx.week_dir / "raw" / "synthesis_transcript.json").write_text(
            json.dumps(result.transcript, indent=2, default=str) + "\n", encoding="utf-8"
        )
        if result.failure:
            ctx.failures.append(f"synthesis: {result.failure}")
        if result.output is not None:
            applied = apply_synthesis(
                result.output,
                cited=result.cited_keys,
                annotations=annotations,
                hypotheses=hyps,
                taxonomy=ctx.taxonomy,
                episodes_by_stream=ctx.episodes,
                forecast_dates=ctx.forecast_dates(),
                runs_per_cutoff=ctx.runs_per_cutoff(),
                week=ctx.week,
                settings=ctx.settings,
            )
            ctx.evidence = applied.evidence
            drafts = applied.drafts
            ctx.manifest["synthesis"].update(
                {
                    "created": applied.created,
                    "updated": applied.updated,
                    "dropped_keys": applied.dropped_keys,
                    "drafts": len(drafts),
                }
            )
            print(
                f"synthesis: {len(applied.created)} new hypotheses, {len(applied.updated)} updated, {len(drafts)} draft(s)"
            )
    if not ctx.evidence:
        current_all = {k: a for per in annotations.values() for k, a in per.items()}
        ctx.evidence = {
            hid: evidence_for(
                h,
                annotations=current_all,
                episodes_by_stream=ctx.episodes,
                forecast_dates=ctx.forecast_dates(),
                runs_per_cutoff=ctx.runs_per_cutoff(),
                taxonomy=ctx.taxonomy,
                settings=ctx.settings,
            )
            for hid, h in hyps.state().items()
            if h.status not in ("retired", "refuted", "merged")
        }
    flat_annotations = {k: a for per in annotations.values() for k, a in per.items()}
    ctx.outcomes = await run_gates(
        drafts,
        hypotheses=hyps.state(),
        evidence=ctx.evidence,
        proposals=props,
        corpora=ctx.corpora,
        annotations=flat_annotations,
        cards_by_key=ctx.cards_by_key,
        client=client,
        settings=ctx.settings,
    )
    state = hyps.state()
    for draft, outcome in zip(drafts, ctx.outcomes, strict=False):
        if outcome.decision != "pass":
            continue
        proposal = to_proposal(
            draft,
            outcome,
            state[outcome.hypothesis_id],
            ctx.evidence.get(outcome.hypothesis_id),
            proposal_id=props.next_id(),
            week=ctx.week,
        )
        props.save(proposal)
        ctx.new_proposals.append(proposal)
    ctx.manifest["gates"] = [o.model_dump(mode="json") for o in ctx.outcomes]
    ctx.manifest["new_proposals"] = [p.id for p in ctx.new_proposals]
    print(
        f"gates: {sum(o.decision == 'pass' for o in ctx.outcomes)} pass / {sum(o.decision == 'downgrade' for o in ctx.outcomes)} downgrade / {sum(o.decision == 'block' for o in ctx.outcomes)} block; {len(ctx.new_proposals)} proposal(s) written"
    )


async def stage_metrics(ctx: WeekContext, client: LlmClient | None, store: AnnotationStore) -> None:
    if client is not None:
        written = await coach_metrics.blind_retriage(
            client, cards=ctx.cards_by_key, store=store, taxonomy=ctx.taxonomy, week=ctx.week, settings=ctx.settings
        )
        ctx.manifest["blind_retriage"] = len(written)
    hyps = HypothesisStore(ctx.settings).state()
    flat = {k: a for per in ctx.annotations(store).values() for k, a in per.items()}
    ctx.metrics = {
        "tag_reliability": coach_metrics.tag_reliability(store, list(ctx.corpora)),
        "predictive_precision": coach_metrics.predictive_precision(hyps, flat, weeks_order=[]),
    }
    coach_metrics.write_metrics(ctx.settings, ctx.week, ctx.metrics)


def stage_ship_watch(ctx: WeekContext) -> None:
    props = ProposalStore(ctx.settings).load_all()
    changes, verdicts = ship_watch.run(ctx.settings, props, ctx.corpora)
    ctx.manifest["ship_watch"] = changes
    ctx.post_ship = verdicts


def stage_render(ctx: WeekContext, ledger: CostLedger) -> Path:
    props = ProposalStore(ctx.settings).load_all()
    hyps = HypothesisStore(ctx.settings).state()
    text = render_report(
        week=ctx.week,
        corpora=ctx.corpora,
        frames=ctx.frames,
        episodes=ctx.episodes,
        proposals=props,
        outcomes=ctx.outcomes,
        hypotheses=hyps,
        evidence=ctx.evidence,
        deep_dives=ctx.deep_dives,
        base_dominated=ctx.base_dominated,
        pairing=[],
        post_ship=ctx.post_ship,
        metrics=ctx.metrics,
        ledger_by_stage=ledger.by_stage(),
        spent_usd=ledger.spent_usd,
        budget_usd=ledger.budget_usd,
        manifest=ctx.manifest,
        failures=ctx.failures,
    )
    ctx.week_dir.mkdir(parents=True, exist_ok=True)
    path = ctx.week_dir / "report.md"
    path.write_text(text, encoding="utf-8")
    (ctx.week_dir / "report.html").write_text(render_html(text, title=f"Coach review {ctx.week}"), encoding="utf-8")
    return path


# -- entry --------------------------------------------------------------------


def make_client(settings: ReviewSettings, ledger: CostLedger, *, no_llm: bool) -> LlmClient:
    if no_llm:
        return FakeClient(settings=settings, ledger=ledger, default=None)
    base, key = load_proxy_credentials()
    return ProxyClient(settings, ledger, api_base=base, api_key=key)


async def run(
    args: argparse.Namespace,
    settings: ReviewSettings = DEFAULT_REVIEW_SETTINGS,
    *,
    corpora: dict[str, StreamCorpus] | None = None,
    client: LlmClient | None = None,
) -> WeekContext:
    streams = tuple(stream_for(s) for s in args.stream) if args.stream else SCHEDULED_STREAMS
    ctx = WeekContext(
        week=args.week or week_id(), settings=settings, state=ReviewState(settings.state_path), taxonomy=Taxonomy.load()
    )
    ctx.manifest = {
        "week": ctx.week,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "model": settings.model,
        "settings": settings.model_dump(mode="json"),
        "no_llm": args.no_llm,
        "streams": [s.stream_id for s in streams],
    }
    ctx.regate = bool(getattr(args, "regate", False))
    budget = args.budget_usd or (settings.bootstrap_budget_usd if args.bootstrap else settings.weekly_budget_usd)
    ledger = CostLedger(budget, settings)
    store = AnnotationStore(settings)

    stage_collect(ctx, streams, corpora=corpora)
    stage_cards(ctx)
    stage_stats(ctx)
    stage_seed(ctx)
    ctx.state.save()
    est = preflight(ctx, ledger)
    print(
        f"pre-flight: {est['triage_calls']} triage call(s) ~${est['triage_usd']:.3f}; bound incl. deep dive + synthesis ~${est['total_usd_bound']:.2f} of ${budget:.2f}"
    )
    if args.dry_run:
        ctx.manifest["stopped"] = "dry_run"
        ctx.save_manifest()
        return ctx

    live_client: LlmClient | None = None
    if not args.no_llm:
        live_client = client or make_client(settings, ledger, no_llm=False)
        if client is not None:
            live_client.ledger = ledger  # type: ignore[attr-defined]
    try:
        if live_client is not None:
            supported = await live_client.probe_reasoning_effort()
            ctx.manifest["reasoning_effort_supported"] = supported
            ctx.manifest["probe_log"] = getattr(live_client, "probe_log", [])
            print(f"probe: reasoning_effort supported={supported}")
            await stage_triage(ctx, live_client, store)
            await stage_deep_dive(ctx, live_client, store)
        await stage_synthesis_and_gates(ctx, live_client, store)
        await stage_metrics(ctx, live_client, store)
        stage_ship_watch(ctx)
    except BudgetExceededError as exc:
        ctx.failures.append(f"budget: {exc}")
        print(f"STOP: {exc}")
    finally:
        ctx.state.save()
        ctx.manifest["usage"] = ledger.by_stage()
        ctx.manifest["cost_usd"] = ledger.spent_usd
        ctx.manifest["failures"] = ctx.failures
        ledger.write_jsonl(ctx.week_dir / "usage.jsonl")
        report = stage_render(ctx, ledger)
        ctx.manifest["report"] = str(report)
        ctx.manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
        ctx.save_manifest()
        print(f"report: {report}")
    return ctx


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stream", action="append", default=None)
    parser.add_argument("--week", default=None, help="ISO week id, default this week")
    parser.add_argument(
        "--no-llm", action="store_true", help="stages A-C, seeds, gates on existing hypotheses, report; zero spend"
    )
    parser.add_argument("--dry-run", action="store_true", help="pre-flight only")
    parser.add_argument("--budget-usd", type=float, default=None)
    parser.add_argument("--bootstrap", action="store_true", help="use the bootstrap budget ceiling")
    parser.add_argument(
        "--regate", action="store_true", help="re-run the gates from this week's stored synthesis (no synthesis call)"
    )
    args = parser.parse_args(argv)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()


__all__ = [
    "WeekContext",
    "make_client",
    "preflight",
    "run",
    "stage_cards",
    "stage_collect",
    "stage_deep_dive",
    "stage_metrics",
    "stage_render",
    "stage_seed",
    "stage_ship_watch",
    "stage_stats",
    "stage_synthesis_and_gates",
    "stage_triage",
    "week_header",
    "week_id",
]
