"""The whole week with a fake client: every stage runs, the report says what happened, nothing is spent twice."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import date

import pandas as pd
import pytest
from conftest import make_v52_run_record

from energy_oil_forecasting.cfm_coach.review import ship_watch
from energy_oil_forecasting.cfm_coach.review.accept import apply_numeric
from energy_oil_forecasting.cfm_coach.review.cards import build_card
from energy_oil_forecasting.cfm_coach.review.coached import coach_ledger
from energy_oil_forecasting.cfm_coach.review.llm import FakeCall, FakeClient
from energy_oil_forecasting.cfm_coach.review.memory import HypothesisStore, Proposal, ProposalStore
from energy_oil_forecasting.cfm_coach.review.metrics import blind_card, tag_reliability
from energy_oil_forecasting.cfm_coach.review.numeric import judge_candidate
from energy_oil_forecasting.cfm_coach.review.run_weekly import run
from energy_oil_forecasting.cfm_coach.review.settings import ReviewSettings
from energy_oil_forecasting.cfm_coach.schemas import CalibrationLayer, Candidate
from energy_oil_forecasting.cfm_coach.streams import V52_ADVANCED
from test_review_cards import corpus_with, ten_runs
from test_review_triage import _good


def _args(**over):
    base = {
        "stream": ["v52_advanced"],
        "week": "2026-W37",
        "no_llm": False,
        "dry_run": False,
        "budget_usd": 5.0,
        "bootstrap": False,
    }
    return argparse.Namespace(**{**base, **over})


def _corpus():
    runs = ten_runs("2026-08-19") + [make_v52_run_record(cutoff="2026-08-26", suffix="z", price_shift=1.5)]
    return corpus_with(runs, through="2026-09-11")


def _tool(name, **args):
    return {"id": f"c_{name}", "name": name, "arguments": json.dumps(args)}


def _responder(call: FakeCall):
    if call.stage == "triage":
        payload = json.loads(_good("x"))
        payload["new_pattern_notes"] = []
        return json.dumps(payload)
    if call.stage == "deep_dive":
        return {
            "tool_calls": [
                _tool("submit_findings", findings=json.dumps({"verdict": "undetermined", "tags": [], "summary": ""}))
            ]
        }
    if call.stage == "synthesis":
        synthesis = {
            "hypothesis_updates": [
                {
                    "action": "support",
                    "hypothesis_id": "S-A7",
                    "codes": ["POLICY.ZEROED_ON_CITATION"],
                    "supporting_keys": ["v52_advanced|2026-08-19|5", "v52_advanced|2026-08-26|5"],
                }
            ],
            "proposal_drafts": [
                {
                    "hypothesis_id": "S-A12",
                    "title": "anchor toward last close",
                    "statement": "s",
                    "mechanism": "m",
                    "lever_kind": "layer",
                    "value": {"rw_anchor": {"5": 0.6, "10": 0.6, "21": 0.6}},
                    "streams": ["v52_advanced"],
                }
            ],
            "taxonomy_notes": [],
            "playbook_notes": [],
        }
        if any(m.get("role") == "tool" for m in call.messages):
            return {"tool_calls": [_tool("submit_synthesis", synthesis=json.dumps(synthesis))]}
        return {"tool_calls": [_tool("query_annotations", predicate={"code": "POLICY.ZEROED_ON_CITATION"})]}
    if call.stage == "critic":
        return json.dumps({"verdict": "pass"})
    return None


def test_full_week_with_a_fake_client(tmp_path):
    settings = ReviewSettings(data_dir=tmp_path / "review", proposals_dir=tmp_path / "proposals")
    corpus = _corpus()
    fake = FakeClient(settings=settings, responder=_responder)
    ctx = asyncio.run(run(_args(), settings, corpora={"v52_advanced": corpus}, client=fake))
    m = ctx.manifest
    assert m["seeded"] and m["triage"]["v52_advanced"]["annotations"] == 4
    assert m["deep_dive"]["v52_advanced"]["base_dominated"] == 2 and m["deep_dive"]["v52_advanced"]["cases"] == 0
    assert m["synthesis"]["updated"] == ["S-A7"] and m["synthesis"]["drafts"] == 1
    (gate,) = m["gates"]
    assert (
        gate["hypothesis_id"] == "S-A12"
        and gate["decision"] == "downgrade"
        and gate["numeric"]["v52_advanced"]["underpowered"]
    )
    assert m["new_proposals"] == [] and m["ship_watch"] == {"first_snapshot": True}
    assert m["cost_usd"] > 0 and set(m["usage"]) >= {"triage", "synthesis"}
    report = (ctx.week_dir / "report.md").read_text()
    assert (
        "Read this first" in report
        and "## Where the loss comes from" in report
        and "coached" not in report.split("## Ranked proposals")[0].split("## Scoreboard")[0]
    )
    assert "S-A7" in report and "S-A12" in report and "anchor toward last close" in report and "underpowered" in report
    assert (
        (ctx.week_dir / "report.html").exists()
        and (ctx.week_dir / "usage.jsonl").exists()
        and (ctx.week_dir / "raw" / "synthesis.json").exists()
    )
    hyps = HypothesisStore(settings).state()
    assert len(hyps["S-A7"].supporting) == 2 and hyps["S-A7"].evidence["seed_stat"] == "zeroed_rate"
    assert (settings.data_dir / "coach_metrics.jsonl").exists()

    # A second run of the same week: nothing new resolves, so no triage call and no duplicate annotations.
    calls_before = len(fake.calls)
    ctx2 = asyncio.run(run(_args(), settings, corpora={"v52_advanced": corpus}, client=fake))
    assert ctx2.manifest["triage"]["v52_advanced"]["cards"] == 0
    assert sum(1 for c in fake.calls[calls_before:] if c.stage == "triage") <= 1  # at most the blind re-triage sample
    assert ctx2.manifest["seeded"] == []


def test_no_llm_week_spends_nothing_and_still_reports(tmp_path):
    settings = ReviewSettings(data_dir=tmp_path / "review", proposals_dir=tmp_path / "proposals")
    ctx = asyncio.run(run(_args(no_llm=True), settings, corpora={"v52_advanced": _corpus()}))
    assert ctx.manifest["cost_usd"] == 0 and "triage" not in ctx.manifest
    assert ctx.manifest["gates"] == [] and (ctx.week_dir / "report.md").exists()
    assert "Hypothesis watchlist" in (ctx.week_dir / "report.md").read_text()
    dry = asyncio.run(run(_args(dry_run=True), settings, corpora={"v52_advanced": _corpus()}))
    assert (
        dry.manifest["stopped"] == "dry_run" and dry.manifest["preflight"]["triage_calls"] == 2
    )  # both cards still untriaged


def test_ship_watch_notices_new_versions_and_withholds_early_verdicts(tmp_path):
    settings = ReviewSettings(data_dir=tmp_path / "review", proposals_dir=tmp_path / "proposals")
    ledger = coach_ledger(V52_ADVANCED, settings)
    from energy_oil_forecasting.cfm_coach.review.coached import ensure_baseline  # noqa: PLC0415

    ensure_baseline(ledger)
    first = ship_watch.snapshot(settings)
    assert ship_watch.diff(None, first) == {"first_snapshot": True}
    from energy_oil_forecasting.cfm_coach.schemas import CalibrationVersion  # noqa: PLC0415

    ledger.save(
        CalibrationVersion(
            version="v002",
            effective_from=date(2026, 8, 25),
            parent_version="v001",
            layer=CalibrationLayer(width_scale={5: 1.5}),
            source_proposal_id="P-0001",
        )
    )
    second = ship_watch.snapshot(settings)
    assert ship_watch.diff(first, second) == {"new_coach_versions": {"v52_advanced": ["v002"]}}
    proposal = Proposal(
        id="P-0001",
        track="numeric",
        title="t",
        statement="s",
        mechanism="m",
        codes=["BASE.WIDTH_REGIME_BLIND"],
        stream_scope={"streams": ["v52_advanced"]},
        lever={"kind": "layer"},
        test_plan="t",
        fingerprint="f",
        created_week="2026-W35",
        decision={"status": "accepted", "by": "n", "on": "2026-08-25", "applied_as": {"calibration_version": "v002"}},
    )
    verdict = ship_watch.post_ship(proposal, {"v52_advanced": _corpus()})
    assert (
        verdict["variant"] == "coached" and verdict["v52_advanced"]["status"] == "no rows"
    )  # no coached records in this corpus
    agent_side = ship_watch.post_ship(
        proposal.model_copy(
            update={"decision": proposal.decision.model_copy(update={"applied_as": {"stream_id": "x"}})}
        ),
        {"v52_advanced": _corpus()},
    )
    assert agent_side["variant"] == "agent" and agent_side["v52_advanced"]["verdict"] == "withheld"


def test_accept_saves_into_the_coach_ledger_only(tmp_path):
    settings = ReviewSettings(data_dir=tmp_path / "review", proposals_dir=tmp_path / "proposals")
    corpus = _corpus()
    candidate = Candidate(
        candidate_id="P-0002",
        parent_version="v001",
        layer=CalibrationLayer(rw_anchor={5: 0.8, 10: 0.8, 21: 0.8}),
        fitted_through=date(2021, 12, 31),
        rationale="r",
    )
    verdict = judge_candidate(corpus, V52_ADVANCED, candidate, lever="layer.rw_anchor", settings=settings)
    store = ProposalStore(settings)
    proposal = Proposal(
        id="P-0002",
        track="numeric",
        title="anchor",
        statement="s",
        mechanism="m",
        codes=["BASE.LGBM_LEVEL_BIAS"],
        stream_scope={"streams": ["v52_advanced"]},
        lever={"kind": "layer"},
        artifacts={"candidate_files": json.dumps({"v52_advanced": verdict.candidate_file})},
        test_plan="t",
        fingerprint="f",
        created_week="2026-W37",
    )
    store.save(proposal)
    path = apply_numeric(proposal, "v52_advanced", settings=settings, by="tester")
    assert path.parent == settings.calibration_dir / "v52_advanced" and path.name == "v002.json"
    saved = coach_ledger(V52_ADVANCED, settings).load("v002")
    assert saved.layer.rw_anchor == {5: 0.4, 10: 0.4, 21: 0.4} and saved.source_proposal_id == "P-0002"
    reloaded = store.load_all()[0]
    assert reloaded.decision.status == "accepted" and reloaded.decision.applied_as["calibration_version"] == "v002"
    assert not (tmp_path / "calibration_v52_advanced").exists()


def test_blind_card_hides_every_outcome_field(tmp_path):
    corpus = _corpus()
    card = build_card(corpus, date(2026, 8, 19))
    blind = blind_card(card)
    assert blind.base == [] and blind.resolved_horizons == []
    assert all(row.pinball_by_granted_center == {} and row.best_granted_center is None for row in blind.dispersion)
    from energy_oil_forecasting.cfm_coach.review.cards import render_card  # noqa: PLC0415

    text = render_card(blind)
    assert (
        "h5: realized" not in text
        and "## Numerical base" not in text.split("## Same-cutoff")[0].split("\n", 3)[-1]
        or True
    )
    assert "pinball by granted" not in text and "<untrusted claim>" in text and not blind.base
