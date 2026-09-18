"""Synthesis apply, gates, challenger drafting and pairing: the closed loop, offline."""

from __future__ import annotations

import asyncio
import json
from datetime import date

import pytest
import yaml
from conftest import make_v52_run_record

from energy_oil_forecasting.cfm_coach.review.cards import build_card
from energy_oil_forecasting.cfm_coach.review.challenger import (
    DraftError,
    draft_challenger,
    manifest_lines,
    verify_manifest,
)
from energy_oil_forecasting.cfm_coach.review.gates import run_gates, to_proposal
from energy_oil_forecasting.cfm_coach.review.llm import FakeClient
from energy_oil_forecasting.cfm_coach.review.memory import (
    AnnotationStore,
    HypothesisStore,
    Proposal,
    ProposalStore,
    Taxonomy,
)
from energy_oil_forecasting.cfm_coach.review.pairing import compare_corpora
from energy_oil_forecasting.cfm_coach.review.settings import ReviewSettings
from energy_oil_forecasting.cfm_coach.review.stats import strata_frame
from energy_oil_forecasting.cfm_coach.review.synthesis import (
    ProposalDraft,
    SynthesisOut,
    SynthesisTools,
    apply_synthesis,
    run_synthesis_loop,
    build_messages,
)
from energy_oil_forecasting.cfm_coach.review.triage import run_triage
from energy_oil_forecasting.cfm_coach.targets import V52
from test_review_cards import corpus_with
from test_review_triage import _good, _setup


TAX = Taxonomy.load()
WEEK = "2026-W37"


def _world(tmp_path):
    settings = ReviewSettings(data_dir=tmp_path / "review", proposals_dir=tmp_path / "proposals")
    _, corpus, state, cards = _setup(tmp_path / "review")
    store = AnnotationStore(settings)
    fake = FakeClient(settings=settings, responder=lambda call: _good("x"))
    asyncio.run(run_triage(fake, corpus=corpus, cards=cards, state=state, store=store, taxonomy=TAX, week=WEEK))
    annotations = {"v52_advanced": store.current("v52_advanced")}
    hyps = HypothesisStore(settings)
    props = ProposalStore(settings)
    frames = {"v52_advanced": strata_frame(corpus)}
    tools = SynthesisTools(
        corpora={"v52_advanced": corpus}, frames=frames, annotations=annotations, hypotheses=hyps, settings=settings
    )
    return settings, corpus, state, cards, store, annotations, hyps, props, tools


def _tool(name, **args):
    return {"id": f"c_{name}", "name": name, "arguments": json.dumps(args)}


def test_synthesis_tools_count_and_only_returned_keys_can_support(tmp_path):
    settings, corpus, state, cards, store, annotations, hyps, props, tools = _world(tmp_path)
    counted = json.loads(tools.count_scope({"stream": "v52_advanced", "horizon": 5}))
    assert counted["rows"] == 11 and counted["distinct_cutoffs"] == 2 and counted["effective_n"] == pytest.approx(0.4)
    listed = json.loads(tools.query_annotations({"code": "POLICY.ZEROED_ON_CITATION"}))
    assert len(listed) == 2 and all("POLICY.ZEROED_ON_CITATION@h5.granted_center" in x["tags"] for x in listed)
    assert tools.cited_keys == {x["key"] for x in listed}
    priced = json.loads(tools.price_counterfactual("v52_advanced", "rw_anchor", {"a": 1.0}))
    assert priced["fidelity"] == "exact" and priced["by_horizon"][0]["horizon"] == 5
    base = json.loads(tools.base_table("v52_advanced"))
    assert {row["horizon"] for row in base} == {5, 10}

    out = SynthesisOut.model_validate(
        {
            "hypothesis_updates": [
                {
                    "action": "create",
                    "statement": "zeroing on citation",
                    "mechanism": "m",
                    "track": "llm",
                    "lever": {"kind": "skill_text", "file": "skills/claim-building/SKILL.md"},
                    "codes": ["POLICY.ZEROED_ON_CITATION"],
                    "scope": {},
                    "prediction": "p",
                    "supporting_keys": [*tools.cited_keys, "v52_advanced|2026-01-01|5"],
                },
                {
                    "action": "create",
                    "statement": "no codes",
                    "mechanism": "m",
                    "track": "llm",
                    "codes": ["NOT.A.CODE"],
                    "prediction": "p",
                },
            ],
            "proposal_drafts": [
                {
                    "hypothesis_id": "H-0001",
                    "title": "t",
                    "statement": "s",
                    "mechanism": "m",
                    "lever_kind": "skill_text",
                    "text_edits": [{"file": "skills/claim-building/SKILL.md", "find": "x", "replace": "y"}],
                    "streams": ["v52_advanced"],
                }
            ],
        }
    )
    applied = apply_synthesis(
        out,
        cited=tools.cited_keys,
        annotations=annotations,
        hypotheses=hyps,
        taxonomy=TAX,
        episodes_by_stream={},
        forecast_dates={},
        runs_per_cutoff={},
        week=WEEK,
        settings=settings,
    )
    assert applied.created == ["H-0001"] and applied.dropped_keys == 1
    h = hyps.state()["H-0001"]
    assert len(h.supporting) == 2 and h.status == "candidate"
    assert applied.evidence["H-0001"].distinct_cutoffs == 2 and not applied.evidence["H-0001"].promotable
    assert [d.hypothesis_id for d in applied.drafts] == ["H-0001"]


def test_synthesis_loop_ends_with_submit_and_applies(tmp_path):
    settings, corpus, state, cards, store, annotations, hyps, props, tools = _world(tmp_path)
    synthesis = json.dumps(
        {"hypothesis_updates": [], "proposal_drafts": [], "taxonomy_notes": ["n"], "playbook_notes": []}
    )
    fake = FakeClient(
        settings=settings,
        script={
            "synthesis": [
                {
                    "tool_calls": [
                        _tool("base_table", stream="v52_advanced"),
                        _tool("query_annotations", predicate={"code": "BASE.LGBM_LEVEL_BIAS"}),
                    ]
                },
                {"tool_calls": [_tool("submit_synthesis", synthesis="{bad")]},
                {"tool_calls": [_tool("submit_synthesis", synthesis=synthesis)]},
            ]
        },
    )
    messages, _ = build_messages(
        week=WEEK, taxonomy=TAX, hypotheses=hyps, proposals=props, week_summary="w", playbook="p"
    )
    result = asyncio.run(run_synthesis_loop(fake, tools, messages, settings=settings))
    assert result.failure is None and result.output.taxonomy_notes == ["n"] and result.turns == 3
    assert result.tool_calls == ["base_table", "query_annotations", "submit_synthesis", "submit_synthesis"]


def _hypothesis(hyps, *, track, codes, supporting, status="promoted", lever=None):
    from energy_oil_forecasting.cfm_coach.review.memory import Hypothesis, HypothesisEvent  # noqa: PLC0415

    hid = hyps.next_id()
    hyps.create(
        Hypothesis(
            hypothesis_id=hid,
            statement="s",
            mechanism="m",
            track=track,
            lever=lever or {},
            codes=codes,
            scope={},
            prediction="p",
            supporting=supporting,
        ),
        week=WEEK,
    )
    if status != "candidate":
        hyps.append(HypothesisEvent(event="status", hypothesis_id=hid, week=WEEK, payload={"status": status}))
    return hid


def test_gates_block_rejected_downgrade_unpromoted_and_judge_numeric(tmp_path):
    settings, corpus, state, cards, store, annotations, hyps, props, tools = _world(tmp_path)
    keys = [str(k) for k in annotations["v52_advanced"]]
    cards_by_key = {(c.stream, c.cutoff): c for c in cards.values()}
    h_llm = _hypothesis(hyps, track="llm", codes=["POLICY.ZEROED_ON_CITATION"], supporting=keys, status="candidate")
    h_prom = _hypothesis(hyps, track="llm", codes=["POLICY.ZEROED_ON_CITATION"], supporting=keys, status="promoted")
    h_num = _hypothesis(hyps, track="numeric", codes=["BASE.LGBM_LEVEL_BIAS"], supporting=keys, status="candidate")
    edits = [{"file": "skills/claim-building/SKILL.md", "find": "x", "replace": "y"}]
    drafts = [
        ProposalDraft(
            hypothesis_id=h_llm,
            title="unpromoted",
            statement="s",
            mechanism="m",
            lever_kind="skill_text",
            text_edits=edits,
            streams=["v52_advanced"],
            pricing_op="no_change",
        ),
        ProposalDraft(
            hypothesis_id=h_prom,
            title="base dominated",
            statement="s",
            mechanism="m",
            lever_kind="persona",
            file_line="agent.py:36",
            streams=["v52_advanced"],
        ),
        ProposalDraft(
            hypothesis_id=h_num,
            title="anchor",
            statement="s",
            mechanism="m",
            lever_kind="layer",
            value={"rw_anchor": {"5": 0.8, "10": 0.8, "21": 0.8}},
            streams=["v52_advanced"],
        ),
        ProposalDraft(
            hypothesis_id=h_num,
            title="denied",
            statement="s",
            mechanism="m",
            lever_kind="settings_overlay",
            field_name="audit_enabled",
            value={"value": False},
            streams=["v52_advanced"],
        ),
    ]
    # A previously rejected persona proposal with the same fingerprint.
    rejected = Proposal(
        id="P-0001",
        track="llm",
        title="old",
        statement="s",
        mechanism="m",
        codes=["POLICY.ZEROED_ON_CITATION"],
        stream_scope={},
        lever={"kind": "persona", "file": "agent.py:36"},
        test_plan="t",
        fingerprint=Proposal.make_fingerprint(
            {"kind": "persona", "file": "agent.py:36"}, ["POLICY.ZEROED_ON_CITATION"]
        ),
        created_week="2026-W30",
    )
    path = props.save(rejected)
    payload = yaml.safe_load(path.read_text())
    payload["decision"] = {"status": "rejected", "by": "n", "on": "2026-09-10", "reason": "r"}
    path.write_text(yaml.safe_dump(payload, sort_keys=False))

    from energy_oil_forecasting.cfm_coach.review.gates import evidence_for  # noqa: PLC0415

    ev = {
        hid: evidence_for(
            hyps.state()[hid],
            annotations=annotations["v52_advanced"],
            episodes_by_stream={},
            forecast_dates={},
            runs_per_cutoff={},
            taxonomy=TAX,
            settings=settings,
        )
        for hid in (h_llm, h_prom, h_num)
    }
    outcomes = asyncio.run(
        run_gates(
            drafts,
            hypotheses=hyps.state(),
            evidence=ev,
            proposals=props,
            corpora={"v52_advanced": corpus},
            annotations=annotations["v52_advanced"],
            cards_by_key=cards_by_key,
            client=None,
            settings=settings,
        )
    )
    by_title = {o.title: o for o in outcomes}
    assert by_title["unpromoted"].decision == "downgrade" and any(
        "is candidate" in r for r in by_title["unpromoted"].reasons
    )
    assert by_title["unpromoted"].pricing is not None and by_title["unpromoted"].rank_score is not None
    assert by_title["base dominated"].decision == "block" and "blocked: P-0001" in by_title["base dominated"].reasons[0]
    assert any("base-attributable" in r for r in by_title["base dominated"].reasons)
    anchor = by_title["anchor"]
    assert anchor.numeric["v52_advanced"]["underpowered"] and anchor.numeric["v52_advanced"]["candidate_file"].endswith(
        "v002.json"
    )
    assert anchor.decision == "downgrade" and anchor.rank_score is not None
    assert by_title["denied"].decision == "block" and "denylist" in by_title["denied"].reasons[0]

    proposal = to_proposal(drafts[2], anchor, hyps.state()[h_num], ev[h_num], proposal_id=props.next_id(), week=WEEK)
    saved = props.save(proposal)
    reloaded = [p for p in props.load_all() if p.id == proposal.id][0]
    assert (
        reloaded.track == "numeric" and "candidate_files" in reloaded.artifacts and reloaded.decision.status == "open"
    )


def test_critic_downgrade_is_honoured_on_passing_items(tmp_path):
    settings, corpus, state, cards, store, annotations, hyps, props, tools = _world(tmp_path)
    keys = [str(k) for k in annotations["v52_advanced"]]
    h = _hypothesis(hyps, track="policy", codes=["POLICY.ZEROED_ON_CITATION"], supporting=keys, status="promoted")
    draft = ProposalDraft(
        hypothesis_id=h,
        title="widen",
        statement="s",
        mechanism="m",
        lever_kind="code",
        file_line="policy/evidence_policy.py:200",
        streams=["v52_advanced"],
    )
    critic = json.dumps(
        {"verdict": "downgrade", "hindsight": "only one rally", "mechanism": "", "confound": "", "blind_spot": ""}
    )
    fake = FakeClient(settings=settings, script={"critic": [critic]})
    from energy_oil_forecasting.cfm_coach.review.gates import evidence_for  # noqa: PLC0415

    ev = {
        h: evidence_for(
            hyps.state()[h],
            annotations=annotations["v52_advanced"],
            episodes_by_stream={},
            forecast_dates={},
            runs_per_cutoff={},
            taxonomy=TAX,
            settings=settings,
        )
    }
    (outcome,) = asyncio.run(
        run_gates(
            [draft],
            hypotheses=hyps.state(),
            evidence=ev,
            proposals=props,
            corpora={"v52_advanced": corpus},
            annotations=annotations["v52_advanced"],
            cards_by_key={},
            client=fake,
            settings=settings,
        )
    )
    assert outcome.decision == "downgrade" and outcome.critic["hindsight"] == "only one rally" and len(fake.calls) == 1


# -- challenger ---------------------------------------------------------------


def test_manifest_regeneration_matches_the_shipped_format():
    mine = manifest_lines(V52.package_root)
    shipped = (V52.package_root / "MANIFEST.sha256").read_text().splitlines()
    # The shipped manifest predates CALIBRATION.md; every entry it has must match byte for byte, in order.
    assert [line for line in mine if not line.endswith("./CALIBRATION.md")] == shipped


def test_draft_challenger_rewrites_identity_applies_edits_and_regenerates_manifest(tmp_path):
    settings = ReviewSettings(data_dir=tmp_path / "review", proposals_dir=tmp_path / "proposals")
    skill = (V52.package_root / "skills" / "claim-building" / "SKILL.md").read_text()
    find = skill.splitlines()[-1]
    proposal = Proposal(
        id="P-0007",
        track="llm",
        title="cite every summary",
        statement="s",
        mechanism="m",
        codes=["POLICY.ZEROED_ON_CITATION"],
        stream_scope={},
        lever={
            "kind": "skill_text",
            "text_edits": [
                {
                    "file": "skills/claim-building/SKILL.md",
                    "find": find,
                    "replace": find + "\nAlways cite the accepted summary id for every material claim.",
                }
            ],
        },
        test_plan="t",
        fingerprint="f",
        created_week=WEEK,
    )
    draft = draft_challenger(proposal, settings=settings, start=date(2026, 9, 15))
    pkg = tmp_path / "review" / "challengers" / "P-0007" / draft.package_name
    assert pkg.exists() and verify_manifest(pkg) and draft.edits_applied == 1
    assert "Always cite the accepted summary id" in (pkg / "skills" / "claim-building" / "SKILL.md").read_text()
    assert f'AGENT_NAME = "{draft.package_name}"' in (pkg / "config.py").read_text()
    assert draft.module in (pkg / "agent.py").read_text() and "cfm_agent_v_5_2" not in (pkg / "agent.py").read_text()
    stream = yaml.safe_load((pkg.parent / "stream.yaml").read_text())
    assert (
        stream["runs_per_day"] == 3 and stream["model"] == "gemini-3.5-flash" and stream["champion"] == "v52_advanced"
    )
    register = (pkg.parent / "REGISTER.md").read_text()
    assert "SCHEDULED_STREAMS" in register and "shasum -a 256 -c" in register and "norecursedirs" in register
    assert (pkg.parent / "calibration" / "v001.json").exists()
    # A fingerprint different from v5.2's, by construction.
    assert (pkg / "MANIFEST.sha256").read_text() != (V52.package_root / "MANIFEST.sha256").read_text()

    twice = Proposal.model_validate(
        {
            **proposal.model_dump(mode="json"),
            "lever": {
                "kind": "skill_text",
                "text_edits": [{"file": "skills/claim-building/SKILL.md", "find": "claim", "replace": "x"}],
            },
        }
    )
    with pytest.raises(DraftError, match="exactly one match"):
        draft_challenger(twice, settings=settings)
    code = Proposal.model_validate(
        {**proposal.model_dump(mode="json"), "lever": {"kind": "code", "file_line": "x.py:1"}}
    )
    with pytest.raises(DraftError, match="not a text lever"):
        draft_challenger(code, settings=settings)


# -- pairing ------------------------------------------------------------------


def test_pairing_recovers_a_planted_effect_and_withholds_below_min_cutoffs(tmp_path):
    settings = ReviewSettings(data_dir=tmp_path)
    import pandas as pd  # noqa: PLC0415

    days = [d.date() for d in pd.bdate_range("2026-07-06", periods=16)]
    champion_records = [
        make_v52_run_record(cutoff=str(d), suffix=f"a{i}", price_shift=0.2 * i) for i, d in enumerate(days)
    ]
    # Challenger = same runs with the whole distribution nudged toward the last close (a real improvement here).
    challenger_records = []
    for r in champion_records:
        last = float(r.diagnostics["latest_value"])
        forecasts = []
        for item in r.forecasts:
            delta = 0.6 * (last - item.final_point_forecast)
            forecasts.append(
                item.model_copy(
                    update={
                        "final_point_forecast": item.final_point_forecast + delta,
                        "final_quantiles": {k: v + delta for k, v in item.final_quantiles.items()},
                    }
                )
            )
        challenger_records.append(r.model_copy(update={"run_id": r.run_id + "__chal", "forecasts": forecasts}))
    champion = corpus_with(champion_records, through="2026-09-30")
    challenger = corpus_with(challenger_records, through="2026-09-30")
    challenger.stream_id = "v53_test_advanced"
    report = compare_corpora(champion, challenger, settings=settings)
    assert report.shared_cutoffs == 16 and report.mean_delta is not None
    assert report.conditions["min_distinct_cutoffs"] and report.effect_pct is not None
    assert all(h.challenger_pinball != h.champion_pinball for h in report.by_horizon)
    short = compare_corpora(
        corpus_with(champion_records[:5], through="2026-09-30"),
        corpus_with(challenger_records[:5], through="2026-09-30"),
        settings=settings,
    )
    assert short.verdict.startswith("withheld") and short.conditions["min_distinct_cutoffs"] is False
    assert "vs" in report.describe()


def test_the_final_turn_forces_submit_synthesis(tmp_path):
    settings, corpus, state, cards, store, annotations, hyps, props, tools = _world(tmp_path)
    synthesis = json.dumps(
        {"hypothesis_updates": [], "proposal_drafts": [], "taxonomy_notes": ["forced"], "playbook_notes": []}
    )

    def responder(call):
        if call.tool_choice not in (None, "auto"):
            return {"tool_calls": [_tool("submit_synthesis", synthesis=synthesis)]}
        return {"tool_calls": [_tool("count_scope", predicate={})]}

    fake = FakeClient(settings=settings, responder=responder)
    messages, _ = build_messages(
        week=WEEK, taxonomy=TAX, hypotheses=hyps, proposals=props, week_summary="w", playbook="p"
    )
    result = asyncio.run(run_synthesis_loop(fake, tools, messages, settings=settings, max_turns=4))
    assert result.failure is None and result.output.taxonomy_notes == ["forced"] and result.turns == 4
    assert [c.tool_choice for c in fake.calls] == [
        "auto",
        "auto",
        "auto",
        {"type": "function", "function": {"name": "submit_synthesis"}},
    ]
    assert result.transcript[-1]["role"] == "assistant" and "budget is spent" in result.transcript[-2]["content"]


def test_synthesis_pricing_supports_ensemble_reweight(tmp_path):
    settings, corpus, state, cards, store, annotations, hyps, props, tools = _world(tmp_path)
    out = json.loads(
        tools.price_counterfactual(
            "v52_advanced", "ensemble_reweight", {"weights": {"arima": 0.5, "kalman": 0.5, "lightgbm": 0.0}}
        )
    )
    assert out["op"] == "ensemble_reweight" and out["fidelity"].startswith("approximate") and out["by_horizon"]
    assert tools.price_counterfactual(
        "v52_advanced", "settings", {"field": "ensemble_weights", "value": {}}
    ).startswith("pricing failed")


def test_a_schema_invalid_forced_synthesis_gets_one_correction_turn(tmp_path):
    settings, corpus, state, cards, store, annotations, hyps, props, tools = _world(tmp_path)
    good = json.dumps(
        {"hypothesis_updates": [], "proposal_drafts": [], "taxonomy_notes": ["fixed"], "playbook_notes": []}
    )
    forced = []

    def responder(call):
        if call.tool_choice not in (None, "auto"):
            forced.append(call)
            bad = json.dumps({"hypothesis_updates": [{"hypothesis_id": "S-A1", "status": "active"}], "notes": "x"})
            return {"tool_calls": [_tool("submit_synthesis", synthesis=bad if len(forced) == 1 else good)]}
        return {"tool_calls": [_tool("count_scope", predicate={})]}

    fake = FakeClient(settings=settings, responder=responder)
    messages, _ = build_messages(
        week=WEEK, taxonomy=TAX, hypotheses=hyps, proposals=props, week_summary="w", playbook="p"
    )
    assert '"hypothesis_updates"' in messages[0]["content"] and "$ref" not in messages[0]["content"]
    result = asyncio.run(run_synthesis_loop(fake, tools, messages, settings=settings, max_turns=3))
    assert result.failure is None and result.output.taxonomy_notes == ["fixed"] and result.turns == 4
    assert [c.tool_choice for c in fake.calls][:2] == ["auto", "auto"] and len(forced) == 2


def test_layer_drafts_map_field_name_and_scalar_values_to_candidates():
    from energy_oil_forecasting.cfm_coach.review.gates import _candidate_for, missing_files  # noqa: PLC0415
    from energy_oil_forecasting.cfm_coach.review.synthesis import ProposalDraft  # noqa: PLC0415

    base = {"hypothesis_id": "S-A12", "title": "t", "statement": "s", "mechanism": "m", "lever_kind": "layer"}
    named = ProposalDraft(**base, field_name="rw_anchor", value={"a": 0.5}, horizons=[5, 10])
    assert _candidate_for(named, "S-A12", "v52_lite").layer.rw_anchor == {5: 0.5, 10: 0.5}
    nested = ProposalDraft(**base, value={"width_scale": {"5": 1.2, "h10": 1.3}})
    assert _candidate_for(nested, "S-A12", "v52_lite").layer.width_scale == {5: 1.2, 10: 1.3}
    with pytest.raises(ValueError, match="no layer parameter"):
        _candidate_for(ProposalDraft(**base, value={"a": 0.5}), "S-A12", "v52_lite")
    with pytest.raises(ValueError, match="field_name"):
        _candidate_for(ProposalDraft(**{**base, "lever_kind": "settings_overlay"}, value={}), "S-A12", "v52_lite")
    code = {**base, "lever_kind": "code"}
    assert missing_files(ProposalDraft(**code, file_line="cfm_agent_v_5_2/pipeline/summarizer.py:45")) == [
        "cfm_agent_v_5_2/pipeline/summarizer.py"
    ]
    assert missing_files(ProposalDraft(**code, file_line="cfm_agent_v_5_2/config.py:11")) == []


def test_gates_downgrade_a_draft_that_names_a_missing_file(tmp_path):
    from energy_oil_forecasting.cfm_coach.review.synthesis import ProposalDraft  # noqa: PLC0415

    settings, corpus, state, cards, store, annotations, hyps, props, tools = _world(tmp_path)
    hid = _hypothesis(hyps, track="code", codes=["EVIDENCE.EMPTY_ACCEPTED_SUMMARY"], supporting=[], status="promoted")
    draft = ProposalDraft(
        hypothesis_id=hid,
        title="t",
        statement="s",
        mechanism="m",
        lever_kind="code",
        file_line="cfm_agent_v_5_2/pipeline/summarizer.py:45",
    )
    (outcome,) = asyncio.run(
        run_gates(
            [draft],
            hypotheses=hyps.state(),
            evidence={},
            proposals=props,
            corpora={},
            annotations={},
            cards_by_key={},
            client=None,
            settings=settings,
        )
    )
    assert outcome.decision == "downgrade" and any("do not exist" in r for r in outcome.reasons)


def test_apply_synthesis_dedupes_creates_remaps_drafts_and_can_run_without_writing(tmp_path):
    from energy_oil_forecasting.cfm_coach.review.synthesis import (
        HypothesisUpdate,
        ProposalDraft,
        SynthesisOut,
        stored_synthesis_output,
    )  # noqa: PLC0415

    settings, corpus, state, cards, store, annotations, hyps, props, tools = _world(tmp_path)
    out = SynthesisOut(
        hypothesis_updates=[
            HypothesisUpdate(
                action="create",
                hypothesis_id="S-NEW",
                statement="summaries arrive empty",
                mechanism="m",
                track="data",
                codes=["EVIDENCE.EMPTY_ACCEPTED_SUMMARY"],
                prediction="p",
            )
        ],
        proposal_drafts=[
            ProposalDraft(
                hypothesis_id="S-NEW",
                title="t",
                statement="s",
                mechanism="m",
                lever_kind="code",
                file_line="cfm_agent_v_5_2/config.py:1",
            )
        ],
    )
    kwargs = dict(
        cited=set(),
        annotations={},
        hypotheses=hyps,
        taxonomy=TAX,
        episodes_by_stream={},
        forecast_dates={},
        runs_per_cutoff={},
        week=WEEK,
        settings=settings,
    )
    first = apply_synthesis(out, **kwargs)
    (created,) = first.created
    assert first.id_map == {"S-NEW": created} and first.drafts[0].hypothesis_id == created
    events_after_first = hyps.path.read_text().count("\n")
    second = apply_synthesis(out, **kwargs)
    assert second.created == [] and second.id_map == {"S-NEW": created} and second.drafts[0].hypothesis_id == created
    assert (
        len(hyps.state()) == len(hyps.state())
        and sum(1 for h in hyps.state().values() if h.statement == "summaries arrive empty") == 1
    )
    events_after_second = hyps.path.read_text().count("\n")
    third = apply_synthesis(out, **kwargs, write_events=False)
    assert (
        third.drafts[0].hypothesis_id == created
        and hyps.path.read_text().count("\n") == events_after_second > events_after_first
    )
    week_dir = settings.weeks_dir / WEEK
    (week_dir / "raw").mkdir(parents=True)
    (week_dir / "raw" / "synthesis.json").write_text(
        json.dumps(
            [
                {"content": None, "tool_calls": [{"id": "c", "name": "count_scope", "arguments": "{}"}]},
                {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c2",
                            "name": "submit_synthesis",
                            "arguments": json.dumps({"synthesis": out.model_dump_json()}),
                        }
                    ],
                },
            ]
        )
    )
    assert stored_synthesis_output(week_dir).proposal_drafts[0].hypothesis_id == "S-NEW"
    assert stored_synthesis_output(tmp_path / "nowhere") is None
