"""Memory: append-only stores, two-unit evidence boundaries, rejection blocks, loud failures on bad YAML."""

from __future__ import annotations

from datetime import date

import pytest
import yaml

from energy_oil_forecasting.cfm_coach.review.gates import evidence_for, refuted, seed_statistics
from energy_oil_forecasting.cfm_coach.review.memory import (
    Annotation,
    AnnotationStore,
    Decision,
    Hypothesis,
    HypothesisEvent,
    HypothesisStore,
    Proposal,
    ProposalStore,
    Tag,
    Taxonomy,
    annotation_id,
    load_seeds,
    seed_hypotheses,
)
from energy_oil_forecasting.cfm_coach.review.settings import ReviewSettings
from energy_oil_forecasting.cfm_coach.review.state import ReviewKey
from energy_oil_forecasting.cfm_coach.review.stats import Episode, strata_frame
from energy_oil_forecasting.cfm_coach.review.cards import build_card
from test_review_cards import corpus_with, ten_runs


TAX = Taxonomy.load()


def _settings(tmp_path) -> ReviewSettings:
    return ReviewSettings(data_dir=tmp_path / "review", proposals_dir=tmp_path / "proposals")


def _annotation(stream, cutoff, horizon, *tags, stage="triage", supersedes=None):
    key = ReviewKey(stream, cutoff, horizon)
    return Annotation(
        annotation_id=annotation_id(key, "hash", "prompt", stage) + (supersedes or ""),
        stream=stream,
        cutoff=cutoff,
        horizon=horizon,
        card_hash="hash",
        card_version=1,
        prompt_hash="prompt",
        stage=stage,
        week="2026-W37",
        model="fake",
        tags=[
            Tag(
                code=c,
                track=TAX.resolve(c).track,
                severity=2,
                confidence=0.8,
                pointer=p,
                outcome_dependent=TAX.resolve(c).outcome_dependent,
            )
            for c, p in tags
        ],
        supersedes=supersedes,
    )


# -- taxonomy ----------------------------------------------------------------


def test_taxonomy_resolves_aliases_and_rejects_unknown_codes():
    tax = Taxonomy(version=2, codes=TAX.codes, aliases={"OLD.CODE": "POLICY.ZEROED_ON_CITATION", "A": "B", "B": "A"})
    assert tax.resolve("OLD.CODE").code == "POLICY.ZEROED_ON_CITATION"
    assert tax.resolve("NOT.A.CODE") is None
    assert tax.resolve("A") is None  # a cycle resolves to nothing rather than looping
    assert "new_pattern_notes" in tax.render()


# -- annotations ---------------------------------------------------------------


def test_annotations_are_append_only_and_dedupe_by_id(tmp_path):
    store = AnnotationStore(_settings(tmp_path))
    a = _annotation("v52_lite", date(2026, 8, 19), 5, ("POLICY.ZEROED_ON_CITATION", "h5.granted_center"))
    assert store.append(a) is True and store.append(a) is False
    b = _annotation(
        "v52_lite", date(2026, 8, 19), 5, ("JUDGMENT.DIRECTION_WRONG", "claim_001"), supersedes=a.annotation_id
    )
    assert store.append(b) is True
    current = store.current("v52_lite")
    assert current[a.key].annotation_id == b.annotation_id
    assert len(store.load("v52_lite")) == 2  # nothing rewritten


# -- hypotheses ---------------------------------------------------------------


def test_hypotheses_fold_from_events_and_refuse_duplicates(tmp_path):
    store = HypothesisStore(_settings(tmp_path))
    h = Hypothesis(
        hypothesis_id="H-0001",
        statement="s",
        mechanism="m",
        track="llm",
        lever={"kind": "skill_text"},
        codes=["POLICY.ZEROED_ON_CITATION"],
        scope={},
        prediction="p",
    )
    store.create(h, week="2026-W37")
    with pytest.raises(ValueError, match="already exists"):
        store.create(h, week="2026-W37")
    store.append(
        HypothesisEvent(
            event="supported", hypothesis_id="H-0001", week="2026-W38", payload={"keys": ["v52_lite|2026-08-19|5"]}
        )
    )
    store.append(
        HypothesisEvent(event="evidence", hypothesis_id="H-0001", week="2026-W38", payload={"new_support": False})
    )
    store.append(HypothesisEvent(event="status", hypothesis_id="H-0001", week="2026-W38", payload={"status": "stale"}))
    state = store.state()["H-0001"]
    assert state.supporting == ["v52_lite|2026-08-19|5"] and state.status == "stale"
    assert state.reviews_without_new_support == 1 and store.next_id() == "H-0002"
    with pytest.raises(ValueError, match="unknown hypothesis"):
        store.append(HypothesisEvent(event="status", hypothesis_id="H-9999", week="w", payload={"status": "stale"}))
        store.state()


# -- evidence: two units --------------------------------------------------------


def _episodes():
    return [
        Episode(
            episode_id="E1",
            start=date(2026, 8, 20),
            end=date(2026, 8, 27),
            direction=-1,
            start_price=85,
            end_price=82,
            threshold=3,
        ),
        Episode(
            episode_id="E2",
            start=date(2026, 8, 28),
            end=date(2026, 9, 12),
            direction=1,
            start_price=82,
            end_price=100,
            threshold=3,
        ),
    ]


def _evidence(h, keys, *, code, pointer="claim_001"):
    anns = {k: _annotation(k.stream, k.cutoff, k.horizon, (code, pointer)) for k in keys}
    return evidence_for(
        h.model_copy(update={"supporting": [str(k) for k in keys]}),
        annotations=anns,
        episodes_by_stream={"v52_lite": _episodes()},
        forecast_dates={k: date.fromordinal(k.cutoff.toordinal() + 14) for k in keys},
        runs_per_cutoff={(k.stream, k.cutoff): 10 for k in keys},
        taxonomy=TAX,
    )


def test_process_tags_count_distinct_cutoffs():
    h = Hypothesis(
        hypothesis_id="H-p",
        statement="s",
        mechanism="m",
        track="llm",
        lever={},
        codes=["POLICY.ZEROED_ON_CITATION"],
        scope={},
        prediction="p",
    )
    three = [ReviewKey("v52_lite", date(2026, 8, d), 5) for d in (19, 20, 21)]
    four = [*three, ReviewKey("v52_lite", date(2026, 8, 24), 5)]
    assert _evidence(h, three, code="POLICY.ZEROED_ON_CITATION", pointer="h5.granted_center").promotable is False
    summary = _evidence(h, four, code="POLICY.ZEROED_ON_CITATION", pointer="h5.granted_center")
    assert summary.unit == "distinct_cutoffs" and summary.promotable and summary.replication_rate == pytest.approx(0.1)


def test_outcome_tags_need_independent_windows_and_both_episode_signs():
    h = Hypothesis(
        hypothesis_id="H-o",
        statement="s",
        mechanism="m",
        track="llm",
        lever={},
        codes=["JUDGMENT.DIRECTION_WRONG"],
        scope={},
        prediction="p",
    )
    daily = [
        ReviewKey("v52_lite", date(2026, 9, d), 10) for d in (1, 2, 3, 4, 7)
    ]  # five cutoffs, one window, one episode
    s = _evidence(h, daily, code="JUDGMENT.DIRECTION_WRONG")
    assert s.unit == "independent_windows" and s.independent_windows == 1 and not s.promotable
    assert any("episode" in r for r in s.reasons) and any("window" in r for r in s.reasons)
    spread = [
        ReviewKey("v52_lite", date(2026, 8, 20), 5),
        ReviewKey("v52_lite", date(2026, 8, 28), 5),
        ReviewKey("v52_lite", date(2026, 9, 8), 5),
    ]
    s2 = _evidence(h, spread, code="JUDGMENT.DIRECTION_WRONG")
    assert s2.independent_windows == 3 and s2.episodes == 2 and s2.episode_signs == [-1, 1] and s2.promotable
    no_pointer = _evidence(h, spread, code="JUDGMENT.DIRECTION_WRONG", pointer="")
    assert not no_pointer.promotable and "pointer" in " ".join(no_pointer.reasons)
    assert not refuted(s2)
    contradicted = s2.model_copy(update={"contradicting_keys": 5})
    assert refuted(contradicted)


# -- proposals ------------------------------------------------------------------


def _proposal(store, **over):
    lever = over.pop("lever", {"kind": "skill_text", "file": "skills/claim-building/SKILL.md"})
    codes = over.pop("codes", ["POLICY.ZEROED_ON_CITATION"])
    return Proposal(
        id=store.next_id(),
        track="llm",
        title="t",
        statement="s",
        mechanism="m",
        codes=codes,
        stream_scope={},
        lever=lever,
        test_plan="register challenger",
        fingerprint=Proposal.make_fingerprint(lever, codes),
        created_week="2026-W37",
        **over,
    )


def test_proposals_write_once_block_rejected_fingerprints_and_fail_loudly_on_bad_yaml(tmp_path):
    settings = _settings(tmp_path)
    store = ProposalStore(settings)
    first = _proposal(store)
    path = store.save(first)
    assert path.name == "P-0001.yaml" and store.next_id() == "P-0002"
    with pytest.raises(FileExistsError):
        store.save(first)

    payload = yaml.safe_load(path.read_text())
    payload["decision"] = {"status": "rejected", "by": "naman", "on": "2026-09-20", "reason": "one rally only"}
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    assert store.load_all()[0].decision.reason == "one rally only"
    blocked, why = store.blocked(first.fingerprint, new_cutoffs_since_rejection=2)
    assert blocked and "2/3" in why
    assert store.blocked(first.fingerprint, new_cutoffs_since_rejection=3)[0] is False
    assert store.blocked("other", new_cutoffs_since_rejection=0) == (False, "")
    assert "P-0001" in store.render_register()

    payload["track"] = "vibes"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    with pytest.raises(ValueError, match="invalid"):
        store.load_all()
    path.write_text("decision: [unclosed")
    with pytest.raises(ValueError, match="invalid"):
        store.load_all()
    assert store.next_brief_id() == "B-0001"
    store.write_brief("B-0001", "# brief")
    assert store.next_brief_id() == "B-0002"


# -- seeds ---------------------------------------------------------------------------


def test_seeds_load_into_hypotheses_with_code_computed_evidence(tmp_path):
    settings = _settings(tmp_path)
    corpus = corpus_with(ten_runs("2026-08-19"), through="2026-09-30")
    stats = seed_statistics(strata_frame(corpus), [build_card(corpus, date(2026, 8, 19))])
    assert stats["zeroed_rate"] == 1.0 and stats["lgbm_below_rate"] in (0.0, 1.0)
    assert stats["novelty_split_rate"] == 1.0 and stats["p50_spread_mean"] is not None
    store = HypothesisStore(settings)
    created = seed_hypotheses(store, load_seeds(), TAX, week="2026-W37", evidence=stats)
    assert "S-A7" in created and seed_hypotheses(store, load_seeds(), TAX, week="2026-W38", evidence=stats) == []
    state = store.state()
    assert state["S-A7"].evidence == {"seed_stat": "zeroed_rate", "seed_value": 1.0}
    assert state["S-A7"].seed_source == "IMPROVEMENTS:A7" and state["S-A1"].track == "numeric"
    assert "S-A7" in store.render_active()
