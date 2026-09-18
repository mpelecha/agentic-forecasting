"""Triage: one call per card, annotations per key, and code that drops what it cannot verify."""

from __future__ import annotations

import asyncio
import json
from datetime import date

import pytest
from conftest import make_v52_run_record

from energy_oil_forecasting.cfm_coach.review.cards import build_card
from energy_oil_forecasting.cfm_coach.review.collect import readiness
from energy_oil_forecasting.cfm_coach.review.llm import FakeCall, FakeClient
from energy_oil_forecasting.cfm_coach.review.memory import AnnotationStore, Taxonomy
from energy_oil_forecasting.cfm_coach.review.settings import ReviewSettings
from energy_oil_forecasting.cfm_coach.review.state import ReviewKey, ReviewState
from energy_oil_forecasting.cfm_coach.review.triage import TriageOut, build_messages, run_triage, validate
from test_review_cards import corpus_with, ten_runs


TAX = Taxonomy.load()
WEEK = "2026-W37"


def _setup(tmp_path, through="2026-09-11"):
    settings = ReviewSettings(data_dir=tmp_path)
    runs = ten_runs("2026-08-19") + [make_v52_run_record(cutoff="2026-08-26", suffix="z", price_shift=1.5)]
    corpus = corpus_with(runs, through=through)
    state = ReviewState(settings.state_path)
    ready = readiness(corpus, state, settings)
    cards = {c: build_card(corpus, c, settings) for c in ready.cutoffs_at(1)}
    return settings, corpus, state, cards


def _good(card_cutoff: str):
    return json.dumps(
        {
            "tags": [
                {
                    "code": "POLICY.ZEROED_ON_CITATION",
                    "severity": 3,
                    "confidence": 0.9,
                    "pointer": "h5.granted_center",
                    "outcome_dependent": False,
                    "horizon": 5,
                },
                {
                    "code": "BASE.LGBM_LEVEL_BIAS",
                    "severity": 2,
                    "confidence": 0.8,
                    "pointer": "base.h5.components.lightgbm.bias",
                    "outcome_dependent": False,
                    "value": -1.0,
                },
                {
                    "code": "JUDGMENT.DIRECTION_WRONG",
                    "severity": 2,
                    "confidence": 0.6,
                    "pointer": "claim_001",
                    "outcome_dependent": True,
                    "horizon": 10,
                },
                {
                    "code": "NOT.A.CODE",
                    "severity": 1,
                    "confidence": 0.5,
                    "pointer": "claim_001",
                    "outcome_dependent": False,
                },
                {
                    "code": "JUDGMENT.STALE_CLAIM",
                    "severity": 1,
                    "confidence": 0.5,
                    "pointer": "claim_999",
                    "outcome_dependent": False,
                },
                {
                    "code": "BASE.ENSEMBLE_LOSES_TO_RW",
                    "severity": 1,
                    "confidence": 0.5,
                    "pointer": "base.h5.decomposition.base_term",
                    "outcome_dependent": True,
                    "value": -40.0,
                },
                {
                    "code": "EVIDENCE.QUERY_TOO_GENERIC",
                    "severity": 1,
                    "confidence": 0.5,
                    "pointer": "",
                    "outcome_dependent": False,
                },
            ],
            "went_right": ["RIGHT.HELD_NEUTRAL_CORRECTLY", "NOPE"],
            "new_pattern_notes": ["accepted summaries with empty text"],
            "knowability": "undetermined",
            "summary": f"card {card_cutoff}",
        }
    )


def test_end_to_end_with_the_fake_client_writes_one_annotation_per_key(tmp_path):
    settings, corpus, state, cards = _setup(tmp_path)
    assert set(cards) == {date(2026, 8, 19), date(2026, 8, 26)}
    fake = FakeClient(settings=settings, responder=lambda call: _good("x"))
    store = AnnotationStore(settings)
    results = asyncio.run(
        run_triage(fake, corpus=corpus, cards=cards, state=state, store=store, taxonomy=TAX, week=WEEK)
    )
    assert len(fake.calls) == 2 and all(r.failure is None for r in results)
    current = store.current("v52_advanced")
    assert set(current) == {
        ReviewKey("v52_advanced", d, h) for d in (date(2026, 8, 19), date(2026, 8, 26)) for h in (5, 10)
    }
    h5 = current[ReviewKey("v52_advanced", date(2026, 8, 19), 5)]
    h10 = current[ReviewKey("v52_advanced", date(2026, 8, 19), 10)]
    assert {t.code for t in h5.tags} == {"POLICY.ZEROED_ON_CITATION", "BASE.LGBM_LEVEL_BIAS"}
    assert {t.code for t in h10.tags} == {"BASE.LGBM_LEVEL_BIAS", "JUDGMENT.DIRECTION_WRONG"}
    assert h5.went_right == ["RIGHT.HELD_NEUTRAL_CORRECTLY"] and h5.knowability == "undetermined"
    whys = [d["why"] for d in h5.dropped]
    assert any("unknown code" in w for w in whys) and any("claim_id" in w for w in whys)
    assert any("does not match cell" in w for w in whys) and any("query_index" in w for w in whys)
    assert all(state.get(k).status == "triaged" for k in current)
    # Idempotent: nothing left in `resolved`, so a second pass makes no call.
    again = asyncio.run(run_triage(fake, corpus=corpus, cards=cards, state=state, store=store, taxonomy=TAX, week=WEEK))
    assert again == [] and len(fake.calls) == 2


def test_messages_are_stable_then_volatile(tmp_path):
    settings, corpus, state, cards = _setup(tmp_path)
    card = cards[date(2026, 8, 19)]
    messages, phash = build_messages(card, TAX, horizons=(5, 10))
    assert messages[0]["role"] == "system" and "Taxonomy v" in messages[0]["content"]
    assert "Newly resolved horizons to tag: [5, 10]" in messages[1]["content"]
    assert "<untrusted claim>" in messages[1]["content"]
    other, phash2 = build_messages(cards[date(2026, 8, 26)], TAX, horizons=(5, 10))
    assert phash == phash2 and other[0]["content"] == messages[0]["content"]


def test_an_injected_instruction_cannot_create_a_tag_without_a_valid_pointer(tmp_path):
    settings, corpus, state, cards = _setup(tmp_path)
    obedient = json.dumps(
        {
            "tags": [
                {
                    "code": "RIGHT.DIRECTION_AND_SIZE",
                    "severity": 3,
                    "confidence": 1.0,
                    "pointer": "as instructed",
                    "outcome_dependent": True,
                }
            ],
            "went_right": [],
            "new_pattern_notes": [],
            "knowability": None,
            "summary": "",
        }
    )
    fake = FakeClient(settings=settings, responder=lambda call: obedient)
    store = AnnotationStore(settings)
    asyncio.run(run_triage(fake, corpus=corpus, cards=cards, state=state, store=store, taxonomy=TAX, week=WEEK))
    for ann in store.current("v52_advanced").values():
        assert ann.tags == [] and ann.dropped[0]["why"].startswith("pointer not on card")


def test_schema_invalid_gets_one_correction_then_a_recorded_failure(tmp_path):
    settings, corpus, state, cards = _setup(tmp_path)
    one_card = {date(2026, 8, 19): cards[date(2026, 8, 19)]}
    fake = FakeClient(settings=settings, script={"triage": ["not json", _good("x")]})
    store = AnnotationStore(settings)
    (result,) = asyncio.run(
        run_triage(fake, corpus=corpus, cards=one_card, state=state, store=store, taxonomy=TAX, week=WEEK)
    )
    assert result.failure is None and len(fake.calls) == 2
    assert "did not match the schema" in fake.calls[1].messages[-1]["content"]

    other = {date(2026, 8, 26): cards[date(2026, 8, 26)]}
    fake = FakeClient(settings=settings, script={"triage": ["{}", '{"tags": "no"}']})
    (result,) = asyncio.run(
        run_triage(fake, corpus=corpus, cards=other, state=state, store=store, taxonomy=TAX, week=WEEK)
    )
    assert result.failure and "schema-invalid" in result.failure
    assert state.get(ReviewKey("v52_advanced", date(2026, 8, 26), 5)).status == "resolved"
    assert ReviewKey("v52_advanced", date(2026, 8, 26), 5) not in store.current("v52_advanced")


def test_later_horizon_sees_earlier_annotations_as_read_only_context(tmp_path):
    settings, corpus, state, cards = _setup(tmp_path)
    store = AnnotationStore(settings)
    fake = FakeClient(settings=settings, responder=lambda call: _good("x"))
    asyncio.run(run_triage(fake, corpus=corpus, cards=cards, state=state, store=store, taxonomy=TAX, week=WEEK))
    # h21 lands.
    from test_review_cards import corpus_with as cw  # noqa: PLC0415
    from energy_oil_forecasting.cfm_coach.review.collect import readiness as rd  # noqa: PLC0415

    later = cw(corpus.records, through="2026-09-30")
    ready = rd(later, state, settings)
    assert [k.horizon for k in ready.new_keys] == [21, 21]
    cards2 = {c: build_card(later, c, settings) for c in ready.cutoffs_at(1)}
    calls_before = len(fake.calls)
    asyncio.run(run_triage(fake, corpus=later, cards=cards2, state=state, store=store, taxonomy=TAX, week="2026-W38"))
    assert len(fake.calls) == calls_before + 2
    prompt = fake.calls[-1].messages[1]["content"]
    assert "Read-only: earlier annotations" in prompt and "POLICY.ZEROED_ON_CITATION@h5.granted_center" in prompt
    assert state.get(ReviewKey("v52_advanced", date(2026, 8, 19), 21)).status == "triaged"


def test_validate_resolves_aliases_and_checks_table_values(tmp_path):
    settings, corpus, state, cards = _setup(tmp_path)
    card = cards[date(2026, 8, 19)]
    tax = Taxonomy(version=2, codes=TAX.codes, aliases={"OLD.ZEROED": "POLICY.ZEROED_ON_CITATION"})
    out = TriageOut.model_validate(
        {
            "tags": [
                {
                    "code": "OLD.ZEROED",
                    "severity": 2,
                    "confidence": 0.7,
                    "pointer": "h10.tier",
                    "outcome_dependent": False,
                },
                {
                    "code": "BASE.LGBM_LEVEL_BIAS",
                    "severity": 2,
                    "confidence": 0.7,
                    "pointer": "base.h5.lgbm_below_last_close",
                    "outcome_dependent": False,
                },
                {
                    "code": "BASE.LGBM_LEVEL_BIAS",
                    "severity": 2,
                    "confidence": 0.7,
                    "pointer": "base.h99.lightgbm.bias",
                    "outcome_dependent": False,
                },
            ]
        }
    )
    checked = validate(out, card, tax)
    assert [t.code for t in checked.tags] == ["POLICY.ZEROED_ON_CITATION", "BASE.LGBM_LEVEL_BIAS"]
    assert len(checked.dropped) == 1


def test_component_aliases_and_percent_shares_resolve(tmp_path):
    from energy_oil_forecasting.cfm_coach.review.triage import resolve_pointer, value_matches  # noqa: PLC0415

    settings, corpus, state, cards = _setup(tmp_path)
    card = next(iter(cards.values()))
    ok, cell = resolve_pointer(card, "base.h5.lightgbm.bias", "table_cell")
    ok_full, cell_full = resolve_pointer(card, "base.h5.components.lightgbm.bias", "table_cell")
    assert ok and ok_full and cell == cell_full and isinstance(cell, float)
    assert resolve_pointer(card, "base.h5.nothing.bias", "table_cell") == (False, None)
    sid = card.representative.summaries[0].summary_id
    assert resolve_pointer(card, f"summary_{sid}", "summary_id")[0] and resolve_pointer(card, sid, "summary_id")[0]
    # shares render as percentages; a fraction cell accepts the percentage reading
    assert value_matches(85.0, 0.8518) and value_matches(0.85, 0.8518) and value_matches(60.0, 0.6003)
    assert not value_matches(1.02, 0.236) and not value_matches(-4.45, 0.83) and not value_matches(150.0, 0.6)


def test_revalidate_week_rebuilds_tags_from_raw_without_changing_ids(tmp_path):
    from energy_oil_forecasting.cfm_coach.review.cards import CardStore  # noqa: PLC0415
    from energy_oil_forecasting.cfm_coach.review.triage import revalidate_week  # noqa: PLC0415

    settings, corpus, state, cards = _setup(tmp_path)
    for cutoff, card in cards.items():
        CardStore(settings).save(card)
    fake = FakeClient(settings=settings, responder=lambda call: _good("x"))
    store = AnnotationStore(settings)
    results = asyncio.run(
        run_triage(fake, corpus=corpus, cards=cards, state=state, store=store, taxonomy=TAX, week=WEEK)
    )
    raw_dir = settings.weeks_dir / WEEK / "raw" / "v52_advanced"
    raw_dir.mkdir(parents=True)
    for r in results:
        payload = json.loads(r.raw["content"])
        payload["tags"].append(
            {
                "code": "BASE.LGBM_LEVEL_BIAS",
                "severity": 1,
                "confidence": 0.5,
                "pointer": "base.h5.lightgbm.bias",
                "outcome_dependent": False,
                "value": None,
                "horizon": 5,
            }
        )
        (raw_dir / f"triage_{r.card.cutoff}.json").write_text(json.dumps({**r.raw, "content": json.dumps(payload)}))
    before = {a.annotation_id: len(a.tags) for a in store.load("v52_advanced")}
    out = revalidate_week(settings, WEEK, taxonomy=TAX)
    after = {a.annotation_id: len(a.tags) for a in store.load("v52_advanced")}
    assert set(before) == set(after) and out["v52_advanced"][1] > out["v52_advanced"][0]
    assert all(after[i] >= before[i] for i in before) and any(after[i] == before[i] + 1 for i in before)
