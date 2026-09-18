"""The coached forecast: bit-identical under the identity, changed only by the coach ledger, never re-issued."""

from __future__ import annotations

from datetime import date, datetime

import pytest
from conftest import make_v52_run_record

from energy_oil_forecasting.cfm_coach.review.coached import (
    COACHED_PROVENANCE,
    coach_ledger,
    coached_run_id,
    load_coached,
    run_stream,
)
from energy_oil_forecasting.cfm_coach.review.collect import COACHED, collect_stream
from energy_oil_forecasting.cfm_coach.review.settings import ReviewSettings
from energy_oil_forecasting.cfm_coach.schemas import CalibrationLayer, CalibrationVersion
from energy_oil_forecasting.cfm_coach.streams import V52_ADVANCED
from test_review_cards import V52_SETTINGS, price_service
from energy_oil_forecasting.cfm_coach.outcomes import OutcomeResolver


def test_identity_ledger_reproduces_the_agent_and_is_idempotent(tmp_path):
    settings = ReviewSettings(data_dir=tmp_path)
    records = [
        make_v52_run_record(cutoff="2026-08-19", suffix="a"),
        make_v52_run_record(cutoff="2026-08-20", suffix="b"),
    ]
    written = run_stream(V52_ADVANCED, settings, records=records, now=datetime(2026, 8, 20, 12, 0))
    assert [r.run_id for r in written] == [coached_run_id(r.run_id) for r in records]
    for source, coached in zip(records, written, strict=True):
        assert coached.provenance == COACHED_PROVENANCE
        assert coached.calibration_version == "v001"
        for h in source.horizons:
            assert coached.horizon(h).final_quantiles == source.horizon(h).final_quantiles
        assert coached.audit_signals["coached"]["source_run_id"] == source.run_id
    assert written[0].audit_signals["coached"]["issued_lag_days"] == 1
    assert run_stream(V52_ADVANCED, settings, records=records) == []
    assert set(load_coached(V52_ADVANCED, settings)) == {r.run_id for r in records}
    assert not (tmp_path.parent / "calibration_v52_advanced").exists()


def test_a_saved_version_changes_new_coached_records_the_way_the_layer_says(tmp_path):
    settings = ReviewSettings(data_dir=tmp_path)
    first = [make_v52_run_record(cutoff="2026-08-19", suffix="a")]
    run_stream(V52_ADVANCED, settings, records=first)
    ledger = coach_ledger(V52_ADVANCED, settings)
    layer = CalibrationLayer(width_scale={5: 1.5, 10: 1.5, 21: 1.5}, rw_anchor={5: 0.5, 10: 0.5, 21: 0.5})
    ledger.save(
        CalibrationVersion(version="v002", effective_from=date(2026, 8, 20), parent_version="v001", layer=layer)
    )

    later = make_v52_run_record(cutoff="2026-08-21", suffix="c")
    (coached,) = run_stream(V52_ADVANCED, settings, records=[*first, later])
    assert coached.calibration_version == "v002"
    last = float(later.diagnostics["latest_value"])
    for h in later.horizons:
        src = later.horizon(h)
        # Exactly what `CoachedCfmPredictor` would have applied live: the layer on the engine's output.
        point, quantiles = layer.apply(
            ensemble_p50=src.ensemble_quantiles[0.5],
            final_point_forecast=src.final_point_forecast,
            final_quantiles=dict(src.final_quantiles),
            horizon=h,
            last_value=last,
        )
        assert coached.horizon(h).final_point_forecast == pytest.approx(point)
        assert coached.horizon(h).final_quantiles == pytest.approx(quantiles)
        assert coached.horizon(h).final_quantiles[0.9] - coached.horizon(h).final_quantiles[0.1] == pytest.approx(
            1.5 * (src.final_quantiles[0.9] - src.final_quantiles[0.1])
        )
        assert coached.horizon(h).forecast_transformation["coach_layer"]["rw_anchor"] == 0.5
    # The record coached earlier under v001 is not re-issued.
    assert load_coached(V52_ADVANCED, settings)[first[0].run_id].calibration_version == "v001"


def test_collect_scores_the_coached_variant_beside_the_agent(tmp_path):
    settings = ReviewSettings(data_dir=tmp_path)
    records = [make_v52_run_record(cutoff="2026-08-19", suffix="a")]
    run_stream(V52_ADVANCED, settings, records=records)
    resolver = OutcomeResolver(V52_SETTINGS, service=price_service("2026-09-04"))
    corpus = collect_stream(V52_ADVANCED, records=records, resolver=resolver, review_settings=settings)
    coached = corpus.score(records[0].run_id, 5, COACHED)
    agent = corpus.score(records[0].run_id, 5, "agent")
    assert coached is not None and coached.pinball == pytest.approx(agent.pinball)
    assert corpus.score(records[0].run_id, 21, COACHED) is None  # unresolved
