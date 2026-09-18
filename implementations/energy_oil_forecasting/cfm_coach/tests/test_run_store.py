import json

import pytest
from energy_oil_forecasting.cfm_agent_v_5_0.config import DEFAULT_SETTINGS as AGENT_DEFAULTS
from energy_oil_forecasting.cfm_coach.backfill import _predictions_from_audit, model_from_predictor_id
from energy_oil_forecasting.cfm_coach.run_store import RunRecordStore, build_run_record


def _write(store: RunRecordStore, predictions: list[dict]):
    return store.write(
        predictions,
        settings=AGENT_DEFAULTS,
        calibration_version="v001",
        agent_model="gemini-3.1-flash-lite-preview",
    )


def test_record_round_trips_unchanged(settings, predictions):
    store = RunRecordStore(settings)
    written = _write(store, predictions)

    loaded = store.load_all()
    assert len(loaded) == 1
    assert loaded[0] == written


def test_record_captures_the_five_identity_fields(settings, predictions):
    """R6: none of these are persisted by the agent today, and all are needed to segment a corpus."""
    record = _write(RunRecordStore(settings), predictions)

    assert record.settings == AGENT_DEFAULTS.model_dump(mode="json")
    assert record.calibration_version == "v001"
    assert record.agent_model == "gemini-3.1-flash-lite-preview"
    assert record.package_fingerprint.startswith("cfm_v5_0_package:sha256:")
    assert record.prompt_version == "cfm_v5_0_builtin"


def test_quantile_keys_survive_json_as_floats(settings, predictions):
    """The agent's metadata is JSON, so quantile keys arrive as strings; replay needs floats."""
    record = _write(RunRecordStore(settings), predictions)
    horizon = record.horizon(5)

    assert horizon.ensemble_quantiles[0.5] == 66.0
    assert horizon.final_quantiles[0.5] == 68.0
    assert set(horizon.component_quantiles) == {"arima", "kalman", "lightgbm"}
    assert horizon.component_quantiles["arima"][0.5] == 66.0


def test_horizons_are_sorted_and_addressable(settings, predictions):
    record = _write(RunRecordStore(settings), predictions)

    assert record.horizons == [5, 21]
    assert record.horizon(21).final_point_forecast == 67.0
    with pytest.raises(KeyError):
        record.horizon(10)


def test_audit_only_signals_are_carried_for_the_trust_report(settings, predictions):
    """R9: these inform trust and never the number, but they have to be kept to inform anything."""
    record = _write(RunRecordStore(settings), predictions)

    assert record.audit_signals["audit_switch"] is True
    assert record.audit_signals["model_disagreement_std"] == {"5": 1.2}
    assert "source_validation_audit" in record.audit_signals
    assert "claim_support_findings" in record.audit_signals


def test_unknown_schema_version_is_refused_not_guessed(settings, predictions):
    store = RunRecordStore(settings)
    record = _write(store, predictions)

    path = store.path_for(record.run_id)
    payload = json.loads(path.read_text())
    payload["schema_version"] = 99
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="schema_version"):
        store.load_all()


def test_both_historical_audit_shapes_are_readable(predictions, e2e_audit):
    """The two pre-coach audit files have incompatible shapes; both must load."""
    assert _predictions_from_audit(predictions) == predictions
    assert _predictions_from_audit(e2e_audit) == predictions

    with pytest.raises(ValueError, match="unrecognised audit-file shape"):
        _predictions_from_audit({"nothing": "useful"})


def test_model_is_recoverable_from_predictor_id():
    """The model name is recorded nowhere else in a pre-R6 audit file."""
    assert (
        model_from_predictor_id("agent_predictor_cfm_agent_v_5_0_gemini-3.1-flash-lite-preview_continuous")
        == "gemini-3.1-flash-lite-preview"
    )


def test_empty_predictions_are_rejected(settings):
    with pytest.raises(ValueError, match="zero predictions"):
        build_run_record(
            [],
            settings=AGENT_DEFAULTS,
            calibration_version="v001",
            agent_model="m",
            schema_version=1,
        )
