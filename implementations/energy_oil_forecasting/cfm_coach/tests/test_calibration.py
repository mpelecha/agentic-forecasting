from datetime import date

import pytest
from energy_oil_forecasting.cfm_agent_v_5_0.config import DEFAULT_SETTINGS as AGENT_DEFAULTS
from energy_oil_forecasting.cfm_coach.config import DEFAULT_SETTINGS
from energy_oil_forecasting.cfm_coach.ledger import CalibrationLedger
from energy_oil_forecasting.cfm_coach.schemas import CalibrationLayer, CalibrationVersion


QUANTILES = {0.1: 64.0, 0.5: 68.0, 0.9: 72.0}


def test_shipped_v001_is_a_true_no_op():
    """The whole learning curve is measured against v001, so it must change nothing."""
    version = CalibrationLedger(DEFAULT_SETTINGS).load("v001")

    assert version.is_baseline
    assert version.settings_overlay == {}
    assert version.layer.is_identity
    assert CalibrationLedger.to_agent_settings(version, base=AGENT_DEFAULTS) == AGENT_DEFAULTS


def test_identity_layer_reproduces_the_agent_exactly():
    point, quantiles = CalibrationLayer().apply(
        ensemble_p50=66.0,
        final_point_forecast=68.0,
        final_quantiles=QUANTILES,
        horizon=21,
    )

    assert point == 68.0
    assert quantiles == QUANTILES


def test_zero_centre_gain_discards_the_llm_overlay():
    """centre_gain=0 answers 'does the overlay help at all?' by removing it."""
    point, quantiles = CalibrationLayer(centre_gain={21: 0.0}).apply(
        ensemble_p50=66.0,
        final_point_forecast=68.0,
        final_quantiles=QUANTILES,
        horizon=21,
    )

    assert point == 66.0
    assert quantiles[0.5] == 66.0
    # Width is untouched when only the centre moves.
    assert quantiles[0.9] - quantiles[0.1] == pytest.approx(8.0)


def test_width_scale_widens_around_the_new_centre():
    _, quantiles = CalibrationLayer(width_scale={21: 2.0}).apply(
        ensemble_p50=66.0,
        final_point_forecast=68.0,
        final_quantiles=QUANTILES,
        horizon=21,
    )

    assert quantiles[0.9] - quantiles[0.1] == pytest.approx(16.0)
    assert quantiles[0.5] == 68.0


def test_layer_applies_per_horizon():
    layer = CalibrationLayer(width_scale={5: 3.0})
    _, untouched = layer.apply(ensemble_p50=66.0, final_point_forecast=68.0, final_quantiles=QUANTILES, horizon=21)

    assert untouched == QUANTILES


def test_crossing_quantiles_fail_loudly():
    """A negative scale inverts the distribution; publishing that would be worse than crashing."""
    with pytest.raises(ValueError, match="crossing quantiles"):
        CalibrationLayer(width_scale={21: -1.0}).apply(
            ensemble_p50=66.0,
            final_point_forecast=68.0,
            final_quantiles=QUANTILES,
            horizon=21,
        )


def test_current_resolves_by_date_not_by_newest(settings):
    ledger = CalibrationLedger(settings)
    ledger.save(CalibrationVersion(version="v001", effective_from=date(2026, 1, 1)))
    ledger.save(CalibrationVersion(version="v002", effective_from=date(2026, 6, 1)))

    assert ledger.current(date(2026, 3, 1)).version == "v001"
    assert ledger.current(date(2026, 6, 1)).version == "v002"
    with pytest.raises(ValueError, match="no calibration version is effective"):
        ledger.current(date(2025, 12, 31))


def test_versions_are_immutable_once_written(settings):
    ledger = CalibrationLedger(settings)
    ledger.save(CalibrationVersion(version="v001", effective_from=date(2026, 1, 1)))

    with pytest.raises(FileExistsError):
        ledger.save(CalibrationVersion(version="v001", effective_from=date(2026, 2, 1)))


def test_overlay_with_an_unknown_field_is_rejected(settings):
    """CfmV50Settings forbids extras, so a typo'd tunable fails instead of being dropped."""
    version = CalibrationVersion(
        version="v002",
        effective_from=date(2026, 6, 1),
        settings_overlay={"not_a_real_constant": 1.0},
    )

    with pytest.raises(ValueError, match="not_a_real_constant"):
        CalibrationLedger.to_agent_settings(version, base=AGENT_DEFAULTS)


def test_overlay_changes_only_the_named_constant(settings):
    version = CalibrationVersion(
        version="v002",
        effective_from=date(2026, 6, 1),
        settings_overlay={"large_action_width_fraction": 0.45},
    )
    tuned = CalibrationLedger.to_agent_settings(version, base=AGENT_DEFAULTS)

    assert tuned.large_action_width_fraction == 0.45
    assert tuned.small_action_width_fraction == AGENT_DEFAULTS.small_action_width_fraction
    assert tuned.tier_3_min_confidence == AGENT_DEFAULTS.tier_3_min_confidence
