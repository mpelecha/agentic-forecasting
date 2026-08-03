"""Scenario contract, calibration-feature, and uncertainty-floor tests."""

import copy

import pytest
from energy_oil_forecasting.cfm_agent_v_3_4.config import CfmV34Settings
from energy_oil_forecasting.cfm_agent_v_3_4.outputs import CfmContextAssessmentOutput
from energy_oil_forecasting.cfm_agent_v_3_4.policy import ConstrainedActionPolicy
from energy_oil_forecasting.cfm_agent_v_3_4.sanitizer import sanitize_assessment
from energy_oil_forecasting.cfm_agent_v_3_4.schemas import ModelHorizonForecast
from pydantic import ValidationError


def ensemble() -> ModelHorizonForecast:
    return ModelHorizonForecast(
        horizon=5,
        forecast_date="2026-03-09",
        point_forecast=67.0,
        quantiles={0.05: 60.0, 0.1: 62.0, 0.5: 67.0, 0.9: 72.0, 0.95: 74.0},
    )


def base_assessment() -> dict:
    """Two eligible publishers, one direct weather claim, a two-sided scenario set."""
    return {
        "reported_search_queries": ["q1", "q2", "q3", "q4"],
        "evidence_sources": [
            {
                "source_id": f"s{i}",
                "title": f"Source {i}",
                "source_url": f"https://publisher{i}.example/story",
                "publisher": f"Publisher {i}",
                "source_tier": "tier_1_primary" if i == 0 else "tier_2_independent",
                "publication_date": "2026-03-01",
                "is_primary_or_official": i == 0,
                "provenance_status": "verified_from_tool",
                "verifier_content_status": "accepted_factual_content",
                "verifier_processing_status": "accepted_clean",
                "verified_evidence_excerpt": "Accepted factual evidence.",
            }
            for i in range(2)
        ],
        "evidence_claims": [
            {
                "claim_id": "c1",
                "statement": "A storm shut in offshore production.",
                "claim_type": "physical_supply",
                "driver_category": "weather",
                "support_status": "direct",
                "supporting_source_ids": ["s0", "s1"],
                "contradicting_source_ids": [],
            },
            {
                "claim_id": "c2",
                "statement": "A stimulus announcement lifted demand expectations.",
                "claim_type": "demand",
                "driver_category": "demand_expectations",
                "support_status": "direct",
                "supporting_source_ids": ["s1"],
                "contradicting_source_ids": [],
            },
        ],
        "physical_status": "partial_disruption",
        "incremental_novelty": "likely_new_relative_to_model_data",
        "material_evidence_conflict": False,
        "confidence": 0.70,
        "scenarios": [
            {
                "scenario_id": "sc1",
                "name": "Disruption persists",
                "probability": 0.55,
                "direction": "up",
                "magnitude": "moderate",
                "is_tail_case": False,
                "cited_claim_ids": ["c1"],
                "rationale": "Shut-ins continue and the supply loss is felt.",
            },
            {
                "scenario_id": "sc2",
                "name": "Rapid restart",
                "probability": 0.30,
                "direction": "down",
                "magnitude": "small",
                "is_tail_case": False,
                "cited_claim_ids": ["c1"],
                "rationale": "Facilities restart quickly and the premium unwinds.",
            },
            {
                "scenario_id": "sc3",
                "name": "Compound infrastructure damage",
                "probability": 0.15,
                "direction": "up",
                "magnitude": "large",
                "is_tail_case": True,
                "cited_claim_ids": ["c1", "c2"],
                "rationale": "Damage proves structural and outages extend for months.",
            },
        ],
        "horizon_actions": [
            {
                "horizon": 5,
                "center_action": "small_up",
                "uncertainty_action": "unchanged",
                "persistence_profile": "temporary",
                "cited_claim_ids": ["c1"],
                "rationale": "Bounded upward pressure from a confirmed shut-in.",
            }
        ],
        "research_summary": "Storm-driven shut-ins with a credible fast-restart path.",
        "overall_rationale": "Two-sided situation with an upward central case.",
    }


def test_valid_scenario_set_parses_and_reports_disagreement() -> None:
    output = CfmContextAssessmentOutput.model_validate(base_assessment())
    # min(P(up)=0.70, P(down)=0.30) = 0.30
    assert output.scenario_disagreement_mass() == pytest.approx(0.30)


def test_absent_scenario_set_is_tolerated() -> None:
    raw = base_assessment()
    raw["scenarios"] = []
    output = CfmContextAssessmentOutput.model_validate(raw)
    assert output.scenario_disagreement_mass() == 0.0


def test_probabilities_must_sum_to_one() -> None:
    raw = base_assessment()
    raw["scenarios"][0]["probability"] = 0.90
    with pytest.raises(ValidationError, match="sum to 1"):
        CfmContextAssessmentOutput.model_validate(raw)


def test_missing_tail_case_is_rejected() -> None:
    raw = base_assessment()
    raw["scenarios"][2]["is_tail_case"] = False
    with pytest.raises(ValidationError, match="is_tail_case=true"):
        CfmContextAssessmentOutput.model_validate(raw)


def test_tail_case_may_not_be_the_most_likely_scenario() -> None:
    raw = base_assessment()
    raw["scenarios"][0]["probability"] = 0.15
    raw["scenarios"][2]["probability"] = 0.55
    with pytest.raises(ValidationError, match="low-probability"):
        CfmContextAssessmentOutput.model_validate(raw)


def test_tail_case_must_be_high_impact() -> None:
    raw = base_assessment()
    raw["scenarios"][2]["magnitude"] = "small"
    with pytest.raises(ValidationError, match="high-impact"):
        CfmContextAssessmentOutput.model_validate(raw)


def test_duplicate_direction_and_magnitude_is_rejected() -> None:
    raw = base_assessment()
    raw["scenarios"][2]["magnitude"] = "moderate"  # now identical to sc1: up/moderate
    with pytest.raises(ValidationError, match="written twice"):
        CfmContextAssessmentOutput.model_validate(raw)


def test_scenario_may_not_cite_unknown_claim() -> None:
    raw = base_assessment()
    raw["scenarios"][0]["cited_claim_ids"] = ["c_missing"]
    with pytest.raises(ValidationError, match="unknown claim_id"):
        CfmContextAssessmentOutput.model_validate(raw)


def test_neutral_scenario_must_be_small() -> None:
    raw = base_assessment()
    raw["scenarios"][1]["direction"] = "neutral"
    raw["scenarios"][1]["magnitude"] = "moderate"
    with pytest.raises(ValidationError, match="neutral scenario must have small magnitude"):
        CfmContextAssessmentOutput.model_validate(raw)


def test_calibration_features_are_fixed_width_and_numeric() -> None:
    output = CfmContextAssessmentOutput.model_validate(base_assessment())
    features = output.calibration_features()

    assert all(isinstance(value, float) for value in features.values())
    # Every driver category has a slot, whether or not it was used this run.
    for category in ("geopolitical", "weather", "operational", "policy", "demand_expectations"):
        assert f"claims_{category}" in features
    assert features["claims_weather"] == 1.0
    assert features["claims_demand_expectations"] == 1.0
    assert features["claims_geopolitical"] == 0.0
    assert features["tail_probability"] == pytest.approx(0.15)
    assert features["scenario_disagreement_mass"] == pytest.approx(0.30)
    # 0.55*(+2) + 0.30*(-1) + 0.15*(+3) = 1.25
    assert features["expected_scenario_impact"] == pytest.approx(1.25)
    assert features["center_action_h5"] == 1.0
    assert features["uncertainty_action_h5"] == 0.0


def test_calibration_feature_keys_are_identical_across_different_content() -> None:
    """The whole point: different findings, same columns."""
    first = CfmContextAssessmentOutput.model_validate(base_assessment())

    raw = base_assessment()
    raw["scenarios"] = []
    raw["evidence_claims"][0]["driver_category"] = "geopolitical"
    raw["evidence_claims"][1]["driver_category"] = "policy"
    second = CfmContextAssessmentOutput.model_validate(raw)

    assert set(first.calibration_features()) == set(second.calibration_features())


def test_scenario_disagreement_floors_uncertainty_widening() -> None:
    """A two-sided scenario set overrides an 'unchanged' uncertainty proposal."""
    settings = CfmV34Settings()
    output = CfmContextAssessmentOutput.model_validate(base_assessment())
    decision = ConstrainedActionPolicy(settings).apply(
        ensemble=ensemble(),
        assessment=output,
        horizon=5,
        horizons=[5],
        latest_price=67.0,
        cutoff_date="2026-03-02",
    )

    assert decision.scenario_disagreement_mass == pytest.approx(0.30)
    assert decision.scenario_uncertainty_floor_applied is True
    assert decision.uncertainty_action == "moderately_wider"
    assert decision.uncertainty_multiplier == pytest.approx(settings.moderately_wider_multiplier)
    # Widening only — the floor must never move the center.
    assert decision.final_point_forecast == pytest.approx(
        ensemble().point_forecast + decision.applied_center_adjustment
    )


def test_one_sided_scenario_set_applies_no_floor() -> None:
    raw = base_assessment()
    raw["scenarios"][1]["direction"] = "neutral"  # removes all downside mass
    raw["scenarios"][1]["magnitude"] = "small"
    output = CfmContextAssessmentOutput.model_validate(raw)
    decision = ConstrainedActionPolicy(CfmV34Settings()).apply(
        ensemble=ensemble(),
        assessment=output,
        horizon=5,
        horizons=[5],
        latest_price=67.0,
        cutoff_date="2026-03-02",
    )

    assert decision.scenario_disagreement_mass == 0.0
    assert decision.scenario_uncertainty_floor_applied is False
    assert decision.uncertainty_action == "unchanged"


def test_floor_can_be_disabled_by_settings() -> None:
    settings = CfmV34Settings(enforce_scenario_uncertainty_floor=False)
    output = CfmContextAssessmentOutput.model_validate(base_assessment())
    decision = ConstrainedActionPolicy(settings).apply(
        ensemble=ensemble(),
        assessment=output,
        horizon=5,
        horizons=[5],
        latest_price=67.0,
        cutoff_date="2026-03-02",
    )

    assert decision.scenario_uncertainty_floor_applied is False
    assert decision.uncertainty_multiplier == pytest.approx(1.0)


def test_scenarios_survive_sanitization_round_trip() -> None:
    """The sanitizer dumps and revalidates; scenarios must not be lost."""
    output = CfmContextAssessmentOutput.model_validate(base_assessment())
    sanitized, _audit = sanitize_assessment(
        output, cutoff_date="2026-03-02", settings=CfmV34Settings()
    )
    assert len(sanitized.scenarios) == 3
    assert sanitized.scenario_disagreement_mass() == pytest.approx(0.30)


def test_driver_category_defaults_do_not_break_legacy_claims() -> None:
    """A claim without driver_category still parses, defaulting to geopolitical."""
    raw = copy.deepcopy(base_assessment())
    del raw["evidence_claims"][0]["driver_category"]
    output = CfmContextAssessmentOutput.model_validate(raw)
    assert output.evidence_claims[0].driver_category == "geopolitical"
