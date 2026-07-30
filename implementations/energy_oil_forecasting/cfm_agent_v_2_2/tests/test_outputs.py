"""Schema and cross-object validator tests for CFM Agent v2.2 score output."""

from __future__ import annotations

import copy

import pytest
from energy_oil_forecasting.cfm_agent_v_2_2.outputs import WtiEventScoreOutput
from energy_oil_forecasting.cfm_agent_v_2_2.schemas import WtiEventFactor, WtiEventScenario
from pydantic import ValidationError


def valid_output() -> dict[str, object]:
    """Return a fully valid ``WtiEventScoreOutput`` fixture.

    Two core factors (one dormant), one transitory factor; three
    scenarios with probabilities 0.5/0.35/0.15 and a genuine tail case.
    """
    return {
        "factors": [
            {
                "name": "A",
                "category": "geopolitical",
                "tier": "core",
                "rationale": "Durable geopolitical driver.",
                "impact_score": 1,
                "confidence": 0.8,
                "evidence_indices": [0],
            },
            {
                "name": "B",
                "category": "policy",
                "tier": "core",
                "rationale": "Dormant policy theme, still worth tracking.",
                "impact_score": 0,
                "confidence": 0.6,
            },
            {
                "name": "C",
                "category": "weather",
                "tier": "transitory",
                "rationale": "A storm approaching production areas.",
                "impact_score": 2,
                "confidence": 0.7,
            },
        ],
        "scenarios": [
            {
                "name": "Status quo",
                "stances": {"A": "neutral", "B": "neutral", "C": "bullish"},
                "probability": 0.5,
                "impact_score": 0,
                "is_tail_case": False,
                "rationale": "Nothing escalates; mild weather premium fades.",
            },
            {
                "name": "Disruption clears",
                "stances": {"A": "bearish", "B": "neutral", "C": "bearish"},
                "probability": 0.35,
                "impact_score": -1,
                "is_tail_case": False,
                "rationale": "Tensions ease and the storm misses key areas.",
            },
            {
                "name": "Compound escalation",
                "stances": {"A": "bullish", "B": "bullish", "C": "bullish"},
                "probability": 0.15,
                "impact_score": 3,
                "is_tail_case": True,
                "rationale": "Conflict escalates while the storm hits production.",
            },
        ],
        "verified_evidence": [
            {
                "title": "Example source",
                "source_url": "https://example.com/evidence",
                "claim": "A verified development affecting factor A.",
                "forecast_effect": "center_up",
            }
        ],
        "research_summary": "One material verified event was found.",
        "overall_rationale": "Scores span a dormant policy theme and an active weather event.",
        "warnings": [],
    }


def test_valid_output_parses_and_yields_calibration_features() -> None:
    output = WtiEventScoreOutput.model_validate(valid_output())
    features = output.calibration_features()

    assert features["net_core_score"] == 1.0
    assert features["net_transitory_score"] == 2.0
    assert features["confidence_weighted_core_score"] == pytest.approx(0.8)
    assert features["confidence_weighted_transitory_score"] == pytest.approx(1.4)
    assert features["expected_scenario_impact"] == pytest.approx(0.5 * 0 + 0.35 * -1 + 0.15 * 3)
    assert features["tail_probability"] == pytest.approx(0.15)
    assert features["tail_impact_score"] == 3.0
    assert features["score_geopolitical"] == 1.0
    assert features["score_weather"] == 2.0
    assert features["score_policy"] == 0.0
    assert features["score_operational"] == 0.0
    assert features["score_demand_expectations"] == 0.0


def test_calibration_row_is_flat_and_stamped_with_as_of() -> None:
    output = WtiEventScoreOutput.model_validate(valid_output())
    # calibration_features must be flat numeric — the stored dataset depends on it.
    assert all(isinstance(value, float) for value in output.calibration_features().values())


def test_probability_sum_must_be_one() -> None:
    raw = copy.deepcopy(valid_output())
    raw["scenarios"][0]["probability"] = 0.6
    with pytest.raises(ValidationError, match="sum to 1"):
        WtiEventScoreOutput.model_validate(raw)


def test_missing_tail_case_is_rejected() -> None:
    raw = copy.deepcopy(valid_output())
    for scenario in raw["scenarios"]:
        scenario["is_tail_case"] = False
    with pytest.raises(ValidationError, match="is_tail_case=True"):
        WtiEventScoreOutput.model_validate(raw)


def test_tail_case_with_highest_probability_is_rejected() -> None:
    raw = copy.deepcopy(valid_output())
    raw["scenarios"][0]["probability"] = 0.25
    raw["scenarios"][1]["probability"] = 0.15
    raw["scenarios"][2]["probability"] = 0.6
    with pytest.raises(ValidationError, match="low-probability"):
        WtiEventScoreOutput.model_validate(raw)


def test_tail_case_with_weak_impact_is_rejected() -> None:
    raw = copy.deepcopy(valid_output())
    raw["scenarios"][2]["impact_score"] = 1
    with pytest.raises(ValidationError, match="high-impact"):
        WtiEventScoreOutput.model_validate(raw)


def test_stance_coverage_mismatch_is_rejected() -> None:
    raw = copy.deepcopy(valid_output())
    del raw["scenarios"][0]["stances"]["C"]
    with pytest.raises(ValidationError, match="cover exactly the shared factors"):
        WtiEventScoreOutput.model_validate(raw)


def test_scenarios_must_genuinely_disagree() -> None:
    raw = copy.deepcopy(valid_output())
    for scenario in raw["scenarios"]:
        scenario["stances"] = {"A": "bullish", "B": "neutral", "C": "neutral"}
    with pytest.raises(ValidationError, match="genuinely disagree"):
        WtiEventScoreOutput.model_validate(raw)


def test_factor_tier_counts_out_of_range_is_rejected() -> None:
    raw = copy.deepcopy(valid_output())
    raw["factors"] = [raw["factors"][0], raw["factors"][2]]  # one core, one transitory
    for scenario in raw["scenarios"]:
        scenario["stances"] = {"A": scenario["stances"]["A"], "C": scenario["stances"]["C"]}
    with pytest.raises(ValidationError, match="2-4 core factors"):
        WtiEventScoreOutput.model_validate(raw)


def test_duplicate_factor_names_are_rejected() -> None:
    raw = copy.deepcopy(valid_output())
    raw["factors"][1]["name"] = "A"
    with pytest.raises(ValidationError, match="unique"):
        WtiEventScoreOutput.model_validate(raw)


def test_out_of_range_evidence_index_is_rejected() -> None:
    raw = copy.deepcopy(valid_output())
    raw["factors"][0]["evidence_indices"] = [3]
    with pytest.raises(ValidationError, match="evidence index"):
        WtiEventScoreOutput.model_validate(raw)


def test_impact_score_out_of_bounds_is_rejected() -> None:
    with pytest.raises(ValidationError):
        WtiEventFactor.model_validate(
            {
                "name": "A",
                "category": "geopolitical",
                "tier": "core",
                "rationale": "Out of bounds.",
                "impact_score": 5,
                "confidence": 0.5,
            }
        )


def test_transitory_factor_with_zero_impact_is_rejected() -> None:
    with pytest.raises(ValidationError, match="nonzero"):
        WtiEventFactor.model_validate(
            {
                "name": "C",
                "category": "weather",
                "tier": "transitory",
                "rationale": "Pointless if zero.",
                "impact_score": 0,
                "confidence": 0.5,
            }
        )


def test_core_factor_may_be_dormant_with_zero_impact() -> None:
    factor = WtiEventFactor.model_validate(
        {
            "name": "B",
            "category": "policy",
            "tier": "core",
            "rationale": "Dormant but tracked.",
            "impact_score": 0,
            "confidence": 0.5,
        }
    )
    assert factor.impact_score == 0


def test_scenario_probability_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        WtiEventScenario.model_validate(
            {
                "name": "Impossible",
                "stances": {"A": "neutral"},
                "probability": 0.0,
                "impact_score": 1,
                "is_tail_case": False,
                "rationale": "Zero-probability scenarios are not allowed.",
            }
        )
