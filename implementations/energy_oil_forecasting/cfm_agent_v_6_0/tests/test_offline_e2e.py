"""Offline end-to-end run of the public predictor with stubbed LLM, search, verifier, and sandbox.

The initial agent turn is replaced by a function that executes the two one-shot
tools exactly as the LLM would and returns a canned structured assessment, so the
whole Python path (task binding, suite, research, policy, transformation, audit)
runs on the synthetic data service without any network access.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime

import pytest
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES
from energy_oil_forecasting.cfm_agent_v_6_0.agent import (
    build_cfm_agent_config,
    build_cfm_agent_predictor,
)
from energy_oil_forecasting.cfm_agent_v_6_0.config import AGENT_NAME, CfmV60Settings
from energy_oil_forecasting.cfm_agent_v_6_0.response_control import (
    StructuredAssessmentController,
)
from energy_oil_forecasting.cfm_agent_v_6_0.schemas import ActiveSource
from energy_oil_forecasting.data import WTI_SERIES_ID


CUTOFF = "2026-09-14"
HORIZONS = [5, 10, 21]
V52_AUDIT_KEYS = (
    "model_suite_id",
    "numerical_suite_audit",
    "research_execution_audit",
    "code_execution_audit",
    "llm_context_assessment",
    "structured_output_control_audit",
    "active_research_packet",
    "cutoff_verification_audit",
    "source_validation_audit",
    "claim_support_audit",
    "claim_support_findings",
    "policy_decision",
    "forecast_transformation",
    "unadjusted_ensemble",
    "market_diagnostics",
    "task_binding",
    "audit_switch",
    "audit_only_controls",
    "active_evidence_declaration",
    "audit_findings_changed_forecast",
    "llm_authored_final_numbers",
)
V60_AUDIT_KEYS = (
    "resolved_settings",
    "floors",
    "history_window",
    "data_fingerprint",
    "data_snapshot",
    "component_seeds",
    "component_dispositions",
    "ensemble_guard_applied",
    "historical_transformation_inputs",
    "llm_effective_parameters",
    "llm_preflight",
    "code_execution_requests",
)
PUBLISHERS = {
    0: ("https://www.eia.gov/a", "eia.gov", "U.S. Energy Information Administration", "tier_1_primary", True),
    1: ("https://www.reuters.com/a", "reuters.com", "Reuters", "tier_2_independent", False),
}


class Verdict:
    clean = True
    confidence = 10
    flagged_claims: list[str] = []
    filtered_text = (
        "Exports from the Gulf terminal were suspended on 2026-09-10 after a fire at the loading berth. "
        "Loadings resumed at half capacity two days later, according to the operator."
    )


def _stub_research(monkeypatch, tool):
    async def fake_search(_):
        return "Original content", [{"url": "https://redirect.test/a", "title": "Provider"}]

    async def fake_verify(**_):
        return Verdict()

    def fake_resolve(source_id, rank, grounding_url, provider_title):
        ordinal = int(source_id.rsplit("_", 1)[-1])
        url, domain, publisher, quality, primary = PUBLISHERS[ordinal % len(PUBLISHERS)]
        return ActiveSource(
            source_id=source_id,
            rank=rank,
            grounding_url=grounding_url,
            provider_title=provider_title,
            resolved_url=url,
            resolved_domain=domain,
            publisher=publisher,
            source_quality=quality,
            is_primary_or_official=primary,
            resolution_status="resolved",
        )

    monkeypatch.setattr(tool, "_do_search", fake_search)
    monkeypatch.setattr(tool, "_verify_no_leakage", fake_verify)
    monkeypatch.setattr(tool, "_resolve_source", fake_resolve)


def _assessment(packet) -> dict:
    accepted = [summary for summary in packet.verified_summaries if summary.status == "accepted"]
    assert accepted, "the stubbed pipeline must accept summaries"
    claims = []
    for index, summary in enumerate(accepted[:2], 1):
        claims.append(
            {
                "claim_id": f"claim_{index:03d}",
                "statement": f"Loadings at the terminal were disrupted ({index}).",
                "claim_type": "physical_supply",
                "supporting_summary_ids": [summary.verified_summary_id],
                "supporting_passage_ids": [summary.passages[0].passage_id],
                "supporting_source_ids": list(summary.associated_source_ids[:1]),
                "contradicting_summary_ids": [],
                "contradicting_passage_ids": [],
                "contradicting_source_ids": [],
                "material_to_forecast": True,
            }
        )
    cited = [claim["claim_id"] for claim in claims]
    return {
        "research_packet_id": packet.packet_id,
        "evidence_claims": claims,
        "physical_status": "partial_disruption",
        "incremental_novelty": "likely_new_relative_to_model_data",
        "material_evidence_conflict": False,
        "confidence": 0.7,
        "horizon_actions": [
            {
                "horizon": 5,
                "center_action": "small_up",
                "uncertainty_action": "small_wider",
                "persistence_profile": "temporary",
                "cited_claim_ids": cited,
                "narrowing_uncertainty_resolved": "",
                "rationale": "Terminal outage supports a small upward step and wider near-term uncertainty.",
            },
            {
                "horizon": 10,
                "center_action": "moderate_up",
                "uncertainty_action": "unchanged",
                "persistence_profile": "decaying",
                "cited_claim_ids": cited,
                "narrowing_uncertainty_resolved": "",
                "rationale": "Partial resumption limits the persistence of the disruption.",
            },
            {
                "horizon": 21,
                "center_action": "no_change",
                "uncertainty_action": "unchanged",
                "persistence_profile": "temporary",
                "cited_claim_ids": [],
                "narrowing_uncertainty_resolved": "",
                "rationale": "No evidence bears on the one-month horizon.",
            },
        ],
        "research_summary": "A terminal outage suspended exports briefly; loadings have partly resumed.",
        "overall_rationale": "Categorical proposals only; Python owns the arithmetic.",
        "warnings": [],
    }


def _run_tools(predictor) -> None:
    predictor.market.run_authoritative_suite(
        cutoff_date=CUTOFF,
        series_ids=[WTI_SERIES_ID],
        target_series_id=WTI_SERIES_ID,
        horizons=HORIZONS,
        frequency="B",
    )
    asyncio.run(
        predictor.research.run_research_pipeline(
            CUTOFF,
            "underlying event query",
            "physical flows query",
            "supply response query",
            "market reaction query",
        )
    )


@pytest.fixture
def predictor(monkeypatch, synthetic_service):
    settings = CfmV60Settings(code_execution_enabled=False)
    config = build_cfm_agent_config(data_service=synthetic_service, settings=settings)
    built = build_cfm_agent_predictor(config)
    _stub_research(monkeypatch, built.research)
    return built


def _install_initial_response(monkeypatch, predictor, raw_factory):
    def fake_initial(self, task, context):
        _run_tools(predictor)
        return raw_factory(predictor.research.last_result)

    monkeypatch.setattr(StructuredAssessmentController, "_initial_response", fake_initial)


def test_full_offline_run_produces_audited_eleven_level_forecasts(
    monkeypatch, predictor, synthetic_task, synthetic_context
):
    _install_initial_response(monkeypatch, predictor, lambda packet: json.dumps(_assessment(packet)))
    predictions = predictor.predict(synthetic_task, synthetic_context)
    assert [prediction.forecast_date.date() for prediction in predictions] == [
        (datetime.fromisoformat(CUTOFF) + __import__("pandas").offsets.BDay(h)).date() for h in HORIZONS
    ]
    for prediction in predictions:
        quantiles = prediction.payload.quantiles
        assert set(quantiles) == set(STANDARD_QUANTILES)
        assert prediction.payload.point_forecast == quantiles[0.5]
        ordered = [quantiles[level] for level in sorted(quantiles)]
        assert ordered == sorted(ordered)
        metadata = prediction.metadata
        for key in V52_AUDIT_KEYS + V60_AUDIT_KEYS:
            assert key in metadata, key
        assert metadata["llm_authored_final_numbers"] is False
        assert metadata["audit_findings_changed_forecast"] is False
        assert metadata["model_suite_id"].startswith("model_suite:sha256:")
        assert metadata["data_fingerprint"].startswith("data:sha256:")
        assert set(metadata["component_dispositions"]) == {"bootstrap", "sparse_ar", "lightgbm"}
        assert all(item["status"] == "ok" for item in metadata["component_dispositions"].values())
        assert metadata["ensemble_guard_applied"] is False
        assert set(metadata["historical_transformation_inputs"]) == {"5", "10", "21"}
        assert metadata["resolved_settings"]["overrides"] == {"code_execution_enabled": False}
        assert metadata["resolved_settings"]["settings_file_sha256"]
        assert metadata["llm_preflight"] == {"executed": False}
        assert metadata["data_snapshot"] is None
        assert metadata["code_execution_requests"] == {"accepted": [], "rejected": []}
        assert metadata["structured_output_control_audit"]["final_disposition"] == "initial_response_accepted"
        assert metadata["forecast_transformation"]["engine_id"] == "python_forecast_engine_v60"
        assert metadata["policy_decision"]["policy_id"] == "graduated_evidence_policy_v60"
    by_horizon = {prediction.metadata["policy_decision"]["horizon"]: prediction for prediction in predictions}
    decision_5 = by_horizon[5].metadata["policy_decision"]
    assert decision_5["evidence_tier_level"] >= 1, decision_5
    transformation_5 = by_horizon[5].metadata["forecast_transformation"]
    assert transformation_5["neutral"] is False
    assert transformation_5["selected_center_percentile"] == 60
    assert transformation_5["uncertainty_multiplier"] > 1.0
    assert transformation_5["final_point_forecast"] > by_horizon[5].metadata["unadjusted_ensemble"]["point_forecast"]
    transformation_21 = by_horizon[21].metadata["forecast_transformation"]
    assert transformation_21["neutral"] is True
    assert {float(level): value for level, value in transformation_21["final_quantiles"].items()} == {
        float(level): value for level, value in by_horizon[21].metadata["unadjusted_ensemble"]["quantiles"].items()
    }


def test_neutral_fallback_reproduces_the_ensemble_exactly(monkeypatch, predictor, synthetic_task, synthetic_context):
    _install_initial_response(monkeypatch, predictor, lambda packet: "this is not json")
    monkeypatch.setattr(StructuredAssessmentController, "_corrected_response", lambda self, **_: "still not json")
    predictions = predictor.predict(synthetic_task, synthetic_context)
    for prediction in predictions:
        audit = prediction.metadata["structured_output_control_audit"]
        assert audit["final_disposition"] == "neutral_fallback_after_retry_failure"
        assert audit["correction_retry_attempted"] is True
        unadjusted = prediction.metadata["unadjusted_ensemble"]
        assert prediction.payload.quantiles == {float(k): v for k, v in unadjusted["quantiles"].items()}
        assert prediction.payload.point_forecast == unadjusted["point_forecast"]
        assert prediction.metadata["forecast_transformation"]["neutral"] is True
        assert prediction.metadata["llm_context_assessment"]["incremental_novelty"] == "indeterminate"
        assert prediction.metadata["llm_context_assessment"]["confidence"] == 0


def test_invalid_then_corrected_response_is_accepted(monkeypatch, predictor, synthetic_task, synthetic_context):
    _install_initial_response(monkeypatch, predictor, lambda packet: "{not valid json")

    def corrected(self, **_):
        return json.dumps(_assessment(predictor.research.last_result))

    monkeypatch.setattr(StructuredAssessmentController, "_corrected_response", corrected)
    predictions = predictor.predict(synthetic_task, synthetic_context)
    for prediction in predictions:
        audit = prediction.metadata["structured_output_control_audit"]
        assert audit["final_disposition"] == "corrected_response_accepted"
        assert audit["retry_succeeded"] is True
    assert predictions[0].metadata["forecast_transformation"]["neutral"] is False


def test_predictor_id_is_computed_from_settings(predictor):
    assert predictor.predictor_id == f"agent_predictor_{AGENT_NAME}_{predictor.settings.main_model}_continuous"
    assert predictor.predictor_id == "agent_predictor_cfm_agent_v_6_0_gemini-3.1-flash-lite-preview_continuous"
