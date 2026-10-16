"""End-to-end v6.0 controller: task binding, one agent turn, policy, transformation, audit.

Specification Sections 5 and 8. The orchestration matches v5.2: the numerical suite
and the research pipeline execute once inside the initial agent turn, Python then
validates the structured response, applies the evidence policy per horizon,
transforms the ensemble with the historical profile, and writes the unified audit
record into every prediction's metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import ContinuousForecast, Prediction
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic import AgentConfig, AgentPredictor
from energy_oil_forecasting.cfm_agent_v_6_0.claim_support_verifier import (
    audit_claim_support,
)
from energy_oil_forecasting.cfm_agent_v_6_0.config import (
    AGENT_NAME,
    CfmV60Settings,
    resolved_settings_audit,
)
from energy_oil_forecasting.cfm_agent_v_6_0.forecast_engine import PythonForecastEngine
from energy_oil_forecasting.cfm_agent_v_6_0.llm_roles import (
    RECORDER,
    main_model_kwargs,
    record_effective_parameters,
)
from energy_oil_forecasting.cfm_agent_v_6_0.outputs import CfmContextAssessmentOutput
from energy_oil_forecasting.cfm_agent_v_6_0.policy import (
    EnsembleLockedPolicy,
    EvidencePolicy,
)
from energy_oil_forecasting.cfm_agent_v_6_0.prompts import CfmV60PromptBuilder
from energy_oil_forecasting.cfm_agent_v_6_0.response_control import (
    StructuredAssessmentController,
)
from energy_oil_forecasting.cfm_agent_v_6_0.tools import (
    AuditedCodeExecutionTool,
    AuthoritativeSuiteTool,
    ResearchPipelineTool,
)


CLAIM_SUPPORT_VERIFIER_ID = "claim_support_verifier_v60_audit_only"
DATA_SNAPSHOT_ENV = "CFM_V60_DATA_SNAPSHOT"
LLM_PREFLIGHT_ENV = "CFM_V60_LLM_PREFLIGHT_JSON"
ACTIVE_EVIDENCE_DECLARATION = (
    "The LLM used the authoritative numerical-suite output, the deterministic diagnostics, and "
    "main-verifier-cleaned summaries with passage identifiers and mechanically resolved source "
    "identities. Destination-page and audit-only content were not exposed. Optional diagnostic "
    "code use is separately declared in code_execution_audit."
)


def data_snapshot_record(path: str | None = None) -> dict[str, Any] | None:
    """Return {path, sha256} for the frozen data snapshot named by the environment, else None."""
    named = path if path is not None else os.environ.get(DATA_SNAPSHOT_ENV)
    if not named:
        return None
    root = Path(named)
    digest_file = root / "DATA_SNAPSHOT.sha256"
    return {
        "path": str(root),
        "sha256": hashlib.sha256(digest_file.read_bytes()).hexdigest() if digest_file.exists() else None,
        "hash_file": str(digest_file) if digest_file.exists() else None,
    }


def llm_preflight_record(result: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the preflight result given, or the one named by the environment, or an unexecuted marker."""
    if result is not None:
        return {"executed": True, **result}
    named = os.environ.get(LLM_PREFLIGHT_ENV)
    if named and Path(named).exists():
        return {"executed": True, **json.loads(Path(named).read_text(encoding="utf-8"))}
    return {"executed": False}


class CfmV60Predictor(Predictor):
    def __init__(
        self,
        config: AgentConfig,
        market: AuthoritativeSuiteTool,
        research: ResearchPipelineTool,
        code_execution: AuditedCodeExecutionTool,
        settings: CfmV60Settings,
    ):
        self.market = market
        self.research = research
        self.code_execution = code_execution
        self.settings = settings
        self.policy = (
            EnsembleLockedPolicy(settings) if settings.policy_mode == "ensemble_locked" else EvidencePolicy(settings)
        )
        self.engine = PythonForecastEngine(settings)
        self.inner = AgentPredictor(
            agent_config=config,
            prompt_builder=CfmV60PromptBuilder(),
            output_schema=CfmContextAssessmentOutput,
        )
        self.assessment_controller = StructuredAssessmentController(self.inner, settings)
        #: Set by the validation scripts after ``release.py preflight-models``; recorded in the audit.
        self.preflight_result: dict[str, Any] | None = None

    @property
    def predictor_id(self) -> str:
        return f"agent_predictor_{AGENT_NAME}_{self.settings.main_model}_continuous"

    @property
    def _runner(self) -> Any:
        return self.inner._runner

    def predict(  # noqa: PLR0912, PLR0915
        self, task: ForecastingTask, context: ForecastContext
    ) -> list[Prediction]:
        cutoff = str(pd.Timestamp(context.as_of).date())
        expected_horizons = [int(horizon) for horizon in task.horizons]
        if expected_horizons != sorted(set(expected_horizons)):
            raise RuntimeError("task horizons must be strictly increasing and unique")
        settings_audit = resolved_settings_audit(self.settings)
        RECORDER.reset()
        record_effective_parameters(
            "main",
            main_model_kwargs(self.settings),
            note=(
                f"AgentConfig.temperature={self.settings.main_temperature}, "
                f"max_output_tokens={self.settings.main_max_tokens}, seed=None"
            ),
        )
        self.market.prepare_workflow(task, context)
        self.research.prepare_workflow(cutoff)
        self.code_execution.prepare_workflow()
        # The initial agent turn executes both one-shot tools. Resolve the raw
        # structured response only after the active research packet exists.
        first_raw = self.assessment_controller._initial_response(task, context)
        result = self.market.last_result
        packet = self.research.last_result
        if result is None or result.model_suite is None or result.model_suite.ensemble is None:
            raise RuntimeError("one authoritative suite result required")
        if packet is None:
            raise RuntimeError("one research pipeline result required")
        research_audit = self.research.audit
        if research_audit["successful_execution_count"] != 1:
            raise RuntimeError("exactly one successful research execution is required")
        suite = result.model_suite
        if suite.cutoff_date != cutoff or packet.cutoff_date != cutoff:
            raise RuntimeError("numerical and research cutoffs must equal the harness-controlled cutoff")
        if suite.target_series_id != task.target_series_id or suite.frequency != task.frequency:
            raise RuntimeError("numerical-suite target or frequency does not match the task")
        if suite.horizons != expected_horizons:
            raise RuntimeError("numerical-suite horizons do not exactly match the task")
        if [item.horizon for item in suite.ensemble.forecasts] != expected_horizons:
            raise RuntimeError("ensemble horizons do not exactly match the task")
        # Validate the initial raw response, perform at most one correction-only
        # LLM retry, and fall back to a Python-owned neutral assessment if needed.
        resolution = self.assessment_controller.resolve(
            task=task,
            context=context,
            packet_id=packet.packet_id,
            initial_raw_response=first_raw,
        )
        placeholders = resolution.predictions
        assessment = resolution.assessment
        structured_output_audit = resolution.audit
        if len(placeholders) != len(expected_horizons):
            raise RuntimeError("placeholder count does not match task horizons")
        if self.settings.audit_enabled:
            findings, support_audit = audit_claim_support(assessment, packet, self.settings)
        else:
            findings = []
            support_audit = {
                "verifier_id": CLAIM_SUPPORT_VERIFIER_ID,
                "audit_only": True,
                "executed": False,
                "completed": False,
                "reason": "audit switch is off",
                "findings": [],
            }
        source_bundle = self.research.last_source_audit
        source_audit = source_bundle.model_dump(mode="json")
        ensemble = {item.horizon: item for item in suite.ensemble.forecasts}
        profiles = suite.diagnostics.historical_profiles
        decisions = {horizon: self.policy.apply(assessment, packet, horizon) for horizon in expected_horizons}
        transforms = {
            horizon: self.engine.transform(
                ensemble[horizon],
                decisions[horizon],
                assessment.incremental_novelty,
                suite.diagnostics.latest_value,
                profiles.get(horizon),
            )
            for horizon in expected_horizons
        }
        by_date = {str(value.forecast_date.date()): value for value in placeholders}
        market_audit = self.market.audit
        code_audit = self.code_execution.audit
        numerical_audit = {
            "model_suite": suite.model_dump(mode="json"),
            "execution": market_audit,
        }
        llm_effective_parameters = RECORDER.snapshot()
        common_metadata = {
            "model_suite_id": suite.model_suite_id,
            "numerical_suite_audit": numerical_audit,
            "research_execution_audit": research_audit,
            "code_execution_audit": code_audit,
            "llm_context_assessment": assessment.model_dump(mode="json"),
            "structured_output_control_audit": structured_output_audit,
            "active_research_packet": packet.model_dump(mode="json"),
            "cutoff_verification_audit": self.research.last_verification_audit.model_dump(mode="json"),
            "source_validation_audit": source_audit,
            "claim_support_audit": support_audit,
            "claim_support_findings": [finding.model_dump(mode="json") for finding in findings],
            "market_diagnostics": suite.diagnostics.model_dump(mode="json"),
            "task_binding": {
                "cutoff": cutoff,
                "target_series_id": task.target_series_id,
                "horizons": expected_horizons,
                "frequency": task.frequency,
                "validated": True,
            },
            "audit_switch": self.settings.audit_enabled,
            "audit_only_controls": {
                "source_validator_executed": source_bundle.executed,
                "claim_support_verifier_executed": bool(support_audit.get("executed")),
                "claim_support_verifier_completed": bool(support_audit.get("completed")),
            },
            "active_evidence_declaration": ACTIVE_EVIDENCE_DECLARATION,
            "audit_findings_changed_forecast": False,
            "llm_authored_final_numbers": False,
            # v6.0 additions (Section 8).
            "resolved_settings": settings_audit,
            "floors": suite.floors.model_dump(mode="json"),
            "history_window": suite.history_window.model_dump(mode="json"),
            "data_fingerprint": suite.data_fingerprint,
            "data_snapshot": data_snapshot_record(),
            "component_seeds": dict(suite.component_seeds),
            "component_dispositions": {
                name: disposition.model_dump(mode="json") for name, disposition in suite.dispositions.items()
            },
            "ensemble_guard_applied": suite.ensemble_guard_applied,
            "historical_transformation_inputs": {
                str(horizon): profile.model_dump(mode="json") for horizon, profile in profiles.items()
            },
            "llm_effective_parameters": llm_effective_parameters,
            "llm_preflight": llm_preflight_record(self.preflight_result),
            "code_execution_requests": {
                "accepted": list(code_audit.get("accepted_requests", [])),
                "rejected": list(code_audit.get("rejected_requests", [])),
            },
        }
        for item in suite.ensemble.forecasts:
            prediction = by_date.get(item.forecast_date)
            if prediction is None:
                raise RuntimeError(f"missing placeholder for forecast date {item.forecast_date}")
            transformation = transforms[item.horizon]
            prediction.payload = ContinuousForecast(
                point_forecast=transformation.final_point_forecast,
                quantiles=transformation.final_quantiles,
            )
            prediction.metadata.update(
                {
                    **common_metadata,
                    "policy_decision": decisions[item.horizon].model_dump(mode="json"),
                    "forecast_transformation": transformation.model_dump(mode="json"),
                    "unadjusted_ensemble": {
                        "point_forecast": item.point_forecast,
                        "quantiles": item.quantiles,
                    },
                }
            )
        return placeholders


__all__ = [
    "ACTIVE_EVIDENCE_DECLARATION",
    "CLAIM_SUPPORT_VERIFIER_ID",
    "DATA_SNAPSHOT_ENV",
    "LLM_PREFLIGHT_ENV",
    "CfmV60Predictor",
    "data_snapshot_record",
    "llm_preflight_record",
]
