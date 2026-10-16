"""Package-local structured-assessment correction and neutral-fallback control.

Specification Section 4.4: one initial assessment, at most one correction-only
resubmission (no tools; the research pipeline and the numerical suite are not
rerun), then a fully audited neutral fallback that the forecast engine turns into
an exact ensemble reproduction. The correction call is role 4 of Section 4.3 and
reads every generation parameter from settings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import Prediction
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic.agent_factory import AS_OF_STATE_KEY
from aieng.forecasting.methods.agentic.predictor import _run_coroutine_sync
from aieng.forecasting.methods.llm_processes._client import strip_markdown_fence
from energy_oil_forecasting.cfm_agent_v_6_0.config import CfmV60Settings
from energy_oil_forecasting.cfm_agent_v_6_0.llm_roles import (
    completion_kwargs,
    dropped_parameters,
    record_effective_parameters,
)
from energy_oil_forecasting.cfm_agent_v_6_0.outputs import CfmContextAssessmentOutput
from energy_oil_forecasting.cfm_agent_v_6_0.schemas import HorizonAction
from pydantic import ValidationError


CONTROL_ID = "structured_assessment_retry_v60"
CORRECTION_ROLE = "structured_output_correction"
NEUTRAL_FALLBACK_WARNING = "structured_output_neutral_fallback"
_CORRECTION_SYSTEM_PROMPT = "You repair structured JSON. Return only corrected JSON."
_CORRECTION_INSTRUCTION = (
    "Correct the prior JSON so it validates exactly against the supplied schema. "
    "Preserve substantive findings unless a validation error requires a correction. "
    "Do not research, call tools, add evidence, or calculate forecasts. Return JSON only."
)


@dataclass(frozen=True)
class AssessmentResolution:
    assessment: CfmContextAssessmentOutput
    predictions: list[Prediction]
    audit: dict[str, Any]


class StructuredAssessmentController:
    """Run one assessment, one correction-only retry, then the neutral fallback."""

    def __init__(self, inner: Any, settings: CfmV60Settings) -> None:
        self.inner = inner
        self.settings = settings

    @staticmethod
    def _errors(exc: BaseException) -> list[dict[str, Any]]:
        if isinstance(exc, ValidationError):
            return exc.errors(include_url=False)
        return [{"type": type(exc).__name__, "msg": str(exc)}]

    @staticmethod
    def _validate_contract(
        assessment: CfmContextAssessmentOutput,
        *,
        packet_id: str,
        horizons: list[int],
    ) -> None:
        if assessment.research_packet_id != packet_id:
            raise ValueError("research_packet_id must equal the active research packet ID")
        if [item.horizon for item in assessment.horizon_actions] != horizons:
            raise ValueError("horizon_actions must exactly match task horizons in task order")

    @staticmethod
    def _neutral(packet_id: str, horizons: list[int]) -> CfmContextAssessmentOutput:
        """Build the neutral fallback: no_change and unchanged everywhere, confidence 0, no claims."""
        return CfmContextAssessmentOutput(
            research_packet_id=packet_id,
            evidence_claims=[],
            physical_status="unknown",
            incremental_novelty="indeterminate",
            material_evidence_conflict=False,
            confidence=0.0,
            horizon_actions=[
                HorizonAction(
                    horizon=horizon,
                    center_action="no_change",
                    uncertainty_action="unchanged",
                    persistence_profile="unknown",
                    cited_claim_ids=[],
                    rationale="Neutral fallback after two invalid structured-output attempts.",
                )
                for horizon in horizons
            ],
            research_summary="Structured assessment could not be validated after one controlled correction retry.",
            overall_rationale="Python applied the audited neutral fallback.",
            warnings=[NEUTRAL_FALLBACK_WARNING],
        )

    def _initial_response(self, task: ForecastingTask, context: ForecastContext) -> str:
        prompt = self.inner.prompt_builder(task=task, context=context)
        initial_state = {AS_OF_STATE_KEY: str(context.as_of)[:10]}  # structural: ISO date prefix of a timestamp
        return _run_coroutine_sync(self.inner._runner.run_text_async(prompt, initial_state=initial_state))

    @staticmethod
    def _correction_request(
        *,
        raw_response: str,
        errors: list[dict[str, Any]],
        packet_id: str,
        horizons: list[int],
    ) -> dict[str, Any]:
        """Build the correction-only request body, recorded verbatim in the audit."""
        return {
            "instruction": _CORRECTION_INSTRUCTION,
            "required_research_packet_id": packet_id,
            "required_horizons_in_order": horizons,
            "validation_errors": errors,
            "prior_response": raw_response,
            "output_schema_template": CfmContextAssessmentOutput.prompt_schema_json(),
        }

    def _corrected_response(
        self,
        *,
        raw_response: str,
        errors: list[dict[str, Any]],
        packet_id: str,
        horizons: list[int],
    ) -> str:
        """One correction-only resubmission through role 4, with its parameters recorded."""
        import litellm  # noqa: PLC0415

        correction = self._correction_request(
            raw_response=raw_response, errors=errors, packet_id=packet_id, horizons=horizons
        )
        kwargs_used = {
            **completion_kwargs(self.settings, CORRECTION_ROLE),
            "messages": [
                {"role": "system", "content": _CORRECTION_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(correction)},
            ],
        }
        response = litellm.completion(**kwargs_used)
        record_effective_parameters(
            CORRECTION_ROLE,
            kwargs_used,
            dropped=dropped_parameters(kwargs_used, model=kwargs_used["model"]),
            response_model=getattr(response, "model", None),
        )
        return response.choices[0].message.content or ""

    def _parse(
        self,
        raw: str,
        *,
        packet_id: str,
        horizons: list[int],
    ) -> CfmContextAssessmentOutput:
        cleaned = strip_markdown_fence(raw)
        try:
            assessment = CfmContextAssessmentOutput.model_validate_json(cleaned)
        except ValidationError:
            assessment = CfmContextAssessmentOutput.model_validate(json.loads(cleaned))
        self._validate_contract(assessment, packet_id=packet_id, horizons=horizons)
        return assessment

    def resolve(
        self,
        *,
        task: ForecastingTask,
        context: ForecastContext,
        packet_id: str,
        initial_raw_response: str | None = None,
    ) -> AssessmentResolution:
        horizons = [int(value) for value in task.horizons]
        first_raw = initial_raw_response if initial_raw_response is not None else self._initial_response(task, context)
        audit: dict[str, Any] = {
            "control_id": CONTROL_ID,
            "maximum_correction_retries": 1,
            "correction_model": self.settings.structured_output_correction_model,
            "initial_response": first_raw,
            "initial_validation_failed": False,
            "initial_validation_errors": [],
            "correction_retry_attempted": False,
            "correction_request": None,
            "retry_response": None,
            "retry_validation_errors": [],
            "retry_succeeded": False,
            "neutral_fallback_applied": False,
            "final_disposition": "initial_response_accepted",
        }
        try:
            assessment = self._parse(first_raw, packet_id=packet_id, horizons=horizons)
        except Exception as first_exc:  # noqa: BLE001
            audit["initial_validation_failed"] = True
            audit["initial_validation_errors"] = self._errors(first_exc)
            audit["correction_retry_attempted"] = True
            audit["correction_request"] = self._correction_request(
                raw_response=first_raw,
                errors=audit["initial_validation_errors"],
                packet_id=packet_id,
                horizons=horizons,
            )
            try:
                retry_raw = self._corrected_response(
                    raw_response=first_raw,
                    errors=audit["initial_validation_errors"],
                    packet_id=packet_id,
                    horizons=horizons,
                )
                audit["retry_response"] = retry_raw
                assessment = self._parse(retry_raw, packet_id=packet_id, horizons=horizons)
                audit["retry_succeeded"] = True
                audit["final_disposition"] = "corrected_response_accepted"
            except Exception as retry_exc:  # noqa: BLE001
                audit["retry_validation_errors"] = self._errors(retry_exc)
                audit["neutral_fallback_applied"] = True
                audit["final_disposition"] = "neutral_fallback_after_retry_failure"
                assessment = self._neutral(packet_id, horizons)
        predictions = assessment.to_predictions(
            task=task,
            context=context,
            predictor_id=self.inner.predictor_id,
            metadata={"context_assessment": assessment.model_dump(mode="json")},
        )
        return AssessmentResolution(assessment=assessment, predictions=predictions, audit=audit)


__all__ = [
    "CONTROL_ID",
    "CORRECTION_ROLE",
    "NEUTRAL_FALLBACK_WARNING",
    "AssessmentResolution",
    "StructuredAssessmentController",
]
