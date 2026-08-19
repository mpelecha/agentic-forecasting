"""Package-local structured-assessment retry and neutral-fallback control."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import Prediction
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic.agent_factory import AS_OF_STATE_KEY
from aieng.forecasting.methods.agentic.predictor import _run_coroutine_sync
from aieng.forecasting.methods.llm_processes._client import strip_markdown_fence
from energy_oil_forecasting.cfm_agent_v_5_2.config import CfmV52Settings
from energy_oil_forecasting.cfm_agent_v_5_2.outputs import CfmContextAssessmentOutput
from energy_oil_forecasting.cfm_agent_v_5_2.schemas import HorizonAction
from pydantic import ValidationError


@dataclass(frozen=True)
class AssessmentResolution:
    assessment: CfmContextAssessmentOutput
    predictions: list[Prediction]
    audit: dict[str, Any]


class StructuredAssessmentController:
    """Run one assessment, one correction-only retry, then neutral fallback."""

    def __init__(self, inner: Any, settings: CfmV52Settings) -> None:
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
            warnings=["structured_output_neutral_fallback"],
        )

    def _initial_response(self, task: ForecastingTask, context: ForecastContext) -> str:
        prompt = self.inner.prompt_builder(task=task, context=context)
        initial_state = {AS_OF_STATE_KEY: str(context.as_of)[:10]}
        return _run_coroutine_sync(self.inner._runner.run_text_async(prompt, initial_state=initial_state))

    def _corrected_response(
        self,
        *,
        raw_response: str,
        errors: list[dict[str, Any]],
        packet_id: str,
        horizons: list[int],
    ) -> str:
        import litellm  # noqa: PLC0415

        model = self.settings.structured_output_retry_model
        model = model if model.startswith("openai/") else f"openai/{model}"
        correction = {
            "instruction": (
                "Correct the prior JSON so it validates exactly against the supplied schema. "
                "Preserve substantive findings unless a validation error requires a correction. "
                "Do not research, call tools, add evidence, or calculate forecasts. Return JSON only."
            ),
            "required_research_packet_id": packet_id,
            "required_horizons_in_order": horizons,
            "validation_errors": errors,
            "prior_response": raw_response,
            "output_schema_template": CfmContextAssessmentOutput.prompt_schema_json(),
        }
        response = litellm.completion(
            model=model,
            api_base=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            messages=[
                {
                    "role": "system",
                    "content": "You repair structured JSON. Return only corrected JSON.",
                },
                {"role": "user", "content": json.dumps(correction)},
            ],
            temperature=0.0,
            max_tokens=8192,
            timeout=self.settings.structured_output_retry_timeout_seconds,
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
            "control_id": "structured_assessment_retry_v52",
            "maximum_correction_retries": 1,
            "initial_response": first_raw,
            "initial_validation_failed": False,
            "initial_validation_errors": [],
            "correction_retry_attempted": False,
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


__all__ = ["AssessmentResolution", "StructuredAssessmentController"]
