"""Identifier-only assessment prompt for CFM Agent v3.4."""

from __future__ import annotations

import json

import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.task import ForecastingTask
from energy_oil_forecasting.cfm_agent_v_3_4.outputs import CfmContextAssessmentOutput


class CfmV34PromptBuilder:
    def __init__(self) -> None:
        self._schema = CfmContextAssessmentOutput.prompt_schema_json()

    def __call__(self, *, task: ForecastingTask, context: ForecastContext) -> str:
        payload = {
            "agent_name": "cfm_agent_v_3_4",
            "task_id": task.task_id,
            "target_series_id": task.target_series_id,
            "description": task.description,
            "as_of": str(pd.Timestamp(context.as_of).date()),
            "horizons": list(task.horizons),
            "frequency": task.frequency,
            "available_series_ids": context.series_ids,
            "raw_history_in_prompt": False,
            "instructions": (
                "Load all required skills. Call run_authoritative_suite exactly once with the exact cutoff, target, required horizons, frequency, and relevant series IDs. Treat its model suite and diagnostics as authoritative. "
                "Execute four distinct search_web calls, one for each part of the fixed research plan: underlying event "
                "and official statements; confirmed physical oil-flow impact; production or "
                "strategic-reserve response; and independent market-reaction reporting. Use only "
                "cutoff-approved evidence. Do not report a planned query as executed. An empty verifier result supplies no facts. Never infer source title, publisher, date, tier, or provenance from an opaque redirect URL. Report verifier_processing_status only when the accepted tool output establishes it; otherwise use unknown. One eligible high-quality publisher may support only a small bounded action; two independent publishers may support a moderate bounded action; three or more strongly corroborated publishers may support the strongest bounded action. Separate facts from interpretation, preserve conflicts, and "
                "select only authorized categorical actions. Do not calculate any final price, "
                "quantile, dollar adjustment, or uncertainty multiplier. Call set_model_response "
                "exactly once with json_response matching output_schema."
            ),
            "output_schema": self._schema,
        }
        return json.dumps(payload, indent=2)


__all__ = ["CfmV34PromptBuilder"]
