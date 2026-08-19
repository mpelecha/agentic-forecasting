"""Prompt for the v5.1 staged research and assessment workflow."""

import json

import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.task import ForecastingTask
from energy_oil_forecasting.cfm_agent_v_5_2.outputs import CfmContextAssessmentOutput


class CfmV51PromptBuilder:
    def __call__(self, *, task: ForecastingTask, context: ForecastContext) -> str:
        return json.dumps(
            {
                "agent_name": "cfm_agent_v_5_2",
                "task_id": task.task_id,
                "target_series_id": task.target_series_id,
                "description": task.description,
                "as_of": str(pd.Timestamp(context.as_of).date()),
                "horizons": list(task.horizons),
                "frequency": task.frequency,
                "available_series_ids": context.series_ids,
                "raw_history_in_prompt": False,
                "instructions": (
                    "Load source-selection before research-planning and claim-building. Call run_authoritative_suite exactly once. Construct exactly four "
                    "neutral queries and call run_research_pipeline exactly once. Use only verifier-approved "
                    "cleaned summaries in the returned active packet. Build atomic claims that cite verified "
                    "summary IDs and provenance-correct associated source IDs, preferring the strongest sources under source-selection. Never use unfiltered "
                    "text, removed claims, destination-page content, or model memory as evidence. Do not invent "
                    "or alter summaries, IDs, URLs, domains, publishers, or verification results. Propose only "
                    "categorical actions. Audit-only components are not visible to you and cannot affect the "
                    "forecast. Do not calculate prices, quantiles, multipliers, or adjustments. Call "
                    "set_model_response exactly once."
                ),
                "output_schema": CfmContextAssessmentOutput.prompt_schema_json(),
            },
            indent=2,
        )


__all__ = ["CfmV51PromptBuilder"]
