"""Identifier-only prompt builder for ``cfm_agent_v_2_0``."""

from __future__ import annotations

import json

import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.task import ForecastingTask
from energy_oil_forecasting.cfm_agent_v_2_0.outputs import CfmRichForecastOutput


class CfmForecastPromptBuilder:
    """Serialize the task without embedding raw target or covariate history."""

    def __init__(self) -> None:
        self._schema = CfmRichForecastOutput.prompt_schema_json()

    def __call__(
        self,
        *,
        task: ForecastingTask,
        context: ForecastContext,
    ) -> str:
        """Build a compact identifier-only forecasting payload."""
        payload = {
            "agent_name": "cfm_agent_v_2_0",
            "task_id": task.task_id,
            "target_series_id": task.target_series_id,
            "description": task.description,
            "as_of": str(pd.Timestamp(context.as_of).date()),
            "horizons": list(task.horizons),
            "frequency": task.frequency,
            "available_series_ids": context.series_ids,
            "raw_history_in_prompt": False,
            "data_access": {
                "tool": "query_market_data",
                "instruction": (
                    "Use the target and covariate series IDs to retrieve cutoff-safe "
                    "data and deterministic model results. Raw observations are not "
                    "embedded in this prompt."
                ),
            },
            "required_skills": [
                "forecasting",
                "model-selection",
                "research",
                "code-analysis",
            ],
            "instructions": (
                "Load forecasting and model-selection. Call query_market_data "
                "with operation get_series_and_run_models exactly once using this "
                "payload's cutoff, target, horizons, and frequency. Preserve the "
                "tool's component-model values exactly in the rich output. Load "
                "research before search_web. Load code-analysis before optional "
                "run_code. Call set_model_response exactly once with json_response "
                "matching output_schema."
            ),
            "output_schema": self._schema,
        }
        return json.dumps(payload, indent=2)


__all__ = ["CfmForecastPromptBuilder"]
