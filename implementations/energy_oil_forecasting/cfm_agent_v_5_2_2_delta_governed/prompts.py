"""Prompt builder for CFM Agent v5.2.2 Delta-Governed.

Forked from
:class:`energy_oil_forecasting.cfm_agent_v_5_2.prompts.CfmV51PromptBuilder`
because it hardcodes ``CfmContextAssessmentOutput.prompt_schema_json()``
directly rather than deriving the embedded schema template from whatever
``output_schema`` the predictor was actually configured with. Identical
payload shape, just pointing at
:class:`~energy_oil_forecasting.cfm_agent_v_5_2_2_delta_governed.outputs.CfmContextAssessmentOutputDeltaGoverned`
(the rank-based schema) instead.
"""

import json

import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.task import ForecastingTask
from energy_oil_forecasting.cfm_agent_v_5_2_2_delta_governed.outputs import (
    CfmContextAssessmentOutputDeltaGoverned,
)


class CfmDeltaGovernedPromptBuilder:
    def __call__(self, *, task: ForecastingTask, context: ForecastContext) -> str:
        return json.dumps(
            {
                "agent_name": "cfm_agent_v_5_2_2_delta_governed",
                "task_id": task.task_id,
                "target_series_id": task.target_series_id,
                "description": task.description,
                "as_of": str(pd.Timestamp(context.as_of).date()),
                "horizons": list(task.horizons),
                "frequency": task.frequency,
                "available_series_ids": context.series_ids,
                "raw_history_in_prompt": False,
                "instructions": (
                    "Load source-selection before research-planning and claim-building. Call "
                    "run_authoritative_suite exactly once. Construct exactly four neutral queries and call "
                    "run_research_pipeline exactly once. Use only verifier-approved cleaned summaries in the "
                    "returned active packet. Build atomic claims that cite verified summary IDs and "
                    "provenance-correct associated source IDs, preferring the strongest sources under "
                    "source-selection. Never use unfiltered text, removed claims, destination-page content, or "
                    "model memory as evidence. Do not invent or alter summaries, IDs, URLs, domains, "
                    "publishers, or verification results. Propose only categorical actions: center_action is a "
                    "discrete integer rank in {-2,-1,0,1,2} — never a string, never a price. Audit-only "
                    "components are not visible to you and cannot affect the forecast. Do not calculate "
                    "prices, quantiles, multipliers, or adjustments. Call set_model_response exactly once."
                ),
                "output_schema": CfmContextAssessmentOutputDeltaGoverned.prompt_schema_json(),
            },
            indent=2,
        )


__all__ = ["CfmDeltaGovernedPromptBuilder"]
