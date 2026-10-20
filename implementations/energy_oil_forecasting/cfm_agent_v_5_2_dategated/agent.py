"""Build CFM Agent v5.2 ARIMA-only with the publication-date gate on research.

Everything except the research tool is v5.2's ``build_cfm_agent_config_arima_only``:
same model, persona, skills, numerical suite, policy, engine, and settings. The agent
keeps v5.2's name and description because ADK writes both into the system
instruction ("Your internal name is ..."), so changing them would change the prompt.
Results are kept apart from v5.2's by the predictor ID alone.
"""

from __future__ import annotations

from typing import Any

from aieng.forecasting.data import DataService
from aieng.forecasting.methods.agentic import AgentConfig
from aieng.forecasting.models import LITE_MODEL
from energy_oil_forecasting.cfm_agent_v_5_2 import agent as _v52_agent
from energy_oil_forecasting.cfm_agent_v_5_2.config import DEFAULT_SETTINGS, CfmV52Settings
from energy_oil_forecasting.cfm_agent_v_5_2.predictor import CfmV51Predictor
from energy_oil_forecasting.cfm_agent_v_5_2_dategated.research import DateGatedResearchPipelineTool


_V52_ID_STEM = "agent_predictor_cfm_agent_v_5_2_arima_only_"
_ID_STEM = "agent_predictor_cfm_agent_v_5_2_arima_only_dategated_"


class DateGatedPredictor(CfmV51Predictor):
    """v5.2's predictor under its own ID, so its results never share a cache file with v5.2's."""

    @property
    def predictor_id(self) -> str:
        inner = self.inner.predictor_id
        if not inner.startswith(_V52_ID_STEM):
            raise RuntimeError(f"unexpected v5.2 predictor id {inner!r}")
        return _ID_STEM + inner[len(_V52_ID_STEM) :]

    def predict(self, task: Any, context: Any) -> list[Any]:
        predictions = super().predict(task, context)
        for prediction in predictions:
            prediction.predictor_id = self.predictor_id
        return predictions


def build_cfm_agent_config_arima_only_dategated(
    model: str = LITE_MODEL,
    *,
    data_service: DataService | None = None,
    settings: CfmV52Settings = DEFAULT_SETTINGS,
    search_model: str = LITE_MODEL,
    covariate_series_ids: list[str] | None = None,
    allow_undated: bool = False,
) -> AgentConfig:
    """Return v5.2's ARIMA-only config with the research tool swapped for the date-gated one."""
    config = _v52_agent.build_cfm_agent_config_arima_only(
        model,
        data_service=data_service,
        settings=settings,
        search_model=search_model,
        covariate_series_ids=covariate_series_ids,
    )
    _, market, research, code_execution, bound_settings = _v52_agent._CONFIG.pop(id(config))
    gated = DateGatedResearchPipelineTool(bound_settings, search_model=search_model, allow_undated=allow_undated)

    tools = list(config.function_tools)
    swapped = [i for i, tool in enumerate(tools) if getattr(tool, "func", None) == research.run_research_pipeline]
    if len(swapped) != 1:
        raise RuntimeError("could not find v5.2's research tool in the config")
    tools[swapped[0]] = gated.as_function_tool()
    gated_config = config.model_copy(update={"function_tools": tools})
    _v52_agent._CONFIG[id(gated_config)] = (gated_config, market, gated, code_execution, bound_settings)
    return gated_config


def build_cfm_agent_predictor_arima_only_dategated(config: AgentConfig) -> DateGatedPredictor:
    """Return the predictor for a config from :func:`build_cfm_agent_config_arima_only_dategated`."""
    bundle = _v52_agent._CONFIG.pop(id(config), None)
    if bundle is None or bundle[0] is not config or not isinstance(bundle[2], DateGatedResearchPipelineTool):
        raise ValueError("Configuration was not created by build_cfm_agent_config_arima_only_dategated.")
    _, market, research, code_execution, settings = bundle
    return DateGatedPredictor(config, market, research, code_execution, settings)
