"""The WTI scenario-memory agent with a publication-date gate on its news search.

Identical to ``wti_scenario_memory_agent`` — same prompt, model, ARIMA anchor,
scenario math, and memory — except that its ``search_web`` tool is swapped for the
date-gated one from ``cfm_agent_v_5_2_dategated.search_web``: the agent sees only text
that the search's grounding metadata attributes to pages dated strictly before the
origin. The tool keeps the shared tool's name, signature, and description, so the
LLM's prompt and tool schema are unchanged.

The predictor ID gains ``_dategated``. Because the scenario memory store is keyed by
predictor ID, the gated agent reads and writes its own memory, never frameworks
produced by the ungated agent from possibly post-origin news.

Requires ``wti_scenario_memory_agent`` and ``cfm_agent_v_5_2_dategated`` (which in
turn needs ``cfm_agent_v_5_2``) in the same package.
"""

from __future__ import annotations

import os
from typing import Any

from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import Prediction
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic import AgentConfig
from aieng.forecasting.models import LITE_MODEL
from energy_oil_forecasting.cfm_agent_v_5_2.config import DEFAULT_SETTINGS, CfmV52Settings
from energy_oil_forecasting.cfm_agent_v_5_2_dategated.page_gate import GATE_ID, PageGate
from energy_oil_forecasting.cfm_agent_v_5_2_dategated.search_web import build_date_gated_search_web
from energy_oil_forecasting.wti_scenario_memory_agent import build_wti_news_scenario_schema_anchored_enhanced_config
from energy_oil_forecasting.wti_scenario_memory_agent.predictor import ScenarioSchemaAnchoredEnhancedPredictor


_STEM = "agent_predictor_wti_analyst_news_scenario_schema_anchored_enhanced"


class DateGatedScenarioMemoryPredictor(ScenarioSchemaAnchoredEnhancedPredictor):
    """The scenario-memory predictor with its ``search_web`` tool replaced by the date-gated one."""

    def __init__(
        self,
        config: AgentConfig,
        *,
        settings: CfmV52Settings = DEFAULT_SETTINGS,
        allow_undated: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        self.gate = PageGate(settings, allow_undated=allow_undated)
        self.gate_log: list[dict[str, Any]] = []
        gated = build_date_gated_search_web(
            config.context_retrieval,
            gate=self.gate,
            log=self.gate_log,
            openai_base_url=config.openai_base_url or os.getenv("OPENAI_BASE_URL") or "",
            openai_api_key=config.openai_api_key,
        )
        tools = self.inner._agent.tools
        slots = [i for i, tool in enumerate(tools) if getattr(tool, "__name__", None) == "search_web"]
        if len(slots) != 1:
            raise RuntimeError(f"expected one search_web tool on the agent, found {len(slots)}")
        tools[slots[0]] = gated

    @property
    def predictor_id(self) -> str:
        inner = self.inner.predictor_id
        if not inner.startswith(_STEM):
            raise RuntimeError(f"unexpected scenario-memory predictor id {inner!r}")
        return f"{_STEM}_dategated{inner[len(_STEM) :]}"

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        self.gate.reset()
        self.gate_log.clear()
        predictions = super().predict(task, context)
        audit = {"gate_id": GATE_ID, "allow_undated": self.gate.allow_undated, "attempts": list(self.gate_log)}
        for prediction in predictions:
            prediction.predictor_id = self.predictor_id
            prediction.metadata["date_gate"] = audit
        return predictions


def build_agent_dategated(
    model: str = LITE_MODEL,
    *,
    log_returns: bool = False,
    memory_limit: int = 2,
    allow_undated: bool = False,
) -> DateGatedScenarioMemoryPredictor:
    """Build the gated agent with the same defaults as ``wti_scenario_memory_agent.build_agent``."""
    config = build_wti_news_scenario_schema_anchored_enhanced_config(model=model, log_returns=log_returns)
    return DateGatedScenarioMemoryPredictor(
        config, anchor_log_returns=log_returns, memory_limit=memory_limit, allow_undated=allow_undated
    )


__all__ = ["DateGatedScenarioMemoryPredictor", "build_agent_dategated"]
