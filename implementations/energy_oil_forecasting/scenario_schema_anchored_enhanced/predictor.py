"""ARIMA-anchored, memory-enhanced WTI Scenario Schema predictor.

Combines two independent extensions of the base Scenario Schema agent:

1. Everything :class:`~energy_oil_forecasting.scenario_schema_anchored.predictor.ScenarioSchemaAnchoredPredictor`
   already does — Python, not the LLM, owns the final point_forecast/
   quantiles, built from a deterministic AutoARIMA anchor shifted and
   widened toward the LLM's scenarios. See that class's docstring for the
   full four-step orchestration; this class reuses it unchanged, via
   inheritance, rather than duplicating it.
2. Persistent scenario memory (new here). Before each call, the last few
   scenario frameworks this predictor produced at EARLIER origins (strictly
   before the current one, chronologically — see
   :mod:`energy_oil_forecasting.scenario_schema_anchored_enhanced.memory`
   for the leakage-safety argument) are read back and injected into the
   prompt as ``prior_frameworks``, so the agent can keep scenario naming and
   thematic structure consistent across origins instead of reinventing them
   from a blank slate every call. After the call, this origin's own
   factors/scenarios are appended to the same store for future origins to
   read.
"""

from __future__ import annotations

from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import Prediction
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic import AgentConfig, AgentPredictor
from energy_oil_forecasting.analyst_agent.agent import WtiScenarioForecastOutput
from energy_oil_forecasting.scenario_schema_anchored.predictor import ScenarioSchemaAnchoredPredictor
from energy_oil_forecasting.scenario_schema_anchored_enhanced.memory import (
    append_framework,
    read_prior_frameworks,
)
from energy_oil_forecasting.scenario_schema_anchored_enhanced.prompt import AnchoredMemoryPromptBuilder


class ScenarioSchemaAnchoredEnhancedPredictor(ScenarioSchemaAnchoredPredictor):
    """ARIMA-anchored Scenario Schema with persistent cross-origin scenario memory.

    Same numerical arithmetic as
    :class:`~energy_oil_forecasting.scenario_schema_anchored.predictor.ScenarioSchemaAnchoredPredictor`
    (Python owns point_forecast/quantiles; see that class's docstring) —
    inherited and called via ``super().predict()``, not reimplemented. The
    only difference is what the LLM sees before it answers: the last
    ``memory_limit`` scenario frameworks this predictor produced at earlier
    origins, and what happens after it answers: this origin's frameworks get
    written back for the next one to read.

    ``__init__`` is a full override rather than a call to
    ``super().__init__()`` — the parent builds its inner
    :class:`AgentPredictor` around a plain
    :class:`~energy_oil_forecasting.scenario_schema_anchored.prompt.AnchoredPromptBuilder`,
    which has no ``prior_frameworks`` field, and :class:`AgentPredictor`
    captures the prompt builder object by reference at construction time —
    so swapping ``self._prompt_builder`` afterward would leave ``self.inner``
    still pointed at the old, memory-blind one.
    """

    def __init__(
        self,
        config: AgentConfig,
        *,
        arima_num_samples: int = 1_000,
        anchor_log_returns: bool = True,
        memory_limit: int = 2,
    ) -> None:
        self.arima_num_samples = arima_num_samples
        self.anchor_log_returns = anchor_log_returns
        self.memory_limit = memory_limit
        self._prompt_builder = AnchoredMemoryPromptBuilder(arima_anchor={}, prior_frameworks=[])
        self.inner = AgentPredictor(
            agent_config=config,
            prompt_builder=self._prompt_builder,
            output_schema=WtiScenarioForecastOutput,
        )

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        as_of = str(context.as_of)[:10]

        prior_frameworks = read_prior_frameworks(
            self.predictor_id, task.task_id, before=as_of, limit=self.memory_limit
        )
        # Stashed here so AnchoredMemoryPromptBuilder.__call__ can read it
        # when the inner AgentPredictor invokes the prompt builder inside
        # super().predict() below — mirrors how the parent stashes
        # arima_anchor onto the same object.
        self._prompt_builder.prior_frameworks = prior_frameworks

        predictions = super().predict(task, context)

        factors = predictions[0].metadata.get("factors", [])
        scenarios = predictions[0].metadata.get("scenarios", [])
        append_framework(
            self.predictor_id, task.task_id, as_of=as_of, factors=factors, scenarios=scenarios
        )

        for pred in predictions:
            pred.metadata["prior_frameworks_used"] = len(prior_frameworks)

        return predictions


def build_wti_scenario_schema_anchored_enhanced_predictor(
    config: AgentConfig,
    *,
    arima_num_samples: int = 1_000,
    anchor_log_returns: bool = True,
    memory_limit: int = 2,
) -> ScenarioSchemaAnchoredEnhancedPredictor:
    """Wrap an :class:`AgentConfig` in a :class:`ScenarioSchemaAnchoredEnhancedPredictor`.

    Parameters
    ----------
    config : AgentConfig
        Config from :func:`~energy_oil_forecasting.scenario_schema_anchored_enhanced.agent.build_wti_news_scenario_schema_anchored_enhanced_config`.
    arima_num_samples : int, default=1_000
        Monte Carlo sample count for the AutoARIMA anchor.
    anchor_log_returns : bool, default=True
        Fit the anchor on log returns rather than the price level. Must
        match the ``log_returns`` passed to the config builder, or the
        cache name will not describe what actually ran.
    memory_limit : int, default=2
        Maximum number of prior scenario frameworks injected into the
        prompt per call.

    Returns
    -------
    ScenarioSchemaAnchoredEnhancedPredictor
    """
    return ScenarioSchemaAnchoredEnhancedPredictor(
        config,
        arima_num_samples=arima_num_samples,
        anchor_log_returns=anchor_log_returns,
        memory_limit=memory_limit,
    )


__all__ = [
    "ScenarioSchemaAnchoredEnhancedPredictor",
    "build_wti_scenario_schema_anchored_enhanced_predictor",
]
