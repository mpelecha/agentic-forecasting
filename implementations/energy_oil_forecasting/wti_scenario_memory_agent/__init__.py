"""WTI Scenario Memory Agent — one self-contained package.

ARIMA-anchored, news-grounded scenario forecaster with persistent
cross-origin scenario memory. Depends only on the shared ``aieng.forecasting``
framework (plus ``energy_oil_forecasting.data``/``paths`` in the scripts) —
nothing else in this repo. See README.md.

Quick start::

    from energy_oil_forecasting.wti_scenario_memory_agent import build_agent
    predictor = build_agent()
    predictions = predictor.predict(task, context)
"""

from aieng.forecasting.models import LITE_MODEL
from energy_oil_forecasting.wti_scenario_memory_agent.agent import (
    build_wti_news_scenario_schema_anchored_enhanced_config,
)
from energy_oil_forecasting.wti_scenario_memory_agent.predictor import (
    ScenarioSchemaAnchoredEnhancedPredictor,
    build_wti_scenario_schema_anchored_enhanced_predictor,
)


def build_agent(
    model: str = LITE_MODEL, *, log_returns: bool = False, memory_limit: int = 2
) -> ScenarioSchemaAnchoredEnhancedPredictor:
    """Build the agent with config and predictor guaranteed to agree on ``log_returns``.

    Defaults reproduce the variant scored in the NB04b backtest/eval
    (lite model, level-price ARIMA anchor), so its predictor_id — and
    therefore its prediction cache and memory file — match those results.
    """
    config = build_wti_news_scenario_schema_anchored_enhanced_config(model=model, log_returns=log_returns)
    return build_wti_scenario_schema_anchored_enhanced_predictor(
        config, anchor_log_returns=log_returns, memory_limit=memory_limit
    )


__all__ = [
    "ScenarioSchemaAnchoredEnhancedPredictor",
    "build_agent",
    "build_wti_news_scenario_schema_anchored_enhanced_config",
    "build_wti_scenario_schema_anchored_enhanced_predictor",
]
