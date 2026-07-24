"""WTI crude oil analyst agent module.

Exports the :class:`AgentConfig` factories, prompt builder, and predictor
convenience factory for the energy/oil reference implementation.
"""

from energy_oil_forecasting.analyst_agent.agent import (
    WtiFactor,
    WtiPriceForecastPromptBuilder,
    WtiScenarioCard,
    WtiScenarioForecastOutput,
    build_wti_agent_predictor,
    build_wti_basic_config,
    build_wti_code_exec_config,
    build_wti_multitask_news_config,
    build_wti_news_config,
    build_wti_news_contrarian_config,
    build_wti_news_factors_v2_config,
    build_wti_news_scenario_schema_config,
    build_wti_scenario_schema_predictor,
    build_wti_tool_config,
    compress_history,
)


__all__ = [
    "WtiFactor",
    "WtiPriceForecastPromptBuilder",
    "WtiScenarioCard",
    "WtiScenarioForecastOutput",
    "build_wti_agent_predictor",
    "build_wti_basic_config",
    "build_wti_code_exec_config",
    "build_wti_multitask_news_config",
    "build_wti_news_config",
    "build_wti_news_contrarian_config",
    "build_wti_news_factors_v2_config",
    "build_wti_news_scenario_schema_config",
    "build_wti_scenario_schema_predictor",
    "build_wti_tool_config",
    "compress_history",
]