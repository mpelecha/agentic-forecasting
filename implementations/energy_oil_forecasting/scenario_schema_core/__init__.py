"""Standalone schema/prompt-builder + price-delta grounding for the scenario schema family.

Everything ``scenario_schema_anchored`` and ``scenario_schema_anchored_enhanced``
need that would otherwise come from ``analyst_agent.agent`` or
``price_deltas`` — vendored here so those two packages depend on nothing in
this project except this package and the shared ``aieng.forecasting``
framework. See ``agent.py`` and ``price_deltas.py`` docstrings for why.
"""

from energy_oil_forecasting.scenario_schema_core.agent import (
    SCENARIO_CONSISTENCY_TOLERANCE,
    WTI_FACTORS_CONTEXT_RETRIEVAL_INSTRUCTION,
    WtiFactor,
    WtiPriceForecastPromptBuilder,
    WtiScenarioCard,
    WtiScenarioForecastOutput,
    compress_history,
)
from energy_oil_forecasting.scenario_schema_core.price_deltas import (
    DEFAULT_PRICE_FLOOR,
    PERCENTILE_LEVELS,
    compute_horizon_delta_percentiles,
)


__all__ = [
    "DEFAULT_PRICE_FLOOR",
    "PERCENTILE_LEVELS",
    "SCENARIO_CONSISTENCY_TOLERANCE",
    "WTI_FACTORS_CONTEXT_RETRIEVAL_INSTRUCTION",
    "WtiFactor",
    "WtiPriceForecastPromptBuilder",
    "WtiScenarioCard",
    "WtiScenarioForecastOutput",
    "compress_history",
    "compute_horizon_delta_percentiles",
]
