"""CFM Agent v2.0 with rich attribution and identifier-only prompts."""

from energy_oil_forecasting.cfm_agent_v_2_0.agent import (
    build_cfm_agent_config,
    build_cfm_agent_predictor,
)
from energy_oil_forecasting.cfm_agent_v_2_0.config import (
    AGENT_NAME,
    DEFAULT_SETTINGS,
    CfmAgentSettings,
)
from energy_oil_forecasting.cfm_agent_v_2_0.models import (
    CfmEnsemblePredictor,
    build_arima_predictor,
    build_kalman_predictor,
    build_lightgbm_predictor,
)
from energy_oil_forecasting.cfm_agent_v_2_0.outputs import CfmRichForecastOutput
from energy_oil_forecasting.cfm_agent_v_2_0.predictor import CfmRichAgentPredictor
from energy_oil_forecasting.cfm_agent_v_2_0.tools import MarketDataTool


__all__ = [
    "AGENT_NAME",
    "CfmAgentSettings",
    "CfmEnsemblePredictor",
    "CfmRichAgentPredictor",
    "CfmRichForecastOutput",
    "DEFAULT_SETTINGS",
    "MarketDataTool",
    "build_arima_predictor",
    "build_cfm_agent_config",
    "build_cfm_agent_predictor",
    "build_kalman_predictor",
    "build_lightgbm_predictor",
]
