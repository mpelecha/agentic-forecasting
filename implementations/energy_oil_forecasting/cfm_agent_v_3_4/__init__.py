"""CFM Agent v3.4: contextual LLM actions with Python-owned forecast arithmetic."""

from energy_oil_forecasting.cfm_agent_v_3_4.agent import (
    build_cfm_agent_config,
    build_cfm_agent_predictor,
)
from energy_oil_forecasting.cfm_agent_v_3_4.config import (
    AGENT_NAME,
    DEFAULT_SETTINGS,
    CfmV34Settings,
)
from energy_oil_forecasting.cfm_agent_v_3_4.models import (
    CfmEnsemblePredictor,
    build_arima_predictor,
    build_kalman_predictor,
    build_lightgbm_predictor,
)
from energy_oil_forecasting.cfm_agent_v_3_4.outputs import CfmContextAssessmentOutput
from energy_oil_forecasting.cfm_agent_v_3_4.policy import (
    ConstrainedActionPolicy,
    EnsembleLockedPolicy,
)
from energy_oil_forecasting.cfm_agent_v_3_4.predictor import CfmV34Predictor
from energy_oil_forecasting.cfm_agent_v_3_4.tools import AuthoritativeSuiteTool, MarketDataTool


__all__ = [
    "AGENT_NAME",
    "CfmContextAssessmentOutput",
    "CfmEnsemblePredictor",
    "CfmV34Predictor",
    "CfmV34Settings",
    "ConstrainedActionPolicy",
    "DEFAULT_SETTINGS",
    "EnsembleLockedPolicy",
    "AuthoritativeSuiteTool",
    "MarketDataTool",
    "build_arima_predictor",
    "build_cfm_agent_config",
    "build_cfm_agent_predictor",
    "build_kalman_predictor",
    "build_lightgbm_predictor",
]
