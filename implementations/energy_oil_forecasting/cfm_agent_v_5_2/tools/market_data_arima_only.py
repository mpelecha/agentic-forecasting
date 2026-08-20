"""Market data tool variant for CFM v5.2 that uses ONLY ARIMA in the ensemble.

This is market_data.py with Kalman and LightGBM stripped out — saves hours of
computation while keeping the full LLM agent, policy, and research components.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from energy_oil_forecasting.cfm_agent_v_5_2.models import (
    CfmEnsemblePredictor,
    build_arima_predictor,
)
from energy_oil_forecasting.cfm_agent_v_5_2.tools.market_data import (
    AuthoritativeSuiteTool,
)

if TYPE_CHECKING:
    from aieng.forecasting.data import DataService

    from energy_oil_forecasting.cfm_agent_v_5_2.config import CfmV52Settings


class AuthoritativeSuiteToolArimaOnly(AuthoritativeSuiteTool):
    """AuthoritativeSuiteTool that uses ONLY ARIMA in the ensemble (no Kalman, no LightGBM)."""

    def __init__(
        self,
        data_service: DataService,
        settings: CfmV52Settings,
        covariate_series_ids: list[str] | None = None,
    ) -> None:
        # Call parent __init__ but we'll override the ensemble after
        super().__init__(data_service, settings, covariate_series_ids)

        # Replace the ensemble with ARIMA-only version
        arima_only_predictors = {
            "arima": build_arima_predictor(num_samples=settings.model_num_samples),
        }
        arima_only_weights = {"arima": 1.0}
        self._ensemble = CfmEnsemblePredictor(arima_only_predictors, weights=arima_only_weights)


__all__ = ["AuthoritativeSuiteToolArimaOnly"]
