"""Policy protocol for v3 forecast transformations."""

from __future__ import annotations

from typing import Protocol

from energy_oil_forecasting.cfm_agent_v_3_4.outputs import CfmContextAssessmentOutput
from energy_oil_forecasting.cfm_agent_v_3_4.schemas import ModelHorizonForecast, PolicyDecision


class ForecastPolicy(Protocol):
    @property
    def policy_id(self) -> str: ...

    def apply(
        self,
        *,
        ensemble: ModelHorizonForecast,
        assessment: CfmContextAssessmentOutput,
        horizon: int,
        horizons: list[int],
        latest_price: float | None,
        cutoff_date: str,
    ) -> PolicyDecision: ...


__all__ = ["ForecastPolicy"]
