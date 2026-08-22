"""Delta-governed schema extensions for CFM Agent v5.2.2.

Extends the CFM v5.2 schemas with a discrete rank encoding for
``center_action`` in place of the named up/down/magnitude category — see
:mod:`energy_oil_forecasting.cfm_agent_v_5_2_2_delta_governed.forecast_engine`
for how the rank is converted to a dollar adjustment.
"""

from __future__ import annotations

from typing import Literal

from energy_oil_forecasting.cfm_agent_v_5_2.schemas import (
    ForecastTransformation,
    HorizonAction,
    PolicyDecision,
)


CenterActionRank = Literal[-2, -1, 0, 1, 2]

# Each rank maps to a target percentile of the empirical h-day price-delta
# distribution (see delta_distribution.py). 0 -> the median historical move
# (no directional view beyond typical drift); +/-2 -> the 90th/10th
# percentile of real historical moves at that horizon.
RANK_TO_PERCENTILE: dict[int, int] = {-2: 10, -1: 25, 0: 50, 1: 75, 2: 90}


class HorizonActionDeltaGoverned(HorizonAction):
    center_action: CenterActionRank = 0


class PolicyDecisionDeltaGoverned(PolicyDecision):
    center_action: CenterActionRank


class ForecastTransformationDeltaGoverned(ForecastTransformation):
    center_action: CenterActionRank
    historical_delta_p10: float
    historical_delta_p50: float
    historical_delta_p90: float
    target_percentile: int


__all__ = [
    "CenterActionRank",
    "RANK_TO_PERCENTILE",
    "HorizonActionDeltaGoverned",
    "PolicyDecisionDeltaGoverned",
    "ForecastTransformationDeltaGoverned",
]
