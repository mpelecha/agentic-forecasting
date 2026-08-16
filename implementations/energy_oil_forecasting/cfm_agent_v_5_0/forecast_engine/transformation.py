"""Locked Component #10 transformations with v5.0 correctness controls."""

import math

from energy_oil_forecasting.cfm_agent_v_5_0.config import CfmV50Settings
from energy_oil_forecasting.cfm_agent_v_5_0.schemas import (
    ForecastTransformation,
    ModelHorizonForecast,
    PolicyDecision,
)


class PythonForecastEngine:
    engine_id = "python_forecast_engine_v50"

    def __init__(self, settings: CfmV50Settings):
        self.s = settings

    def transform(
        self,
        ensemble: ModelHorizonForecast,
        decision: PolicyDecision,
        novelty: str,
        latest_price: float | None,
    ) -> ForecastTransformation:
        q = dict(ensemble.quantiles)
        missing = {0.1, 0.5, 0.9} - set(q)
        if missing:
            raise ValueError(f"required ensemble quantiles missing: {sorted(missing)}")
        if not all(math.isfinite(value) for value in q.values()):
            raise ValueError("non-finite ensemble quantile")
        if latest_price is not None and not math.isfinite(latest_price):
            raise ValueError("latest_price must be finite")
        p50 = q[0.5]
        width = q[0.9] - q[0.1]
        if width < 0:
            raise ValueError("P10-P90 width must be non-negative")
        action = decision.center_action
        level = (
            0
            if action == "no_change"
            else 1
            if action.startswith("small")
            else 2
            if action.startswith("moderate")
            else 3
        )
        fraction = {
            0: 0.0,
            1: self.s.small_action_width_fraction,
            2: self.s.moderate_action_width_fraction,
            3: self.s.large_action_width_fraction,
        }[level]
        direction = -1.0 if action.endswith("down") else 1.0 if action.endswith("up") else 0.0
        raw = direction * fraction * width
        novelty_multiplier = self.s.partly_reflected_multiplier if novelty == "possibly_partly_reflected" else 1.0
        novelty_adjusted = raw * novelty_multiplier
        price_cap = (
            abs(latest_price) * self.s.emergency_center_cap_price_fraction
            if latest_price is not None
            else self.s.emergency_center_cap_usd
        )
        cap = min(self.s.emergency_center_cap_usd, price_cap)
        applied = max(-cap, min(cap, novelty_adjusted)) if decision.eligible or action == "no_change" else 0.0
        multipliers = {
            "unchanged": 1.0,
            "small_wider": self.s.small_wider_multiplier,
            "small_narrower": self.s.small_narrower_multiplier,
            "moderately_wider": self.s.moderately_wider_multiplier,
            "moderately_narrower": self.s.moderately_narrower_multiplier,
            "substantially_wider": self.s.substantially_wider_multiplier,
            "substantially_narrower": self.s.substantially_narrower_multiplier,
        }
        uncertainty_multiplier = multipliers[decision.uncertainty_action]
        neutral = action == "no_change" and decision.uncertainty_action == "unchanged"
        pre_floor = (
            dict(q)
            if neutral
            else {key: p50 + applied + uncertainty_multiplier * (value - p50) for key, value in q.items()}
        )
        warnings: list[str] = []
        final = dict(pre_floor)
        floor_applied = False
        if any(value < 0 for value in pre_floor.values()):
            warnings.append("Negative WTI quantile produced.")
            if self.s.nonnegative_price_floor is not None:
                final = {key: max(self.s.nonnegative_price_floor, value) for key, value in pre_floor.items()}
                floor_applied = True
                warnings.append(f"Applied explicit floor {self.s.nonnegative_price_floor}.")
        values = [final[key] for key in sorted(final)]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("non-finite forecast")
        if values != sorted(values):
            raise ValueError("quantile crossing")
        if neutral and final != q:
            raise ValueError("neutral transformation did not exactly reproduce baseline")
        return ForecastTransformation(
            horizon=ensemble.horizon,
            original_point_forecast=p50,
            original_quantiles=q,
            p10_p90_width=width,
            center_action=action,
            action_fraction=fraction,
            raw_center_adjustment=raw,
            novelty_multiplier=novelty_multiplier,
            novelty_adjusted_amount=novelty_adjusted,
            emergency_cap=cap,
            applied_center_adjustment=applied,
            uncertainty_action=decision.uncertainty_action,
            uncertainty_multiplier=uncertainty_multiplier,
            pre_floor_quantiles=pre_floor,
            final_point_forecast=final[0.5],
            final_quantiles=final,
            floor_applied=floor_applied,
            warnings=warnings,
        )
