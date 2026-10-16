"""Historical forecast transformations (specification Section 3).

The LLM proposes categorical actions; this engine selects the historical
percentile, performs every arithmetic step, and records enough for a replay from
the transformation record and the settings alone.

Order: preconditions -> exact neutral (literal copy) -> profile validity ->
centre (percentile, directional guard, novelty multiplier, emergency cap) ->
coverage-shaped uncertainty ladder -> final floor -> finite and ordered check
-> point = P50.
"""

from __future__ import annotations

import math

from energy_oil_forecasting.cfm_agent_v_6_0.config import CfmV60Settings
from energy_oil_forecasting.cfm_agent_v_6_0.schemas import (
    ForecastTransformation,
    HistoricalReturnProfile,
    ModelHorizonForecast,
    PolicyDecision,
)


P10_LEVEL = 0.1  # structural: the lower bound of the ensemble's 80% band
P50_LEVEL = 0.5  # structural: the median quantile level
P90_LEVEL = 0.9  # structural: the upper bound of the ensemble's 80% band
NEUTRAL_CENTER = "no_change"
NEUTRAL_UNCERTAINTY = "unchanged"


def _direction(action: str) -> float:
    if action.endswith("_down"):
        return -1.0
    if action.endswith("_up"):
        return 1.0
    return 0.0


class PythonForecastEngine:
    """Python-owned transformation of the authoritative ensemble (Section 3)."""

    engine_id = "python_forecast_engine_v60"

    def __init__(self, settings: CfmV60Settings) -> None:
        self.s = settings

    # ------------------------------------------------------------ helpers --

    def _profile_problem(  # noqa: PLR0911 - one check per profile field
        self, profile: HistoricalReturnProfile | None, center_action: str
    ) -> str | None:
        """Return why the profile cannot drive a transformation, or ``None`` when it can."""
        if profile is None:
            return "no historical return profile was supplied"
        if not profile.eligible:
            return profile.ineligibility_reason or (
                f"{profile.n_eligible} eligible historical returns, below the minimum {profile.min_required}"
            )
        if profile.median is None or not math.isfinite(profile.median):
            return "historical median is missing or not finite"
        if center_action != NEUTRAL_CENTER:
            percentile = self.s.center_percentiles[center_action]
            value = profile.return_percentiles.get(percentile)
            if value is None or not math.isfinite(value):
                return f"historical return percentile P{percentile} is missing or not finite"
        anchor = self.s.uncertainty_anchor_coverage
        anchor_band = profile.absolute_deviation_bands.get(anchor)
        if anchor_band is None or not math.isfinite(anchor_band) or anchor_band <= 0:
            return f"historical anchor band at {anchor}% coverage is missing, not finite, or zero"
        for coverage in self.s.uncertainty_coverage.values():
            band = profile.absolute_deviation_bands.get(coverage)
            if band is None or not math.isfinite(band):
                return f"historical band at {coverage}% coverage is missing or not finite"
        return None

    def _emergency_cap(self, latest_price: float | None) -> float:
        cap_usd = self.s.emergency_center_cap_usd
        if latest_price is None:
            return cap_usd
        return min(cap_usd, self.s.emergency_center_cap_price_fraction * abs(latest_price))

    # ---------------------------------------------------------- transform --

    def transform(  # noqa: PLR0912, PLR0915 - the normative Section 3 sequence, kept in one place
        self,
        ensemble: ModelHorizonForecast,
        decision: PolicyDecision,
        novelty: str,
        latest_price: float | None,
        profile: HistoricalReturnProfile | None = None,
    ) -> ForecastTransformation:
        """Apply the decided centre and uncertainty actions to the ensemble quantiles."""
        q = dict(ensemble.quantiles)
        missing = {P10_LEVEL, P50_LEVEL, P90_LEVEL} - set(q)
        if missing:
            raise ValueError(f"required ensemble quantiles missing: {sorted(missing)}")
        if not all(math.isfinite(value) for value in q.values()):
            raise ValueError("non-finite ensemble quantile")
        if latest_price is not None and not math.isfinite(latest_price):
            raise ValueError("latest_price must be finite")
        if novelty not in self.s.novelty_multipliers:
            raise ValueError(f"unknown novelty classification {novelty!r}")
        p50 = q[P50_LEVEL]
        width = q[P90_LEVEL] - q[P10_LEVEL]
        if width < 0:
            raise ValueError("P10-P90 width must be non-negative")
        center_action = decision.center_action
        uncertainty_action = decision.uncertainty_action
        novelty_multiplier = float(self.s.novelty_multipliers[novelty])
        cap = self._emergency_cap(latest_price)
        floor = self.s.final_price_floor
        warnings: list[str] = []

        requested_neutral = center_action == NEUTRAL_CENTER and uncertainty_action == NEUTRAL_UNCERTAINTY
        neutral_reason: str | None = None
        if requested_neutral:
            neutral_reason = "requested: center_action no_change and uncertainty_action unchanged"
        elif width <= 0:
            neutral_reason = "baseline P10-P90 width is zero; both levers forced neutral"
        else:
            problem = self._profile_problem(profile, center_action)
            if problem is not None:
                neutral_reason = f"historical profile unusable: {problem}; both levers forced neutral"
        neutral = neutral_reason is not None
        if neutral and not requested_neutral:
            warnings.append(neutral_reason)

        record: dict[str, object] = {
            "engine_id": self.engine_id,
            "horizon": ensemble.horizon,
            "original_point_forecast": p50,
            "original_quantiles": q,
            "p10_p90_width": width,
            "latest_price": latest_price,
            "neutral": neutral,
            "neutral_reason": neutral_reason,
            "center_action": center_action,
            "novelty": novelty,
            "novelty_multiplier": novelty_multiplier,
            "emergency_cap": cap,
            "uncertainty_action": uncertainty_action,
            "anchor_coverage": self.s.uncertainty_anchor_coverage,
            "final_price_floor": floor,
            "historical_profile": profile,
        }

        if neutral:
            pre_floor = dict(q)
            final = {level: max(floor, value) for level, value in pre_floor.items()}
            floor_applied = final != pre_floor
            if floor_applied:
                warnings.append("final floor raised a neutral quantile")
            if final != q:
                raise ValueError("neutral transformation did not exactly reproduce the baseline")
            self._check_output(final)
            return ForecastTransformation(
                **record,
                selected_center_percentile=None,
                selected_center_value=None,
                center_sign_guard_applied=False,
                raw_center_adjustment=0.0,
                novelty_adjusted_amount=0.0,
                applied_center_adjustment=0.0,
                adjusted_p50=p50,
                uncertainty_coverage_target=self.s.uncertainty_coverage[uncertainty_action],
                anchor_band=None,
                target_band=None,
                uncertainty_multiplier=1.0,
                uncertainty_clamp_applied=False,
                pre_floor_quantiles=pre_floor,
                floor_applied=floor_applied,
                final_point_forecast=final[P50_LEVEL],
                final_quantiles=final,
                warnings=warnings,
            )

        assert profile is not None
        # -- centre (Section 3.1) -------------------------------------------
        direction = _direction(center_action)
        selected_percentile: int | None = None
        selected_value: float | None = None
        guarded_return = 0.0
        sign_guard = False
        if center_action != NEUTRAL_CENTER:
            selected_percentile = self.s.center_percentiles[center_action]
            selected_value = float(profile.return_percentiles[selected_percentile])
            if direction * selected_value < 0:
                sign_guard = True
                warnings.append(
                    f"directional guard: {center_action} selected P{selected_percentile} = {selected_value:.6f} "
                    "with the wrong sign; centre adjustment set to zero"
                )
            else:
                guarded_return = selected_value
        raw = p50 * guarded_return
        novelty_adjusted = raw * novelty_multiplier
        applied = max(-cap, min(cap, novelty_adjusted))
        if applied != novelty_adjusted:
            warnings.append(f"emergency cap {cap:.6f} limited the centre adjustment {novelty_adjusted:.6f}")
        if not decision.eligible and center_action != NEUTRAL_CENTER:
            applied = 0.0
            warnings.append("policy decision ineligible; centre adjustment set to zero")
        adjusted_p50 = p50 + applied

        # -- uncertainty (Section 3.3) --------------------------------------
        coverage = self.s.uncertainty_coverage[uncertainty_action]
        anchor = self.s.uncertainty_anchor_coverage
        anchor_band = float(profile.absolute_deviation_bands[anchor])
        target_band = float(profile.absolute_deviation_bands[coverage])
        clamp_applied = False
        if uncertainty_action == NEUTRAL_UNCERTAINTY:
            multiplier = 1.0
        else:
            multiplier = target_band / anchor_band
            wider = uncertainty_action.endswith("_wider")
            if (wider and multiplier < 1.0) or (not wider and multiplier > 1.0):
                raise ValueError(
                    f"uncertainty ladder violated: {uncertainty_action} gave multiplier {multiplier!r} "
                    f"(target band {target_band!r}, anchor band {anchor_band!r})"
                )
            if multiplier == 1.0:
                clamp_applied = True
                warnings.append(
                    f"uncertainty ladder tie: {uncertainty_action} band equals the anchor band; multiplier 1.0"
                )
        pre_floor = {level: adjusted_p50 + multiplier * (value - p50) for level, value in q.items()}

        # -- final floor and checks (Section 1.6 step 10) -------------------
        final = {level: max(floor, value) for level, value in pre_floor.items()}
        floor_applied = final != pre_floor
        if floor_applied:
            warnings.append(f"final floor {floor} applied")
        self._check_output(final)
        return ForecastTransformation(
            **record,
            selected_center_percentile=selected_percentile,
            selected_center_value=selected_value,
            center_sign_guard_applied=sign_guard,
            raw_center_adjustment=raw,
            novelty_adjusted_amount=novelty_adjusted,
            applied_center_adjustment=applied,
            adjusted_p50=adjusted_p50,
            uncertainty_coverage_target=coverage,
            anchor_band=anchor_band,
            target_band=target_band,
            uncertainty_multiplier=multiplier,
            uncertainty_clamp_applied=clamp_applied,
            pre_floor_quantiles=pre_floor,
            floor_applied=floor_applied,
            final_point_forecast=final[P50_LEVEL],
            final_quantiles=final,
            warnings=warnings,
        )

    @staticmethod
    def _check_output(final: dict[float, float]) -> None:
        values = [final[level] for level in sorted(final)]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("non-finite forecast")
        if values != sorted(values):
            raise ValueError("quantile crossing")


__all__ = ["PythonForecastEngine"]
