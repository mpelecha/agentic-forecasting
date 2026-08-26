"""Delta-governed forecast engine for CFM Agent v5.2.2.

Same transformation shape as
:class:`energy_oil_forecasting.cfm_agent_v_5_2.forecast_engine.PythonForecastEngine`
(center shift + uncertainty-width scaling around the ensemble's own
quantiles, floor/validation guarantees preserved) — but the governor for
*how big* those adjustments are is the empirical distribution of real
historical h-day WTI price deltas, not the ensemble's own self-reported
quantile width:

- **Center shift**: the LLM's rank ``{-2,-1,0,1,2}`` maps to a target
  percentile of the historical delta distribution (see
  ``schemas.RANK_TO_PERCENTILE``); the shift is the gap between that target
  percentile and the historical median — never a number the LLM invents.
- **Uncertainty width**: the categorical wider/narrower multiplier now
  targets a fraction of the *empirical* delta spread (P90-P10 of real
  historical moves) instead of a fixed multiple of the ensemble's own,
  possibly self-referentially narrow, quantile spread.
"""

from __future__ import annotations

import math

from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.task import ForecastingTask
from energy_oil_forecasting.cfm_agent_v_5_2.config import CfmV52Settings
from energy_oil_forecasting.cfm_agent_v_5_2.forecast_engine import PythonForecastEngine
from energy_oil_forecasting.cfm_agent_v_5_2.schemas import ModelHorizonForecast, PolicyDecision
from energy_oil_forecasting.cfm_agent_v_5_2_2_delta_governed.delta_distribution import (
    compute_horizon_delta_percentiles,
)
from energy_oil_forecasting.cfm_agent_v_5_2_2_delta_governed.schemas import (
    RANK_TO_PERCENTILE,
    ForecastTransformationDeltaGoverned,
)


class PythonForecastEngineDeltaGoverned(PythonForecastEngine):
    engine_id = "python_forecast_engine_v522_delta_governed"

    def __init__(self, settings: CfmV52Settings):
        super().__init__(settings)
        self._delta_percentiles: dict[int, dict[int, float]] | None = None

    def prepare(self, context: ForecastContext, task: ForecastingTask) -> None:
        """Compute and stash the empirical delta distribution for this call.

        Must be called once per ``predict()`` invocation, before
        ``transform()`` — see
        :meth:`energy_oil_forecasting.cfm_agent_v_5_2_2_delta_governed.predictor.CfmDeltaGovernedPredictor.predict`.
        """
        self._delta_percentiles = compute_horizon_delta_percentiles(context, task.target_series_id, task.horizons)

    def transform(  # noqa: PLR0912, PLR0915
        self,
        ensemble: ModelHorizonForecast,
        decision: PolicyDecision,
        novelty: str,
        latest_price: float | None,
    ) -> ForecastTransformationDeltaGoverned:  # noqa: PLR0912, PLR0915
        if self._delta_percentiles is None:
            raise RuntimeError("prepare() must be called before transform().")
        percentiles = self._delta_percentiles.get(ensemble.horizon)
        if percentiles is None:
            raise RuntimeError(f"No historical delta distribution prepared for horizon {ensemble.horizon}.")

        q = dict(ensemble.quantiles)
        missing = {0.1, 0.5, 0.9} - set(q)
        if missing:
            raise ValueError(f"required ensemble quantiles missing: {sorted(missing)}")
        if not all(math.isfinite(value) for value in q.values()):
            raise ValueError("non-finite ensemble quantile")
        if latest_price is not None and not math.isfinite(latest_price):
            raise ValueError("latest_price must be finite")
        p50 = q[0.5]
        ensemble_width = q[0.9] - q[0.1]
        if ensemble_width < 0:
            raise ValueError("P10-P90 width must be non-negative")

        # ---- Center shift: rank -> target percentile of REAL historical deltas ----
        rank = decision.center_action
        target_percentile = RANK_TO_PERCENTILE[rank]
        raw = percentiles[target_percentile] - percentiles[50]
        novelty_multiplier = self.s.partly_reflected_multiplier if novelty == "possibly_partly_reflected" else 1.0
        novelty_adjusted = raw * novelty_multiplier
        price_cap = (
            abs(latest_price) * self.s.emergency_center_cap_price_fraction
            if latest_price is not None
            else self.s.emergency_center_cap_usd
        )
        cap = min(self.s.emergency_center_cap_usd, price_cap)
        applied = max(-cap, min(cap, novelty_adjusted)) if decision.eligible or rank == 0 else 0.0

        # ---- Uncertainty width: multiplier now targets the EMPIRICAL delta spread ----
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
        empirical_width = percentiles[90] - percentiles[10]
        target_width = uncertainty_multiplier * empirical_width
        # Rescale the ensemble's own quantile shape to hit the empirically-anchored
        # target width, instead of multiplying the ensemble's own (possibly
        # self-referentially narrow) spread directly. target_width alone isn't
        # safe to divide straight into scale, though: if the ensemble's own
        # width already exceeds a "wider" target (common when the ensemble is
        # itself reacting to real volatility), naively using scale =
        # target_width / ensemble_width silently NARROWS the interval despite
        # the action being "wider" -- and symmetrically a "narrower" action
        # could silently widen it. Floor/ceiling the effective target against
        # the ensemble's own width so the action's direction is never inverted;
        # "unchanged" always means scale == 1.0 exactly, not "rescale toward
        # the empirical width regardless."
        if decision.uncertainty_action == "unchanged":
            # "unchanged" is the LLM declining to form a view on width, not an
            # assertion that the ensemble's own width is right. Treating it as
            # the latter is what made this governor almost inert: the action is
            # "unchanged" on 131 of 167 predictions, so the empirical width was
            # computed and then discarded 78% of the time, and the interval was
            # whatever the ensemble said.
            #
            # Floor it at what real h-day moves justify instead. Never narrower
            # than history; wider is left alone, since an ensemble that has
            # widened for genuine current volatility knows something the
            # unconditional historical distribution does not.
            #
            # Scored offline across every cached predictor on the 2014-2024
            # grid, this floor moved coverage to roughly nominal wherever it
            # fired (Scenario Schema 60.5% -> 79.0%, News Original 52.1% ->
            # 77.2%) for at most +0.06 CRPS, and left already-calibrated
            # predictors untouched to the decimal. It replicated on the
            # 2024-2026 window, which little else in this project has.
            effective_target = max(ensemble_width, empirical_width)
        elif decision.uncertainty_action.endswith("wider"):
            effective_target = max(target_width, ensemble_width)
        else:  # *_narrower
            # Deliberately NOT floored. An explicit narrow request has already
            # passed evidence-tier gating, so the LLM is claiming to know
            # something specific; overriding that would make the action
            # meaningless. Only the "no view" case defers to history.
            effective_target = min(target_width, ensemble_width)
        scale = effective_target / ensemble_width if ensemble_width > 0 else uncertainty_multiplier

        # The old short-circuit returned the ensemble's quantiles untouched
        # whenever rank == 0 and the action was "unchanged" -- which is exactly
        # the case the floor above now needs to act on. Keep the exactness it
        # was there for, but condition it on nothing actually changing rather
        # than on the LLM having no opinion.
        unchanged_exactly = applied == 0.0 and scale == 1.0
        pre_floor = (
            dict(q)
            if unchanged_exactly
            else {key: p50 + applied + scale * (value - p50) for key, value in q.items()}
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
        # Guards the identity case only. This used to key on "the LLM had no
        # view", which the width floor now makes a *changing* case -- a neutral
        # decision can legitimately widen the band to what history justifies.
        # Keyed on nothing having actually changed, the invariant still holds
        # and still catches an accidental transform, without forbidding the fix.
        if unchanged_exactly and not floor_applied and final != q:
            raise ValueError("identity transformation did not exactly reproduce baseline")

        return ForecastTransformationDeltaGoverned(
            horizon=ensemble.horizon,
            original_point_forecast=p50,
            original_quantiles=q,
            p10_p90_width=ensemble_width,
            center_action=rank,
            action_fraction=target_percentile / 100,
            raw_center_adjustment=raw,
            novelty_multiplier=novelty_multiplier,
            novelty_adjusted_amount=novelty_adjusted,
            emergency_cap=cap,
            applied_center_adjustment=applied,
            uncertainty_action=decision.uncertainty_action,
            uncertainty_multiplier=scale,
            pre_floor_quantiles=pre_floor,
            final_point_forecast=final[0.5],
            final_quantiles=final,
            floor_applied=floor_applied,
            warnings=warnings,
            historical_delta_p10=percentiles[10],
            historical_delta_p50=percentiles[50],
            historical_delta_p90=percentiles[90],
            target_percentile=target_percentile,
        )


__all__ = ["PythonForecastEngineDeltaGoverned"]
