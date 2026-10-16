"""Weighted linear pool of the return-path components (specification Section 1.6).

Order of operations, per horizon, on already validated component quantiles:
component point (P50) -> weighted pool over the active components with the
configured weights renormalised over survivors -> cumulative-maximum guard
(defensive; a weighted sum of non-decreasing vectors is non-decreasing, so any
change is recorded as ``guard_applied`` and is a build failure) -> ensemble P50
read after the guard. Ensemble construction never reruns a model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from aieng.forecasting.data import ForecastContext
from aieng.forecasting.evaluation.prediction import ContinuousForecast, Prediction
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask
from energy_oil_forecasting.cfm_agent_v_6_0.config import ENSEMBLE_COMPONENTS, CfmV60Settings
from energy_oil_forecasting.cfm_agent_v_6_0.models.paths import (
    MEDIAN_LEVEL,
    QUANTILE_LEVELS,
    ComponentFailure,
    validate_quantiles,
)
from energy_oil_forecasting.cfm_agent_v_6_0.schemas import ComponentDisposition


class ReturnPathEnsemble(Predictor):
    """Weighted pool over the active components; fails closed when none survives."""

    def __init__(self, components: Mapping[str, Predictor], settings: CfmV60Settings) -> None:
        unknown = sorted(set(components) - set(ENSEMBLE_COMPONENTS))
        if unknown:
            raise ValueError(f"unknown ensemble components: {unknown}")
        self._components = {name: components[name] for name in ENSEMBLE_COMPONENTS if name in components}
        if not self._components:
            raise ValueError("at least one component is required")
        self._settings = settings
        self._weights = {name: float(settings.ensemble_weights[name]) for name in self._components}
        self.guard_applied = False

    @property
    def predictor_id(self) -> str:
        """Stable ensemble identifier."""
        return "cfm_v60_return_ensemble"

    @property
    def components(self) -> dict[str, Predictor]:
        """The components in spawn order."""
        return dict(self._components)

    @property
    def configured_weights(self) -> dict[str, float]:
        """Configured weights for the components present."""
        return dict(self._weights)

    def normalized_weights(self, active: Sequence[str]) -> dict[str, float]:
        """Return the configured weights renormalised over the active components, in spawn order."""
        names = [name for name in ENSEMBLE_COMPONENTS if name in active and name in self._weights]
        if not names:
            return {}
        total = sum(self._weights[name] for name in names)
        return {name: self._weights[name] / total for name in names}

    def collect(
        self,
        task: ForecastingTask,
        context: ForecastContext,
    ) -> tuple[dict[str, list[Prediction]], dict[str, ComponentDisposition]]:
        """Run every component; return the successful predictions and every disposition."""
        successful: dict[str, list[Prediction]] = {}
        dispositions: dict[str, ComponentDisposition] = {}
        for name, component in self._components.items():
            try:
                predictions = component.predict(task, context)
                if len(predictions) != len(task.horizons):
                    raise ComponentFailure(
                        "shape_mismatch",
                        f"expected {len(task.horizons)} predictions; got {len(predictions)}",
                    )
                for prediction in predictions:
                    payload = prediction.payload
                    if not isinstance(payload, ContinuousForecast):
                        raise ComponentFailure("non_finite_output", f"{name} did not return a ContinuousForecast")
                    validate_quantiles(payload.quantiles)
                successful[name] = predictions
                dispositions[name] = ComponentDisposition(status="ok")
            except ComponentFailure as exc:
                dispositions[name] = ComponentDisposition(status="error", failure_code=exc.code, reason=exc.message)
            except Exception as exc:  # noqa: BLE001 - any other exception is a library_error disposition
                dispositions[name] = ComponentDisposition(
                    status="error",
                    failure_code=f"library_error:{type(exc).__name__}",
                    reason=str(exc),
                )
        return successful, dispositions

    def check_dispositions(self, dispositions: Mapping[str, ComponentDisposition]) -> list[str]:
        """Return the active component names, or raise when the suite must fail closed."""
        failed = [name for name, disposition in dispositions.items() if disposition.status != "ok"]
        active = [name for name in ENSEMBLE_COMPONENTS if name in dispositions and dispositions[name].status == "ok"]
        if self._settings.require_all_components and failed:
            raise RuntimeError(f"require_all_components is set and components failed: {failed}")
        if not active:
            codes = {name: dispositions[name].failure_code for name in failed}
            raise RuntimeError(f"every positively weighted component failed: {codes}")
        return active

    def combine(
        self,
        task: ForecastingTask,
        context: ForecastContext,
        predictions_by_model: Mapping[str, Sequence[Prediction]],
    ) -> list[Prediction]:
        """Pool already computed component results (Section 1.6 steps 5 to 8)."""
        active = [name for name in ENSEMBLE_COMPONENTS if name in predictions_by_model]
        if not active:
            raise RuntimeError("no positively weighted component produced a forecast")
        weights = self.normalized_weights(active)
        outputs: list[Prediction] = []
        issued_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        guard_applied_any = False
        for index, horizon in enumerate(task.horizons):
            payloads: dict[str, ContinuousForecast] = {}
            for name in active:
                payload = predictions_by_model[name][index].payload
                if not isinstance(payload, ContinuousForecast):
                    raise TypeError(f"{name} did not return a ContinuousForecast")
                validate_quantiles(payload.quantiles)
                if payload.point_forecast != payload.quantiles[MEDIAN_LEVEL]:
                    raise ValueError(f"{name} point forecast does not equal its P50")
                payloads[name] = payload
            component_points = {name: payloads[name].quantiles[MEDIAN_LEVEL] for name in active}
            pooled = {
                level: float(sum(weights[name] * payloads[name].quantiles[level] for name in active))
                for level in QUANTILE_LEVELS
            }
            guarded = dict(pooled)
            running = float("-inf")
            for level in QUANTILE_LEVELS:
                running = max(running, guarded[level])
                guarded[level] = running
            guard_applied = guarded != pooled
            guard_applied_any = guard_applied_any or guard_applied
            point = guarded[MEDIAN_LEVEL]
            forecast_date = predictions_by_model[active[0]][index].forecast_date
            outputs.append(
                Prediction(
                    predictor_id=self.predictor_id,
                    task_id=task.task_id,
                    issued_at=issued_at,
                    as_of=context.as_of,
                    forecast_date=pd.Timestamp(forecast_date).to_pydatetime(),
                    payload=ContinuousForecast(point_forecast=point, quantiles=guarded),
                    metadata={
                        "component_models": active,
                        "normalized_weights": weights,
                        "component_points": component_points,
                        "model_disagreement_std": float(np.std(list(component_points.values()), ddof=0)),
                        "guard_applied": guard_applied,
                        "horizon": int(horizon),
                    },
                )
            )
        self.guard_applied = guard_applied_any
        return outputs

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Run the components and pool the survivors; raise when the suite fails closed."""
        successful, dispositions = self.collect(task, context)
        self.check_dispositions(dispositions)
        return self.combine(task, context, successful)


__all__ = ["ReturnPathEnsemble"]
