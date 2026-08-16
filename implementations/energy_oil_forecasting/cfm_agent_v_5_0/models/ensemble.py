"""Deterministic quantile ensemble for CFM model forecasts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone

import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import (
    ContinuousForecast,
    Prediction,
)
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask


class CfmEnsemblePredictor(Predictor):
    """Weighted linear pool over successful continuous predictors.

    Quantiles are averaged level-by-level using normalized positive weights.
    A cumulative maximum is applied afterward to guarantee non-crossing output.
    Model failures are tolerated when at least one component succeeds.
    """

    def __init__(
        self,
        predictors: Mapping[str, Predictor],
        weights: Mapping[str, float] | None = None,
    ) -> None:
        if not predictors:
            raise ValueError("predictors cannot be empty.")
        self._predictors = dict(predictors)
        raw_weights = dict(weights or dict.fromkeys(predictors, 1.0))
        self._weights = {
            name: float(raw_weights.get(name, 0.0)) for name in predictors if float(raw_weights.get(name, 0.0)) > 0
        }
        if not self._weights:
            raise ValueError("at least one ensemble weight must be positive.")

    def normalized_weights(self, active_models: list[str]) -> dict[str, float]:
        """Return configured weights renormalized over successful models."""
        active = [name for name in active_models if name in self._weights]
        if not active:
            return {}
        total = sum(self._weights[name] for name in active)
        return {name: self._weights[name] / total for name in active}

    @property
    def predictor_id(self) -> str:
        """Return a stable identifier."""
        return "cfm_arima_kalman_lightgbm_ensemble"

    def collect(
        self,
        task: ForecastingTask,
        context: ForecastContext,
    ) -> tuple[dict[str, list[Prediction]], dict[str, str]]:
        """Run every component and return successful predictions plus failures."""
        successful: dict[str, list[Prediction]] = {}
        failures: dict[str, str] = {}
        for name, predictor in self._predictors.items():
            try:
                predictions = predictor.predict(task, context)
                if len(predictions) != len(task.horizons):
                    raise ValueError(f"expected {len(task.horizons)} predictions; got {len(predictions)}")
                successful[name] = predictions
            except Exception as exc:  # noqa: BLE001 - failures are model evidence
                failures[name] = f"{type(exc).__name__}: {exc}"
        return successful, failures

    def combine(
        self,
        task: ForecastingTask,
        context: ForecastContext,
        predictions_by_model: Mapping[str, Sequence[Prediction]],
    ) -> list[Prediction]:
        """Combine already-computed component results without rerunning models."""
        active = [name for name in predictions_by_model if name in self._weights]
        if not active:
            raise RuntimeError("no positively weighted model produced a forecast.")
        total_weight = sum(self._weights[name] for name in active)
        normalized = {name: self._weights[name] / total_weight for name in active}
        outputs: list[Prediction] = []
        issued_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        for index, horizon in enumerate(task.horizons):
            payloads = {}
            for name in active:
                payload = predictions_by_model[name][index].payload
                if not isinstance(payload, ContinuousForecast):
                    raise TypeError(f"{name} did not return ContinuousForecast.")
                payloads[name] = payload
            common_quantiles = set.intersection(*(set(payload.quantiles) for payload in payloads.values()))
            if 0.5 not in common_quantiles:
                raise ValueError("component forecasts do not share the median quantile.")
            quantiles = {
                q: sum(normalized[name] * payloads[name].quantiles[q] for name in active)
                for q in sorted(common_quantiles)
            }
            previous = float("-inf")
            for q in sorted(quantiles):
                quantiles[q] = max(previous, quantiles[q])
                previous = quantiles[q]
            forecast_date = predictions_by_model[active[0]][index].forecast_date
            outputs.append(
                Prediction(
                    predictor_id=self.predictor_id,
                    task_id=task.task_id,
                    issued_at=issued_at,
                    as_of=context.as_of,
                    forecast_date=pd.Timestamp(forecast_date).to_pydatetime(),
                    payload=ContinuousForecast(
                        point_forecast=quantiles[0.5],
                        quantiles=quantiles,
                    ),
                    metadata={
                        "component_models": active,
                        "normalized_weights": normalized,
                        "horizon": horizon,
                    },
                )
            )
        return outputs

    def predict(
        self,
        task: ForecastingTask,
        context: ForecastContext,
    ) -> list[Prediction]:
        """Run components and return the weighted ensemble."""
        successful, failures = self.collect(task, context)
        if not successful:
            raise RuntimeError(f"all component models failed: {failures}")
        return self.combine(task, context, successful)


__all__ = ["CfmEnsemblePredictor"]
