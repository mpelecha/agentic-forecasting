"""Ensemble combination tests without fitting external numerical libraries."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from aieng.forecasting.evaluation import ContinuousForecast, Prediction, Predictor
from aieng.forecasting.evaluation.task import ForecastingTask
from energy_oil_forecasting.cfm_agent_v_2_0.models import CfmEnsemblePredictor


class ConstantPredictor(Predictor):
    def __init__(self, name: str, value: float) -> None:
        self._name = name
        self._value = value

    @property
    def predictor_id(self) -> str:
        return self._name

    def predict(self, task, context):
        issued = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        offset = pd.tseries.frequencies.to_offset(task.frequency)
        return [
            Prediction(
                predictor_id=self.predictor_id,
                task_id=task.task_id,
                issued_at=issued,
                as_of=context.as_of,
                forecast_date=(pd.Timestamp(context.as_of) + offset * h).to_pydatetime(),
                payload=ContinuousForecast(
                    point_forecast=self._value,
                    quantiles={0.1: self._value - 1, 0.5: self._value, 0.9: self._value + 1},
                ),
            )
            for h in task.horizons
        ]


def test_weighted_ensemble(synthetic_service) -> None:
    task = ForecastingTask(
        task_id="test",
        target_series_id="wti_crude_oil_price",
        horizons=[1, 5],
        frequency="B",
        description="test",
    )
    context = synthetic_service.context(pd.Timestamp("2021-06-01").to_pydatetime())
    ensemble = CfmEnsemblePredictor(
        {"a": ConstantPredictor("a", 10.0), "b": ConstantPredictor("b", 20.0)},
        weights={"a": 0.25, "b": 0.75},
    )
    predictions = ensemble.predict(task, context)
    assert len(predictions) == 2
    for prediction in predictions:
        assert prediction.payload.point_forecast == 17.5
        assert prediction.payload.quantiles[0.5] == 17.5
