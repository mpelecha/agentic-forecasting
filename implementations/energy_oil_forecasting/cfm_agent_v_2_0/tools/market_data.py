"""Cutoff-safe market data, model forecasts, and model training audit."""

from __future__ import annotations

import threading
from datetime import datetime

import numpy as np
import pandas as pd
from aieng.forecasting.data import DataService, ForecastContext
from aieng.forecasting.evaluation.prediction import ContinuousForecast, Prediction
from aieng.forecasting.evaluation.task import ForecastingTask
from energy_oil_forecasting.cfm_agent_v_2_0.config import (
    DEFAULT_SETTINGS,
    CfmAgentSettings,
)
from energy_oil_forecasting.cfm_agent_v_2_0.models import (
    CfmEnsemblePredictor,
    build_arima_predictor,
    build_kalman_predictor,
    build_lightgbm_predictor,
)
from energy_oil_forecasting.cfm_agent_v_2_0.schemas import (
    MarketDataToolResult,
    ModelForecastResult,
    ModelHorizonForecast,
    ModelSuiteResult,
    ModelTrainingData,
    SeriesObservation,
    SeriesSnapshot,
)


try:
    from google.adk.tools.function_tool import FunctionTool
except ModuleNotFoundError as exc:
    raise ImportError("The market-data tool requires the agentic extra.") from exc


_ALLOWED_OPERATIONS = {
    "list_series",
    "get_series",
    "run_models",
    "get_series_and_run_models",
}


class MarketDataTool:
    """Query registered series and run the auditable three-model suite."""

    def __init__(
        self,
        data_service: DataService,
        *,
        settings: CfmAgentSettings = DEFAULT_SETTINGS,
        covariate_series_ids: list[str] | None = None,
    ) -> None:
        self._data_service = data_service
        self._settings = settings
        self._lock = threading.Lock()
        self._last_result: MarketDataToolResult | None = None
        available = set(data_service.series_ids)
        self._covariate_series_ids = [series_id for series_id in (covariate_series_ids or []) if series_id in available]
        predictors = {
            "arima": build_arima_predictor(num_samples=settings.model_num_samples),
            "kalman": build_kalman_predictor(
                num_samples=settings.model_num_samples,
                dim_x=settings.kalman_dim_x,
            ),
            "lightgbm": build_lightgbm_predictor(
                covariate_series_ids=self._covariate_series_ids or None,
                lags=settings.lightgbm_lags,
                lags_past_covariates=settings.lightgbm_covariate_lags,
                num_samples=settings.model_num_samples,
            ),
        }
        self._ensemble = CfmEnsemblePredictor(
            predictors,
            weights=settings.ensemble_weights,
        )

    @property
    def last_result(self) -> MarketDataToolResult | None:
        """Return a defensive copy of the most recent tool result."""
        with self._lock:
            return self._last_result.model_copy(deep=True) if self._last_result else None

    def as_function_tool(self) -> FunctionTool:
        """Return the ready-to-register ADK tool."""
        return FunctionTool(func=self.query_market_data)

    def query_market_data(
        self,
        operation: str,
        cutoff_date: str,
        series_ids: list[str],
        target_series_id: str = "wti_crude_oil_price",
        horizons: list[int] | None = None,
        frequency: str = "B",
        lookback: int = 260,
    ) -> str:
        """Get cutoff-safe data and optionally run ARIMA, Kalman, LightGBM.

        Args:
            operation: ``list_series``, ``get_series``, ``run_models``, or
                ``get_series_and_run_models``.
            cutoff_date: Information cutoff in YYYY-MM-DD format.
            series_ids: Registered series to return for data operations.
            target_series_id: Registered series forecast by the model suite.
            horizons: Positive forecast steps, for example [5, 10, 21].
            frequency: Pandas forecast frequency such as "B".
            lookback: Maximum observations returned per requested series.

        Returns
        -------
            JSON with bounded histories, component and ensemble forecasts,
            configured and active weights, model disagreement, and per-model
            training-data usage. ``effective_training_examples_estimate`` is a
            transparent design-matrix estimate because Darts does not expose its
            final internal row count.
        """
        operation = operation.strip().lower()
        if operation not in _ALLOWED_OPERATIONS:
            return self._store_error(
                operation,
                cutoff_date,
                f"operation must be one of {sorted(_ALLOWED_OPERATIONS)}.",
            )
        try:
            as_of = datetime.strptime(cutoff_date, "%Y-%m-%d")
        except ValueError:
            return self._store_error(operation, cutoff_date, "cutoff_date must be YYYY-MM-DD.")
        try:
            clean_lookback = max(
                1,
                min(int(lookback), self._settings.max_data_rows_per_series),
            )
            clean_horizons = sorted({int(h) for h in (horizons or [])})
        except (TypeError, ValueError):
            return self._store_error(operation, cutoff_date, "lookback and horizons must be integers.")
        if operation in {"run_models", "get_series_and_run_models"} and (
            not clean_horizons or any(h < 1 for h in clean_horizons)
        ):
            return self._store_error(
                operation,
                cutoff_date,
                "model operations require positive horizons.",
            )

        context = self._data_service.context(as_of=as_of)
        snapshots: list[SeriesSnapshot] = []
        warnings: list[str] = []
        if operation in {"get_series", "get_series_and_run_models"}:
            for series_id in dict.fromkeys(series_ids):
                try:
                    snapshots.append(self._snapshot(context, series_id, clean_lookback))
                except KeyError:
                    warnings.append(f"Series {series_id!r} is not registered and was omitted.")
        suite = None
        if operation in {"run_models", "get_series_and_run_models"}:
            try:
                suite = self._run_model_suite(
                    context=context,
                    target_series_id=target_series_id,
                    horizons=clean_horizons,
                    frequency=frequency,
                    cutoff_date=cutoff_date,
                )
            except KeyError as exc:
                return self._store_error(operation, cutoff_date, str(exc))
            if suite.failed_models:
                warnings.append("One or more models failed; ensemble weights were renormalized over successful models.")
        result = MarketDataToolResult(
            status="partial" if warnings or (suite and suite.failed_models) else "ok",
            operation=operation,
            cutoff_date=cutoff_date,
            available_series=self._data_service.series_ids,
            series=snapshots,
            model_suite=suite,
            warnings=warnings,
        )
        return self._store(result)

    def _store(self, result: MarketDataToolResult) -> str:
        with self._lock:
            self._last_result = result.model_copy(deep=True)
        return result.model_dump_json(indent=2)

    def _store_error(self, operation: str, cutoff_date: str, message: str) -> str:
        return self._store(
            MarketDataToolResult(
                status="error",
                operation=operation,
                cutoff_date=cutoff_date,
                error=message,
            )
        )

    @staticmethod
    def _snapshot(context: ForecastContext, series_id: str, lookback: int) -> SeriesSnapshot:
        metadata = context.get_metadata(series_id)
        history = context.get_series(series_id).copy()
        history["timestamp"] = pd.to_datetime(history["timestamp"])
        selected = history.tail(lookback)
        return SeriesSnapshot(
            series_id=series_id,
            description=metadata.description,
            source=metadata.source,
            units=metadata.units,
            frequency=metadata.frequency,
            n_observations_at_cutoff=len(history),
            returned_observations=len(selected),
            first_available_date=(
                str(pd.Timestamp(history["timestamp"].iloc[0]).date()) if not history.empty else None
            ),
            last_available_date=(
                str(pd.Timestamp(history["timestamp"].iloc[-1]).date()) if not history.empty else None
            ),
            last_value=float(history["value"].iloc[-1]) if not history.empty else None,
            observations=[
                SeriesObservation(
                    date=str(pd.Timestamp(row.timestamp).date()),
                    value=float(row.value),
                )
                for row in selected.itertuples(index=False)
            ],
        )

    def _training_audit(
        self,
        context: ForecastContext,
        target_series_id: str,
        max_horizon: int,
    ) -> dict[str, ModelTrainingData]:
        target = context.get_series(target_series_id).copy()
        target["timestamp"] = pd.to_datetime(target["timestamp"])
        target_count = len(target)
        first_date = str(target["timestamp"].iloc[0].date()) if target_count else None
        last_date = str(target["timestamp"].iloc[-1].date()) if target_count else None
        base = {
            "target_observations": target_count,
            "first_target_date": first_date,
            "last_target_date": last_date,
        }
        covariate_counts: dict[str, int] = {}
        frame_starts = [target["timestamp"].min()]
        frame_ends = [target["timestamp"].max()]
        for series_id in self._covariate_series_ids:
            frame = context.get_series(series_id).copy()
            frame["timestamp"] = pd.to_datetime(frame["timestamp"])
            covariate_counts[series_id] = len(frame)
            if not frame.empty:
                frame_starts.append(frame["timestamp"].min())
                frame_ends.append(frame["timestamp"].max())
        if frame_starts and frame_ends:
            aligned_start = max(frame_starts)
            aligned_end = min(frame_ends)
            aligned_count = len(pd.bdate_range(aligned_start, aligned_end)) if aligned_start <= aligned_end else 0
        else:
            aligned_count = 0
        lag_requirement = max(
            self._settings.lightgbm_lags,
            self._settings.lightgbm_covariate_lags,
        )
        effective_estimate = max(0, aligned_count - lag_requirement - max_horizon + 1)
        simple = ModelTrainingData(
            **base,
            notes="All cutoff-safe target observations are supplied to the Darts model.",
        )
        lightgbm = ModelTrainingData(
            **base,
            aligned_observations=aligned_count,
            effective_training_examples_estimate=effective_estimate,
            target_lags=self._settings.lightgbm_lags,
            covariate_lags=self._settings.lightgbm_covariate_lags,
            covariates=list(self._covariate_series_ids),
            covariate_observations=covariate_counts,
            notes=(
                "Aligned observations use the common business-day span after Darts-style "
                "calendar regularization. Effective examples are an auditable estimate: "
                "aligned_observations - max(lags) - max_horizon + 1."
            ),
        )
        return {"arima": simple, "kalman": simple.model_copy(deep=True), "lightgbm": lightgbm}

    def _run_model_suite(
        self,
        *,
        context: ForecastContext,
        target_series_id: str,
        horizons: list[int],
        frequency: str,
        cutoff_date: str,
    ) -> ModelSuiteResult:
        task = ForecastingTask(
            task_id=f"cfm_v2_{target_series_id}_{cutoff_date}",
            target_series_id=target_series_id,
            horizons=horizons,
            frequency=frequency,
            description=f"CFM v2 deterministic model suite for {target_series_id}.",
        )
        audits = self._training_audit(context, target_series_id, max(horizons))
        successful, failures = self._ensemble.collect(task, context)
        model_results: dict[str, ModelForecastResult] = {}
        for name in ("arima", "kalman", "lightgbm"):
            if name in successful:
                model_results[name] = self._format_model_result(
                    name,
                    successful[name],
                    horizons,
                    audits[name],
                )
            else:
                model_results[name] = ModelForecastResult(
                    model_name=name,
                    predictor_id=name,
                    status="error",
                    training_data=audits[name],
                    error=failures.get(name, "model did not return a result."),
                )
        ensemble_result = None
        active_weights = self._ensemble.normalized_weights(list(successful))
        if successful:
            ensemble_predictions = self._ensemble.combine(task, context, successful)
            ensemble_training = ModelTrainingData(
                target_observations=audits["arima"].target_observations,
                first_target_date=audits["arima"].first_target_date,
                last_target_date=audits["arima"].last_target_date,
                aligned_observations=audits["lightgbm"].aligned_observations,
                effective_training_examples_estimate=(audits["lightgbm"].effective_training_examples_estimate),
                covariates=list(self._covariate_series_ids),
                notes="Ensemble consumes forecast distributions, not raw observations.",
            )
            ensemble_result = self._format_model_result(
                "ensemble",
                ensemble_predictions,
                horizons,
                ensemble_training,
            )
        disagreement = {}
        for index, horizon in enumerate(horizons):
            values = [
                predictions[index].payload.point_forecast
                for predictions in successful.values()
                if isinstance(predictions[index].payload, ContinuousForecast)
            ]
            disagreement[horizon] = float(np.std(values, ddof=0)) if values else 0.0
        return ModelSuiteResult(
            target_series_id=target_series_id,
            cutoff_date=cutoff_date,
            horizons=horizons,
            models=model_results,
            ensemble=ensemble_result,
            configured_ensemble_weights=dict(self._settings.ensemble_weights),
            active_ensemble_weights=active_weights,
            successful_models=list(successful),
            failed_models=list(failures),
            model_disagreement_std=disagreement,
        )

    @staticmethod
    def _format_model_result(
        model_name: str,
        predictions: list[Prediction],
        horizons: list[int],
        training_data: ModelTrainingData,
    ) -> ModelForecastResult:
        forecasts = []
        for horizon, prediction in zip(horizons, predictions, strict=True):
            payload = prediction.payload
            if not isinstance(payload, ContinuousForecast):
                raise TypeError("model suite requires ContinuousForecast payloads.")
            forecasts.append(
                ModelHorizonForecast(
                    horizon=horizon,
                    forecast_date=str(pd.Timestamp(prediction.forecast_date).date()),
                    point_forecast=payload.point_forecast,
                    quantiles=payload.quantiles,
                )
            )
        return ModelForecastResult(
            model_name=model_name,
            predictor_id=predictions[0].predictor_id,
            status="ok",
            forecasts=forecasts,
            training_data=training_data,
        )


__all__ = ["MarketDataTool"]
