"""Cutoff-safe market data, model forecasts, and model training audit."""

from __future__ import annotations

import random
import threading
from datetime import datetime

import numpy as np
import pandas as pd
from aieng.forecasting.data import DataService, ForecastContext
from aieng.forecasting.evaluation.prediction import ContinuousForecast, Prediction
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic.agent_factory import AS_OF_STATE_KEY
from energy_oil_forecasting.cfm_agent_v_5_2.config import (
    DEFAULT_SETTINGS,
    CfmV52Settings,
)
from energy_oil_forecasting.cfm_agent_v_5_2.diagnostics import (
    compute_market_diagnostics,
)
from energy_oil_forecasting.cfm_agent_v_5_2.fingerprints import stable_fingerprint
from energy_oil_forecasting.cfm_agent_v_5_2.models import (
    CfmEnsemblePredictor,
    build_arima_predictor,
    build_kalman_predictor,
    build_lightgbm_predictor,
)
from energy_oil_forecasting.cfm_agent_v_5_2.schemas import (
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
    from google.adk.tools.tool_context import ToolContext
except ModuleNotFoundError as exc:
    raise ImportError("The market-data tool requires the agentic extra.") from exc


_ALLOWED_OPERATIONS = {
    "list_series",
    "get_series",
    "run_models",
    "get_series_and_run_models",
}


class AuthoritativeSuiteTool:
    """Run exactly one successful authoritative suite with auditable attempts."""

    def __init__(
        self,
        data_service: DataService,
        *,
        settings: CfmV52Settings = DEFAULT_SETTINGS,
        covariate_series_ids: list[str] | None = None,
    ) -> None:
        self._data_service = data_service
        self._settings = settings
        self._lock = threading.Lock()
        self._last_result: MarketDataToolResult | None = None
        self._attempt_count = 0
        self._validation_failure_count = 0
        self._successful_execution_count = 0
        self._attempt_errors: list[str] = []
        self._expected_task: dict[str, object] | None = None
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

    def prepare_workflow(self, task: ForecastingTask, context: ForecastContext) -> None:
        """Reset and bind the one-run tool to the current task."""
        expected_cutoff = str(pd.Timestamp(context.as_of).date())
        task_horizons = [int(value) for value in task.horizons]
        if task_horizons != sorted(set(task_horizons)):
            raise ValueError("task horizons must be strictly increasing and unique")
        with self._lock:
            self._last_result = None
            self._attempt_count = 0
            self._validation_failure_count = 0
            self._successful_execution_count = 0
            self._attempt_errors = []
            self._expected_task = {
                "cutoff_date": expected_cutoff,
                "target_series_id": task.target_series_id,
                "horizons": task_horizons,
                "frequency": task.frequency,
            }

    @property
    def last_result(self) -> MarketDataToolResult | None:
        """Return a defensive copy of the most recent tool result."""
        with self._lock:
            return self._last_result.model_copy(deep=True) if self._last_result else None

    @property
    def audit(self) -> dict[str, object]:
        """Return workflow-level execution counters."""
        with self._lock:
            return {
                "tool_name": "run_authoritative_suite",
                "attempt_count": self._attempt_count,
                "validation_failure_count": self._validation_failure_count,
                "successful_execution_count": self._successful_execution_count,
                "attempt_errors": list(self._attempt_errors),
                "rng_seed": (
                    self._last_result.model_suite.rng_seed
                    if self._last_result and self._last_result.model_suite
                    else None
                ),
                "rng_seed_derivation": (
                    self._last_result.model_suite.rng_seed_derivation
                    if self._last_result and self._last_result.model_suite
                    else None
                ),
                "rng_seed_input_fingerprint": (
                    self._last_result.model_suite.rng_seed_input_fingerprint
                    if self._last_result and self._last_result.model_suite
                    else None
                ),
            }

    def as_function_tool(self) -> FunctionTool:
        """Return the narrow agent-facing tool with required horizons."""
        return FunctionTool(func=self.run_authoritative_suite)

    def run_authoritative_suite(  # noqa: PLR0911, PLR0912, PLR0917
        self,
        cutoff_date: str,
        series_ids: list[str],
        target_series_id: str,
        horizons: list[int],
        frequency: str = "B",
        lookback: int = 260,
        tool_context: ToolContext | None = None,
    ) -> str:
        """Run one task-bound cutoff-safe numerical suite."""
        harness_cutoff = tool_context.state.get(AS_OF_STATE_KEY) if tool_context is not None else None
        with self._lock:
            self._attempt_count += 1
            attempt = self._attempt_count
            already_succeeded = self._successful_execution_count > 0
            expected = dict(self._expected_task or {})
        authoritative_cutoff = str(harness_cutoff or expected.get("cutoff_date") or cutoff_date)
        try:
            supplied_horizons = [int(value) for value in horizons] if horizons else []
        except (TypeError, ValueError):
            return self._record_validation_error(authoritative_cutoff, "horizons must contain only integers.")
        mismatches = []
        if cutoff_date != authoritative_cutoff:
            mismatches.append(f"cutoff_date must equal authoritative cutoff {authoritative_cutoff}")
        if expected and target_series_id != expected["target_series_id"]:
            mismatches.append(f"target_series_id must equal {expected['target_series_id']!r}")
        if expected and supplied_horizons != expected["horizons"]:
            mismatches.append(f"horizons must equal {expected['horizons']}")
        if expected and frequency != expected["frequency"]:
            mismatches.append(f"frequency must equal {expected['frequency']!r}")
        if mismatches:
            return self._record_validation_error(authoritative_cutoff, "; ".join(mismatches))
        cutoff_date = authoritative_cutoff
        if already_succeeded:
            return MarketDataToolResult(
                status="error",
                operation="run_authoritative_suite",
                cutoff_date=cutoff_date,
                error="authoritative numerical suite already executed successfully for this workflow.",
            ).model_dump_json(indent=2)
        if attempt > self._settings.max_pre_execution_attempts:
            return self._record_validation_error(cutoff_date, "pre-execution attempt limit exceeded.")
        try:
            as_of = datetime.strptime(cutoff_date, "%Y-%m-%d")
            clean_lookback = max(1, min(int(lookback), self._settings.max_data_rows_per_series))
            clean_horizons = [int(horizon) for horizon in horizons]
        except (TypeError, ValueError):
            return self._record_validation_error(
                cutoff_date,
                "cutoff_date must be YYYY-MM-DD; lookback and horizons must be integers.",
            )
        if not clean_horizons or any(horizon < 1 for horizon in clean_horizons):
            return self._record_validation_error(cutoff_date, "positive horizons are required.")
        available = set(self._data_service.series_ids)
        if target_series_id not in available:
            return self._record_validation_error(cutoff_date, f"target series {target_series_id!r} is not registered.")
        context = self._data_service.context(as_of=as_of)
        snapshots: list[SeriesSnapshot] = []
        warnings: list[str] = []
        for series_id in dict.fromkeys(series_ids):
            try:
                snapshots.append(self._snapshot(context, series_id, clean_lookback))
            except KeyError:
                warnings.append(f"Series {series_id!r} is not registered and was omitted.")
        try:
            suite = self._run_model_suite(
                context=context,
                target_series_id=target_series_id,
                horizons=clean_horizons,
                frequency=frequency,
                cutoff_date=cutoff_date,
            )
        except (KeyError, TypeError, ValueError) as exc:
            return self._record_validation_error(cutoff_date, str(exc))
        if suite.failed_models:
            warnings.append("One or more models failed; ensemble weights were renormalized over successful models.")
        result = MarketDataToolResult(
            status="partial" if warnings or suite.failed_models else "ok",
            operation="run_authoritative_suite",
            cutoff_date=cutoff_date,
            available_series=self._data_service.series_ids,
            series=snapshots,
            model_suite=suite,
            warnings=warnings,
        )
        with self._lock:
            self._successful_execution_count += 1
            self._last_result = result.model_copy(deep=True)
        return result.model_dump_json(indent=2)

    def _record_validation_error(self, cutoff_date: str, message: str) -> str:
        with self._lock:
            self._validation_failure_count += 1
            self._attempt_errors.append(message)
        return MarketDataToolResult(
            status="error",
            operation="run_authoritative_suite",
            cutoff_date=cutoff_date,
            error=message,
        ).model_dump_json(indent=2)

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
        return {
            "arima": simple,
            "kalman": simple.model_copy(deep=True),
            "lightgbm": lightgbm,
        }

    def _run_model_suite(
        self,
        *,
        context: ForecastContext,
        target_series_id: str,
        horizons: list[int],
        frequency: str,
        cutoff_date: str,
    ) -> ModelSuiteResult:
        seed_inputs = {
            "target_series_id": target_series_id,
            "cutoff_date": cutoff_date,
            "horizons": horizons,
            "frequency": frequency,
            "model_num_samples": self._settings.model_num_samples,
            "lightgbm_lags": self._settings.lightgbm_lags,
            "lightgbm_covariate_lags": self._settings.lightgbm_covariate_lags,
            "kalman_dim_x": self._settings.kalman_dim_x,
            "covariate_series_ids": list(self._covariate_series_ids),
            "ensemble_weights": dict(self._settings.ensemble_weights),
        }
        seed_fingerprint = stable_fingerprint(seed_inputs, prefix="rng_seed_input")
        rng_seed = int(seed_fingerprint.rsplit(":", 1)[-1][:8], 16)
        task = ForecastingTask(
            task_id=f"cfm_v5_2_{target_series_id}_{cutoff_date}",
            target_series_id=target_series_id,
            horizons=horizons,
            frequency=frequency,
            description=f"CFM v5.2 authoritative numerical suite for {target_series_id}.",
        )
        audits = self._training_audit(context, target_series_id, max(horizons))
        # Set both process-global RNGs immediately before component execution.
        random.seed(rng_seed)
        np.random.seed(rng_seed)
        successful, failures = self._ensemble.collect(task, context)
        model_results: dict[str, ModelForecastResult] = {}
        for name in ("arima", "kalman", "lightgbm"):
            if name in successful:
                model_results[name] = self._format_model_result(
                    name,
                    successful[name],
                    horizons,
                    audits[name],
                    self._settings.model_num_samples,
                )
            else:
                model_results[name] = ModelForecastResult(
                    model_name=name,
                    predictor_id=name,
                    status="error",
                    training_data=audits[name],
                    num_samples=self._settings.model_num_samples,
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
                self._settings.model_num_samples,
            )
        disagreement = {}
        for index, horizon in enumerate(horizons):
            values = [
                predictions[index].payload.point_forecast
                for predictions in successful.values()
                if isinstance(predictions[index].payload, ContinuousForecast)
            ]
            disagreement[horizon] = float(np.std(values, ddof=0)) if values else 0.0
        target_frame = context.get_series(target_series_id)
        covariate_frames = {series_id: context.get_series(series_id) for series_id in self._covariate_series_ids}
        diagnostics = compute_market_diagnostics(
            target_frame,
            as_of=context.as_of,
            covariates=covariate_frames,
        )
        suite_identity = {
            "target_series_id": target_series_id,
            "cutoff_date": cutoff_date,
            "horizons": horizons,
            "frequency": frequency,
            "model_num_samples": self._settings.model_num_samples,
            "rng_seed": rng_seed,
            "rng_seed_derivation": "sha256_first_32_bits_of_canonical_task_and_numerical_configuration",
            "rng_seed_input_fingerprint": seed_fingerprint,
            "models": {name: result.model_dump(mode="json") for name, result in model_results.items()},
            "ensemble": ensemble_result.model_dump(mode="json") if ensemble_result else None,
            "active_weights": active_weights,
        }
        return ModelSuiteResult(
            model_suite_id=stable_fingerprint(suite_identity, prefix="model_suite"),
            target_series_id=target_series_id,
            cutoff_date=cutoff_date,
            horizons=horizons,
            frequency=frequency,
            model_num_samples=self._settings.model_num_samples,
            rng_seed=rng_seed,
            rng_seed_derivation="sha256_first_32_bits_of_canonical_task_and_numerical_configuration",
            rng_seed_input_fingerprint=seed_fingerprint,
            models=model_results,
            ensemble=ensemble_result,
            configured_ensemble_weights=dict(self._settings.ensemble_weights),
            active_ensemble_weights=active_weights,
            successful_models=list(successful),
            failed_models=list(failures),
            model_disagreement_std=disagreement,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _format_model_result(
        model_name: str,
        predictions: list[Prediction],
        horizons: list[int],
        training_data: ModelTrainingData,
        num_samples: int,
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
            num_samples=num_samples,
        )


MarketDataTool = AuthoritativeSuiteTool

__all__ = ["AuthoritativeSuiteTool", "MarketDataTool"]
