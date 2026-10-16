"""Cutoff-safe market data and the authoritative numerical suite (specification Sections 1 and 2).

The tool runs the return-path suite exactly once per workflow: it builds the
:class:`PriceHistory`, derives the suite seed and the three component
generators from the canonical seed inputs, runs the three components, pools the
survivors, computes the diagnostics, and records the model-suite identity.
Paths never leave Python; the LLM sees capped series snapshots, the component
and ensemble quantiles, and the diagnostics.
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from aieng.forecasting.data import DataService, ForecastContext
from aieng.forecasting.evaluation.prediction import ContinuousForecast, Prediction
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic.agent_factory import AS_OF_STATE_KEY
from energy_oil_forecasting.cfm_agent_v_6_0.config import (
    DEFAULT_SETTINGS,
    ENSEMBLE_COMPONENTS,
    CfmV60Settings,
    identity_settings,
)
from energy_oil_forecasting.cfm_agent_v_6_0.diagnostics import compute_market_diagnostics
from energy_oil_forecasting.cfm_agent_v_6_0.fingerprints import identity, seed_from_identity
from energy_oil_forecasting.cfm_agent_v_6_0.history import PriceHistory
from energy_oil_forecasting.cfm_agent_v_6_0.models.bootstrap import HistoricalBootstrapReturn
from energy_oil_forecasting.cfm_agent_v_6_0.models.ensemble import ReturnPathEnsemble
from energy_oil_forecasting.cfm_agent_v_6_0.models.lightgbm_return import LightGBMReturn
from energy_oil_forecasting.cfm_agent_v_6_0.models.paths import ReturnPathComponent
from energy_oil_forecasting.cfm_agent_v_6_0.models.sparse_ar import SparseARReturn
from energy_oil_forecasting.cfm_agent_v_6_0.schemas import (
    ComponentDisposition,
    FloorRecord,
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


SEED_DERIVATION = (
    "suite_seed = int(sha256(canonical_json(seed_inputs))[0:8], 16); "
    "component seeds = numpy.random.SeedSequence(suite_seed).spawn(3) in the order "
    "bootstrap, sparse_ar, lightgbm, each read as generate_state(1, dtype=uint32)[0]; "
    "component generators = numpy.random.default_rng(component_seed); "
    "seed_inputs = task binding + identity settings + covariate ids + data fingerprint"
)

_COMPONENT_CLASSES: dict[str, type[ReturnPathComponent]] = {
    "bootstrap": HistoricalBootstrapReturn,
    "sparse_ar": SparseARReturn,
    "lightgbm": LightGBMReturn,
}


def derive_component_seeds(suite_seed: int) -> dict[str, int]:
    """Spawn one 32-bit seed per component from the suite seed, in spawn order."""
    children = np.random.SeedSequence(suite_seed).spawn(len(ENSEMBLE_COMPONENTS))
    return {
        name: int(child.generate_state(1, dtype=np.uint32)[0]) for name, child in zip(ENSEMBLE_COMPONENTS, children)
    }


class AuthoritativeSuiteTool:
    """Run exactly one successful authoritative suite with auditable attempts."""

    def __init__(
        self,
        data_service: DataService,
        *,
        settings: CfmV60Settings = DEFAULT_SETTINGS,
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
        # Configured covariates are never filtered against the service: an absent
        # covariate must surface as a LightGBM covariate_missing disposition.
        self._covariate_series_ids = (
            list(settings.covariate_series_ids) if covariate_series_ids is None else list(covariate_series_ids)
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
        """Return workflow-level execution counters and the suite's identity inputs."""
        with self._lock:
            suite = self._last_result.model_suite if self._last_result else None
            return {
                "tool_name": "run_authoritative_suite",
                "attempt_count": self._attempt_count,
                "validation_failure_count": self._validation_failure_count,
                "successful_execution_count": self._successful_execution_count,
                "attempt_errors": list(self._attempt_errors),
                "suite_seed": suite.suite_seed if suite else None,
                "seed_derivation": suite.seed_derivation if suite else None,
                "seed_input_fingerprint": suite.seed_input_fingerprint if suite else None,
                "data_fingerprint": suite.data_fingerprint if suite else None,
                "component_seeds": dict(suite.component_seeds) if suite else None,
                "component_dispositions": (
                    {name: disposition.model_dump(mode="json") for name, disposition in suite.dispositions.items()}
                    if suite
                    else None
                ),
                "floors": suite.floors.model_dump(mode="json") if suite else None,
                "ensemble_guard_applied": suite.ensemble_guard_applied if suite else None,
                "history_window": suite.history_window.model_dump(mode="json") if suite else None,
            }

    def code_execution_payload(self) -> dict[str, Any] | None:
        """Return the sandbox preamble data: diagnostics, quantiles, and the task binding; no paths or text."""
        with self._lock:
            suite = self._last_result.model_suite if self._last_result else None
            if suite is None:
                return None
            quantiles: dict[str, dict[str, dict[str, float]]] = {}
            for name, result in suite.components.items():
                if result.status == "ok":
                    quantiles[name] = {
                        str(forecast.horizon): {str(level): value for level, value in forecast.quantiles.items()}
                        for forecast in result.forecasts
                    }
            if suite.ensemble is not None:
                quantiles["ensemble"] = {
                    str(forecast.horizon): {str(level): value for level, value in forecast.quantiles.items()}
                    for forecast in suite.ensemble.forecasts
                }
            return {
                "DIAGNOSTICS": suite.diagnostics.model_dump(mode="json"),
                "MODEL_SUITE_QUANTILES": quantiles,
                "TASK_BINDING": {
                    "cutoff": suite.cutoff_date,
                    "target_series_id": suite.target_series_id,
                    "horizons": list(suite.horizons),
                    "frequency": suite.frequency,
                },
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
        if max(clean_horizons) > self._settings.simulation_length_business_days:
            return self._record_validation_error(
                cutoff_date,
                f"horizons must not exceed the simulation length {self._settings.simulation_length_business_days}.",
            )
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
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            return self._record_validation_error(cutoff_date, str(exc))
        if suite.failed_components:
            codes = {name: suite.dispositions[name].failure_code for name in suite.failed_components}
            warnings.append(
                f"Components failed and were excluded ({codes}); ensemble weights were renormalised over survivors."
            )
        result = MarketDataToolResult(
            status="partial" if warnings else "ok",
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

    # ------------------------------------------------------------- suite --

    def build_history(self, context: ForecastContext, target_series_id: str) -> PriceHistory:
        """Build the floored, windowed history with the configured covariates."""
        return PriceHistory.from_context(context, target_series_id, self._covariate_series_ids, self._settings)

    def seed_inputs(
        self,
        *,
        target_series_id: str,
        cutoff_date: str,
        horizons: list[int],
        frequency: str,
        data_fingerprint: str,
    ) -> dict[str, Any]:
        """Return the canonical seed-input payload (Section 1.7)."""
        return {
            "task": {
                "target_series_id": target_series_id,
                "cutoff_date": cutoff_date,
                "horizons": [int(h) for h in horizons],
                "frequency": frequency,
            },
            "identity_settings": identity_settings(self._settings),
            "covariate_series_ids": list(self._covariate_series_ids),
            "data_fingerprint": data_fingerprint,
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
        settings = self._settings
        history = self.build_history(context, target_series_id)
        seed_inputs = self.seed_inputs(
            target_series_id=target_series_id,
            cutoff_date=cutoff_date,
            horizons=horizons,
            frequency=frequency,
            data_fingerprint=history.data_fingerprint,
        )
        seed_fingerprint = identity(seed_inputs, prefix="suite_seed_input")
        suite_seed = seed_from_identity(seed_fingerprint)
        component_seeds = derive_component_seeds(suite_seed)
        components: dict[str, ReturnPathComponent] = {
            name: _COMPONENT_CLASSES[name](
                history,
                settings,
                np.random.default_rng(component_seeds[name]),
                component_seeds[name],
            )
            for name in ENSEMBLE_COMPONENTS
        }
        ensemble = ReturnPathEnsemble(components, settings)
        task = ForecastingTask(
            task_id=f"cfm_v6_0_{target_series_id}_{cutoff_date}",
            target_series_id=target_series_id,
            horizons=horizons,
            frequency=frequency,
            description=f"CFM v6.0 authoritative numerical suite for {target_series_id}.",
        )
        successful, dispositions = ensemble.collect(task, context)
        active = ensemble.check_dispositions(dispositions)
        active_weights = ensemble.normalized_weights(active)
        ensemble_predictions = ensemble.combine(task, context, successful)
        guard_applied = ensemble.guard_applied

        component_results: dict[str, ModelForecastResult] = {}
        for name in ENSEMBLE_COMPONENTS:
            component = components[name]
            disposition = dispositions[name]
            training = component.training_data or ModelTrainingData(
                target_observations=int(len(history.prices)),
                first_target_date=history.first_date,
                last_target_date=history.last_date,
                returns_available=int(len(history.returns)),
            )
            if disposition.status == "ok":
                component_results[name] = self._format_model_result(
                    name,
                    successful[name],
                    horizons,
                    training,
                    settings.model_num_paths,
                    component_seed=component_seeds[name],
                    floor_activations=component.simulated_floor_activations,
                )
            else:
                component_results[name] = ModelForecastResult(
                    model_name=name,
                    predictor_id=component.predictor_id,
                    status="error",
                    training_data=training,
                    num_paths=settings.model_num_paths,
                    component_seed=component_seeds[name],
                    simulated_floor_activations=component.simulated_floor_activations,
                    failure_code=disposition.failure_code,
                    error=disposition.reason,
                )
        ensemble_result = self._format_model_result(
            "ensemble",
            ensemble_predictions,
            horizons,
            ModelTrainingData(
                target_observations=int(len(history.prices)),
                first_target_date=history.first_date,
                last_target_date=history.last_date,
                returns_available=int(len(history.returns)),
                covariates=list(self._covariate_series_ids),
                notes="Ensemble consumes component quantiles, not raw observations.",
            ),
            settings.model_num_paths,
            component_seed=None,
            floor_activations=0,
        )
        disagreement: dict[int, float] = {}
        for index, horizon in enumerate(horizons):
            points = [successful[name][index].payload.point_forecast for name in active]
            disagreement[int(horizon)] = float(np.std(points, ddof=0))
        diagnostics = compute_market_diagnostics(
            history,
            as_of=context.as_of,
            settings=settings,
            model_disagreement=disagreement,
            horizons=horizons,
        )
        floors = FloorRecord(
            input_price_floor=settings.input_price_floor,
            simulated_price_floor=settings.simulated_price_floor,
            final_price_floor=settings.final_price_floor,
            input_floor_activation_count=history.input_floor_activation_count,
            input_floor_activation_dates=list(history.input_floor_activation_dates),
            simulated_floor_activation_counts={
                name: components[name].simulated_floor_activations for name in ENSEMBLE_COMPONENTS
            },
        )
        suite_identity = {
            "seed_inputs": seed_inputs,
            "components": {
                name: {
                    "status": dispositions[name].status,
                    "failure_code": dispositions[name].failure_code,
                    "component_seed": component_seeds[name],
                    "quantiles": (
                        {
                            str(forecast.horizon): {str(level): value for level, value in forecast.quantiles.items()}
                            for forecast in component_results[name].forecasts
                        }
                        if dispositions[name].status == "ok"
                        else None
                    ),
                }
                for name in ENSEMBLE_COMPONENTS
            },
            "active_weights": active_weights,
            "dispositions": {
                name: {"status": dispositions[name].status, "failure_code": dispositions[name].failure_code}
                for name in ENSEMBLE_COMPONENTS
            },
        }
        return ModelSuiteResult(
            model_suite_id=identity(suite_identity, prefix="model_suite"),
            target_series_id=target_series_id,
            cutoff_date=cutoff_date,
            horizons=horizons,
            frequency=frequency,
            model_num_paths=settings.model_num_paths,
            simulation_length_business_days=settings.simulation_length_business_days,
            suite_seed=suite_seed,
            seed_derivation=SEED_DERIVATION,
            seed_input_fingerprint=seed_fingerprint,
            data_fingerprint=history.data_fingerprint,
            history_window=history.window,
            component_seeds=component_seeds,
            components=component_results,
            ensemble=ensemble_result,
            configured_ensemble_weights=dict(settings.ensemble_weights),
            active_ensemble_weights=active_weights,
            successful_components=list(active),
            failed_components=[name for name in ENSEMBLE_COMPONENTS if dispositions[name].status != "ok"],
            dispositions={
                name: ComponentDisposition(**dispositions[name].model_dump()) for name in ENSEMBLE_COMPONENTS
            },
            floors=floors,
            ensemble_guard_applied=guard_applied,
            model_disagreement_std=disagreement,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _format_model_result(
        model_name: str,
        predictions: list[Prediction],
        horizons: list[int],
        training_data: ModelTrainingData,
        num_paths: int,
        *,
        component_seed: int | None,
        floor_activations: int,
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
            num_paths=num_paths,
            component_seed=component_seed,
            simulated_floor_activations=floor_activations,
        )


MarketDataTool = AuthoritativeSuiteTool

__all__ = ["SEED_DERIVATION", "AuthoritativeSuiteTool", "MarketDataTool", "derive_component_seeds"]
