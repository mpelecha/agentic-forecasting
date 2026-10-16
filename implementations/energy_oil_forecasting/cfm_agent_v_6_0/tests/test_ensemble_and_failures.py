"""Ensemble order of operations, failure dispositions, and identity rules (specification Section 1.6)."""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pytest
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES, ContinuousForecast, Prediction
from energy_oil_forecasting.cfm_agent_v_6_0.config import ENSEMBLE_COMPONENTS, CfmV60Settings
from energy_oil_forecasting.cfm_agent_v_6_0.models import ensemble as ensemble_module
from energy_oil_forecasting.cfm_agent_v_6_0.models.bootstrap import HistoricalBootstrapReturn
from energy_oil_forecasting.cfm_agent_v_6_0.models.ensemble import ReturnPathEnsemble
from energy_oil_forecasting.cfm_agent_v_6_0.models.lightgbm_return import LightGBMReturn
from energy_oil_forecasting.cfm_agent_v_6_0.models.paths import ComponentFailure
from energy_oil_forecasting.cfm_agent_v_6_0.tools.market_data import AuthoritativeSuiteTool
from energy_oil_forecasting.data import WTI_SERIES_ID


CUTOFF = datetime(2026, 9, 14)
LEVELS = [float(level) for level in STANDARD_QUANTILES]


class StubComponent:
    """A component returning fixed quantiles, or raising, for the ensemble arithmetic tests."""

    def __init__(self, name: str, quantiles_by_horizon: dict[int, dict[float, float]] | None, error=None):
        self.name = name
        self.quantiles_by_horizon = quantiles_by_horizon
        self.error = error

    @property
    def predictor_id(self):
        return f"stub_{self.name}"

    def predict(self, task, context):
        if self.error is not None:
            raise self.error
        assert self.quantiles_by_horizon is not None
        return [
            Prediction(
                predictor_id=self.predictor_id,
                task_id=task.task_id,
                issued_at=CUTOFF,
                as_of=context.as_of,
                forecast_date=CUTOFF,
                payload=ContinuousForecast(
                    point_forecast=self.quantiles_by_horizon[h][0.5], quantiles=self.quantiles_by_horizon[h]
                ),
            )
            for h in task.horizons
        ]


def _quantiles(center: float, spread: float) -> dict[float, float]:
    return {level: center + spread * (level - 0.5) for level in LEVELS}


def _stub_suite(task, **overrides):
    base = {
        "bootstrap": {h: _quantiles(80.0 + h, 10.0) for h in task.horizons},
        "sparse_ar": {h: _quantiles(82.0 + h, 12.0) for h in task.horizons},
        "lightgbm": {h: _quantiles(78.0 + h, 8.0) for h in task.horizons},
    }
    return {name: StubComponent(name, overrides.get(name, base[name])) for name in ENSEMBLE_COMPONENTS}


def _run(service, settings, horizons=(5, 10, 21)):
    tool = AuthoritativeSuiteTool(service, settings=settings)
    return tool._run_model_suite(
        context=service.context(as_of=CUTOFF),
        target_series_id=WTI_SERIES_ID,
        horizons=list(horizons),
        frequency="B",
        cutoff_date="2026-09-14",
    )


def test_order_of_operations_weighted_pool_then_guard_then_p50(synthetic_task, synthetic_context):
    settings = CfmV60Settings()
    components = _stub_suite(synthetic_task)
    ensemble = ReturnPathEnsemble(components, settings)
    successful, dispositions = ensemble.collect(synthetic_task, synthetic_context)
    assert all(d.status == "ok" for d in dispositions.values())
    outputs = ensemble.combine(synthetic_task, synthetic_context, successful)
    weights = settings.ensemble_weights
    for index, horizon in enumerate(synthetic_task.horizons):
        payload = outputs[index].payload
        for level in LEVELS:
            expected = sum(weights[name] * components[name].quantiles_by_horizon[horizon][level] for name in weights)
            assert payload.quantiles[level] == pytest.approx(expected, abs=1e-12)
        assert payload.point_forecast == payload.quantiles[0.5]
        assert outputs[index].metadata["guard_applied"] is False
        assert outputs[index].metadata["normalized_weights"] == weights
        assert outputs[index].predictor_id == "cfm_v60_return_ensemble"
    assert ensemble.guard_applied is False


def test_guard_is_flagged_when_forced(monkeypatch, synthetic_task, synthetic_context):
    settings = CfmV60Settings()
    crossing = _quantiles(80.0, 10.0)
    crossing[0.6] = crossing[0.4] - 1.0  # a decrease in the level
    components = _stub_suite(synthetic_task, bootstrap=dict.fromkeys(synthetic_task.horizons, crossing))
    ensemble = ReturnPathEnsemble(components, settings)
    successful, dispositions = ensemble.collect(synthetic_task, synthetic_context)
    # Validation excludes the crossing component before the pool.
    assert dispositions["bootstrap"].failure_code == "quantile_crossing"
    assert "bootstrap" not in successful
    # Force the crossing past validation: the guard then changes a value and must say so.
    monkeypatch.setattr(ensemble_module, "validate_quantiles", lambda quantiles: None)
    forced = {name: components[name].predict(synthetic_task, synthetic_context) for name in ENSEMBLE_COMPONENTS}
    outputs = ensemble.combine(synthetic_task, synthetic_context, forced)
    assert ensemble.guard_applied is True
    assert all(output.metadata["guard_applied"] for output in outputs)
    ordered = [outputs[0].payload.quantiles[level] for level in LEVELS]
    assert ordered == sorted(ordered)


def test_renormalisation_is_fifty_fifty_when_bootstrap_fails(synthetic_task, synthetic_context):
    settings = CfmV60Settings()
    components = _stub_suite(synthetic_task)
    components["bootstrap"] = StubComponent("bootstrap", None, ComponentFailure("insufficient_history", "too short"))
    ensemble = ReturnPathEnsemble(components, settings)
    successful, dispositions = ensemble.collect(synthetic_task, synthetic_context)
    assert dispositions["bootstrap"].status == "error"
    assert dispositions["bootstrap"].failure_code == "insufficient_history"
    assert dispositions["bootstrap"].reason == "too short"
    active = ensemble.check_dispositions(dispositions)
    assert active == ["sparse_ar", "lightgbm"]
    assert ensemble.normalized_weights(active) == {"sparse_ar": 0.5, "lightgbm": 0.5}
    outputs = ensemble.combine(synthetic_task, synthetic_context, successful)
    horizon = synthetic_task.horizons[0]
    for level in LEVELS:
        expected = 0.5 * (
            components["sparse_ar"].quantiles_by_horizon[horizon][level]
            + components["lightgbm"].quantiles_by_horizon[horizon][level]
        )
        assert outputs[0].payload.quantiles[level] == pytest.approx(expected, abs=1e-12)


def test_other_exceptions_become_library_error_dispositions(synthetic_task, synthetic_context):
    settings = CfmV60Settings()
    components = _stub_suite(synthetic_task)
    components["lightgbm"] = StubComponent("lightgbm", None, ZeroDivisionError("division by zero"))
    ensemble = ReturnPathEnsemble(components, settings)
    _, dispositions = ensemble.collect(synthetic_task, synthetic_context)
    assert dispositions["lightgbm"].failure_code == "library_error:ZeroDivisionError"
    assert dispositions["lightgbm"].reason == "division by zero"


def test_fail_closed_when_every_component_fails(synthetic_task, synthetic_context):
    settings = CfmV60Settings()
    components = {
        name: StubComponent(name, None, ComponentFailure("insufficient_history", name)) for name in ENSEMBLE_COMPONENTS
    }
    ensemble = ReturnPathEnsemble(components, settings)
    successful, dispositions = ensemble.collect(synthetic_task, synthetic_context)
    assert successful == {}
    with pytest.raises(RuntimeError, match="every positively weighted component failed"):
        ensemble.check_dispositions(dispositions)
    with pytest.raises(RuntimeError):
        ensemble.combine(synthetic_task, synthetic_context, successful)


def test_require_all_components_fails_the_suite_on_any_failure(synthetic_task, synthetic_context):
    strict = CfmV60Settings(require_all_components=True)
    components = _stub_suite(synthetic_task)
    components["lightgbm"] = StubComponent("lightgbm", None, ComponentFailure("covariate_missing", "gold"))
    ensemble = ReturnPathEnsemble(components, strict)
    _, dispositions = ensemble.collect(synthetic_task, synthetic_context)
    with pytest.raises(RuntimeError, match="require_all_components"):
        ensemble.check_dispositions(dispositions)
    lenient = ReturnPathEnsemble(components, CfmV60Settings())
    assert lenient.check_dispositions(dispositions) == ["bootstrap", "sparse_ar"]


def test_suite_fails_closed_through_the_tool_on_short_history(synthetic_service_factory, synthetic_task):
    service = synthetic_service_factory(n_observations=30)
    tool = AuthoritativeSuiteTool(service, settings=CfmV60Settings())
    tool.prepare_workflow(synthetic_task, service.context(as_of=CUTOFF))
    result = json.loads(
        tool.run_authoritative_suite(
            cutoff_date="2026-09-14",
            series_ids=[WTI_SERIES_ID],
            target_series_id=WTI_SERIES_ID,
            horizons=[5, 10, 21],
            frequency="B",
        )
    )
    assert result["status"] == "error"
    assert "every positively weighted component failed" in result["error"]
    assert "insufficient_history" in result["error"]
    assert tool.audit["successful_execution_count"] == 0
    assert tool.code_execution_payload() is None


def test_real_suite_excludes_lightgbm_when_a_covariate_is_missing(synthetic_service_factory):
    settings = CfmV60Settings()
    present = [series_id for series_id in settings.covariate_series_ids if series_id != "vix_level_l1b"]
    suite = _run(synthetic_service_factory(covariate_ids=present), settings)
    assert suite.failed_components == ["lightgbm"]
    assert suite.successful_components == ["bootstrap", "sparse_ar"]
    assert suite.dispositions["lightgbm"].failure_code == "covariate_missing"
    assert suite.components["lightgbm"].status == "error"
    assert suite.components["lightgbm"].failure_code == "covariate_missing"
    assert suite.components["lightgbm"].component_seed == suite.component_seeds["lightgbm"]
    assert suite.active_ensemble_weights == {"bootstrap": pytest.approx(2 / 3), "sparse_ar": pytest.approx(1 / 3)}
    assert suite.diagnostics.data_quality.missing_covariates == ["vix_level_l1b"]
    assert suite.diagnostics.vix_level is None
    with pytest.raises(RuntimeError, match="require_all_components"):
        _run(synthetic_service_factory(covariate_ids=present), CfmV60Settings(require_all_components=True))


def test_failure_code_enters_the_suite_id_but_the_message_does_not(monkeypatch, synthetic_service):
    settings = CfmV60Settings()

    def failing(message):
        def _simulate(self):
            raise ComponentFailure("insufficient_aligned_rows", message)

        return _simulate

    monkeypatch.setattr(LightGBMReturn, "_simulate_returns", failing("first message"))
    first = _run(synthetic_service, settings)
    monkeypatch.setattr(LightGBMReturn, "_simulate_returns", failing("second message"))
    second = _run(synthetic_service, settings)
    assert first.dispositions["lightgbm"].reason == "first message"
    assert second.dispositions["lightgbm"].reason == "second message"
    assert first.model_suite_id == second.model_suite_id
    monkeypatch.setattr(
        LightGBMReturn,
        "_simulate_returns",
        lambda self: (_ for _ in ()).throw(ComponentFailure("covariate_missing", "first message")),
    )
    third = _run(synthetic_service, settings)
    assert third.model_suite_id != first.model_suite_id
    monkeypatch.undo()
    healthy = _run(synthetic_service, settings)
    assert healthy.model_suite_id != first.model_suite_id
    assert healthy.failed_components == []


def test_component_paths_do_not_depend_on_another_components_failure(monkeypatch, synthetic_service):
    settings = CfmV60Settings()
    healthy = _run(synthetic_service, settings)
    monkeypatch.setattr(
        HistoricalBootstrapReturn,
        "_simulate_returns",
        lambda self: (_ for _ in ()).throw(ComponentFailure("insufficient_history", "forced")),
    )
    degraded = _run(synthetic_service, settings)
    for name in ("sparse_ar", "lightgbm"):
        assert degraded.components[name].forecasts == healthy.components[name].forecasts
    assert degraded.active_ensemble_weights == {"sparse_ar": 0.5, "lightgbm": 0.5}
    assert degraded.component_seeds == healthy.component_seeds
    assert degraded.floors.simulated_floor_activation_counts["bootstrap"] == 0


def test_suite_result_is_complete(synthetic_service):
    settings = CfmV60Settings()
    suite = _run(synthetic_service, settings)
    assert list(suite.components) == list(ENSEMBLE_COMPONENTS)
    assert suite.successful_components == list(ENSEMBLE_COMPONENTS)
    assert suite.failed_components == []
    assert suite.configured_ensemble_weights == settings.ensemble_weights
    assert suite.active_ensemble_weights == settings.ensemble_weights
    assert suite.ensemble is not None and suite.ensemble.predictor_id == "cfm_v60_return_ensemble"
    assert suite.ensemble_guard_applied is False
    assert set(suite.component_seeds) == set(ENSEMBLE_COMPONENTS)
    assert suite.data_fingerprint.startswith("data:sha256:")
    assert suite.seed_input_fingerprint.startswith("suite_seed_input:sha256:")
    assert suite.model_suite_id.startswith("model_suite:sha256:")
    assert suite.history_window.rows == 3000
    assert suite.floors.input_price_floor == settings.input_price_floor
    assert set(suite.floors.simulated_floor_activation_counts) == set(ENSEMBLE_COMPONENTS)
    assert set(suite.model_disagreement_std) == {5, 10, 21}
    assert suite.diagnostics.model_disagreement_std == suite.model_disagreement_std
    assert set(suite.diagnostics.historical_profiles) == {5, 10, 21}
    for horizon in suite.horizons:
        points = [
            next(f.point_forecast for f in suite.components[name].forecasts if f.horizon == horizon)
            for name in ENSEMBLE_COMPONENTS
        ]
        assert suite.model_disagreement_std[horizon] == pytest.approx(float(np.std(points, ddof=0)))
    for name in ENSEMBLE_COMPONENTS:
        component = suite.components[name]
        assert component.component_seed == suite.component_seeds[name]
        assert [f.horizon for f in component.forecasts] == [5, 10, 21]
        assert all(len(f.quantiles) == 11 for f in component.forecasts)
    assert [f.forecast_date for f in suite.ensemble.forecasts] == ["2026-09-21", "2026-09-28", "2026-10-13"]
