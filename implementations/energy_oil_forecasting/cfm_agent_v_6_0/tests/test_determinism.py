"""Reproducibility of the numerical suite and the model-suite identity (specification Section 1.7)."""

from __future__ import annotations

import json
import random
import subprocess
import sys
from datetime import datetime

import numpy as np
from aieng.forecasting.evaluation.task import ForecastingTask
from energy_oil_forecasting.cfm_agent_v_6_0.config import ENSEMBLE_COMPONENTS, CfmV60Settings
from energy_oil_forecasting.cfm_agent_v_6_0.fingerprints import identity, seed_from_identity
from energy_oil_forecasting.cfm_agent_v_6_0.tools.market_data import (
    AuthoritativeSuiteTool,
    derive_component_seeds,
)
from energy_oil_forecasting.data import WTI_SERIES_ID


CUTOFF = datetime(2026, 9, 14)
CUTOFF_TEXT = "2026-09-14"
HORIZONS = [5, 10, 21]


def _run_suite(service, settings=None, horizons=HORIZONS, frequency="B"):
    tool = AuthoritativeSuiteTool(service, settings=settings or CfmV60Settings())
    return tool._run_model_suite(
        context=service.context(as_of=CUTOFF),
        target_series_id=WTI_SERIES_ID,
        horizons=list(horizons),
        frequency=frequency,
        cutoff_date=CUTOFF_TEXT,
    )


def _run_tool(service, settings=None):
    tool = AuthoritativeSuiteTool(service, settings=settings or CfmV60Settings())
    task = ForecastingTask(
        task_id="det", target_series_id=WTI_SERIES_ID, horizons=HORIZONS, frequency="B", description="d"
    )
    tool.prepare_workflow(task, service.context(as_of=CUTOFF))
    payload = json.loads(
        tool.run_authoritative_suite(
            cutoff_date=CUTOFF_TEXT,
            series_ids=[WTI_SERIES_ID],
            target_series_id=WTI_SERIES_ID,
            horizons=HORIZONS,
            frequency="B",
        )
    )
    assert payload["status"] == "ok", payload
    return payload["model_suite"]


def test_two_fresh_runs_are_bit_equal_including_lightgbm(synthetic_service_factory):
    first = _run_suite(synthetic_service_factory())
    second = _run_suite(synthetic_service_factory())
    assert first.model_dump() == second.model_dump()
    assert first.successful_components == list(ENSEMBLE_COMPONENTS)
    for name in ENSEMBLE_COMPONENTS:
        assert first.components[name].forecasts == second.components[name].forecasts
    assert first.ensemble_guard_applied is False


def test_task_derived_seed_and_suite_id_are_repeatable(synthetic_service):
    tool = AuthoritativeSuiteTool(synthetic_service, settings=CfmV60Settings())
    context = synthetic_service.context(as_of=CUTOFF)
    kwargs = {
        "context": context,
        "target_series_id": WTI_SERIES_ID,
        "horizons": [5],
        "frequency": "B",
        "cutoff_date": CUTOFF_TEXT,
    }
    first = tool._run_model_suite(**kwargs)
    second = tool._run_model_suite(**kwargs)
    assert first.suite_seed == second.suite_seed
    assert first.seed_input_fingerprint == second.seed_input_fingerprint
    assert first.data_fingerprint == second.data_fingerprint
    assert first.component_seeds == second.component_seeds
    assert first.model_suite_id == second.model_suite_id
    assert first.suite_seed == seed_from_identity(first.seed_input_fingerprint)
    assert first.component_seeds == derive_component_seeds(first.suite_seed)
    assert list(first.component_seeds) == list(ENSEMBLE_COMPONENTS)


def test_seed_inputs_are_task_settings_covariates_and_data(synthetic_service):
    tool = AuthoritativeSuiteTool(synthetic_service, settings=CfmV60Settings())
    suite = _run_suite(synthetic_service)
    seed_inputs = tool.seed_inputs(
        target_series_id=WTI_SERIES_ID,
        cutoff_date=CUTOFF_TEXT,
        horizons=HORIZONS,
        frequency="B",
        data_fingerprint=suite.data_fingerprint,
    )
    assert identity(seed_inputs, prefix="suite_seed_input") == suite.seed_input_fingerprint
    assert set(seed_inputs) == {"task", "identity_settings", "covariate_series_ids", "data_fingerprint"}


def test_identity_setting_changes_the_suite_id_and_llm_setting_does_not(synthetic_service_factory):
    baseline = _run_suite(synthetic_service_factory())
    llm_changed = _run_suite(synthetic_service_factory(), CfmV60Settings(main_temperature=0.9))
    assert llm_changed.model_suite_id == baseline.model_suite_id
    assert llm_changed.suite_seed == baseline.suite_seed
    assert llm_changed.ensemble.forecasts == baseline.ensemble.forecasts
    ridge_changed = _run_suite(synthetic_service_factory(), CfmV60Settings(ar_ridge_penalty=1e-3))
    assert ridge_changed.model_suite_id != baseline.model_suite_id
    assert ridge_changed.suite_seed != baseline.suite_seed
    audit_changed = _run_suite(synthetic_service_factory(), CfmV60Settings(audit_enabled=True))
    assert audit_changed.model_suite_id == baseline.model_suite_id


def test_task_binding_changes_the_seed(synthetic_service_factory):
    baseline = _run_suite(synthetic_service_factory())
    assert _run_suite(synthetic_service_factory(), horizons=[5, 10]).suite_seed != baseline.suite_seed
    assert _run_suite(synthetic_service_factory(), frequency="D").suite_seed != baseline.suite_seed


def test_data_change_changes_the_seed(synthetic_service_factory):
    baseline = _run_suite(synthetic_service_factory())
    changed = _run_suite(synthetic_service_factory(seed=1))
    assert changed.data_fingerprint != baseline.data_fingerprint
    assert changed.suite_seed != baseline.suite_seed


def test_global_random_state_is_never_touched(synthetic_service_factory):
    np.random.seed(123)
    random.seed(123)
    expected_np = np.random.RandomState(123).random()
    expected_py = random.Random(123).random()
    _run_suite(synthetic_service_factory())
    assert np.random.random() == expected_np
    assert random.random() == expected_py


def test_subprocess_run_equals_in_process_run(synthetic_service_factory, tmp_path):
    settings = CfmV60Settings()
    service = synthetic_service_factory()
    context = service.context(as_of=CUTOFF)
    ids = [WTI_SERIES_ID, *settings.covariate_series_ids]
    for series_id in ids:
        context.get_series(series_id).to_csv(tmp_path / f"{series_id}.csv", index=False, float_format="%.17g")
    in_process = _run_tool(service, settings)
    script = f"""
import json, pandas as pd
from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.features import StaticFrameAdapter
from aieng.forecasting.evaluation.task import ForecastingTask
from energy_oil_forecasting.cfm_agent_v_6_0.config import CfmV60Settings
from energy_oil_forecasting.cfm_agent_v_6_0.tools.market_data import AuthoritativeSuiteTool
from datetime import datetime
service = DataService()
for series_id in {ids!r}:
    frame = pd.read_csv({str(tmp_path)!r} + "/" + series_id + ".csv", parse_dates=["timestamp", "released_at"], float_precision="round_trip")
    service.register(series_id, StaticFrameAdapter(frame), SeriesMetadata(series_id=series_id, description="d", source="s", units="u", frequency="B"))
tool = AuthoritativeSuiteTool(service, settings=CfmV60Settings())
task = ForecastingTask(task_id="det", target_series_id={WTI_SERIES_ID!r}, horizons={HORIZONS!r}, frequency="B", description="d")
tool.prepare_workflow(task, service.context(as_of=datetime(2026, 9, 14)))
payload = json.loads(tool.run_authoritative_suite(cutoff_date={CUTOFF_TEXT!r}, series_ids=[{WTI_SERIES_ID!r}], target_series_id={WTI_SERIES_ID!r}, horizons={HORIZONS!r}, frequency="B"))
print(json.dumps(payload["model_suite"]))
"""
    child = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    child_suite = json.loads(child.stdout.strip().splitlines()[-1])
    assert child_suite["data_fingerprint"] == in_process["data_fingerprint"]
    assert child_suite["suite_seed"] == in_process["suite_seed"]
    assert child_suite["component_seeds"] == in_process["component_seeds"]
    assert child_suite["model_suite_id"] == in_process["model_suite_id"]
    for name in ENSEMBLE_COMPONENTS:
        assert child_suite["components"][name]["forecasts"] == in_process["components"][name]["forecasts"]
    assert child_suite["ensemble"]["forecasts"] == in_process["ensemble"]["forecasts"]
    assert child_suite["diagnostics"] == in_process["diagnostics"]
