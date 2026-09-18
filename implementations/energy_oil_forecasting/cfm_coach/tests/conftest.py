from datetime import datetime, timedelta

import pandas as pd
import pytest
from energy_oil_forecasting.cfm_agent_v_5_0.config import CfmV50Settings
from energy_oil_forecasting.cfm_coach.config import CoachSettings
from energy_oil_forecasting.cfm_coach.schemas import HorizonRecord, RunRecord


# Quantile keys are strings on purpose: every record round-trips through JSON, so
# the schemas must coerce string keys back to floats or replay silently breaks.
def _quantiles(p10: float, p50: float, p90: float) -> dict[str, float]:
    return {"0.1": p10, "0.5": p50, "0.9": p90}


def _prediction(horizon: int, forecast_date: str, ensemble_p50: float, final_p50: float) -> dict:
    spread = 4.0
    return {
        "predictor_id": "agent_predictor_cfm_agent_v_5_0_gemini-3.1-flash-lite-preview_continuous",
        "task_id": "cfm_coach_test_wti_crude_oil_price_2026-03-02",
        "issued_at": "2026-03-02T12:00:00",
        "as_of": "2026-03-02T00:00:00",
        "forecast_date": f"{forecast_date}T00:00:00",
        "payload": {
            "point_forecast": final_p50,
            "quantiles": _quantiles(final_p50 - spread, final_p50, final_p50 + spread),
        },
        "metadata": {
            "llm_context_assessment": {
                "physical_status": "confirmed_disruption",
                "incremental_novelty": "likely_new_relative_to_model_data",
                "confidence": 0.9,
                "material_evidence_conflict": False,
                "overall_rationale": "test",
            },
            "active_research_packet": {"packet_id": "research_packet:sha256:test", "sources": []},
            "market_diagnostics": {"realized_volatility_21b": 0.38, "jump_zscore_63b": 2.93},
            "unadjusted_ensemble": {
                "point_forecast": ensemble_p50,
                "quantiles": _quantiles(ensemble_p50 - spread, ensemble_p50, ensemble_p50 + spread),
            },
            "forecast_transformation": {
                "horizon": horizon,
                "final_point_forecast": final_p50,
                "final_quantiles": _quantiles(final_p50 - spread, final_p50, final_p50 + spread),
            },
            "policy_decision": {"horizon": horizon, "evidence_tier": "strong", "eligible": True},
            "task_binding": {"cutoff": "2026-03-02", "horizons": [5, 21], "validated": True},
            "numerical_suite_audit": {
                "model_suite": {
                    "models": {
                        "arima": {"forecasts": [{"horizon": horizon, "quantiles": _quantiles(60.0, 66.0, 72.0)}]},
                        "kalman": {"forecasts": [{"horizon": horizon, "quantiles": _quantiles(61.0, 67.0, 73.0)}]},
                        "lightgbm": {"forecasts": [{"horizon": horizon, "quantiles": _quantiles(59.0, 65.0, 71.0)}]},
                    },
                    "model_disagreement_std": {str(horizon): 1.2},
                    "successful_models": ["arima", "kalman", "lightgbm"],
                    "failed_models": [],
                }
            },
            "audit_switch": True,
            "audit_only_controls": {"source_validator_executed": True},
            "source_validation_audit": {"executed": True, "records": []},
            "claim_support_audit": {"executed": True},
            "claim_support_findings": [],
            "research_execution_audit": {"successful_execution_count": 1},
            "cutoff_verification_audit": {},
            "code_execution_audit": {},
        },
    }


@pytest.fixture
def predictions() -> list[dict]:
    """Two horizons of one run, in the shape `run_cfm_agent_v_5_0_interactive.py` writes."""
    return [
        _prediction(5, "2026-03-09", ensemble_p50=66.0, final_p50=68.0),
        _prediction(21, "2026-03-31", ensemble_p50=65.0, final_p50=67.0),
    ]


@pytest.fixture
def e2e_audit(predictions: list[dict]) -> dict:
    """Return the other audit shape: a dict wrapping each prediction under `forecasts`."""
    return {
        "settings": {"audit_enabled": True},
        "forecasts": [
            {"horizon": item["metadata"]["forecast_transformation"]["horizon"], "prediction": item}
            for item in predictions
        ],
    }


@pytest.fixture
def settings(tmp_path) -> CoachSettings:
    return CoachSettings(
        runs_dir=tmp_path / "runs",
        calibration_dir=tmp_path / "calibration",
        proposals_dir=tmp_path / "proposals",
        horizons=(5, 21),
    )


# -- synthetic corpus, for the M1 harness ------------------------------------
#
# The real corpus has one live_forward record and nothing resolved, so the
# statistical half of the gate cannot be exercised against it for another two
# months. These build a corpus with known properties instead: the assertions are
# about the machinery, and the machinery has to be right before the data arrives.

#: A neutral assessment takes `EvidencePolicy`'s early-return branch and the
#: engine's `neutral` branch, which reproduces the ensemble exactly. That makes a
#: synthetic record trivially replay-faithful, so a test that fails is failing on
#: what it meant to test rather than on a hand-built assessment being subtly invalid.
NEUTRAL_ASSESSMENT = {
    "research_packet_id": "research_packet:sha256:synthetic",
    "evidence_claims": [],
    "physical_status": "normal",
    "incremental_novelty": "likely_reflected_in_model_data",
    "material_evidence_conflict": False,
    "confidence": 0.7,
    "research_summary": "synthetic",
    "overall_rationale": "synthetic",
    "warnings": [],
}

EMPTY_PACKET = {
    "packet_id": "research_packet:sha256:synthetic",
    "cutoff_date": "2026-01-01",
    "queries": [],
    "verified_summaries": [],
    "sources": [],
    "warnings": [],
}

STANDARD_GRID = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)

#: Half-width at each level for a spread of 1.0, so a caller sets one number.
_SHAPE = {0.05: -1.6, 0.1: -1.0, 0.2: -0.6, 0.3: -0.35, 0.4: -0.15, 0.5: 0.0}


def quantile_grid(centre: float, spread: float) -> dict[float, float]:
    """Symmetric quantiles around `centre`; `spread` is the P10-P90 half-width."""
    grid = {}
    for level in STANDARD_GRID:
        offset = _SHAPE.get(level)
        if offset is None:
            offset = -_SHAPE[round(1.0 - level, 2)]
        grid[level] = centre + spread * offset
    return grid


def make_run_record(
    *,
    cutoff: str,
    horizons: tuple[int, ...] = (5, 10, 21),
    centre: float = 80.0,
    spread: float = 2.0,
    provenance: str = "live_forward",
    prompt_version: str = "cfm_v5_0_builtin",
    package_fingerprint: str = "cfm_v5_0_package:sha256:synthetic",
    issued_at: str | None = None,
    agent_id: str = "cfm_agent_v_5_0",
    agent_model: str = "synthetic-model",
) -> RunRecord:
    """One replay-faithful synthetic run with a neutral assessment."""
    cutoff_date = datetime.fromisoformat(cutoff)
    forecasts = []
    for horizon in horizons:
        quantiles = quantile_grid(centre, spread)
        forecasts.append(
            HorizonRecord(
                horizon=horizon,
                forecast_date=cutoff_date + timedelta(days=horizon),
                ensemble_point_forecast=centre,
                ensemble_quantiles=dict(quantiles),
                final_point_forecast=centre,
                final_quantiles=dict(quantiles),
            )
        )
    assessment = {
        **NEUTRAL_ASSESSMENT,
        "horizon_actions": [
            {
                "horizon": horizon,
                "center_action": "no_change",
                "uncertainty_action": "unchanged",
                "rationale": "synthetic",
            }
            for horizon in horizons
        ],
    }
    return RunRecord(
        schema_version=1,
        run_id=f"synthetic__{cutoff}",
        agent_id=agent_id,
        agent_model=agent_model,
        predictor_id="synthetic_predictor",
        package_fingerprint=package_fingerprint,
        calibration_version="v001",
        prompt_version=prompt_version,
        settings=CfmV50Settings().model_dump(mode="json"),
        provenance=provenance,
        task_id=f"synthetic_{cutoff}",
        cutoff=cutoff_date.date(),
        horizons=list(horizons),
        issued_at=datetime.fromisoformat(issued_at) if issued_at else cutoff_date,
        assessment=assessment,
        research_packet=EMPTY_PACKET,
        diagnostics={"latest_value": centre},
        forecasts=forecasts,
    )


class FakeService:
    """Minimal `DataService` stand-in: one series, from an explicit date->value map."""

    def __init__(self, values: dict[str, float]):
        self.values = values
        self.series_ids = ["wti_crude_oil_price"]

    def get_series(self, series_id: str, as_of=None) -> pd.DataFrame:  # noqa: ANN001, ARG002
        return pd.DataFrame(
            {
                "timestamp": pd.to_datetime(list(self.values)),
                "value": list(self.values.values()),
            }
        )


# -- a real v5.2 record, for the review agent ---------------------------------
#
# The synthetic v5.0 record above takes every neutral branch. The review agent
# reads claims, actions, policy tiers and component quantiles, so its tests need
# a record with all of those populated *and* valid under v5.2's own schemas --
# which a hand-built packet is not (`ResearchPacket` carries fields like
# `active_evidence_type`). This is one live v52_advanced run (cutoff 2026-09-08)
# with the bulky audit blocks dropped; replay fidelity is asserted in
# `test_review_cards.py` so the fixture cannot silently rot.

import json
from copy import deepcopy
from datetime import date
from pathlib import Path

from energy_oil_forecasting.cfm_coach.schemas import CalibrationVersion

V52_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "v52_record.json"
V52_IDENTITY = CalibrationVersion(version="v001", effective_from=date(2004, 1, 1))
_V52_PAYLOAD: dict | None = None


def _v52_payload() -> dict:
    global _V52_PAYLOAD  # noqa: PLW0603
    if _V52_PAYLOAD is None:
        _V52_PAYLOAD = json.loads(V52_FIXTURE_PATH.read_text(encoding="utf-8"))
    return deepcopy(_V52_PAYLOAD)


def _shift_quantiles(quantiles: dict, delta: float) -> dict:
    return {level: float(value) + delta for level, value in quantiles.items()}


def make_v52_run_record(
    *,
    cutoff: str = "2026-09-08",
    suffix: str = "a",
    provenance: str = "live_forward",
    agent_model: str | None = None,
    run_id_prefix: str = "cfm_agent_v_5_2__advanced",
    price_shift: float = 0.0,
    assessment_patch: dict | None = None,
    issued_at: str | None = None,
) -> RunRecord:
    """The fixture run re-dated to `cutoff`, optionally translated in price and re-judged.

    Translation keeps replay exact: the engine's overlay is a fraction of the
    interval width and the layer is affine around the centre, neither of which
    depends on the level. `assessment_patch` is merged over the stored assessment
    (e.g. a different `horizon_actions` list) -- fidelity to the *stored* final
    quantiles is then deliberately broken, so tests that patch must not assert it.
    """
    payload = _v52_payload()
    original_cutoff = date.fromisoformat(payload["cutoff"])
    new_cutoff = date.fromisoformat(cutoff)
    delta_days = (new_cutoff - original_cutoff).days

    def shift_date(text: str) -> str:
        return (datetime.fromisoformat(text) + timedelta(days=delta_days)).isoformat()

    payload["cutoff"] = cutoff
    payload["task_id"] = f"cfm_coach_wti_crude_oil_price_{cutoff}"
    payload["run_id"] = f"{run_id_prefix}__{cutoff}__{suffix}"
    payload["issued_at"] = issued_at or shift_date(payload["issued_at"])
    payload["provenance"] = provenance
    if agent_model is not None:
        payload["agent_model"] = agent_model
    payload["research_packet"]["cutoff_date"] = cutoff
    diagnostics = payload["diagnostics"]
    diagnostics["latest_observation_date"] = shift_date(diagnostics["latest_observation_date"])[:10]
    diagnostics["latest_value"] = float(diagnostics["latest_value"]) + price_shift
    for forecast in payload["forecasts"]:
        forecast["forecast_date"] = shift_date(forecast["forecast_date"])
        if price_shift:
            forecast["component_quantiles"] = {
                name: _shift_quantiles(q, price_shift) for name, q in forecast["component_quantiles"].items()
            }
            forecast["ensemble_quantiles"] = _shift_quantiles(forecast["ensemble_quantiles"], price_shift)
            forecast["ensemble_point_forecast"] += price_shift
            forecast["final_quantiles"] = _shift_quantiles(forecast["final_quantiles"], price_shift)
            forecast["final_point_forecast"] += price_shift
    if assessment_patch:
        payload["assessment"] = {**payload["assessment"], **assessment_patch}
    return RunRecord.model_validate(payload)
