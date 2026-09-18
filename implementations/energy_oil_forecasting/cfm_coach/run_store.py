"""Persist and load `RunRecord`s -- the coach's corpus.

This is the whole of R6, and it is why nothing in ``cfm_agent_v_5_0`` has to change.
The five identity fields a calibration corpus needs (settings, calibration version,
model, package fingerprint, prompt version) are all known *to the caller* before the
run starts: a coach-side runner constructs the settings itself and injects them via
``build_cfm_agent_config(settings=...)``. So the writer takes the predictions the
agent hands back and pairs them with what it already had. The agent never learns the
coach exists.

Corpus cannot be backfilled without contamination, so this module is the one piece
that must exist before daily running begins -- every day without a record is
evidence that cannot be recovered later.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from aieng.forecasting.evaluation.prediction import Prediction
from energy_oil_forecasting.cfm_coach.config import (
    DEFAULT_SETTINGS,
    CoachSettings,
)
from energy_oil_forecasting.cfm_coach.schemas import (
    HorizonRecord,
    Provenance,
    RunRecord,
    SettingsSource,
)
from energy_oil_forecasting.cfm_coach.targets import DEFAULT_TARGET, AgentTarget
from pydantic import BaseModel


def agent_package_fingerprint(target: AgentTarget = DEFAULT_TARGET) -> str:
    """Fingerprint the agent package that produced a run.

    Hashes the shipped ``MANIFEST.sha256``, which already covers every source file
    including ``config.py``. So if an agent's own constants ever change underneath
    the corpus, the fingerprint changes with them and the affected records become
    visibly a different system -- which matters because a calibration fitted to one
    set of defaults says nothing about another.

    ``target`` is not optional in spirit even though it has a default. This
    function used to hardcode v5.0's manifest path and prefix, and calling it
    unqualified for a v5.2 run would stamp that run with v5.0's identity -- which
    is exactly the confusion ``ComparisonPolicy.single_package_fingerprint`` exists
    to detect, defeated at the source.
    """
    manifest = target.manifest_path
    if not manifest.exists():  # pragma: no cover - the manifest ships with the package
        return f"{target.fingerprint_prefix}:sha256:unavailable"
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    return f"{target.fingerprint_prefix}:sha256:{digest}"


def _as_dicts(predictions: list[Prediction] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item if isinstance(item, dict) else item.model_dump(mode="json") for item in predictions]


def _component_quantiles_by_horizon(model_suite: dict[str, Any]) -> dict[int, dict[str, dict[str, float]]]:
    """Per-model quantiles keyed by horizon, so ensemble weights stay replayable offline."""
    out: dict[int, dict[str, dict[str, float]]] = {}
    for name, model in (model_suite.get("models") or {}).items():
        for forecast in model.get("forecasts") or []:
            out.setdefault(int(forecast["horizon"]), {})[name] = forecast["quantiles"]
    return out


def build_run_record(
    predictions: list[Prediction] | list[dict[str, Any]],
    *,
    settings: BaseModel,
    calibration_version: str,
    agent_model: str,
    schema_version: int,
    target: AgentTarget = DEFAULT_TARGET,
    prompt_version: str | None = None,
    provenance: Provenance = "live_forward",
    settings_source: SettingsSource = "recorded",
    run_id_prefix: str | None = None,
    issued_at: datetime | None = None,
) -> RunRecord:
    """Assemble one `RunRecord` from a completed agent run plus the caller's own identity fields.

    The per-horizon material (ensemble input, recorded output, policy decision) is
    split out; the assessment and research packet are stored once rather than
    repeated per horizon as the agent's metadata does.

    ``run_id_prefix`` defaults to the target's agent id, which is the form the
    existing v5.0 corpus was written with. Streams that share an agent id but not
    an LLM pass a prefix carrying the stream, so two corpora can never collide on
    a run id even if they are ever merged into one directory.
    """
    rows = _as_dicts(predictions)
    if not rows:
        raise ValueError("cannot build a run record from zero predictions")

    head = rows[0]["metadata"]
    suite = head["numerical_suite_audit"]["model_suite"]
    components = _component_quantiles_by_horizon(suite)
    binding = head.get("task_binding", {})

    issued = issued_at or datetime.fromisoformat(rows[0]["issued_at"])
    cutoff = binding.get("cutoff") or str(datetime.fromisoformat(rows[0]["as_of"]).date())

    forecasts: list[HorizonRecord] = []
    for row in rows:
        metadata = row["metadata"]
        transformation = metadata["forecast_transformation"]
        horizon = int(transformation["horizon"])
        ensemble = metadata["unadjusted_ensemble"]
        forecasts.append(
            HorizonRecord(
                horizon=horizon,
                forecast_date=row["forecast_date"],
                component_quantiles=components.get(horizon, {}),
                ensemble_point_forecast=ensemble["point_forecast"],
                ensemble_quantiles=ensemble["quantiles"],
                final_point_forecast=transformation["final_point_forecast"],
                final_quantiles=transformation["final_quantiles"],
                policy_decision=metadata.get("policy_decision", {}),
                forecast_transformation=transformation,
            )
        )
    forecasts.sort(key=lambda item: item.horizon)

    prefix = run_id_prefix or target.agent_id
    return RunRecord(
        schema_version=schema_version,
        run_id=f"{prefix}__{cutoff}__{issued:%Y%m%dT%H%M%SZ}",
        agent_id=target.agent_id,
        agent_model=agent_model,
        predictor_id=rows[0]["predictor_id"],
        package_fingerprint=agent_package_fingerprint(target),
        calibration_version=calibration_version,
        prompt_version=prompt_version or target.prompt_version,
        settings=settings.model_dump(mode="json"),
        settings_source=settings_source,
        provenance=provenance,
        task_id=rows[0]["task_id"],
        cutoff=cutoff,
        horizons=[record.horizon for record in forecasts],
        issued_at=issued,
        assessment=head["llm_context_assessment"],
        research_packet=head["active_research_packet"],
        diagnostics=head.get("market_diagnostics", {}),
        audit_signals={
            "audit_switch": head.get("audit_switch"),
            "audit_only_controls": head.get("audit_only_controls", {}),
            "source_validation_audit": head.get("source_validation_audit", {}),
            "claim_support_audit": head.get("claim_support_audit", {}),
            "claim_support_findings": head.get("claim_support_findings", []),
            "research_execution_audit": head.get("research_execution_audit", {}),
            "cutoff_verification_audit": head.get("cutoff_verification_audit", {}),
            "code_execution_audit": head.get("code_execution_audit", {}),
            "model_disagreement_std": suite.get("model_disagreement_std", {}),
            "successful_models": suite.get("successful_models", []),
            "failed_models": suite.get("failed_models", []),
        },
        forecasts=forecasts,
    )


class RunRecordStore:
    store_id = "cfm_coach_run_store_v1"

    def __init__(self, settings: CoachSettings = DEFAULT_SETTINGS):
        self.s = settings

    @property
    def directory(self) -> Path:
        return self.s.runs_dir

    def path_for(self, run_id: str) -> Path:
        return self.directory / f"{run_id}.json"

    def write(
        self,
        predictions: list[Prediction] | list[dict[str, Any]],
        *,
        settings: BaseModel,
        calibration_version: str,
        agent_model: str,
        target: AgentTarget = DEFAULT_TARGET,
        prompt_version: str | None = None,
        provenance: Provenance = "live_forward",
        settings_source: SettingsSource = "recorded",
        run_id_prefix: str | None = None,
        issued_at: datetime | None = None,
    ) -> RunRecord:
        record = build_run_record(
            predictions,
            settings=settings,
            calibration_version=calibration_version,
            agent_model=agent_model,
            schema_version=self.s.run_record_schema_version,
            target=target,
            prompt_version=prompt_version,
            provenance=provenance,
            settings_source=settings_source,
            run_id_prefix=run_id_prefix,
            issued_at=issued_at,
        )
        return self.save(record)

    def save(self, record: RunRecord) -> RunRecord:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path_for(record.run_id)
        path.write_text(
            json.dumps(record.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return record

    def load(self, path: Path) -> RunRecord:
        payload = json.loads(path.read_text(encoding="utf-8"))
        version = payload.get("schema_version")
        if version != self.s.run_record_schema_version:
            raise ValueError(
                f"{path.name} has run-record schema_version {version!r}, but this coach "
                f"understands {self.s.run_record_schema_version!r}. Refusing to guess at its shape."
            )
        return RunRecord.model_validate(payload)

    def load_all(self) -> list[RunRecord]:
        if not self.directory.exists():
            return []
        records = [self.load(path) for path in sorted(self.directory.glob("*.json"))]
        return sorted(records, key=lambda record: (record.cutoff, record.issued_at))


__all__ = ["RunRecordStore", "agent_package_fingerprint", "build_run_record"]
