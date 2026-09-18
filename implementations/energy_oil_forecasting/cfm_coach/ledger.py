"""Calibration versions on disk, and how they become agent settings.

The learned artifact is data, not code. A calibration version is a dated JSON
document naming the agent-settings fields to override; the agent package is never
edited, so its ``MANIFEST.sha256`` and the provenance claims in its
``BUILD_VERIFICATION.md`` stay true.

Each run stream keeps its own ledger directory. A calibration is fitted against
one agent's realized error under one LLM; carrying it across to another stream
would assert a transfer nothing has measured.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from energy_oil_forecasting.cfm_coach.config import DEFAULT_SETTINGS, CoachSettings
from energy_oil_forecasting.cfm_coach.schemas import CalibrationVersion
from energy_oil_forecasting.cfm_coach.targets import DEFAULT_TARGET, AgentTarget
from pydantic import BaseModel


class CalibrationLedger:
    ledger_id = "cfm_coach_calibration_ledger_v1"

    def __init__(self, settings: CoachSettings = DEFAULT_SETTINGS):
        self.s = settings

    @property
    def directory(self) -> Path:
        return self.s.calibration_dir

    def _paths(self) -> list[Path]:
        return sorted(path for path in self.directory.glob("v*.json") if path.name != "ledger.json")

    def load(self, version: str) -> CalibrationVersion:
        path = self.directory / f"{version}.json"
        if not path.exists():
            raise FileNotFoundError(f"no calibration version {version!r} at {path}")
        return CalibrationVersion.model_validate_json(path.read_text(encoding="utf-8"))

    def all_versions(self) -> list[CalibrationVersion]:
        versions = [CalibrationVersion.model_validate_json(path.read_text(encoding="utf-8")) for path in self._paths()]
        return sorted(versions, key=lambda item: (item.effective_from, item.version))

    def current(self, as_of: date) -> CalibrationVersion:
        """Return the version in force on `as_of` -- the latest one that had taken effect.

        Scoring a past run means replaying it under the calibration that actually
        produced it, so this is a lookup by date rather than "the newest file".
        """
        eligible = [version for version in self.all_versions() if version.effective_from <= as_of]
        if not eligible:
            raise ValueError(f"no calibration version is effective on {as_of}; is {self.directory} populated?")
        return eligible[-1]

    def save(self, version: CalibrationVersion) -> Path:
        path = self.directory / f"{version.version}.json"
        if path.exists():
            raise FileExistsError(
                f"calibration {version.version!r} already exists; versions are immutable once written"
            )
        self.directory.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(version.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    @staticmethod
    def to_agent_settings(
        version: CalibrationVersion,
        *,
        base: BaseModel,
        target: AgentTarget = DEFAULT_TARGET,
    ) -> Any:
        """Apply a calibration overlay to run-level agent settings.

        `base` carries choices that are not calibration -- whether the audit-only
        controls run, whether code execution is enabled, which LLM answers -- so the
        overlay only ever names constants the coach is allowed to tune. Unknown
        field names raise here rather than being silently dropped, because both
        settings classes are declared ``extra="forbid"``.

        `target` decides which settings class is rebuilt. v5.2 carries a field v5.0
        does not (``structured_output_retry_model``), so reconstructing a v5.2 base
        through v5.0's class would raise on that field -- correctly, but with a
        confusing message and only by luck rather than by design.
        """
        if not version.settings_overlay:
            return base
        named_models = sorted(set(version.settings_overlay) & set(target.model_settings_fields))
        if named_models:
            raise ValueError(
                f"calibration {version.version!r} names LLM field(s) {named_models} in its settings_overlay. "
                "Which model answered is an identity fact about a run, like the package fingerprint -- "
                "it belongs to the run stream, not to a fitted calibration. Applying it here would let a "
                "fit silently change models underneath a corpus."
            )
        return target.settings_cls(**{**base.model_dump(), **version.settings_overlay})


__all__ = ["CalibrationLedger"]
