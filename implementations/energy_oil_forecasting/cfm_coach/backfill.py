"""One-off conversion of pre-R6 audit files into `RunRecord`s.

Two audit files exist from before the coach did, and they have *different shapes*:
the e2e harness wrote a dict with a ``forecasts`` list wrapping each prediction,
while ``run_cfm_agent_v_5_0_interactive.py`` wrote a bare list of predictions. Both
are handled here so the corpus starts with everything that exists.

Both were also issued months after their cutoff -- the agent was replayed at a
historical date against today's live web -- so they are recorded as
``replayed_live_search`` and are excluded from fitting evidence by construction.
They remain useful for replay-fidelity checks and for exercising the harness.

Usage::

    uv run python -m energy_oil_forecasting.cfm_coach.backfill
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from energy_oil_forecasting.cfm_agent_v_5_0.config import AGENT_NAME, CfmV50Settings
from energy_oil_forecasting.cfm_coach.config import (
    BASELINE_CALIBRATION_VERSION,
    DEFAULT_SETTINGS,
    CoachSettings,
)
from energy_oil_forecasting.cfm_coach.run_store import RunRecordStore, build_run_record
from energy_oil_forecasting.cfm_coach.schemas import RunRecord


_PREDICTOR_PREFIX = f"agent_predictor_{AGENT_NAME}_"
_PREDICTOR_SUFFIXES = ("_continuous", "_binary", "_categorical")

# What `run_cfm_agent_v_5_0_interactive.py::build_predictor` uses. The interactive
# audit predates R6 and never persisted its settings, so they are reconstructed
# here and flagged `inferred` -- never presented as observed.
_INTERACTIVE_SETTINGS = CfmV50Settings(
    audit_enabled=True,
    code_execution_enabled=True,
    policy_mode="constrained_actions",
)


def model_from_predictor_id(predictor_id: str) -> str:
    """Recover the model name from a predictor id, the only place it is recorded today."""
    name = predictor_id.removeprefix(_PREDICTOR_PREFIX)
    for suffix in _PREDICTOR_SUFFIXES:
        name = name.removesuffix(suffix)
    return name or "unknown"


def _predictions_from_audit(payload: Any) -> list[dict[str, Any]]:
    """Normalise either audit shape into a plain list of prediction dicts."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "forecasts" in payload:
        return [entry["prediction"] for entry in payload["forecasts"]]
    raise ValueError("unrecognised audit-file shape: expected a list of predictions or a dict with 'forecasts'")


def record_from_audit_file(path: Path, *, schema_version: int) -> RunRecord:
    payload = json.loads(path.read_text(encoding="utf-8"))
    predictions = _predictions_from_audit(payload)

    # The e2e harness happened to dump its settings; the interactive script did not.
    recorded = payload.get("settings") if isinstance(payload, dict) else None
    settings = CfmV50Settings(**recorded) if recorded else _INTERACTIVE_SETTINGS

    return build_run_record(
        predictions,
        settings=settings,
        calibration_version=BASELINE_CALIBRATION_VERSION,
        agent_model=model_from_predictor_id(predictions[0]["predictor_id"]),
        schema_version=schema_version,
        # Issued long after the cutoff: the research pipeline searched today's web
        # for a date months in the past, so the source selection is shaped by what
        # turned out to matter. Not fitting evidence.
        provenance="replayed_live_search",
        settings_source="recorded" if recorded else "inferred",
    )


def backfill(paths: list[Path], settings: CoachSettings = DEFAULT_SETTINGS) -> list[RunRecord]:
    store = RunRecordStore(settings)
    records = []
    for path in paths:
        record = record_from_audit_file(path, schema_version=settings.run_record_schema_version)
        store.save(record)
        lag = (record.issued_at.date() - record.cutoff).days
        print(
            f"  {path.name}\n"
            f"    -> {record.run_id}\n"
            f"       cutoff={record.cutoff}  issued={record.issued_at.date()}  (+{lag} days after cutoff)\n"
            f"       horizons={record.horizons}  settings={record.settings_source}  provenance={record.provenance}"
        )
        records.append(record)
    return records


def main() -> None:
    repo_root = Path.cwd()
    paths = sorted(repo_root.glob("cfm_agent_v_5_0_*audit*.json"))
    if not paths:
        print(f"No pre-coach audit files found in {repo_root}.")
        return
    print(f"Backfilling {len(paths)} audit file(s) into RunRecords:\n")
    records = backfill(paths)
    print(f"\nWrote {len(records)} record(s) to {DEFAULT_SETTINGS.runs_dir}")
    print(
        "\nNote: every backfilled record is provenance='replayed_live_search' and is "
        "excluded from fitting evidence (R7).\nThey are usable for replay-fidelity "
        "checks and for exercising the harness, not for calibrating constants."
    )


if __name__ == "__main__":
    main()
