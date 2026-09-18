"""The coach's own forecast: yesterday's v5.2 runs, replayed under the coach-owned ledger, published beside them.

Decided 2026-09-13: numeric proposals never touch v5.2 or its ledgers. Instead,
every scheduled stream gets a *coached* sibling corpus. Each morning's records
are replayed -- exactly, no LLM, no network (`ReplayEngine.replay`) -- under the
version in force in ``review/calibration/<stream>/`` and written to
``review/coached/<stream>/`` as records of their own. The report scores them as
the ``coached`` variant beside random walk, ensemble and agent, which is the
scoreboard's ``current`` column made real and dated.

Honesty: a coached record is issued hours after the run it derives from, and
says so (``audit_signals.coached.issued_lag_days``). Its provenance is
``coached_replay``, which `fitting_eligible_provenance` excludes, so the coach
never fits on its own output.

Usage (from the repo root; the launchd job runs this weekdays at 12:00)::

    uv run python -m energy_oil_forecasting.cfm_coach.review.coached [--stream v52_lite] [--dry-run]
"""

from __future__ import annotations

import argparse
from datetime import date, datetime
from typing import Any

from energy_oil_forecasting.cfm_coach.config import CoachSettings
from energy_oil_forecasting.cfm_coach.ledger import CalibrationLedger
from energy_oil_forecasting.cfm_coach.replay import ReplayEngine
from energy_oil_forecasting.cfm_coach.review.settings import DEFAULT_REVIEW_SETTINGS, ReviewSettings
from energy_oil_forecasting.cfm_coach.run_store import RunRecordStore
from energy_oil_forecasting.cfm_coach.schemas import CalibrationVersion, HorizonRecord, RunRecord
from energy_oil_forecasting.cfm_coach.streams import SCHEDULED_STREAMS, RunStream, stream_for


COACHED_SUFFIX = "__coached"
COACHED_PROVENANCE = "coached_replay"
COACHED_PREDICTOR_ID = "cfm_coach_coached_predictor"
BASELINE_VERSION = "v001"


def coach_ledger(stream: RunStream, settings: ReviewSettings = DEFAULT_REVIEW_SETTINGS) -> CalibrationLedger:
    """The coach-owned ledger for one stream, separate from the stream's own ``calibration_*`` directory."""
    return CalibrationLedger(
        CoachSettings(target_agent=stream.target.agent_id, calibration_dir=settings.calibration_dir / stream.stream_id)
    )


def ensure_baseline(ledger: CalibrationLedger) -> CalibrationVersion:
    """Write the identity ``v001`` if the ledger is empty, so `current()` resolves for any cutoff."""
    try:
        return ledger.load(BASELINE_VERSION)
    except FileNotFoundError:
        baseline = CalibrationVersion(
            version=BASELINE_VERSION,
            effective_from=date(2004, 1, 1),
            notes="Coach-owned baseline: identity. Accepted numeric proposals are saved as later versions here.",
        )
        ledger.save(baseline)
        return baseline


def coached_store(stream: RunStream, settings: ReviewSettings = DEFAULT_REVIEW_SETTINGS) -> RunRecordStore:
    return RunRecordStore(
        CoachSettings(target_agent=stream.target.agent_id, runs_dir=settings.coached_dir / stream.stream_id)
    )


def coached_run_id(source_run_id: str) -> str:
    return f"{source_run_id}{COACHED_SUFFIX}"


def coached_record(
    record: RunRecord,
    version: CalibrationVersion,
    replay: ReplayEngine,
    *,
    issued_at: datetime | None = None,
) -> RunRecord:
    """`record` replayed under `version`, as a record of its own. Bit-identical under the identity."""
    issued = issued_at or datetime.now()
    replayed = {item.horizon: item for item in replay.replay(record, version)}
    forecasts: list[HorizonRecord] = []
    for item in record.forecasts:
        out = replayed[item.horizon]
        forecasts.append(
            item.model_copy(
                update={
                    "final_point_forecast": out.point_forecast,
                    "final_quantiles": dict(out.quantiles),
                    "policy_decision": out.decision.model_dump(mode="json")
                    if hasattr(out.decision, "model_dump")
                    else dict(out.decision),
                    "forecast_transformation": {
                        **out.transformation,
                        "coach_layer": {
                            "calibration_version": version.version,
                            "centre_gain": version.layer.gain_for(item.horizon),
                            "width_scale": version.layer.scale_for(item.horizon),
                            "rw_anchor": version.layer.anchor_for(item.horizon),
                            "engine_point_forecast": out.engine_point_forecast,
                        },
                    },
                }
            )
        )
    settings = replay.agent_settings_for(record, version)
    audit: dict[str, Any] = {
        **record.audit_signals,
        "coached": {
            "source_run_id": record.run_id,
            "source_calibration_version": record.calibration_version,
            "issued_lag_days": (issued.date() - record.cutoff).days,
            "coach_version": version.version,
        },
    }
    return record.model_copy(
        update={
            "run_id": coached_run_id(record.run_id),
            "predictor_id": COACHED_PREDICTOR_ID,
            "calibration_version": version.version,
            "settings": settings.model_dump(mode="json"),
            "provenance": COACHED_PROVENANCE,
            "issued_at": issued,
            "forecasts": forecasts,
            "audit_signals": audit,
        }
    )


def run_stream(
    stream: RunStream,
    settings: ReviewSettings = DEFAULT_REVIEW_SETTINGS,
    *,
    records: list[RunRecord] | None = None,
    dry_run: bool = False,
    now: datetime | None = None,
) -> list[RunRecord]:
    """Coach every record in `stream` that has no coached sibling yet. Idempotent."""
    ledger = coach_ledger(stream, settings)
    ensure_baseline(ledger)
    store = coached_store(stream, settings)
    existing = {path.stem for path in store.directory.glob("*.json")} if store.directory.exists() else set()
    source = records if records is not None else RunRecordStore(stream.coach_settings).load_all()
    replay = ReplayEngine(stream.coach_settings, target=stream.target)
    written: list[RunRecord] = []
    for record in source:
        if record.provenance == COACHED_PROVENANCE or coached_run_id(record.run_id) in existing:
            continue
        version = ledger.current(record.cutoff)
        coached = coached_record(record, version, replay, issued_at=now)
        if not dry_run:
            store.save(coached)
        written.append(coached)
    return written


def load_coached(stream: RunStream, settings: ReviewSettings = DEFAULT_REVIEW_SETTINGS) -> dict[str, RunRecord]:
    """Coached records keyed by their *source* run id."""
    store = coached_store(stream, settings)
    return {r.audit_signals["coached"]["source_run_id"]: r for r in store.load_all()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stream", action="append", default=None, help="repeatable; default: every scheduled stream")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    streams = tuple(stream_for(s) for s in args.stream) if args.stream else SCHEDULED_STREAMS
    for stream in streams:
        written = run_stream(stream, dry_run=args.dry_run)
        ledger = coach_ledger(stream)
        version = ledger.current(date.today())
        print(
            f"{stream.stream_id}: {'would write' if args.dry_run else 'wrote'} {len(written)} coached record(s) under {version.version}"
            f" ({'identity' if version.is_baseline else 'calibrated'}); dir {coached_store(stream).directory}"
        )
        for record in written[:5]:
            lag = record.audit_signals["coached"]["issued_lag_days"]
            print(f"  {record.run_id} (lag {lag}d)")


if __name__ == "__main__":
    main()


__all__ = [
    "BASELINE_VERSION",
    "COACHED_PREDICTOR_ID",
    "COACHED_PROVENANCE",
    "COACHED_SUFFIX",
    "coach_ledger",
    "coached_record",
    "coached_run_id",
    "coached_store",
    "ensure_baseline",
    "load_coached",
    "run_stream",
]
