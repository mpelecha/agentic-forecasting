"""Workflow A -- one live forecast, recorded.

    stream -> calibration in force -> agent settings -> run agent -> calibration layer -> RunRecord

This supersedes ``run_cfm_agent_v_5_0_interactive.py`` for daily use. That script keeps
working and is untouched; it simply does not persist the five identity fields a
calibration corpus needs (R6), which is the only thing added here.

Every business day produces a record whether or not anything is learned from it.
This loop *is* the corpus, and the corpus cannot be recovered after the fact:
re-running at a past cutoff searches today's web, which is why such runs are marked
``replayed_live_search`` and excluded from fitting evidence.

**Which LLM answers is a property of the stream, not an argument.** A run makes five
distinct LLM calls -- the main agent, the grounded search, the search leakage
verifier, the claim-support verifier, and (v5.2 only) the structured-output retry.
Two are constructor arguments and three are settings fields, so binding only
``model=`` would leave three calls on the package default and make a nominally
single-model run a mixed-model one. `RunStream.build_config` and
`RunStream.base_settings` bind both halves together, and `run_once` prints every
resolved model before spending a token so a mis-binding is visible in the log
rather than buried in a record.

Usage::

    uv run python -m energy_oil_forecasting.cfm_coach.run_daily [YYYY-MM-DD] [--stream=ID]
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

from aieng.forecasting.evaluation.task import ForecastingTask
from energy_oil_forecasting.cfm_coach.config import CoachSettings
from energy_oil_forecasting.cfm_coach.ledger import CalibrationLedger
from energy_oil_forecasting.cfm_coach.run_store import RunRecordStore
from energy_oil_forecasting.cfm_coach.schemas import Provenance, RunRecord
from energy_oil_forecasting.cfm_coach.streams import (
    BASE_SETTINGS_OVERRIDES,
    DEFAULT_STREAM,
    RunStream,
    stream_for,
)
from energy_oil_forecasting.data import (
    DEFAULT_WTI_COVARIATE_SERIES_IDS,
    build_wti_multivariate_service,
)


#: Kept as a module-level name because it documented the run-level (non-calibration)
#: choices for two milestones and is referenced from HANDOFF.md. The values now live
#: in `streams.BASE_SETTINGS_OVERRIDES` so every stream shares one definition.
BASE_AGENT_SETTINGS = DEFAULT_STREAM.target.settings(**BASE_SETTINGS_OVERRIDES)


_load_dotenv: Callable[..., Any] | None
try:
    from dotenv import load_dotenv as _load_dotenv
except ImportError:
    _load_dotenv = None


def _repo_root() -> Path | None:
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "aieng-forecasting").is_dir():
            return candidate
    return None


def load_credentials() -> None:
    """Populate os.environ from the repo-root ``.env`` (same pattern as ``scripts/fetch_fred.py``).

    This runs unattended on a schedule, so it cannot assume an interactive shell
    has already exported ``GEMINI_API_KEY`` and the proxy variables.
    """
    root = _repo_root()
    if _load_dotenv is None or root is None:
        return
    _load_dotenv(root / ".env", override=False)


def build_task(cutoff: str, settings: CoachSettings, *, agent_id: str) -> ForecastingTask:
    return ForecastingTask(
        task_id=f"cfm_coach_daily_{settings.target_series_id}_{cutoff}",
        target_series_id=settings.target_series_id,
        horizons=list(settings.horizons),
        frequency=settings.frequency,
        description=(
            f"Daily coach-recorded probabilistic WTI forecast. Produced by {agent_id} "
            "under a versioned calibration; recorded as a RunRecord for later replay."
        ),
    )


def classify_provenance(cutoff: date, today: date | None = None) -> Provenance:
    """Mark a run as forward evidence only when it is issued on its own cutoff.

    Anything else re-searches today's web for a past date. Even when every cited
    source predates the cutoff, the *selection* is shaped by what turned out to
    matter -- a bias the agent's leakage verifier cannot see, because it inspects
    content rather than retrieval.
    """
    return "live_forward" if cutoff >= (today or date.today()) else "replayed_live_search"


def run_once(cutoff: str, *, stream: RunStream = DEFAULT_STREAM) -> RunRecord:
    """Run `stream` once at `cutoff` and write the record to that stream's corpus."""
    load_credentials()
    settings = stream.coach_settings
    ledger = CalibrationLedger(settings)
    calibration = ledger.current(date.fromisoformat(cutoff))
    agent_settings = ledger.to_agent_settings(
        calibration,
        base=stream.base_settings,
        target=stream.target,
    )
    provenance = classify_provenance(date.fromisoformat(cutoff))

    print(f"Stream:      {stream.stream_id}  --  {stream.description}")
    print(f"Agent:       {stream.target.agent_id}")
    print(f"Models:      {stream.models_in_use(agent_settings)}")
    print(f"Cutoff:      {cutoff}   provenance={provenance}")
    print(f"Corpus:      {settings.runs_dir}")
    print(f"Calibration: {calibration.version}  (baseline={calibration.is_baseline})")
    if not calibration.is_baseline:
        print(f"             overlay={calibration.settings_overlay}  layer={calibration.layer.model_dump()}")

    service = build_wti_multivariate_service()
    covariates = [name for name in DEFAULT_WTI_COVARIATE_SERIES_IDS if name in service.series_ids]
    task = build_task(cutoff, settings, agent_id=stream.target.agent_id)
    context = service.context(as_of=datetime.strptime(cutoff, "%Y-%m-%d"))

    config = stream.build_config(
        data_service=service,
        settings=agent_settings,
        covariate_series_ids=covariates,
    )
    predictor = stream.target.build_predictor(config)

    print("\nRunning task-bound workflow...\n")
    predictions = predictor.predict(task, context)

    store = RunRecordStore(settings)
    record = store.write(
        predictions,
        settings=agent_settings,
        calibration_version=calibration.version,
        agent_model=stream.model,
        target=stream.target,
        provenance=provenance,
        run_id_prefix=stream.run_id_prefix,
    )
    _report(record, calibration.layer, store)
    return record


def _report(record: RunRecord, layer, store: RunRecordStore) -> None:  # noqa: ANN001 - CalibrationLayer, kept loose
    print("\nForecasts (after calibration layer):")
    for horizon_record in record.forecasts:
        point, quantiles = layer.apply(
            ensemble_p50=horizon_record.ensemble_quantiles[0.5],
            final_point_forecast=horizon_record.final_point_forecast,
            final_quantiles=horizon_record.final_quantiles,
            horizon=horizon_record.horizon,
        )
        ensemble_p50 = horizon_record.ensemble_quantiles[0.5]
        print(
            f"  h={horizon_record.horizon:<3} {horizon_record.forecast_date.date()}  "
            f"P10={quantiles[0.1]:7.2f}  P50={point:7.2f}  P90={quantiles[0.9]:7.2f}   "
            f"width={quantiles[0.9] - quantiles[0.1]:5.2f}  "
            f"(ensemble P50={ensemble_p50:.2f}, overlay {point - ensemble_p50:+.2f})"
        )

    assessment = record.assessment
    print(
        f"\nAssessment:  physical_status={assessment['physical_status']}  "
        f"novelty={assessment['incremental_novelty']}  confidence={assessment['confidence']}"
    )
    tiers = {item.horizon: item.policy_decision.get("evidence_tier") for item in record.forecasts}
    print(f"Evidence:    tiers={tiers}  conflict={assessment['material_evidence_conflict']}")
    print(f"\nRecord:      {store.path_for(record.run_id)}")


def parse_args(argv: list[str]) -> tuple[str, tuple[RunStream, ...] | None]:
    """Accept ``[cutoff] [--stream=ID ...]`` in any order.

    ``--stream`` may be repeated, so a morning can be run for a subset without
    editing the schedule -- the case that motivated it was v5.0 having already
    recorded its runs for a cutoff while the v5.2 streams still needed theirs.
    Returns ``None`` for the streams when none was named, which lets each entry
    point apply its own default rather than guessing here.
    """
    cutoff: str | None = None
    streams: list[RunStream] = []
    for arg in argv:
        if arg.startswith("--stream="):
            for name in arg.split("=", 1)[1].split(","):
                stream = stream_for(name.strip())
                if stream not in streams:
                    streams.append(stream)
        elif cutoff is None:
            cutoff = arg
        else:
            raise SystemExit(f"unexpected argument {arg!r}; usage: [YYYY-MM-DD] [--stream=ID ...]")
    return cutoff or date.today().isoformat(), tuple(streams) or None


def main() -> None:
    cutoff, streams = parse_args(sys.argv[1:])
    if streams and len(streams) > 1:
        raise SystemExit("run_daily runs one stream; use run_daily_all for several")
    run_once(cutoff, stream=streams[0] if streams else DEFAULT_STREAM)


if __name__ == "__main__":
    main()
