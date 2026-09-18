"""Apply a proposal the team has accepted. Run by a human; never by the weekly job.

- numeric: copy the emitted candidate JSON into the coach-owned ledger for the
  stream (`CalibrationLedger.save`, refuses overwrite) and record `applied_as`;
- text: draft the challenger package (`review/challenger.py`) and record its path;
- code/data: nothing to apply; the brief is the deliverable.

It never writes into the live streams' ``calibration_v52_*`` directories.

Usage (from the repo root)::

    uv run python -m energy_oil_forecasting.cfm_coach.review.accept --proposal P-0003 --stream v52_lite --by naman
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import yaml
from energy_oil_forecasting.cfm_coach.review.challenger import TEXT_LEVERS, draft_challenger
from energy_oil_forecasting.cfm_coach.review.coached import coach_ledger, ensure_baseline
from energy_oil_forecasting.cfm_coach.review.memory import Proposal, ProposalStore
from energy_oil_forecasting.cfm_coach.review.numeric import next_business_day, next_version_name
from energy_oil_forecasting.cfm_coach.review.settings import DEFAULT_REVIEW_SETTINGS, ReviewSettings
from energy_oil_forecasting.cfm_coach.schemas import CalibrationVersion
from energy_oil_forecasting.cfm_coach.streams import stream_for


def _write_decision(store: ProposalStore, proposal: Proposal, **fields) -> None:
    path = store.path_for(proposal.id)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["decision"] = {**payload.get("decision", {}), **fields}
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    Proposal.model_validate(payload)


def apply_numeric(
    proposal: Proposal, stream_id: str, *, settings: ReviewSettings = DEFAULT_REVIEW_SETTINGS, by: str
) -> Path:
    files = json.loads(proposal.artifacts.get("candidate_files", "{}"))
    candidate_path = files.get(stream_id)
    if not candidate_path:
        raise SystemExit(f"{proposal.id} has no candidate file for {stream_id}; candidates: {files}")
    stream = stream_for(stream_id)
    ledger = coach_ledger(stream, settings)
    ensure_baseline(ledger)
    if "calibration_v52" in str(ledger.directory):
        raise SystemExit("refusing to write into a live stream's ledger")
    candidate = CalibrationVersion.model_validate_json(Path(candidate_path).read_text(encoding="utf-8"))
    version = candidate.model_copy(
        update={
            "version": next_version_name(ledger),
            "effective_from": next_business_day(date.today()),
            "parent_version": ledger.current(date.today()).version,
            "source_proposal_id": proposal.id,
        }
    )
    ledger.to_agent_settings(version, base=stream.base_settings, target=stream.target)
    path = ledger.save(version)
    _write_decision(
        ProposalStore(settings),
        proposal,
        status="accepted",
        by=by,
        on=date.today().isoformat(),
        applied_as={
            "calibration_version": version.version,
            "stream": stream_id,
            "effective_from": version.effective_from.isoformat(),
        },
    )
    return path


def apply_text(proposal: Proposal, *, settings: ReviewSettings = DEFAULT_REVIEW_SETTINGS, by: str) -> Path:
    draft = draft_challenger(proposal, settings=settings)
    _write_decision(
        ProposalStore(settings),
        proposal,
        status="accepted",
        by=by,
        on=date.today().isoformat(),
        applied_as={
            "challenger_path": draft.package_dir,
            "stream_id": draft.stream_id,
            "register": draft.register_path,
        },
    )
    return Path(draft.register_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--proposal", required=True)
    parser.add_argument("--stream", default=None, help="for numeric proposals: which stream's coach ledger")
    parser.add_argument("--by", required=True)
    args = parser.parse_args(argv)
    store = ProposalStore()
    proposal = next((p for p in store.load_all() if p.id == args.proposal), None)
    if proposal is None:
        raise SystemExit(f"no proposal {args.proposal}")
    kind = proposal.lever.get("kind")
    if kind in ("layer", "settings_overlay", "ensemble_weights"):
        if not args.stream:
            raise SystemExit("--stream is required for a numeric proposal")
        print(f"saved {apply_numeric(proposal, args.stream, by=args.by)}")
    elif kind in TEXT_LEVERS:
        print(f"drafted; follow {apply_text(proposal, by=args.by)}")
    else:
        _write_decision(
            store,
            proposal,
            status="accepted",
            by=args.by,
            on=date.today().isoformat(),
            applied_as={"brief": proposal.artifacts.get("brief_file", "")},
        )
        print(f"{proposal.id} accepted; nothing to apply automatically (lever {kind})")


if __name__ == "__main__":
    main()


__all__ = ["apply_numeric", "apply_text"]
