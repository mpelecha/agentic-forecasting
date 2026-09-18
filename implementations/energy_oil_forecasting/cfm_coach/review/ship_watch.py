"""What changed underneath the coach, and what it did to the numbers.

Watched weekly: the v5.2 package fingerprint (a new build is a new agent), the
coach-owned ledgers (an accepted numeric version), and the stream registry (a
registered challenger). For each accepted proposal with an `applied_as`, the
before/after comparison is by distinct cutoff and withheld below six cutoffs.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import pandas as pd
from energy_oil_forecasting.cfm_coach.review.coached import coach_ledger
from energy_oil_forecasting.cfm_coach.review.collect import AGENT, COACHED, RANDOM_WALK, StreamCorpus
from energy_oil_forecasting.cfm_coach.review.memory import Proposal
from energy_oil_forecasting.cfm_coach.review.settings import DEFAULT_REVIEW_SETTINGS, ReviewSettings
from energy_oil_forecasting.cfm_coach.run_store import agent_package_fingerprint
from energy_oil_forecasting.cfm_coach.streams import STREAMS
from energy_oil_forecasting.cfm_coach.targets import V52


MIN_POST_SHIP_CUTOFFS = 6


def snapshot(settings: ReviewSettings = DEFAULT_REVIEW_SETTINGS) -> dict[str, Any]:
    streams = {s.stream_id: {"model": s.model, "agent": s.target.agent_id} for s in STREAMS}
    ledgers = {}
    for stream in STREAMS:
        try:
            ledgers[stream.stream_id] = [v.version for v in coach_ledger(stream, settings).all_versions()]
        except Exception:  # noqa: BLE001
            ledgers[stream.stream_id] = []
    return {
        "fingerprint_v52": agent_package_fingerprint(V52),
        "streams": streams,
        "coach_ledgers": ledgers,
        "at": date.today().isoformat(),
    }


def diff(previous: dict[str, Any] | None, current: dict[str, Any]) -> dict[str, Any]:
    if not previous:
        return {"first_snapshot": True}
    out: dict[str, Any] = {}
    if previous.get("fingerprint_v52") != current.get("fingerprint_v52"):
        out["fingerprint_changed"] = {"from": previous.get("fingerprint_v52"), "to": current.get("fingerprint_v52")}
    new_streams = sorted(set(current["streams"]) - set(previous.get("streams", {})))
    if new_streams:
        out["new_streams"] = new_streams
    for sid, versions in current["coach_ledgers"].items():
        added = sorted(set(versions) - set(previous.get("coach_ledgers", {}).get(sid, [])))
        if added:
            out.setdefault("new_coach_versions", {})[sid] = added
    return out


def post_ship(proposal: Proposal, corpora: dict[str, StreamCorpus]) -> dict[str, Any]:
    """Before vs after the applied date, by distinct cutoff, on the variant the proposal changed."""
    applied = proposal.decision.applied_as or {}
    on = proposal.decision.on
    if not applied or on is None:
        return {"status": "not applied"}
    variant = COACHED if applied.get("calibration_version") else AGENT
    out: dict[str, Any] = {"applied_as": applied, "on": on.isoformat(), "variant": variant}
    for sid in proposal.stream_scope.get("streams") or list(corpora):
        corpus = corpora.get(sid)
        if corpus is None:
            continue
        rows = []
        for (run_id, h, v), card in corpus.scores.items():
            if v != variant:
                continue
            rw = corpus.score(run_id, h, RANDOM_WALK)
            rows.append(
                {
                    "cutoff": card.cutoff,
                    "horizon": h,
                    "after": card.cutoff > on,
                    "gap_vs_rw": card.pinball - (rw.pinball if rw else float("nan")),
                    "covered": float(card.covered_80),
                }
            )
        frame = pd.DataFrame(rows)
        if frame.empty:
            out[sid] = {"status": "no rows"}
            continue
        after = frame[frame["after"]]
        before = frame[~frame["after"]]
        n_after = after["cutoff"].nunique()
        out[sid] = {
            "cutoffs_before": int(before["cutoff"].nunique()),
            "cutoffs_after": int(n_after),
            "gap_vs_rw_before": float(before["gap_vs_rw"].mean()) if not before.empty else None,
            "gap_vs_rw_after": float(after["gap_vs_rw"].mean()) if not after.empty else None,
            "coverage_before": float(before["covered"].mean()) if not before.empty else None,
            "coverage_after": float(after["covered"].mean()) if not after.empty else None,
            "verdict": "withheld"
            if n_after < MIN_POST_SHIP_CUTOFFS
            else (
                "improved"
                if not before.empty and after["gap_vs_rw"].mean() < before["gap_vs_rw"].mean()
                else "not improved"
            ),
        }
    return out


def run(
    settings: ReviewSettings, proposals: list[Proposal], corpora: dict[str, StreamCorpus]
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = settings.data_dir / "ship_watch.json"
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    current = snapshot(settings)
    changes = diff(previous, current)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
    verdicts = {p.id: post_ship(p, corpora) for p in proposals if p.decision.status == "accepted"}
    return changes, verdicts


__all__ = ["MIN_POST_SHIP_CUTOFFS", "diff", "post_ship", "run", "snapshot"]
