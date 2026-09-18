"""Generate a shareable HTML browser for the whole run corpus.

Every record the coach has ever written, filterable, with the full JSON one click
away. Built as a script rather than a hand-made file so it can be regenerated as
the corpus grows -- the numbers move every weekday.

    uv run python -m energy_oil_forecasting.cfm_coach.make_corpus_page

The derived columns (fidelity, resolution, scores) come from the coach's own
modules rather than being recomputed here, so this page and ``report.py`` can
never disagree about what a record says.

**Size.** Each record embeds its complete JSON so the page is self-contained --
no server, no fetch, works from a file:// URL or a published artifact. That costs
~75 KB minified per record, so the whole page grows ~18 MB/year at daily cadence
and will eventually exceed the 16 MB artifact ceiling. ``SIZE_BUDGET_BYTES``
guards it: past the soft limit the script drops the bulkiest audit blob (the
source-validation page archive, ~55 KB of each record) from the embedded copy and
says so on the page, rather than silently truncating.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

from aieng.forecasting.models import ADVANCED_MODEL, LITE_MODEL
from energy_oil_forecasting.cfm_coach.config import DEFAULT_SETTINGS
from energy_oil_forecasting.cfm_coach.outcomes import OutcomeResolver
from energy_oil_forecasting.cfm_coach.replay import ReplayEngine
from energy_oil_forecasting.cfm_coach.run_store import RunRecordStore
from energy_oil_forecasting.cfm_coach.schemas import HorizonRecord, RunRecord
from energy_oil_forecasting.cfm_coach.scoring import ForecastScorer
from energy_oil_forecasting.cfm_coach.streams import STREAMS, RunStream, stream_for


#: One page covering every stream. The corpora stay separate on disk and must never
#: be *pooled* for fitting, so every record carries its stream, agent and model, the
#: summary tiles are per stream rather than over the union, and one filter isolates
#: a stream. What a reader actually wants is the comparison -- what did v5.0 and each
#: v5.2 arm say on the same cutoff -- and three pages make that a tab-switching chore.
OUTPUT = Path(__file__).resolve().parent / "corpus_page.html"

#: Soft ceiling for the embedded JSON. The artifact hard cap is 16 MB; stop well
#: short so the surrounding markup, CSS and script still fit comfortably.
SIZE_BUDGET_BYTES = 11 * 1024 * 1024

#: Dropped first when the budget is exceeded: a page-fetch archive with extracted
#: passages, by far the largest block and the least useful to a human reader.
HEAVY_AUDIT_KEY = "source_validation_audit"

#: The adaptive analyst (`adaptive_agent/`) is architecturally different from every
#: cfm_coach stream -- a single persistent analyst that mutates its own skill file,
#: with no numerical ensemble, no evidence-tier gate, and no coach calibration layer.
#: `adaptive_agent/run_daily.py`'s own docstring is explicit that it does not fit
#: `RunStream`/`AgentTarget`'s shape and was deliberately given its own script and
#: launchd job rather than a fourth entry in `streams.STREAMS`. Its records are read
#: directly here (not through `RunRecordStore`, which is scoped to a `CoachSettings`
#: this agent doesn't have) and rendered with a card of their own -- shown on the
#: same page for reference, never implied comparable to the three streams above it.
ADAPTIVE_AGENT_ID = "adaptive_agent"
ADAPTIVE_RUNS_DIR = Path(__file__).resolve().parent.parent / "adaptive_agent" / "runs"
#: Same 5/10/21 business-day convention as every cfm_coach stream (`adaptive_agent
#: /run_daily.py`'s `HORIZONS`), and in the same order the agent writes its
#: predictions -- confirmed against every run file on disk, not assumed.
ADAPTIVE_HORIZONS = (5, 10, 21)


def _short_fingerprint(value: str) -> str:
    return value.rsplit(":", 1)[-1][:12] if ":" in value else value[:12]


def _adaptive_model(predictor_id: str) -> str:
    """Recover the model from a predictor id by checking for the two known constants.

    Not a prefix/suffix strip like `backfill.model_from_predictor_id` -- that helper
    is keyed to `cfm_agent_v_5_0`'s predictor-id shape. This agent's differs, and a
    substring check against the models this codebase actually names is more robust
    than parsing a string format that isn't this module's to own.
    """
    for model in (ADVANCED_MODEL, LITE_MODEL):
        if model in predictor_id:
            return model
    return "unknown"


def _load_adaptive_records() -> list[RunRecord]:
    """Read the adaptive agent's own run files as genuine, not fabricated, `RunRecord`s.

    Only fields this agent actually produces are populated: the real cutoff, the real
    per-horizon point forecast and quantiles, the real rationale text. Fields that
    presuppose cfm_coach's ensemble-plus-policy architecture -- a structured
    assessment, a research packet, audit signals, a package fingerprint -- are left
    genuinely empty rather than filled with placeholders, because this agent does not
    produce that kind of output; inventing plausible-looking values for them would be
    worse than leaving them out.

    `ensemble_*` and `final_*` are set to the same real forecast. This agent has no
    separate numerical baseline for an LLM to overlay -- one model proposes the
    forecast directly -- so there is nothing else honest to put in the "ensemble"
    slot. The page's overlay/tier machinery is not applied to these rows for exactly
    this reason (see `collect_adaptive`).

    Reusing `RunRecord`/`HorizonRecord` here (rather than a bespoke shape) is what
    lets `OutcomeResolver` and `ForecastScorer` resolve and score these forecasts
    against realized WTI honestly, with no reimplementation -- both only ever read
    `run_id`, `cutoff`, and each horizon's `forecast_date`/`final_quantiles`.
    """
    records: list[RunRecord] = []
    for path in sorted(ADAPTIVE_RUNS_DIR.glob("adaptive_agent__*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        preds = payload["predictions"]
        model = _adaptive_model(preds[0]["predictor_id"])
        forecasts = [
            HorizonRecord(
                horizon=horizon,
                forecast_date=pred["forecast_date"],
                ensemble_point_forecast=pred["payload"]["point_forecast"],
                ensemble_quantiles={float(k): v for k, v in pred["payload"]["quantiles"].items()},
                final_point_forecast=pred["payload"]["point_forecast"],
                final_quantiles={float(k): v for k, v in pred["payload"]["quantiles"].items()},
                forecast_transformation={"horizon_rationale": pred.get("metadata", {}).get("horizon_rationale", "")},
            )
            for horizon, pred in zip(ADAPTIVE_HORIZONS, preds, strict=True)
        ]
        records.append(
            RunRecord(
                schema_version=1,
                run_id=payload["run_id"],
                agent_id=ADAPTIVE_AGENT_ID,
                agent_model=model,
                predictor_id=preds[0]["predictor_id"],
                package_fingerprint="adaptive_agent:no_manifest",
                calibration_version="n/a",
                prompt_version="adaptive_agent_skill_live",
                settings={},
                provenance=payload["provenance"],
                task_id=payload["task_id"],
                cutoff=payload["cutoff"],
                horizons=list(ADAPTIVE_HORIZONS),
                issued_at=preds[0]["issued_at"],
                assessment={"rationale": preds[0].get("metadata", {}).get("rationale", "")},
                research_packet={},
                forecasts=forecasts,
            )
        )
    return records


def collect_stream(stream: RunStream) -> dict[str, Any]:
    """Assemble one stream's records plus its derived evaluation columns."""
    settings = stream.coach_settings
    target = stream.target
    store = RunRecordStore(settings)
    records = store.load_all()
    replay = ReplayEngine(settings, target=target)
    resolver = OutcomeResolver(settings)
    scorer = ForecastScorer()

    resolution = resolver.resolve_all(records)
    resolved_by = {(item.run_id, item.horizon): item for item in resolution.resolved}
    pending_by = {(item.run_id, item.horizon): item for item in resolution.pending}

    rows: list[dict[str, Any]] = []
    for record in records:
        report = replay.check_fidelity(record)
        assessment = record.assessment
        packet = record.research_packet
        audit = record.audit_signals

        proposed = {int(action["horizon"]): action for action in assessment.get("horizon_actions", [])}

        horizons: list[dict[str, Any]] = []
        for item in record.forecasts:
            decision = item.policy_decision or {}
            action = proposed.get(item.horizon, {})
            key = (record.run_id, item.horizon)
            resolved = resolved_by.get(key)
            pending = pending_by.get(key)

            entry: dict[str, Any] = {
                "horizon": item.horizon,
                "forecast_date": str(item.forecast_date.date()),
                "ensemble_p50": item.ensemble_quantiles[0.5],
                "p10": item.final_quantiles[0.1],
                "p50": item.final_point_forecast,
                "p90": item.final_quantiles[0.9],
                "width": item.final_quantiles[0.9] - item.final_quantiles[0.1],
                "overlay": item.final_point_forecast - item.ensemble_quantiles[0.5],
                "proposed_center": action.get("center_action", "--"),
                "proposed_uncertainty": action.get("uncertainty_action", "--"),
                "granted_center": decision.get("center_action", "--"),
                "granted_uncertainty": decision.get("uncertainty_action", "--"),
                "tier": decision.get("evidence_tier", "none"),
                "tier_level": decision.get("evidence_tier_level", 0),
                "eligible": bool(decision.get("eligible", False)),
                "publishers": len(decision.get("resolved_publishers", []) or []),
                "reasons": decision.get("eligibility_reasons", []) or [],
                "capped": (
                    action.get("center_action") not in (None, decision.get("center_action"))
                    or action.get("uncertainty_action") not in (None, decision.get("uncertainty_action"))
                ),
            }

            if resolved is not None:
                card = scorer.score_recorded(record, resolved, variant="agent")
                ensemble_card = scorer.score_ensemble(record, resolved)
                entry |= {
                    "status": "resolved",
                    "realized": resolved.realized_value,
                    "observation_date": str(resolved.realized_observation_date),
                    "pinball": card.pinball,
                    "ensemble_pinball": ensemble_card.pinball,
                    "abs_error": card.absolute_error,
                    "covered": card.covered_80,
                }
            else:
                entry |= {
                    "status": "pending",
                    "pending_reason": pending.reason if pending else "not resolved",
                }
            horizons.append(entry)

        overlays = [abs(item["overlay"]) for item in horizons]
        tiers = [item["tier"] for item in horizons]
        rows.append(
            {
                # -- which system produced this (see `streams.py`) --------------
                # Carried per record rather than per page section: the corpora are
                # separate on disk precisely because they must never be pooled, and
                # a reader scrolling one list needs to know at a glance which agent
                # and model a card belongs to.
                "stream": stream.stream_id,
                "agent_id": record.agent_id,
                "runs_dir": stream.runs_dirname,
                "run_id": record.run_id,
                "cutoff": str(record.cutoff),
                "issued_at": record.issued_at.isoformat(timespec="seconds"),
                "issued_time": record.issued_at.strftime("%H:%M"),
                "provenance": record.provenance,
                "fitting": record.provenance in settings.fitting_eligible_provenance,
                "settings_source": record.settings_source,
                "calibration_version": record.calibration_version,
                "agent_model": record.agent_model,
                "prompt_version": record.prompt_version,
                "fingerprint": _short_fingerprint(record.package_fingerprint),
                "faithful": report.faithful,
                "fidelity_error": report.max_absolute_error,
                # -- the judgment ---------------------------------------------
                "physical_status": assessment.get("physical_status", "--"),
                "novelty": assessment.get("incremental_novelty", "--"),
                "confidence": assessment.get("confidence"),
                "conflict": bool(assessment.get("material_evidence_conflict", False)),
                "claim_count": len(assessment.get("evidence_claims", [])),
                "claim_types": sorted(
                    {claim.get("claim_type", "?") for claim in assessment.get("evidence_claims", [])}
                ),
                # The statements are the substance of the run -- what the agent
                # actually believed and cited. Summarising them to a count would
                # hide the only part a reader can independently judge.
                "claims": [
                    {
                        "id": claim.get("claim_id", ""),
                        "type": claim.get("claim_type", "?"),
                        "statement": claim.get("statement", ""),
                        "material": bool(claim.get("material_to_forecast", False)),
                        "sources": len(claim.get("supporting_source_ids", []) or []),
                    }
                    for claim in assessment.get("evidence_claims", [])
                ],
                "rationale": assessment.get("overall_rationale", ""),
                "research_summary": assessment.get("research_summary", ""),
                # -- research --------------------------------------------------
                "queries": len(packet.get("queries", []) or []),
                "sources": len(packet.get("sources", []) or []),
                "resolved_domains": sum(1 for source in (packet.get("sources") or []) if source.get("resolved_domain")),
                "accepted_summaries": sum(
                    1 for summary in (packet.get("verified_summaries") or []) if summary.get("status") == "accepted"
                ),
                # -- audit ------------------------------------------------------
                "model_disagreement": audit.get("model_disagreement_std", {}) or {},
                "claim_support_findings": len(audit.get("claim_support_findings", []) or []),
                "successful_models": audit.get("successful_models", []) or [],
                "failed_models": audit.get("failed_models", []) or [],
                "latest_observation": record.diagnostics.get("latest_observation_date"),
                "latest_value": record.diagnostics.get("latest_value"),
                "vix": (record.diagnostics.get("covariate_latest_values") or {}).get("vix_level_l1b"),
                "realized_vol": record.diagnostics.get("realized_volatility_21b"),
                # -- rollups ----------------------------------------------------
                "max_overlay": max(overlays) if overlays else 0.0,
                "moved": any(value > 1e-9 for value in overlays),
                "best_tier": max(tiers, key=lambda name: _TIER_ORDER.get(name, 0)) if tiers else "none",
                "resolved_count": sum(1 for item in horizons if item["status"] == "resolved"),
                "horizons": horizons,
            }
        )

    return {
        "data_through": str(resolution.data_through) if resolution.data_through else None,
        "calibration_in_force": replay.ledger.current(date.today()).version,
        "records": rows,
    }


def collect_adaptive() -> dict[str, Any]:
    """Assemble the adaptive agent's own corpus, in the row shape the page reads.

    Deliberately produces far fewer fields than `collect_stream`: no evidence tier,
    no claims, no fingerprint, no calibration version worth showing -- because none
    of those exist for this agent. `best_tier` is set to the sentinel ``"n/a"``
    (not ``"none"``, which means something specific in the coach's own vocabulary --
    a gate that ran and found no evidence) and `claim_types`/`claims` are empty
    lists so the page's existing search and tier-sort logic, written for records
    that always have these keys, doesn't have to special-case an absent one.

    Never merged into `payload["streams"]` -- that list drives the three-box
    comparability strip, and this agent is not a fourth option in it.
    """
    records = _load_adaptive_records()
    resolver = OutcomeResolver(DEFAULT_SETTINGS)
    scorer = ForecastScorer()
    resolution = resolver.resolve_all(records)
    resolved_by = {(item.run_id, item.horizon): item for item in resolution.resolved}
    pending_by = {(item.run_id, item.horizon): item for item in resolution.pending}

    rows: list[dict[str, Any]] = []
    for record in records:
        horizons: list[dict[str, Any]] = []
        for item in record.forecasts:
            key = (record.run_id, item.horizon)
            resolved = resolved_by.get(key)
            pending = pending_by.get(key)
            entry: dict[str, Any] = {
                "horizon": item.horizon,
                "forecast_date": str(item.forecast_date.date()),
                "p10": item.final_quantiles.get(0.1),
                "p50": item.final_point_forecast,
                "p90": item.final_quantiles.get(0.9),
                "width": item.final_quantiles.get(0.9, 0.0) - item.final_quantiles.get(0.1, 0.0),
                "rationale": item.forecast_transformation.get("horizon_rationale", ""),
            }
            if resolved is not None:
                card = scorer.score_recorded(record, resolved, variant="agent")
                entry |= {
                    "status": "resolved",
                    "realized": resolved.realized_value,
                    "observation_date": str(resolved.realized_observation_date),
                    "pinball": card.pinball,
                    "abs_error": card.absolute_error,
                    "covered": card.covered_80,
                }
            else:
                entry |= {
                    "status": "pending",
                    "pending_reason": pending.reason if pending else "not resolved",
                }
            horizons.append(entry)

        rows.append(
            {
                "kind": "adaptive",
                "stream": ADAPTIVE_AGENT_ID,
                "agent_id": record.agent_id,
                "runs_dir": "adaptive_agent/runs",
                "run_id": record.run_id,
                "cutoff": str(record.cutoff),
                "issued_at": record.issued_at.isoformat(timespec="seconds"),
                "issued_time": record.issued_at.strftime("%H:%M"),
                "provenance": record.provenance,
                "fitting": False,
                "agent_model": record.agent_model,
                "rationale": record.assessment.get("rationale", ""),
                "claim_types": [],
                "claims": [],
                "best_tier": "n/a",
                "resolved_count": sum(1 for item in horizons if item["status"] == "resolved"),
                "horizons": horizons,
            }
        )
    return {"records": rows}


def collect(streams: tuple[RunStream, ...] = STREAMS, *, include_adaptive: bool = True) -> dict[str, Any]:
    """Assemble every stream's corpus into one payload, each record tagged with its stream.

    One page rather than three because the question a reader actually has is
    comparative -- what did v5.0 say, and what did each v5.2 arm say, on the same
    cutoff. Three pages make that a tab-switching exercise.

    The tagging is what keeps that safe. These corpora may never be *pooled* for
    fitting (``ComparisonPolicy`` enforces one prompt version, one package
    fingerprint and one model), so every record carries its stream, agent and model,
    the summary tiles are computed per stream rather than over the union, and the
    Stream filter defaults to showing everything but is one click from isolating one.

    ``include_adaptive`` folds in the adaptive agent's own corpus (`collect_adaptive`)
    alongside the three streams -- on the same page, tagged distinctly, but never as
    a fourth entry in ``streams`` (see that function's docstring for why).
    """
    per_stream = {stream.stream_id: collect_stream(stream) for stream in streams}
    records = [row for part in per_stream.values() for row in part["records"]]
    data_through = [p["data_through"] for p in per_stream.values() if p["data_through"]]

    adaptive_rows = collect_adaptive()["records"] if include_adaptive else []
    records += adaptive_rows
    adaptive_models = sorted({row["agent_model"] for row in adaptive_rows})

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_through": max(data_through) if data_through else None,
        "adaptive": (
            {
                "id": ADAPTIVE_AGENT_ID,
                "agent_id": ADAPTIVE_AGENT_ID,
                "model": " / ".join(adaptive_models) if adaptive_models else "—",
                "runs_dir": "adaptive_agent/runs",
                "count": len(adaptive_rows),
            }
            if adaptive_rows
            else None
        ),
        "streams": [
            {
                "id": stream.stream_id,
                "agent_id": stream.target.agent_id,
                "model": stream.model,
                "runs_dir": stream.runs_dirname,
                "calibration_dir": stream.calibration_dirname,
                "calibration_in_force": per_stream[stream.stream_id]["calibration_in_force"],
                "code_execution": stream.code_execution_enabled,
                "runs_per_day": stream.runs_per_day,
                "description": stream.description,
                "count": len(per_stream[stream.stream_id]["records"]),
            }
            for stream in streams
        ],
        "records": records,
    }


_TIER_ORDER = {"none": 0, "limited": 1, "corroborated": 2, "strong": 3}


def embed_raw(
    streams: tuple[RunStream, ...] = STREAMS,
    budget: int = SIZE_BUDGET_BYTES,
    *,
    include_adaptive: bool = True,
) -> tuple[dict[str, Any], str | None]:
    """Full JSON per record across every stream, trimmed only if the whole thing will not fit.

    Keyed by ``run_id``, which is unique across streams: the two v5.2 streams share an
    agent id and are kept apart by `RunStream.run_id_prefix`. The adaptive agent's own
    files already use the same ``<run_id>.json`` naming, so they merge in the same way.
    """
    raw: dict[str, Any] = {}
    for stream in streams:
        for path in sorted(RunRecordStore(stream.coach_settings).directory.glob("*.json")):
            raw[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    if include_adaptive:
        for path in sorted(ADAPTIVE_RUNS_DIR.glob("adaptive_agent__*.json")):
            raw[path.stem] = json.loads(path.read_text(encoding="utf-8"))

    size = len(json.dumps(raw, separators=(",", ":")))
    if size <= budget:
        return raw, None

    for payload in raw.values():
        signals = payload.get("audit_signals") or {}
        if HEAVY_AUDIT_KEY in signals:
            signals[HEAVY_AUDIT_KEY] = {
                "_omitted": (
                    "Dropped from this page to stay inside the artifact size limit. "
                    "The complete block is in the record on disk."
                )
            }
    trimmed = len(json.dumps(raw, separators=(",", ":")))
    note = (
        f"Embedded JSON was {size / 1024 / 1024:.1f} MB, over the {budget / 1024 / 1024:.0f} MB budget. "
        f"The {HEAVY_AUDIT_KEY} block (page-fetch archive) is omitted from this page, bringing it to "
        f"{trimmed / 1024 / 1024:.1f} MB. Every other field is complete, and the full block remains in "
        f"the record on disk."
    )
    return raw, note


def render(payload: dict[str, Any], raw: dict[str, Any], trim_note: str | None) -> str:
    return (
        _TEMPLATE.replace("/*__DATA__*/", json.dumps(payload, separators=(",", ":")))
        .replace("/*__RAW__*/", json.dumps(raw, separators=(",", ":")))
        .replace("/*__NOTE__*/", json.dumps(trim_note))
    )


def main() -> None:
    # Defaults to every stream, plus the adaptive agent. `--stream=` narrows the
    # cfm_coach streams; `--no-adaptive` drops the adaptive agent's corpus -- both
    # are escape hatches for once the combined page approaches the size ceiling.
    selected = [stream_for(a.split("=", 1)[1].strip()) for a in sys.argv[1:] if a.startswith("--stream=")]
    streams = tuple(selected) or STREAMS
    include_adaptive = "--no-adaptive" not in sys.argv[1:]

    payload = collect(streams, include_adaptive=include_adaptive)
    raw, note = embed_raw(streams, include_adaptive=include_adaptive)
    OUTPUT.write_text(render(payload, raw, note), encoding="utf-8")
    size = OUTPUT.stat().st_size
    print(f"{OUTPUT}  ({size / 1024 / 1024:.2f} MB, {len(payload['records'])} records)")
    for entry in payload["streams"]:
        print(f"  {entry['id']:<14} {entry['count']:>3} record(s)  {entry['agent_id']}  {entry['model']}")
    if payload.get("adaptive"):
        a = payload["adaptive"]
        print(f"  {a['id']:<14} {a['count']:>3} record(s)  {a['agent_id']}  {a['model']}  (not a stream)")
    if note:
        print(f"  note: {note}")
    if size > 14 * 1024 * 1024:
        print("  WARNING: approaching the 16 MB artifact ceiling.")
        print("  Narrow with --stream=<id> or drop --no-adaptive to publish less at a time.")


_TEMPLATE = r"""<meta charset="utf-8">
<title>CFM Coach Corpus</title>
<style>
  /* ---- tokens: light is the base, both dark paths redefine only these ---- */
  :root {
    --ground:      #F5F6F4;
    --panel:       #FFFFFF;
    --panel-2:     #F0F2EF;
    --ink:         #14181D;
    --ink-2:       #3D444C;
    --muted:       #656B73;
    --rule:        #DFE1DC;
    --rule-2:      #C8CBC4;
    /* teal = the deterministic machinery */
    --accent:      #0E7C86;
    --accent-soft: #D6EBED;
    --on-accent:   #FFFFFF;
    /* amber = the LLM's contribution, used nowhere else */
    --signal:      #C2610F;
    --signal-soft: #F7E4CF;
    --ok:          #2F7D4F;
    --warn:        #9A7000;
    --crit:        #B3261E;
    --crit-soft:   #F7DEDC;
    --shadow:      0 1px 2px rgba(20,24,29,.06), 0 4px 16px rgba(20,24,29,.05);

    /* Chart-series identities for the Performance tab. Same hue families as the
       page's stream colors but re-stepped to pass the categorical-palette checks
       (lightness band, chroma floor, CVD + normal-vision separation, contrast)
       on this mode's chart surface -- the page tokens themselves fail them, and
       adaptive's gray cannot do identity work at all, so it gets a real hue.
       Identity never rides on color alone: each agent also has a fixed marker
       shape (circle / square / triangle / diamond). */
    --ch-v50:      #0390A6;
    --ch-v52a:     #C96A15;
    --ch-v52l:     #1E7245;
    --ch-adapt:    #7A4FBF;

    --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
    --sans: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --ground:      #101317;
      --panel:       #171B21;
      --panel-2:     #1E232A;
      --ink:         #E8EAED;
      --ink-2:       #B6BCC4;
      --muted:       #8C949E;
      --rule:        #2A3038;
      --rule-2:      #3A424C;
      --accent:      #35B8C4;
      --accent-soft: #14343A;
      --on-accent:   #06181B;
      --signal:      #E8913A;
      --signal-soft: #3A2A16;
      --ok:          #5FBF87;
      --warn:        #D9B04A;
      --crit:        #F2837B;
      --crit-soft:   #3B1F1D;
      --shadow:      0 1px 2px rgba(0,0,0,.4), 0 4px 16px rgba(0,0,0,.3);
      --ch-v50:      #1D97A5;
      --ch-v52a:     #C97625;
      --ch-v52l:     #3F9A63;
      --ch-adapt:    #9877D6;
    }
  }
  :root[data-theme="dark"] {
    --ground:      #101317;
    --panel:       #171B21;
    --panel-2:     #1E232A;
    --ink:         #E8EAED;
    --ink-2:       #B6BCC4;
    --muted:       #8C949E;
    --rule:        #2A3038;
    --rule-2:      #3A424C;
    --accent:      #35B8C4;
    --accent-soft: #14343A;
    --on-accent:   #06181B;
    --signal:      #E8913A;
    --signal-soft: #3A2A16;
    --ok:          #5FBF87;
    --warn:        #D9B04A;
    --crit:        #F2837B;
    --crit-soft:   #3B1F1D;
    --shadow:      0 1px 2px rgba(0,0,0,.4), 0 4px 16px rgba(0,0,0,.3);
    --ch-v50:      #1D97A5;
    --ch-v52a:     #C97625;
    --ch-v52l:     #3F9A63;
    --ch-adapt:    #9877D6;
  }

  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--ground);
    color: var(--ink);
    font-family: var(--sans);
    font-size: 15px;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
  }
  .wrap { max-width: 1180px; margin: 0 auto; padding: 28px 20px 80px; }

  /* ---- masthead ---- */
  .masthead { display: flex; flex-wrap: wrap; align-items: baseline; gap: 12px 20px; margin-bottom: 6px; }
  h1 {
    font-family: var(--mono); font-size: 22px; font-weight: 600;
    letter-spacing: -0.01em; margin: 0; text-wrap: balance;
  }
  .sub { color: var(--muted); font-size: 13px; max-width: 68ch; margin: 0 0 22px; }
  .stamp { font-family: var(--mono); font-size: 11px; color: var(--muted); letter-spacing: .04em; }

  /* ---- tab bar ---- */
  .tabbar { display: flex; gap: 4px; border-bottom: 1px solid var(--rule); margin-bottom: 18px; }
  .tab {
    font-family: var(--mono); font-size: 12.5px; letter-spacing: .04em; cursor: pointer;
    background: none; border: none; border-bottom: 2px solid transparent;
    color: var(--muted); padding: 8px 14px 9px; margin-bottom: -1px;
  }
  .tab:hover { color: var(--ink-2); }
  .tab[aria-selected="true"] { color: var(--ink); border-bottom-color: var(--accent); font-weight: 600; }
  .tab:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; border-radius: 2px; }

  /* ---- performance tab ---- */
  .perf-rail {
    display: flex; flex-wrap: wrap; gap: 14px 22px; align-items: center;
    background: var(--panel); border: 1px solid var(--rule); border-radius: 5px;
    padding: 12px 16px; margin-bottom: 14px; box-shadow: var(--shadow);
  }
  .perf-group { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  .perf-glabel { font-family: var(--mono); font-size: 10px; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
  .agent-check {
    display: inline-flex; align-items: center; gap: 7px; cursor: pointer; user-select: none;
    font-family: var(--mono); font-size: 12px; color: var(--ink-2);
    padding: 4px 10px 4px 7px; border: 1px solid var(--rule); border-radius: 20px; background: var(--panel);
  }
  .agent-check input { accent-color: var(--accent); margin: 0; }
  .agent-check[data-off="1"] { opacity: .45; }
  .agent-check svg { flex: none; }
  .seg { display: inline-flex; border: 1px solid var(--rule); border-radius: 4px; overflow: hidden; }
  .seg button {
    font-family: var(--mono); font-size: 12px; background: var(--panel); color: var(--ink-2);
    border: none; padding: 5px 13px; cursor: pointer; border-right: 1px solid var(--rule);
  }
  .seg button:last-child { border-right: none; }
  .seg button[aria-pressed="true"] { background: var(--accent); color: var(--on-accent); font-weight: 600; }
  .seg button:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }

  .chart-panel {
    background: var(--panel); border: 1px solid var(--rule); border-radius: 5px;
    padding: 14px 16px 10px; margin-bottom: 14px; box-shadow: var(--shadow);
  }
  .chart-title { font-family: var(--mono); font-size: 12.5px; font-weight: 600; letter-spacing: .03em; }
  .chart-sub { font-size: 12px; color: var(--muted); margin: 3px 0 10px; }
  .chart-scroll { overflow-x: auto; }
  .chart-scroll svg { display: block; min-width: 640px; width: 100%; height: auto; }
  .chart-grid2 .chart-scroll svg { min-width: 340px; }
  .axt { font-family: var(--mono); font-size: 10px; fill: var(--muted); }
  .hbar-row { display: flex; align-items: center; gap: 10px; margin: 7px 0; }
  .hbar-name { font-family: var(--mono); font-size: 11.5px; width: 110px; flex: none; color: var(--ink-2); display: flex; align-items: center; gap: 6px; }
  .hbar-track { flex: 1; height: 14px; background: var(--panel-2); border-radius: 3px; position: relative; overflow: hidden; }
  .hbar-fill { position: absolute; inset: 0 auto 0 0; border-radius: 3px 0 0 3px; }
  .hbar-target { position: absolute; top: -2px; bottom: -2px; width: 0; border-left: 2px dashed var(--rule-2); }
  .hbar-val { font-family: var(--mono); font-size: 11px; color: var(--ink-2); width: 200px; flex: none; }
  .chart-note { font-size: 11.5px; color: var(--muted); margin-top: 8px; }
  .chart-empty { font-size: 12.5px; color: var(--muted); padding: 26px 0 30px; text-align: center; }
  .chart-grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  @media (max-width: 860px) { .chart-grid2 { grid-template-columns: 1fr; } }

  #perf-tip {
    position: fixed; z-index: 30; pointer-events: none; display: none;
    background: var(--panel); border: 1px solid var(--rule-2); border-radius: 4px;
    box-shadow: var(--shadow); padding: 8px 10px;
    font-family: var(--mono); font-size: 11px; line-height: 1.6; color: var(--ink-2);
    max-width: 300px;
  }
  #perf-tip b { color: var(--ink); }

  /* ---- stream band ----
     One colour per stream, reused on every card badge, so a reader learns the
     mapping once at the top and can then scan the list without re-reading labels. */
  .streams { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 10px; margin-bottom: 14px; }
  .sbox { background: var(--panel); border: 1px solid var(--rule); border-left-width: 3px; border-radius: 4px; padding: 11px 13px; }
  .sname { font-family: var(--mono); font-size: 12px; font-weight: 600; letter-spacing: .04em; display: flex; justify-content: space-between; gap: 8px; }
  .scount { color: var(--muted); font-weight: 400; }
  .smeta { font-family: var(--mono); font-size: 11px; color: var(--ink-2); margin-top: 3px; }
  .sdir { font-family: var(--mono); font-size: 10px; color: var(--muted); margin-top: 5px; line-height: 1.5; }
  .sbox.s-v50_lite { border-left-color: var(--accent); }
  .sbox.s-v52_advanced { border-left-color: var(--signal); }
  .sbox.s-v52_lite { border-left-color: var(--ok); }
  /* Dashed, muted, and no fill -- the visual signal that this box is not a fourth
     option alongside the three above, but a different kind of thing shown for
     reference. Reused on its card badge below for the same reason. */
  .sbox.adaptive { border-left-style: dashed; border-left-color: var(--muted); }
  .chip.stream { font-weight: 600; }
  .chip.s-v50_lite { border-color: var(--accent); color: var(--accent); background: var(--accent-soft); }
  .chip.s-v52_advanced { border-color: var(--signal); color: var(--signal); background: var(--signal-soft); }
  .chip.s-v52_lite { border-color: var(--ok); color: var(--ok); }
  .chip.s-adaptive_agent { border-color: var(--muted); color: var(--muted); background: var(--panel-2); border-style: dashed; }

  /* ---- adaptive card: no tier/claims/fingerprint machinery, a rationale instead ---- */
  .card.adaptive .hrationale { font-size: 12px; color: var(--ink-2); margin-top: 6px; line-height: 1.5; }

  /* ---- summary tiles ---- */
  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(148px, 1fr)); gap: 10px; margin-bottom: 22px; }
  .tile { background: var(--panel); border: 1px solid var(--rule); border-radius: 4px; padding: 12px 14px; }
  .tile .k { font-family: var(--mono); font-size: 10px; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
  .tile .v { font-family: var(--mono); font-size: 24px; font-variant-numeric: tabular-nums; margin-top: 3px; }
  .tile .n { font-size: 11px; color: var(--muted); margin-top: 1px; }
  .tile.good .v { color: var(--ok); }
  .tile.warn .v { color: var(--warn); }

  .banner {
    border-left: 3px solid var(--warn); background: var(--panel);
    padding: 10px 14px; border-radius: 0 4px 4px 0; font-size: 13px;
    color: var(--ink-2); margin-bottom: 18px;
  }

  /* ---- filter rail ---- */
  .rail {
    position: sticky; top: 0; z-index: 20; background: var(--ground);
    padding: 12px 0 10px; border-bottom: 1px solid var(--rule); margin-bottom: 16px;
  }
  .rail-row { display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: center; }
  .fgroup { display: flex; align-items: center; gap: 6px; }
  .flabel {
    font-family: var(--mono); font-size: 10px; letter-spacing: .08em;
    text-transform: uppercase; color: var(--muted);
  }
  select, input[type="search"] {
    font-family: var(--mono); font-size: 12px; color: var(--ink);
    background: var(--panel); border: 1px solid var(--rule-2);
    border-radius: 3px; padding: 5px 8px;
  }
  input[type="search"] { min-width: 220px; }
  select:focus-visible, input:focus-visible, button:focus-visible, .card-head:focus-visible {
    outline: 2px solid var(--accent); outline-offset: 2px;
  }
  .toggle {
    font-family: var(--mono); font-size: 11px; letter-spacing: .03em;
    background: var(--panel); color: var(--ink-2); border: 1px solid var(--rule-2);
    border-radius: 3px; padding: 5px 9px; cursor: pointer;
  }
  .toggle[aria-pressed="true"] { background: var(--accent); border-color: var(--accent); color: var(--on-accent); }
  .count { font-family: var(--mono); font-size: 11px; color: var(--muted); margin-left: auto; }

  /* ---- record card ---- */
  .card {
    background: var(--panel); border: 1px solid var(--rule);
    border-radius: 5px; margin-bottom: 12px; box-shadow: var(--shadow);
    overflow: hidden;
  }
  .card.replayed { background: repeating-linear-gradient(135deg, var(--panel), var(--panel) 9px, var(--panel-2) 9px, var(--panel-2) 18px); }
  .card-head {
    display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: center;
    padding: 12px 16px; border-bottom: 1px solid var(--rule);
    cursor: pointer; width: 100%; text-align: left; background: none;
    border-left: 0; border-right: 0; border-top: 0; color: inherit; font: inherit;
  }
  .card-head:hover { background: var(--panel-2); }
  .cut { font-family: var(--mono); font-size: 16px; font-weight: 600; letter-spacing: -.01em; }
  .at { font-family: var(--mono); font-size: 11px; color: var(--muted); }

  .chip {
    font-family: var(--mono); font-size: 10px; letter-spacing: .05em; text-transform: uppercase;
    padding: 2px 7px; border-radius: 2px; border: 1px solid var(--rule-2); color: var(--ink-2);
    white-space: nowrap;
  }
  .chip.live { background: var(--accent-soft); border-color: var(--accent); color: var(--accent); }
  .chip.replay { background: transparent; border-style: dashed; color: var(--muted); }
  .chip.moved { background: var(--signal-soft); border-color: var(--signal); color: var(--signal); }
  .chip.bad { background: var(--crit-soft); border-color: var(--crit); color: var(--crit); }
  .chip.ok { border-color: var(--ok); color: var(--ok); }

  .card-body { padding: 4px 16px 14px; }

  /* ---- interval strips: the motif ---- */
  .strip { padding: 11px 0 9px; border-bottom: 1px dashed var(--rule); }
  .strip:last-of-type { border-bottom: 0; }
  .strip-top {
    display: flex; flex-wrap: wrap; gap: 6px 12px; align-items: baseline;
    font-family: var(--mono); font-size: 11px; color: var(--muted); margin-bottom: 7px;
  }
  .h-label { font-size: 12px; font-weight: 600; color: var(--ink); min-width: 42px; }
  .num { font-variant-numeric: tabular-nums; color: var(--ink-2); }
  .num b { color: var(--ink); font-weight: 600; }
  .ov { color: var(--signal); font-variant-numeric: tabular-nums; }
  .ov.zero { color: var(--muted); }

  .track { position: relative; height: 26px; }
  .axis { position: absolute; left: 0; right: 0; top: 13px; height: 1px; background: var(--rule); }
  .band {
    position: absolute; top: 7px; height: 12px; border-radius: 2px;
    background: var(--accent-soft); border: 1px solid var(--accent);
  }
  .replayed .band { opacity: .55; }
  .tick { position: absolute; top: 3px; width: 2px; height: 20px; background: var(--ink); border-radius: 1px; }
  .tick.ens { background: transparent; border-left: 2px dotted var(--muted); }
  .realized { position: absolute; top: 1px; width: 2px; height: 24px; background: var(--crit); }
  .realized.in { background: var(--ok); }
  .realized::after {
    content: ""; position: absolute; left: -3px; top: -4px; width: 8px; height: 8px;
    border-radius: 50%; background: inherit;
  }
  .scale { display: flex; justify-content: space-between; font-family: var(--mono); font-size: 10px; color: var(--muted); margin-top: 2px; }

  .verdict { font-family: var(--mono); font-size: 11px; margin-top: 6px; color: var(--muted); }
  .verdict .in { color: var(--ok); }
  .verdict .out { color: var(--crit); }

  /* ---- judgment row ---- */
  .judge {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(126px, 1fr));
    gap: 8px 16px; padding: 12px 0 2px; border-top: 1px solid var(--rule); margin-top: 8px;
  }
  .j .k { font-family: var(--mono); font-size: 10px; letter-spacing: .07em; text-transform: uppercase; color: var(--muted); }
  .j .v { font-family: var(--mono); font-size: 12px; color: var(--ink); word-break: break-word; }
  .claims { margin-top: 12px; border-top: 1px solid var(--rule); padding-top: 10px; }
  .claims .k { font-family: var(--mono); font-size: 10px; letter-spacing: .07em; text-transform: uppercase; color: var(--muted); margin-bottom: 6px; }
  .claim { display: grid; grid-template-columns: 132px 1fr auto; gap: 10px; align-items: baseline; padding: 3px 0; font-size: 13px; }
  .ctype { font-family: var(--mono); font-size: 10.5px; color: var(--accent); letter-spacing: .03em; }
  .ctext { color: var(--ink-2); max-width: 78ch; }
  .cmeta { font-family: var(--mono); font-size: 10px; color: var(--muted); white-space: nowrap; }
  @media (max-width: 700px) { .claim { grid-template-columns: 1fr; gap: 2px; } }
  .more {
    margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap;
  }
  .btn {
    font-family: var(--mono); font-size: 11px; letter-spacing: .04em;
    background: var(--panel-2); color: var(--ink); border: 1px solid var(--rule-2);
    border-radius: 3px; padding: 6px 11px; cursor: pointer;
  }
  .btn:hover { border-color: var(--accent); color: var(--accent); }

  /* ---- json drawer ---- */
  dialog {
    border: 1px solid var(--rule-2); border-radius: 6px; background: var(--panel);
    color: var(--ink); padding: 0; width: min(1000px, 94vw); height: min(84vh, 900px);
    box-shadow: var(--shadow);
  }
  dialog::backdrop { background: rgba(8,10,13,.55); }
  .dlg-head {
    display: flex; align-items: center; gap: 12px; padding: 12px 16px;
    border-bottom: 1px solid var(--rule); position: sticky; top: 0; background: var(--panel);
  }
  .dlg-title { font-family: var(--mono); font-size: 13px; font-weight: 600; word-break: break-all; }
  .dlg-actions { margin-left: auto; display: flex; gap: 8px; }
  pre.json {
    margin: 0; padding: 14px 16px; overflow: auto; height: calc(100% - 52px);
    font-family: var(--mono); font-size: 11.5px; line-height: 1.55; color: var(--ink-2);
    white-space: pre; tab-size: 2;
  }
  .jk { color: var(--accent); }
  .js { color: var(--ink); }
  .jn { color: var(--signal); font-variant-numeric: tabular-nums; }
  .jb { color: var(--warn); }

  .empty { text-align: center; padding: 48px 20px; color: var(--muted); font-family: var(--mono); font-size: 13px; }
  .legend {
    display: flex; flex-wrap: wrap; gap: 6px 18px; font-family: var(--mono); font-size: 10.5px;
    color: var(--muted); margin: 14px 0 20px; letter-spacing: .02em;
  }
  .legend i { font-style: normal; color: var(--ink-2); }
  .swatch { display: inline-block; width: 20px; height: 8px; vertical-align: middle; margin-right: 5px; border-radius: 2px; }
  footer { margin-top: 34px; padding-top: 14px; border-top: 1px solid var(--rule); font-size: 12px; color: var(--muted); }
  footer code { font-family: var(--mono); font-size: 11px; }
  @media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
</style>

<div class="wrap">
  <div class="masthead">
    <h1>CFM Coach Corpus</h1>
    <span class="stamp" id="stamp"></span>
  </div>
  <p class="sub">
    Every forecast the coach has recorded, across all three run streams, with what the model
    proposed, what the evidence policy allowed, and how the published interval compares to the raw
    numerical ensemble. Only <b>live_forward</b> records are fitting evidence &mdash; replayed cutoffs are
    shown hatched and excluded from any calibration.
  </p>
  <p class="sub">
    <b>The streams are separate corpora and are never pooled.</b> Each has its own calibration ledger,
    and a fit may span only one agent build, one prompt version and one model. They share a page so the
    same cutoff can be read across all three &mdash; every card is tagged with the stream that produced it,
    and the <b>Stream</b> filter isolates one.
  </p>
  <p class="sub">
    The dashed <b>adaptive_agent</b> box is a fourth, unrelated forecaster shown alongside for reference
    &mdash; a single persistent analyst with no ensemble, no evidence-tier gate, and no coach calibration
    layer, so it isn't part of the comparison above and never enters any cfm_coach fit.
  </p>

  <div id="banner"></div>

  <div class="tabbar" role="tablist">
    <button class="tab" id="tabbtn-records" role="tab" aria-selected="true" aria-controls="tab-records">Records</button>
    <button class="tab" id="tabbtn-perf" role="tab" aria-selected="false" aria-controls="tab-perf">Performance</button>
  </div>

  <div id="tab-records" role="tabpanel">
  <div class="streams" id="streams"></div>
  <div class="tiles" id="tiles"></div>

  <div class="legend">
    <span><i class="swatch" style="background:var(--accent-soft);border:1px solid var(--accent)"></i>P10&ndash;P90 interval</span>
    <span><i class="swatch" style="background:var(--ink);width:2px;height:14px"></i>published P50</span>
    <span><i class="swatch" style="border-left:2px dotted var(--muted);width:2px;height:14px"></i>ensemble P50</span>
    <span><i class="swatch" style="background:var(--signal)"></i>overlay &mdash; the LLM's contribution</span>
    <span><i class="swatch" style="background:var(--ok)"></i>realized, inside interval</span>
    <span><i class="swatch" style="background:var(--crit)"></i>realized, outside</span>
  </div>

  <div class="rail">
    <div class="rail-row">
      <div class="fgroup"><span class="flabel">Stream</span>
        <select id="f-stream"><option value="">all</option></select></div>
      <div class="fgroup"><span class="flabel">Cutoff</span>
        <select id="f-cutoff"><option value="">all</option></select></div>
      <div class="fgroup"><span class="flabel">Provenance</span>
        <select id="f-prov"><option value="">all</option></select></div>
      <div class="fgroup"><span class="flabel">Tier</span>
        <select id="f-tier"><option value="">all</option></select></div>
      <div class="fgroup"><span class="flabel">Status</span>
        <select id="f-status">
          <option value="">all</option>
          <option value="resolved">has resolved</option>
          <option value="pending">all pending</option>
        </select></div>
      <div class="fgroup"><span class="flabel">Sort</span>
        <select id="f-sort">
          <option value="cutoff-desc">newest cutoff</option>
          <option value="cutoff-asc">oldest cutoff</option>
          <option value="issued-desc">newest run</option>
          <option value="overlay-desc">largest overlay</option>
        </select></div>
    </div>
    <div class="rail-row" style="margin-top:8px">
      <input type="search" id="f-text" placeholder="search id, rationale, status&hellip;" aria-label="Search records">
      <button class="toggle" id="t-fitting" aria-pressed="false">fitting evidence only</button>
      <button class="toggle" id="t-moved" aria-pressed="false">overlay &ne; 0</button>
      <button class="toggle" id="t-drift" aria-pressed="false">fidelity drift</button>
      <button class="toggle" id="t-reset">reset</button>
      <span class="count" id="count"></span>
    </div>
  </div>

  <div id="list"></div>
  </div><!-- /tab-records -->

  <div id="tab-perf" role="tabpanel" hidden></div>
  <div id="perf-tip"></div>

  <footer>
    Regenerate with <code>uv run python -m energy_oil_forecasting.cfm_coach.make_corpus_page</code>.
    Derived columns come from <code>replay.py</code>, <code>outcomes.py</code> and <code>scoring.py</code>,
    so this page and <code>report.py</code> cannot disagree.
  </footer>
</div>

<dialog id="dlg">
  <div class="dlg-head">
    <span class="dlg-title" id="dlg-title"></span>
    <span class="dlg-actions">
      <button class="btn" id="dlg-copy">copy json</button>
      <button class="btn" id="dlg-close">close</button>
    </span>
  </div>
  <pre class="json" id="dlg-json"></pre>
</dialog>

<script>
const DATA = /*__DATA__*/;
const RAW  = /*__RAW__*/;
const NOTE = /*__NOTE__*/;

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const money = (v) => (v == null ? "\u2014" : v.toFixed(2));
const signed = (v) => (v > 0 ? "+" : "") + v.toFixed(2);

/* ---------- summary ---------- */
const recs = DATA.records;
$("stamp").textContent = `generated ${DATA.generated_at.replace("T", " ")} \u00b7 prices through ${DATA.data_through || "\u2014"}`;

/* ---------- streams ----------
   Rendered as its own band above the tiles because it is the first thing a reader
   needs: three agents answered the same question here, and a card means nothing
   until you know which one produced it. Calibration is per stream, so it belongs
   here rather than in the page stamp. */
const streamBoxes = DATA.streams.map(s => `
  <div class="sbox s-${esc(s.id)}">
    <div class="sname">${esc(s.id)}<span class="scount">${s.count} record${s.count === 1 ? "" : "s"}</span></div>
    <div class="smeta">${esc(s.agent_id)}</div>
    <div class="smeta">${esc(s.model)}</div>
    <div class="sdir">${esc(s.runs_dir)}/ \u00b7 calibration ${esc(s.calibration_in_force)} \u00b7 ${s.runs_per_day}/day \u00b7 code exec ${s.code_execution ? "on" : "off"}</div>
  </div>`).join("");
// Dashed border, no fill -- deliberately reads as "not a fourth stream" (see the
// `.sbox.adaptive` rule). Absent entirely when the page was built with --no-adaptive.
const adaptiveBox = DATA.adaptive ? `
  <div class="sbox adaptive">
    <div class="sname">${esc(DATA.adaptive.id)}<span class="scount">${DATA.adaptive.count} record${DATA.adaptive.count === 1 ? "" : "s"}</span></div>
    <div class="smeta">${esc(DATA.adaptive.agent_id)}</div>
    <div class="smeta">${esc(DATA.adaptive.model)}</div>
    <div class="sdir">${esc(DATA.adaptive.runs_dir)}/ \u00b7 different architecture &mdash; no ensemble, no evidence-tier gate, no coach calibration layer. Shown for reference, not comparable to the three streams above.</div>
  </div>` : "";
$("streams").innerHTML = streamBoxes + adaptiveBox;

if (NOTE) $("banner").innerHTML = `<div class="banner">${esc(NOTE)}</div>`;

const fitting = recs.filter(r => r.fitting);
// Explicit `=== false`, not `!r.faithful`: adaptive rows carry no `faithful` key at
// all (fidelity/replay is not a concept that applies to them), and `!undefined` is
// true -- the looser check would have silently counted every one of them as drifted.
// Fidelity is only checked for rows that carry the key at all -- adaptive rows
// never do, since replay is not a concept that applies to them. The tile's "all N
// reproduce" denominator must be this checked set, not `recs.length`, or it
// overclaims fidelity for rows that were never tested.
const fidelityChecked = recs.filter(r => r.faithful !== undefined);
const drifted = fidelityChecked.filter(r => r.faithful === false);
const allH = recs.flatMap(r => r.horizons);
const resolvedH = allH.filter(h => h.status === "resolved");
const covered = resolvedH.filter(h => h.covered).length;
// Same reasoning as `fidelityChecked`: "left the ensemble alone" presupposes there
// was an ensemble to leave alone, which adaptive rows don't have. Scope the
// denominator to rows where `moved` is actually a meaningful concept.
const ensembleApplicable = recs.filter(r => r.moved !== undefined);
const movedRuns = ensembleApplicable.filter(r => r.moved).length;

$("tiles").innerHTML = [
  ["records", recs.length, `${fitting.length} fitting evidence`, ""],
  ["horizons resolved", `${resolvedH.length}/${allH.length}`, `${allH.length - resolvedH.length} pending`, ""],
  ["80% coverage", resolvedH.length ? `${Math.round(covered / resolvedH.length * 100)}%` : "\u2014",
    resolvedH.length ? `${covered} of ${resolvedH.length} inside` : "nothing resolved", covered/Math.max(resolvedH.length,1) >= .8 ? "good" : "warn"],
  ["runs that moved", movedRuns, `${ensembleApplicable.length - movedRuns} left the ensemble alone`, ""],
  ["replay fidelity", drifted.length ? `${drifted.length} drift` : "exact",
    drifted.length ? "invalidates fits" : `all ${fidelityChecked.length} checked reproduce`, drifted.length ? "warn" : "good"],
].map(([k, v, n, cls]) => `<div class="tile ${cls}"><div class="k">${k}</div><div class="v">${v}</div><div class="n">${esc(n)}</div></div>`).join("");

/* ---------- filter options ---------- */
const opts = (sel, values) => {
  values.forEach(v => { const o = document.createElement("option"); o.value = v; o.textContent = v; $(sel).appendChild(o); });
};
// Derived from the records themselves, not DATA.streams -- so a stream option
// appears whenever a record actually carries it, adaptive_agent included, without
// this filter needing its own list of what counts as a "stream."
opts("f-stream", [...new Set(recs.map(r => r.stream))].sort());
opts("f-cutoff", [...new Set(recs.map(r => r.cutoff))].sort().reverse());
opts("f-prov",   [...new Set(recs.map(r => r.provenance))].sort());
const TIER_ORDER = {none:0, limited:1, corroborated:2, strong:3, "n/a":-1};
opts("f-tier",   [...new Set(recs.map(r => r.best_tier))].sort((a,b) => (TIER_ORDER[a] ?? -1) - (TIER_ORDER[b] ?? -1)));

const state = { fitting: false, moved: false, drift: false };

/* ---------- rendering ---------- */
function strip(rec, h) {
  const lo = Math.min(h.p10, h.ensemble_p50, h.status === "resolved" ? h.realized : h.p10);
  const hi = Math.max(h.p90, h.ensemble_p50, h.status === "resolved" ? h.realized : h.p90);
  const pad = (hi - lo) * 0.10 || 1;
  const a = lo - pad, b = hi + pad, span = b - a;
  const pct = (v) => ((v - a) / span * 100).toFixed(2) + "%";

  const marks = [
    `<div class="band" style="left:${pct(h.p10)};width:${((h.p90 - h.p10) / span * 100).toFixed(2)}%"></div>`,
    `<div class="tick ens" style="left:${pct(h.ensemble_p50)}" title="ensemble P50 ${money(h.ensemble_p50)}"></div>`,
    `<div class="tick" style="left:${pct(h.p50)}" title="published P50 ${money(h.p50)}"></div>`,
  ];
  let verdict = `<span class="verdict">pending \u2014 ${esc(h.pending_reason || "")}</span>`;
  if (h.status === "resolved") {
    marks.push(`<div class="realized ${h.covered ? "in" : ""}" style="left:${pct(h.realized)}" title="realized ${money(h.realized)}"></div>`);
    // Equal is the common case, not a tie to be broken: when the policy grants
    // no_change the published forecast IS the ensemble, so the scores are identical
    // by construction and calling that "worse" would misreport it.
    const gap = h.ensemble_pinball - h.pinball;
    const verdictWord = Math.abs(gap) < 5e-4 ? "identical" : (gap > 0 ? "agent better" : "agent worse");
    verdict = `<span class="verdict">realized <b>${money(h.realized)}</b> \u2014
      <span class="${h.covered ? "in" : "out"}">${h.covered ? "inside" : "outside"} P10\u2013P90</span> \u00b7
      error ${money(h.abs_error)} \u00b7 pinball ${h.pinball.toFixed(3)}
      (ensemble ${h.ensemble_pinball.toFixed(3)}, ${verdictWord})</span>`;
  }

  const capped = h.proposed_center !== h.granted_center || h.proposed_uncertainty !== h.granted_uncertainty;
  return `<div class="strip">
    <div class="strip-top">
      <span class="h-label">h=${h.horizon}</span>
      <span>${h.forecast_date}</span>
      <span class="num">P10 <b>${money(h.p10)}</b> \u00b7 P50 <b>${money(h.p50)}</b> \u00b7 P90 <b>${money(h.p90)}</b> \u00b7 width ${money(h.width)}</span>
      <span class="ov ${Math.abs(h.overlay) < 1e-9 ? "zero" : ""}">overlay ${signed(h.overlay)}</span>
      <span class="chip ${h.tier === "none" ? "" : "ok"}">tier ${h.tier}${h.publishers ? " \u00b7 " + h.publishers + " pub" : ""}</span>
      <span class="num">proposed ${esc(h.proposed_center)}/${esc(h.proposed_uncertainty)}${capped ? ` \u2192 granted ${esc(h.granted_center)}/${esc(h.granted_uncertainty)}` : ""}</span>
      ${capped ? '<span class="chip bad">capped by policy</span>' : ""}
    </div>
    <div class="track">${marks.join("")}<div class="axis"></div></div>
    <div class="scale"><span>${money(a)}</span><span>${money(b)}</span></div>
    ${verdict}
  </div>`;
}

// The adaptive agent has no ensemble baseline, no evidence-tier gate, and no
// coach calibration layer -- so its strip has no "ens" tick and no overlay chip
// (there is nothing honest to put in either), and its card has no judge/claims
// section (nothing structured was produced to put there). A rationale paragraph
// stands in for both.
function stripAdaptive(h) {
  const lo = Math.min(h.p10, h.status === "resolved" ? h.realized : h.p10);
  const hi = Math.max(h.p90, h.status === "resolved" ? h.realized : h.p90);
  const pad = (hi - lo) * 0.10 || 1;
  const a = lo - pad, b = hi + pad, span = b - a;
  const pct = (v) => ((v - a) / span * 100).toFixed(2) + "%";

  const marks = [
    `<div class="band" style="left:${pct(h.p10)};width:${((h.p90 - h.p10) / span * 100).toFixed(2)}%"></div>`,
    `<div class="tick" style="left:${pct(h.p50)}" title="P50 ${money(h.p50)}"></div>`,
  ];
  let verdict = `<span class="verdict">pending — ${esc(h.pending_reason || "")}</span>`;
  if (h.status === "resolved") {
    marks.push(`<div class="realized ${h.covered ? "in" : ""}" style="left:${pct(h.realized)}" title="realized ${money(h.realized)}"></div>`);
    verdict = `<span class="verdict">realized <b>${money(h.realized)}</b> —
      <span class="${h.covered ? "in" : "out"}">${h.covered ? "inside" : "outside"} P10–P90</span> ·
      error ${money(h.abs_error)} · pinball ${h.pinball.toFixed(3)}</span>`;
  }

  return `<div class="strip">
    <div class="strip-top">
      <span class="h-label">h=${h.horizon}</span>
      <span>${h.forecast_date}</span>
      <span class="num">P10 <b>${money(h.p10)}</b> · P50 <b>${money(h.p50)}</b> · P90 <b>${money(h.p90)}</b> · width ${money(h.width)}</span>
    </div>
    <div class="track">${marks.join("")}<div class="axis"></div></div>
    <div class="scale"><span>${money(a)}</span><span>${money(b)}</span></div>
    ${verdict}
    ${h.rationale ? `<div class="hrationale">${esc(h.rationale)}</div>` : ""}
  </div>`;
}

function adaptiveCard(rec) {
  return `<article class="card adaptive replayed">
    <button class="card-head" aria-expanded="false" data-run="${esc(rec.run_id)}">
      <span class="chip stream s-${esc(rec.stream)}" title="${esc(rec.agent_id)} · ${esc(rec.agent_model)} · ${esc(rec.runs_dir)}/">${esc(rec.stream)}</span>
      <span class="cut">${rec.cutoff}</span>
      <span class="at">issued ${rec.issued_at.replace("T", " ")}</span>
      <span class="chip live">${rec.provenance}</span>
      <span class="chip">${esc(rec.agent_model)}</span>
    </button>
    <div class="card-body">
      ${rec.horizons.map(h => stripAdaptive(h)).join("")}
      ${rec.rationale ? `<div class="claims">
        <div class="k">market view</div>
        <div class="claim"><span class="ctext">${esc(rec.rationale)}</span></div>
      </div>` : ""}
      <div class="more">
        <button class="btn" data-json="${esc(rec.run_id)}">view full json</button>
      </div>
    </div>
  </article>`;
}

function card(rec) {
  if (rec.kind === "adaptive") return adaptiveCard(rec);
  const j = (k, v) => `<div class="j"><div class="k">${k}</div><div class="v">${esc(v)}</div></div>`;
  return `<article class="card ${rec.fitting ? "" : "replayed"}">
    <button class="card-head" aria-expanded="false" data-run="${esc(rec.run_id)}">
      <span class="chip stream s-${esc(rec.stream)}" title="${esc(rec.agent_id)} · ${esc(rec.agent_model)} · ${esc(rec.runs_dir)}/">${esc(rec.stream)}</span>
      <span class="cut">${rec.cutoff}</span>
      <span class="at">issued ${rec.issued_at.replace("T", " ")}</span>
      <span class="chip ${rec.fitting ? "live" : "replay"}">${rec.provenance}</span>
      <span class="chip">${rec.novelty}</span>
      <span class="chip">conf ${rec.confidence ?? "\u2014"}</span>
      ${rec.moved ? `<span class="chip moved">overlay ${signed(rec.max_overlay)}</span>` : ""}
      ${rec.best_tier !== "none" ? `<span class="chip ok">tier ${rec.best_tier}</span>` : ""}
      ${rec.conflict ? '<span class="chip bad">evidence conflict</span>' : ""}
      ${rec.faithful ? "" : '<span class="chip bad">fidelity drift</span>'}
      ${rec.settings_source === "inferred" ? '<span class="chip">settings inferred</span>' : ""}
    </button>
    <div class="card-body">
      ${rec.horizons.map(h => strip(rec, h)).join("")}
      <div class="judge">
        ${j("physical status", rec.physical_status)}
        ${j("claims", `${rec.claim_count}${rec.claim_types.length ? " \u00b7 " + rec.claim_types.join(", ") : ""}`)}
        ${j("research", `${rec.sources} sources \u00b7 ${rec.resolved_domains} resolved \u00b7 ${rec.accepted_summaries}/${rec.queries} summaries`)}
        ${j("last observation", `${rec.latest_observation ?? "\u2014"} @ ${money(rec.latest_value)}`)}
        ${j("vix / realized vol", `${rec.vix ? rec.vix.toFixed(2) : "\u2014"} / ${rec.realized_vol ? rec.realized_vol.toFixed(3) : "\u2014"}`)}
        ${j("stream", `${rec.stream} \u00b7 ${rec.runs_dir}/`)}
        ${j("agent / model", `${rec.agent_id} \u00b7 ${rec.agent_model}`)}
        ${j("calibration", `${rec.calibration_version} \u00b7 ${rec.prompt_version}`)}
        ${j("package", rec.fingerprint)}
      </div>
      ${rec.claims.length ? `<div class="claims">
        <div class="k">evidence claims</div>
        ${rec.claims.map(c => `<div class="claim">
          <span class="ctype">${esc(c.type)}</span>
          <span class="ctext">${esc(c.statement)}</span>
          <span class="cmeta">${c.sources} src${c.material ? "" : " \u00b7 not material"}</span>
        </div>`).join("")}
      </div>` : ""}
      <div class="more">
        <button class="btn" data-json="${esc(rec.run_id)}">view full json</button>
        <button class="btn" data-why="${esc(rec.run_id)}">why this tier</button>
      </div>
    </div>
  </article>`;
}

function apply() {
  const stream = $("f-stream").value,
        cutoff = $("f-cutoff").value, prov = $("f-prov").value, tier = $("f-tier").value,
        status = $("f-status").value, sort = $("f-sort").value,
        q = $("f-text").value.trim().toLowerCase();

  let out = recs.filter(r => {
    if (stream && r.stream !== stream) return false;
    if (cutoff && r.cutoff !== cutoff) return false;
    if (prov && r.provenance !== prov) return false;
    if (tier && r.best_tier !== tier) return false;
    if (status === "resolved" && r.resolved_count === 0) return false;
    if (status === "pending" && r.resolved_count > 0) return false;
    if (state.fitting && !r.fitting) return false;
    if (state.moved && !r.moved) return false;
    // `!== false`, not the bare flag: adaptive rows carry no `faithful` key (replay
    // fidelity isn't a concept that applies to them), and the toggle must exclude
    // them rather than have `undefined` read as "faithful, so hide it."
    if (state.drift && r.faithful !== false) return false;
    if (q) {
      const hay = [r.run_id, r.rationale, r.research_summary, r.physical_status, r.novelty,
                   r.provenance, r.stream, r.agent_id, r.agent_model, r.claim_types.join(" "),
                   r.claims.map(c => c.type + " " + c.statement).join(" ")].join(" ").toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });

  const by = {
    "cutoff-desc": (a, b) => b.cutoff.localeCompare(a.cutoff) || b.issued_at.localeCompare(a.issued_at),
    "cutoff-asc":  (a, b) => a.cutoff.localeCompare(b.cutoff) || a.issued_at.localeCompare(b.issued_at),
    "issued-desc": (a, b) => b.issued_at.localeCompare(a.issued_at),
    "overlay-desc":(a, b) => b.max_overlay - a.max_overlay,
  }[sort];
  out = out.slice().sort(by);

  $("count").textContent = `${out.length} of ${recs.length} records`;
  $("list").innerHTML = out.length
    ? out.map(card).join("")
    : `<div class="empty">No records match these filters.</div>`;
}

/* ---------- json viewer ---------- */
function highlight(obj) {
  const text = JSON.stringify(obj, null, 2);
  return esc(text)
    .replace(/&quot;([^&]*?)&quot;(\s*:)/g, '<span class="jk">&quot;$1&quot;</span>$2')
    .replace(/:\s(&quot;.*?&quot;)/g, ': <span class="js">$1</span>')
    .replace(/:\s(-?\d+\.?\d*(e[-+]?\d+)?)/gi, ': <span class="jn">$1</span>')
    .replace(/:\s(true|false|null)/g, ': <span class="jb">$1</span>');
}

let currentRaw = null;
document.addEventListener("click", (ev) => {
  const jsonBtn = ev.target.closest("[data-json]");
  const whyBtn  = ev.target.closest("[data-why]");
  const head    = ev.target.closest(".card-head");

  if (jsonBtn) {
    const id = jsonBtn.getAttribute("data-json");
    currentRaw = RAW[id];
    $("dlg-title").textContent = id + ".json";
    $("dlg-json").innerHTML = highlight(currentRaw);
    $("dlg").showModal();
  } else if (whyBtn) {
    const id = whyBtn.getAttribute("data-why");
    const rec = recs.find(r => r.run_id === id);
    const why = { run_id: id, novelty: rec.novelty, confidence: rec.confidence,
      material_evidence_conflict: rec.conflict,
      horizons: rec.horizons.map(h => ({ horizon: h.horizon, proposed: h.proposed_center + "/" + h.proposed_uncertainty,
        granted: h.granted_center + "/" + h.granted_uncertainty, tier: h.tier,
        resolved_publishers: h.publishers, eligible: h.eligible, eligibility_reasons: h.reasons })) };
    $("dlg-title").textContent = "evidence policy \u2014 " + id;
    currentRaw = why;
    $("dlg-json").innerHTML = highlight(why);
    $("dlg").showModal();
  } else if (head) {
    const body = head.nextElementSibling;
    const open = body.style.display !== "none";
    body.style.display = open ? "none" : "";
    head.setAttribute("aria-expanded", String(!open));
  }
});

$("dlg-close").addEventListener("click", () => $("dlg").close());
$("dlg-copy").addEventListener("click", async (e) => {
  try {
    await navigator.clipboard.writeText(JSON.stringify(currentRaw, null, 2));
    e.target.textContent = "copied";
    setTimeout(() => { e.target.textContent = "copy json"; }, 1400);
  } catch { e.target.textContent = "copy blocked"; }
});

["f-stream","f-cutoff","f-prov","f-tier","f-status","f-sort"].forEach(id => $(id).addEventListener("change", apply));
$("f-text").addEventListener("input", apply);
[["t-fitting","fitting"],["t-moved","moved"],["t-drift","drift"]].forEach(([id, key]) => {
  $(id).addEventListener("click", () => {
    state[key] = !state[key];
    $(id).setAttribute("aria-pressed", String(state[key]));
    apply();
  });
});
$("t-reset").addEventListener("click", () => {
  ["f-stream","f-cutoff","f-prov","f-tier","f-status"].forEach(id => $(id).value = "");
  $("f-sort").value = "cutoff-desc"; $("f-text").value = "";
  Object.keys(state).forEach(k => state[k] = false);
  ["t-fitting","t-moved","t-drift"].forEach(id => $(id).setAttribute("aria-pressed", "false"));
  apply();
});

apply();

/* ================= Performance tab =================
   Continuous monitoring of forecasts vs actuals across all four agents.
   Charts are hand-rolled SVG: the artifact CSP forbids external libs, and the
   ~500 flattened horizon rows here need nothing heavier. All series colors come
   from the --ch-* tokens (validated per-mode categorical palette); identity is
   also carried by a fixed marker shape per agent, never color alone. */

const AGENTS = [
  { id: "v50_lite",       label: "v5.0 lite",     color: "var(--ch-v50)",   shape: "circle"   },
  { id: "v52_advanced",   label: "v5.2 advanced", color: "var(--ch-v52a)",  shape: "square"   },
  { id: "v52_lite",       label: "v5.2 lite",     color: "var(--ch-v52l)",  shape: "triangle" },
  { id: "adaptive_agent", label: "adaptive",      color: "var(--ch-adapt)", shape: "diamond"  },
];
const AGENT_BY_ID = Object.fromEntries(AGENTS.map(a => [a.id, a]));
/* Entity-fixed x-dodge so same-day markers from different agents never overprint;
   fixed per agent (not per selection) so a point never moves when a checkbox toggles. */
const DODGE = { v50_lite: -9, v52_advanced: -3, v52_lite: 3, adaptive_agent: 9 };
/* Below this, a centre call (or a realized move) is treated as "no call" rather
   than scored directionally -- a $0.02 nudge is not a directional opinion. */
const FLAT_EPS = 0.05;

/* ---------- flatten to per-(run, horizon) rows ---------- */
const lastCloseByCutoff = {};
recs.forEach(r => { if (r.latest_value != null) lastCloseByCutoff[r.cutoff] = r.latest_value; });

/* Monitoring is live_forward-only, matching the corpus's own evidence rule: a
   replayed cutoff re-searched today's web, so it is not a measurement of how the
   agent does in production -- and the two March replays would stretch the time
   axis across five empty months besides. Stated in the tab subtitle. */
const perfRows = [];
recs.filter(r => r.provenance === "live_forward").forEach(r => r.horizons.forEach(h => perfRows.push({
  stream: r.stream, run_id: r.run_id, cutoff: r.cutoff,
  horizon: h.horizon, target: h.forecast_date,
  p10: h.p10, p50: h.p50, p90: h.p90,
  ens: h.ensemble_p50 !== undefined ? h.ensemble_p50 : null,
  overlay: h.overlay !== undefined ? h.overlay : null,
  /* adaptive records carry no latest_value; borrow the same-cutoff close from a
     coach record (same series, same release rule). Null if no coach run that day. */
  last: r.latest_value != null ? r.latest_value : (lastCloseByCutoff[r.cutoff] ?? null),
  status: h.status, realized: h.status === "resolved" ? h.realized : null,
  covered: h.status === "resolved" ? h.covered : null,
})));

const realizedByTarget = {};
perfRows.forEach(row => { if (row.realized != null) realizedByTarget[row.target] = row.realized; });

/* ---------- state ---------- */
const PERF_LS_KEY = "cfm_perf_v1";
const perfState = { agents: Object.fromEntries(AGENTS.map(a => [a.id, true])), horizon: 5, agg: "med" };
try {
  const saved = JSON.parse(localStorage.getItem(PERF_LS_KEY) || "{}");
  if (saved.agents) AGENTS.forEach(a => { if (typeof saved.agents[a.id] === "boolean") perfState.agents[a.id] = saved.agents[a.id]; });
  if ([5, 10, 21].includes(saved.horizon)) perfState.horizon = saved.horizon;
  if (["med", "all"].includes(saved.agg)) perfState.agg = saved.agg;
} catch (e) { /* private mode etc. -- defaults are fine */ }
function persistPerf() { try { localStorage.setItem(PERF_LS_KEY, JSON.stringify(perfState)); } catch (e) { /* ignore */ } }

/* ---------- small helpers ---------- */
const dayMs = 86400000;
const toMs = (d) => Date.parse(d + "T00:00:00Z");
const fmtDay = (ms) => new Date(ms).toISOString().slice(5, 10);
const median = (xs) => { const s = [...xs].sort((a, b) => a - b); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
const lin = (d0, d1, r0, r1) => (v) => r0 + (v - d0) / (d1 - d0 || 1) * (r1 - r0);
const seldRows = () => perfRows.filter(r => perfState.agents[r.stream] && r.horizon === perfState.horizon);

function marker(shape, x, y, r, attrs) {
  const a = attrs || "";
  if (shape === "circle")   return `<circle cx="${x}" cy="${y}" r="${r}" ${a}/>`;
  if (shape === "square")   return `<rect x="${x - r}" y="${y - r}" width="${2 * r}" height="${2 * r}" ${a}/>`;
  if (shape === "triangle") return `<path d="M${x} ${y - r * 1.2} L${x + r * 1.1} ${y + r * 0.9} L${x - r * 1.1} ${y + r * 0.9} Z" ${a}/>`;
  return `<path d="M${x} ${y - r * 1.3} L${x + r * 1.3} ${y} L${x} ${y + r * 1.3} L${x - r * 1.3} ${y} Z" ${a}/>`;
}
function glyph(agent) {
  return `<svg width="12" height="12" viewBox="-7 -7 14 14" aria-hidden="true">${marker(agent.shape, 0, 0, 5, `fill="${agent.color}"`)}</svg>`;
}
function yTicks(min, max, n) {
  const span = max - min || 1, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= n) || step0;
  const out = []; for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(v);
  return out;
}

/* ---------- charts ---------- */
function chartTimeline(rows) {
  const resolved = rows.filter(r => r.status === "resolved"), pending = rows.filter(r => r.status !== "resolved");
  if (!rows.length) return `<div class="chart-empty">No forecasts for this selection.</div>`;

  const W = 940, H = 330, M = { t: 14, r: 16, b: 26, l: 46 };
  const realizedPts = Object.entries(realizedByTarget).map(([d, v]) => [toMs(d), v]).sort((a, b) => a[0] - b[0]);
  const xsAll = rows.map(r => toMs(r.target)).concat(realizedPts.map(p => p[0]));
  const x0 = Math.min(...xsAll) - 2 * dayMs, x1 = Math.max(...xsAll) + 2 * dayMs;
  const ysAll = rows.flatMap(r => [r.p10, r.p90]).concat(realizedPts.map(p => p[1]));
  const yPad = (Math.max(...ysAll) - Math.min(...ysAll)) * 0.06 || 1;
  const y0 = Math.min(...ysAll) - yPad, y1 = Math.max(...ysAll) + yPad;
  const X = lin(x0, x1, M.l, W - M.r), Y = lin(y0, y1, H - M.b, M.t);

  let s = "";
  yTicks(y0, y1, 6).forEach(v => {
    s += `<line x1="${M.l}" y1="${Y(v)}" x2="${W - M.r}" y2="${Y(v)}" stroke="var(--rule)" stroke-width="1"/>`;
    s += `<text x="${M.l - 7}" y="${Y(v) + 3.5}" text-anchor="end" class="axt">${v.toFixed(0)}</text>`;
  });
  const nXT = Math.min(10, Math.round((x1 - x0) / (3 * dayMs)));
  for (let i = 0; i <= nXT; i++) {
    const ms = x0 + (x1 - x0) * i / nXT;
    s += `<text x="${X(ms)}" y="${H - 8}" text-anchor="middle" class="axt">${fmtDay(ms)}</text>`;
  }
  if (DATA.data_through) {
    const fx = X(toMs(DATA.data_through));
    s += `<line x1="${fx}" y1="${M.t}" x2="${fx}" y2="${H - M.b}" stroke="var(--rule-2)" stroke-width="1.5" stroke-dasharray="5 4"/>`;
    // Flip the label to the left of the line when it would clip the right edge.
    const flip = fx > W - 130;
    s += `<text x="${fx + (flip ? -5 : 5)}" y="${M.t + 9}" ${flip ? 'text-anchor="end"' : ""} class="axt">prices through ${DATA.data_through.slice(5)}</text>`;
  }

  const groups = {};
  rows.forEach(r => (groups[r.stream + "|" + r.cutoff + "|" + r.status] ||= []).push(r));

  function drawOne(agent, x, p10, p50, p90, hollow, tip, whisk) {
    const c = agent.color, fill = hollow ? "var(--panel)" : c;
    let g = `<line x1="${x}" y1="${Y(p10)}" x2="${x}" y2="${Y(p90)}" stroke="${c}" stroke-width="4" stroke-linecap="round" opacity="${hollow ? 0.35 : 0.28}"/>`;
    if (whisk) g += `<line x1="${x}" y1="${Y(whisk[0])}" x2="${x}" y2="${Y(whisk[1])}" stroke="${c}" stroke-width="1.5"/>`;
    g += marker(agent.shape, x, Y(p50), 4.5, `fill="${fill}" stroke="${c}" stroke-width="1.6"`);
    return `<g class="ht" data-tip="${esc(tip)}">${marker(agent.shape, x, Y(p50), 11, 'fill="transparent"')}${g}</g>`;
  }

  if (realizedPts.length > 1) s += `<polyline points="${realizedPts.map(p => X(p[0]) + "," + Y(p[1]).toFixed(1)).join(" ")}" fill="none" stroke="var(--ink)" stroke-width="2" stroke-linejoin="round"/>`;

  if (perfState.agg === "med") {
    Object.values(groups).forEach(g => {
      const agent = AGENT_BY_ID[g[0].stream], hollow = g[0].status !== "resolved";
      const x = X(toMs(g[0].target)) + DODGE[g[0].stream];
      const p50s = g.map(r => r.p50);
      const tip = `<b>${agent.label}</b> · ${g.length} run${g.length > 1 ? "s" : ""} · cutoff ${g[0].cutoff}<br>` +
        `median P50 <b>${median(p50s).toFixed(2)}</b> (${Math.min(...p50s).toFixed(2)}–${Math.max(...p50s).toFixed(2)})<br>` +
        `median band ${median(g.map(r => r.p10)).toFixed(2)}–${median(g.map(r => r.p90)).toFixed(2)}` +
        (g[0].realized != null ? `<br>realized <b>${g[0].realized.toFixed(2)}</b>` : "<br>pending");
      s += drawOne(agent, x, median(g.map(r => r.p10)), median(p50s), median(g.map(r => r.p90)), hollow, tip,
        g.length > 1 ? [Math.min(...p50s), Math.max(...p50s)] : null);
    });
  } else {
    rows.forEach(r => {
      const agent = AGENT_BY_ID[r.stream], hollow = r.status !== "resolved";
      const x = X(toMs(r.target)) + DODGE[r.stream] + (Math.abs(hashJitter(r.run_id)) % 5) - 2;
      const tip = `<b>${agent.label}</b> · ${r.run_id}<br>P10 ${r.p10.toFixed(2)} · P50 <b>${r.p50.toFixed(2)}</b> · P90 ${r.p90.toFixed(2)}` +
        (r.realized != null ? `<br>realized <b>${r.realized.toFixed(2)}</b> · err ${(r.p50 - r.realized).toFixed(2)}` : "<br>pending");
      s += `<g opacity="0.75">${drawOne(agent, x, r.p10, r.p50, r.p90, hollow, tip, null)}</g>`;
    });
  }
  const nR = resolved.length, nP = pending.length;
  return `<div class="chart-scroll"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Forecast vs actual timeline">${s}</svg></div>
    <div class="chart-note">Solid line = realized WTI. Filled marker = resolved forecast, hollow = pending (right of the dashed frontier).
    Band = P10–P90${perfState.agg === "med" ? " (median across same-cutoff runs; thin whisker = min–max of P50s)" : " (every run drawn)"}.
    ${nR} resolved · ${nP} pending horizon${nR + nP === 1 ? "" : "s"} shown.</div>`;
}
function hashJitter(s) { let h = 0; for (const ch of s) h = (h * 31 + ch.charCodeAt(0)) | 0; return h; }

function chartCoverage(rows) {
  const res = rows.filter(r => r.status === "resolved");
  if (!res.length) return `<div class="chart-empty">Nothing resolved yet for this selection.</div>`;
  const active = AGENTS.filter(a => perfState.agents[a.id] && res.some(r => r.stream === a.id));

  let bars = "";
  active.forEach(a => {
    const mine = res.filter(r => r.stream === a.id), inN = mine.filter(r => r.covered).length;
    const pct = Math.round(inN / mine.length * 100);
    bars += `<div class="hbar-row"><span class="hbar-name">${glyph(a)}${a.label}</span>
      <span class="hbar-track"><span class="hbar-fill" style="width:${pct}%;background:${a.color}"></span><span class="hbar-target" style="left:80%"></span></span>
      <span class="hbar-val">${pct}% inside · ${inN}/${mine.length}</span></div>`;
  });

  const W = 940, rowH = 26, M = { t: 8, r: 16, b: 24, l: 118 };
  const H = M.t + active.length * rowH + M.b;
  const xs = res.map(r => toMs(r.target));
  const x0 = Math.min(...xs) - dayMs, x1 = Math.max(...xs) + dayMs;
  const X = lin(x0, x1, M.l, W - M.r);
  let s = "";
  const nXT = Math.min(10, Math.round((x1 - x0) / (3 * dayMs)) || 1);
  for (let i = 0; i <= nXT; i++) {
    const ms = x0 + (x1 - x0) * i / nXT;
    s += `<text x="${X(ms)}" y="${H - 7}" text-anchor="middle" class="axt">${fmtDay(ms)}</text>`;
  }
  active.forEach((a, i) => {
    const cy = M.t + i * rowH + rowH / 2;
    s += `<text x="${M.l - 10}" y="${cy + 3.5}" text-anchor="end" class="axt">${a.label}</text>`;
    s += `<line x1="${M.l}" y1="${cy}" x2="${W - M.r}" y2="${cy}" stroke="var(--rule)"/>`;
    res.filter(r => r.stream === a.id).forEach(r => {
      const x = X(toMs(r.target)) + DODGE[r.stream] / 3;
      const tip = `<b>${a.label}</b> · ${r.run_id}<br>band ${r.p10.toFixed(2)}–${r.p90.toFixed(2)} · realized ${r.realized.toFixed(2)}<br><b>${r.covered ? "inside" : "outside"}</b> P10–P90`;
      s += r.covered
        ? `<g class="ht" data-tip="${esc(tip)}">${marker("circle", x, cy, 8, 'fill="transparent"')}${marker(a.shape, x, cy, 3.6, `fill="${a.color}"`)}</g>`
        : `<g class="ht" data-tip="${esc(tip)}"><circle cx="${x}" cy="${cy}" r="8" fill="transparent"/><path d="M${x - 4} ${cy - 4} L${x + 4} ${cy + 4} M${x - 4} ${cy + 4} L${x + 4} ${cy - 4}" stroke="${a.color}" stroke-width="2"/></g>`;
    });
  });
  return bars + `<div class="chart-scroll" style="margin-top:8px"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Coverage by target date">${s}</svg></div>
    <div class="chart-note">Dashed mark in each bar = the 80% design target. Strip: marker = actual inside the P10–P90 band, ✕ = outside. Every run counts individually.</div>`;
}

function dirBars(title, entries, note) {
  let rows = "";
  entries.forEach(e => {
    if (e.na) { rows += `<div class="hbar-row"><span class="hbar-name">${glyph(e.agent)}${e.agent.label}</span><span class="chart-note" style="margin:0">n/a — ${e.na}</span></div>`; return; }
    const decided = e.hits + e.misses, pct = decided ? Math.round(e.hits / decided * 100) : 0;
    rows += `<div class="hbar-row"><span class="hbar-name">${glyph(e.agent)}${e.agent.label}</span>
      <span class="hbar-track"><span class="hbar-fill" style="width:${pct}%;background:${e.agent.color}"></span><span class="hbar-target" style="left:50%"></span></span>
      <span class="hbar-val">${decided ? pct + "% of " + decided + " calls" : "no directional calls"}${e.nocalls ? " · " + e.nocalls + " no-call" : ""}</span></div>`;
  });
  return `<div class="chart-title">${title}</div><div class="chart-sub">${note}</div>${rows}`;
}

function chartDirection(rows) {
  const res = rows.filter(r => r.status === "resolved");
  const active = AGENTS.filter(a => perfState.agents[a.id]);
  if (!res.length) return `<div class="chart-empty">Nothing resolved yet for this selection.</div>`;

  const stats = (mine, callOf, outcomeOf) => {
    let hits = 0, misses = 0, nocalls = 0;
    mine.forEach(r => {
      const call = callOf(r), out = outcomeOf(r);
      if (Math.abs(call) < FLAT_EPS || Math.abs(out) < FLAT_EPS) { nocalls++; return; }
      if (Math.sign(call) === Math.sign(out)) hits++; else misses++;
    });
    return { hits, misses, nocalls };
  };

  const panelA = active.map(agent => {
    const mine = res.filter(r => r.stream === agent.id && r.last != null);
    const skipped = res.filter(r => r.stream === agent.id && r.last == null).length;
    const e = { agent, ...stats(mine, r => r.p50 - r.last, r => r.realized - r.last) };
    if (skipped) e.nocalls += skipped;
    return e;
  });
  const panelB = active.map(agent => {
    const mine = res.filter(r => r.stream === agent.id && r.ens != null && r.overlay != null);
    if (!mine.length) return { agent, na: "no numerical baseline to measure against" };
    return { agent, ...stats(mine, r => r.overlay, r => r.realized - r.ens) };
  });

  return `<div class="chart-grid2">
    <div>${dirBars("vs last close at cutoff", panelA, "Did the published P50 call the direction the price actually moved? All agents. Dashed mark = coin-flip 50%.")}</div>
    <div>${dirBars("vs numerical ensemble (the LLM overlay alone)", panelB, "Did the overlay push the ensemble the way the price actually went? Zero-overlay runs are no-calls.")}</div>
  </div>
  <div class="chart-note">Calls or realized moves smaller than $${FLAT_EPS.toFixed(2)} count as no-call, not as wrong.</div>`;
}

function scatterSvg(pts, opts) {
  const W = 460, H = 360, M = { t: 12, r: 14, b: 34, l: 46 };
  const X = lin(opts.x0, opts.x1, M.l, W - M.r), Y = lin(opts.y0, opts.y1, H - M.b, M.t);
  let s = "";
  if (opts.quadrants) {
    s += `<rect x="${X(0)}" y="${M.t}" width="${W - M.r - X(0)}" height="${Y(0) - M.t}" fill="var(--panel-2)"/>`;
    s += `<rect x="${M.l}" y="${Y(0)}" width="${X(0) - M.l}" height="${H - M.b - Y(0)}" fill="var(--panel-2)"/>`;
    s += `<text x="${W - M.r - 6}" y="${M.t + 12}" text-anchor="end" class="axt">directionally helpful</text>`;
    s += `<text x="${M.l + 6}" y="${H - M.b - 8}" class="axt">directionally helpful</text>`;
  }
  yTicks(opts.y0, opts.y1, 6).forEach(v => {
    s += `<line x1="${M.l}" y1="${Y(v)}" x2="${W - M.r}" y2="${Y(v)}" stroke="var(--rule)"/>`;
    s += `<text x="${M.l - 6}" y="${Y(v) + 3.5}" text-anchor="end" class="axt">${v.toFixed(1)}</text>`;
  });
  yTicks(opts.x0, opts.x1, 6).forEach(v => {
    s += `<text x="${X(v)}" y="${H - M.b + 14}" text-anchor="middle" class="axt">${v.toFixed(1)}</text>`;
  });
  if (opts.zeroAxes) s += `<line x1="${X(0)}" y1="${M.t}" x2="${X(0)}" y2="${H - M.b}" stroke="var(--rule-2)"/><line x1="${M.l}" y1="${Y(0)}" x2="${W - M.r}" y2="${Y(0)}" stroke="var(--rule-2)"/>`;
  const dLo = Math.max(opts.x0, opts.y0), dHi = Math.min(opts.x1, opts.y1);
  s += `<line x1="${X(dLo)}" y1="${Y(dLo)}" x2="${X(dHi)}" y2="${Y(dHi)}" stroke="var(--rule-2)" stroke-dasharray="5 4" stroke-width="1.5"/>`;
  pts.forEach(p => {
    s += `<g class="ht" data-tip="${esc(p.tip)}">${marker(p.agent.shape, X(p.x), Y(p.y), 10, 'fill="transparent"')}` +
      marker(p.agent.shape, X(p.x), Y(p.y), 4.2,
        p.hollow ? `fill="var(--panel)" stroke="${p.agent.color}" stroke-width="1.6"` : `fill="${p.agent.color}" fill-opacity="0.85"`) + `</g>`;
  });
  s += `<text x="${(M.l + W - M.r) / 2}" y="${H - 4}" text-anchor="middle" class="axt">${opts.xlabel}</text>`;
  s += `<text transform="translate(11 ${(M.t + H - M.b) / 2}) rotate(-90)" text-anchor="middle" class="axt">${opts.ylabel}</text>`;
  return `<div class="chart-scroll"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${opts.xlabel} vs ${opts.ylabel}">${s}</svg></div>`;
}

function chartShift(rows) {
  const pts = rows.filter(r => r.status === "resolved" && r.ens != null && r.overlay != null).map(r => ({
    agent: AGENT_BY_ID[r.stream], x: r.realized - r.ens, y: r.overlay,
    tip: `<b>${AGENT_BY_ID[r.stream].label}</b> · ${r.run_id}<br>needed shift ${(r.realized - r.ens).toFixed(2)} · proposed ${r.overlay.toFixed(2)}`,
  }));
  if (!pts.length) return `<div class="chart-empty">No resolved runs with an ensemble baseline in this selection.</div>`;
  const mx = Math.max(...pts.map(p => Math.abs(p.x)), 0.5) * 1.1, my = Math.max(...pts.map(p => Math.abs(p.y)), 0.5) * 1.1;
  const m = Math.max(mx, my);
  return scatterSvg(pts, { x0: -m, x1: m, y0: -m, y1: m, quadrants: true, zeroAxes: true,
    xlabel: "needed shift = realized − ensemble P50 ($)", ylabel: "proposed shift = overlay ($)" }) +
    `<div class="chart-note">Diagonal = perfectly sized shift. Shaded quadrants = shift pushed the right way. Points on y=0 are runs where the policy granted no move. Coach streams only — the adaptive agent has no ensemble baseline. n=${pts.length}.</div>`;
}

function chartWidth(rows) {
  const pts = rows.filter(r => r.status === "resolved").map(r => ({
    agent: AGENT_BY_ID[r.stream], x: Math.abs(r.realized - r.p50), y: (r.p90 - r.p10) / 2, hollow: !r.covered,
    tip: `<b>${AGENT_BY_ID[r.stream].label}</b> · ${r.run_id}<br>|realized − P50| ${Math.abs(r.realized - r.p50).toFixed(2)} · half-width ${((r.p90 - r.p10) / 2).toFixed(2)}<br>${r.covered ? "inside" : "outside"} P10–P90`,
  }));
  if (!pts.length) return `<div class="chart-empty">Nothing resolved yet for this selection.</div>`;
  const m = Math.max(...pts.map(p => Math.max(p.x, p.y)), 1) * 1.12;
  return scatterSvg(pts, { x0: 0, x1: m, y0: 0, y1: m, quadrants: false, zeroAxes: false,
    xlabel: "actual miss = |realized − P50| ($)", ylabel: "promised half-width = (P90−P10)/2 ($)" }) +
    `<div class="chart-note">Above the diagonal ≈ the promised range was wide enough for the miss; below = too narrow. Hollow marker = realized fell outside P10–P90. n=${pts.length}.</div>`;
}

/* ---------- assembly ---------- */
let perfBuilt = false;
function buildPerf() {
  const root = $("tab-perf");
  root.innerHTML = `
    <div class="perf-rail">
      <div class="perf-group"><span class="perf-glabel">Agents</span>${AGENTS.map(a => `
        <label class="agent-check" data-agent="${a.id}">${glyph(a)}<input type="checkbox" ${perfState.agents[a.id] ? "checked" : ""}>${a.label}</label>`).join("")}
      </div>
      <div class="perf-group"><span class="perf-glabel">Horizon</span>
        <span class="seg" id="perf-h">${[5, 10, 21].map(h => `<button aria-pressed="${perfState.horizon === h}" data-h="${h}">${h}B</button>`).join("")}</span>
      </div>
      <div class="perf-group"><span class="perf-glabel">Same-day runs</span>
        <span class="seg" id="perf-agg">
          <button aria-pressed="${perfState.agg === "med"}" data-agg="med">median + spread</button>
          <button aria-pressed="${perfState.agg === "all"}" data-agg="all">all runs</button>
        </span>
      </div>
    </div>
    <div class="chart-panel"><div class="chart-title">Forecast vs actual</div>
      <div class="chart-sub" id="sub-timeline"></div><div id="ch-timeline"></div></div>
    <div class="chart-panel"><div class="chart-title">Was the actual inside P10–P90?</div>
      <div class="chart-sub">Target 80% by construction — the band is the 10th-to-90th percentile.</div><div id="ch-coverage"></div></div>
    <div class="chart-panel"><div class="chart-title">Directional hit-rate</div>
      <div class="chart-sub">Two references: the market (last close when the forecast was made), and the numerical ensemble (what the LLM overlay added to it).</div>
      <div id="ch-direction"></div></div>
    <div class="chart-grid2">
      <div class="chart-panel"><div class="chart-title">Shift quality — proposed vs needed</div>
        <div class="chart-sub">Each resolved run: how far the LLM moved the centre vs how far it should have.</div><div id="ch-shift"></div></div>
      <div class="chart-panel"><div class="chart-title">Width calibration — promised vs actual miss</div>
        <div class="chart-sub">Was the interval sized to the error that actually happened?</div><div id="ch-width"></div></div>
    </div>`;

  root.querySelectorAll(".agent-check input").forEach(cb => cb.addEventListener("change", (ev) => {
    const id = ev.target.closest(".agent-check").dataset.agent;
    perfState.agents[id] = ev.target.checked; persistPerf(); renderPerf();
  }));
  $("perf-h").querySelectorAll("button").forEach(b => b.addEventListener("click", () => {
    perfState.horizon = +b.dataset.h; persistPerf();
    $("perf-h").querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", String(x === b)));
    renderPerf();
  }));
  $("perf-agg").querySelectorAll("button").forEach(b => b.addEventListener("click", () => {
    perfState.agg = b.dataset.agg; persistPerf();
    $("perf-agg").querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", String(x === b)));
    renderPerf();
  }));

  root.addEventListener("mousemove", (e) => {
    const t = e.target.closest(".ht"), tip = $("perf-tip");
    if (!t) { tip.style.display = "none"; return; }
    tip.innerHTML = t.dataset.tip; tip.style.display = "block";
    const vw = window.innerWidth, r = tip.getBoundingClientRect();
    tip.style.left = Math.min(e.clientX + 14, vw - r.width - 8) + "px";
    tip.style.top = (e.clientY + 14) + "px";
  });
  root.addEventListener("mouseleave", () => { $("perf-tip").style.display = "none"; });
  perfBuilt = true;
}

function renderPerf() {
  const rows = seldRows();
  root_sync_checks();
  $("sub-timeline").textContent = `h=${perfState.horizon} business days · plotted at target date · live_forward runs only (replayed cutoffs excluded) · ${rows.length} forecast-horizons in selection`;
  $("ch-timeline").innerHTML = chartTimeline(rows);
  $("ch-coverage").innerHTML = chartCoverage(rows);
  $("ch-direction").innerHTML = chartDirection(rows);
  $("ch-shift").innerHTML = chartShift(rows);
  $("ch-width").innerHTML = chartWidth(rows);
}
function root_sync_checks() {
  document.querySelectorAll(".agent-check").forEach(l => l.dataset.off = perfState.agents[l.dataset.agent] ? "" : "1");
}

function showTab(which) {
  const perf = which === "perf";
  $("tab-records").hidden = perf; $("tab-perf").hidden = !perf;
  $("tabbtn-records").setAttribute("aria-selected", String(!perf));
  $("tabbtn-perf").setAttribute("aria-selected", String(perf));
  if (perf) { if (!perfBuilt) buildPerf(); renderPerf(); } else { $("perf-tip").style.display = "none"; }
}
$("tabbtn-records").addEventListener("click", () => showTab("records"));
$("tabbtn-perf").addEventListener("click", () => showTab("perf"));
</script>
"""


if __name__ == "__main__":
    main()
