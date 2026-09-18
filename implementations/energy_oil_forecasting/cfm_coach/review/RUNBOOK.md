# Coach review agent — runbook

What it is: a weekly agent that reads resolved v5.2 runs, works out where the loss to the random walk comes from, tags repeated errors with pointers, and writes ranked proposals. It never edits v5.2, the live streams, or their ledgers. Design: `~/.claude/plans/review-the-plan-for-dazzling-kitten.md`.

## Commands (repo root)

```bash
# zero-spend: collect, cards, stats, seeds, gates on existing hypotheses, report
uv run python -m energy_oil_forecasting.cfm_coach.review.run_weekly --no-llm
# pre-flight only
uv run python -m energy_oil_forecasting.cfm_coach.review.run_weekly --dry-run
# the real thing (weekly budget, hard-stopped)          | the one-time bootstrap ($10 ceiling)
uv run python -m energy_oil_forecasting.cfm_coach.review.run_weekly
uv run python -m energy_oil_forecasting.cfm_coach.review.run_weekly --bootstrap
# the coach's own forecast (weekdays 12:00 via launchd; idempotent)
uv run python -m energy_oil_forecasting.cfm_coach.review.coached
# apply a decision you recorded in proposals/P-NNNN.yaml
uv run python -m energy_oil_forecasting.cfm_coach.review.accept --proposal P-0003 --stream v52_lite --by <you>
# compare a registered challenger with its champion
uv run python -m energy_oil_forecasting.cfm_coach.review.pairing --champion v52_advanced --challenger v53_<slug>_advanced
```

Tests: `uv run pytest -q implementations/energy_oil_forecasting/cfm_coach/tests -k review`.

## What lands where

| path | what | who writes |
|---|---|---|
| `review/weeks/<YYYY-Www>/` | `manifest.json` (inputs, probe result, usage, cost, gates), `report.md` (source of truth), `report.html`, `usage.jsonl`, `raw/` | weekly job |
| `review/cards/<stream>/<cutoff>.json` | case cards, cached by (runs, resolved horizons, builder version) | weekly job |
| `review/annotations/<stream>.jsonl` | triage / deep-dive / blind re-triage annotations, append-only | weekly job |
| `review/hypotheses.jsonl` | hypothesis events (seeded from IMPROVEMENTS.md on first run) | weekly job |
| `cfm_coach/proposals/P-NNNN.yaml` | proposals; **only `decision` is yours to edit** | weekly job / you |
| `review/candidates/<stream>/` | ready-to-accept calibration JSONs for numeric proposals | weekly job |
| `review/calibration/<stream>/vNNN.json` | the coach-owned ledger the coached forecast runs under | `accept.py` |
| `review/coached/<stream>/` | coached records: each v5.2 run replayed under the coach ledger, `provenance=coached_replay` | coached daily job |
| `review/challengers/<P-id>/` | drafted v5.3 package copy + `stream.yaml` + `REGISTER.md` for accepted text proposals | `accept.py` |
| `review/hindsight/` | post-cutoff search material; never rendered into a proposal | deep dive |
| `review/state.json`, `review/ship_watch.json`, `review/coach_metrics.jsonl` | progress, what changed underneath, self-metrics | weekly job |

## Deciding

Read `report.md` top down: the scope line says what can and cannot be concluded this week; the decomposition says where the loss sits; proposals are ranked by gain against the random walk and labelled by track. For each proposal set `decision.status` to `accepted`, `rejected` or `deferred` with `by`, `on`, `reason` in its YAML. Rejections are remembered by fingerprint and block re-proposal until three new cutoffs exist.

- **numeric** (layer / settings / ensemble weights): `accept.py` copies the candidate into the coach ledger; the coached forecast changes from the next business day; v5.2 does not.
- **text** (skill / persona / prompt): `accept.py` drafts a challenger package; follow its `REGISTER.md` by hand (targets.py, streams.py, the test assertion, pyproject).
- **code / data**: the brief is the deliverable; implement in a Claude Code session.

## Live-call steps that need a human go-ahead

1. P0 probe: the first real run sends one tiny call per reasoning-effort value; a 400 on `reasoning_effort` is recorded and the run continues without it.
2. Bootstrap (`--bootstrap`) after `cfm_coach.history` has finished and `fit_width` has run, so the numeric track has its omega candidate.

## Honest expectations

Numeric candidates within days of the history run finishing; process-error proposals can promote at bootstrap; outcome-dependent LLM proposals need ~8–10 weeks (three independent h10 windows across two episodes); a challenger verdict needs 12 shared cutoffs plus the h21 lag. Weekly cost is bounded by the pre-flight and the ledger, realistically $2–6 at litellm's price map.

## Raw transcripts and zero-cost re-validation

Every LLM stage keeps the model's raw output under `review/weeks/<week>/raw/`:
`<stream>/triage_<cutoff>.json` (the parsed triage object), `<stream>/deep_dive_<cutoff>.json`
(`raw` turns plus the full `transcript` with tool results), `synthesis.json` and
`synthesis_transcript.json`. Read the transcripts first when a loop reports
"no findings" or "no synthesis": they show which tools the model called and what it got back.

When the pointer rules change (a new alias, a tolerance), re-run validation over the stored
triage responses instead of paying for triage again. Annotation ids are unchanged, so the
rest of the week can be re-run with `run_weekly` and triage is skipped:

```bash
uv run python -m energy_oil_forecasting.cfm_coach.review.triage --revalidate --week 2026-W37
```

First live run (2026-09-13, $0.83): triage worked but dropped 42 tags on two validation bugs
(component rows cited as `base.h5.lightgbm.bias`, shares cited as percentages); both loops ran
out of turns without submitting. Fixed the same day: alias and percent handling, a forced
`submit_*` final turn with a countdown, transcripts saved, and the pricing tool now refuses
settings fields that exact replay cannot express (`ensemble_weights` goes through
`ensemble_reweight`). Outputs of that first pass are kept in `weeks/2026-W37/run1/`.
