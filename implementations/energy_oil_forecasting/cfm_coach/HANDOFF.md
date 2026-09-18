# CFM Coach — handoff

**Last updated:** 2026-08-18 · **Stage 1a complete and verified. Stage 1b built but deliberately not
wired in. Stage 1c (M1 harness) built and tested. Corpus: 11 records, 7 `live_forward`. Triple-run
measurement window running 2026-08-19 → 2026-10-30 (see finding 6 / decision below).**

Design doc (requirements, workflows, full task list): `~/.claude/plans/enchanted-discovering-tower.md`

Branch `main` (local only): `e191f81` (v5.0 package) → `52553ed` (coach stage 1a) → `607d21f` (run
corpus + v001 fix) → M1 harness. **Nothing pushed, and nothing should be.**
Daily job **installed and loaded**: `com.cfm.coach-daily`, weekdays 08:00 local.

---

## What this is

A standalone package that reads what `cfm_agent_v_5_0` produces, waits for forecasts to resolve, and
proposes — never applies — calibration corrections. Goal: a WTI forecast that can be relied on for both
direction and risk, **plus an explicit signal for when to distrust it**.

Three learning mechanisms, in scope order:

| | Mechanism | Who learns | Status |
|---|---|---|---|
| **M1** | Calibrate the constants converting judgment into quantiles | the machinery | **harness built; needs corpus** |
| **M2** | Memory of the agent's own track record | the LLM | not built (stage 1d) |
| **M3a** | Analogue lookup → trust report only, no prompt change | nobody (a read) | built, not wired |
| **M3b** | Analogue injection into the prompt | the LLM | deferred to stage 2 |

**M1 is the measurement instrument for M2.** M2 changes what the LLM says and nothing else can tell you
whether that helped. Build M1 first for that reason.

## Hard constraints (do not violate)

1. **`cfm_agent_v_5_0` is never modified.** Verify with
   `cd implementations/energy_oil_forecasting/cfm_agent_v_5_0 && shasum -a 256 -c MANIFEST.sha256`
   (43 files, all must be OK). Calibration is applied by *injecting* settings via
   `build_cfm_agent_config(settings=...)`, never by editing the package.
2. **Only `provenance="live_forward"` records are fitting evidence.** A run issued after its cutoff
   re-searches today's web; even when every cited source predates the cutoff, the *selection* is shaped
   by what turned out to matter. v5.0's leakage verifier inspects content, not retrieval, so it cannot
   catch this.
3. **Audit-only signals may inform trust, never the number.** The trust report is computed strictly after
   the forecast is final.
4. **Never pool records across `prompt_version` when fitting.** A different prompt is a different agent.
5. **Nothing auto-applies.** Every proposed change waits for a human.

## Commands

```bash
# one live forecast, recorded (~4 min, needs .env at repo root)
uv run python -m energy_oil_forecasting.cfm_coach.run_daily            # cutoff = today
uv run python -m energy_oil_forecasting.cfm_coach.run_daily 2026-08-14 # explicit cutoff

# what the scheduled job actually calls, 2026-08-19 through 2026-10-30: N runs at
# one cutoff, N = 3 inside the measurement window, 1 outside it (see run_daily_batch.py)
uv run python -m energy_oil_forecasting.cfm_coach.run_daily_batch            # cutoff = today
uv run python -m energy_oil_forecasting.cfm_coach.run_daily_batch 2026-08-19 # explicit cutoff

# convert pre-coach audit files at repo root into RunRecords (idempotent-ish: overwrites)
uv run python -m energy_oil_forecasting.cfm_coach.backfill

# the scoreboard: fidelity + resolved outcomes + ensemble/agent/current, no LLM, ~2s
uv run python -m energy_oil_forecasting.cfm_coach.report

uv run pytest implementations/energy_oil_forecasting/cfm_coach/tests/ -q
uv run ruff format implementations/energy_oil_forecasting/cfm_coach/
uv run ruff check  implementations/energy_oil_forecasting/cfm_coach/
```

## File map

| File | Does | State |
|---|---|---|
| `config.py` | `CoachSettings` (frozen, extra=forbid). All thresholds live here. | done |
| `schemas.py` | `RunRecord`, `HorizonRecord`, `ResolvedForecast`, `CalibrationVersion`, `CalibrationLayer` (with `.apply()`), `TrustReport` | done |
| `ledger.py` | `CalibrationLedger` — `.current(as_of)` by date, `.save()` immutable, `.to_agent_settings()` | done |
| `run_store.py` | `RunRecordStore`, `build_run_record()`, `agent_package_fingerprint()` — this is all of R6 | done |
| `run_daily.py` | Workflow A entry point, `classify_provenance()`, loads `.env` | done |
| `backfill.py` | Both historical audit shapes → `RunRecord` | done |
| `diagnostics.py` | `STATE_FEATURES` (8), `historical_states()`, `forward_moves()` | done |
| `case_base.py` | `MarketCaseBase` — 4,700 cases, `.neighbours()`, `.assess_novelty()`, `.move_dispersion()` | done |
| `trust.py` | `TrustReporter` — tier + one-line driver | **built, does not discriminate — see below** |
| `calibration/v001.json` | empty overlay + identity layer = true no-op | done |
| `runs/` | the corpus. ~90-120 KB per record | 11 records |
| `replay.py` | `ReplayEngine.replay()` / `.verify_fidelity()` / `.check_fidelity()`, `CoachedCfmPredictor` | done |
| `scoring.py` | `pinball_loss` (**new to the repo**), `crps_from_quantiles`, `crps_ensemble`, `ForecastScorer` | done |
| `outcomes.py` | `OutcomeResolver` — resolves, pends, and refuses stale prices | done |
| `report.py` | `CoachReporter` — the scoreboard + learning curve. Entry point. | done |
| `policy/comparison_policy.py` | `ComparisonPolicy` — the locked gate, nine conditions | done |
| `schemas.py` additions | `ScoreCard`, `Candidate`, `ConditionResult`, `ComparisonVerdict` | done |
| `proposal.py`, `run_cycle.py`, `self_audit.py` | workflow B + M2 | not built |
| `run_daily_batch.py` | `run_batch()`, `repeats_for()` — bounded triple-run window, 2026-08-19→10-30 | done |

## Corpus state (11 records)

| cutoff | issued | provenance | fitting? | note |
|---|---|---|---|---|
| 2026-03-02 | 2026-08-13 23:08 (+164d) | `replayed_live_search` | no | settings recorded |
| 2026-03-02 | 2026-08-14 21:25 (+164d) | `replayed_live_search` | no | repeat of the same cutoff |
| 2026-03-03 | 2026-08-14 06:34 (+164d) | `replayed_live_search` | no | settings **inferred** (pre-R6) |
| 2026-03-03 | 2026-08-14 21:30 (+164d) | `replayed_live_search` | no | repeat of the same cutoff |
| 2026-08-14 | 2026-08-14 20:20 | `live_forward` | **yes** | first clean record |
| 2026-08-17 | 2026-08-17 08:04 | `live_forward` | **yes** | first *scheduled* run (job's first fire) |
| 2026-08-17 | 2026-08-17 20:06 | `live_forward` | **yes** | manual re-run — **first non-zero live overlay** |
| 2026-08-17 | 2026-08-17 20:10 | `live_forward` | **yes** | manual re-run, 4 min after the last |
| 2026-08-18 | 2026-08-18 08:03 | `live_forward` | **yes** | first *batched-schedule-eligible* day, still ran singly (batch starts 08-19) |
| 2026-08-18 | 2026-08-18 21:09 | `live_forward` | **yes** | manual re-run — see finding 9 |
| 2026-08-18 | 2026-08-18 21:14 | `live_forward` | **yes** | manual re-run, 4 min after the last |

The 2026-08-14 run: h=5 $80.20, h=10 $80.39, h=21 $79.58; overlay contributed **$0.00** because the agent
called `likely_reflected_in_model_data` and proposed `no_change` on every horizon (tier `none`,
"Neutral proposal requires no evidence"). Its three horizons resolve 2026-08-21 / 08-28 / 09-14.

The three 2026-08-17 runs are the same cutoff executed three times — see finding 6. The job fired for the
first time that morning (installed Friday evening with `RunAtLoad=false`, so it correctly skipped Friday
08:00 and the weekend).

## Findings that changed the design

**1. Both pre-existing audit files were replays, not live runs.** Issued 164 days after their cutoff. The
"+10.9% overlay improvement on 6/6" measured early in this project comes from those two runs and is
**not** evidence the overlay works live.

**2. The trust tier does not discriminate.** `TrustReporter` returns `high` for all three records —
including the two that were off by $27–$35 with 0/6 interval coverage. R4's retirement condition fired
before a single forecast resolved. Neither sub-check catches March 2026:
- OOD flag: 2026-03-03 sits at the 87.9th percentile of k-NN distance (threshold p99). Correct for a
  history containing 2008/2020/2022, but it means no warning.
- Analogue dispersion: implied/ensemble width ratios 1.5x / 0.9x / 1.1x at h=5/10/21 against realized
  moves of +$12 / +$25 / +$29. The 5 nearest analogues gave no warning.

**3. The agent's citation bookkeeping, not its market judgment, is the dominant source of run-to-run
variance — and it silently zeroes the overlay.** Running the same cutoff twice, same code, same model,
~15 hours apart:

| cutoff | issued | LLM proposed (h=21) | policy gave | why |
|---|---|---|---|---|
| 2026-03-03 | 06:34 | `large_up` / `substantially_wider` | `large_up`, tier **strong** | — |
| 2026-03-03 | 21:30 | `large_up` / `substantially_wider` | `no_change`, tier **none** | cited claims failed the source-subset contract |
| 2026-03-02 | 23:08 | `moderate_up` | `moderate_up`, tier **corroborated** | 11 resolved publishers |
| 2026-03-02 | 21:25 | `large_up` | `small_up`, tier **limited** | only 5 resolved publishers |

On 2026-03-03 the agent formed the **identical** view both times — same actions on all three horizons,
same confidence (0.9), same novelty — and the output went from **+$3.58 to +$0.00** purely because the
second run wired its claim → summary → source citations in a way `EvidencePolicy` rejected
("Every cited material claim must link to an accepted verified summary and cite at least one mechanically
resolved source associated with that summary").

Consequences, and they are large:

- Calibrating λ / ω / the action fractions means fitting a signal that is being **randomly zeroed by an
  unrelated failure mode**. Any λ fitted across such runs is biased toward 0 with inflated variance, and
  ~40 records will not be enough if a meaningful share of them are nullified this way.
- The highest-leverage fix is **not a calibration constant**. It is the claim-building step — a prompt or
  skill issue in `skills/claim-building/SKILL.md`, not a number in `config.py`.
- This is measurable **immediately**, with no waiting for horizons to resolve, which makes it exactly the
  kind of fast text-level signal M2 was meant to learn from.
- It was invisible before, because nothing had ever compared two runs of the same cutoff.

**4. The contamination test did not answer the contamination question.** Both the "original" and the
"re-run" of each March cutoff were replays issued in August, so this compared replay to replay, not live
to replay. It measured agent variance instead, which turned out to be the more valuable answer. A real
contamination test is now possible and cheap: `2026-08-14` is a `live_forward` record, so replaying that
cutoff in a month or two and diffing the assessments gives a clean live-vs-replayed comparison.

**5. Measured over 4,679 historical days** — Spearman rank correlation vs |forward move|:

| signal | \|move_5b\| | \|move_10b\| | \|move_21b\| |
|---|---|---|---|
| `vix_level_l1b` | **0.255** | **0.231** | **0.236** |
| `realized_volatility_21b` | 0.175 | 0.144 | 0.120 |
| `drawdown_63b` | −0.174 | −0.133 | −0.109 |
| `return_1b` | 0.103 | 0.070 | 0.074 |
| `oil_curve_contango_l1b` | −0.086 | −0.087 | −0.097 |
| `jump_zscore_63b` | 0.045 | 0.018 | 0.020 |

- **`jump_zscore_63b` has no predictive power.** This kills the earlier hypothesis that the overlay's
  reach should scale with market abnormality (which was built on the 2.93 jump z-score of 2026-03-03).
  A big jump today does not predict a big move next.
- **Best state signal is VIX, and it is still weak** (ρ ≈ 0.24). No state-based trust tier will
  discriminate strongly.
- **The usable effect is in the tail, not the median.** Calmest → most violent vol quintile: median
  |21-day move| rises only $4.11 → $5.40, but p90 rises $9.21 → **$16.07**. Width should scale with
  volatility; the centre should not. This is the leading M1 calibration hypothesis and it serves R1.

**6. The agent's *action selection* varies run to run on identical data — a second, distinct source of
overlay noise.** Cutoff 2026-08-17 executed three times. All three saw the **same inputs**
(`latest_observation=2026-08-14`, `latest_value=82.40`), confirmed from the stored diagnostics:

| run | physical_status | novelty | conf | proposed h=5 | tier h=5 | overlay h=5 / h=10 |
|---|---|---|---|---|---|---|
| A 08:04 | `elevated_risk` | likely_reflected | 0.85 | `no_change` | none (0 pub) | $0.00 / $0.00 |
| B 20:06 | `confirmed_disruption` | possibly_partly_reflected | 0.80 | **`small_up`/`moderately_wider`** | **corroborated (12 pub)** | **+$0.38 / +$0.48** |
| C 20:10 | `confirmed_disruption` | likely_reflected | 0.85 | `no_change` | none (0 pub) | $0.00 / $0.00 |

**B and C ran four minutes apart** and disagreed on whether to act at all. A→B is confounded (12 hours,
so `elevated_risk` → `confirmed_disruption` may be real news); **B→C is not.**

This is a *different mechanism* from finding 3. There the LLM formed one view twice and `EvidencePolicy`
zeroed the second on citation wiring. Here the LLM's own categorical action differs between draws.

Two consequences:

- **The evidence bar is clearable live.** B is the first `live_forward` record with a non-zero overlay,
  reaching tier `corroborated` on 12 resolved publishers. The zero-overlay pattern is not structural.
- **λ is threatened, ω is not.** On one day, one horizon, one dataset, the published overlay was $0.00 /
  $0.38 / $0.00. The run-to-run spread is the size of the signal. Fitting a centre gain across ~40
  records means estimating a mean whose per-observation noise dominates — biased toward zero with
  inflated variance, exactly as finding 3 predicted, by a different route. The width scale is unaffected:
  it acts post-engine on every run regardless of what the LLM proposes.

**Before committing corpus to λ, measure this.** Run N=5 at a single cutoff on a few days and quantify the
action-selection distribution. ~20 minutes per day sampled, and it answers whether λ is fittable at all.

**7. The interval problem is an *ensemble* problem, and it is calibratable on history right now.**
This inverts a priority the plan had recorded as a low-value side quest, so it is worth stating carefully.

From the scoreboard, on the 12 resolved horizons:

| variant | mean P10–P90 width | 80% coverage |
|---|---|---|
| `ensemble` (no LLM overlay at all) | 9.74 | **0%** |
| `agent` (as published) | 11.08 | **0%** |

**The ensemble's own intervals are too narrow before the LLM touches anything.** R1 is therefore mostly a
base-layer calibration problem, not an overlay problem.

And the coach's width lever sits exactly there. When the agent proposes `no_change`/`unchanged` — **3 of
the 4 live_forward runs so far** — `PythonForecastEngine`'s `neutral` branch returns the ensemble
quantiles *exactly*, so the published forecast **is** the ensemble. Scaling ω on such a run is scaling the
ensemble's interval, nothing else.

Which means **ω is fittable on ensemble-only runs**: `CfmEnsemblePredictor`
(`cfm_agent_v_5_0/models/ensemble.py:18`, import only) runs with no LLM and no web search. No research
means no research *selection*, so R7's contamination argument — which is specifically about selection
being shaped by what turned out to matter — does not apply to it. Years of history, available today,
minutes to compute.

Contrast with λ, which needs ~40 live records **and** is threatened by finding 6's action variance. The
two levers are in completely different positions and should stop being planned as one job.

**The design decision this forces.** `Provenance` already includes an `ensemble_only` literal, but
`CoachSettings.fitting_eligible_provenance` is `("live_forward",)`, so such records fail the gate's first
condition today. The argument for admitting them **for width-only fits** is that R7 exists to exclude
contaminated research selection, and a run with no research cannot have any. That is the rule applied to a
case it was not written for, not a loophole — but it must be an explicit, argued change to `config.py`,
and it must stay scoped to ω. The moment a fit touches the centre, `live_forward` only applies again.

**8. Cutoff honesty held under a live test.** The price parquet gained 2026-08-17's close *during* run B
(5686 → 5687 rows, file rewritten 20:06). All three runs still reported `latest_observation=2026-08-14`,
because `released_at` is the next business day. Building the data service also **refreshes the yfinance
cache as a side effect**, so the daily job keeps prices current and resolution is not blocked on a manual
fetch step.

**9. A second same-cutoff triple, decided 2026-08-18 because of finding 6, confirms the variance is real
and shows a new shape.** Cutoff 2026-08-18 executed three times (08:03 scheduled, 21:09, 21:14 manual).
Unlike 2026-08-17, all three agreed on *direction* -- `confirmed_disruption`, every horizon moved up, and
the evidence bar was cleared every time (tier never dropped to `none`):

| run | h=5 proposed | h=5 tier (publishers) | overlay h=5 / h=10 / h=21 |
|---|---|---|---|
| A 08:03 | `moderate_up`/`moderately_wider` | strong (6) | +1.60 / +2.11 / +1.37 |
| B 21:09 | `small_up`/`substantially_wider` | corroborated (9) | +0.85 / +2.16 / +1.50 |
| C 21:14 | `moderate_up`/`substantially_wider` | strong (8) | +1.73 / +2.07 / +1.46 |

All three saw identical inputs (`latest_observation=2026-08-17`, `latest_value=84.50` in all three) --
cutoff honesty held again. So this is not finding 6's on/off pattern (act vs. decline to act). It is a
*third* pattern: the LLM reliably decides to act and on which side, but the **tier it clears and the size
of the move it is granted still vary** -- h=5's overlay ranges +0.85 to +1.73, essentially 2x, while h=10
and h=21 stayed tight (spreads of 0.09 and 0.13). Shorter horizons appear to carry more of the variance;
worth watching as more triples accumulate.

Net effect on the plan is unchanged from finding 6: this is more evidence that λ needs the variance
measured before it is fitted, not less. **The triple-run schedule (2026-08-19 → 2026-10-30, weekdays,
`run_daily_batch.py`) exists to accumulate this measurement systematically** instead of via ad hoc
manual reruns -- see the decision below.

## Decisions taken (2026-08-14)

| Decision | Choice |
|---|---|
| Trust tier | **Hold all of it.** Nothing trust-related is wired into `run_daily` until M1 can rank it against realized error (R4). `trust.py` and `case_base.py` stay in the tree, unused by the daily path. |
| Daily runs | **Scheduled on this Mac.** `scripts/launchd/com.cfm.coach-daily.plist`, installed to `~/Library/LaunchAgents/`, weekdays 08:00, `RunAtLoad=false`. Log: `~/Library/Logs/cfm-coach-daily.log`. Since 2026-08-19 it runs `run_daily_all` (all streams), not `run_daily_batch` (one). |
| Version control | **Commit everything, including `runs/`.** The corpus is irreplaceable and ~20 MB/year, so git doubles as its backup. |
| Contamination test | **Run it** — re-execute the two March cutoffs today and diff the assessments against the live originals. |
| Triple-run window (2026-08-18) | **Weekdays 2026-08-19 → 2026-10-30 run each cutoff 3x**, via `run_daily_batch.py` (the daily job now calls this instead of `run_daily` directly). Purpose: accumulate finding 6/9's action-selection variance systematically instead of via manual reruns, so there is enough evidence to decide whether λ is fittable before the corpus is spent on it. Bounded on purpose — 3x LLM spend and 3x corpus growth is not something to pay indefinitely. Reverts to 1 run/day automatically on 2026-11-02; no plist change needed then, see `streams.py`'s `TRIPLE_RUN_FROM`/`TRIPLE_RUN_THROUGH`. |
| v5.2 streams (2026-08-19) | **Two new daily streams for `cfm_agent_v_5_2`**, alongside the unchanged v5.0 one: `v52_advanced` (3 runs/day, every LLM call on `gemini-3.5-flash`) and `v52_lite` (10 runs/day, every call on `gemini-3.1-flash-lite-preview`). See the section below. |

## M1 (stage 1c) — built, tested, waiting on corpus

50 coach tests pass (19 pre-existing + 31 new); 175 repo tests pass. `ruff check`/`format` clean. v5.0
still verifies 43/43 and has zero git changes.

**The replay is exact.** All 5 records reproduce their recorded quantiles bit-for-bit under their own
calibration — `max |error| = 0.0`, not merely within tolerance. That includes
`2026-03-03__20260814T063442Z`, whose settings were **inferred** by the backfill, which independently
confirms the reconstruction was right. `verify_fidelity()` raises on any mismatch; the report prints
drift at the top instead of quietly scoring on top of it.

**Both calibration levers are live and independent**, checked on the real 2026-03-03 record at h=21:

| calibration | P50 | overlay | P10–P90 width |
|---|---|---|---|
| v001 (identity) | 72.964 | +3.583 | 15.527 |
| layer `width_scale=1.5` | 72.964 | +3.583 | **23.290** |
| `large_action_width_fraction` 0.30→0.20 | **71.770** | **+2.389** | 15.527 |

Width scales with the centre untouched, and the agent constant scales the overlay by exactly 0.20/0.30.

**The scoreboard, as of today** (`report.py`). All 12 resolved rows are `replayed_live_search`, so **none
of it is fitting evidence** — the report bands it separately and says so:

| variant | n | pinball | CRPS | MAE | cover80 | width |
|---|---|---|---|---|---|---|
| agent | 12 | 11.5434 | 21.4171 | 25.58 | **0.0%** | 11.08 |
| current (v001) | 12 | 11.5434 | 21.4171 | 25.58 | 0.0% | 11.08 |
| ensemble | 12 | 12.3681 | 22.8239 | 26.93 | 0.0% | 9.74 |

`current` == `agent` to the digit, which is the live proof that v001 is a true no-op. The overlay beats
the raw ensemble by 6.7% on pinball here — **do not repeat the earlier mistake and read that as evidence
the overlay works.** It is two March cutoffs, replayed, with ~0.7 effective independent observations.

**First real evidence lands 2026-08-21**, when the 2026-08-14 `live_forward` run's h=5 resolves.

### Gate design decisions worth knowing

- **The bootstrap blocks on origin, never on (origin, horizon).** A run's three horizons share one
  assessment, one market state and overlapping windows. Row-resampling shrinks the SE by ~√3 and
  manufactures significance; there is a test asserting the naive version looks more significant than the
  blocked one on identical data.
- **`Candidate.fitted_through` declares the holdout boundary.** Origins after it are the holdout; a
  candidate fitted through the last origin has no holdout and therefore cannot pass. `None` means nothing
  was fitted, so everything is out-of-sample.
- **Fidelity is a precondition, not a gate condition** — hence nine conditions, not ten. A corpus the
  coach cannot reproduce does not yield a failed comparison, it yields a meaningless one.
- **A pass is a proposal, not an application.** The verdict carries the candidate shrunk halfway toward
  the incumbent (`shrinkage_factor` 0.50) and still waits for a human.
- **Pinball is the primary metric.** CRPS-via-`crps_ensemble` blends everything into one number; the
  coach needs to know whether the *centre* or the *width* moved, so it needs a level-decomposable loss.
  `crps_ensemble` is still recorded per row so coach numbers stay comparable with the repo leaderboard.

Leading calibration hypothesis, already measured and now testable end-to-end: **width should scale with
volatility, the centre should not.** Calmest → most violent vol quintile moves the median |21-day move|
only $4.11 → $5.40 but the p90 $9.21 → $16.07.

## v5.2 streams (2026-08-19)

`cfm_agent_v_5_2` now runs daily alongside v5.0. The coach was hardwired to v5.0 in three ways, all of
which had to go first:

1. **Eight modules imported `cfm_agent_v_5_0` directly.** Now one does — `targets.py`. Everything else
   takes an `AgentTarget`. Adding v5.3 is one entry there, not a grep.
2. **`agent_package_fingerprint()` hardcoded v5.0's manifest path and prefix.** Reused unqualified for a
   v5.2 run it would have stamped that run with **v5.0's** fingerprint — a wrong identity field, and one
   that specifically defeats `single_package_fingerprint`, the condition whose whole job is to notice a
   corpus spanning two builds.
3. **One `runs/` directory, globbed wholesale.** v5.2 records landing there would not have added noise;
   they would have failed `single_package_fingerprint` and rejected every v5.0 candidate outright.

### The three streams

`streams.py` is now the only place a model or a run count is chosen.

| stream | agent | model | runs/day | corpus | ledger |
|---|---|---|---|---|---|
| `v50_lite` | v5.0 | `gemini-3.1-flash-lite-preview` | 3 → 1 after 2026-10-30 | `runs/` | `calibration/` |
| `v52_advanced` | v5.2 | `gemini-3.5-flash` | 3 | `runs_v52_advanced/` | `calibration_v52_advanced/` |
| `v52_lite` | v5.2 | `gemini-3.1-flash-lite-preview` | 10 | `runs_v52_lite/` | `calibration_v52_lite/` |

**A run makes five LLM calls, not one** — main agent, grounded search, search leakage verifier,
claim-support verifier, and v5.2's new structured-output retry. Two are constructor arguments and three
are settings fields, so a stream binds both halves; binding only `model=` would have left three calls on
the package default and made a nominally single-model run a mixed-model one. `AgentTarget` discovers the
settings fields by their `_model` suffix, and `test_streams.py` asserts the discovered set explicitly so
a rename breaks a test rather than silently unbinding a call.

**`v50_lite` deliberately does not bind all five.** v5.0 ships its two verifiers on `gemini-3.5-flash`
and every record in `runs/` was produced that way. Binding them would change what the agent *is* partway
through a live corpus, invisibly — `package_fingerprint` hashes the manifest, and the manifest does not
record which model a verifier called. This was caught during implementation, after the first version of
`streams.py` bound all three streams uniformly.

### What else changed

- **Tenth gate condition, `single_agent_model`.** The fingerprint cannot see the model. Directory
  separation is what actually keeps the corpora apart; this is what notices if that is ever bypassed.
- **A calibration overlay may not name an LLM field.** `to_agent_settings` raises. Which model answered
  is identity, like the fingerprint — not something a fit is allowed to change.
- **`ReplayEngine` refuses a record from another agent.** v5.0's and v5.2's engines agree *today*
  (`python_forecast_engine_v50` vs `_v52`), so a cross-replay would look perfectly faithful and start
  lying the moment they diverge.
- **`run_id` carries the stream** for the v5.2 corpora (`cfm_agent_v_5_2__lite__…`), since the two share
  an agent id. v5.0's ids are unchanged, so the 14 existing records stay addressable.
- **The scheduled job runs `run_daily_all`**, which walks every stream and absorbs a stream-level failure
  so a v5.0 problem cannot cost v5.2 its day. `--stream=` is repeatable and comma-separated.
- **Whole-repo `pytest` works again.** The two agents' test directories hold five identically-named
  modules; collection failed outright until `--import-mode=importlib` (plus a `pythonpath` entry for the
  coach's shared fakes). Renaming was not an option — each `MANIFEST.sha256` hashes its test filenames.

### Finding: `v52_advanced` hangs, and nothing below the coach bounds it (2026-08-19)

The first `v52_advanced` attempt ran **66 minutes** without producing a record or raising — CPU almost
idle, the LLM socket open and established, nothing arriving — and blocked `v52_lite` behind it. Killed.

What was ruled out:

- **The model is healthy.** Probed directly through the proxy: `gemini-3.5-flash` answers in ~1s. It is a
  **reasoning model** — every response carries `thinking_tokens`. At `max_tokens=16` it returns
  `finish_reason=length` with **empty content**, having spent the whole budget thinking; at 512 it answers
  normally. So an empty reply from it is token starvation, not an outage.
- **v5.2 itself is fine.** `v52_lite` runs the identical agent end-to-end in ~4 minutes.

The mechanism is `AdkTextRunner.run_text_async` in `aieng-forecasting`
(`methods/agentic/adk_runner.py`): `drain_run()` iterates ADK's `run_async` with **no timeout and no
iteration cap**, returning only when ADK emits a final response. A thinking model that keeps producing
tool calls, or never emits a final response, spins there forever. Note v5.2's *own* structured-output
retry is bounded (`structured_output_retry_timeout_seconds`) — it is only the initial ADK turn that is not.

Hypothesis, not yet proven: `max_output_tokens=24576` is shared between thinking and output, and under a
long persona + 5 skills + 3 tools the model exhausts it on reasoning, never reaches
`set_model_response`, and ADK retries. Testing this means either capping thinking or raising the budget —
both changes to `cfm_agent_v_5_2` or `aieng-forecasting`, neither of which the coach may make.

**Contained at the coach layer instead.** `run_daily_batch._time_limit` puts a
`ATTEMPT_TIMEOUT_SECONDS` (15 min, vs a ~4 min norm) `SIGALRM` ceiling on every attempt; a timeout is
counted as a failed attempt and retried, and logged louder than an ordinary failure because a stall will
probably stall again. `SIGALRM` rather than a worker thread because a hung attempt is blocked in a socket
read and a thread cannot be killed — a `concurrent.futures` timeout would hand back control while the run
kept holding its connection.

**This is a real defect in the first version of the daily job, not just a v5.2 quirk.** The retry budget
bounded *failures*; nothing bounded a *hang*, and from the outside the two are indistinguishable. Had this
shipped, an 08:00 fire could have hung indefinitely, produced nothing, and starved every stream behind it
— silently, since it never errors.

### Open

- **`v52_advanced` works — the blocker was E2B, not the model.** It sat 66 minutes inside one agent turn
  on its first attempt; with `code_execution_enabled=False` the same run completed in **6.4 minutes**,
  evidence tier `corroborated` at all three horizons. Cause: `AuditedCodeExecutionTool.run_code` calls
  `CodeInterpreter.run_code(code=...)` with no timeout, and the interpreter defaults to
  `code_execution_timeout_seconds=None` / `request_timeout_seconds=None` /
  `sandbox_create_max_attempts=12`, so a sandbox that will not create blocks the run with nothing to cut
  it off. A `faulthandler` dump caught it in `response_control.py:82 -> _run_coroutine_sync ->
  selectors.select`. Why only this stream: all 11 `v52_lite` runs recorded
  `code_execution_call_count: 0` — lite never reached for the tool, and `gemini-3.5-flash` is a thinking
  model (lite reports `thinking_tokens=0` on every probe) so it is far likelier to. Both v5.2 streams now
  run with the tool off; it is diagnostics-only and barred from Components #9/#10, so it cannot affect a
  forecast (R9). `v50_lite` keeps it — all 14 records in `runs/` were produced with it registered, and the
  tool set is not in `package_fingerprint`.
- **Worth raising with the v5.2 / `aieng-forecasting` owner**, two separate unbounded waits that turning
  the tool off does not fix: (1) `AuditedCodeExecutionTool.run_code` should pass a timeout — every agent
  using `CodeInterpreter` is exposed; (2) `build_adk_agent` wraps the main model in `LiteLlm` with no
  `timeout`, while every other LLM call in the repo passes one (`timeout=60.0` for search and the leakage
  verifier). That is why a stalled turn ran 66 minutes instead of failing and being retried.
- **`gemini-3.5-flash` thinking cost cannot be capped.** It rose steeply with payload in isolation
  (2 KB → 1.5 s, 40 KB → 2.3 s, 120 KB → 16.2 s / 3289 thinking tokens) and is erratic — two identical
  120 KB calls returned 3.4 s and 16.2 s. The proxy rejects `reasoning_effort` with a 400 and litellm
  rejects `thinking` outright, so there is no lever short of changing model. Re-measure runtime if the
  prompt grows.
- **Cost.** A weekday morning went from 3 runs to 16 — about an hour serial, >5x the LLM spend, with
  `v52_advanced` on the pricier model. Unlike `v50_lite`, the v5.2 streams have **no end date**, because
  none was specified. Set `window` on them once action variance is answered.
- The v5.2 corpora start empty. Nothing can be fitted on them for months, and `single_package_fingerprint`
  means they can never be pooled with v5.0's.

## Next

In priority order:

1. **Fit ω on the deterministic base corpus** (finding 7). `CfmEnsemblePredictor` over the existing
   51 + 18 backtest origins, or further back — no LLM, no contamination, minutes to run. This is the only
   part of M1 that does **not** wait two months for corpus, and it attacks R1 (coverage 0-of-12) directly.
   Needs the scoped `ensemble_only` decision in finding 7 taken first.
2. **Let the triple-run window accumulate** (findings 6, 9; decided 2026-08-18). The daily job now runs
   each cutoff 3x through 2026-10-30 on its own — nothing further to do here except periodically look at
   whether λ's estimability question has enough evidence to answer. Do not manually re-run cutoffs the
   batch already covers; that just duplicates rows within one origin without adding information.
3. **Keep the daily job running.** The centre work waits on corpus. First resolution: **2026-08-21**
   (h=5 of the 2026-08-14 run) — the first genuine evidence this project will have produced. The measured
   vol-quintile tail effect ($9.21 → $16.07 at p90) remains the ω hypothesis to test.
4. `proposal.py` + `run_cycle.py` — workflow B wiring (`CandidateGenerator` → gate → `ReviewQueue`).
   Buildable now; says nothing until the corpus arrives.
5. Side quest, cheap and independent: replay cutoff `2026-08-14` in a month or two and diff the
   assessment against the live original. That is the clean live-vs-replayed contamination test that the
   March comparison could not be.
6. **Raise with the v5.0 owner:** the three 2026-08-17 runs are a concrete, reproducible case of the agent
   reaching `confirmed_disruption` and then declining to act on it. That is an action-selection /
   claim-building question in the agent's territory, not a coach constant.

## Gotchas

- `build_cfm_agent_predictor(config)` pops from a module-level `_CONFIG` keyed by `id(config)` and raises
  if consumed twice. Any loop must call `build_cfm_agent_config()` fresh.
- `policy_mode="ensemble_locked"` **still runs the LLM**. For a genuinely offline numerical run use
  `models/ensemble.py::CfmEnsemblePredictor` directly.
- Quantile keys round-trip through JSON as strings; the schemas coerce them back to floats. There is a
  test for this — do not remove it.
- `run_daily.py` loads the repo-root `.env` itself. Without it you get "No API key was provided."
- **Re-running a cutoff is allowed and never overwrites.** `run_id` embeds the issue timestamp to the
  second, and nothing in the write path checks for an existing record. Two runs of the same cutoff are
  two records; the gate counts *origins* (a set of cutoffs), so they stay one origin.
- **`v001.effective_from` is deliberately `2004-01-01`**, not the day the coach was written. These are the
  constants v5.0 has always shipped with, so v001 was in force for any cutoff a replay might ask about.
  Dating it to "today" asserts the agent had no calibration before then (false) and makes
  `CalibrationLedger.current()` raise for every historical cutoff — which is exactly how it first broke.
- Langfuse 401s and LiteLLM `LoggingWorker` timeouts in the run log are telemetry noise, not failures.
  `tests/test_integration.py::test_langfuse_auth` fails at collection for the same reason and is
  **pre-existing** — verified by stashing. Run the repo suite with `--ignore=tests/test_integration.py`.
- ~~There is no pinball function in the repo~~ → `scoring.py::pinball_loss` now provides one.
- `crps_from_quantiles` integrates over a grid spanning [0.05, 0.95], so it is a consistent **lower
  bound** on true CRPS, not an absolute CRPS. Fine for paired comparison on a shared grid; do not quote
  it against outside numbers.
- `cfm_coach/tests/` is deliberately **not** a package (matching `cfm_agent_v_5_0/tests/`), so
  `test_m1_harness.py` uses `from conftest import ...`, not a relative import. Adding `__init__.py`
  would break that line.
- `OutcomeResolver` refuses to resolve against an observation more than `MAX_STALENESS_DAYS` (5) before
  the target date. That guard exists so a frozen parquet cannot silently score a forecast against a price
  from before it was issued.
- `data/yfinance/*.parquet` runs to 2026-08-14; `usl` (from 2007-12-06) is what bounds the case base.
