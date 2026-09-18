# Running CFM Coach

A practical guide for teammates. What each command does, what it costs, and which ones call an LLM.

**Run everything from the repo root** (`CFMv6_0/`). Several paths — notably the price cache at
`data/yfinance/` — resolve relative to your current directory, so running from inside the package
produces confusing "file not found" errors.

---

## Start here

If you only ever run one command, run this one:

```bash
uv run python -m energy_oil_forecasting.cfm_coach.report
```

It prints the scoreboard: whether every stored run still replays exactly, which forecast horizons have
resolved, and how the agent compares against the raw numerical ensemble. It takes about two seconds,
needs no API key, makes no network calls, and costs nothing.

---

## Which commands use an LLM, and which do not

This is the most important thing to understand about the coach, because it decides what is free to run
and what is not.

| Command | Calls an LLM? | Network? | Needs `.env`? | Time | Cost |
|---|---|---|---|---|---|
| `report` | **No** | No | No | ~2 s | free |
| `pytest` (coach tests) | **No** | No | No | ~1 s | free |
| `backfill` | **No** | No | No | < 1 s | free |
| Replay / scoring / the gate (Python API) | **No** | No | No | ms | free |
| `run_daily` | **Yes** | Yes — 4 web searches | **Yes** | ~4 min | LLM tokens |
| `run_daily_batch` | **Yes** | Yes | **Yes** | ~12–40 min | one stream's daily count |
| `run_daily_all` | **Yes** | Yes | **Yes** | **~50 min** | both scheduled streams, 13 runs |

**Only the `run_daily*` entry points call an LLM.** Everything else is arithmetic over records
that already exist.

### Why almost none of it needs an LLM

This is the design decision the whole harness rests on, and it is worth understanding before you use it.

`cfm_agent_v_5_0`'s prompt contains **no numeric constants**. The LLM never emits a price. It emits
*categorical* judgments — `large_up`, `substantially_wider`, `likely_new_relative_to_model_data` — and
Python owns every number that turns those judgments into quantiles.

So once a run is recorded, its LLM output is fixed and stored. Asking "what would this same forecast have
looked like under different constants?" means re-running Python over a stored judgment. No LLM, no web
search, no tokens, and the answer is **exact** rather than an approximation.

That is why the coach can test a calibration change across the entire corpus in milliseconds, and why
only the one command that produces *new* evidence costs anything.

> **Gotcha:** setting v5.0's `policy_mode="ensemble_locked"` does **not** give you an offline agent run —
> it still calls the LLM and then discards the result. For a genuinely offline numerical forecast, use
> `models/ensemble.py::CfmEnsemblePredictor` directly.

---

## Setup

You need the full repo, not just the `cfm_coach/` folder. The coach imports v5.0's own `EvidencePolicy`
and `PythonForecastEngine` rather than reimplementing them — deliberately, so the coach's arithmetic can
never drift from the agent's — plus `energy_oil_forecasting/data.py` and the `aieng-forecasting`
framework.

```bash
cd CFMv6_0
uv sync
```

Check your checkout is healthy — offline, ~1 second:

```bash
uv run pytest implementations/energy_oil_forecasting/cfm_coach/tests/ -q
```

Expect `50 passed`. If anything fails here, stop and fix that before trusting any other output.

An API key in the repo-root `.env` is needed **only** for `run_daily`. Everything else works without one.

---

## 1. Read the scoreboard (offline)

```bash
uv run python -m energy_oil_forecasting.cfm_coach.report
```

Four things to look at, in order:

**Fidelity.** `all N record(s) replay exactly` means the coach can still reproduce every stored forecast
from its inputs. If you instead see `!! FIDELITY DRIFT`, stop — something changed underneath the corpus,
and every number below that line is untrustworthy until it is explained.

**The provenance bands.** The report prints scores twice: once for everything, and once for
`FITTING EVIDENCE ONLY (live_forward)`. **Only the second band is evidence.** A run executed at a past
cutoff re-searches today's web, so even when every cited source predates the cutoff, the *selection* of
sources is shaped by what turned out to matter. Those runs are scored and shown, but they cannot be used
to fit anything.

**The three variants.**

| Variant | What it is |
|---|---|
| `ensemble` | ARIMA + Kalman + LightGBM with no LLM overlay at all — the floor the agent must beat |
| `agent` | what was actually published — the frozen baseline |
| `current` | every record replayed under the calibration in force today |

`current` and `agent` are identical until a calibration is approved. That is expected, and it is a live
check that the baseline calibration (`v001`) really is a no-op.

**Effective independent observations.** Horizons overlap, so 60 daily runs at h=21 contain roughly 3
genuinely independent observations. The report prints this number so `n=60` never reads as more precision
than it carries.

---

## 2. Produce a new forecast and record it (uses an LLM)

This is the only command that creates new evidence, and the only one that costs anything.

```bash
# cutoff = today, default stream (v50_lite)
uv run python -m energy_oil_forecasting.cfm_coach.run_daily

# a specific cutoff
uv run python -m energy_oil_forecasting.cfm_coach.run_daily 2026-08-14

# a specific stream
uv run python -m energy_oil_forecasting.cfm_coach.run_daily --stream=v52_lite
```

What happens, in order:

1. Look up which calibration version is in force for that cutoff, **in that stream's own ledger**.
2. Turn it into the target agent's settings and inject it via `build_cfm_agent_config(settings=...)` —
   the agent package itself is never edited.
3. Run the agent: 4 cutoff-aware web searches → LLM assessment → evidence policy → forecast engine.
4. Apply the coach's post-engine calibration layer (currently the identity).
5. Write a `RunRecord` to that stream's runs directory (~90 KB).

Takes about 4 minutes and needs `GEMINI_API_KEY` in the repo-root `.env`. Without the key you get
`No API key was provided.`

### The four streams

A *stream* is one (agent package, LLM, cadence) triple with its own corpus and its own calibration
ledger. They are defined in `cfm_coach/streams.py` and are the only place a model or a run count is
chosen.

| stream | agent | model | runs/day | corpus |
|---|---|---|---|---|
| `v50_lite` | `cfm_agent_v_5_0` | `gemini-3.1-flash-lite-preview` | **retired 2026-09-08** — 0 | `runs/` |
| `v52_advanced` | `cfm_agent_v_5_2` | `gemini-3.5-flash` | 3 | `runs_v52_advanced/` |
| `v52_lite` | `cfm_agent_v_5_2` | `gemini-3.1-flash-lite-preview` | 10 | `runs_v52_lite/` |
| `v522_arima_lite` | `cfm_agent_v_5_2` **ARIMA-only** | `gemini-3.1-flash-lite-preview` | **not scheduled** — 0 | `runs_v522_arima_lite/` |

**`v50_lite` is retired from the schedule, not deleted.** Its last cutoff is 2026-09-08 (a full 3/3, 54
records total). The corpus is still read, fitted, scored and published on the corpus page; it simply
stops growing. Two names keep those halves apart: `STREAMS` is every stream that exists, and
`SCHEDULED_STREAMS` is what the weekday job walks. Run it by hand any time with
`--stream=v50_lite`. Note this is deliberately *not* done by closing its `window` — outside a window a
stream falls back to `OFF_WINDOW_RUNS_PER_DAY`, which is one run a day, not none.

**`v522_arima_lite` is registered but not yet scheduled, and has no corpus.** It is `v52_lite`'s twin
in every respect but one: ARIMA alone in the ensemble, no Kalman and no LightGBM, built through
`build_cfm_agent_config_arima_only` rather than `build_cfm_agent_config`. Model, cadence, tool set and
settings are held identical on purpose — that is what makes the pair a controlled comparison of the
ensemble itself. Start it by adding `V522_ARIMA_LITE` to `SCHEDULED_STREAMS`; until then it runs by hand
with `--stream=v522_arima_lite`, at the usual ~4 minutes and five LLM calls per run.

Two things to know before reading anything it produces:

- **Two numeric levers go inert.** `settings_overlay.ensemble_weights` has nothing to reweight when one
  model produces the forecast, and `model_disagreement_std` is identically zero. Both stay live on the
  three ensemble streams, which is why this is a fourth stream rather than a change to an existing one.
- **`history_v52_ensemble/` is not its history.** Those dates were re-run with all three models, so they
  are the look-ahead-free base for fitting the *ensemble* streams' range and anchor, not this one's. An
  ARIMA-only backfill is its own job, and nothing stops a fit from using the wrong one but this note.

**It shares a package with `v52_lite`, so its fingerprint prefix does the work the manifest cannot.**
Both targets hash the same `MANIFEST.sha256` — same files, different entry point — so `V52_ARIMA_ONLY`
carries `cfm_v5_2_arima_only_package` to keep `single_package_fingerprint` able to tell a three-model
corpus from a one-model one.

**The corpora are separate on purpose and must stay separate.** `ComparisonPolicy` requires the fitting
corpus to span exactly one package fingerprint *and* one model; a mixed directory does not produce a
noisier fit, it produces a rejected one. Three conditions defend this — `single_prompt_version`,
`single_package_fingerprint`, and `single_agent_model` — and the directory layout means none of them
should ever fire.

**One run makes five LLM calls, not one.** The main agent, the grounded search, the search leakage
verifier, the claim-support verifier, and (v5.2 only) the structured-output retry. Two are constructor
arguments and three are settings fields, so a stream binds both halves together; `run_daily` prints
every resolved model before spending a token.

The v5.2 streams put **all five** on their stream's model. `v50_lite` is the deliberate exception: v5.0
ships its two verifiers on `gemini-3.5-flash`, every record already in `runs/` was produced that way, and
changing it now would alter what the agent *is* partway through a live corpus — invisibly, because the
package fingerprint hashes the manifest and the manifest does not record which model a verifier called.

### What the cutoff date does to your record

`run_daily` classifies the run automatically, and the classification is permanent:

| You run | Provenance | Fitting evidence? |
|---|---|---|
| today's cutoff | `live_forward` | **yes** |
| a past cutoff | `replayed_live_search` | no — visible in the corpus, excluded from fitting |

So if you want to watch the agent work end to end, running a past cutoff is completely safe. It lands in
the corpus clearly labelled, and the report and the statistical gate both exclude it on their own. You
cannot accidentally contaminate the evidence base by doing this.

### A recommendation about the daily run

Each `live_forward` record is irreplaceable — a day without one is evidence that cannot be recovered
later, because re-running that cutoff tomorrow produces a `replayed_live_search` record instead.

A scheduled job already runs this on Naman's machine each weekday morning at 08:00
(`scripts/launchd/com.cfm.coach-daily.plist`, logging to `~/Library/Logs/cfm-coach-daily.log`). It runs
`run_daily_all`, which walks every **scheduled** stream in order:

```bash
# what the scheduled job runs -- both v5.2 streams, 13 runs, about fifty minutes
uv run python -m energy_oil_forecasting.cfm_coach.run_daily_all

# one stream's full daily count -- including the retired v50_lite, if ever wanted
uv run python -m energy_oil_forecasting.cfm_coach.run_daily_all --stream=v52_lite
```

**It takes about fifty minutes.** Thirteen runs at roughly four minutes each, serial — an 08:00 fire is
still working close to 09:00. That is the cost of ten draws a day, not a hang.

**Sleep does not just delay it — it disarms the watchdog.** `run_daily_batch`'s 15-minute per-attempt
ceiling is armed with `SIGALRM`, and that timer does not advance while the machine is asleep, so it only
counts awake time. On 2026-09-08 the job fired at 08:13 on battery, slept through ten cycles, and one
attempt spanned 7h40m of wall clock for 13 minutes of CPU without the cap ever firing; the first record
was written one second after the lid opened at 15:52. Keep the machine awake and on power on mornings
the runs matter.

A stream that fails outright is logged and the next one still runs, so a v5.0 problem cannot cost v5.2 its
day. Within a stream a failed *attempt* is retried toward the target count, bounded by a budget that
scales with the target (2 extra for a 3-run target, 5 for a 10-run one). A shortfall is a spent budget,
not a config change — and it cannot be filled tomorrow, because tomorrow's rerun records under tomorrow's
cutoff.

You are welcome to run `run_daily` yourself whenever it is useful. As the corpus becomes the thing we fit
calibrations on, though, we'd suggest we converge on **one machine owning `runs/`** — most naturally the
one with the scheduled job. Two machines each producing live records for the same day means two copies of
the corpus that have to be reconciled by hand, and the reconciliation is easy to get subtly wrong. Until
then, if you do generate records worth keeping, mention it so they can be merged deliberately rather than
discovered later.

---

## 3. Convert older audit files into records (offline)

```bash
uv run python -m energy_oil_forecasting.cfm_coach.backfill
```

Reads the pre-coach audit JSON files at the repo root — the ones written by
`run_cfm_agent_v_5_0_interactive.py` — and converts them into `RunRecord`s. Pure format conversion: no
LLM, no network. It overwrites existing records for the same runs, so it is safe to re-run.

Everything it produces is marked `replayed_live_search`, because those runs were executed months after
their cutoff.

---

## 4. Use the harness directly (offline)

Everything below is plain Python over stored records. No LLM, no network, milliseconds.

**Check the corpus still replays exactly:**

```python
from energy_oil_forecasting.cfm_coach import ReplayEngine, RunRecordStore

records = RunRecordStore().load_all()
for report in ReplayEngine().verify_all(records, strict=True):   # raises on any drift
    print(report.describe())
```

**Ask what a different calibration would have produced:**

```python
from datetime import date
from energy_oil_forecasting.cfm_coach import CalibrationLayer, ReplayEngine
from energy_oil_forecasting.cfm_coach.schemas import CalibrationVersion

wider = CalibrationVersion(
    version="experiment",
    effective_from=date(2004, 1, 1),
    layer=CalibrationLayer(width_scale={5: 1.5, 10: 1.5, 21: 1.5}),
)
for horizon in ReplayEngine().replay(records[0], wider):
    print(horizon.horizon, horizon.point_forecast, horizon.p10_p90_width)
```

**Score a candidate through the locked gate:**

```python
from energy_oil_forecasting.cfm_coach import ComparisonPolicy, OutcomeResolver
from energy_oil_forecasting.cfm_coach.ledger import CalibrationLedger
from energy_oil_forecasting.cfm_coach.schemas import Candidate

resolution = OutcomeResolver().resolve_all(records)
verdict = ComparisonPolicy().evaluate(
    Candidate(candidate_id="wider_intervals", parent_version="v001",
              layer=CalibrationLayer(width_scale={5: 1.5, 10: 1.5, 21: 1.5})),
    CalibrationLedger().load("v001"),
    records=records, resolution=resolution,
)
print(verdict.describe())
```

The gate checks ten conditions and prints why each passed or failed. It will reject nearly everything
today, because it needs 12 resolved `live_forward` origins and we currently have none — that is the gate
working, not a bug.

---

## The two rules that matter

**1. `cfm_agent_v_5_0` is never modified.** The coach applies calibration by *injecting settings*, never
by editing the agent. Verify before and after any change you make:

```bash
cd implementations/energy_oil_forecasting/cfm_agent_v_5_0 && shasum -a 256 -c MANIFEST.sha256
```

All 43 files must report `OK`.

**2. Only `live_forward` records are fitting evidence.** Read the fitting-evidence band of the report,
not the top-line number. This is enforced in code — the gate's first condition — but it is also the
easiest thing to forget when reading a table.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `No API key was provided.` | `run_daily` needs `GEMINI_API_KEY` in the repo-root `.env`. The report and tests do not. |
| File-not-found on parquet data | You are not in the repo root. `data/yfinance` resolves relative to your working directory. |
| Langfuse `401`, LiteLLM `LoggingWorker` timeouts | Telemetry noise, not a failure. The run completed. |
| `tests/test_integration.py::test_langfuse_auth` fails | Pre-existing and unrelated to the coach. Run the repo suite with `--ignore=tests/test_integration.py`. |
| `!! FIDELITY DRIFT` in the report | The agent package, a calibration file, or a record changed. Investigate before trusting any score. |
| Report shows scores but an empty fitting-evidence band | Normal today. No `live_forward` horizon has resolved yet; the first resolves 2026-08-21. |

---

## Where things live

| Path | What |
|---|---|
| `cfm_coach/streams.py` | **the three run streams** — the only place a model or a run count is chosen |
| `cfm_coach/targets.py` | **the only place an agent package is named** — add v5.3 here, nowhere else |
| `cfm_coach/run_daily.py` | one live forecast, recorded (LLM) |
| `cfm_coach/run_daily_batch.py` | one stream's daily count, with retry toward target (LLM) |
| `cfm_coach/run_daily_all.py` | **what the scheduled job runs** — every stream, one cutoff (LLM) |
| `cfm_coach/report.py` | the scoreboard (offline) |
| `cfm_coach/backfill.py` | old audit files → records (offline) |
| `cfm_coach/replay.py` | re-derive a forecast under any calibration (offline) |
| `cfm_coach/scoring.py` | pinball, CRPS, 80% coverage |
| `cfm_coach/outcomes.py` | join horizons to realized WTI |
| `cfm_coach/policy/comparison_policy.py` | the locked statistical gate |
| `cfm_coach/runs/` | **the `v50_lite` corpus** — irreplaceable, committed to git as its backup |
| `cfm_coach/runs_v52_advanced/`, `runs_v52_lite/` | the v5.2 corpora, one per model |
| `cfm_coach/calibration*/` | versioned calibration constants, one ledger per stream |
| `cfm_coach/HANDOFF.md` | current state, findings, and open work |
| `run_cfm_agent_v_5_0_interactive.py` (repo root) | the pre-coach runner. Still works, but does **not** write records the coach can replay. |
