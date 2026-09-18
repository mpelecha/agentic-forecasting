"""The daily run streams -- what gets run, how many times, and where it lands.

A *stream* is one (agent package, LLM, cadence) triple with its own corpus on
disk. Three exist:

================  =====  =============================  ====  ======================
stream_id         agent  model                          /day  runs dir
================  =====  =============================  ====  ======================
``v50_lite``      v5.0   gemini-3.1-flash-lite-preview     3  ``runs/``
                         *agent and search only*
``v52_advanced``  v5.2   gemini-3.5-flash                  3  ``runs_v52_advanced/``
                         *every LLM call*
``v52_lite``      v5.2   gemini-3.1-flash-lite-preview    10  ``runs_v52_lite/``
                         *every LLM call*
================  =====  =============================  ====  ======================

``v50_lite`` is the one exception to "one stream, one model": v5.0 ships its search
and claim-support verifiers on ``gemini-3.5-flash``, and every record already in
``runs/`` was produced that way. See `RunStream.bind_every_llm_call`.

**Why separate directories rather than one corpus with a filter.** ``RunRecordStore``
globs its directory wholesale, and ``ComparisonPolicy`` requires the fitting corpus
to span exactly one package fingerprint. Dropping v5.2 records into ``runs/`` would
not merely add noise -- it would fail ``single_package_fingerprint`` and reject every
v5.0 candidate outright. Separate stores make the streams independent by
construction; the added ``single_agent_model`` gate is then defence in depth rather
than the only thing standing between two models and a pooled average.

**Why the model is a stream property and not a calibration field.** A calibration
version is the coach's *learned* artifact -- constants fitted against realized error.
Which LLM answered is an identity fact about the run, like the package fingerprint.
Putting it in the overlay would let a fit silently change models;
``CalibrationLedger.to_agent_settings`` now rejects an overlay that names one.

**What runs daily is a subset of what exists.** ``STREAMS`` is the registry of every
stream; ``SCHEDULED_STREAMS`` is what the weekday job walks. ``v50_lite`` was retired
from the schedule on 2026-09-08 and appears only in the former -- its corpus is still
read, fitted and published, it is simply no longer added to.

**Cost.** ``v50_lite`` was the whole schedule at first: 3 runs, ~4 min each. The two
v5.2 streams took a weekday morning to 16 runs -- roughly an hour of wall clock and
more than 5x the LLM spend, with ``v52_advanced`` on the pricier model. Retiring
``v50_lite`` brings that to 13. The v5.2 streams still have no end date, because none
was specified; set ``window`` on them the day that changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

from aieng.forecasting.models import ADVANCED_MODEL, LITE_MODEL
from energy_oil_forecasting.cfm_coach.config import PACKAGE_ROOT, CoachSettings
from energy_oil_forecasting.cfm_coach.targets import V50, V52, AgentTarget


#: Audit-only controls stay on: their findings are exactly the trust-bearing signals
#: the trust report consumes (R2). They remain barred from the forecast itself (R9).
#: These are run-level choices, not calibration, which is why they live outside the
#: overlay -- see `CalibrationLedger.to_agent_settings`.
#:
#: ``code_execution_enabled`` is deliberately *not* here: it varies by stream, so it
#: is a `RunStream` field. See `RunStream.code_execution_enabled`.
BASE_SETTINGS_OVERRIDES: dict[str, Any] = {
    "audit_enabled": True,
    "policy_mode": "constrained_actions",
}

#: Decided 2026-08-18. `v50_lite`'s bounded measurement window for whether the LLM's
#: own action selection varies enough on identical inputs to threaten fitting a
#: centre gain (HANDOFF.md finding 6). Ends itself -- outside it the stream reverts
#: to one run a day with no plist change and no date anyone must remember.
TRIPLE_RUN_FROM = date(2026, 8, 19)
TRIPLE_RUN_THROUGH = date(2026, 10, 30)

#: Runs a windowed stream makes outside its window.
OFF_WINDOW_RUNS_PER_DAY = 1

#: Retry budget as a fraction of the target, floored at `MIN_RETRY_BUDGET`. A fixed
#: 2 was right for a 3-run target and too thin for a 10-run one: the observed
#: failure mode is per-call LLM variance, so expected failures scale with attempts.
#: The bound exists so a genuine provider outage gives up instead of looping.
RETRY_BUDGET_FRACTION = 0.5
MIN_RETRY_BUDGET = 2

#: Wall-clock ceiling on one `run_once`, after which the attempt is abandoned and
#: counted as a failure so the retry budget applies to it.
#:
#: Added 2026-08-19 after the first `v52_advanced` attempt ran 66 minutes without
#: producing a record or an error -- CPU almost idle, the LLM socket open, nothing
#: arriving. The retry budget bounds *failed* attempts; nothing bounded a *hung*
#: one, so a single stalled call blocked the whole morning and starved every stream
#: behind it. A hang is indistinguishable from slowness from the outside, so the
#: only honest defence is a clock.
#:
#: 15 minutes against a ~4 minute norm: generous enough that a genuinely slow run
#: is never cut off, short enough that a stall costs one attempt rather than a day.
ATTEMPT_TIMEOUT_SECONDS = 900.0


@dataclass(frozen=True)
class RunStream:
    """One daily run stream: an agent, an LLM, a cadence, and a corpus of its own."""

    stream_id: str
    target: AgentTarget
    #: Every LLM call in this stream goes here -- the main agent, the grounded
    #: search, and every ``*_model`` settings field. See `build_config`.
    model: str
    runs_per_day: int
    runs_dirname: str
    calibration_dirname: str
    #: Prefixes ``run_id``. ``v50_lite`` keeps the bare agent id so the 14 records
    #: already on disk stay addressable by the ids they were written with.
    run_id_prefix: str
    description: str
    #: When set, `runs_per_day` applies only inside ``[from, through]`` and the
    #: stream falls back to `OFF_WINDOW_RUNS_PER_DAY` outside it.
    window: tuple[date, date] | None = None
    #: Wall-clock ceiling on one attempt. See `ATTEMPT_TIMEOUT_SECONDS`.
    attempt_timeout_seconds: float = ATTEMPT_TIMEOUT_SECONDS
    #: Whether `model` also overrides the ``*_model`` settings fields, or only the
    #: two constructor-side calls (main agent, grounded search).
    #:
    #: **False for `v50_lite`, and that is not an oversight.** v5.0 ships its two
    #: verifiers on ``gemini-3.5-flash`` while the agent runs on the lite model, and
    #: all 14 records in ``runs/`` were produced that way. Binding them to the lite
    #: model would change what the agent *is* partway through a live corpus -- new
    #: records would not be poolable with old ones, and no existing gate would catch
    #: it, because ``package_fingerprint`` hashes the manifest and the manifest does
    #: not record which model a verifier called. The v5.2 streams are new corpora, so
    #: "every call on one model" costs them nothing.
    bind_every_llm_call: bool = True
    #: Whether the agent is given the E2B-backed ``run_code`` tool at all.
    #:
    #: **Off for both v5.2 streams.** ``AuditedCodeExecutionTool.run_code`` calls
    #: ``CodeInterpreter.run_code(code=...)`` passing no timeout, and the interpreter
    #: defaults to ``code_execution_timeout_seconds=None`` and
    #: ``request_timeout_seconds=None`` with ``sandbox_create_max_attempts=12``. A
    #: sandbox that will not create, or an execution that stalls, therefore blocks the
    #: whole run with nothing to cut it off -- inside a single agent turn, which is
    #: exactly where ``v52_advanced`` was parked for 66 minutes on 2026-08-19.
    #:
    #: Cheap to give up. The tool is diagnostics-only and barred from Components #9
    #: and #10, so it cannot influence a forecast (R9), and all 11 ``v52_lite`` runs
    #: recorded ``code_execution_call_count: 0`` -- lite never once reached for it.
    #: A thinking model is far likelier to, which is the difference that matters here.
    #:
    #: Left **on** for ``v50_lite`` for the same reason its verifier models are left
    #: alone: all 14 records in ``runs/`` were produced with the tool registered, and
    #: changing the tool set changes what the agent is, partway through a live corpus.
    code_execution_enabled: bool = True

    # -- cadence --------------------------------------------------------------

    def repeats_for(self, cutoff: date) -> int:
        """How many *successful* runs this cutoff needs."""
        if self.window is None or self.window[0] <= cutoff <= self.window[1]:
            return self.runs_per_day
        return OFF_WINDOW_RUNS_PER_DAY

    @staticmethod
    def retry_budget_for(target: int) -> int:
        """Extra attempts allowed on top of `target` successes."""
        return max(MIN_RETRY_BUDGET, math.ceil(target * RETRY_BUDGET_FRACTION))

    # -- storage --------------------------------------------------------------

    @property
    def coach_settings(self) -> CoachSettings:
        """Coach settings pointed at this stream's own corpus and ledger.

        Everything else -- the task binding, the locked comparison gate, the trust
        thresholds -- is deliberately shared, so the streams differ only in what
        they run and where it lands.
        """
        return CoachSettings(
            target_agent=self.target.agent_id,
            runs_dir=PACKAGE_ROOT / self.runs_dirname,
            calibration_dir=PACKAGE_ROOT / self.calibration_dirname,
        )

    # -- the agent ------------------------------------------------------------

    @property
    def base_settings(self) -> Any:
        """Run-level agent settings, with the LLM-naming fields bound if this stream binds them."""
        settings = self.target.settings(
            **BASE_SETTINGS_OVERRIDES,
            code_execution_enabled=self.code_execution_enabled,
        )
        return self.target.bind_models(settings, self.model) if self.bind_every_llm_call else settings

    def build_config(self, **kwargs: Any) -> Any:
        """Build the agent config with both constructor-side models bound to `model`.

        The other half of the binding -- ``search_verifier_model``,
        ``claim_support_verifier_model``, and v5.2's ``structured_output_retry_model``
        -- rides in on the ``settings`` the caller passes, via `base_settings`.
        Setting only ``model=`` here would leave three of five calls on the package
        default, making a nominally single-model run a mixed-model one.
        """
        return self.target.build_config(model=self.model, search_model=self.model, **kwargs)

    def models_in_use(self, settings: Any) -> dict[str, str]:
        """Every LLM this stream will call, by knob name -- for logging and tests."""
        return {
            "agent": self.model,
            "search": self.model,
            **{name: getattr(settings, name) for name in self.target.model_settings_fields},
        }


V50_LITE = RunStream(
    stream_id="v50_lite",
    target=V50,
    model=LITE_MODEL,
    runs_per_day=3,
    runs_dirname="runs",
    calibration_dirname="calibration",
    run_id_prefix=V50.agent_id,
    description="The original corpus. Unchanged: same models, same cadence, same directory.",
    window=(TRIPLE_RUN_FROM, TRIPLE_RUN_THROUGH),
    # Leaves the two verifiers on v5.0's shipped gemini-3.5-flash, which is what
    # every record already in `runs/` was produced with. See `bind_every_llm_call`.
    bind_every_llm_call=False,
)

V52_ADVANCED = RunStream(
    stream_id="v52_advanced",
    target=V52,
    model=ADVANCED_MODEL,
    runs_per_day=3,
    runs_dirname="runs_v52_advanced",
    calibration_dirname="calibration_v52_advanced",
    run_id_prefix=f"{V52.agent_id}__advanced",
    description="v5.2 with every LLM call on the advanced model, no code execution.",
    # The stream that stranded on an untimed E2B call. See `code_execution_enabled`.
    code_execution_enabled=False,
)

V52_LITE = RunStream(
    stream_id="v52_lite",
    target=V52,
    model=LITE_MODEL,
    runs_per_day=10,
    runs_dirname="runs_v52_lite",
    calibration_dirname="calibration_v52_lite",
    run_id_prefix=f"{V52.agent_id}__lite",
    description=(
        "v5.2 with every LLM call on the lite model, no code execution; ten draws a day to measure action variance."
    ),
    # Off here too, so the two v5.2 streams differ only in model. The 11 records
    # written on 2026-08-19 had the tool available and never called it, so this
    # changes the recorded settings without changing any forecast.
    code_execution_enabled=False,
)

#: Every stream that exists, in run order. This is the *registry*: what
#: `stream_for` resolves, what the corpus page reads, what the invariant tests
#: walk. A stream stays here for as long as its corpus is worth reading, which
#: outlasts the last day it was run.
STREAMS: tuple[RunStream, ...] = (V50_LITE, V52_ADVANCED, V52_LITE)

#: What the scheduled weekday job actually runs -- a subset of `STREAMS`.
#:
#: **`v50_lite` was retired from the schedule on 2026-09-08, at the user's
#: request, effective the next run.** Its last recorded cutoff is 2026-09-08,
#: which landed a full 3/3. Retiring it is deliberately *not* expressed by
#: ending its `window`: outside a window a stream falls back to
#: `OFF_WINDOW_RUNS_PER_DAY`, which is one run a day, not none. Nor is it
#: expressed by removing it from `STREAMS`, which would break `stream_for`
#: and unpublish 54 records from the corpus page. Stopping the schedule and
#: keeping the corpus readable are separate facts, so they get separate names.
#:
#: The stream remains fully runnable by hand for a backfill or a replay::
#:
#:     uv run python -m energy_oil_forecasting.cfm_coach.run_daily_all --stream=v50_lite
SCHEDULED_STREAMS: tuple[RunStream, ...] = (V52_ADVANCED, V52_LITE)

STREAMS_BY_ID: dict[str, RunStream] = {stream.stream_id: stream for stream in STREAMS}

DEFAULT_STREAM = V50_LITE


def stream_for(stream_id: str) -> RunStream:
    if stream_id not in STREAMS_BY_ID:
        raise KeyError(f"no run stream {stream_id!r}; known streams are {sorted(STREAMS_BY_ID)}")
    return STREAMS_BY_ID[stream_id]


__all__ = [
    "ATTEMPT_TIMEOUT_SECONDS",
    "BASE_SETTINGS_OVERRIDES",
    "DEFAULT_STREAM",
    "MIN_RETRY_BUDGET",
    "OFF_WINDOW_RUNS_PER_DAY",
    "RETRY_BUDGET_FRACTION",
    "SCHEDULED_STREAMS",
    "STREAMS",
    "STREAMS_BY_ID",
    "TRIPLE_RUN_FROM",
    "TRIPLE_RUN_THROUGH",
    "V50_LITE",
    "V52_ADVANCED",
    "V52_LITE",
    "RunStream",
    "stream_for",
]
