"""Workflow A, repeated -- N live forecasts at one cutoff, each recorded separately.

Wraps :func:`run_daily.run_once`; `run_daily.py` itself still works standalone for a
single manual run.

**Why more than one run a day.** Three same-cutoff runs on 2026-08-17 showed the LLM's
own categorical action selection varying between draws on *identical* inputs -- one run
proposed `small_up`/`moderately_wider` and cleared evidence tier `corroborated`, another
four minutes later proposed `no_change` on everything (see ``HANDOFF.md`` finding 6).
That variance directly threatens fitting a centre gain (λ): if the published overlay on
one day is a coin flip, ~40 records will not be enough to separate signal from noise.
Repeating each cutoff makes that variance directly measurable instead of stumbled into.

**How many, and for how long, is a property of the stream.** ``v50_lite`` runs three
times inside a bounded window that expires on its own (decided 2026-08-18); the two
v5.2 streams run three and ten times with no end date, because none was specified.
See :mod:`cfm_coach.streams` -- this module owns the retry loop and nothing else.

**Attempts are bounded in time as well as in count.** Added 2026-08-19, the day the
first `v52_advanced` attempt ran 66 minutes without producing a record or an error --
CPU idle, LLM socket open, nothing arriving -- and blocked every stream behind it. The
retry budget bounded failures; nothing bounded a hang, and from the outside a hang and
a slow run look identical. `_time_limit` makes the difference observable by putting a
clock on it, and a timed-out attempt is retried like any other failure.

**Attempts retry toward the target; they are not a fixed count.** The first version of this
script made exactly ``repeats_for()`` attempts and accepted however many succeeded --
2026-08-19 landed 2 of 3 because one attempt hit a Pydantic validation error (the LLM
emitted ``physical_status="disrupted"``, not one of v5.0's five allowed values) and nothing
retried it. Since the failure is call-to-call LLM variance rather than a structural problem
-- the very next attempt that same day succeeded -- a retry is the right response, not a
shrug. ``run_batch`` keeps attempting until it reaches the target count of *successes*,
bounded by the stream's retry budget so a genuinely bad day (a real outage, not sampling
noise) cannot turn into an unbounded loop of LLM calls. The budget scales with the target
rather than being a flat 2: expected failures scale with attempts, so a 10-run target that
kept a 3-run target's allowance would routinely finish short.

Usage::

    uv run python -m energy_oil_forecasting.cfm_coach.run_daily_batch            # v50_lite, cutoff = today
    uv run python -m energy_oil_forecasting.cfm_coach.run_daily_batch 2026-08-18
    uv run python -m energy_oil_forecasting.cfm_coach.run_daily_batch --stream=v52_lite
"""

from __future__ import annotations

import signal
import sys
import threading
import traceback
from contextlib import contextmanager
from datetime import date

from energy_oil_forecasting.cfm_coach.run_daily import parse_args, run_once
from energy_oil_forecasting.cfm_coach.streams import DEFAULT_STREAM, RunStream


class AttemptTimeoutError(TimeoutError):
    """One attempt exceeded the stream's wall-clock ceiling and was abandoned."""


@contextmanager
def _time_limit(seconds: float, label: str):
    """Abandon the enclosed block after `seconds`, as a normal Python exception.

    ``SIGALRM`` rather than a worker thread, because a hung attempt is hung inside
    a blocking socket read and a thread cannot be killed -- a `concurrent.futures`
    timeout would return control while the run kept going, holding its connection
    and its share of the rate limit. The signal interrupts the syscall and raises
    on the main thread, so the attempt actually stops.

    Only arms on the main thread of a Unix process. Elsewhere (a worker thread, a
    test harness) it yields unguarded rather than failing: a missing watchdog must
    not be the thing that breaks a run.
    """
    can_arm = threading.current_thread() is threading.main_thread() and hasattr(signal, "SIGALRM")
    if not can_arm or seconds <= 0:
        yield
        return

    def _fire(signum, frame):  # noqa: ANN001, ARG001 - signal handler signature
        raise AttemptTimeoutError(f"{label} exceeded {seconds:.0f}s and was abandoned")

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def repeats_for(cutoff: date, stream: RunStream = DEFAULT_STREAM) -> int:
    """How many *successful* runs this cutoff needs for `stream`."""
    return stream.repeats_for(cutoff)


def run_batch(cutoff: str, stream: RunStream = DEFAULT_STREAM) -> list[str]:
    """Run `cutoff` until the stream's target is met, or the retry budget runs out.

    Returns the run_ids that succeeded -- normally `repeats_for(cutoff, stream)` of
    them. A provider hiccup on one attempt does not cost the corpus a run: it is
    logged and retried, because each `run_once` call is independent (fresh config
    and predictor every time) and a validation failure on one call says nothing
    about the next.
    """
    target = stream.repeats_for(date.fromisoformat(cutoff))
    max_attempts = target + stream.retry_budget_for(target)
    print(
        f"Batch:       [{stream.stream_id}] target {target} successful run(s) "
        f"for cutoff {cutoff}, up to {max_attempts} attempt(s)"
    )

    run_ids: list[str] = []
    attempt = 0
    while len(run_ids) < target and attempt < max_attempts:
        attempt += 1
        print(
            f"\n{'=' * 70}\n[{stream.stream_id}] Attempt {attempt}/{max_attempts}  "
            f"({len(run_ids)}/{target} succeeded so far)  --  cutoff {cutoff}\n{'=' * 70}"
        )
        try:
            with _time_limit(stream.attempt_timeout_seconds, f"attempt {attempt} of {stream.stream_id}"):
                record = run_once(cutoff, stream=stream)
            run_ids.append(record.run_id)
        except AttemptTimeoutError as exc:
            # Deliberately louder than a normal failure: a timeout is not sampling
            # noise the next attempt will shrug off, it is a stalled call that will
            # probably stall again. Worth a human looking, not just a retry.
            print(f"\n!! [{stream.stream_id}] Attempt {attempt}/{max_attempts} TIMED OUT: {exc}")
            print("   A hang, not a validation error. Check the model and the agent loop before rerunning.")
        except Exception:  # noqa: BLE001 - one bad attempt must not sink the batch
            print(f"\n!! [{stream.stream_id}] Attempt {attempt}/{max_attempts} failed:")
            traceback.print_exc()

    print(
        f"\n{'=' * 70}\nBatch complete: [{stream.stream_id}] {len(run_ids)}/{target} "
        f"succeeded for cutoff {cutoff} in {attempt} attempt(s)\n{'=' * 70}"
    )
    if len(run_ids) < target:
        print(f"  WARNING: shortfall of {target - len(run_ids)} -- retry budget exhausted, not a config change.")
    for run_id in run_ids:
        print(f"  {run_id}")
    return run_ids


def main() -> None:
    cutoff, streams = parse_args(sys.argv[1:])
    if streams and len(streams) > 1:
        raise SystemExit("run_daily_batch runs one stream; use run_daily_all for several")
    run_batch(cutoff, streams[0] if streams else DEFAULT_STREAM)


if __name__ == "__main__":
    main()
