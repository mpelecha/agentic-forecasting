"""Tests for `run_daily_batch`: the retry-toward-target loop, not the LLM call itself.

`run_once` is patched out everywhere here -- these tests are about the retry policy,
not about producing a real forecast, and must not touch the network or an API key.
"""

from __future__ import annotations

import time
from dataclasses import replace
from datetime import date

import pytest
from energy_oil_forecasting.cfm_coach import run_daily_batch as batch
from energy_oil_forecasting.cfm_coach.streams import (
    OFF_WINDOW_RUNS_PER_DAY,
    V50_LITE,
    V52_ADVANCED,
    V52_LITE,
)


IN_WINDOW = "2026-09-01"
OUTSIDE_WINDOW = "2026-11-03"


class _FakeRecord:
    def __init__(self, run_id: str):
        self.run_id = run_id


def _fake(outcomes: list[bool], calls: list[bool]):
    """Build a `run_once` stand-in that succeeds or raises per `outcomes`, then always succeeds."""
    it = iter(outcomes)

    def fake_run_once(cutoff: str, *, stream=None):  # noqa: ARG001
        ok = next(it, True)
        calls.append(ok)
        if not ok:
            raise ValueError("simulated validation failure")
        return _FakeRecord(f"run_{len(calls)}")

    return fake_run_once


# -- cadence ------------------------------------------------------------------


def test_v50_window_still_governs_the_original_stream():
    """The bounded measurement window decided 2026-08-18 is unchanged by the refactor."""
    assert batch.repeats_for(date.fromisoformat(IN_WINDOW), V50_LITE) == 3
    assert batch.repeats_for(V50_LITE.window[0], V50_LITE) == 3
    assert batch.repeats_for(V50_LITE.window[1], V50_LITE) == 3
    assert batch.repeats_for(date.fromisoformat(OUTSIDE_WINDOW), V50_LITE) == OFF_WINDOW_RUNS_PER_DAY


def test_repeats_for_defaults_to_the_v50_stream():
    assert batch.repeats_for(date.fromisoformat(IN_WINDOW)) == V50_LITE.runs_per_day


@pytest.mark.parametrize(("stream", "expected"), [(V52_ADVANCED, 3), (V52_LITE, 10)])
def test_v52_streams_run_their_count_on_every_date(stream, expected):
    """No window was specified for the v5.2 streams, so the count must not expire."""
    assert stream.window is None
    for cutoff in (IN_WINDOW, OUTSIDE_WINDOW, "2030-01-04"):
        assert stream.repeats_for(date.fromisoformat(cutoff)) == expected


def test_retry_budget_scales_with_the_target():
    """A 10-run target that kept a 3-run target's flat allowance would finish short."""
    assert V50_LITE.retry_budget_for(3) == 2
    assert V52_ADVANCED.retry_budget_for(3) == 2
    assert V52_LITE.retry_budget_for(10) == 5
    assert V50_LITE.retry_budget_for(1) >= 1  # a lone run must still get a retry


# -- the retry loop -----------------------------------------------------------


def test_one_failed_attempt_is_retried_to_reach_the_target(monkeypatch):
    """The 2026-08-19 case: attempt 2 fails, but the target is still 3 successes, not 2."""
    calls: list[bool] = []
    monkeypatch.setattr(batch, "run_once", _fake([True, False, True, True], calls))
    run_ids = batch.run_batch(IN_WINDOW, V50_LITE)

    assert len(run_ids) == 3
    assert len(calls) == 4  # 3 successes + 1 retried failure, not 3 flat attempts


def test_retry_budget_is_bounded_on_a_genuinely_bad_day(monkeypatch):
    """A day where every attempt fails must give up, not retry forever."""
    call_count = 0

    def always_fails(cutoff: str, *, stream=None):  # noqa: ARG001
        nonlocal call_count
        call_count += 1
        raise RuntimeError("simulated provider outage")

    monkeypatch.setattr(batch, "run_once", always_fails)
    run_ids = batch.run_batch(IN_WINDOW, V50_LITE)

    target = V50_LITE.repeats_for(date.fromisoformat(IN_WINDOW))
    assert run_ids == []
    assert call_count == target + V50_LITE.retry_budget_for(target)


def test_the_ten_run_stream_is_also_bounded(monkeypatch):
    """The largest target must still terminate on a total outage."""
    call_count = 0

    def always_fails(cutoff: str, *, stream=None):  # noqa: ARG001
        nonlocal call_count
        call_count += 1
        raise RuntimeError("simulated provider outage")

    monkeypatch.setattr(batch, "run_once", always_fails)
    assert batch.run_batch(IN_WINDOW, V52_LITE) == []
    assert call_count == 10 + V52_LITE.retry_budget_for(10)


@pytest.mark.parametrize("stream", [V50_LITE, V52_ADVANCED, V52_LITE])
def test_no_wasted_attempts_when_every_call_succeeds(monkeypatch, stream):
    calls: list[bool] = []
    monkeypatch.setattr(batch, "run_once", _fake([], calls))
    run_ids = batch.run_batch(IN_WINDOW, stream)

    target = stream.repeats_for(date.fromisoformat(IN_WINDOW))
    assert len(run_ids) == target
    assert len(calls) == target  # no attempts spent once the target is met


def test_the_batch_runs_the_stream_it_was_given(monkeypatch):
    """A stream argument that never reaches `run_once` would silently run the default."""
    seen: list[str] = []

    def record_stream(cutoff: str, *, stream=None):  # noqa: ARG001
        seen.append(stream.stream_id)
        return _FakeRecord(f"run_{len(seen)}")

    monkeypatch.setattr(batch, "run_once", record_stream)
    batch.run_batch(IN_WINDOW, V52_ADVANCED)

    assert seen == ["v52_advanced"] * 3


def test_outside_the_window_a_single_failure_still_retries_once(monkeypatch):
    """Target is 1 outside the window, but a failed attempt must still be retried."""
    calls: list[bool] = []
    monkeypatch.setattr(batch, "run_once", _fake([False, True], calls))
    run_ids = batch.run_batch(OUTSIDE_WINDOW, V50_LITE)

    assert len(run_ids) == 1
    assert len(calls) == 2


@pytest.mark.parametrize("cutoff", [IN_WINDOW, OUTSIDE_WINDOW])
def test_run_ids_are_returned_in_the_order_they_succeeded(monkeypatch, cutoff):
    calls: list[bool] = []
    monkeypatch.setattr(batch, "run_once", _fake([], calls))
    run_ids = batch.run_batch(cutoff, V50_LITE)

    assert run_ids == [f"run_{i}" for i in range(1, len(run_ids) + 1)]


# -- the watchdog -------------------------------------------------------------


def test_a_hung_attempt_is_abandoned_and_retried(monkeypatch):
    """A hang must cost one attempt, not the whole morning.

    The 2026-08-19 case: `v52_advanced` attempt 1 ran 66 minutes with an open socket
    and no output, starving every stream queued behind it.
    """
    calls: list[str] = []

    def hangs_once(cutoff: str, *, stream=None):  # noqa: ARG001
        calls.append("call")
        if len(calls) == 1:
            time.sleep(5)  # longer than the patched ceiling
        return _FakeRecord(f"run_{len(calls)}")

    slow = replace(V50_LITE, runs_per_day=1, window=None, attempt_timeout_seconds=0.25)
    monkeypatch.setattr(batch, "run_once", hangs_once)
    run_ids = batch.run_batch(IN_WINDOW, slow)

    assert run_ids == ["run_2"], "the hung attempt should be abandoned and the next one counted"
    assert len(calls) == 2


def test_the_watchdog_does_not_fire_on_a_normal_run(monkeypatch):
    calls: list[bool] = []
    monkeypatch.setattr(batch, "run_once", _fake([], calls))
    assert len(batch.run_batch(IN_WINDOW, V50_LITE)) == 3


def test_every_stream_carries_a_finite_attempt_ceiling():
    for stream in (V50_LITE, V52_ADVANCED, V52_LITE):
        assert 0 < stream.attempt_timeout_seconds < 3600
