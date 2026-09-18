"""The scheduled weekday job -- every scheduled stream, one cutoff, one log.

    for stream in SCHEDULED_STREAMS: run_batch(cutoff, stream)

`SCHEDULED_STREAMS`, not `STREAMS`: `v50_lite` was retired from the schedule on
2026-09-08 and is no longer run daily, while its corpus stays readable and it
stays runnable by hand via ``--stream=v50_lite``. See ``streams.py``.

One process rather than one launchd job per stream, for two reasons. The streams
share a rate-limited LLM proxy and a single cached parquet, so running them
concurrently buys less wall clock than it looks like it should while making a
429 storm easy to trigger. And a single log read top to bottom answers "what did
this morning actually produce" without correlating three files by timestamp.

**A failed stream does not stop the ones after it.** `run_batch` already absorbs a
failed *attempt*; this absorbs a failed *stream* -- an import error, an exhausted
quota, a corpus directory that will not create. Losing v5.2's records to a v5.0
problem would be the schedule punishing the wrong corpus, and a day's records
cannot be recovered later: re-running at a past cutoff searches today's web.

**Runtime.** Thirteen runs at roughly four minutes each is about fifty minutes,
serial (sixteen until `v50_lite` left the schedule). That is the intended cost --
see ``streams.py`` -- but it means an 08:00 fire is still working close to 09:00,
and a machine asleep at 08:00 runs the whole stretch whenever it wakes. Note that
sleep does not merely delay the job: `run_daily_batch._time_limit` arms
``SIGALRM``, whose timer does not advance while the machine is asleep, so the
per-attempt ceiling only counts awake time. On 2026-09-08 one attempt spanned
7h40m of wall clock across ten sleep cycles without the 15-minute cap firing.

Usage::

    uv run python -m energy_oil_forecasting.cfm_coach.run_daily_all              # cutoff = today
    uv run python -m energy_oil_forecasting.cfm_coach.run_daily_all 2026-08-20
    uv run python -m energy_oil_forecasting.cfm_coach.run_daily_all --stream=v52_lite   # just one
    uv run python -m energy_oil_forecasting.cfm_coach.run_daily_all --stream=v52_advanced,v52_lite

``--stream`` is repeatable and comma-separated, so a morning can be completed for a
subset -- the case that motivated it was v5.0 having already recorded its runs for a
cutoff while the v5.2 streams still needed theirs.
"""

from __future__ import annotations

import sys
import traceback
from datetime import date

from energy_oil_forecasting.cfm_coach.run_daily import parse_args
from energy_oil_forecasting.cfm_coach.run_daily_batch import run_batch
from energy_oil_forecasting.cfm_coach.streams import SCHEDULED_STREAMS, RunStream


def run_all(cutoff: str, streams: tuple[RunStream, ...] = SCHEDULED_STREAMS) -> dict[str, list[str]]:
    """Run every stream at `cutoff`. Returns stream_id -> run_ids that succeeded."""
    planned = {stream.stream_id: stream.repeats_for(date.fromisoformat(cutoff)) for stream in streams}
    print(f"\n{'#' * 70}\nCFM Coach daily -- cutoff {cutoff}\n{'#' * 70}")
    for stream in streams:
        print(
            f"  {stream.stream_id:<14} {planned[stream.stream_id]:>2} run(s)  {stream.model}  -> {stream.runs_dirname}/"
        )
    print(f"  {'total':<14} {sum(planned.values()):>2} run(s)\n")

    results: dict[str, list[str]] = {}
    for stream in streams:
        print(f"\n{'#' * 70}\n# stream {stream.stream_id}\n{'#' * 70}")
        try:
            results[stream.stream_id] = run_batch(cutoff, stream)
        except Exception:  # noqa: BLE001 - one stream's failure must not cost the others their day
            print(f"\n!! stream {stream.stream_id} failed outright:")
            traceback.print_exc()
            results[stream.stream_id] = []

    print(f"\n{'#' * 70}\nDaily complete -- cutoff {cutoff}\n{'#' * 70}")
    shortfall = False
    for stream in streams:
        got, want = len(results[stream.stream_id]), planned[stream.stream_id]
        flag = "" if got == want else "   << SHORT"
        shortfall = shortfall or got != want
        print(f"  {stream.stream_id:<14} {got}/{want}{flag}")
    if shortfall:
        print("\n  A shortfall is a spent retry budget, not a config change. Investigate before rerunning:")
        print("  a rerun tomorrow records under tomorrow's cutoff and does not fill today's gap.")
    return results


def main() -> None:
    cutoff, streams = parse_args(sys.argv[1:])
    run_all(cutoff, streams or SCHEDULED_STREAMS)


if __name__ == "__main__":
    main()
