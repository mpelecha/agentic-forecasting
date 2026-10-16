---
name: code-analysis
description: >-
  Use CFM Agent v6.0 code execution only for one planned, self-contained
  diagnostic script. Never recreate the numerical suite, the research
  pipeline, the evidence policy, or the forecast engine.
---

# Optional Code Analysis (CFM Agent v6.0)

run_code executes one Python script in an isolated sandbox with no internet
access. It exists for optional diagnostics on the numbers Python has already
produced. Most assessments do not need it.

## One accepted call

Exactly one request is accepted per assessment. Rejected requests are counted
separately, and after a small budget of rejections every further request is
refused. Plan the whole diagnostic before calling: decide every quantity you
want, write one self-contained script that computes and prints all of them,
and submit it once with a purpose that states why the diagnostic is needed.
Do not submit exploratory or incremental scripts.

## Available names

The sandbox receives a preamble that assigns three names as plain Python
values loaded from JSON. Read them; do not recompute them.

- `DIAGNOSTICS`: the market diagnostics returned by run_authoritative_suite
  (windowed returns, realised volatility, z-scores, drawdowns, jumps, curve
  state).
- `MODEL_SUITE_QUANTILES`: the ensemble quantiles per horizon from the
  numerical suite.
- `TASK_BINDING`: the task target, horizons, frequency, and cutoff date.

Nothing else is provided: no file paths, no research text, no settings, and no
credentials.

## Prohibited operations

A request that does any of the following is rejected before the sandbox is
created, and the refusal names every failed rule:

- Forecasting, simulating, or recreating the numerical suite, or computing
  alternative forecasts.
- Recomputing, verifying, or reinterpreting task-bound values: the cutoff
  date, the target series, the horizons, and the frequency are authoritative.
- Evidence decisions: scoring claims, sources, tiers, or novelty.
- Forecast transformations, adjustments, coverages, or final pricing.
- Filesystem, package, environment, skill, or runtime discovery: imports of
  or attribute access on operating-system, process, path, import-machinery,
  or network modules; calls such as open, exec, eval, compile, __import__,
  globals, locals, vars, breakpoint, or input; and names such as environ,
  getenv, __builtins__, __loader__, __spec__, or __file__.
- Network access of any kind.
- A script longer than the configured character limit, or a script that does
  not parse as Python.

## Output

Print what you need. stdout and stderr are captured and truncated at a
configured limit, with a flag when truncation occurred. Use the output as
context for your assessment only; it is not evidence and cannot support a
claim or an action.
