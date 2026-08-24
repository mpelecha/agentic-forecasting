---
name: action-proposal
description: >-
  Propose only exact delta-governed categorical uncertainty, persistence,
  physical-status, and novelty values, plus an integer center_action rank.
  Never invent substitute labels or a string center_action.
---

# Delta-Governed Action Proposal

Use only the exact categorical values listed below. `center_action` is the
one exception — it is an integer rank, not a category (see below).

## Exact Physical-Status Values

`physical_status` must be exactly one of:

- `normal`
- `elevated_risk`
- `partial_disruption`
- `confirmed_disruption`
- `unknown`

Do not use substitutes such as:

- `tight`
- `loose`
- `bullish`
- `bearish`
- `high`
- `low`

## Exact Incremental-Novelty Values

`incremental_novelty` must be exactly one of:

- `likely_new_relative_to_model_data`
- `possibly_partly_reflected`
- `likely_reflected_in_model_data`
- `indeterminate`

Do not use substitutes such as:

- `high`
- `medium`
- `low`
- `new`
- `old`
- `reflected`

## center_action Is an Integer Rank, Not a Category

`center_action` must be a bare JSON integer, exactly one of:

- `-2` — strongly bearish
- `-1` — mildly bearish
- `0` — no directional view beyond the median historical move
- `1` — mildly bullish
- `2` — strongly bullish

Never a string. Never `"no_change"`, `"small_up"`, `"large_down"`, or any
other named category — those belonged to an earlier version of this agent
and do not apply here. Never a price or a percentage. Python converts the
rank into a dollar adjustment using the real historical distribution of
WTI price moves — never compute that adjustment yourself.

## Exact Uncertainty Actions

- `unchanged`
- `small_wider`
- `small_narrower`
- `moderately_wider`
- `moderately_narrower`
- `substantially_wider`
- `substantially_narrower`

## Exact Persistence Profiles

- `temporary`
- `decaying`
- `persistent`
- `delayed`
- `unknown`

## Rules

- Default to `0` and `unchanged`.
- Cite only material claim IDs.
- Every non-zero `center_action` proposal must cite supporting claim IDs.
- A narrowing proposal must name the specific resolved uncertainty in
  `narrowing_uncertainty_resolved`.
- Do not calculate prices, width fractions, dollar shifts, multipliers,
  quantiles, or final numerical forecasts.
