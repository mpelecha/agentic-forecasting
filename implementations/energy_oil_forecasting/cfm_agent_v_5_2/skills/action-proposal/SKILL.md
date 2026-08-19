---
name: action-proposal
description: >-
  Propose only exact v5.1 categorical center, uncertainty, persistence,
  physical-status, and novelty values. Never invent substitute labels.
---

# Categorical Action Proposal

Use only the exact categorical values listed below.

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

## Exact Center Actions

- `no_change`
- `small_up`
- `small_down`
- `moderate_up`
- `moderate_down`
- `large_up`
- `large_down`

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

- Default to `no_change` and `unchanged`.
- Cite only material claim IDs.
- Every non-neutral proposal must cite supporting claim IDs.
- A narrowing proposal must name the specific resolved uncertainty in
  `narrowing_uncertainty_resolved`.
- Do not calculate prices, width fractions, dollar shifts, multipliers,
  quantiles, or final numerical forecasts.
