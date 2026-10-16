---
name: action-proposal
description: >-
  Propose only exact CFM Agent v6.0 categorical center, uncertainty,
  persistence, physical-status, and novelty values. Never invent substitute
  labels. Python turns the values into numbers.
---

# Categorical Action Proposal (CFM Agent v6.0)

Use only the exact categorical values listed below. Python maps each value to a
number from settings; you never calculate the number.

## Exact Physical-Status Values

`physical_status` must be exactly one of:

- `normal`
- `elevated_risk`
- `partial_disruption`
- `confirmed_disruption`
- `unknown`

Do not use substitutes such as `tight`, `loose`, `bullish`, `bearish`, `high`,
or `low`.

## Exact Incremental-Novelty Values

`incremental_novelty` must be exactly one of:

- `likely_new_relative_to_model_data`
- `possibly_partly_reflected`
- `likely_reflected_in_model_data`
- `indeterminate`

Do not use substitutes such as `high`, `medium`, `low`, `new`, `old`, or
`reflected`.

### What novelty does

All four classes act. Novelty never switches an action off; it scales the size
of the centre adjustment. Python multiplies the centre adjustment by a
multiplier read from settings (`[transformation.novelty_multipliers]`), whose
shipped values are 1.00 for `likely_new_relative_to_model_data`, 0.75 for
`possibly_partly_reflected`, 0.50 for `likely_reflected_in_model_data`, and
0.50 for `indeterminate`. The numerical suite is fitted on price history up to
the cutoff, so evidence that the market had already priced in before the cutoff
is partly or wholly reflected in the model data; choose the class on that
basis. Use `indeterminate` when you cannot judge.

## Exact Center Actions

- `no_change`
- `small_up`
- `small_down`
- `moderate_up`
- `moderate_down`
- `large_up`
- `large_down`

Each non-neutral centre action selects a percentile of the historical h-step
return distribution (settings `[transformation.center_percentiles]`); the
adjustment is then scaled by the novelty multiplier and limited by the emergency
cap (the smaller of a fixed dollar amount and a fraction of the latest WTI
price, both from settings). Propose the direction and size class only.

## Exact Uncertainty Actions

- `unchanged`
- `small_wider`
- `small_narrower`
- `moderately_wider`
- `moderately_narrower`
- `substantially_wider`
- `substantially_narrower`

### What the uncertainty rungs mean

Each rung names a nominal historical coverage for the forecast band. The
ensemble's P10 to P90 band is treated as the 80 percent rung. The shipped
ladder (settings `[transformation.uncertainty_coverage]`) is:

| Uncertainty action        | Nominal historical coverage |
|---------------------------|-----------------------------|
| `substantially_narrower`  | 65 percent                  |
| `moderately_narrower`     | 70 percent                  |
| `small_narrower`          | 75 percent                  |
| `unchanged`               | 80 percent                  |
| `small_wider`             | 85 percent                  |
| `moderately_wider`        | 90 percent                  |
| `substantially_wider`     | 95 percent                  |

Python converts the chosen coverage into a width ratio against the 80 percent
anchor from the historical h-step returns, so `unchanged` yields a ratio of
exactly 1.0. Widening says the band should cover more of the historical
outcomes than the ensemble band does; narrowing says less.

### The narrowing gate

Narrowing is a claim that a specific uncertainty has been resolved. A narrowing
proposal must name that resolved uncertainty in
`narrowing_uncertainty_resolved`, regardless of tier, and cite the claim ids
that show it resolved. General confidence is insufficient. Python sets the
uncertainty action of a narrowing proposal with an empty
`narrowing_uncertainty_resolved` back to `unchanged`.

### The cap ladder

Python evaluates the evidence tier of each horizon's cited claims (independent
publishers and confidence, thresholds from settings `[evidence]`) and caps the
proposal at that tier; it never upgrades a proposal.

| Tier   | Requirement (shipped)                 | Maximum centre | Maximum uncertainty                    |
|--------|---------------------------------------|----------------|----------------------------------------|
| Tier 0 | requirements not met                  | `no_change`    | `unchanged`                            |
| Tier 1 | 1 publisher, confidence 0.40          | small          | `small_wider` / `small_narrower`       |
| Tier 2 | 2 publishers, confidence 0.60         | moderate       | `moderately_wider` / `moderately_narrower` |
| Tier 3 | 2 publishers, confidence 0.80         | large          | `substantially_wider` / `substantially_narrower` |

A proposal above the qualified tier is reduced to that tier's maximum on both
levers. A material evidence conflict forces Tier 0 on the horizon. A neutral
proposal needs no evidence.

## Exact Persistence Profiles

- `temporary`
- `decaying`
- `persistent`
- `delayed`
- `unknown`

## Rules

- Default to `no_change` and `unchanged`.
- Cite only material claim ids. Every non-neutral proposal must cite supporting
  claim ids that qualify under the tier the action size requires (see the cap
  ladder). Python checks the tier and reduces an action whose evidence falls
  short.
- Market diagnostics are informational context. They cannot satisfy an
  evidence tier; an action supported only by diagnostics is rejected.
- Do not calculate prices, coverages, width ratios, dollar shifts,
  multipliers, quantiles, or final numerical forecasts.
