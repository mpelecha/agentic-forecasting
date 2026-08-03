---
name: forecast-action-selection
description: Select only authorized categorical forecast actions; Python owns all final arithmetic.
---
# Forecast Action Selection

## Default
no_change and unchanged uncertainty are the default.

## Authorized center actions
- no_change
- small_up
- moderate_up
- small_down
- moderate_down

## Authorized uncertainty actions
- unchanged
- moderately_wider
- substantially_wider
- moderately_narrower

## Authorized persistence profiles
- temporary
- decaying
- persistent
- delayed
- unknown

## Competing scenarios

Before selecting any action, state 2-3 competing scenarios for how the
assessed context resolves. Scenarios discipline the action: they force you to
name the alternative before committing to a direction.

- Give each scenario a `probability`. Probabilities must sum to 1.
- Give each a `direction` (up, down, neutral) and a `magnitude` (small,
  moderate, large). A neutral scenario must be small. These are categorical,
  like actions: never state a price or a percentage.
- Exactly the tail role: at least one scenario sets `is_tail_case: true`. A
  tail case must be genuinely low-probability, never the most likely
  scenario, and must be moderate or large in magnitude.
- Keep at least one non-tail mainline scenario.
- No two scenarios may share the same direction *and* magnitude. An intensity
  ladder is valid, but the rungs must differ: if two storylines resolve the
  same way with the same force, they are one scenario written twice.
- Cite the claim IDs each scenario rests on.

Python reads the scenario set to size uncertainty. When your scenarios place
real probability on both an up and a down outcome, Python widens the interval
whatever uncertainty action you propose. So an honest two-sided scenario set
and an `unchanged` uncertainty action contradict each other; choose the
uncertainty action your own scenarios imply.

## Rules
- Select one action record for every requested horizon.
- Cite only normalized material claim IDs.
- A non-neutral action requires direct eligible evidence, no unresolved material conflict, a clear WTI transmission mechanism, and sufficient incremental novelty. Python assigns a graduated evidence tier and may cap or reject the proposal.
- Do not output a final price, quantile, dollar adjustment, percentage adjustment, interval multiplier, or Python-policy parameter.
- Python may reject the selected action if evidence gates fail.
- Keep horizon actions coherent with the chosen persistence profile.
