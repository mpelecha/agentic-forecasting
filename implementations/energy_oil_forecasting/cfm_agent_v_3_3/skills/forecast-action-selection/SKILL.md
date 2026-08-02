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

## Rules
- Select one action record for every requested horizon.
- Cite only normalized material claim IDs.
- A non-neutral action requires direct eligible evidence, no unresolved material conflict, a clear WTI transmission mechanism, and sufficient incremental novelty. Python assigns a graduated evidence tier and may cap or reject the proposal.
- Do not output a final price, quantile, dollar adjustment, percentage adjustment, interval multiplier, or Python-policy parameter.
- Python may reject the selected action if evidence gates fail.
- Keep horizon actions coherent with the chosen persistence profile.
