---
name: research-planning
description: >-
  Construct exactly four neutral, cutoff-aware WTI research queries covering
  the underlying event, physical oil flows, supply response, and independent
  reporting on the WTI market reaction.
---

# Research Planning

Construct exactly four neutral, cutoff-aware WTI research queries.

## Required Areas

1. `underlying_event`
   - Underlying event and official statements.

2. `physical_flows`
   - Confirmed physical oil-flow and shipping effects.

3. `supply_response`
   - Production, inventory, refinery, OPEC, or strategic-reserve responses.

4. `market_reaction`
   - Reputable independent reporting on the WTI market reaction.

## Rules

- Use exactly one distinct query for each required area.
- Include WTI or oil, the relevant event or topic, and the relevant month or date.
- Keep every query factual, neutral, and non-leading.
- Do not assume an outcome.
- Call `run_research_pipeline` exactly once.
- Pass the four queries through these exact arguments:
  - `underlying_event_query`
  - `physical_flows_query`
  - `supply_response_query`
  - `market_reaction_query`

