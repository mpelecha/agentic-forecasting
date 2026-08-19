---
name: research-planning
description: >-
  Construct exactly four neutral, cutoff-aware WTI research queries while following the package-local source-selection hierarchy.
---

## Research Planning

Load and follow source-selection before constructing exactly four neutral queries.

### Required areas
- underlying_event: the event and original official statements or records.
- physical_flows: observed oil-flow, shipping, production, refinery, or logistics effects without assuming disruption.
- supply_response: production, inventory, refinery, OPEC, or strategic-reserve responses.
- market_reaction: strong independent reporting on WTI market pricing and reaction.

### Rules
- Use exactly one distinct query for each required area.
- Include WTI or oil, the relevant event or topic, and the relevant month or date.
- Keep every query factual, neutral, and non-leading. Do not assume an outcome, confirmation, causality, direction, or materiality.
- Apply the source-selection hierarchy: seek original primary records and strong independent reporting before commentary.
- When reporting points to an official release or statement, design the query to find the original source as well.
- Call run_research_pipeline exactly once using underlying_event_query, physical_flows_query, supply_response_query, and market_reaction_query.
