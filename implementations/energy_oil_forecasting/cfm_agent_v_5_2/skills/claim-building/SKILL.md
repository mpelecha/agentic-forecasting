---
name: claim-building
description: >-
  Build atomic material claims only from accepted cleaned summaries and associated sources, preferring the strongest sources under source-selection.
---

## Cleaned-Summary Claim Building

Load and follow source-selection. Build claims only from accepted, main-verifier-cleaned summaries in the active research packet.

### Exact claim types
- physical_supply
- shipping
- production_policy
- strategic_reserves
- inventory
- demand
- market_reaction
- other

### Evidence rules
- Use only accepted verified-summary IDs returned by run_research_pipeline.
- Every selected source ID must be associated with at least one cited accepted summary exactly as returned.
- Prefer the strongest associated source or provenance-correct source subset under source-selection when multiple associated sources support the same atomic fact.
- Prefer direct primary evidence, then strong independent corroboration, then recognized specialist sources.
- Each claim must contain one atomic material fact.
- Claims may identify supporting and contradicting accepted summaries and their associated sources.
- Omit a claim when accepted summaries and associated sources do not support it.

### Prohibited evidence
Never use unfiltered search text, verifier-removed claims, destination-page passages or metadata, audit-only findings, model memory, invented provenance, or a source not associated with the cited summary.

### Boundary
Source preference is behavioral guidance. Python-owned association checks, mechanical resolution, publisher counting, and evidence tiers remain authoritative.
