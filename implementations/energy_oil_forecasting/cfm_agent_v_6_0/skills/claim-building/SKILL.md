---
name: claim-building
description: >-
  Build atomic material claims for CFM Agent v6.0 only from accepted cleaned
  summaries, citing the passage ids that state each fact and the associated
  sources, preferring the strongest sources under source-selection.
---

## Cleaned-Summary Claim Building (CFM Agent v6.0)

Load and follow source-selection. Build claims only from accepted,
verifier-cleaned summaries in the active research packet returned by
run_research_pipeline.

### Exact claim types
- physical_supply
- shipping
- production_policy
- strategic_reserves
- inventory
- demand
- market_reaction
- other

### Passages
Each accepted summary is split into numbered passages. A passage id has the form
`SS.NN`: `SS` is the 1-based, zero-padded position of the summary in the packet
and `NN` the 1-based, zero-padded position of the passage in that summary
(for example `01.02` is the second passage of the first summary). Cite the
passage ids that state the fact alongside the summary ids and source ids. Cite a
passage only when its text states the fact; do not cite a whole summary in
place of the passage that carries the fact.

### Evidence rules
- Use only accepted verified-summary ids returned by run_research_pipeline.
- Cite, for every claim, the supporting summary ids, the supporting passage ids
  within those summaries, and the supporting source ids.
- Every selected source id must be associated with at least one cited accepted
  summary exactly as returned.
- Prefer the strongest associated source or provenance-correct source subset
  under source-selection when multiple associated sources support the same
  atomic fact.
- Prefer direct primary evidence, then strong independent corroboration, then
  recognized specialist sources.
- Each claim must contain one atomic material fact.
- Claims may identify supporting and contradicting accepted summaries, their
  passages, and their associated sources.
- Omit a claim when accepted summaries and associated sources do not support
  it.
- Market diagnostics from run_authoritative_suite are not evidence and cannot
  be cited as a claim's support.

### Prohibited evidence
Never use unfiltered search text, verifier-removed claims, destination-page
passages or metadata, audit-only findings, model memory, invented provenance,
a passage id that does not exist in the cited summary, or a source not
associated with the cited summary.

### Boundary
Source preference is behavioral guidance. Python-owned association checks,
passage checks, mechanical resolution, publisher counting, and evidence tiers
remain authoritative. The audit-only claim-support verifier later checks each
claim against its cited passages; its findings do not change the forecast.
