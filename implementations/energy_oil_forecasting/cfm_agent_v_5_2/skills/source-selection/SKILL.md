---
name: source-selection
description: >-
  Define the source-discovery and source-preference hierarchy used by CFM v5.2 research planning and claim building. This is behavioral guidance only.
---

## Source Selection

Use this hierarchy when designing searches and selecting associated sources for claims.

### Preference hierarchy
1. Original official or primary sources directly responsible for the fact, data, decision, statement, or physical-market record.
2. Strong independent reporting with identifiable editorial standards, especially Reuters, Bloomberg, Associated Press, Financial Times, and The Wall Street Journal.
3. Recognized specialist and institutional sources with direct domain expertise, transparent attribution, and relevant original analysis.
4. Secondary commentary only when a stronger source is unavailable, and never portray commentary as primary evidence.
5. Do not use social media, Wikipedia, anonymous blogs, generic prediction sites, content farms, or unattributed aggregators for material claims.

### Discovery rules
- Seek the original primary source when independent reporting mentions an official release, filing, statement, dataset, shipping notice, inventory report, or policy decision.
- Seek strong independent corroboration for consequential facts, particularly physical disruptions and market reactions.
- Do not write queries that assume disruption, confirmation, causality, direction, or materiality.
- Source preference never overrides the strictly-before-cutoff requirement.

### Boundary
This skill guides LLM behavior only. Python-owned source resolution, evidence eligibility, publisher counting, and evidence tiers remain authoritative.
