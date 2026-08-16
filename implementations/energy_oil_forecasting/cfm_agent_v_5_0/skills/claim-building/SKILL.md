---
name: claim-building
description: >-
  Build atomic material claims only from main-verifier-cleaned summaries in
  the active research packet. Use only the exact claim types and provenance
  fields defined by the v5.0 output contract.
---

# Cleaned-Summary Claim Building

Build atomic claims only from accepted, main-verifier-cleaned summaries in the
active research packet.

## Exact Claim Types

Every claim must use exactly one of these values for `claim_type`:

- `physical_supply`
- `shipping`
- `production_policy`
- `strategic_reserves`
- `inventory`
- `demand`
- `market_reaction`
- `other`

Do not use labels such as:

- `factual`
- `fact`
- `news`
- `geopolitical`
- `bullish`
- `bearish`

If none of the specific categories applies, use `other`.

## Evidence Rules

- Use only accepted verified-summary IDs returned by
  `run_research_pipeline`.
- For every cited verified summary, list every source ID associated with that
  summary exactly as returned.
- Each claim must contain one atomic material fact.
- Claims may identify supporting and contradicting verified summaries.

## Prohibited Evidence

Never use:

- unfiltered search text;
- claims removed by the cutoff verifier;
- destination-page passages;
- destination-page metadata;
- audit-only findings;
- model memory.

## Provenance Rules

Never invent, alter, or infer:

- cleaned summary text;
- verified-summary IDs;
- source IDs;
- URLs;
- domains;
- publishers;
- verification results.

If the cleaned summaries do not support a material claim, omit the claim.
