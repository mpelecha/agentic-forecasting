---
name: evidence-assessment
description: Build a corroborated, claim-level evidence record from cutoff-approved web research.
---
# Evidence Assessment

## Search procedure
Run distinct searches covering:
1. the underlying event and official statements;
2. confirmed physical oil-flow or shipping effects;
3. production, inventory, or strategic-reserve responses;
4. reputable independent reporting on market reaction.

## Source standards
- Prefer official/primary sources and reputable independent reporting.
- One eligible high-quality publisher may support only a small bounded action. Independent corroboration progressively unlocks moderate and strong bounded authority.
- Treat broker, bank, trading-site, and analyst material as commentary unless it directly links to primary evidence.
- Do not count repeated articles from the same publisher as independent corroboration.
- For each source, record publisher, URL, title, date, source tier, and whether it is primary/official.

## Pillar boundary: record events, not statistics
This agent is one of three forecasting inputs. A quantitative pillar already
reads price and financial market data. A physical-fundamentals pillar already
reads supply and demand data. This agent covers what is visible through news
and text, and should not restate what the other two already carry.

Two kinds of material do not belong in a claim:
- Structured market data: dollar index, VIX, oil curve contango,
  cross-commodity returns, CFTC positioning, and the price history itself.
- Statistical levels and revisions published on a schedule: EIA, IEA, or OPEC
  inventory levels, production and supply figures, and demand forecasts or
  their revisions taken from a weekly or monthly report.

The test for a candidate claim is whether it records an event or a situation
rather than a statistic. Ask, "would the quantitative or physical pillar
eventually read this in its own input data?" If yes, do not record it as a
material claim.

This does not narrow the claim types. A production, inventory, or
strategic-reserve claim remains legitimate when its substance is an event: a
decision, an announced release, a confirmed disruption, or an outage. It is
not legitimate when its substance is a level or a revision reported in a
scheduled statistical publication. Search step 3 asks for such responses;
record the response itself, not the statistic that describes it.

Prefer primary reporting about a specific event over market-wrap or outlook
articles. A market overview may point you toward an event; it is not itself
evidence for one.

## Claim standards
- Separate confirmed facts from source interpretation and agent assessment.
- Normalize each material fact into one claim.
- Mark support as direct, partial, unsupported, or conflicting.
- Link every claim to supporting and contradicting source IDs.
- Never promote threatened, feared, or possible disruption into confirmed physical disruption.
- Preserve unresolved conflicts; do not choose the most dramatic account.
- If evidence is weak or conflicting, keep the action neutral. A directly verified single high-quality source may support only a small action when novelty and transmission are clear.


### Provenance hardening
- Execute all four searches; report the exact executed query strings, not plan labels.
- If verifier-approved factual text is empty, the search contributes no evidence.
- Never reconstruct removed claims from memory or from URLs alone.
- Never use placeholder publishers such as Various, Unknown, Multiple sources, News reports, or Google Search.
- Do not infer title, publisher, publication date, source tier, or official status from an opaque redirect URL.
- Set provenance_status to verified_from_tool only when the accepted tool output explicitly establishes the source identity fields; otherwise use inferred_by_agent or unresolved.
- Set verifier_content_status to accepted_factual_content only when accepted factual text is present.
- Copy a concise exact accepted excerpt into verified_evidence_excerpt.
- Prefer resolved direct publisher URLs. When accepted tool output explicitly establishes publisher, title, date, factual excerpt, source type, and cutoff compliance, an imperfect URL may be retained with an audit warning rather than automatically discarded.
