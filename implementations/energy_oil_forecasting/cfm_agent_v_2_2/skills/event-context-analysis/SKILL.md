---
name: event-context-analysis
description: Score all news-visible WTI drivers across five categories, with a probability-weighted scenario set. Scores only, never prices.
---

# Event Context Analysis

Load `references/scoring-examples.md` via
`load_skill_resource("event-context-analysis", "references/scoring-examples.md")`
before writing any factors or scenarios.

## Scope

Cover every driver that is visible only through news and text. Five
categories:

- `geopolitical`: conflict, sanctions, shipping-lane risk, OPEC+ policy,
  diplomatic developments.
- `weather`: hurricanes, freezes, floods, and other natural events that
  disrupt production, refining, or transport.
- `operational`: strikes, accidents, outages, cyberattacks on energy
  infrastructure.
- `policy`: domestic government action — SPR releases and refills,
  drilling rules, tariffs, price caps.
- `demand_expectations`: sudden headline shocks to expected demand — a
  stimulus announcement, a lockdown, a fast-moving recession scare.
  Agency demand forecasts and their revisions (IEA, EIA, OPEC) are NOT
  in this category — they are fundamentals data, not news events.

Do not restate a driver another pillar already covers. Two banned
classes, not just the named examples:

- Structured market data: dollar index, VIX, oil curve contango,
  cross-commodity returns, CFTC positioning, and the price history
  itself.
- Supply/demand statistics and agency publications: EIA, IEA, or OPEC
  inventory levels, production and supply figures, demand forecasts and
  their revisions, and any number taken from a weekly or monthly report.

The boundary test for every candidate factor: it must be an event or a
situation, not a statistic or a forecast. Ask: "Would the quant pillar
or the physical fundamentals pillar eventually see this in its input
data?" If yes, drop the factor.

## Search strategy

1. Search for current *events* that could move oil — conflict, storms,
   strikes, sanctions, policy action — as of the cutoff date. Do not
   search for "market drivers" or "market outlook": those queries return
   market-wrap articles dominated by supply/demand framing.
2. Run 1-2 follow-up searches on whatever the first search actually
   surfaced as relevant — do not default to a fixed topic list.
3. Actively search for sources that disagree with each other, and for a
   past episode that resembles the current situation. Report both sides
   when sources conflict; do not report only the majority view.
4. Use a market-overview article only to discover that an event exists.
   Never cite one as factor evidence — verify each event from primary
   reporting about that specific event.

## Scoring rubric

`impact_score` is a signed integer from -3 to +3. The sign is price
direction: positive means upward pressure on WTI. The magnitude:

- 1: mild — could plausibly move price up to about 2%.
- 2: significant — could plausibly move price 2-5%.
- 3: extreme — regime-level, could plausibly move price more than 5%.

`confidence` is 0 to 1 and measures evidence quality behind the score,
not how likely the factor is to matter:

- 0.8-1.0: multiple independent, verifier-approved sources agree.
- 0.4-0.7: a single solid source, or sources that partly conflict.
- 0.1-0.3: thin, indirect, or rumor-level evidence.

Never output a price level, a price range, or a quantile — in any field,
including rationale text. A separate calibrated code layer maps your
scores to prices.

## Factors

1. Identify 2-4 core factors: themes durable enough to plausibly matter
   in five years or more. A core factor may score 0 when the theme is
   currently dormant but still worth tracking.
2. Identify 1-3 transitory factors: situational developments that could
   resolve, reverse, or become irrelevant within months. A transitory
   factor must have a nonzero score — if it has no impact, do not list it.
3. Tag every factor with exactly one category, and cite its evidence by
   index in `evidence_indices`.
4. Identify this factor set once. Every scenario tags the same shared set.

## Scenarios

1. Build 2-3 named, competing scenarios, each with a `stances` entry
   (bullish/bearish/neutral) for every factor, a `probability`, a signed
   `impact_score` (net price pressure if the scenario plays out), and a
   one-to-two sentence `rationale`.
2. Probabilities must sum to 1 across the scenario set.
3. Scenarios must genuinely disagree: at least two scenarios must differ
   in stance on at least two factors, not just differ in tone.
4. Exactly the tail role: at least one scenario sets `is_tail_case: true`.
   A tail case must be genuinely low-probability (never the most likely
   scenario) and genuinely high-impact (|impact_score| of 2 or 3).
5. An intensity ladder (de-escalation / sustained / escalation) is a
   valid scenario set: those scenarios may share stances, but must then
   differ in impact_score — the mainline is not the most extreme case,
   so do not give it the same score as the tail. Two scenarios that are
   identical in both stances and impact_score are one scenario written
   twice, and will be rejected.

**No scripts in this skill. Do not call `run_skill_script`.**
