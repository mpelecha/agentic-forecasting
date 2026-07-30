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
- `demand_expectations`: headline shocks to expected demand — stimulus or
  lockdown news, recession fear waves — before any indicator prints.

Do not restate a driver already covered by structured data: dollar index,
VIX, oil curve contango, cross-commodity returns, EIA inventory, CFTC
positioning. Never derive a factor from the price history itself. If a
candidate factor is just structured data restated in words, drop it.

## Search strategy

1. Start broad: search current market drivers as of the cutoff date.
2. Run 1-2 follow-up searches on whatever the first search actually
   surfaced as relevant — do not default to a fixed topic list.
3. Actively search for sources that disagree with each other, and for a
   past episode that resembles the current situation. Report both sides
   when sources conflict; do not report only the majority view.

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

**No scripts in this skill. Do not call `run_skill_script`.**
