---
name: geopolitical-analysis
description: Build a two-tier core/transitory geopolitical factor set and a genuinely disagreeing scenario set for WTI.
---

# Geopolitical Analysis

Load `references/factor-examples.md` via
`load_skill_resource("geopolitical-analysis", "references/factor-examples.md")`
before writing any factors or scenarios.

## Scope

Cover geopolitical drivers only: conflict, sanctions, shipping-lane risk,
OPEC+ policy decisions, and diplomatic developments. Do not restate a
factor already covered by structured market or fundamentals data — dollar
index, VIX, oil curve contango, cross-commodity returns, EIA inventory, or
CFTC positioning. If a candidate factor is just one of those restated in
words, drop it.

## Search strategy

1. Start broad: search current market drivers as of the cutoff date.
2. Run 1-2 follow-up searches on whatever the first search actually
   surfaced as relevant — do not default to a fixed topic list.
3. Actively search for sources that disagree with each other, and for a
   past episode that resembles the current situation. Report both sides
   when sources conflict; do not report only the majority view.

## Factors

1. Identify 2-5 core factors: geopolitical themes durable enough to
   plausibly matter in five years or more.
2. Identify 1-2 transitory factors: situational developments that could
   resolve, reverse, or become irrelevant within months. Each requires an
   `impact_score` (low/medium/high). Core factors must not set one.
3. Give each factor a `rationale` grounded in retrieved evidence.
4. Identify this factor set once. Every scenario tags the same shared set.

## Scenarios

1. Build 2-3 named, competing scenarios, each with a `stances` entry
   (bullish/bearish/neutral) for every factor.
2. Scenarios must genuinely disagree: at least two scenarios must differ
   in stance on at least two factors, not just differ in tone.
3. At least one scenario must set `is_tail_case: true` — a genuine
   low-probability, high-impact case, not a milder variant of the main
   narrative.
4. Give each scenario a real `price_low`/`price_high` range at the
   longest horizon. `price_high` must exceed `price_low` by a meaningful
   margin — a single point value is a modeling error, not a valid choice.

## Final forecast

The longest horizon's point forecast must fall inside the full range
spanned by every scenario's price bounds, and its 90% interval must not
be narrower than 15% of that scenario spread. A forecast that ignores its
own scenarios will be rejected and retried by the calling harness.

**No scripts in this skill. Do not call `run_skill_script`.**
