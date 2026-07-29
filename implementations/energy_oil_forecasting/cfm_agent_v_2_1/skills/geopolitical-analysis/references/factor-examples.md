# Geopolitical Factor and Scenario Examples

These are worked examples of the `WtiGeoFactor` and `WtiGeoScenario` shape.
They illustrate structure and tone only. Do not copy the numbers or
claims into a real forecast — always ground a real factor or scenario in
evidence retrieved for that forecast's own cutoff date.

---

## Example 1: A past OPEC+ production decision (core factor)

A production-policy decision is usually a **core** factor: the cartel's
supply stance is a durable, always-relevant theme, even though any single
meeting's outcome is itself short-lived news.

```json
{
  "name": "OPEC+ production policy",
  "tier": "core",
  "rationale": "OPEC+ controls a large share of spare capacity, so its stated production path is a persistent anchor on the supply side of every WTI forecast, independent of any single meeting's outcome."
}
```

A scenario tags this factor with a stance, not a restatement of the
decision itself:

```json
{
  "name": "Cohesion holds",
  "stances": {"OPEC+ production policy": "bullish", "...": "..."},
  "price_low": 68.0,
  "price_high": 76.0,
  "is_tail_case": false
}
```

---

## Example 2: A past shipping-lane incident (transitory factor)

A specific tanker seizure, strait closure, or attack is usually a
**transitory** factor: it carries a large, uncertain price effect right
now, but it does not durably define the market once it resolves.

```json
{
  "name": "Strait of Hormuz transit disruption",
  "tier": "transitory",
  "rationale": "A reported vessel seizure near the Strait raised insurance costs and briefly cut tanker throughput; this factor could resolve within weeks or persist for months depending on the response.",
  "impact_score": "high"
}
```

Note the required `impact_score` — core factors must never set one, and
transitory factors must always set one.

---

## Example 3: A full scenario set (illustrating genuine disagreement)

Two scenarios must differ on at least two factors to pass the
disagreement check. A weak pair like this does **not** pass, because
both scenarios take the same stance on `OPEC+ production policy`:

```json
[
  {"name": "A", "stances": {"OPEC+ production policy": "bullish", "Strait disruption": "bullish"}},
  {"name": "B", "stances": {"OPEC+ production policy": "bullish", "Strait disruption": "neutral"}}
]
```

A strong pair differs on both factors, and is a genuine scenario split:

```json
[
  {"name": "Disruption persists", "stances": {"OPEC+ production policy": "neutral", "Strait disruption": "bullish"}, "price_low": 82.0, "price_high": 95.0, "is_tail_case": false},
  {"name": "Disruption resolves, supply resumes", "stances": {"OPEC+ production policy": "bearish", "Strait disruption": "bearish"}, "price_low": 66.0, "price_high": 74.0, "is_tail_case": false},
  {"name": "Full regional escalation", "stances": {"OPEC+ production policy": "neutral", "Strait disruption": "bullish"}, "price_low": 100.0, "price_high": 130.0, "is_tail_case": true}
]
```

The third scenario is the required tail case: low-probability, high-impact,
and a genuinely different story from the first scenario, not a milder
version of it.
