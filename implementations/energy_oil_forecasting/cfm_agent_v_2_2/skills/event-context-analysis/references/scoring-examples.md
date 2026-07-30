# Event Scoring Examples

These are worked examples of the `WtiEventFactor` and `WtiEventScenario`
shape. They illustrate structure, scoring, and tone only. Do not copy the
numbers or claims into a real scoring run — always ground a real factor
or scenario in evidence retrieved for that run's own cutoff date.

---

## Example 1: A Gulf hurricane (weather, transitory)

A named storm heading for Gulf of Mexico production is a **weather**
factor and almost always **transitory**: large potential impact now, but
it resolves within weeks. It is not in any structured data series until
after the fact — exactly the kind of driver this agent exists to score.

```json
{
  "name": "Gulf hurricane approaching production areas",
  "category": "weather",
  "tier": "transitory",
  "rationale": "A major storm is forecast to cross key offshore production and refining areas within days; operators have begun evacuating platforms.",
  "impact_score": 2,
  "confidence": 0.8,
  "evidence_indices": [0]
}
```

Score logic: shut-in production supports price (+), scale is significant
but not regime-level (2), and multiple independent sources confirm the
evacuation reports (0.8).

---

## Example 2: OPEC+ production policy (geopolitical, core, currently dormant)

The cartel's supply stance is a durable **core** theme. In a quiet
period, it can carry a score of 0 — still listed, because scenarios need
a stance on it, but currently exerting no directional pressure.

```json
{
  "name": "OPEC+ cohesion and spare capacity",
  "category": "geopolitical",
  "tier": "core",
  "rationale": "No meeting is imminent and current quotas are holding; the theme is dormant but remains the dominant supply-side lever.",
  "impact_score": 0,
  "confidence": 0.6,
  "evidence_indices": []
}
```

Note: a transitory factor must never score 0. Only core themes may be
dormant.

---

## Example 3: China stimulus headlines (demand_expectations, transitory)

Stimulus or lockdown headlines move expected demand before any indicator
prints. The *headline shock* belongs to this agent; the *indicator* (PMI,
imports) is structured macro data and does not.

```json
{
  "name": "China stimulus headlines lifting demand expectations",
  "category": "demand_expectations",
  "tier": "transitory",
  "rationale": "State media signaled a larger-than-expected fiscal package; commodity demand expectations rose across markets, ahead of any data print.",
  "impact_score": 1,
  "confidence": 0.5,
  "evidence_indices": [1]
}
```

---

## Example 4: A full scenario set

Three scenarios over the factors above. Probabilities sum to 1. The
first two differ in stance on two factors (genuine disagreement). The
third is the tail case: least likely, high impact.

```json
[
  {
    "name": "Storm weakens, status quo holds",
    "stances": {"Gulf hurricane approaching production areas": "neutral", "OPEC+ cohesion and spare capacity": "neutral", "China stimulus headlines lifting demand expectations": "bullish"},
    "probability": 0.5,
    "impact_score": 0,
    "is_tail_case": false,
    "rationale": "The storm misses key infrastructure; mild demand tailwind persists."
  },
  {
    "name": "Direct hit with extended shut-ins",
    "stances": {"Gulf hurricane approaching production areas": "bullish", "OPEC+ cohesion and spare capacity": "neutral", "China stimulus headlines lifting demand expectations": "neutral"},
    "probability": 0.35,
    "impact_score": 2,
    "is_tail_case": false,
    "rationale": "The storm crosses production areas; shut-ins and refinery outages last weeks."
  },
  {
    "name": "Compound shock: hurricane plus OPEC+ discipline break",
    "stances": {"Gulf hurricane approaching production areas": "bullish", "OPEC+ cohesion and spare capacity": "bearish", "China stimulus headlines lifting demand expectations": "neutral"},
    "probability": 0.15,
    "impact_score": -2,
    "is_tail_case": true,
    "rationale": "Storm damage coincides with a quota dispute that opens the taps; the supply surge dominates once outages clear."
  }
]
```

Note what the tail case is: a genuinely different storyline, not a
stronger version of the mainline. Its probability (0.15) is the lowest
in the set, and its |impact_score| is 2 — both are hard requirements.
