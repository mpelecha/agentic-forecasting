## CFM Agent v3.4

> **Status: proposed, not approved.** This package changes files that *CFM
> Agent v3.3 — Prompt and Skill Work Guidelines* protects: `schemas.py`,
> `outputs.py`, `policy/constrained.py`, `predictor.py`, and `config.py`. Under
> those guidelines it belongs to the "separately approved development task"
> lane, not the prompt-work lane. Read `PROPOSAL.md` first: it states each
> change, its cost, and a suggested adoption order. v3.3 itself is untouched.

Self-contained successor to v3.3. V3.4 keeps v3.3's numerical architecture,
graduated evidence policy, and sanitizer unchanged, and merges three ideas from
the v2.2 event-scoring line: a pillar boundary, a competing-scenario contract,
and a fixed-width calibration feature row.

### What v3.4 adds

**Pillar boundary.** The evidence skill now states the boundary against the
other two forecasting pillars explicitly. Structured market data (dollar index,
VIX, contango, cross-commodity returns, CFTC positioning, price history) and
supply/demand statistics or agency publications (EIA, IEA, OPEC inventory,
production, and demand forecasts) are banned as claims. The test for a
candidate claim is whether the quant or physical pillar would eventually see it
in its own input data; if so, this agent does not record it.

**Driver categories.** Every claim carries a `driver_category` of
`geopolitical`, `weather`, `operational`, `policy`, or `demand_expectations`.
This is orthogonal to `claim_type`: `claim_type` says what kind of fact a claim
is, `driver_category` says what kind of driver produced it. A hurricane that
shuts in production is `physical_supply` and `weather`.

**Competing scenarios.** The assessment states 2-3 categorical scenarios with
probabilities summing to 1, a direction, a magnitude, and one required
low-probability, high-impact tail case. Scenarios are categorical for the same
reason actions are: the LLM never states a price. Python validates the set's
well-formedness and rejects a malformed one, but tolerates an absent one — the
agent then degrades to v3.3 behaviour rather than failing.

**Scenario-derived uncertainty floor.** Scenarios are load-bearing, not
decorative. When the assessment's own scenario set places at least
`scenario_disagreement_floor_mass` (default 0.20) of probability on the losing
price direction, Python widens the interval to at least `moderately_wider`,
whatever uncertainty action the LLM proposed. The floor is independent of the
evidence tier because it widens only and never moves the center, so it cannot
manufacture a directional view from weak evidence. The evidence gate governs
the center; the scenario set governs the spread.

**Calibration feature row.** Each prediction's metadata now carries
`calibration_features`: a fixed-width numeric row built from the *sanitized*
assessment, so rejected evidence cannot inflate a feature. Every key exists on
every run, whatever the LLM found — that is what makes it usable as regression
input, since claim text changes between origins but the slots do not. Paired
with the `unadjusted_ensemble` already recorded in metadata, it supplies both
sides of the future score-to-price study: the features, and the residual of the
actual move against the quant baseline's expectation.

### Why the calibration row matters

V3.3 moved the adjustment decision out of the LLM and into a deterministic,
auditable, versioned config. That is a real improvement. But the constants in
that config — the action width fractions, the center caps, the uncertainty
multipliers, the persistence-profile coefficients — are still judgment calls,
not fitted values. The calibration row is the input to the study that would
replace them with numbers learned from data.

### Graduated evidence policy (unchanged from v3.3)

- Tier 0, none: no qualifying evidence, so no contextual adjustment.
- Tier 1, limited: at least one eligible high-quality publisher, direct support for every cited material claim, confidence at least 0.50, sufficient novelty, and no material conflict. Python permits at most a small bounded center action and moderate uncertainty widening.
- Tier 2, corroborated: at least two eligible independent publishers, direct support for every cited material claim, confidence at least 0.65, sufficient novelty, and no material conflict. Python permits at most a moderate bounded center action and substantial uncertainty widening.
- Tier 3, strong: at least three eligible independent publishers, every cited material claim directly supported by at least two publishers, at least one primary source, confirmed material physical impact, confidence at least 0.80, likely-new evidence, and no material conflict. Python grants the strongest bounded authority available under the existing categorical action vocabulary.

Evidence classified as possibly partly reflected receives a 50% center-adjustment discount. Material conflict, reflected or indeterminate novelty, unsupported cited claims, or failure to reach Tier 1 neutralizes the proposal.

### Metadata and recording

- Retains the original LLM assessment, sanitized assessment, and Python-approved assessment separately.
- Retains the original LLM action and Python-approved action separately for every horizon.
- Removes rejected citations from approved actions.
- Neutralizes rejected actions to no_change, unchanged uncertainty, and unknown persistence.
- Downgrades unsupported physical-status labels.
- Records source-level and claim-level eligibility diagnostics and rejection reasons.
- Retains the authoritative unadjusted ensemble as a control beside the final forecast.
- Records the scenario disagreement mass and whether the uncertainty floor was applied.
- Records the fixed-width calibration feature row.

### Preserved from v3.3

ARIMA, Kalman, LightGBM, equal-weight ensemble, 1,000 samples, one successful
authoritative-suite execution, cutoff-safe research, the four graduated
evidence tiers, the deterministic sanitizer, the ensemble-locked control
policy, optional diagnostic E2B, and Python-owned final arithmetic are all
unchanged.

### Validation

Run inside the complete agentic-forecasting repository:

```bash
uv run ruff check implementations/energy_oil_forecasting/cfm_agent_v_3_4
uv run python -m compileall -q implementations/energy_oil_forecasting/cfm_agent_v_3_4
uv run pytest -q implementations/energy_oil_forecasting/cfm_agent_v_3_4/tests
```

The live multi-date end-to-end run and the calibration study are the next
validation stages.
