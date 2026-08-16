# CFM Agent v5.0 Architecture

This file is the authoritative design reference for `cfm_agent_v_5_0`. When code, tests, or other documentation appear ambiguous, this contract governs the intended package behavior.

## Design goal

CFM Agent v5.0 combines a cutoff-safe numerical ensemble with constrained LLM research judgment. Python owns task binding, source identity, evidence eligibility, action caps, forecast arithmetic, validation, and audit serialization. The LLM may research, form atomic claims from cleaned summaries, and propose categorical actions only.

## Eleven-component pipeline

1. **Task binding** validates the target, cutoff, ordered horizons, and frequency.
2. **Cutoff-safe data service** supplies target and covariate history available at the cutoff.
3. **ARIMA** produces a probabilistic forecast.
4. **Kalman** produces a probabilistic forecast.
5. **LightGBM** produces a probabilistic forecast using lagged target and covariates.
6. **Equal-weight ensemble** combines successful model distributions without rerunning components.
7. **Deterministic diagnostics** summarize the cutoff-safe market state.
8. **Research pipeline** performs exactly four neutral searches and exposes only main-verifier-cleaned summaries plus mechanically resolved source identities.
9. **Evidence Policy** converts categorical proposals into evidence-gated, tier-capped actions.
10. **Python Forecast Engine** applies bounded center and uncertainty transformations to the ensemble.
11. **Controller and audits** validate the workflow, attach complete metadata, and isolate audit-only findings.

## Active evidence provenance

A `VerifiedSummary` is an accepted cleaned summary and its complete `associated_source_ids`. An `EvidenceClaim` cites one or more accepted summary IDs and selects supporting source IDs. Source identities are mechanically resolved from URLs; destination-page passages are not active evidence.

### Component #9 source-subset contract

For each material claim cited by a non-neutral horizon action:

- every cited supporting summary must exist and be accepted;
- the claim must select at least one supporting source ID;
- every selected source must be associated with at least one cited supporting summary;
- at least one selected source must be mechanically resolved;
- the claim may select all associated sources or a proper subset;
- unselected associated sources neither help nor hurt the claim;
- only selected, resolved sources count toward publisher thresholds and per-claim publisher requirements;
- an empty selection or any unassociated selected source invalidates the claim.

The preferred LLM behavior remains listing every associated source returned for each cited summary. The subset rule is a policy tolerance for provenance-correct source selections, not an instruction to omit sources.

## Component #9 evidence tiers

Neutral `no_change` plus `unchanged` requires no evidence. Non-neutral actions require material claims, no declared material conflict, and novelty of `likely_new_relative_to_model_data` or `possibly_partly_reflected`.

- **Tier 1, limited:** at least one selected resolved publisher and confidence at least 0.40. Caps center and uncertainty changes at small.
- **Tier 2, corroborated:** at least two selected resolved publishers and confidence at least 0.60. Caps changes at moderate.
- **Tier 3, strong:** at least three selected resolved publishers, confidence at least 0.80, at least two selected resolved publishers per qualifying claim, a Tier 1 primary publisher, physical-market evidence, and likely-new novelty. Caps changes at large or substantial.

The policy caps proposals but never upgrades them. Narrowing additionally requires a named, specifically resolved uncertainty.

## Component #10 transformation

The engine uses the original ensemble P10-P90 width:

- small center action: 10% of width;
- moderate center action: 20% of width;
- large center action: 30% of width;
- likely-new novelty multiplier: 1.00;
- possibly-partly-reflected multiplier: 0.50;
- emergency center cap: the lesser of USD 100/bbl and 100% of the latest WTI price;
- uncertainty multipliers: 1.10/0.90, 1.20/0.80, and 1.30/0.70.

Quantiles are transformed around the original P50 after the center shift. The engine checks finite non-crossing output, records any negative quantile, applies the configured explicit floor if enabled, and reproduces the ensemble exactly for a fully neutral decision.

## Execution invariants

- One successful authoritative numerical-suite execution per workflow.
- One successful four-query research execution per workflow.
- Numerical target, cutoff, horizons, and frequency exactly match the task.
- The LLM references the active research packet and exact task horizons.
- The LLM does not author final numerical forecasts.
- Audit-only findings cannot influence Components #9 or #10.
- Ensemble-locked mode reproduces the unadjusted ensemble.

## Audit-only controls

The Source Validator may inspect destination pages, publication dates, extraction quality, and forecast eligibility. The Claim-Support Verifier may assess semantic entailment from cited cleaned summaries. Both controls are serialized for diagnostics only. Their findings are deliberately excluded from the active evidence policy and forecast engine in v5.0.

## v5.0 relationship to the authoritative baseline

The authoritative baseline is `cfm_agent_v_4_4_1_revised_evidence_policy`, which already implements the Component #9 source-subset contract. v5.0 preserves that behavior and completes package isolation, deterministic test coverage, authoritative documentation, and clean release packaging. All numerical models, ensemble logic, research workflow, thresholds, action caps, transformation parameters, schemas, prompts, skills, and audit isolation remain unchanged apart from version identifiers and supporting tests/documentation.
