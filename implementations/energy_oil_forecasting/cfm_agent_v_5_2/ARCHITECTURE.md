## CFM Agent v5.2 Architecture

This is the authoritative design contract for `cfm_agent_v_5_2`. The package is an isolated sibling of locked v5.0 and does not modify shared `main`.

### Preserved pipeline
v5.1 preserves v5.0 task binding, strict-before cutoff verification, ARIMA/Kalman/LightGBM models, equal-weight ensemble, diagnostics, four-query research, evidence thresholds and source-subset policy, bounded forecast transformations, audit-only controls, and ensemble-locked mode.

### Structured-output robustness
The initial decision-LLM response is validated against `CfmContextAssessmentOutput` and the active packet/task contract. On failure, Python sends the raw response and exact validation errors to one correction-only LLM call. The correction call has no tools and does not rerun research or the numerical suite. If corrected JSON validates, processing continues normally. If it fails, Python creates a neutral assessment for every horizon (`no_change`, `unchanged`, `unknown` persistence, no claims, zero confidence). Neutral fallback must exactly reproduce the authoritative ensemble. Both raw attempts, validation errors, retry outcome, and final disposition are serialized.

### Source-selection skill architecture
`source-selection` defines behavioral source-discovery and preference guidance. `research-planning` must use it while producing exactly four neutral queries. `claim-building` may form claims only from accepted cleaned summaries and associated source IDs, preferring the strongest provenance-correct sources. Python-owned resolution, association checks, publisher counting, and evidence tiers remain authoritative. No active pre-summary eligibility control is added in v5.1.

### Deterministic numerical suite
The package-owned authoritative suite derives a 32-bit RNG seed from the SHA-256 fingerprint of stable task and numerical-configuration inputs: target, cutoff, ordered horizons, frequency, sample count, model lags, Kalman dimension, covariate IDs, and ensemble weights. It sets Python `random` and NumPy RNG state immediately before component model execution. The seed, derivation label, and seed-input fingerprint are recorded in `ModelSuiteResult`, numerical execution audit, and the model-suite identity payload. Identical task/configuration inputs must produce identical component forecasts, ensemble forecasts, and model-suite IDs in an unchanged environment.

### Explicitly deferred
Active pre-summary source eligibility, frozen-packet testing, improved article extraction, and any cutoff-convention change are deferred beyond v5.1. Destination-page and claim-support controls remain audit-only.
