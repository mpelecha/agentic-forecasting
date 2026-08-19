### Changelog

#### CFM Agent v5.2

##### Fixed
- Await the package-local E2B code-interpreter coroutine before returning tool output to Google ADK.
- Preserve diagnostic code execution while directing the agent not to recompute Python-supplied task bindings.
- Add focused regression coverage for async awaiting, serializable tool output, and code-execution auditing.

##### Preserved
- All v5.1 forecasting models, ensemble behavior, research workflow, evidence policy, transformations, strict-before cutoff convention, audit-only controls, structured-output correction/fallback controls, and deterministic RNG seeding.

## Changelog

### CFM Agent v5.2

#### Added
- One correction-only LLM resubmission after invalid `CfmContextAssessmentOutput`.
- Fully audited Python-owned neutral fallback after a second invalid response.
- Package-local `source-selection` skill and coordinated updates to `research-planning` and `claim-building`.
- Deterministic task-derived RNG seed, seed audit fields, and seed participation in model-suite identity.
- Focused retry/fallback, skill-contract, and seed reproducibility tests.

#### Preserved
- Shared `main` and locked v5.0 are unchanged.
- Existing numerical models, ensemble behavior, evidence thresholds, transformation rules, strict-before cutoff convention, source-subset policy, and audit-only controls are unchanged.

#### Deferred
- Active pre-summary source eligibility, frozen-packet testing, improved article extraction, and cutoff-convention changes.
