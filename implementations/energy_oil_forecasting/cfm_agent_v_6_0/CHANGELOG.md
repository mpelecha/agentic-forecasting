## Changelog

### CFM Agent v6.0 (from v5.2)

#### Numerical foundation
- Replaced AutoARIMA, Kalman, and price-level LightGBM with three package-owned return-path
  models: Historical Bootstrap Return, Sparse AR Return, and Lightweight LightGBM Return, each
  simulating exactly 1,000 daily return paths of 21 steps (DD-03).
- Weights 0.50 / 0.25 / 0.25, renormalised over surviving components; fail closed when every
  positively weighted component fails; `require_all_components` switch (DD-04).
- Eleven quantile levels per component and ensemble, computed with `numpy.quantile` linear
  interpolation on the simulated prices (DD-05).
- $1.00 input, simulated, and final price floors; no return clipping or demeaning; the April 2020
  observations stay in every fit; a sensitivity report is produced at release (DD-06, DD-08).
- Full cutoff-safe history by default with one optional window setting; the LLM snapshot stays at
  520 rows (DD-07).
- Fixed order of operations for quantiles, the guard, the P50, and the floors; the ordering guard is
  defensive and its activation is a build failure (DD-18).
- Fixed failure-code vocabulary; the code enters the suite identity, the message stays in the audit
  (DD-17).
- One canonical JSON encoder for every identity payload; v5.2's `stable_fingerprint` with its
  `str()` fallback is removed (DD-17).
- Suite seed from the task binding, the identity settings, the covariate ids, and the data
  fingerprint; one generator per component spawned from it; process-global RNG seeding removed.

#### Diagnostics
- Multi-window cumulative returns, realised volatility, volatility z-scores, drawdowns, and jump
  z-scores with windows read from settings; curve state with a flat threshold; VIX level; data
  quality; the historical return profile per horizon.

#### Transformations
- Centre steps are historical horizon-matched return percentiles applied to the baseline P50, with
  a directional guard, novelty multiplier, and emergency cap (DD-09).
- Uncertainty rungs are nominal historical coverages; the multiplier is the ratio of the band at
  the rung's coverage to the 80% anchor band, applied to the ensemble's own distances around the
  adjusted P50 (DD-10; ADR 0001).
- Novelty multipliers 1.00 / 0.75 / 0.50 / 0.50 apply to all four classes; the v5.2 eligibility
  gate and the partly-reflected multiplier are removed (DD-11).
- Fewer than 500 eligible historical returns force a neutral transformation with the reason
  recorded; exact neutral takes precedence over the final floor and cannot conflict with it.
- Regime conditioning (`historical_conditioning`) ships as a setting, off (DD-23).

#### Research and evidence
- Verifier-cleaned summaries are split into passages by a deterministic algorithm; claims cite
  passage ids; passages are part of the summary and packet fingerprints (DD-13).
- Tier 3 requires two independent publishers and 0.80 confidence; the per-claim publisher count,
  primary-source, physical-claim, and likely-new-novelty requirements are removed (DD-12).
- Six per-claim validity clauses; failing clauses and the tier evaluation are recorded in the
  policy decision.

#### LLM configuration
- Five independently configurable roles (main, search, cutoff verifier, structured-output
  correction, claim-support verifier) with model, temperature, maximum tokens, and timeout, plus
  common generation parameters; the main role is a package-built `LiteLlm` passed as
  `AgentConfig.model`; every call site records the parameters it handed over (DD-14).
- `LITE_MODEL` alias no longer used; no import from `aieng.forecasting.models`.
- Model availability preflight before any paid run; substitution prohibited.

#### E2B
- Python-defined payload, static prohibited-operation screen, one accepted and three rejected
  requests per workflow, 60 s execution, 10,000 characters in, 50,000 out, internet off; rejected
  requests never launch a sandbox; every request audited (DD-15).

#### Configuration and audit
- Every tunable value lives in `settings.toml`; key names equal `CfmV60Settings` field names; the
  loader rejects duplicate, unknown, and missing keys; a literal-scan test guards the numerical,
  policy, and LLM modules (DD-16).
- Audit record extended with the resolved settings, floors, history window, data fingerprint and
  snapshot, component seeds and dispositions, guard flag, historical transformation inputs,
  effective LLM parameters, preflight result, and E2B requests.

#### Removed settings
`model_num_samples`, `lightgbm_lags`, `lightgbm_covariate_lags`, `kalman_dim_x`,
`tier_3_min_publishers_per_claim`, `partly_reflected_multiplier`, the three width fractions, the
six wider/narrower multipliers, `nonnegative_price_floor`, `search_verifier_model`,
`structured_output_retry_model`, `structured_output_retry_timeout_seconds`.

#### Preserved
- Task binding, strict-before cutoff, one-shot tools, one correction-only retry with neutral
  fallback, source-selection hierarchy, four neutral research categories, audit-only source
  validator and claim-support verifier, the coach-facing module layout, and all seven v5.2 test
  files (adapted; DD-21).

#### Deferred
- Coach registration, streams, replay support, scheduled runs, backtests, regime conditioning on
  by default (DD-24).
