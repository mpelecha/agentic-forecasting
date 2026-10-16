# CFM Agent v6.0 glossary

Package-local vocabulary, mirrored from Section 13 of `SPECIFICATION.pdf`. Code, tests, and
documents use these terms with exactly these meanings.

| Term | Meaning |
|---|---|
| Return path | One simulated sequence of 21 daily simple returns produced by a component |
| Step | One WTI observation (a trading day); horizons count steps, forecast dates count pandas business days |
| Component | One of the three return models; each yields 1,000 return paths and eleven quantiles per horizon |
| Eligible return | A horizon-matched cumulative return computed from cutoff-safe, floored prices inside the history window |
| Historical return profile | Per horizon: eligible count, window bounds, median, centre percentiles, and absolute-deviation bands |
| Coverage rung | An uncertainty action expressed as the nominal share of historical moves its band contains |
| Anchor band | The historical 80% band, to which the ensemble's own P10-to-P50 half-width is equated |
| Directional guard | The rule that a centre action can only move the centre in its own direction |
| Evidence tier | The Python-determined qualification level (0 to 3) that caps both levers |
| Passage | A sentence-level slice of a verifier-cleaned summary, with identifier SS.NN |
| Accepted / rejected execution | An E2B request that passed or failed the Python screen; only accepted requests launch a sandbox |
| Neutral transformation | A literal copy of the authoritative ensemble, produced when both actions are neutral or when history is insufficient |
| Failure code | The fixed-vocabulary reason a component was excluded; part of the suite identity, unlike the verbatim message |
| Identity setting | A settings.toml key tagged [identity]; its value is bound into the model-suite ID |
| Settings file | settings.toml, the single home of every tunable value; hashed into the audit and the manifest |
| Data snapshot | A frozen, hashed copy of the yfinance cache used for release validation runs |
| Protected file | A file outside the package, selected by PROTECTED_SCOPE.txt, whose hash must be unchanged at release |
| Model-suite ID | The SHA-256 identity of the numerical suite: task, identity settings, data fingerprint, seeds, and results |
| Data fingerprint | The SHA-256 of the canonical serialization of the floored, windowed, cutoff-safe observations the suite received |

## Package-specific names

| Name | Meaning |
|---|---|
| `CfmV60Settings` | The validated settings object; bare construction returns the values of `settings.toml` |
| `IDENTITY_FIELDS` | The settings fields tagged `[identity]`, bound into the model-suite ID in this order |
| `ComponentFailure` | The exception a component raises to report a failure code with an audit-only message |
| Suite seed | The first 32 bits of the identity of the seed inputs (task binding, identity settings, covariate ids, data fingerprint) |
| Component seed | The 32-bit integer drawn from the component's spawned `SeedSequence`; seeds its generator and LightGBM's `random_state` |
| Baseline P50 | The ensemble median read after the ordering guard; the centre the transformation moves |
| Adjusted P50 | Baseline P50 plus the capped centre adjustment; the centre the uncertainty ladder scales around |
| Passage id `SS.NN` | Summary index and passage index, 1-based and zero-padded to two digits |
| Tier evaluation | The recorded pass or fail of each evidence tier for one horizon: publishers, confidence, thresholds |
| Neutral reason | Why a transformation was forced neutral: insufficient history, invalid baseline width, or both actions neutral |
