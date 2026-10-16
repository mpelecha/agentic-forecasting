## CFM Agent v6.0 build verification

Recorded from command output on 2026-09-16 (build step 9). Commands were run from the repository
root with `uv run`; nothing below was recorded without being executed.

### Isolation
- Isolated sibling package created from v5.2 with all identifiers renamed: yes (commit "Scaffold cfm_agent_v_6_0").
- `release.py verify-protected`: OK, 294 files unchanged against `PROTECTED_BASELINE.sha256`.
- v5.0 manifest: 43 files OK, 0 failed. v5.2 manifest: 46 files OK, 0 failed.
- `git status` under shared main (`implementations/energy_oil_forecasting/*.py`, `aieng-forecasting/`): no change.

### Configuration
- `settings.toml` SHA-256: `f600f29407e441fc598ca5998659fe60c3eb444cccdc8110bfe90a0f892aadd3`.
- `SPECIFICATION.pdf` SHA-256: `2376b07584422a784a0aefc0982cb2a932188cc02165095f0f2c2a55ba455992`.
- `CfmV60Settings()` equals the file; every field has one key and every key one field; the seventeen `[identity]` tags equal `IDENTITY_FIELDS`: passed (`tests/test_settings_file.py`, 18 tests).
- Literal-scan test over the eleven guarded modules: passed (`tests/test_no_tunable_literals.py`).

### Static quality
- `ruff format --check`: 57 files already formatted.
- `ruff check`: all checks passed.
- `python -m compileall -q`: ok, 57 Python files.
- Public imports: the 20 modules listed in `release.PUBLIC_MODULES` import.

### Tests
| Suite | Result | Duration |
|---|---|---|
| v6.0 package (`tests/`, 24 files) | 326 passed, 0 failed | 28.0 s |
| v5.2 package (unchanged) | 42 passed, 0 failed | 1.0 s |

### Numerical validation on the frozen data snapshot
- Snapshot: `implementations/energy_oil_forecasting/outputs/cfm_agent_v_6_0/data_snapshot_20260916T052728Z/` (9 parquet files, `DATA_SNAPSHOT.sha256` written; not committed).
- Cutoff 2026-09-15 (latest cutoff-safe WTI observation 2026-09-14, $101.39); full history window 2004-01-05 to 2026-09-14, 5,706 prices.
- Two fresh processes agree on the data fingerprint, suite seed, three component seeds, every component and ensemble quantile, dispositions, active weights, guard flag, and the model-suite ID: yes.
- Changing `ar_ridge_penalty` changes the model-suite ID; changing `main_temperature` leaves it unchanged: yes.
- Ordering guard never activated (`ensemble_guard_applied = false`): yes.
- Input floor activated once, on 2020-04-20; simulated floor activated 0 times in every component.

| Horizon | Ensemble P10 | P50 | P90 | Eligible returns |
|---|---|---|---|---|
| 5 | 94.77 | 101.71 | 107.50 | 5,701 |
| 10 | 91.84 | 101.73 | 110.87 | 5,696 |
| 21 | 86.88 | 101.54 | 115.97 | 5,685 |

- Centre percentiles and ladder multipliers computed at this cutoff equal the specification's illustrative tables (Sections 3.1 and 3.3) to two decimals; Sparse AR intercept +0.224% per day and residual standard deviation 12.53% per day equal Section 1.8.

### Production data
- `build_wti_multivariate_service(cache_dir=<snapshot>)` registers WTI plus all seven configured covariates; LightGBM aligned frame 4,713 rows after fill and drop.
- Service build 2.3 s; full suite (three components, 1,000 paths each, LightGBM fit) 0.4 s per execution.

### Model availability
- `release.py preflight-models`: method `models_endpoint`; `gemini-3.1-flash-lite-preview` available; `gemini-3.5-flash` available; checked 2026-09-16T05:28Z. Result saved for the live runs.

### Sensitivity
- `release.py sensitivity --cutoff 2026-09-15 --cache-dir <snapshot>`: table written to CALIBRATION.md Section 5. Removing the two April 2020 observations changes the Sparse AR intercept from +0.224% to +0.057% per day and the residual standard deviation from 12.53% to 2.63%; the ensemble P50 at h=21 moves from 101.54 to 101.03; centre percentiles and ladder multipliers change in the fourth decimal.

### Live validation
- Recorded in `LIVE_VALIDATION.md`: two live runs numerically identical, correction path exercised, effective parameters recorded, one E2B sandbox execution at the tool level, total spend 0.40 USD of the 3.00 USD ceiling. The Langfuse gate is not met (401 for the configured credentials).

### Packaging (step 10)
- `release.py verify-protected`: OK, 294 files unchanged (rerun before packaging).
- `release.py clean-caches`, then `write-manifest`: every package file hashed into `MANIFEST.sha256` (the manifest excludes itself, `__pycache__`, `.pyc`, and `.DS_Store`, in the coach's format); `verify-manifest`: OK.
- `release.py package`: archive `cfm_agent_v_6_0_<UTC>.zip` with its SHA-256 beside it under `outputs/cfm_agent_v_6_0/` (not committed; the archive name and hash are in the step 10 commit message).
- `release.py verify-archive`: archive hash matches; fresh extraction next to a copy of the shared modules; `compileall`, manifest, and the 20 public imports resolved from the extraction; the full package test suite passed from the extracted copy.
