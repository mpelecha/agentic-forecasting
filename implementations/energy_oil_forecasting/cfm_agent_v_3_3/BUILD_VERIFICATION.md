## Build verification

- Native v3.2 package cloned as a self-contained v3.3 package: passed.
- Legacy v3.2 package-name references in v3.3 source: none found.
- Python compileall across the package: passed.
- Isolated graduated-policy checks: 5 passed.
- Isolated sanitizer checks: 2 passed.
- Numerical architecture comparison after version normalization: ARIMA, Kalman, LightGBM, ensemble, diagnostics, fingerprints, and authoritative suite tool preserved exactly.
- Native repository test suite included: 17 tests across agent contract, outputs, diagnostics, hardening, and policy.
- Full pytest and Ruff execution require the repository's `aieng` environment and are intentionally deferred to repository installation.
- Five-date live end-to-end and Langfuse validation: next stage.
