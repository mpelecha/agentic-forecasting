## Build verification

Run inside the complete agentic-forecasting repository, in the `aieng`
environment. Unlike v3.3, full Ruff and pytest execution was not deferred.

- Native v3.3 package cloned as a self-contained v3.4 package: passed.
- Legacy v3.3 package-name references in v3.4 source: none found.
- Python compileall across the package: passed.
- Ruff across the package: passed, no errors.
- Full pytest suite: 33 passed.
  - 17 inherited tests across agent contract, outputs, diagnostics, hardening,
    and policy, all passing unchanged after the merge.
  - 16 new tests in `tests/test_scenarios.py` covering the scenario contract,
    calibration-feature stability, and the scenario uncertainty floor.
- Numerical architecture comparison after version normalization: ARIMA,
  Kalman, LightGBM, ensemble, diagnostics, fingerprints, and the authoritative
  suite tool preserved exactly from v3.3.
- Graduated evidence policy, sanitizer, and ensemble-locked control policy
  preserved from v3.3; the scenario uncertainty floor is additive and widens
  only.
- Backward compatibility: a claim without `driver_category` parses and
  defaults to `geopolitical`; an assessment without `scenarios` parses and
  applies no uncertainty floor, reproducing v3.3 behaviour.

### Not yet done

- Live multi-date end-to-end run against real search and Langfuse.
- Evidence-tier distribution study: confirm the sanitizer does not reject
  nearly all real sources, and that Tier 2 and Tier 3 are reachable in
  practice.
- The calibration study that would replace the hand-set policy constants with
  fitted values.
