# Changelog

## CFM Agent v5.0

### Baseline

CFM Agent v5.0 is derived exclusively from `cfm_agent_v_4_4_1_revised_evidence_policy`. That baseline already implemented the Component #9 source-subset policy. v5.0 preserves that active behavior without further policy changes.

### Added

- Authoritative `ARCHITECTURE.md`.
- Deterministic full-set, subset, empty-set, unassociated-source, unresolved-source, and selected-publisher-counting tests.
- March 2 source-subset regression coverage.
- Clean v5.0 package isolation and identifiers.

### Changed

- Renamed the isolated package and version-specific identifiers from v4.4.1/v441 to v5.0/v50.
- Replaced the stale all-associated-sources-required test with tests matching the active revised policy.
- Removed the commented historical policy implementation from `policy/evidence_policy.py`.
- Updated release, installation, build-verification, and live-validation documentation.

### Preserved

- Eleven-component architecture and LLM/Python boundary.
- The revised Component #9 source-subset behavior from the authoritative baseline.
- ARIMA, Kalman, LightGBM, equal-weight ensemble, diagnostics, and 1,000 samples.
- Four-query cutoff-verified research pipeline.
- Evidence thresholds, novelty gates, action caps, and narrowing rule.
- Component #10 transformation parameters and validation.
- Audit-only isolation and ensemble-locked control.
