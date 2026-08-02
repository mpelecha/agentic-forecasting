## CFM Agent v3.3

Self-contained successor to v3.2. V3.3 preserves the complete native package structure and numerical architecture while integrating a graduated evidence policy and stronger approved-output consistency.

### Graduated evidence policy

- Tier 0, none: no qualifying evidence, so no contextual adjustment.
- Tier 1, limited: at least one eligible high-quality publisher, direct support for every cited material claim, confidence at least 0.50, sufficient novelty, and no material conflict. Python permits at most a small bounded center action and moderate uncertainty widening.
- Tier 2, corroborated: at least two eligible independent publishers, direct support for every cited material claim, confidence at least 0.65, sufficient novelty, and no material conflict. Python permits at most a moderate bounded center action and substantial uncertainty widening.
- Tier 3, strong: at least three eligible independent publishers, every cited material claim directly supported by at least two publishers, at least one primary source, confirmed material physical impact, confidence at least 0.80, likely-new evidence, and no material conflict. Python grants the strongest bounded authority available under the existing categorical action vocabulary.

Evidence classified as possibly partly reflected receives a 50% center-adjustment discount. Material conflict, reflected or indeterminate novelty, unsupported cited claims, or failure to reach Tier 1 neutralizes the proposal.

### Metadata and recording improvements

- Retains the original LLM assessment, sanitized assessment, and Python-approved assessment separately.
- Retains the original LLM action and Python-approved action separately for every horizon.
- Removes rejected citations from approved actions.
- Neutralizes rejected actions to no_change, unchanged uncertainty, and unknown persistence.
- Downgrades unsupported physical-status labels.
- Records source-level and claim-level eligibility diagnostics and rejection reasons.
- Retains the authoritative unadjusted ensemble as a control beside the final forecast.

### Preserved architecture

ARIMA, Kalman, LightGBM, equal-weight ensemble, 1,000 samples, one successful authoritative-suite execution, cutoff-safe research, optional diagnostic E2B, and Python-owned final arithmetic are unchanged.

### Validation

Run inside the complete agentic-forecasting repository:

```bash
uv run ruff check implementations/energy_oil_forecasting/cfm_agent_v_3_3
uv run python -m compileall -q implementations/energy_oil_forecasting/cfm_agent_v_3_3
uv run pytest -q implementations/energy_oil_forecasting/cfm_agent_v_3_3/tests
```

The live five-date end-to-end and Langfuse validation is the next validation stage.
