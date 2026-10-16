## CFM Agent v6.0 architecture

The authoritative contract is `SPECIFICATION.pdf`. This note describes how the package is laid out
and which component owns each decision.

### Flow

1. **Task binding** (`predictor.py`): the harness cutoff, target, horizons, and frequency are bound
   before any tool runs; `settings.toml` is loaded and hashed into the audit.
2. **Data preparation** (`history.py`): cutoff-safe WTI and the seven covariates, the history
   window, the $1.00 input floor, covariate alignment with bounded forward-fill, and the data
   fingerprint over the served observations.
3. **Diagnostics** (`diagnostics.py`): multi-window returns, realised volatility and its z-score,
   drawdowns, jump z-scores, curve state, VIX, data quality, and the historical return profile per
   horizon. Deterministic and free of random numbers.
4. **Numerical suite** (`tools/market_data.py`, `models/`): suite seed from the task binding, the
   identity settings, the covariate ids, and the data fingerprint; three component generators;
   1,000 return paths per component; compounding with the simulated floor; eleven quantiles;
   validation with fixed failure codes; weighted pooling over the active components; the ordering
   guard; the baseline P50. The suite runs once per workflow and paths never leave Python.
5. **Research** (`tools/research_pipeline.py`, `passages.py`, `source_registry.py`,
   `tools/url_safety.py`): four neutral categories, grounded search, safe resolution, strict
   before-cutoff verification, verifier-cleaned summaries split into passages with stable ids.
6. **Assessment** (`agent.py`, `prompts.py`, `skills/`): the main LLM role proposes categorical
   centre, uncertainty, novelty, and conflict findings with claims linked to summaries, passages,
   and sources; an optional single accepted E2B diagnostic (`tools/code_execution.py`).
7. **Structured-output control** (`response_control.py`): one correction-only resubmission, then
   an audited neutral fallback that reproduces the ensemble exactly.
8. **Evidence policy** (`policy/evidence_policy.py`): six per-claim validity clauses, publisher
   union, tier qualification, conflict handling, the shared cap ladder, the narrowing gate, and
   recorded rejection reasons.
9. **Transformation** (`forecast_engine/transformation.py`): historical-percentile centre step
   with the directional guard, novelty multiplier, and emergency cap; the coverage-shaped
   uncertainty ladder; the final floor; exact neutral reproduction.
10. **Audit-only oversight** (`source_validator.py`, `claim_support_verifier.py`): never reaches
    the packet, the policy, or the numbers.
11. **Audit record** (`predictor.py`, `fingerprints.py`): every v5.2 key plus the v6.0 additions
    listed in specification Section 8.

### Ownership

| Concern | Owner | Module |
|---|---|---|
| Configuration and identity fields | Python | `settings.toml`, `config.py` |
| Public API | Python | `agent.py`, `predictor.py` |
| Task and data | Python | `tools/market_data.py`, `history.py` |
| Numerical suite | Python | `models/paths.py`, `bootstrap.py`, `sparse_ar.py`, `lightgbm_return.py`, `ensemble.py` |
| Diagnostics | Python | `diagnostics.py`, `history.py` |
| Research | LLM under Python controls | `tools/research_pipeline.py`, `passages.py`, `source_registry.py`, `tools/url_safety.py`, `skills/` |
| Claims and assessment | Main LLM role under Python validation | `outputs.py`, `schemas.py`, `prompts.py`, `skills/action-proposal` |
| Correction | Correction LLM role | `response_control.py` |
| Evidence | Python | `policy/evidence_policy.py` |
| Transformation | Python | `forecast_engine/transformation.py` |
| LLM control | Python | `llm_roles.py` |
| E2B | LLM request, Python enforcement | `tools/code_execution.py` |
| Oversight | Audit-only | `source_validator.py`, `claim_support_verifier.py` |
| Release | Python | `release.py`, `PROTECTED_SCOPE.txt` |

### Coach-facing contract

Preserved without coach changes: `config.AGENT_NAME`, `config.PACKAGE_ROOT`,
`config.CfmV60Settings` (pydantic, extra forbidden, frozen, bare construction returns the
`settings.toml` values), `agent.build_cfm_agent_config(data_service=, settings=,
covariate_series_ids=, model=, search_model=)`, `agent.build_cfm_agent_predictor(config)`,
`forecast_engine.transformation.PythonForecastEngine(settings)` with engine id
`python_forecast_engine_v60`, `policy.evidence_policy.EvidencePolicy(settings)` with policy id
`graduated_evidence_policy_v60`, `outputs.CfmContextAssessmentOutput`, `schemas.ResearchPacket`,
`schemas.ModelHorizonForecast`, the settings fields ending in `_model`, `audit_enabled`,
`policy_mode`, `code_execution_enabled`, and `MANIFEST.sha256` at the package root. Registering
v6.0 in the coach is a separate, later task (DD-24).

### Explicitly out of scope

Coach registration, streams, replay support, scheduled runs, backtests, regime conditioning on by
default, staged releases, and any change to shared main, v5.0, v5.2, or the coach.
