## CFM Agent v3.4 changes from v3.3

Merges three ideas from the v2.2 event-scoring line into v3.3's architecture.
The numerical models, graduated evidence tiers, sanitizer, and Python-owned
arithmetic are unchanged.

- Added an explicit pillar boundary to the evidence skill: structured market
  data and supply/demand statistics or agency publications are banned as
  claims, with a general boundary test rather than a list of named series.
- Added `driver_category` to every claim (geopolitical, weather, operational,
  policy, demand_expectations), orthogonal to the existing `claim_type`.
- Added a categorical `ContextScenario` contract: 2-3 competing scenarios with
  probabilities summing to 1, a direction, a magnitude, and one required
  low-probability, high-impact tail case. Scenarios state no prices.
- Added scenario well-formedness validators: unique IDs, known claim
  references, probability sum, tail-case presence and dominance, a required
  mainline case, and rejection of two scenarios sharing direction and
  magnitude. An absent scenario set is tolerated and degrades to v3.3
  behaviour.
- Added a scenario-derived uncertainty floor to the policy: when the scenario
  set places at least `scenario_disagreement_floor_mass` on the losing price
  direction, Python widens the interval to at least `moderately_wider`
  regardless of the proposed action. The floor widens only and never moves the
  center, so it is deliberately independent of the evidence tier.
- Added `scenario_disagreement_mass` and `scenario_uncertainty_floor_applied`
  to `PolicyDecision` and to prediction metadata.
- Added `calibration_features()`: a fixed-width numeric row recorded in
  prediction metadata, built from the sanitized assessment, for the future
  score-to-price calibration study.
- Added `tests/test_scenarios.py` with 16 tests covering the scenario
  contract, feature-row stability, and the uncertainty floor.

## CFM Agent v3.3 changes from v3.2

- Replaced the single binary evidence threshold with Python-assigned limited, corroborated, and strong evidence tiers.
- Added confidence thresholds of 0.50, 0.65, and 0.80.
- Added tier-specific action caps and a 50% discount for possibly partly reflected evidence.
- Permitted explicitly verified source identity metadata with imperfect URLs, with audit warnings.
- Separated LLM-proposed, sanitized, and Python-approved assessments and actions.
- Neutralized rejected actions and removed rejected citations from approved output.
- Added source-level and claim-level evidence diagnostics.
- Retained the unadjusted authoritative ensemble as a control.
- Preserved all numerical-model and one-execution architecture from v3.2.
