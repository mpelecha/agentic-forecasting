# Live validation sequence

1. Start a fresh Coder process or notebook kernel.
2. Import only `cfm_agent_v_5_0` and create fresh `CfmV50Settings`, configuration, and predictor objects.
3. Run audit-off first.
4. Require exactly one successful numerical execution and one successful four-query research execution.
5. Require suite frequency, target, cutoff, and ordered horizons to equal the task.
6. Require finite, ordered P10, P50, and P90 for every horizon.
7. Confirm proper source subsets are accepted when non-empty, associated, and mechanically resolved.
8. Confirm only selected resolved sources and publishers count toward the evidence tier.
9. Require exact neutral reproduction when both actions are neutral.
10. Save the complete JSON result.
11. Create a fresh audit-enabled configuration and predictor and rerun the same task.
12. Confirm audit-only source and claim-support outputs do not enter Components #9 or #10 and cannot alter the forecast.
13. Run at least three additional audit-off forecasts using fresh predictors to assess repeated-run structural stability.
14. Preserve all raw outputs and compare structural invariants, policy decisions, and transformation audits.
