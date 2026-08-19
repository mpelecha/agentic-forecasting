## Live validation sequence

1. Start a fresh process, import only v5.1, and run audit-off acceptance.
2. Require one successful numerical-suite execution and one successful four-query research execution.
3. Verify task binding and strict-before cutoff behavior.
4. Test a valid initial LLM assessment.
5. Test an invalid initial assessment corrected on the one allowed retry; numerical and research successful-execution counts must remain one.
6. Test two invalid assessments; require neutral actions and exact ensemble reproduction.
7. Verify both attempts, errors, retry outcome, and final disposition are serialized.
8. Repeat identical numerical tasks with fresh predictors; require identical RNG seed, component forecasts, ensemble, and model-suite ID.
9. Run audit-on and confirm audit-only findings cannot alter Components #9 or #10.
10. Save all artifacts directly under `implementations/energy_oil_forecasting/outputs`.
