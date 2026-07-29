---
name: forecasting
description: Produce a cutoff-safe rich probabilistic forecast with auditable metadata.
---

# Forecasting Contract

1. Respect the exact target series ID, `as_of`, horizons, and frequency.
2. Raw target and covariate history are not embedded in the prompt. Retrieve
   required data through `query_market_data`.
3. Call `query_market_data` exactly once with operation
   `get_series_and_run_models`, the exact cutoff and horizons, and relevant
   registered series IDs.
4. Copy every ARIMA, Kalman, LightGBM, and ensemble value exactly into the rich
   response. Copy failed status and error text instead of inventing values.
5. Use the ensemble as the default final distribution. Quantify every departure
   as `final_p50 - ensemble_p50`.
6. Return all standard quantiles, non-crossing, with point forecast equal to p50.
7. Include model-selection rationale, final rationale, verified evidence, E2B
   disclosure, and warnings. Keep rationales concise and auditable.
8. Call `set_model_response` exactly once.
