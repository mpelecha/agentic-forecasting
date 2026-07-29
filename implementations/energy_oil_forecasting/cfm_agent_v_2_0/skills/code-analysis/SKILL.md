---
name: code-analysis
description: Use E2B for bounded diagnostics and disclose its contribution.
---

# Code Analysis

1. `run_code` is optional.
2. Do not recreate ARIMA, Kalman, LightGBM, the ensemble, cutoff filtering, or
   standard quantiles in E2B.
3. Use only data supplied by the task or `query_market_data`.
4. Use E2B for a specific incremental diagnostic, with deterministic code and
   fixed random seeds.
5. Set `e2b_used` accurately and summarize the material contribution. If E2B was
   not used, set `e2b_used` false and leave `e2b_summary` empty.
6. Never inspect or expose credentials or environment secrets.
