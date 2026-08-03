---
name: code-analysis
description: Use E2B only for bounded diagnostics and disclose its contribution.
---
# Code Analysis

- run_code is optional.
- Do not recreate ARIMA, Kalman, LightGBM, the ensemble, cutoff filtering, policy arithmetic, or final quantiles.
- Use only data supplied by the task or run_authoritative_suite.
- Use E2B only for a specific incremental diagnostic.
- Set e2b_used accurately and summarize the contribution.
- Never inspect or expose credentials or environment secrets.
