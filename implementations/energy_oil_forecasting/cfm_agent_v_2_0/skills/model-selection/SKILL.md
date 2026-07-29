---
name: model-selection
description: Interpret ARIMA, Kalman, LightGBM, ensemble weights, disagreement, and failures.
---

# Model Selection

- ARIMA is the univariate autocorrelation and differencing anchor.
- Kalman is the adaptive latent-state anchor.
- LightGBM captures nonlinear target-lag and configured covariate relationships.
- The deterministic ensemble is the reproducible default distribution.

## Required procedure

1. Inspect status, point forecast, and quantiles for all components.
2. Preserve tool values exactly in `component_models`.
3. Report successful and failed models accurately.
4. Use the tool-reported model-disagreement standard deviation.
5. Use active ensemble weights, which are renormalized over successful models.
6. Discount a successful model only for an explicit data or stability reason.
7. Never claim historical superiority without backtest evidence.
8. Explain any final adjustment from the ensemble and whether the evidence affects
   the center, uncertainty, or context only.
