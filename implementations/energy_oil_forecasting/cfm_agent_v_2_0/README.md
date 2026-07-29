# cfm_agent_v_2_0

CFM Agent v2.0 preserves the three-tool and three-model architecture of v1.0
while adding auditable rich output and removing raw WTI observations from the
normal LLM prompt. Version 1.0 can remain installed alongside this package.

## What changed from v1.0

- The prompt carries series IDs, cutoff, horizons, frequency, and available
  series IDs. It does not embed target or covariate observations.
- `query_market_data` reports per-model training-data usage, configured and
  active ensemble weights, failures, and model disagreement.
- The agent response includes component forecasts, evidence, model-selection
  rationale, final rationale, ensemble adjustment, E2B disclosure, and warnings.
- `CfmRichAgentPredictor` converts the rich response to the standard
  `ContinuousForecast` payload and stores attribution in `Prediction.metadata`.
- Component forecasts in metadata are overwritten with the authoritative
  in-process `MarketDataTool.last_result`. The agent-reported copy remains under
  `agent_reported_component_models`, with `component_copy_matches_tool` showing
  whether the two agree exactly.

## Component error-field contract

The rich component output uses a status-dependent error contract:

- For `status="ok"`, `error` may be omitted, `null`, or an empty string.
- For `status="error"`, `error` must be a nonempty string describing the
  failure.

This accepts normal JSON representations of “no error” while retaining a strict
failure-disclosure requirement.

## Tools

1. `search_web`: repository web search with independent cutoff verification.
2. `run_code`: E2B sandbox for optional incremental diagnostics.
3. `query_market_data`: cutoff-safe data, ARIMA, Kalman, LightGBM, ensemble, and
   model-audit output.

## Models

- Darts AutoARIMA on all cutoff-safe target observations.
- Darts KalmanForecaster on all cutoff-safe target observations.
- Darts LightGBM quantile regression on target lags and configured cutoff-safe
  covariates.
- Weighted quantile ensemble, renormalized over successful models.

## Training-data audit

ARIMA and Kalman report exact target observation counts and target date range.
LightGBM reports target and covariate counts, common regularized business-day
span, lag settings, and `effective_training_examples_estimate`. The final
training-example count is explicitly labelled an estimate because Darts does not
expose its internal post-transformation design-matrix row count.

## Build

```python
from energy_oil_forecasting.cfm_agent_v_2_0 import (
    build_cfm_agent_config,
    build_cfm_agent_predictor,
)
from energy_oil_forecasting.data import build_wti_multivariate_service

service = build_wti_multivariate_service()
config = build_cfm_agent_config(data_service=service)
predictor = build_cfm_agent_predictor(config)
```

The config and predictor must be built in the same Python process. This lets the
predictor recover the exact `MarketDataTool` instance registered on the config
and attach its authoritative result after each agent run.

## Prediction metadata

Each `Prediction.payload` remains a normal `ContinuousForecast`. Metadata adds:

- `component_models`
- `agent_reported_component_models`
- `component_copy_matches_tool`
- `training_data`
- `configured_ensemble_weights`
- `active_ensemble_weights`
- `model_disagreement_std`
- `successful_models` and `failed_models`
- `ensemble_to_final_adjustment`
- `model_selection_rationale`
- `final_forecast_rationale`
- `verified_evidence` and `research_summary`
- `e2b_used` and `e2b_summary`
- `warnings`
- Langfuse trace identifiers supplied by the shared agent predictor

## Validation

```bash
uv run ruff check implementations/energy_oil_forecasting/cfm_agent_v_2_0
uv run python -m compileall -q implementations/energy_oil_forecasting/cfm_agent_v_2_0
uv run pytest -q implementations/energy_oil_forecasting/cfm_agent_v_2_0/tests
```

The complete Version 2.0 suite should now report `19 passed`.
