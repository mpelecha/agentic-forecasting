# cfm_agent_v_2_2

CFM Agent v2.2 is an event-context scorer, not a forecaster. It reads
news-visible drivers of WTI and expresses them as bounded scores. It
never states a price, a price range, or a quantile.

## The division of labor

- **Job 1 — assess context (this package).** The LLM searches verified,
  cutoff-safe news and sorts it into scored factors and
  probability-weighted scenarios. Interpretation is what an LLM is
  suited for.
- **Job 2 — turn context into a number (not this package).** A
  calibration layer — plain code, fit against quant-baseline residuals,
  versioned and backtested — maps scores to price effects. It consumes
  `WtiEventScoreOutput.calibration_features()`.

## What changed from v2.1

- Scope widened from geopolitics-only to all news-visible drivers, in
  five tagged categories: `geopolitical`, `weather`, `operational`,
  `policy`, `demand_expectations`.
- The output contract lost its `forecasts` block entirely. Factors carry
  a signed `impact_score` (-3..+3) and a `confidence` (0..1); scenarios
  carry a `probability` and a conditional `impact_score`. No price
  fields exist anywhere in the schema.
- The package ships `CfmEventScorer` instead of an `AgentPredictor`
  wiring: the output is not a forecast, so it cannot convert to harness
  `Prediction` objects. The scorer drives the same ADK machinery
  (verified search, skill toolset, Langfuse tracing) and owns a bounded
  validation-retry loop.

## Hard validators (rejected and retried, not silently accepted)

- 2-4 core factors (score 0 allowed: dormant theme), 1-3 transitory
  factors (score 0 forbidden).
- Scenario probabilities sum to 1 (±0.02).
- At least one tail case and one non-tail scenario; a tail case must
  never carry the highest probability and must have |impact_score| ≥ 2.
- Every scenario's stances cover exactly the shared factor names.
- At least two scenarios differ in stance on at least two factors.
- Factor evidence indices must point into `verified_evidence`.

## Build and run

```python
from datetime import datetime

from energy_oil_forecasting.cfm_agent_v_2_2 import CfmEventScorer
from energy_oil_forecasting.data import build_wti_service

service = build_wti_service()
scorer = CfmEventScorer()
result = scorer.score(service.context(datetime(2026, 3, 2)))

result.output.factors          # scored factors, by category and tier
result.output.scenarios        # probability-weighted scenario set
result.calibration_row()       # flat row for the calibration dataset
```

## Validation

```bash
uv run ruff check implementations/energy_oil_forecasting/cfm_agent_v_2_2
uv run python -m compileall -q implementations/energy_oil_forecasting/cfm_agent_v_2_2
uv run pytest -q implementations/energy_oil_forecasting/cfm_agent_v_2_2/tests
```
