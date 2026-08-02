---
name: market-context-assessment
description: Interpret eligible evidence alongside authoritative numerical diagnostics without inventing prices.
---
# Market Context Assessment

- Treat run_authoritative_suite model results and diagnostics as authoritative and immutable for the run.
- Do not recalculate ARIMA, Kalman, LightGBM, the ensemble, or final quantiles.
- Do not claim that an event is "already priced." Use only these narrower classifications:
  - likely_new_relative_to_model_data
  - possibly_partly_reflected
  - likely_reflected_in_model_data
  - indeterminate
- Use Python-reported latest-observation timing when discussing novelty.
- Distinguish normal conditions, elevated risk, partial disruption, confirmed disruption, and unknown status.
- Assess the transmission channel to WTI, expected direction, persistence, and uncertainty separately.
- Dramatic language is not evidence of magnitude.
- If timing, support, transmission, or novelty is indeterminate, prefer no_change. Evidence that is possibly partly reflected may justify a discounted adjustment; Python applies the discount.
