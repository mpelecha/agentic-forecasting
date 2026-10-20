# wti_scenario_memory_agent_dategated

The WTI scenario-memory agent (`wti_scenario_memory_agent`) with the publication-date
gate on its news search. Everything else is unchanged: prompt, model, AutoARIMA anchor,
scenario math, and memory.

## What changes

The agent's shared `search_web` tool is replaced by the gated one from
`cfm_agent_v_5_2_dategated/search_web.py`. Each search attempt:

1. dates every page the search was grounded on (URL date or page metadata; undated,
   live, and unresolvable pages fail);
2. keeps only the sentences that Google's grounding metadata attributes solely to pages
   dated strictly before the origin;
3. runs the same LLM leakage verifier as the shared tool on the kept text.

If nothing survives, the search retries naming the rejected pages, and after the
attempt budget returns the shared tool's `[SEARCH_VERIFICATION_FAILED]` sentinel, which
the agent's instruction already handles. The gated tool has the shared tool's name,
signature, and description, so the LLM's prompt and tool schema are identical (tested).

The predictor ID gains `_dategated`. The scenario memory store is keyed by predictor ID,
so the gated agent keeps its own memory and never reads frameworks the ungated agent
built from possibly post-origin news.

Each prediction's `metadata["date_gate"]` records every search attempt: page verdicts,
sentences kept and dropped, and outcome.

## Requires

`wti_scenario_memory_agent` and `cfm_agent_v_5_2_dategated` (which needs
`cfm_agent_v_5_2`) in the same package.

## Use

```python
from energy_oil_forecasting.wti_scenario_memory_agent_dategated import build_agent_dategated

predictor = build_agent_dategated()  # same defaults as wti_scenario_memory_agent.build_agent()
```

The weekly June 2025 – June 2026 comparison against AutoARIMA runs with
`scripts/run_v52_weekly_extended.py --variant memory_dategated`. Origins run in date
order, so memory builds up as it would live.

## Tests

```bash
uv run pytest -q implementations/energy_oil_forecasting/wti_scenario_memory_agent_dategated/tests
```
