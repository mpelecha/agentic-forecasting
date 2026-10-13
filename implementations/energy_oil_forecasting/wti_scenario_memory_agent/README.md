# WTI Scenario Memory Agent

One folder, one agent. A WTI crude-oil forecaster that:

1. Fits a deterministic AutoARIMA anchor in Python (no LLM).
2. Asks a news-searching LLM analyst for 2–3 competing named scenarios, each
   with a probability (must sum to 1.0), a price range, and a stance on a
   shared set of factors. Before answering, the LLM sees its own scenarios
   from the last two forecast origins, so scenario names stay stable over time.
3. Shifts the ARIMA quantile grid toward the scenarios' probability-weighted
   price (re-grounded in real historical price moves, never the LLM's raw
   dollar figures) and widens its tails toward the probability-weighted
   scenario mixture.

Python owns every final number; the LLM only supplies scenarios and
probabilities.

## Dependencies

Everything in this folder depends only on itself and the shared
`aieng.forecasting` framework (install with the `numerical` and `agentic`
extras). The scripts additionally use `energy_oil_forecasting.data`,
`energy_oil_forecasting.paths`, and `specs/energy_oil_backtest.yaml` — all part
of the base `energy_oil_forecasting` package in every fork.

LLM/search access uses the same ADK / Vertex / Langfuse setup as every other
news agent in this repo.

## Install into another fork

Copy this folder into `implementations/energy_oil_forecasting/` of the fork.
Nothing else needs to change.

## Use it

```python
from energy_oil_forecasting.wti_scenario_memory_agent import build_agent

predictor = build_agent()          # lite model, level-price ARIMA anchor
predictions = predictor.predict(task, context)
```

`build_agent()` defaults reproduce the variant scored in the NB04b
quarterly backtest/eval, so its `predictor_id`
(`agent_predictor_wti_analyst_news_scenario_schema_anchored_enhanced_gemini-3.1-flash-lite-preview_continuous`)
matches those results. Pass `log_returns=True` for the log-return anchor
variant; this changes `predictor_id`, so it gets its own cache and memory.

To add it to a backtest registry:

```python
PredictorEntry(
    f"News Agent SS Anchored + Memory ({LITE_MODEL})",
    lambda: build_agent(LITE_MODEL),
    enabled=True,
)
```

## Scripts

Run from `implementations/energy_oil_forecasting/`:

| Command | What it does |
|---|---|
| `uv run python wti_scenario_memory_agent/scripts/smoke_test.py` | 3 weekly origins through the real backtest harness, under a separate `_smoketest` name so it never touches real memory or cache |
| `uv run python wti_scenario_memory_agent/scripts/daily_update.py` | One live forecast for today, retried up to 3 times |
| `uv run python wti_scenario_memory_agent/scripts/show_scenario_evolution.py` | Read-only: prints how scenarios continued or changed across a cached backtest |

## Where results are saved

All under `implementations/data/` (gitignored):

| Path | Written by |
|---|---|
| `scenario_memory/<predictor_id>/<task_id>.jsonl` | Every `predict()` call — one line of factors + scenarios per origin |
| `live_forecasts/<predictor_id>/<task_id>.jsonl` | `daily_update.py` — one line per horizon with point, quantiles, metadata |
| `predictions/<spec_id>/<predictor_id>__<task_id>.yaml` | Backtests via `cached_multi_backtest` (including the smoke test) |

Each prediction's `metadata` holds the factors, scenarios, raw ARIMA anchor,
center shift, widening targets, `prior_frameworks_used`, and
`unverified_continuations` (continuation claims that did not match a prior
scenario name).

## Caveats

- **Stateful across origins.** Origin N's prompt depends on what earlier
  origins wrote to memory in the same run. Interrupted/resumed runs, or two
  processes sharing one `predictor_id`, may not reproduce identical scenario
  continuity.
- **Memory is scoped by `predictor_id`.** Changing model, `log_returns`, or
  the config name starts a fresh memory file.
- **Results so far.** On the 10-year quarterly grid this agent scored mean
  CRPS 4.13 (backtest, 43 origins) and 5.10 (eval, 8 origins). Its eval score
  is dominated by the 2026-04-01 origin (Strait of Hormuz closure), where the
  63-day forecast missed badly.

## Files

| File | Contents |
|---|---|
| `__init__.py` | `build_agent()` entry point |
| `agent.py` | LLM instruction text and `AgentConfig` builder |
| `predictor.py` | Memory read → anchored prediction → memory write |
| `anchored_predictor.py` | ARIMA anchor, probability-weighted center shift, tail widening |
| `arima_anchor.py` | AutoARIMA anchor computation |
| `anchored_prompt.py`, `prompt.py` | Prompt payload builders (ARIMA anchor, then prior frameworks) |
| `core.py` | Factor/scenario schemas and validators (probabilities sum to 1, point forecast consistency, etc.) |
| `schema.py` | Memory-aware scenario card (`continues_prior_scenario` / `continued_from`) |
| `memory.py` | Append-only JSONL memory store, leakage-safe reads |
| `price_deltas.py` | Historical price-delta percentiles used to ground the center shift |
