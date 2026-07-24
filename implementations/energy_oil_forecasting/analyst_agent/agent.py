"""WTI crude oil analyst agent configurations and prompt builder.

Provides four :class:`~aieng.forecasting.methods.agentic.agent_factory.AgentConfig`
factories that define progressive agent capability levels:

1. :func:`build_wti_basic_config` — LLM reasons from price history alone (no tools).
2. :func:`build_wti_news_config` — Adds bounded Google Search via a
   :class:`~aieng.forecasting.methods.agentic.agent_factory.ContextRetrievalConfig`
   sub-agent with strict temporal cutoffs.
3. :func:`build_wti_code_exec_config` — Adds Gemini native code execution and
   three forecasting skills on top of the news-grounded configuration.
4. :func:`build_wti_tool_config` — Adds a conventional
   :class:`~aieng.forecasting.methods.agentic.forecast_tool.ForecastTool`
   (AutoARIMA) on top of news grounding — a rigid, pre-specified alternative to
   open-ended code execution.
5. :func:`build_wti_news_factors_v2_config` — Adds a two-tier core/transitory
   factor framework to scenario decomposition, plus a more open search
   strategy, on top of the contrarian news-grounded configuration — a
   prompt-only test of whether separating structural from situational
   factors improves calibration, before investing in a structured output
   schema.

   
Also provides:

- :class:`WtiPriceForecastPromptBuilder`: Pydantic ``BaseModel`` that serialises
  the task and history into a structured JSON payload for the agent.
- :func:`build_wti_agent_predictor`: convenience factory that wires a config to
  an :class:`~aieng.forecasting.methods.agentic.predictor.AgentPredictor`.

Module-level ``__getattr__`` exposes ``root_agent`` lazily so ``adk web`` can
load this module for interactive (schema-free) use without importing the full
predictor stack.
"""

from __future__ import annotations

import json
from math import isfinite, sqrt
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from aieng.forecasting.data import DataService
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES, Prediction
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic import (
    AgentPredictor,
    ContinuousAgentForecastOutput,
    ForecastTool,
    build_adk_agent,
)
from aieng.forecasting.methods.agentic.agent_factory import (
    AgentConfig,
    CodeExecutionConfig,
    ContextRetrievalConfig,
)
from aieng.forecasting.methods.numerical.darts_arima import DartsAutoARIMAPredictor
from aieng.forecasting.models import ADVANCED_MODEL, LITE_MODEL
from energy_oil_forecasting.data import WTI_SERIES_ID, build_wti_service
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# System prompt (root analyst agent)
# ---------------------------------------------------------------------------

_WTI_MULTITASK_ANALYST_INSTRUCTION = """\
## Role

You are an expert WTI crude oil market analyst.

## Input

You will receive a JSON payload containing:
- `task_spec`: the exact question and required JSON output schema
- `as_of`: the forecast origin date (temporal cutoff)
- `origin_price_usd_bbl`: WTI close on the origin date
- `target_history_csv`: compressed WTI daily close history

When context retrieval is enabled, call ``search_web`` BEFORE answering.

## Output contract

Read the data (and briefing, if retrieved) carefully, then execute the task \
in `task_spec` precisely.

If a `set_model_response` tool is available, call it with your complete JSON \
as `json_response` — the exact schema is described in `task_spec`. Otherwise \
return the JSON directly as plain text with no preamble.\
"""


def _build_wti_analyst_instruction() -> str:
    """Build the WTI analyst instruction, embedding the output schema from the class.

    Using a function instead of a static string ensures the ``## Output schema``
    block is always in sync with ``ContinuousAgentForecastOutput`` —
    no manual JSON to maintain.
    """
    schema = ContinuousAgentForecastOutput.prompt_schema_json()
    return (
        "## Role\n\n"
        "You are an expert WTI crude oil market analyst. You produce calibrated "
        "probabilistic price forecasts for WTI crude oil futures, grounded in "
        "supply/demand fundamentals, geopolitical risk, and historical price dynamics.\n\n"
        "## Forecasting contract\n\n"
        "You will receive a JSON payload containing:\n"
        "- `task`: the task identifier\n"
        "- `as_of`: the forecast origin date in YYYY-MM-DD format\n"
        "- `horizons`: a list of integer horizon steps (business days ahead)\n"
        "- `standard_quantiles`: the exact quantile levels you must produce\n"
        "- `target_summary`: last close price, 52-week range, and observation count\n"
        "- `target_history_csv`: WTI daily close history (recent 6 months daily, "
        "older history as weekly averages)\n\n"
        "Rules:\n"
        "1. Produce one forecast for each horizon listed in `horizons`.\n"
        "2. Use exactly the quantile levels from `standard_quantiles` — no additions, no omissions.\n"
        "3. `point_forecast` must exactly equal the 0.50 quantile value.\n"
        "4. Quantile values must be strictly non-decreasing as quantile levels increase.\n"
        "5. Document your reasoning in the `rationale` fields.\n"
        "6. When tools are enabled, conclude with `set_model_response` to return the structured forecast.\n\n"
        "## Output schema\n\n"
        "Call `set_model_response` with a `json_response` string matching **exactly**:\n\n"
        "```json\n" + schema + "\n```\n\n"
        'Critical: use `"horizon"` (integer, not `"horizon_days"`). '
        '`"quantiles"` is a **list** of `{"quantile": <level>, "value": <price>}` '
        "objects — not a dict. Omit any field not shown above.\n\n"
        "## Analysis discipline\n\n"
        "When context retrieval is available, call ``search_web`` to gather market "
        "intelligence BEFORE producing forecasts.\n\n"
        "Call ``search_web`` with ``query`` and ``cutoff_date`` (set to the ``as_of`` "
        "date from the payload). The ``cutoff_date`` MUST always equal ``as_of`` — "
        "this is the temporal fence that prevents post-origin information from "
        "contaminating historical backtests.\n\n"
        "If ``search_web`` returns a result beginning with "
        "``[SEARCH_VERIFICATION_FAILED]``, treat it as no verified news context for "
        "that query. Do not use your own background knowledge to fill the gap or "
        "speculate about what the news might have said — proceed with price-history "
        "and other available signals only, and note the gap in your rationale.\n\n"
        "Recommended queries (call ``search_web`` once per topic):\n"
        '- ``search_web(query="WTI crude oil price trend and OPEC+ supply decisions", cutoff_date=<as_of>)``\n'
        '- ``search_web(query="Persian Gulf geopolitical risk shipping lane disruptions", cutoff_date=<as_of>)``\n'
        '- ``search_web(query="US Strategic Petroleum Reserve policy and global demand outlook", cutoff_date=<as_of>)``\n\n'
        "## Scenario decomposition (required before finalizing quantiles)\n\n"
        "Before producing your final quantiles, explicitly write out 2-3 named, "
        "competing scenarios for how WTI could move over the forecast window — "
        "grounded in the search context you retrieved. For each scenario, briefly "
        "tag how it resolves each of these four factors as bullish, bearish, or "
        "neutral for price: OPEC+ supply policy, Persian Gulf / shipping-lane risk, "
        "SPR & demand outlook, and inventory levels. Your scenarios must genuinely "
        "disagree: at least two of your named scenarios must differ in their tag on "
        "at least two of these four factors — not just differ in tone while tagging "
        "all four the same way. At least one scenario must be a genuine tail case — "
        "explicitly labeled low-probability-but-large-impact (e.g., \"full regional "
        "war\", \"sudden diplomatic resolution\") — not a milder variant of your main "
        "narrative. For each scenario, state:\n"
        "- A short name (e.g., \"Escalation continues\", \"Diplomatic de-escalation\", "
        "\"Surplus reasserts\")\n"
        "- Its approximate probability (they should roughly sum to 1.0 — these are "
        "illustrative judgment calls, not a rigorous elicitation, so do not over-invest "
        "in precise-looking probability values)\n"
        "- A rough price level it implies at your longest horizon\n"
        "- The key driver behind it, drawn from what you actually found in search\n\n"
        "Your final quantile grid must be consistent with the SPREAD across these "
        "scenarios — not just your single most likely one. Concretely: your 80% "
        "interval (0.1/0.9 quantiles) should be wide enough to plausibly cover your "
        "named scenarios' price levels, including your tail scenario. If your "
        "scenarios disagree by $10+, a narrow interval is not consistent with your "
        "own analysis.\n\n"
        "Summarize the scenarios and how they map to your final interval width in "
        "the `rationale` field.\n\n"
        "Document your key assumptions (OPEC+ policy, shipping lane risk, inventory "
        "levels, macro demand) in the `rationale` fields of your forecast output."
    )


_WTI_ANALYST_INSTRUCTION = _build_wti_analyst_instruction()



def _build_wti_analyst_instruction_factors_v2() -> str:
    """Variant of :func:`_build_wti_analyst_instruction` testing a two-tier
    core/transitory factor framework in scenario decomposition, plus a more
    open search strategy that doesn't pre-name specific topics.

    Prompt-only test: no output schema changes. Differs from the base
    instruction in three places — the query strategy (broad-then-follow-up
    instead of three fixed named topics), the scenario decomposition section
    (core/transitory factors chosen once per origin instead of a fixed
    four-factor checklist), and a reinforcement line tying the rationale
    field's contents back to the output schema. Tests whether separating
    "always relevant" structural factors from "currently dominant but not
    permanent" situational ones improves calibration, before investing in a
    structured output schema to enforce it.
    """
    schema = ContinuousAgentForecastOutput.prompt_schema_json()
    return (
        "## Role\n\n"
        "You are an expert WTI crude oil market analyst. You produce calibrated "
        "probabilistic price forecasts for WTI crude oil futures, grounded in "
        "supply/demand fundamentals, geopolitical risk, and historical price dynamics.\n\n"
        "## Forecasting contract\n\n"
        "You will receive a JSON payload containing:\n"
        "- `task`: the task identifier\n"
        "- `as_of`: the forecast origin date in YYYY-MM-DD format\n"
        "- `horizons`: a list of integer horizon steps (business days ahead)\n"
        "- `standard_quantiles`: the exact quantile levels you must produce\n"
        "- `target_summary`: last close price, 52-week range, and observation count\n"
        "- `target_history_csv`: WTI daily close history (recent 6 months daily, "
        "older history as weekly averages)\n\n"
        "Rules:\n"
        "1. Produce one forecast for each horizon listed in `horizons`.\n"
        "2. Use exactly the quantile levels from `standard_quantiles` — no additions, no omissions.\n"
        "3. `point_forecast` must exactly equal the 0.50 quantile value.\n"
        "4. Quantile values must be strictly non-decreasing as quantile levels increase.\n"
        "5. Document your reasoning in the `rationale` fields.\n"
        "6. When tools are enabled, conclude with `set_model_response` to return the structured forecast.\n\n"
        "## Output schema\n\n"
        "Call `set_model_response` with a `json_response` string matching **exactly**:\n\n"
        "```json\n" + schema + "\n```\n\n"
        'Critical: use `"horizon"` (integer, not `"horizon_days"`). '
        '`"quantiles"` is a **list** of `{"quantile": <level>, "value": <price>}` '
        "objects — not a dict. Omit any field not shown above.\n\n"
        "For the `rationale` field specifically, make sure it actually contains: "
        "your identified core and transitory factors, your named scenarios with "
        "their factor tags, and how your interval maps to the scenario spread — "
        "not a short summary that omits this.\n\n"
        "## Analysis discipline\n\n"
        "When context retrieval is available, call ``search_web`` to gather market "
        "intelligence BEFORE producing forecasts.\n\n"
        "Call ``search_web`` with ``query`` and ``cutoff_date`` (set to the ``as_of`` "
        "date from the payload). The ``cutoff_date`` MUST always equal ``as_of`` — "
        "this is the temporal fence that prevents post-origin information from "
        "contaminating historical backtests.\n\n"
        "If ``search_web`` returns a result beginning with "
        "``[SEARCH_VERIFICATION_FAILED]``, treat it as no verified news context for "
        "that query. Do not use your own background knowledge to fill the gap or "
        "speculate about what the news might have said — proceed with price-history "
        "and other available signals only, and note the gap in your rationale.\n\n"
        "Recommended search strategy (adapt the specific wording each time — do "
        "not repeat the same named topics regardless of what is actually "
        "relevant):\n"
        "1. Start broad: ``search_web(query=\"WTI crude oil price current market "
        "drivers\", cutoff_date=<as_of>)`` to see what is actually moving the "
        "market right now.\n"
        "2. Based on what that surfaces, run 1-2 follow-up ``search_web`` calls "
        "targeting whatever specific structural or situational factors the first "
        "search actually pointed to — do not default to OPEC+, the Persian Gulf, "
        "or the US SPR unless the first search genuinely surfaced them as "
        "currently relevant.\n\n"
        "## Scenario decomposition (required before finalizing quantiles)\n\n"
        "Before producing your final quantiles, first identify the factors "
        "relevant to this forecast, using two tiers:\n\n"
        "- **Core factors**: macro, financial, or geopolitical themes durable "
        "enough that they would still plausibly matter in five years or more. "
        "Identify 3-5 that are genuinely relevant to the current search context — "
        "the test for inclusion is durability, not topic; do not default to a "
        "fixed checklist.\n"
        "- **Transitory factors**: macro, financial, or geopolitical factors "
        "that do NOT meet that decade-plus durability bar — situational "
        "developments that could plausibly resolve, reverse, or become "
        "irrelevant within months. Identify 1-2 from the current search "
        "context. For each, assign an impact score (low / medium / high) "
        "reflecting how large a price effect it could plausibly have, "
        "independent of its bullish/bearish direction. Do not treat a "
        "transitory factor as a permanent structural driver, even if it is "
        "currently dominating price action.\n\n"
        "Identify this factor set ONCE for the forecast — do not pick a "
        "different set of core or transitory factors per scenario. Then write "
        "2-3 named, competing scenarios for how WTI could move over the "
        "forecast window, each representing a different combination of "
        "stances (bullish, bearish, or neutral for price) across that SAME "
        "shared factor set, including each transitory factor's impact "
        "score. Your scenarios must genuinely disagree: at least two of "
        "your named scenarios must differ in their tag on at least two of "
        "your identified factors — not just differ in tone while tagging "
        "everything the same way. At least one scenario must be a genuine "
        "tail case — explicitly labeled low-probability-but-large-impact "
        "(e.g., \"full regional war\", \"sudden diplomatic resolution\") — "
        "not a milder variant of your main narrative. For each scenario, "
        "state:\n"
        "- A short name (e.g., \"Escalation continues\", \"Diplomatic de-escalation\", "
        "\"Surplus reasserts\")\n"
        "- Its approximate probability (they should roughly sum to 1.0 — these are "
        "illustrative judgment calls, not a rigorous elicitation, so do not over-invest "
        "in precise-looking probability values)\n"
        "- A rough price level it implies at your longest horizon\n"
        "- The key driver behind it, drawn from what you actually found in search\n\n"
        "Your final quantile grid must be consistent with the SPREAD across these "
        "scenarios — not just your single most likely one. Concretely: your 80% "
        "interval (0.1/0.9 quantiles) should be wide enough to plausibly cover your "
        "named scenarios' price levels, including your tail scenario. If your "
        "scenarios disagree by $10+, a narrow interval is not consistent with your "
        "own analysis.\n\n"
        "Summarize the scenarios and how they map to your final interval width in "
        "the `rationale` field. State clearly which factors you classified as core "
        "versus transitory and why.\n\n"
        "Document your key assumptions in the `rationale` fields of your forecast output."
    )


_WTI_ANALYST_INSTRUCTION_FACTORS_V2 = _build_wti_analyst_instruction_factors_v2()


def _build_wti_analyst_instruction_scenario_schema() -> str:
    """Variant of :func:`_build_wti_analyst_instruction_factors_v2` that moves
    the core/transitory scenario decomposition from free-text ``rationale``
    into real, schema-validated ``factors``/``scenarios`` fields.

    Same search strategy and factor-tier reasoning as the ``factors_v2``
    prompt-only test — the only change is *where* the model is told to put
    the result: structured JSON fields (enforced by
    :class:`WtiScenarioForecastOutput`) instead of prose it may or may not
    follow consistently.
    """
    schema = WtiScenarioForecastOutput.prompt_schema_json()
    return (
        "## Role\n\n"
        "You are an expert WTI crude oil market analyst. You produce calibrated "
        "probabilistic price forecasts for WTI crude oil futures, grounded in "
        "supply/demand fundamentals, geopolitical risk, and historical price dynamics.\n\n"
        "## Forecasting contract\n\n"
        "You will receive a JSON payload containing:\n"
        "- `task`: the task identifier\n"
        "- `as_of`: the forecast origin date in YYYY-MM-DD format\n"
        "- `horizons`: a list of integer horizon steps (business days ahead)\n"
        "- `standard_quantiles`: the exact quantile levels you must produce\n"
        "- `target_summary`: last close price, 52-week range, and observation count\n"
        "- `target_history_csv`: WTI daily close history (recent 6 months daily, "
        "older history as weekly averages)\n\n"
        "Rules:\n"
        "1. Produce one forecast for each horizon listed in `horizons`.\n"
        "2. Use exactly the quantile levels from `standard_quantiles` — no additions, no omissions.\n"
        "3. `point_forecast` must exactly equal the 0.50 quantile value.\n"
        "4. Quantile values must be strictly non-decreasing as quantile levels increase.\n"
        "5. Document your reasoning in the `rationale` fields.\n"
        "6. When tools are enabled, conclude with `set_model_response` to return the structured forecast.\n\n"
        "## Output schema\n\n"
        "Call `set_model_response` with a `json_response` string matching **exactly**:\n\n"
        "```json\n" + schema + "\n```\n\n"
        'Critical: use `"horizon"` (integer, not `"horizon_days"`). '
        '`"quantiles"` is a **list** of `{"quantile": <level>, "value": <price>}` '
        "objects — not a dict. `stances` keys in each scenario must exactly match "
        "the `name` of every entry in `factors` — no extra keys, none missing. "
        "Omit any field not shown above.\n\n"
        "## Analysis discipline\n\n"
        "When context retrieval is available, call ``search_web`` to gather market "
        "intelligence BEFORE producing forecasts.\n\n"
        "Call ``search_web`` with ``query`` and ``cutoff_date`` (set to the ``as_of`` "
        "date from the payload). The ``cutoff_date`` MUST always equal ``as_of`` — "
        "this is the temporal fence that prevents post-origin information from "
        "contaminating historical backtests.\n\n"
        "If ``search_web`` returns a result beginning with "
        "``[SEARCH_VERIFICATION_FAILED]``, treat it as no verified news context for "
        "that query. Do not use your own background knowledge to fill the gap or "
        "speculate about what the news might have said — proceed with price-history "
        "and other available signals only, and note the gap in your rationale.\n\n"
        "Recommended search strategy (adapt the specific wording each time — do "
        "not repeat the same named topics regardless of what is actually "
        "relevant):\n"
        "1. Start broad: ``search_web(query=\"WTI crude oil price current market "
        "drivers\", cutoff_date=<as_of>)`` to see what is actually moving the "
        "market right now.\n"
        "2. Based on what that surfaces, run 1-2 follow-up ``search_web`` calls "
        "targeting whatever specific structural or situational factors the first "
        "search actually pointed to.\n\n"
        "## Factors and scenarios (populate the `factors` and `scenarios` fields)\n\n"
        "Before producing your final quantiles, populate `factors` with the "
        "factors relevant to this forecast, using two tiers:\n\n"
        "- **Core factors** (2-5 entries, `tier: \"core\"`): macro, financial, or "
        "geopolitical themes durable enough that they would still plausibly "
        "matter in five years or more. The test for inclusion is durability, "
        "not topic; do not default to a fixed checklist.\n"
        "- **Transitory factors** (1-2 entries, `tier: \"transitory\"`): macro, "
        "financial, or geopolitical factors that do NOT meet that decade-plus "
        "durability bar — situational developments that could plausibly "
        "resolve, reverse, or become irrelevant within months. Each requires "
        "an `impact_score` (low/medium/high) reflecting how large a price "
        "effect it could plausibly have, independent of direction. Do not "
        "treat a transitory factor as a permanent structural driver, even if "
        "it is currently dominating price action.\n\n"
        "Identify this factor set ONCE for the forecast — every scenario tags "
        "the SAME shared factors, not its own set. Then populate `scenarios` "
        "with 2-3 named, competing scenarios, each with a `stances` entry for "
        "every factor in `factors` (bullish/bearish/neutral). Your scenarios "
        "must genuinely disagree: at least two scenarios must differ in stance "
        "on at least two factors — not just differ in tone while tagging "
        "everything the same way. At least one scenario must set "
        "`is_tail_case: true` — a genuine low-probability, high-impact case, "
        "not a milder variant of your main narrative.\n\n"
        "Your final quantile grid must be consistent with the SPREAD across "
        "your scenarios' `price_low`/`price_high` ranges — not just your "
        "single most likely one. If your scenarios disagree by $10+, a narrow "
        "interval is not consistent with your own analysis.\n\n"
        "Use the `rationale` field for a brief narrative summary of your "
        "reasoning — the structured detail belongs in `factors` and "
        "`scenarios`, not repeated at length in prose."
    )




# ---------------------------------------------------------------------------
# Context retrieval instruction (sub-agent)
# ---------------------------------------------------------------------------

_WTI_CONTEXT_RETRIEVAL_INSTRUCTION = """\
You are an oil market intelligence specialist with access to web search.

Search for information relevant to the query and return a concise structured \
markdown summary (3-5 paragraphs) covering relevant aspects of:
- WTI/Brent crude price level and recent trend
- OPEC+ production decisions and supply outlook
- Geopolitical risks in the Persian Gulf, Middle East, key shipping lanes
- US Strategic Petroleum Reserve and energy policy signals
- Notable tanker/shipping incidents or supply disruption signals
- Published analyst forecasts or unusual price-target revisions

Ground your summary in the search results you actually retrieve. \
When a cutoff date is specified, do not report or speculate about events \
that occurred after that date.

Before finalizing your summary, reason step by step: (1) for each candidate \
fact, judge its actual recency from the substance of the result itself, \
never from a source's claimed publish date or byline timestamp — those are \
frequently stale or updated after original publication; (2) discard \
anything you cannot confidently place before the cutoff date; (3) only then \
write your summary. Do not supplement the search results with your own \
background/training knowledge — if the results are insufficient, say so \
explicitly rather than filling gaps from memory.\
"""


_WTI_CONTRARIAN_CONTEXT_RETRIEVAL_INSTRUCTION = """\
You are an oil market intelligence specialist with access to web search.

Search for information relevant to the query and return a concise structured \
markdown summary (3-5 paragraphs) covering relevant aspects of:
- WTI/Brent crude price level and recent trend
- OPEC+ production decisions and supply outlook
- Geopolitical risks in the Persian Gulf, Middle East, key shipping lanes
- US Strategic Petroleum Reserve and energy policy signals
- Notable tanker/shipping incidents or supply disruption signals
- Published analyst and forecast views — including major banks, \
government/international energy agencies, and large physical oil trading \
houses — noting any unusual price-target revisions or shifts in outlook
- Where credible sources actively disagree (e.g. surplus vs. tight-market \
calls, bullish vs. bearish institutional forecasts) — name both sides \
rather than reporting only the majority view

If the current situation resembles a past market episode (e.g. a prior \
OPEC+ supply cut, a prior Strait of Hormuz scare, a prior demand-shock \
period), search for that precedent explicitly and report: what happened, \
how WTI/Brent moved in the following weeks, and how similar the current \
setup actually is versus superficially similar. Only include an analogue \
if you can ground it in a retrieved source — do not construct one from \
memory. If no clear precedent surfaces in search results, say so rather \
than forcing a comparison.

Ground your summary in the search results you actually retrieve. \
When a cutoff date is specified, do not report or speculate about events \
that occurred after that date.

Before finalizing your summary, reason step by step: (1) for each candidate \
fact, judge its actual recency from the substance of the result itself, \
never from a source's claimed publish date or byline timestamp — those are \
frequently stale or updated after original publication; (2) discard \
anything you cannot confidently place before the cutoff date; (3) only then \
write your summary. Do not supplement the search results with your own \
background/training knowledge — if the results are insufficient, say so \
explicitly rather than filling gaps from memory.\
"""


_WTI_FACTORS_V2_CONTEXT_RETRIEVAL_INSTRUCTION = """\
You are an oil market intelligence specialist with access to web search.

Search for information relevant to the query you were given and return a \
concise, grounded markdown summary (3-5 paragraphs). Report what is \
actually driving price action according to the sources you retrieve — do \
not impose a fixed checklist of topics; let the search results themselves \
determine what is significant factors, and useful right now.

Where credible sources actively disagree on a macro, financial, or \
geopolitical driver — name both sides rather than reporting only the \
majority view.

If the current situation resembles a past market episode, search for that \
precedent explicitly and report: what happened, how WTI/Brent moved in the \
following weeks, and how similar the current setup actually is versus \
superficially similar. Only include an analogue if you can ground it in a \
retrieved source — do not construct one from memory. If no clear precedent \
surfaces in search results, say so rather than forcing a comparison.

Ground your summary in the search results you actually retrieve. \
When a cutoff date is specified, do not report or speculate about events \
that occurred after that date!

Before finalizing your summary, reason step by step: (1) for each candidate \
fact, judge its actual recency from the substance of the result itself, \
never from a source's claimed publish date or byline timestamp — those are \
frequently stale or updated after original publication; (2) discard \
anything you cannot confidently place before the cutoff date; (3) only then \
write your summary. Do not supplement the search results with your own \
background/training knowledge — if the results are insufficient, say so \
explicitly rather than filling gaps from memory.\
"""



# ---------------------------------------------------------------------------
# Skills supplement (appended to instruction when skills are attached)
# ---------------------------------------------------------------------------

_CODE_EXEC_SKILLS_SUPPLEMENT = """

## Skills

You have access to two forecasting skills via the SkillToolset. All data
available to code execution comes from the JSON payload in your context —
there are no disk files to read.

**Recommended invocation order:**

1. `statistical-analysis` — run first. Provides diagnostic code patterns
   for interrogating the price series you have been given: vol regime
   classification, anomaly detection, and adaptive trend-window selection.
   The output of Pattern 3 (trend window) is the input to the projection
   skill below.

2. `trend-projection` — run second. Provides code patterns for fitting a
   linear trend on the window chosen above, projecting point forecasts to
   each horizon, and calibrating 80% prediction interval widths.

**To use a skill:**
1. Call `list_skills` to see available skill names and descriptions.
2. Call `load_skill(<name>)` to read the skill's full instructions.
3. Call `load_skill_resource(<skill_name>, <file_path>)` to load a
   reference file (e.g. `references/wti_benchmarks.json`).

These skills have NO scripts. Do not call `run_skill_script`.\
"""

# ---------------------------------------------------------------------------
# Forecast tool supplement (appended to instruction when the forecast tool is attached)
# ---------------------------------------------------------------------------

_FORECAST_TOOL_SUPPLEMENT = f"""

## Statistical forecast tool

You have access to `run_forecast`, a conventional statistical baseline
(AutoARIMA) you can call directly. Unlike open-ended code, this tool has a fixed,
auditable interface and returns a structured forecast you can reason from.

Call it ONCE before producing your forecast, with:
- `series_id`: "{WTI_SERIES_ID}"
- `cutoff_date`: the `as_of` date from the payload (YYYY-MM-DD). This is the
  information cutoff — the model uses only data on or before it.
- `horizons`: the `horizons` list from the payload.
- `frequency`: "B" (WTI trades on business days).

The tool returns JSON with point forecasts and 80%/90% prediction intervals per
horizon. Treat it as a disciplined statistical anchor: combine it with the
market context from the search sub-agent. You may adjust away from the baseline
when fundamentals or geopolitical risk justify it — document your reasoning in
the `rationale` fields.\
"""

# ---------------------------------------------------------------------------
# Skill directories
# ---------------------------------------------------------------------------

_SKILLS_ROOT = Path(__file__).parent / "skills"


# ---------------------------------------------------------------------------
# History compression
# ---------------------------------------------------------------------------


def compress_history(df: pd.DataFrame) -> str:
    """Compress WTI daily history to stay within context limits.

    Returns daily bars for the most recent 6 months and weekly averages for
    older history.  The CSV header is ``date,close``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ``timestamp`` and ``value``.

    Returns
    -------
    str
        CSV string with header ``date,close``.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    cutoff = df["timestamp"].max() - pd.DateOffset(months=6)

    recent = df[df["timestamp"] >= cutoff].copy()
    old = df[df["timestamp"] < cutoff].copy()

    rows: list[str] = ["date,close"]

    if not old.empty:
        old_indexed = old.set_index("timestamp")["value"]
        weekly: pd.Series = old_indexed.resample("W").mean().dropna()
        for date, val in weekly.items():
            rows.append(f"{date.date()},{val:.2f}")

    for _, row in recent.iterrows():
        rows.append(f"{row['timestamp'].date()},{row['value']:.2f}")

    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


class WtiPriceForecastPromptBuilder(BaseModel):
    """Prompt builder for WTI crude oil price forecasting tasks.

    Produces a structured JSON payload for the analyst agent containing the
    task specification, compressed price history, and a data summary.
    The payload includes ``standard_quantiles`` explicitly so the agent knows
    the exact grid it must produce.

    Implements the
    :class:`~aieng.forecasting.methods.agentic.predictor.ForecastPromptBuilder`
    protocol (structural typing — no explicit inheritance required).
    """

    model_config = {"extra": "forbid"}

    def __call__(self, *, task: ForecastingTask, context: ForecastContext) -> str:
        """Serialise the task and context into a JSON string for the agent.

        Parameters
        ----------
        task : ForecastingTask
            The forecasting task — supplies ``task_id``, ``horizons``.
        context : ForecastContext
            The information state at forecast time.

        Returns
        -------
        str
            JSON-serialised payload with task metadata, compressed history, and
            the standard quantile grid the agent must populate.
        """
        df = context.get_series(task.target_series_id)
        compressed = compress_history(df)

        last_row = df.iloc[-1]
        last_close = float(last_row["value"])
        last_date = str(pd.Timestamp(last_row["timestamp"]).date())
        trailing_252 = df["value"].tail(252)

        payload: dict[str, Any] = {
            "task": task.task_id,
            "as_of": str(context.as_of)[:10],
            "horizons": list(task.horizons),
            "standard_quantiles": list(STANDARD_QUANTILES),
            "target_summary": {
                "last_close_usd_bbl": last_close,
                "last_date": last_date,
                "n_trading_days": int(len(df)),
                "52w_high": float(trailing_252.max()),
                "52w_low": float(trailing_252.min()),
            },
            "target_history_csv": compressed,
        }

        return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# Structured scenario output schema
# ---------------------------------------------------------------------------


class WtiFactor(BaseModel):
    """One core or transitory factor identified for this forecast.

    Attributes
    ----------
    name : str
        Short factor label.
    category : Literal["macro", "financial", "geopolitical"]
        Broad category type — deliberately generic; the model chooses the
        specific factor freely within these three types.
    tier : Literal["core", "transitory"]
        ``"core"`` for themes durable enough to plausibly matter in ten
        years or more; ``"transitory"`` for situational developments that
        could resolve, reverse, or become irrelevant within months.
    impact_score : Literal["low", "medium", "high"] or None
        Required for transitory factors (magnitude of potential price
        effect, independent of direction); must be omitted for core factors.
    """

    model_config = {"extra": "ignore"}

    name: str = Field(min_length=1, description="Short factor label.")
    category: Literal["macro", "financial", "geopolitical"] = Field(description="Broad category type.")
    tier: Literal["core", "transitory"] = Field(
        description="'core' for decade-plus durable themes; 'transitory' for situational developments."
    )
    impact_score: Literal["low", "medium", "high"] | None = Field(
        default=None, description="Required for transitory factors; omit for core factors."
    )

    @model_validator(mode="after")
    def _transitory_factors_have_impact_score(self) -> "WtiFactor":
        """Require an impact score for transitory factors; forbid it for core factors."""
        if self.tier == "transitory" and self.impact_score is None:
            raise ValueError("Transitory factors must set impact_score.")
        if self.tier == "core" and self.impact_score is not None:
            raise ValueError("Core factors must not set impact_score (only transitory factors carry one).")
        return self


class WtiScenarioCard(BaseModel):
    """One named, competing scenario, with stances against the shared factor set.

    Attributes
    ----------
    name : str
        Short scenario label, e.g. ``"Escalation continues"``.
    probability : float
        Approximate probability in ``[0, 1]``. Illustrative, not a rigorous
        elicitation — scenarios need not sum to exactly 1.0.
    price_low : float
        Lower end of this scenario's implied WTI price range at the
        forecast's longest horizon. Equal to ``price_high`` for a single
        point estimate rather than a range.
    price_high : float
        Upper end of this scenario's implied price range.
    is_tail_case : bool
        ``True`` for the required low-probability, high-impact scenario.
    stances : dict[str, Literal["bullish", "bearish", "neutral"]]
        This scenario's stance on each factor in the forecast's shared
        ``factors`` list, keyed by factor name. Must cover exactly the
        shared factor names — enforced at the
        :class:`WtiScenarioForecastOutput` level, where the full factor
        list is available for cross-checking.
    """

    model_config = {"extra": "ignore"}

    name: str = Field(min_length=1, description="Short scenario name.")
    probability: float = Field(ge=0.0, le=1.0, description="Approximate probability; illustrative, not rigorous.")
    price_low: float = Field(description="Lower end of this scenario's implied price range at the longest horizon.")
    price_high: float = Field(description="Upper end of this scenario's implied price range.")
    is_tail_case: bool = Field(
        default=False, description="True for the required low-probability, high-impact scenario."
    )
    stances: dict[str, Literal["bullish", "bearish", "neutral"]] = Field(
        description="This scenario's stance on each shared factor, keyed by factor name."
    )

    @field_validator("price_low", "price_high")
    @classmethod
    def _prices_are_finite(cls, value: float) -> float:
        """Reject NaN and infinite prices."""
        if not isfinite(value):
            raise ValueError("Scenario prices must be finite numbers.")
        return value

    @model_validator(mode="after")
    def _price_range_is_ordered(self) -> "WtiScenarioCard":
        """Reject an inverted price range."""
        if self.price_low > self.price_high:
            raise ValueError(f"price_low ({self.price_low}) must be <= price_high ({self.price_high}).")
        return self


# Tolerance for the point-forecast-vs-scenario consistency check, expressed
# as a fraction of the scenario price spread (not an absolute dollar amount)
# so it scales with how much the scenarios actually disagree. Floored at
# $1 in the check itself to avoid a degenerate zero-tolerance when all
# scenarios cluster tightly together.
_SCENARIO_CONSISTENCY_TOLERANCE = 0.15


class WtiScenarioForecastOutput(ContinuousAgentForecastOutput):
    """Continuous WTI forecast output with a required, structured scenario decomposition.

    Extends :class:`~aieng.forecasting.methods.agentic.ContinuousAgentForecastOutput`
    with ``factors`` (the shared core/transitory factor set, identified once)
    and ``scenarios`` (2-3 named scenarios tagging that same set), and
    overrides :meth:`to_predictions` to widen each horizon's outermost
    quantiles to at least span the model's own stated scenario price range
    when they don't already — a code-enforced consistency check, not a
    prompt request the model can silently ignore.

    Attributes
    ----------
    factors : list[WtiFactor]
        2-5 core factors and 1-2 transitory factors, identified once for
        the whole forecast.
    scenarios : list[WtiScenarioCard]
        2 or more named, competing scenarios. At least one must set
        ``is_tail_case=True``, and at least two scenarios must differ in
        their stance on at least two shared factors.
    """

    model_config = {"extra": "ignore"}

    factors: list[WtiFactor] = Field(
        description="The shared core/transitory factor set for this forecast, identified once."
    )
    scenarios: list[WtiScenarioCard] = Field(
        min_length=2,
        description="2-3 named, competing scenarios, each tagging the shared factor set.",
    )

    @model_validator(mode="after")
    def _factor_tier_counts_are_valid(self) -> "WtiScenarioForecastOutput":
        """Require 2-5 core factors and 1-2 transitory factors."""
        core = [factor for factor in self.factors if factor.tier == "core"]
        transitory = [factor for factor in self.factors if factor.tier == "transitory"]
        if not (2 <= len(core) <= 5):
            raise ValueError(f"Expected 2-5 core factors, got {len(core)}.")
        if not (1 <= len(transitory) <= 2):
            raise ValueError(f"Expected 1-2 transitory factors, got {len(transitory)}.")
        return self

    @model_validator(mode="after")
    def _scenarios_include_a_tail_case(self) -> "WtiScenarioForecastOutput":
        """Require at least one scenario explicitly marked as the tail case."""
        if not any(scenario.is_tail_case for scenario in self.scenarios):
            raise ValueError("At least one scenario must set is_tail_case=True.")
        return self

    @model_validator(mode="after")
    def _scenario_stances_cover_every_factor(self) -> "WtiScenarioForecastOutput":
        """Require each scenario's stances to cover exactly the shared factor names."""
        expected = {factor.name for factor in self.factors}
        for scenario in self.scenarios:
            actual = set(scenario.stances)
            if actual != expected:
                missing = sorted(expected - actual)
                extra = sorted(actual - expected)
                raise ValueError(
                    f"Scenario '{scenario.name}' stances must cover exactly the shared factors. "
                    f"Missing: {missing}; extra: {extra}."
                )
        return self

    @model_validator(mode="after")
    def _scenarios_genuinely_disagree(self) -> "WtiScenarioForecastOutput":
        """Require at least two scenarios to differ in stance on at least two shared factors."""
        factor_names = [factor.name for factor in self.factors]
        for i in range(len(self.scenarios)):
            for j in range(i + 1, len(self.scenarios)):
                first, second = self.scenarios[i], self.scenarios[j]
                differences = sum(1 for name in factor_names if first.stances.get(name) != second.stances.get(name))
                if differences >= 2:
                    return self
        raise ValueError(
            "No two scenarios differ in stance on at least two shared factors — "
            "scenarios must genuinely disagree, not just differ in tone."
        )


    @model_validator(mode="after")
    def _point_forecast_consistent_with_scenarios(self) -> "WtiScenarioForecastOutput":
        """Require the longest horizon's point_forecast to track the scenarios' probability-weighted price.

        Scenario prices are defined "at the forecast's longest horizon" (see
        ``WtiScenarioCard``), so only that horizon's ``point_forecast`` is checked.
        Deliberately a model validator, not a check inside ``to_predictions`` — a
        violation now raises during ``model_validate_json()``, the same as the
        other four scenario-consistency checks on this class, so the calling
        harness's retry wrapper gets a chance to re-run the origin instead of
        this failure being caught locally by ``AgentPredictor.predict()`` and
        silently returning zero predictions with no further attempt.
        """
        total_probability = sum(scenario.probability for scenario in self.scenarios)
        if total_probability <= 0:
            raise ValueError("Scenario probabilities must sum to a positive value.")
        weighted_price = (
            sum(
                scenario.probability * (scenario.price_low + scenario.price_high) / 2
                for scenario in self.scenarios
            )
            / total_probability
        )

        max_horizon = max(forecast.horizon for forecast in self.forecasts)
        longest_horizon_forecast = next(f for f in self.forecasts if f.horizon == max_horizon)

        scenario_low = min(scenario.price_low for scenario in self.scenarios)
        scenario_high = max(scenario.price_high for scenario in self.scenarios)
        tolerance = max((scenario_high - scenario_low) * _SCENARIO_CONSISTENCY_TOLERANCE, 1.0)

        deviation = abs(longest_horizon_forecast.point_forecast - weighted_price)
        if deviation > tolerance:
            raise ValueError(
                f"point_forecast ({longest_horizon_forecast.point_forecast:.2f}) at horizon "
                f"{max_horizon} deviates from the probability-weighted scenario price "
                f"({weighted_price:.2f}) by {deviation:.2f}, exceeding the "
                f"{_SCENARIO_CONSISTENCY_TOLERANCE:.0%}-of-spread tolerance ({tolerance:.2f}). "
                "The model's point forecast is inconsistent with its own stated scenarios."
            )
        return self




        
    @classmethod
    def prompt_schema_json(cls) -> str:
        """Return a JSON template for use in agent instruction strings.

        Extends the base template with the ``factors`` and ``scenarios``
        blocks, so the schema embedded in the instruction always matches
        what this class actually validates.

        Returns
        -------
        str
            Indented JSON string showing the exact structure the agent must
            pass to ``set_model_response``.
        """
        quantile_entries = [{"quantile": float(q), "value": "<float>"} for q in STANDARD_QUANTILES]
        template: dict[str, object] = {
            "forecasts": [
                {
                    "horizon": "<integer — one entry per horizon from the task>",
                    "point_forecast": "<float — must equal the 0.50 quantile value>",
                    "quantiles": quantile_entries,
                    "rationale": "<string>",
                }
            ],
            "factors": [
                {
                    "name": "<string>",
                    "category": "<'macro' | 'financial' | 'geopolitical'>",
                    "tier": "<'core' | 'transitory'>",
                    "impact_score": "<'low' | 'medium' | 'high' — required for transitory, omit for core>",
                }
            ],
            "scenarios": [
                {
                    "name": "<string>",
                    "probability": "<float in [0, 1]>",
                    "price_low": "<float>",
                    "price_high": "<float — equal to price_low for a point estimate>",
                    "is_tail_case": "<true for exactly one low-probability/high-impact scenario>",
                    "stances": {"<factor name>": "<'bullish' | 'bearish' | 'neutral'>"},
                }
            ],
            "rationale": "<string, optional overall explanation>",
        }
        return json.dumps(template, indent=2)

    def to_predictions(
        self,
        *,
        task: ForecastingTask,
        context: ForecastContext,
        predictor_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Prediction]:
        """Widen outermost quantiles toward the scenario range, scaled per horizon.

        Widens (never narrows) each horizon's outermost quantiles toward the
        scenario price range, scaled by ``sqrt(horizon / max_horizon)`` so
        shorter horizons don't inherit the full longest-horizon spread. This
        can only ever increase each interval — moving the outermost quantiles
        further from their own current value cannot violate the non-decreasing
        quantile constraint already enforced at construction time.

        Point-forecast/scenario consistency is enforced separately, as
        ``_point_forecast_consistent_with_scenarios`` above — a violation there
        is retried by the calling harness rather than reaching this method.

        Also stamps ``factors`` and ``scenarios`` onto ``Prediction.metadata``
        so downstream analysis can inspect the full decomposition alongside
        the forecast.
        """
        scenario_low = min(scenario.price_low for scenario in self.scenarios)
        scenario_high = max(scenario.price_high for scenario in self.scenarios)
        lowest_quantile = min(STANDARD_QUANTILES)
        highest_quantile = max(STANDARD_QUANTILES)
        max_horizon = max(forecast.horizon for forecast in self.forecasts)

        for forecast in self.forecasts:
            scale = sqrt(forecast.horizon / max_horizon)
            for quantile_forecast in forecast.quantiles:
                if quantile_forecast.quantile == lowest_quantile and quantile_forecast.value > scenario_low:
                    quantile_forecast.value -= (quantile_forecast.value - scenario_low) * scale
                elif quantile_forecast.quantile == highest_quantile and quantile_forecast.value < scenario_high:
                    quantile_forecast.value += (scenario_high - quantile_forecast.value) * scale

        merged_metadata: dict[str, Any] = dict(metadata) if metadata is not None else {}
        merged_metadata["factors"] = [factor.model_dump() for factor in self.factors]
        merged_metadata["scenarios"] = [scenario.model_dump() for scenario in self.scenarios]

        return super().to_predictions(
            task=task,
            context=context,
            predictor_id=predictor_id,
            metadata=merged_metadata,
        )
    

_WTI_ANALYST_INSTRUCTION_SCENARIO_SCHEMA = _build_wti_analyst_instruction_scenario_schema()


# ---------------------------------------------------------------------------
# AgentConfig factories
# ---------------------------------------------------------------------------


def build_wti_basic_config(model: str = LITE_MODEL) -> AgentConfig:
    """Build a :class:`AgentConfig` with no tools.

    The agent reasons purely from the price history in the prompt payload.
    Useful as a low-cost baseline or starting point when comparing capability
    levels.

    Parameters
    ----------
    model : str
        Gemini model identifier.

    Returns
    -------
    AgentConfig
    """
    return AgentConfig(
        name="wti_analyst_basic",
        model=model,
        instruction=_WTI_ANALYST_INSTRUCTION,
    )


def build_wti_multitask_news_config(
    model: str = LITE_MODEL,
    search_model: str = LITE_MODEL,
    verifier_model: str = ADVANCED_MODEL,
    verifier_max_attempts: int = 3,
    verifier_confidence_threshold: int = 8,
) -> AgentConfig:
    """News-grounded config for the one-agent-three-tasks demo (NB3).

    Uses a task-agnostic analyst instruction; the task schema is supplied in
    the user prompt payload via :class:`~energy_oil_forecasting.tasks.WtiMultitaskPromptBuilder`.

    Parameters
    ----------
    model : str
        Model for the top-level analyst agent.
    search_model : str
        Model for the context-retrieval (web-search) sub-tool. Defaults to
        the lite model (``gemini-3.1-flash-lite-preview``) independently of ``model`` so that Gemini
        handles Google Search even when the analyst uses a different provider.
    verifier_model : str
        Model for the independent temporal-leakage verifier that audits each
        ``search_web`` result against ``cutoff_date`` before it is returned.
        Defaults to the advanced model so it doesn't share ``search_model``'s
        blind spots.
    verifier_max_attempts : int
        Maximum search-then-verify attempts before giving up and returning
        the ``[SEARCH_VERIFICATION_FAILED]`` sentinel.
    verifier_confidence_threshold : int
        Minimum verifier confidence (1-10) required to accept a result.
    """
    return AgentConfig(
        name="wti_analyst_multitask",
        model=model,
        instruction=_WTI_MULTITASK_ANALYST_INSTRUCTION,
        context_retrieval=ContextRetrievalConfig(
            enabled=True,
            instruction=_WTI_CONTEXT_RETRIEVAL_INSTRUCTION,
            search_model=search_model,
            verifier_model=verifier_model,
            verifier_max_attempts=verifier_max_attempts,
            verifier_confidence_threshold=verifier_confidence_threshold,
        ),
    )


def build_wti_news_config(
    model: str = LITE_MODEL,
    search_model: str = LITE_MODEL,
    max_output_tokens: int = 16_384,  # ← NEW — was implicitly 4096 (agent_factory.py default)
    verifier_model: str = ADVANCED_MODEL,
    verifier_max_attempts: int = 3,
    verifier_confidence_threshold: int = 8,
) -> AgentConfig:
    """Build an :class:`AgentConfig` with bounded Google Search.

    Wires a :class:`~aieng.forecasting.methods.agentic.agent_factory.ContextRetrievalConfig`
    sub-agent that enforces a temporal cutoff on every search call, preventing
    future information from contaminating historical backtests. An
    independent verifier call audits each search result against the cutoff
    before it reaches the analyst (see :class:`ContextRetrievalConfig`).

    Parameters
    ----------
    model : str
        Model for the top-level analyst agent.
    search_model : str
        Model for the context-retrieval (web-search) sub-tool. Defaults to
        the lite model (``gemini-3.1-flash-lite-preview``) independently of ``model`` so that Gemini
        handles Google Search even when the analyst uses a different provider.
    verifier_model : str
        Model for the independent temporal-leakage verifier that audits each
        ``search_web`` result against ``cutoff_date`` before it is returned.
        Defaults to the advanced model so it doesn't share ``search_model``'s
        blind spots.
    verifier_max_attempts : int
        Maximum search-then-verify attempts before giving up and returning
        the ``[SEARCH_VERIFICATION_FAILED]`` sentinel.
    verifier_confidence_threshold : int
        Minimum verifier confidence (1-10) required to accept a result.

    Returns
    -------
    AgentConfig
    """
    return AgentConfig(
        name="wti_analyst_news_scenario",
        model=model,
        instruction=_WTI_ANALYST_INSTRUCTION,
        max_output_tokens=max_output_tokens,  # ← NEW
        context_retrieval=ContextRetrievalConfig(
            enabled=True,
            instruction=_WTI_CONTEXT_RETRIEVAL_INSTRUCTION,
            search_model=search_model,
            verifier_model=verifier_model,
            verifier_max_attempts=verifier_max_attempts,
            verifier_confidence_threshold=verifier_confidence_threshold,
        ),
    )

def build_wti_news_contrarian_config(
    model: str = LITE_MODEL,
    search_model: str = LITE_MODEL,
    max_output_tokens: int = 16_384,
    verifier_model: str = ADVANCED_MODEL,
    verifier_max_attempts: int = 3,
    verifier_confidence_threshold: int = 8,
) -> AgentConfig:
    """Like :func:`build_wti_news_config`, but instructs the search sub-agent to
    surface disagreeing sources and search for historical precedent/analogues."""
    return AgentConfig(
        name="wti_analyst_news_contrarian_scenario",
        model=model,
        instruction=_WTI_ANALYST_INSTRUCTION,
        max_output_tokens=max_output_tokens,  # ← add this line
        context_retrieval=ContextRetrievalConfig(
            enabled=True,
            instruction=_WTI_CONTRARIAN_CONTEXT_RETRIEVAL_INSTRUCTION,
            search_model=search_model,
            verifier_model=verifier_model,
            verifier_max_attempts=verifier_max_attempts,
            verifier_confidence_threshold=verifier_confidence_threshold,
        ),
    )


def build_wti_news_factors_v2_config(
    model: str = LITE_MODEL,
    search_model: str = LITE_MODEL,
    max_output_tokens: int = 16_384,
    verifier_model: str = ADVANCED_MODEL,
    verifier_max_attempts: int = 3,
    verifier_confidence_threshold: int = 8,
) -> AgentConfig:
    """Like :func:`build_wti_news_contrarian_config`, but tests a two-tier
    core/transitory factor framework in scenario decomposition and a more
    open, non-topic-anchored search strategy — a prompt-only experiment,
    no output schema changes."""
    return AgentConfig(
        name="wti_analyst_news_factors_v2",
        model=model,
        instruction=_WTI_ANALYST_INSTRUCTION_FACTORS_V2,
        max_output_tokens=max_output_tokens,
        context_retrieval=ContextRetrievalConfig(
            enabled=True,
            instruction=_WTI_FACTORS_V2_CONTEXT_RETRIEVAL_INSTRUCTION,
            search_model=search_model,
            verifier_model=verifier_model,
            verifier_max_attempts=verifier_max_attempts,
            verifier_confidence_threshold=verifier_confidence_threshold,
        ),
    )

def build_wti_news_scenario_schema_config(
    model: str = LITE_MODEL,
    search_model: str = LITE_MODEL,
    max_output_tokens: int = 16_384,
    verifier_model: str = ADVANCED_MODEL,
    verifier_max_attempts: int = 3,
    verifier_confidence_threshold: int = 8,
) -> AgentConfig:
    """Like :func:`build_wti_news_factors_v2_config`, but the core/transitory
    scenario decomposition is a real, schema-validated output (see
    :class:`WtiScenarioForecastOutput`) instead of free text inside
    `rationale` — pair with :func:`build_wti_scenario_schema_predictor`,
    not :func:`build_wti_agent_predictor`."""
    return AgentConfig(
        name="wti_analyst_news_scenario_schema",
        model=model,
        instruction=_WTI_ANALYST_INSTRUCTION_SCENARIO_SCHEMA,
        max_output_tokens=max_output_tokens,
        context_retrieval=ContextRetrievalConfig(
            enabled=True,
            instruction=_WTI_FACTORS_V2_CONTEXT_RETRIEVAL_INSTRUCTION,
            search_model=search_model,
            verifier_model=verifier_model,
            verifier_max_attempts=verifier_max_attempts,
            verifier_confidence_threshold=verifier_confidence_threshold,
        ),
    )

def build_wti_code_exec_config(
    model: str = LITE_MODEL,
    search_model: str = LITE_MODEL,
    max_output_tokens: int = 16_384,
    verifier_model: str = ADVANCED_MODEL,
    verifier_max_attempts: int = 3,
    verifier_confidence_threshold: int = 8,
) -> AgentConfig:
    """Build an :class:`AgentConfig` with E2B code execution and forecasting skills.

    Combines bounded Google Search (temporal cutoff enforced) with E2B sandbox
    code execution and two forecasting skills:

    - ``statistical-analysis``: diagnostic patterns for the payload data
      (vol regime, anomaly detection, adaptive trend window).
    - ``trend-projection``: linear trend fit, CI calibration, and plausibility
      guard using the window determined by statistical-analysis.

    Parameters
    ----------
    model : str
        Model for the top-level analyst agent.
    search_model : str
        Model for the context-retrieval (web-search) sub-tool. Defaults to
        the lite model (``gemini-3.1-flash-lite-preview``) independently of ``model`` so that Gemini
        handles Google Search even when the analyst uses a different provider.
    max_output_tokens : int, default=16_384
        Maximum tokens per model response.  The default is set well above
        LiteLLM's OpenAI-compatible endpoint default of 4096, which is not
        enough for Claude to write a complete ``run_code`` Python script in a
        single function call — causing repeated retries with empty arguments.
    verifier_model : str
        Model for the independent temporal-leakage verifier that audits each
        ``search_web`` result against ``cutoff_date`` before it is returned.
        Defaults to the advanced model so it doesn't share ``search_model``'s
        blind spots.
    verifier_max_attempts : int
        Maximum search-then-verify attempts before giving up and returning
        the ``[SEARCH_VERIFICATION_FAILED]`` sentinel.
    verifier_confidence_threshold : int
        Minimum verifier confidence (1-10) required to accept a result.

    Returns
    -------
    AgentConfig
    """
    return AgentConfig(
        name="wti_analyst_code",
        model=model,
        instruction=_WTI_ANALYST_INSTRUCTION + _CODE_EXEC_SKILLS_SUPPLEMENT,
        max_output_tokens=max_output_tokens,
        context_retrieval=ContextRetrievalConfig(
            enabled=True,
            instruction=_WTI_CONTEXT_RETRIEVAL_INSTRUCTION,
            search_model=search_model,
            verifier_model=verifier_model,
            verifier_max_attempts=verifier_max_attempts,
            verifier_confidence_threshold=verifier_confidence_threshold,
        ),
        code_execution=CodeExecutionConfig(enabled=True),
        skills_dirs=[
            _SKILLS_ROOT / "statistical-analysis",
            _SKILLS_ROOT / "trend-projection",
        ],
    )


def build_wti_tool_config(
    model: str = LITE_MODEL,
    search_model: str = LITE_MODEL,
    *,
    data_service: DataService | None = None,
    num_samples: int = 200,
    verifier_model: str = ADVANCED_MODEL,
    verifier_max_attempts: int = 3,
    verifier_confidence_threshold: int = 8,
) -> AgentConfig:
    """Build an :class:`AgentConfig` with a conventional statistical forecast tool.

    This is the fourth analyst capability level. It combines bounded Google
    Search (temporal cutoff enforced) with a
    :class:`~aieng.forecasting.methods.agentic.forecast_tool.ForecastTool`
    that runs AutoARIMA on the WTI series. In contrast to
    :func:`build_wti_code_exec_config` — which gives the agent open-ended code
    execution — this path exposes a rigid, pre-specified tool, trading
    flexibility for control and reproducibility.

    Parameters
    ----------
    model : str
        Model for the top-level analyst agent.
    search_model : str
        Model for the context-retrieval (web-search) sub-tool. Defaults to
        the lite model (``gemini-3.1-flash-lite-preview``) independently of ``model`` so that Gemini
        handles Google Search even when the analyst uses a different provider.
    data_service : DataService or None
        Pre-populated data service with the WTI series registered. When
        ``None``, one is constructed via
        :func:`~energy_oil_forecasting.data.build_wti_service` (cache-backed).
        Series data is read by the tool but never enters the LLM context.
    num_samples : int, default=200
        Monte Carlo sample count for AutoARIMA. Kept modest to bound agent
        latency, since AutoARIMA can be slow per origin.
    verifier_model : str
        Model for the independent temporal-leakage verifier that audits each
        ``search_web`` result against ``cutoff_date`` before it is returned.
        Defaults to the advanced model so it doesn't share ``search_model``'s
        blind spots.
    verifier_max_attempts : int
        Maximum search-then-verify attempts before giving up and returning
        the ``[SEARCH_VERIFICATION_FAILED]`` sentinel.
    verifier_confidence_threshold : int
        Minimum verifier confidence (1-10) required to accept a result.

    Returns
    -------
    AgentConfig
    """
    service = data_service if data_service is not None else build_wti_service()
    forecast_tool = ForecastTool(service, predictor=DartsAutoARIMAPredictor(num_samples=num_samples))

    return AgentConfig(
        name="wti_analyst_tool",
        model=model,
        instruction=_WTI_ANALYST_INSTRUCTION + _FORECAST_TOOL_SUPPLEMENT,
        context_retrieval=ContextRetrievalConfig(
            enabled=True,
            instruction=_WTI_CONTEXT_RETRIEVAL_INSTRUCTION,
            search_model=search_model,
            verifier_model=verifier_model,
            verifier_max_attempts=verifier_max_attempts,
            verifier_confidence_threshold=verifier_confidence_threshold,
        ),
        function_tools=[forecast_tool.as_function_tool()],
    )


# ---------------------------------------------------------------------------
# Predictor convenience factory
# ---------------------------------------------------------------------------


def build_wti_agent_predictor(config: AgentConfig) -> AgentPredictor:
    """Wrap an :class:`AgentConfig` in an :class:`AgentPredictor`.

    Uses :class:`WtiPriceForecastPromptBuilder` and
    :class:`~aieng.forecasting.methods.agentic.outputs.ContinuousAgentForecastOutput`
    as the output schema.

    Parameters
    ----------
    config : AgentConfig
        Any of the configs produced by :func:`build_wti_basic_config`,
        :func:`build_wti_news_config`, or :func:`build_wti_code_exec_config`.

    Returns
    -------
    AgentPredictor
    """
    return AgentPredictor(
        agent_config=config,
        prompt_builder=WtiPriceForecastPromptBuilder(),
        output_schema=ContinuousAgentForecastOutput,
    )


def build_wti_scenario_schema_predictor(config: AgentConfig) -> AgentPredictor:
    """Like :func:`build_wti_agent_predictor`, but validates against
    :class:`WtiScenarioForecastOutput` instead of the plain
    ``ContinuousAgentForecastOutput`` — use with
    :func:`build_wti_news_scenario_schema_config`."""
    return AgentPredictor(
        agent_config=config,
        prompt_builder=WtiPriceForecastPromptBuilder(),
        output_schema=WtiScenarioForecastOutput,
    )


# ---------------------------------------------------------------------------
# Lazy root_agent for `adk web` interactive use
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Lazy root_agent for `adk web` interactive use
# ---------------------------------------------------------------------------


def __getattr__(name: str) -> Any:
    """Expose ``root_agent`` lazily for schema-free interactive use via ``adk web``."""
    if name == "root_agent":
        return build_adk_agent(build_wti_basic_config())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
