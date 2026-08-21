"""Enhanced WTI Scenario Schema agent with memory and structured output persistence.

Extends the basic Scenario Schema agent (see analyst_agent.agent.py) with:

1. **Memory capability**: Agent reads previous scenario frameworks before each
   new run, allowing it to track thematic consistency and detect shifts in
   baseline assumptions.

2. **Explicit output persistence**: Factors and scenarios are not only generated
   but also persisted in the prediction's metadata so they can be retrieved
   post-backtest without requiring Langfuse traces.

3. **Outcome tracking**: Records which scenarios actually occurred (scaffold for
   future calibration cycles).

The output schema and factor-tier logic remain identical to the base Scenario
Schema — this module focuses on *persistence* and *memory*, not algorithmic change.
"""

from typing import Any

from aieng.forecasting.methods.agentic.agent_factory import (
    AgentConfig,
    ContextRetrievalConfig,
)
from aieng.forecasting.methods.agentic import AgentPredictor
from aieng.forecasting.models import ADVANCED_MODEL, LITE_MODEL
from energy_oil_forecasting.analyst_agent.agent import (
    WtiPriceForecastPromptBuilder,
    WtiScenarioForecastOutput,
    _WTI_FACTORS_V2_CONTEXT_RETRIEVAL_INSTRUCTION,
)


def _build_wti_analyst_instruction_scenario_schema_enhanced() -> str:
    """Enhanced variant of Scenario Schema instruction with memory and persistence.

    Adds two sections to the base instruction:
    1. Memory reading: Agent reads previous scenario frameworks to assess consistency
    2. Output metadata: Reminder to ensure json_response is properly persisted
    """
    schema = WtiScenarioForecastOutput.prompt_schema_json()
    return (
        "## Role\n\n"
        "You are an expert WTI crude oil market analyst. You produce calibrated "
        "probabilistic price forecasts for WTI crude oil futures, grounded in "
        "supply/demand fundamentals, geopolitical risk, and historical price dynamics.\n\n"
        "## Memory and Framework Consistency\n\n"
        "IMPORTANT: Before generating new scenarios, check if previous scenario "
        "frameworks are available in the context. If provided, review:\n"
        "- Which factors were identified in prior forecasts (core vs transitory)\n"
        "- How scenarios were structured and weighted in prior periods\n"
        "- Whether current market signals suggest a shift in baseline assumptions\n\n"
        "Use this memory to:\n"
        "1. Maintain thematic continuity when factors remain structurally relevant\n"
        "2. Explicitly signal when a new factor becomes material (newly core or transitory)\n"
        "3. Explain any probability weight shifts across scenarios\n"
        "4. Track which prior scenarios actually occurred (outcome recording)\n\n"
        "If no prior frameworks are available, proceed with standard analysis. "
        "Memory is a consistency aid, not a constraint — your forecast must reflect "
        "current market facts, not historical continuity for its own sake.\n\n"
        "## Forecasting contract\n\n"
        "You will receive a JSON payload containing:\n"
        "- `task`: the task identifier\n"
        "- `as_of`: the forecast origin date in YYYY-MM-DD format\n"
        "- `horizons`: a list of integer horizon steps (business days ahead)\n"
        "- `standard_quantiles`: the exact quantile levels you must produce\n"
        "- `target_summary`: last close price, 52-week range, and observation count\n"
        "- `target_history_csv`: WTI daily close history (recent 6 months daily, "
        "older history as weekly averages)\n"
        "- `prior_frameworks`: (optional) JSON array of previous scenario frameworks\n\n"
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
        "**PERSISTENCE**: Your `set_model_response` call will be captured in the "
        "prediction metadata as `json_response`. Downstream consumers will extract "
        "factors and scenarios directly from this metadata — ensure the JSON is "
        "well-formed and complete.\n\n"
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
        "financial, or geopolitical factors that do NOT meet that five-year-plus "
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
        "not a milder variant of your main narrative. For each scenario, give "
        "`price_low` and `price_high` as a genuine plausible price range under "
        "that scenario, not a single point — even a confident scenario has "
        "some width to its outcome; collapsing `price_low` to equal "
        "`price_high` is a modeling error, not a valid choice.\n\n"
        "Your final quantile grid must be consistent with the SPREAD across "
        "your scenarios' `price_low`/`price_high` ranges — not just your "
        "single most likely one. If your scenarios disagree by $10+, a narrow "
        "interval is not consistent with your own analysis.\n\n"
        "Use the `rationale` field for a brief narrative summary of your "
        "reasoning — the structured detail belongs in `factors` and "
        "`scenarios`, not repeated at length in prose."
    )


_WTI_ANALYST_INSTRUCTION_SCENARIO_SCHEMA_ENHANCED = (
    _build_wti_analyst_instruction_scenario_schema_enhanced()
)


def build_wti_news_scenario_schema_enhanced_config(
    model: str = LITE_MODEL,
    search_model: str = LITE_MODEL,
    max_output_tokens: int = 16_384,
    verifier_model: str = ADVANCED_MODEL,
    verifier_max_attempts: int = 3,
    verifier_confidence_threshold: int = 8,
) -> AgentConfig:
    """Build config for the enhanced Scenario Schema agent with memory.

    Same structure as the base Scenario Schema, but with:
    - Memory reading instruction (agent reads prior frameworks)
    - Output persistence emphasis (json_response captured to metadata)

    Parameters
    ----------
    model : str
        Model for the analyst agent.
    search_model : str
        Model for the context-retrieval (web-search) sub-tool.
    max_output_tokens : int, default=16_384
        Maximum tokens per model response.
    verifier_model : str
        Model for the independent temporal-leakage verifier.
    verifier_max_attempts : int
        Maximum search-then-verify attempts.
    verifier_confidence_threshold : int
        Minimum verifier confidence (1-10) required to accept a result.

    Returns
    -------
    AgentConfig
    """
    return AgentConfig(
        name="wti_analyst_news_scenario_schema_enhanced",
        model=model,
        instruction=_WTI_ANALYST_INSTRUCTION_SCENARIO_SCHEMA_ENHANCED,
        max_output_tokens=max_output_tokens,
        temperature=0.0,  # 2026.08.20 — pin determinism for reproducible scenarios
        context_retrieval=ContextRetrievalConfig(
            enabled=True,
            instruction=_WTI_FACTORS_V2_CONTEXT_RETRIEVAL_INSTRUCTION,
            search_model=search_model,
            verifier_model=verifier_model,
            verifier_max_attempts=verifier_max_attempts,
            verifier_confidence_threshold=verifier_confidence_threshold,
        ),
    )


def build_wti_scenario_schema_enhanced_predictor(config: AgentConfig) -> AgentPredictor:
    """Wrap the enhanced Scenario Schema config in an AgentPredictor.

    Uses :class:`WtiPriceForecastPromptBuilder` and
    :class:`WtiScenarioForecastOutput` as the output schema.

    Parameters
    ----------
    config : AgentConfig
        Config from :func:`build_wti_news_scenario_schema_enhanced_config`.

    Returns
    -------
    AgentPredictor
    """
    return AgentPredictor(
        agent_config=config,
        prompt_builder=WtiPriceForecastPromptBuilder(),
        output_schema=WtiScenarioForecastOutput,
    )
