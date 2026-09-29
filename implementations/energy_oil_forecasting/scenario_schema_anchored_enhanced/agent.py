"""ARIMA-anchored, memory-enhanced WTI Scenario Schema agent.

Combines the two independent extensions of the base Scenario Schema agent
(see ``scenario_schema_core.agent``) this module's siblings ship separately:

1. From :mod:`energy_oil_forecasting.scenario_schema_anchored`: a
   deterministic AutoARIMA forecast, computed in Python and fed into the
   prompt as ``arima_anchor`` — a statistical baseline the LLM reasons from
   rather than inventing price levels independently. Python, not the LLM,
   owns the final point_forecast/quantiles (see
   :mod:`energy_oil_forecasting.scenario_schema_anchored_enhanced.predictor`).
2. From :mod:`energy_oil_forecasting.scenario_schema_anchored_enhanced.memory`:
   the last few scenario frameworks this predictor produced at earlier
   origins, fed into the prompt as ``prior_frameworks`` — unlike
   ``scenario_schema_enhanced``, this field is actually populated (that
   module's own field was aspirational prompt text with nothing behind it;
   see this package's memory module for the real read/write path), so
   scenario names and thematic structure can persist across a backtest run
   instead of the LLM reinventing them from a blank slate every call.
"""

from aieng.forecasting.methods.agentic.agent_factory import (
    AgentConfig,
    ContextRetrievalConfig,
)
from aieng.forecasting.models import ADVANCED_MODEL, LITE_MODEL
from energy_oil_forecasting.scenario_schema_anchored_enhanced.schema import WtiMemoryScenarioForecastOutput
from energy_oil_forecasting.scenario_schema_core import WTI_FACTORS_CONTEXT_RETRIEVAL_INSTRUCTION


def _build_wti_analyst_instruction_scenario_schema_anchored_enhanced() -> str:
    """Scenario Schema instruction aware of both the ARIMA anchor and prior memory.

    Same factor-tier / scenario-decomposition structure as the base Scenario
    Schema, plus:

    - the ``arima_anchor`` field description from
      ``scenario_schema_anchored`` (reason from the statistical baseline,
      don't invent price levels independently of it), and
    - the "Memory and Framework Consistency" section from
      ``scenario_schema_enhanced`` (read prior frameworks before generating
      new ones, signal explicitly when something changes rather than
      silently renaming it).
    """
    schema = WtiMemoryScenarioForecastOutput.prompt_schema_json()
    return (
        "## Role\n\n"
        "You are an expert WTI crude oil market analyst. You produce calibrated "
        "probabilistic price forecasts for WTI crude oil futures, grounded in "
        "supply/demand fundamentals, geopolitical risk, and historical price dynamics.\n\n"
        "## Memory and Framework Consistency\n\n"
        "IMPORTANT: Before generating new scenarios, check `prior_frameworks` in "
        "the payload. If it is non-empty, review:\n"
        "- Which factors were identified in prior forecasts (core vs transitory)\n"
        "- How scenarios were structured and named in prior periods\n"
        "- Whether current market signals suggest a shift in baseline assumptions\n\n"
        "Use this memory to:\n"
        "1. Reuse the SAME scenario and factor names across origins when the "
        "underlying theme is unchanged — do not rename a continuing scenario just "
        "because you are generating it fresh; a reader tracking this forecast over "
        "time relies on stable names to see how probabilities and ranges evolve.\n"
        "2. Explicitly signal when a new factor becomes material (newly core or "
        "transitory) or a prior one drops out, rather than silently reshuffling "
        "the set.\n"
        "3. Explain any large probability weight shifts across scenarios relative "
        "to the prior framework.\n"
        "4. For EVERY scenario, set `continues_prior_scenario` to true and "
        "`continued_from` to the EXACT scenario name from `prior_frameworks` it "
        "continues — or set `continues_prior_scenario` to false (and omit "
        "`continued_from`) for a genuinely new scenario with no prior counterpart. "
        "This is a required declaration, not an optional note: every scenario, on "
        "every origin including the first (when `prior_frameworks` is empty, every "
        "scenario is necessarily new), must set this explicitly.\n\n"
        "If `prior_frameworks` is empty (no memory yet, e.g. the first origin), "
        "proceed with standard analysis. Memory is a consistency aid, not a "
        "constraint — your forecast must reflect current market facts, not "
        "historical continuity for its own sake; a genuine regime change should "
        "produce genuinely new scenarios (`continues_prior_scenario: false`), "
        "explicitly flagged as such rather than forced to fit the old names.\n\n"
        "## Forecasting contract\n\n"
        "You will receive a JSON payload containing:\n"
        "- `task`: the task identifier\n"
        "- `as_of`: the forecast origin date in YYYY-MM-DD format\n"
        "- `horizons`: a list of integer horizon steps (business days ahead)\n"
        "- `standard_quantiles`: the exact quantile levels you must produce\n"
        "- `target_summary`: last close price, 52-week range, and observation count\n"
        "- `target_history_csv`: WTI daily close history (recent 6 months daily, "
        "older history as weekly averages)\n"
        "- `arima_anchor`: a deterministic AutoARIMA point forecast and quantile grid "
        "per horizon, fit on price history alone (no news, no judgment). Treat this as "
        "your statistical baseline — the price the market would imply with no new "
        "information. Your scenarios should be reasoned deviations from it, justified "
        "by what you find in search, not numbers invented independently of it. A "
        "scenario's price range may still land far from the anchor when the evidence "
        "genuinely supports it (e.g. a live supply shock) — just make that justification "
        "explicit in the scenario's rationale.\n"
        "- `prior_frameworks`: (list, possibly empty) your own factors/scenarios from "
        "earlier origins, oldest first — see Memory and Framework Consistency above.\n\n"
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
        "typically 20% probability or less. A 35% scenario is not a tail case "
        "no matter how extreme its price range sounds — if your most extreme "
        "scenario is that likely, it belongs in your main narrative, not tagged "
        "as tail. Do not use the tail-case flag as a label for your highest-impact "
        "scenario regardless of its probability; use it only when the probability "
        "itself is genuinely low. For each scenario, give "
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


_WTI_ANALYST_INSTRUCTION_SCENARIO_SCHEMA_ANCHORED_ENHANCED = (
    _build_wti_analyst_instruction_scenario_schema_anchored_enhanced()
)


def build_wti_news_scenario_schema_anchored_enhanced_config(
    model: str = LITE_MODEL,
    *,
    log_returns: bool = True,
    search_model: str = LITE_MODEL,
    max_output_tokens: int = 16_384,
    verifier_model: str = ADVANCED_MODEL,
    verifier_max_attempts: int = 3,
    verifier_confidence_threshold: int = 8,
) -> AgentConfig:
    """Build config for the ARIMA-anchored, memory-enhanced Scenario Schema agent.

    Pairs with :func:`energy_oil_forecasting.scenario_schema_anchored_enhanced.predictor.build_wti_scenario_schema_anchored_enhanced_predictor`.

    Parameters
    ----------
    model : str
        Model for the analyst agent.
    log_returns : bool, default=True
        Passed through to the predictor's ARIMA anchor — see
        :class:`~energy_oil_forecasting.scenario_schema_anchored.predictor.ScenarioSchemaAnchoredPredictor`.
        Must match what the predictor is constructed with, or the cache name
        will not describe what actually ran (same "_logret" suffix
        discipline as the plain anchored config, below).
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
        # Same "_logret" suffix discipline as scenario_schema_anchored's
        # config builder, and for the same reason: predictor_id is the
        # cache key, and it cannot tell the anchor specification changed
        # underneath it, nor -- here -- that the memory store's contents
        # differ from the plain (non-memory) anchored agent's. A distinct
        # name keeps this variant's memory file and prediction cache both
        # addressable on their own.
        name="wti_analyst_news_scenario_schema_anchored_enhanced_logret"
        if log_returns
        else "wti_analyst_news_scenario_schema_anchored_enhanced",
        model=model,
        instruction=_WTI_ANALYST_INSTRUCTION_SCENARIO_SCHEMA_ANCHORED_ENHANCED,
        max_output_tokens=max_output_tokens,
        temperature=0.0,  # pin determinism, same rationale as the anchored and enhanced configs
        context_retrieval=ContextRetrievalConfig(
            enabled=True,
            instruction=WTI_FACTORS_CONTEXT_RETRIEVAL_INSTRUCTION,
            search_model=search_model,
            verifier_model=verifier_model,
            verifier_max_attempts=verifier_max_attempts,
            verifier_confidence_threshold=verifier_confidence_threshold,
        ),
    )


__all__ = ["build_wti_news_scenario_schema_anchored_enhanced_config"]
