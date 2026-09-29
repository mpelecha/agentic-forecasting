"""Standalone WTI scenario schema/prompt-builder — no dependency on analyst_agent.

Vendored (not imported) from ``energy_oil_forecasting.analyst_agent.agent`` so
that ``scenario_schema_anchored`` / ``scenario_schema_anchored_enhanced`` do
not depend on that 1,400+ line shared module, which is used and evolved by
many other predictors in this project. This file only ever needs to change
when the scenario-schema family itself changes, not when unrelated agents do.

Depends only on the shared ``aieng.forecasting`` framework and pydantic/pandas
— nothing else in ``energy_oil_forecasting``.
"""

from __future__ import annotations

import json
from math import isfinite, sqrt
from typing import Any, Literal

import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES, Prediction
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic import ContinuousAgentForecastOutput
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Context-retrieval (web-search) instruction
# ---------------------------------------------------------------------------

WTI_FACTORS_CONTEXT_RETRIEVAL_INSTRUCTION = """\
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
# History compression
# ---------------------------------------------------------------------------


def compress_history(df: pd.DataFrame) -> str:
    """Compress daily price history to stay within context limits.

    Three-tier compression, most granular near the forecast origin:
    - last 63 trading days: daily bars
    - 63 trading days to 1 year back: weekly averages
    - older than 1 year: quarterly averages

    The CSV header is ``date,close``.

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
    max_date = df["timestamp"].max()

    daily_cutoff = max_date - pd.tseries.offsets.BDay(63)
    weekly_cutoff = max_date - pd.DateOffset(years=1)

    daily = df[df["timestamp"] >= daily_cutoff].copy()
    weekly_band = df[(df["timestamp"] >= weekly_cutoff) & (df["timestamp"] < daily_cutoff)].copy()
    quarterly_band = df[df["timestamp"] < weekly_cutoff].copy()

    rows: list[str] = ["date,close"]

    if not quarterly_band.empty:
        quarterly_indexed = quarterly_band.set_index("timestamp")["value"]
        quarterly: pd.Series = quarterly_indexed.resample("QE").mean().dropna()
        for date, val in quarterly.items():
            rows.append(f"{date.date()},{val:.2f}")

    if not weekly_band.empty:
        weekly_indexed = weekly_band.set_index("timestamp")["value"]
        weekly: pd.Series = weekly_indexed.resample("W").mean().dropna()
        for date, val in weekly.items():
            rows.append(f"{date.date()},{val:.2f}")

    for _, row in daily.iterrows():
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
        ``"core"`` for themes durable enough to plausibly matter in five
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
        description="'core' for five-year-plus durable themes; 'transitory' for situational developments."
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
        forecast's longest horizon.
    price_high : float
        Upper end of this scenario's implied price range. Should exceed
        ``price_low`` by a meaningful margin to reflect genuine within-
        scenario uncertainty.
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
SCENARIO_CONSISTENCY_TOLERANCE = 0.15


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
        tolerance = max((scenario_high - scenario_low) * SCENARIO_CONSISTENCY_TOLERANCE, 1.0)

        deviation = abs(longest_horizon_forecast.point_forecast - weighted_price)
        if deviation > tolerance:
            raise ValueError(
                f"point_forecast ({longest_horizon_forecast.point_forecast:.2f}) at horizon "
                f"{max_horizon} deviates from the probability-weighted scenario price "
                f"({weighted_price:.2f}) by {deviation:.2f}, exceeding the "
                f"{SCENARIO_CONSISTENCY_TOLERANCE:.0%}-of-spread tolerance ({tolerance:.2f}). "
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
                    "price_high": "<float — upper end; must exceed price_low by a meaningful margin, reflecting genuine uncertainty within this scenario, not a point estimate>",
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


__all__ = [
    "SCENARIO_CONSISTENCY_TOLERANCE",
    "WTI_FACTORS_CONTEXT_RETRIEVAL_INSTRUCTION",
    "WtiFactor",
    "WtiPriceForecastPromptBuilder",
    "WtiScenarioCard",
    "WtiScenarioForecastOutput",
    "compress_history",
]
