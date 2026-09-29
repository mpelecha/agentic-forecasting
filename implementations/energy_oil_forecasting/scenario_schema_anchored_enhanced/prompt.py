"""Prompt builder that embeds both the ARIMA anchor and prior scenario memory."""

from __future__ import annotations

import json

from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.task import ForecastingTask
from energy_oil_forecasting.scenario_schema_anchored.prompt import AnchoredPromptBuilder


class AnchoredMemoryPromptBuilder(AnchoredPromptBuilder):
    """Adds a ``prior_frameworks`` block on top of :class:`AnchoredPromptBuilder`.

    Both ``arima_anchor`` (inherited) and ``prior_frameworks`` (added here)
    are mutable and stashed by
    :class:`~energy_oil_forecasting.scenario_schema_anchored_enhanced.predictor.ScenarioSchemaAnchoredEnhancedPredictor`
    immediately before each call to the inner :class:`AgentPredictor` — the
    same "compute, stash, then call" pattern ``AnchoredPromptBuilder`` itself
    documents.
    """

    prior_frameworks: list[dict] = []

    def __call__(self, *, task: ForecastingTask, context: ForecastContext) -> str:
        payload = json.loads(super().__call__(task=task, context=context))
        payload["prior_frameworks"] = self.prior_frameworks
        return json.dumps(payload, indent=2)


__all__ = ["AnchoredMemoryPromptBuilder"]
