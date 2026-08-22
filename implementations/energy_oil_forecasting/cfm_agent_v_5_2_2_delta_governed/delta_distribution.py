"""Empirical h-day price-delta distribution — re-exported from the shared module.

Moved to :mod:`energy_oil_forecasting.price_deltas` so
:mod:`energy_oil_forecasting.scenario_schema_anchored` can share the same
grounding logic without importing this whole CFM-specific package (its
``AgentConfig`` builders, tools, predictor) just for one generic function.
This module re-exports the same name so existing imports keep working
unchanged.
"""

from __future__ import annotations

from energy_oil_forecasting.price_deltas import compute_horizon_delta_percentiles


__all__ = ["compute_horizon_delta_percentiles"]
