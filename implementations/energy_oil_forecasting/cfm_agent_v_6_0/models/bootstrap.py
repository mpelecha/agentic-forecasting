"""Historical Bootstrap Return (specification Section 1.3).

All eligible overlapping blocks of ``block_length_business_days`` daily returns
are sampled uniformly with replacement from the component generator,
concatenated, and truncated to the simulation length. Sampled returns are used
as observed: the history's drift and the compounding drag are part of the
forecast (DD-08).
"""

from __future__ import annotations

import numpy as np
from energy_oil_forecasting.cfm_agent_v_6_0.models.paths import (
    ComponentFailure,
    ReturnPathComponent,
    overlapping_block_paths,
)
from energy_oil_forecasting.cfm_agent_v_6_0.schemas import ModelTrainingData


class HistoricalBootstrapReturn(ReturnPathComponent):
    """Block bootstrap of historical daily simple returns."""

    component_name = "bootstrap"

    def _simulate_returns(self) -> np.ndarray:
        settings = self.settings
        returns = self.history.returns
        block = settings.block_length_business_days
        minimum = block + settings.min_training_rows_margin
        self.training_data = ModelTrainingData(
            **self._base_training_data(),
            minimum_returns_required=minimum,
            eligible_blocks=max(0, len(returns) - block + 1),
            block_length=block,
            notes=(
                "Overlapping blocks of daily simple returns sampled uniformly with replacement "
                "from the component generator; returns are used as observed."
            ),
        )
        if len(returns) < minimum:
            raise ComponentFailure(
                "insufficient_history",
                f"{len(returns)} daily returns available; the bootstrap needs at least {minimum}",
            )
        return overlapping_block_paths(
            self.rng,
            returns,
            n_paths=settings.model_num_paths,
            path_length=settings.simulation_length_business_days,
            block=block,
        )


__all__ = ["HistoricalBootstrapReturn"]
