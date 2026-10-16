"""Lightweight LightGBM Return (specification Section 1.5).

Features are the lagged WTI daily returns in ``return_lags`` followed by the
covariate values from the aligned frame, in ``covariate_series_ids`` order. The
data layer already lags every covariate by one business day, so the covariate
row dated ``t`` is used unchanged as a feature for the return realised on ``t``.

Future-covariate rule
---------------------
During the recursive simulation the WTI lag features are fed by observed and
already simulated returns. A covariate whose id contains ``"contango"`` or
``"vix"`` is a level and is held at its latest cutoff-safe served value; every
other covariate is a return and is set to zero (Section 1.5 table).

Missing-data policy
-------------------
A configured covariate absent from the data service fails the component with
``covariate_missing``. Gaps are forward-filled up to
``covariate_forward_fill_limit`` consecutive rows and rows still missing a value
are dropped; fewer than ``max(return_lags) + block_length_business_days +
min_training_rows_margin`` aligned rows fails with ``insufficient_aligned_rows``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from energy_oil_forecasting.cfm_agent_v_6_0.models.paths import (
    ComponentFailure,
    ReturnPathComponent,
    lag_matrix,
    overlapping_block_paths,
    recursive_lag_simulation,
)
from energy_oil_forecasting.cfm_agent_v_6_0.schemas import ModelTrainingData
from lightgbm import LGBMRegressor


LEVEL_COVARIATE_MARKERS: tuple[str, ...] = ("contango", "vix")


def is_level_covariate(series_id: str) -> bool:
    """Whether a covariate is a level held at its latest value in the simulation."""
    lowered = series_id.lower()
    return any(marker in lowered for marker in LEVEL_COVARIATE_MARKERS)


class LightGBMReturn(ReturnPathComponent):
    """LightGBM regression on lagged WTI returns and covariates with residual block innovations."""

    component_name = "lightgbm"

    def _simulate_returns(self) -> np.ndarray:
        settings = self.settings
        history = self.history
        lags = list(settings.return_lags)
        max_lag = max(lags)
        block = settings.block_length_business_days
        minimum = max_lag + block + settings.min_training_rows_margin
        covariate_ids = list(history.covariate_ids)
        base = self._base_training_data()
        alignment = {
            "covariates": covariate_ids,
            "covariate_observations": {
                series_id: history.covariates[series_id].observations for series_id in covariate_ids
            },
            "rows_before_alignment": history.rows_before_alignment,
            "covariate_fill_counts": dict(history.covariate_fill_counts),
            "rows_dropped": history.rows_dropped,
            "rows_remaining": history.rows_remaining,
        }
        if history.missing_covariates:
            self.training_data = ModelTrainingData(
                **base,
                **alignment,
                minimum_returns_required=minimum,
                block_length=block,
                return_lags=lags,
                notes="A configured covariate is absent from the data service.",
            )
            raise ComponentFailure(
                "covariate_missing",
                f"configured covariates absent from the data service: {history.missing_covariates}",
            )
        if history.rows_remaining < minimum:
            self.training_data = ModelTrainingData(
                **base,
                **alignment,
                minimum_returns_required=minimum,
                block_length=block,
                return_lags=lags,
                notes="The aligned frame after forward-fill and row dropping is below the minimum.",
            )
            raise ComponentFailure(
                "insufficient_aligned_rows",
                f"{history.rows_remaining} aligned rows after fill and drop; LightGBM needs at least {minimum}",
            )
        returns = history.returns
        aligned = history.aligned[covariate_ids]
        # The return realised on date position j is returns[j - 1]; a training row needs every
        # lagged return observed (j - 1 >= max_lag) and a complete covariate row dated j.
        positions = history.dates.get_indexer(aligned.index)
        keep = positions - 1 >= max_lag
        rows = positions[keep] - 1
        covariate_values = aligned.to_numpy(dtype=float)[keep]
        feature_names = [f"wti_return_lag_{lag}" for lag in lags] + covariate_ids
        features = pd.DataFrame(
            np.column_stack([lag_matrix(returns, lags, rows), covariate_values]), columns=feature_names
        )
        target = returns[rows]
        model = LGBMRegressor(
            objective="regression",
            n_estimators=settings.lightgbm_n_estimators,
            learning_rate=settings.lightgbm_learning_rate,
            num_leaves=settings.lightgbm_num_leaves,
            max_depth=settings.lightgbm_max_depth,
            deterministic=True,
            force_row_wise=True,
            num_threads=1,
            random_state=self.component_seed,
            verbose=-1,
        )
        model.fit(features, target)
        residuals = target - model.predict(features)
        residual_std = float(np.std(residuals, ddof=0))
        future_covariates = np.array(
            [
                (history.covariates[series_id].latest_value or 0.0) if is_level_covariate(series_id) else 0.0
                for series_id in covariate_ids
            ],
            dtype=float,
        )
        self.training_data = ModelTrainingData(
            **base,
            **alignment,
            minimum_returns_required=minimum,
            eligible_blocks=max(0, len(residuals) - block + 1),
            block_length=block,
            return_lags=lags,
            fitted_parameters={
                "n_estimators": float(settings.lightgbm_n_estimators),
                "learning_rate": float(settings.lightgbm_learning_rate),
                "num_leaves": float(settings.lightgbm_num_leaves),
                "max_depth": float(settings.lightgbm_max_depth),
                "training_rows": float(len(rows)),
                **{f"future_{series_id}": float(value) for series_id, value in zip(covariate_ids, future_covariates)},
            },
            residual_std=residual_std,
            notes=(
                "L2 regression on lagged WTI returns and the aligned covariates; innovations are "
                "overlapping blocks of in-sample residuals; recursive simulation with covariate "
                "returns set to zero and levels held at their latest cutoff-safe values."
            ),
        )
        innovations = overlapping_block_paths(
            self.rng,
            residuals,
            n_paths=settings.model_num_paths,
            path_length=settings.simulation_length_business_days,
            block=block,
        )

        def conditional_mean(lagged: np.ndarray) -> np.ndarray:
            tiled = np.broadcast_to(future_covariates, (lagged.shape[0], len(future_covariates)))
            return model.predict(pd.DataFrame(np.column_stack([lagged, tiled]), columns=feature_names))

        return recursive_lag_simulation(returns, lags, innovations, conditional_mean)


__all__ = ["LEVEL_COVARIATE_MARKERS", "LightGBMReturn", "is_level_covariate"]
