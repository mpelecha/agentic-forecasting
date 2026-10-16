"""Sparse AR Return (specification Section 1.4).

Closed-form ridge regression of the next daily simple return on the lagged
returns in ``return_lags``; the penalty applies to the coefficients and not to
the intercept. Innovations are overlapping blocks of in-sample residuals, and
the recursion builds each step's lag vector from observed returns and returns
already simulated on that path.
"""

from __future__ import annotations

import numpy as np
from energy_oil_forecasting.cfm_agent_v_6_0.models.paths import (
    ComponentFailure,
    ReturnPathComponent,
    lag_matrix,
    overlapping_block_paths,
    recursive_lag_simulation,
)
from energy_oil_forecasting.cfm_agent_v_6_0.schemas import ModelTrainingData


def ridge_fit(features: np.ndarray, target: np.ndarray, *, penalty: float, fit_intercept: bool) -> np.ndarray:
    """Solve ``(X'X + penalty * D) beta = X'y`` with ``D`` the identity except zero for the intercept.

    Returns ``beta`` with the intercept first when ``fit_intercept`` is true.
    """
    design = np.column_stack([np.ones(len(features)), features]) if fit_intercept else np.asarray(features, float)
    ridge = np.eye(design.shape[1]) * float(penalty)
    if fit_intercept:
        ridge[0, 0] = 0.0
    gram = design.T @ design + ridge
    return np.linalg.solve(gram, design.T @ np.asarray(target, float))


class SparseARReturn(ReturnPathComponent):
    """Ridge AR on sparse daily lags with residual block innovations."""

    component_name = "sparse_ar"

    def _simulate_returns(self) -> np.ndarray:
        settings = self.settings
        lags = list(settings.return_lags)
        max_lag = max(lags)
        returns = self.history.returns
        block = settings.block_length_business_days
        minimum = max_lag + block + settings.min_training_rows_margin
        rows = np.arange(max_lag, len(returns))
        base = self._base_training_data()
        if len(returns) < minimum:
            self.training_data = ModelTrainingData(
                **base,
                minimum_returns_required=minimum,
                block_length=block,
                return_lags=lags,
                notes=f"{len(rows)} training rows available before the eligibility check.",
            )
            raise ComponentFailure(
                "insufficient_history",
                f"{len(returns)} daily returns available; the sparse AR needs at least {minimum}",
            )
        features = lag_matrix(returns, lags, rows)
        target = returns[rows]
        beta = ridge_fit(
            features,
            target,
            penalty=settings.ar_ridge_penalty,
            fit_intercept=settings.ar_fit_intercept,
        )
        if settings.ar_fit_intercept:
            intercept, coefficients = float(beta[0]), beta[1:]
        else:
            intercept, coefficients = 0.0, beta
        fitted = intercept + features @ coefficients
        residuals = target - fitted
        residual_std = float(np.std(residuals, ddof=0))
        self.training_data = ModelTrainingData(
            **base,
            minimum_returns_required=minimum,
            eligible_blocks=max(0, len(residuals) - block + 1),
            block_length=block,
            return_lags=lags,
            fitted_parameters={
                "intercept": intercept,
                **{f"coef_lag_{lag}": float(value) for lag, value in zip(lags, coefficients)},
                "ridge_penalty": float(settings.ar_ridge_penalty),
                "training_rows": float(len(rows)),
            },
            residual_std=residual_std,
            notes=(
                "Closed-form ridge on the lagged daily returns with an unpenalised intercept; "
                "innovations are overlapping blocks of in-sample residuals; recursive simulation."
            ),
        )
        innovations = overlapping_block_paths(
            self.rng,
            residuals,
            n_paths=settings.model_num_paths,
            path_length=settings.simulation_length_business_days,
            block=block,
        )
        return recursive_lag_simulation(
            returns,
            lags,
            innovations,
            lambda lagged: intercept + lagged @ coefficients,
        )


__all__ = ["SparseARReturn", "ridge_fit"]
