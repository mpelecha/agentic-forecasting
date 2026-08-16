"""LightGBM factory for CFM v5.0 using the unchanged shared predictor."""

from aieng.forecasting.methods.numerical import DartsLightGBMPredictor


def build_lightgbm_predictor(
    *,
    covariate_series_ids: list[str] | None = None,
    lags: int = 21,
    lags_past_covariates: int = 21,
    num_samples: int = 1_000,
) -> DartsLightGBMPredictor:
    return DartsLightGBMPredictor(
        lags=lags,
        lags_past_covariates=lags_past_covariates,
        covariate_series_ids=covariate_series_ids,
        num_samples=num_samples,
        lgbm_kwargs={"verbose": -1, "random_state": 42},
    )


__all__ = ["build_lightgbm_predictor"]
