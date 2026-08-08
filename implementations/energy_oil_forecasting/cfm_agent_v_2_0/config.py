"""Configuration models and constants for ``cfm_agent_v_2_0``."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


AGENT_NAME = "cfm_agent_v_2_0"
PACKAGE_ROOT = Path(__file__).resolve().parent
SKILLS_ROOT = PACKAGE_ROOT / "skills"


class CfmAgentSettings(BaseModel):
    """Runtime settings for the CFM agent and deterministic model suite."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_num_samples: int = Field(default=200, ge=50, le=5_000)
    lightgbm_lags: int = Field(default=21, ge=3)
    # Held at 21 (one trading month) deliberately, and reviewed when the
    # expanded covariate panel was added in NB04. Mixing weekly/monthly series
    # into a daily lag window is redundant rather than harmful: a monthly series
    # expanded onto business days is constant across all 21 lags, so it
    # contributes ~21 near-identical columns that dilute LightGBM's per-split
    # feature sampling without adding information. It is not a sample-size
    # problem — the panel spans thousands of aligned business days. Revisit
    # against measured feature importances rather than lowering it pre-emptively;
    # note this setting drives the CFM agent only, not the NB04 backtest, which
    # carries its own LAGS constant.
    lightgbm_covariate_lags: int = Field(default=21, ge=3)
    kalman_dim_x: int = Field(default=2, ge=1, le=20)
    max_data_rows_per_series: int = Field(default=520, ge=1, le=5_000)
    ensemble_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "arima": 1.0 / 3.0,
            "kalman": 1.0 / 3.0,
            "lightgbm": 1.0 / 3.0,
        }
    )


DEFAULT_SETTINGS = CfmAgentSettings()


__all__ = [
    "AGENT_NAME",
    "CfmAgentSettings",
    "DEFAULT_SETTINGS",
    "PACKAGE_ROOT",
    "SKILLS_ROOT",
]
