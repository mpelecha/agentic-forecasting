"""Configuration for the self-contained CFM Agent v3.3 package."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


AGENT_NAME = "cfm_agent_v_3_3"
PACKAGE_ROOT = Path(__file__).resolve().parent
SKILLS_ROOT = PACKAGE_ROOT / "skills"


class CfmV33Settings(BaseModel):
    """Runtime settings for numerical models, research gates, and policy bounds."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Uses existing public predictor interfaces; v3 does not change shared code.
    model_num_samples: int = Field(default=1_000, ge=50, le=5_000)
    lightgbm_lags: int = Field(default=21, ge=3)
    lightgbm_covariate_lags: int = Field(default=21, ge=3)
    kalman_dim_x: int = Field(default=2, ge=1, le=20)
    max_data_rows_per_series: int = Field(default=520, ge=1, le=5_000)
    max_pre_execution_attempts: int = Field(default=2, ge=1, le=3)
    ensemble_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "arima": 1.0 / 3.0,
            "kalman": 1.0 / 3.0,
            "lightgbm": 1.0 / 3.0,
        }
    )

    # Graduated evidence gates. Python assigns the tier; the LLM cannot.
    tier_1_min_publishers: int = Field(default=1, ge=1)
    tier_2_min_publishers: int = Field(default=2, ge=1)
    tier_3_min_publishers: int = Field(default=3, ge=1)
    tier_1_min_confidence: float = Field(default=0.50, ge=0.0, le=1.0)
    tier_2_min_confidence: float = Field(default=0.65, ge=0.0, le=1.0)
    tier_3_min_confidence: float = Field(default=0.80, ge=0.0, le=1.0)
    tier_1_min_claim_support: int = Field(default=1, ge=1)
    tier_2_min_claim_support: int = Field(default=1, ge=1)
    tier_3_min_claim_support: int = Field(default=2, ge=1)
    require_high_quality_source: bool = True
    allow_verified_identity_with_imperfect_url: bool = True
    partly_reflected_adjustment_multiplier: float = Field(default=0.50, ge=0.0, le=1.0)

    # Bounded forecast-action policy. Fractions scale with ensemble p10-p90 width.
    small_action_width_fraction: float = Field(default=0.10, ge=0.0, le=1.0)
    moderate_action_width_fraction: float = Field(default=0.20, ge=0.0, le=1.0)
    max_center_adjustment_usd: float = Field(default=3.0, ge=0.0)
    max_center_adjustment_price_fraction: float = Field(default=0.05, ge=0.0, le=0.50)
    moderately_wider_multiplier: float = Field(default=1.10, ge=1.0)
    substantially_wider_multiplier: float = Field(default=1.25, ge=1.0)
    moderately_narrower_multiplier: float = Field(default=0.90, gt=0.0, le=1.0)
    policy_mode: Literal["constrained_actions", "ensemble_locked"] = "constrained_actions"


DEFAULT_SETTINGS = CfmV33Settings()

__all__ = [
    "AGENT_NAME",
    "CfmV33Settings",
    "DEFAULT_SETTINGS",
    "PACKAGE_ROOT",
    "SKILLS_ROOT",
]
