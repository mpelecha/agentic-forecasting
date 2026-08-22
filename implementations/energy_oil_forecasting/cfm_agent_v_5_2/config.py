"""Configuration for the self-contained CFM Agent v5.2 package."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


AGENT_NAME = "cfm_agent_v_5_2"
PACKAGE_ROOT = Path(__file__).resolve().parent
SKILLS_ROOT = PACKAGE_ROOT / "skills"


class CfmV52Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model_num_samples: int = Field(default=1_000, ge=50, le=5_000)
    lightgbm_lags: int = Field(default=21, ge=3)
    lightgbm_covariate_lags: int = Field(default=21, ge=3)
    kalman_dim_x: int = Field(default=2, ge=1, le=20)
    max_data_rows_per_series: int = Field(default=520, ge=1, le=5_000)
    max_pre_execution_attempts: int = Field(default=2, ge=1, le=3)
    ensemble_weights: dict[str, float] = Field(
        default_factory=lambda: {"arima": 1 / 3, "kalman": 1 / 3, "lightgbm": 1 / 3}
    )
    tier_1_min_publishers: int = 1
    tier_2_min_publishers: int = 2
    tier_3_min_publishers: int = 3
    tier_1_min_confidence: float = 0.40
    tier_2_min_confidence: float = 0.60
    tier_3_min_confidence: float = 0.80
    tier_3_min_publishers_per_claim: int = 2
    partly_reflected_multiplier: float = 0.50
    small_action_width_fraction: float = 0.10
    moderate_action_width_fraction: float = 0.20
    large_action_width_fraction: float = 0.30
    emergency_center_cap_usd: float = 100.0
    emergency_center_cap_price_fraction: float = 1.0
    small_wider_multiplier: float = 1.10
    small_narrower_multiplier: float = 0.90
    moderately_wider_multiplier: float = 1.20
    moderately_narrower_multiplier: float = 0.80
    substantially_wider_multiplier: float = 1.30
    substantially_narrower_multiplier: float = 0.70
    nonnegative_price_floor: float | None = 0.0
    source_resolution_timeout_seconds: float = 8.0
    source_resolution_max_redirects: int = 5
    research_max_results_per_query: int = 5
    research_max_passages_per_source: int = 5
    research_passage_chars: int = 900
    search_verifier_model: str = "gemini-3.5-flash"
    search_verifier_max_attempts: int = Field(default=3, ge=1, le=5)
    search_verifier_confidence_threshold: int = Field(default=8, ge=1, le=10)
    structured_output_retry_model: str = "gemini-3.1-flash-lite-preview"  # 2026.08.19: changed from 3.5-flash to minimize token usage in backtest comparison
    structured_output_retry_timeout_seconds: float = Field(default=60.0, gt=0.0, le=300.0)
    audit_enabled: bool = False
    code_execution_enabled: bool = True
    claim_support_verifier_model: str = "gemini-3.1-flash-lite-preview"  # 2026.08.19: changed from 3.5-flash to minimize token usage in backtest comparison
    claim_support_verifier_timeout_seconds: float = Field(default=60.0, gt=0.0, le=300.0)
    policy_mode: Literal["constrained_actions", "ensemble_locked"] = "constrained_actions"


DEFAULT_SETTINGS = CfmV52Settings()
__all__ = [
    "AGENT_NAME",
    "PACKAGE_ROOT",
    "SKILLS_ROOT",
    "CfmV52Settings",
    "DEFAULT_SETTINGS",
]
