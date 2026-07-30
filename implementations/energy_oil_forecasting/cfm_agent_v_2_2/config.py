"""Configuration models and constants for ``cfm_agent_v_2_2``."""

from __future__ import annotations

from pathlib import Path

from aieng.forecasting.models import ADVANCED_MODEL
from pydantic import BaseModel, ConfigDict, Field


AGENT_NAME = "cfm_agent_v_2_2"
PACKAGE_ROOT = Path(__file__).resolve().parent
SKILLS_ROOT = PACKAGE_ROOT / "skills"


class CfmEventScorerSettings(BaseModel):
    """Runtime settings for the event-context scoring agent.

    Verifier defaults match ``cfm_agent_v_2_1``. ``max_validation_attempts``
    bounds the scorer's own retry loop: unlike the forecast predictors, the
    scorer runs outside the backtest harness and its retry wrapper, so it
    must re-run the agent itself when the output fails schema validation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    verifier_model: str = ADVANCED_MODEL
    verifier_max_attempts: int = Field(default=3, ge=1)
    verifier_confidence_threshold: int = Field(default=8, ge=1, le=10)
    max_output_tokens: int = Field(default=24_576, ge=1)
    max_validation_attempts: int = Field(default=3, ge=1)


DEFAULT_SETTINGS = CfmEventScorerSettings()


__all__ = [
    "AGENT_NAME",
    "CfmEventScorerSettings",
    "DEFAULT_SETTINGS",
    "PACKAGE_ROOT",
    "SKILLS_ROOT",
]
