"""Configuration models and constants for ``cfm_agent_v_2_1``."""

from __future__ import annotations

from pathlib import Path

from aieng.forecasting.models import ADVANCED_MODEL
from pydantic import BaseModel, ConfigDict, Field


AGENT_NAME = "cfm_agent_v_2_1"
PACKAGE_ROOT = Path(__file__).resolve().parent
SKILLS_ROOT = PACKAGE_ROOT / "skills"


class CfmGeoAgentSettings(BaseModel):
    """Runtime settings for the geopolitical-only CFM agent.

    Defaults for ``verifier_model``, ``verifier_max_attempts``, and
    ``verifier_confidence_threshold`` match the values used across the
    repository's other search-grounded agents. ``max_output_tokens`` matches
    the value ``cfm_agent_v_2_0`` passes to ``AgentConfig``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    verifier_model: str = ADVANCED_MODEL
    verifier_max_attempts: int = Field(default=3, ge=1)
    verifier_confidence_threshold: int = Field(default=8, ge=1, le=10)
    max_output_tokens: int = Field(default=24_576, ge=1)


DEFAULT_SETTINGS = CfmGeoAgentSettings()


__all__ = [
    "AGENT_NAME",
    "CfmGeoAgentSettings",
    "DEFAULT_SETTINGS",
    "PACKAGE_ROOT",
    "SKILLS_ROOT",
]
