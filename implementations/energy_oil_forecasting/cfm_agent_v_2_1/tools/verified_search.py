"""Verified cutoff-aware web-search configuration.

Copied from ``cfm_agent_v_2_0.tools.verified_search`` (read-only
reference), with the verifier parameters wired to
``CfmGeoAgentSettings`` so that package's defaults are not silently
unused.
"""

from __future__ import annotations

from aieng.forecasting.methods.agentic.agent_factory import ContextRetrievalConfig
from aieng.forecasting.models import LITE_MODEL


_SEARCH_INSTRUCTION = """\
You are a cutoff-aware market research specialist. Search for evidence relevant
to the user's forecasting question. Separate facts from interpretation and cite
the retrieved sources. When a cutoff date is supplied, include only information
that was publicly knowable on or before that date. Never fill gaps from model
memory. If verified evidence is insufficient or conflicting, say so explicitly.
Prioritize primary sources and reputable independent reporting.\
"""


def build_verified_search_config(
    *,
    search_model: str = LITE_MODEL,
    verifier_model: str,
    verifier_max_attempts: int,
    verifier_confidence_threshold: int,
) -> ContextRetrievalConfig:
    """Return search configuration with the repository's claim verifier enabled."""
    return ContextRetrievalConfig(
        enabled=True,
        instruction=_SEARCH_INSTRUCTION,
        search_model=search_model,
        enforce_cutoff=True,
        verifier_model=verifier_model,
        verifier_max_attempts=verifier_max_attempts,
        verifier_confidence_threshold=verifier_confidence_threshold,
    )


__all__ = ["build_verified_search_config"]
