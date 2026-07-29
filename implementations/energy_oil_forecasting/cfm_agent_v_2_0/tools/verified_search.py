"""Verified cutoff-aware web-search configuration."""

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
) -> ContextRetrievalConfig:
    """Return search configuration with the repository's claim verifier enabled."""
    return ContextRetrievalConfig(
        enabled=True,
        instruction=_SEARCH_INSTRUCTION,
        search_model=search_model,
        enforce_cutoff=True,
    )


__all__ = ["build_verified_search_config"]
