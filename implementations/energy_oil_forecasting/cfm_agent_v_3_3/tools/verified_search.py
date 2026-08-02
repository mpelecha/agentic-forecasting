"""Cutoff-aware search configuration with v3 corroboration instructions."""

from aieng.forecasting.methods.agentic.agent_factory import ContextRetrievalConfig
from aieng.forecasting.models import LITE_MODEL


_SEARCH_INSTRUCTION = """\
You are a cutoff-aware research specialist. Search only for information publicly
knowable on or before the supplied cutoff. For material oil-market claims,
prioritize official/primary sources and reputable independent reporting.
Separate confirmed facts from interpretation. Preserve conflicting accounts.
Broker, trading-site, and analyst commentary may provide context but cannot by
itself establish geopolitical facts, physical disruption, production changes,
or strategic-reserve action. Never fill gaps from model memory.
"""


def build_verified_search_config(*, search_model: str = LITE_MODEL) -> ContextRetrievalConfig:
    return ContextRetrievalConfig(
        enabled=True,
        instruction=_SEARCH_INSTRUCTION,
        search_model=search_model,
        enforce_cutoff=True,
    )


__all__ = ["build_verified_search_config"]
