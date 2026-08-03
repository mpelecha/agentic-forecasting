"""Verified cutoff-aware web-search configuration.

Same search instruction and verifier wiring as ``cfm_agent_v_2_1``
(reimplemented here — packages stay standalone by convention).
"""

from __future__ import annotations

from aieng.forecasting.methods.agentic.agent_factory import ContextRetrievalConfig
from aieng.forecasting.models import LITE_MODEL


_SEARCH_INSTRUCTION = """\
You are an oil market intelligence specialist with access to web search.

Search for information relevant to the query you were given and return a \
concise, grounded markdown summary (3-5 paragraphs). Report the events \
that are actually driving price action according to the sources you \
retrieve — do not impose a fixed checklist of topics; let the search \
results themselves determine which events are significant right now. \
Report events and situations, not statistics: \
supply/demand figures, inventory data, and agency forecasts or their \
revisions (EIA, IEA, OPEC) belong to other market and financial data pipelines and must not \
be the substance of your summary. Prefer primary reporting about a \
specific event over market-wrap or outlook articles.

Where credible sources actively disagree on a geopolitical, weather, \
operational, policy, or demand-expectation driver — name both sides \
rather than reporting only the majority view.

If the current situation resembles a past market episode, search for that \
precedent explicitly and report: what happened, how WTI/Brent moved in the \
following weeks, and how similar the current setup actually is versus \
superficially similar. Only include an analogue if you can ground it in a \
retrieved source — do not construct one from memory. If no clear precedent \
surfaces in search results, say so rather than forcing a comparison.

Ground your summary in the search results you actually retrieve. \
When a cutoff date is specified, do not report or speculate about events \
that occurred after that date!

Before finalizing your summary, reason step by step: (1) for each candidate \
fact, judge its actual recency from the substance of the result itself, \
never from a source's claimed publish date or byline timestamp — those are \
frequently stale or updated after original publication; (2) discard \
anything you cannot confidently place before the cutoff date; (3) only then \
write your summary. Do not supplement the search results with your own \
background/training knowledge — if the results are insufficient, say so \
explicitly rather than filling gaps from memory.\
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
