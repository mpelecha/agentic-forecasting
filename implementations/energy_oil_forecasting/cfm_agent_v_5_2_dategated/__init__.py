"""CFM Agent v5.2 ARIMA-only with a publication-date gate on its news research.

v5.2 searches today's web for past cutoffs and relies on an LLM to remove
post-cutoff content. Cached v5.2 backtests show that this lets through pages
published after the cutoff. This variant passes on only the sentences that the
search's grounding metadata attributes to pages provably dated strictly before the
cutoff; see ``date_gate`` for the page rule and ``research`` for the filtering.
Nothing else differs from v5.2, including the prompt.
"""

from energy_oil_forecasting.cfm_agent_v_5_2_dategated.agent import (
    DateGatedPredictor,
    build_cfm_agent_config_arima_only_dategated,
    build_cfm_agent_predictor_arima_only_dategated,
)
from energy_oil_forecasting.cfm_agent_v_5_2_dategated.date_gate import date_source, url_dates
from energy_oil_forecasting.cfm_agent_v_5_2_dategated.research import DateGatedResearchPipelineTool


__all__ = [
    "DateGatedPredictor",
    "DateGatedResearchPipelineTool",
    "build_cfm_agent_config_arima_only_dategated",
    "build_cfm_agent_predictor_arima_only_dategated",
    "date_source",
    "url_dates",
]
