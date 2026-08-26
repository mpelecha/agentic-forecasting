"""Readable names for predictor_ids, shared by the analysis scripts.

Four predictor_ids share the prefix ``wti_analyst_news_scenario_schema``, so any
table that truncates to a fixed column width renders them identically and a
predictor appears to be missing from its own results. Both
``width_recalibration.py`` and ``conformal_calibration.py`` hit this, hence one
map rather than two.
"""

from __future__ import annotations


LABELS: dict[str, str] = {
    "wti_analyst_news_scenario_schema_anchored_logret": "SS Anchored (log ret)",
    "wti_analyst_news_scenario_schema_anchored": "SS Anchored (level)",
    "wti_analyst_news_scenario_schema_enhanced": "SS Enhanced",
    "wti_analyst_news_scenario_schema_temp0": "SS zero-temp",
    "wti_analyst_news_scenario_schema": "SS base",
    "wti_analyst_news_scenario": "News Agent Scenario",
    "wti_analyst_news_factors_v2": "News Agent Factors v2",
    "wti_analyst_news": "News Agent Original",
    "cfm_agent_v_5_2_2_delta_governed": "CFM v5.2.2 Delta-Gov",
    "cfm_agent_v_5_2_arima_only": "CFM v5.2.2 ARIMA-only",
    "cfm_agent_v_5_2": "CFM v5.2 (full)",
    "ecm_regression_logtgt_expanded_yh_levelonly": "ECM exp lvl-only logtgt",
    "ecm_regression_expanded_yh_levelonly": "ECM expanded lvl-only",
    "ecm_regression_expanded_yh": "ECM expanded",
    "ecm_regression_logtgt": "ECM (log target)",
    "ecm_regression": "ECM (level)",
    "darts_autoarima_logret": "AutoARIMA (log ret)",
    "darts_autoarima": "AutoARIMA",
    "darts_kalman": "Kalman",
    "darts_lightgbm_cov": "LightGBM + cov",
    "last_value_naive": "Naive",
}


def label(predictor_id: str) -> str:
    """Human-readable name; longest matching key wins so prefixes cannot collide."""
    stem = predictor_id.replace("agent_predictor_", "")
    stem = stem.replace("_gemini-3.1-flash-lite-preview", "").replace("_continuous", "")
    for key in sorted(LABELS, key=len, reverse=True):
        if stem.startswith(key):
            return LABELS[key]
    return stem
