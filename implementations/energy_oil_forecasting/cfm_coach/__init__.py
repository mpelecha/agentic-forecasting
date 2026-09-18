"""CFM Coach public API.

A standalone package that reads what an agent package produces, waits for the
world to resolve each forecast, and proposes -- never applies -- calibration
corrections. It imports from the agent packages but never writes to them.

Two agents are targeted (`targets.V50`, `targets.V52`) across three daily run
streams (`streams.STREAMS`), each with its own corpus and its own ledger. Entry
points still default to v5.0, so anything written before v5.2 existed keeps its
old behaviour without being passed a target.
"""

from energy_oil_forecasting.cfm_coach.config import (
    BUILTIN_PROMPT_VERSION,
    COACH_NAME,
    DEFAULT_SETTINGS,
    RUN_RECORD_SCHEMA_VERSION,
    CoachSettings,
)
from energy_oil_forecasting.cfm_coach.ledger import CalibrationLedger
from energy_oil_forecasting.cfm_coach.outcomes import OutcomeResolver
from energy_oil_forecasting.cfm_coach.policy.comparison_policy import ComparisonPolicy
from energy_oil_forecasting.cfm_coach.replay import (
    CoachedCfmPredictor,
    FidelityError,
    ReplayEngine,
)
from energy_oil_forecasting.cfm_coach.report import CoachReporter
from energy_oil_forecasting.cfm_coach.run_store import (
    RunRecordStore,
    agent_package_fingerprint,
    build_run_record,
)
from energy_oil_forecasting.cfm_coach.schemas import (
    CalibrationLayer,
    CalibrationVersion,
    Candidate,
    ComparisonVerdict,
    HorizonRecord,
    Provenance,
    ResolvedForecast,
    RunRecord,
    ScoreCard,
    TrustReport,
    TrustTier,
)
from energy_oil_forecasting.cfm_coach.scoring import ForecastScorer, pinball_loss
from energy_oil_forecasting.cfm_coach.streams import (
    STREAMS,
    V50_LITE,
    V52_ADVANCED,
    V52_LITE,
    RunStream,
    stream_for,
)
from energy_oil_forecasting.cfm_coach.targets import (
    V50,
    V52,
    AgentTarget,
    target_for,
)


__all__ = [
    "BUILTIN_PROMPT_VERSION",
    "COACH_NAME",
    "DEFAULT_SETTINGS",
    "RUN_RECORD_SCHEMA_VERSION",
    "STREAMS",
    "V50",
    "V50_LITE",
    "V52",
    "V52_ADVANCED",
    "V52_LITE",
    "AgentTarget",
    "CalibrationLayer",
    "CalibrationLedger",
    "CalibrationVersion",
    "Candidate",
    "CoachReporter",
    "CoachSettings",
    "CoachedCfmPredictor",
    "ComparisonPolicy",
    "ComparisonVerdict",
    "FidelityError",
    "ForecastScorer",
    "HorizonRecord",
    "OutcomeResolver",
    "Provenance",
    "ReplayEngine",
    "ResolvedForecast",
    "RunRecord",
    "RunRecordStore",
    "RunStream",
    "ScoreCard",
    "TrustReport",
    "TrustTier",
    "agent_package_fingerprint",
    "build_run_record",
    "pinball_loss",
    "stream_for",
    "target_for",
]
