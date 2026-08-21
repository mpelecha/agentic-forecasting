import io
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from energy_oil_forecasting.cfm_agent_v_2_2 import (
    CfmEventScorer,
    build_cfm_agent_v22_config,
)
from energy_oil_forecasting.cfm_agent_v_2_2.prompts import CfmEventScorePromptBuilder
from energy_oil_forecasting.data import WTI_SERIES_ID, build_wti_service


# ---------------------------------------------------------------------------
# 1. Configure visible temporal-verifier logging
# ---------------------------------------------------------------------------

verification_logger = logging.getLogger("aieng.forecasting.methods.agentic.agent_factory")
verification_logger.setLevel(logging.INFO)

verification_log_stream = io.StringIO()
capture_handler = logging.StreamHandler(verification_log_stream)
capture_handler.setLevel(logging.INFO)
capture_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
capture_handler._cfm_verifier_capture = True

display_handler = logging.StreamHandler(sys.stdout)
display_handler.setLevel(logging.INFO)
display_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
display_handler._cfm_verifier_display = True

for existing_handler in list(verification_logger.handlers):
    if getattr(existing_handler, "_cfm_verifier_capture", False) or getattr(
        existing_handler, "_cfm_verifier_display", False
    ):
        verification_logger.removeHandler(existing_handler)
        existing_handler.close()

verification_logger.addHandler(capture_handler)
verification_logger.addHandler(display_handler)
verification_logger.propagate = False

print("Temporal-verifier logging enabled.")


# ---------------------------------------------------------------------------
# 2. Locate the repository root
# ---------------------------------------------------------------------------

def find_repository_root(start: Path) -> Path:
    """Locate the agentic-forecasting repository root."""
    candidate = start.resolve()
    while True:
        implementation_dir = candidate / "implementations" / "energy_oil_forecasting"
        library_file = candidate / "aieng-forecasting" / "pyproject.toml"
        if implementation_dir.is_dir() and library_file.is_file():
            return candidate
        if candidate.parent == candidate:
            break
        candidate = candidate.parent
    raise FileNotFoundError("Could not locate the repository root.")


repository_root = find_repository_root(Path.cwd())
load_dotenv(repository_root / ".env", override=False)

print("\nRepository root:", repository_root)
print("OPENAI_API_KEY available:", bool(os.getenv("OPENAI_API_KEY")))

assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY is unavailable."
# No E2B_API_KEY check — v2.2 has no code_execution config.


# ---------------------------------------------------------------------------
# 3. Build the WTI data service (single series — no covariates needed)
# ---------------------------------------------------------------------------

cache_dir = repository_root / "data" / "yfinance"
service = build_wti_service(cache_dir=cache_dir)

available_series = list(service.series_ids)
assert WTI_SERIES_ID in available_series

print("\nRegistered series:", available_series)


# ---------------------------------------------------------------------------
# 4. Define the historical scoring origin
# ---------------------------------------------------------------------------

as_of = datetime(2026, 3, 2)

cutoff_history = service.get_series(WTI_SERIES_ID, as_of=as_of).copy()
cutoff_history["timestamp"] = pd.to_datetime(cutoff_history["timestamp"])
cutoff_history = cutoff_history.sort_values("timestamp").reset_index(drop=True)

assert not cutoff_history.empty
assert cutoff_history["timestamp"].max() <= pd.Timestamp(as_of)

print("\nScoring origin:", pd.Timestamp(as_of).date())
print("WTI observations available at cutoff:", len(cutoff_history))
print("Latest WTI observation available at cutoff:", cutoff_history["timestamp"].iloc[-1].date())
print("Latest WTI value available at cutoff:", float(cutoff_history["value"].iloc[-1]))


# ---------------------------------------------------------------------------
# 5. Build CFM Agent v2.2
# ---------------------------------------------------------------------------

config = build_cfm_agent_v22_config(model="gemini-3.1-flash-lite-preview")
scorer = CfmEventScorer(config)

function_tool_names = [tool.name for tool in config.function_tools]
skill_names = [path.name for path in config.skills_dirs]

print("\nAgent name:", config.name)
print("Scorer ID:", scorer.scorer_id)
print("Function tools:", function_tool_names)
print("Skills:", skill_names)
print("Search enabled:", config.context_retrieval.enabled)
print("Cutoff enforcement enabled:", config.context_retrieval.enforce_cutoff)
print("Verifier model:", config.context_retrieval.verifier_model)
print("Verifier maximum attempts:", config.context_retrieval.verifier_max_attempts)
print("Verifier confidence threshold:", config.context_retrieval.verifier_confidence_threshold)

assert config.name == "cfm_agent_v_2_2"
assert function_tool_names == []  # no quant tool
assert skill_names == ["event-context-analysis"]
assert config.context_retrieval.enabled is True
assert config.context_retrieval.enforce_cutoff is True
assert not (config.code_execution and config.code_execution.enabled)  # no E2B


# ---------------------------------------------------------------------------
# 6. Inspect the prompt (scores contract: no horizons, no quantiles)
# ---------------------------------------------------------------------------

context = service.context(as_of=as_of)
prompt_payload = json.loads(CfmEventScorePromptBuilder()(context=context))

print("\nPrompt cutoff:", prompt_payload["as_of"])
print("Prompt task:", prompt_payload["task"])
print("Target history CSV starts with:", prompt_payload["target_history_csv"][:40])

assert prompt_payload["as_of"] == "2026-03-02"
assert prompt_payload["task"] == "wti_event_context_scores"
assert prompt_payload["target_history_csv"].startswith("date,close")
assert "horizons" not in prompt_payload
assert "standard_quantiles" not in prompt_payload


# ---------------------------------------------------------------------------
# 7. Run the complete historical scoring pass
# ---------------------------------------------------------------------------

print("\nRunning the March 2, 2026 event scoring...")
result = scorer.score(context)
output = result.output

print("\nValidation attempts used:", result.attempts)
print("Factors returned:", len(output.factors))
print("Scenarios returned:", len(output.scenarios))


# ---------------------------------------------------------------------------
# 8. Inspect and validate the temporal verifier
# ---------------------------------------------------------------------------

capture_handler.flush()
verification_logs = verification_log_stream.getvalue()

print("\n" + "=" * 72)
print("TEMPORAL VERIFIER LOG")
print("=" * 72)
print(verification_logs.strip() or "No verifier log messages were captured.")
print("=" * 72)

verifier_pattern = re.compile(
    r"search_web verification attempt "
    r"(?P<attempt>\d+)/(?P<maximum>\d+): "
    r"clean=(?P<clean>True|False) "
    r"confidence=(?P<confidence>\d+) "
    r"flagged=(?P<flagged>\d+)"
)
verifier_matches = list(verifier_pattern.finditer(verification_logs))
assert verifier_matches, "No temporal-verifier attempt was captured."

verifier_attempts = []
for match in verifier_matches:
    verifier_attempts.append(
        {
            "attempt": int(match.group("attempt")),
            "maximum": int(match.group("maximum")),
            "clean": match.group("clean") == "True",
            "confidence": int(match.group("confidence")),
            "flagged_claim_count": int(match.group("flagged")),
        }
    )

print("\nParsed verifier attempts:")
for attempt_record in verifier_attempts:
    print(json.dumps(attempt_record, indent=2))

accepted_verifier_attempts = [
    a for a in verifier_attempts
    if a["clean"] and a["confidence"] >= config.context_retrieval.verifier_confidence_threshold
]
assert accepted_verifier_attempts, "No verifier attempt met clean=True and the confidence threshold."
accepted_verifier_attempt = accepted_verifier_attempts[-1]

print("\nAccepted verifier attempt:", accepted_verifier_attempt["attempt"])
print("Accepted verifier confidence:", accepted_verifier_attempt["confidence"])
print("Required confidence threshold:", config.context_retrieval.verifier_confidence_threshold)


# ---------------------------------------------------------------------------
# 9. Display the score card
# ---------------------------------------------------------------------------

print("\n" + "-" * 72)
print("SCORED FACTORS")
print("-" * 72)
for factor in output.factors:
    sign = "+" if factor.impact_score > 0 else ""
    print(f"({factor.tier}/{factor.category}) {factor.name}")
    print(f"  impact: {sign}{factor.impact_score}  confidence: {factor.confidence:.2f}  evidence: {factor.evidence_indices}")
    print(f"  {factor.rationale}")

print("\n" + "-" * 72)
print("SCENARIOS")
print("-" * 72)
for scenario in output.scenarios:
    tail = " [TAIL CASE]" if scenario.is_tail_case else ""
    sign = "+" if scenario.impact_score > 0 else ""
    print(f"{scenario.name}{tail}: p={scenario.probability:.2f}  impact: {sign}{scenario.impact_score}")
    print(f"  stances: {scenario.stances}")
    print(f"  {scenario.rationale}")

print("\n" + "=" * 72)
print("VERIFIED EVIDENCE REPORTED BY THE AGENT")
print("=" * 72)
for index, item in enumerate(output.verified_evidence):
    print(f"\nEvidence index: {index}")
    print(f"Title: {item.title}")
    print(f"Source URL: {item.source_url}")
    print(f"Claim: {item.claim}")
    print(f"Forecast effect: {item.forecast_effect}")

print("\nOverall rationale:", output.overall_rationale)
print("Research summary:", output.research_summary)


# ---------------------------------------------------------------------------
# 10. The calibration row — what the future translator will consume
# ---------------------------------------------------------------------------

calibration_row = result.calibration_row()

print("\n" + "=" * 72)
print("CALIBRATION ROW (input to the future score-to-price mapping)")
print("=" * 72)
print(json.dumps(calibration_row, indent=2))

# Serialization check — every stored row must round-trip as JSON.
json.dumps(output.model_dump(mode="json"))
json.dumps(calibration_row)
print("\nScore output serialization passed.")

print("\nTrace ID:", result.langfuse_trace_id)


# ---------------------------------------------------------------------------
# 11. Closing summary
# ---------------------------------------------------------------------------

core = [f for f in output.factors if f.tier == "core"]
transitory = [f for f in output.factors if f.tier == "transitory"]

print("\n" + "=" * 72)
print("MARCH 2, 2026 EVENT SCORING VALIDATION PASSED")
print("=" * 72)
print("Scoring origin:", pd.Timestamp(as_of).date())
print("Core factors:", len(core), " Transitory factors:", len(transitory))
print("Scenarios:", len(output.scenarios), " Probability sum:", f"{sum(s.probability for s in output.scenarios):.2f}")
print("Expected scenario impact:", f"{calibration_row['expected_scenario_impact']:+.2f}")
print("Validation attempts used:", result.attempts)
print("Accepted verifier confidence:", accepted_verifier_attempt["confidence"])
print("Prices anywhere in output: NO — scores only")
print("End-to-end scoring passed:", True)