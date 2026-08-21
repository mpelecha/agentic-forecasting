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

from aieng.forecasting.evaluation.task import ForecastingTask
from energy_oil_forecasting.cfm_agent_v_2_1 import (
    build_cfm_agent_v21_config,
    build_cfm_agent_v21_predictor,
)
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
# No E2B_API_KEY check — v2.1 has no code_execution config.


# ---------------------------------------------------------------------------
# 3. Build the WTI data service (single series — no covariates needed)
# ---------------------------------------------------------------------------

cache_dir = repository_root / "data" / "yfinance"
service = build_wti_service(cache_dir=cache_dir)

available_series = list(service.series_ids)
assert WTI_SERIES_ID in available_series

print("\nRegistered series:", available_series)


# ---------------------------------------------------------------------------
# 4. Define the historical forecast origin
# ---------------------------------------------------------------------------

as_of = datetime(2026, 3, 2)
horizons = [5, 10, 21]

cutoff_history = service.get_series(WTI_SERIES_ID, as_of=as_of).copy()
cutoff_history["timestamp"] = pd.to_datetime(cutoff_history["timestamp"])
cutoff_history = cutoff_history.sort_values("timestamp").reset_index(drop=True)

assert not cutoff_history.empty
assert cutoff_history["timestamp"].max() <= pd.Timestamp(as_of)

print("\nForecast origin:", pd.Timestamp(as_of).date())
print("Requested horizons:", horizons)
print("WTI observations available at cutoff:", len(cutoff_history))
print("Latest WTI observation available at cutoff:", cutoff_history["timestamp"].iloc[-1].date())
print("Latest WTI value available at cutoff:", float(cutoff_history["value"].iloc[-1]))


# ---------------------------------------------------------------------------
# 5. Build CFM Agent v2.1
# ---------------------------------------------------------------------------

config = build_cfm_agent_v21_config(model="gemini-3.1-flash-lite-preview")
predictor = build_cfm_agent_v21_predictor(config)

function_tool_names = [tool.name for tool in config.function_tools]
skill_names = [path.name for path in config.skills_dirs]

print("\nAgent name:", config.name)
print("Predictor ID:", predictor.predictor_id)
print("Function tools:", function_tool_names)
print("Skills:", skill_names)
print("Search enabled:", config.context_retrieval.enabled)
print("Cutoff enforcement enabled:", config.context_retrieval.enforce_cutoff)
print("Verifier model:", config.context_retrieval.verifier_model)
print("Verifier maximum attempts:", config.context_retrieval.verifier_max_attempts)
print("Verifier confidence threshold:", config.context_retrieval.verifier_confidence_threshold)

assert config.name == "cfm_agent_v_2_1"
assert function_tool_names == []  # no quant tool
assert skill_names == ["geopolitical-analysis"]
assert config.context_retrieval.enabled is True
assert config.context_retrieval.enforce_cutoff is True
assert config.context_retrieval.verifier_max_attempts >= 1
assert config.context_retrieval.verifier_confidence_threshold >= 1
assert not (config.code_execution and config.code_execution.enabled)  # no E2B


# ---------------------------------------------------------------------------
# 6. Define the forecasting task and context
# ---------------------------------------------------------------------------

task = ForecastingTask(
    task_id="cfm_agent_v_2_1_2026_03_02_verified_forecast",
    target_series_id=WTI_SERIES_ID,
    horizons=horizons,
    frequency="B",
    description=(
        "Forecast WTI continuous front-month futures prices as of March 2, 2026, "
        "at 5, 10, and 21 business-day horizons using geopolitical scenario analysis "
        "and independently verified web research. No quant models."
    ),
)

context = service.context(as_of=as_of)


# ---------------------------------------------------------------------------
# 7. Inspect the prompt (v2.1 carries compressed history, unlike v2.0)
# ---------------------------------------------------------------------------

prompt_text = predictor.prompt_builder(task=task, context=context)
prompt_payload = json.loads(prompt_text)

print("\nPrompt cutoff:", prompt_payload["as_of"])
print("Prompt target:", prompt_payload["target_series_id"])
print("Target history CSV starts with:", prompt_payload["target_history_csv"][:40])

assert prompt_payload["as_of"] == "2026-03-02"
assert prompt_payload["target_series_id"] == WTI_SERIES_ID
assert prompt_payload["horizons"] == horizons
assert prompt_payload["target_history_csv"].startswith("date,close")


# ---------------------------------------------------------------------------
# 8. Run the complete historical forecast
# ---------------------------------------------------------------------------

print("\nRunning the March 2, 2026 forecast...")
predictions = predictor.predict(task, context)

print("\nPredictions returned:", len(predictions))
assert len(predictions) == len(horizons), "The agent did not return one prediction for each requested horizon."


# ---------------------------------------------------------------------------
# 9. Inspect and validate the temporal verifier (same harness-level logging)
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

accepted = [
    m for m in verifier_matches
    if m.group("clean") == "True" and int(m.group("confidence")) >= config.context_retrieval.verifier_confidence_threshold
]
assert accepted, "The verifier ran, but no attempt met clean=True and the confidence threshold."
print("\nTemporal-verifier validation passed.")


# ---------------------------------------------------------------------------
# 10. Validate the forecast and its geopolitical scenario decomposition
# ---------------------------------------------------------------------------

for prediction in predictions:
    forecast = prediction.payload
    metadata = prediction.metadata

    quantile_levels = sorted(forecast.quantiles)
    quantile_values = [forecast.quantiles[level] for level in quantile_levels]
    assert quantile_values == sorted(quantile_values), "Quantiles are not non-decreasing."
    assert forecast.point_forecast == forecast.quantiles[0.50], "point_forecast does not equal p50."

    # Factor/scenario audit — replaces v2.0's component_models audit (no quant tool here)
    assert "factors" in metadata
    core = [f for f in metadata["factors"] if f["tier"] == "core"]
    transitory = [f for f in metadata["factors"] if f["tier"] == "transitory"]
    print(f"\nHorizon result — core factors: {len(core)}, transitory factors: {len(transitory)}")

    assert "scenarios" in metadata
    assert any(s["is_tail_case"] for s in metadata["scenarios"]), "No tail-case scenario present."

    assert metadata["overall_rationale"].strip()
    assert isinstance(metadata["verified_evidence"], list)
    assert isinstance(metadata["warnings"], list)

print("\nAll v2.1 forecast validations passed.")

# ---------------------------------------------------------------------------
# 11. Parse and print each verifier attempt (matches Kam's v2.0 script)
# ---------------------------------------------------------------------------

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
assert accepted_verifier_attempts, "No accepted verifier attempt met clean=True and the confidence threshold."
accepted_verifier_attempt = accepted_verifier_attempts[-1]

print("\nAccepted verifier attempt:", accepted_verifier_attempt["attempt"])
print("Accepted verifier status:", accepted_verifier_attempt["clean"])
print("Accepted verifier confidence:", accepted_verifier_attempt["confidence"])
print("Required confidence threshold:", config.context_retrieval.verifier_confidence_threshold)
print("Flagged claims on accepted attempt:", accepted_verifier_attempt["flagged_claim_count"])


# ---------------------------------------------------------------------------
# 12. Display per-horizon forecasts
# ---------------------------------------------------------------------------

for prediction in predictions:
    forecast = prediction.payload
    metadata = prediction.metadata
    horizon_days = len(
        pd.bdate_range(start=pd.Timestamp(prediction.as_of), end=pd.Timestamp(prediction.forecast_date))
    ) - 1

    print("\n" + "-" * 72)
    print(f"Forecast date: {prediction.forecast_date.date()}")
    print(f"Horizon: {horizon_days}B")
    print(f"Final forecast: ${forecast.point_forecast:.2f}")
    print(f"p05 / p50 / p95: ${forecast.quantiles[0.05]:.2f} / ${forecast.quantiles[0.5]:.2f} / ${forecast.quantiles[0.95]:.2f}")

    print("Factors:")
    for factor in metadata["factors"]:
        tag = f" [{factor['impact_score']}]" if factor.get("impact_score") else ""
        print(f"  ({factor['tier']}) {factor['name']}{tag}")

    print("Scenarios:")
    for scenario in metadata["scenarios"]:
        tail = " [TAIL CASE]" if scenario["is_tail_case"] else ""
        print(f"  {scenario['name']}{tail}: ${scenario['price_low']}-${scenario['price_high']}")

    print("Verified evidence count:", len(metadata["evidence_indices"]))
    print(f"Rationale: {metadata['rationale']}")


# ---------------------------------------------------------------------------
# 13. Verified evidence — printed once, not per horizon
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("VERIFIED EVIDENCE REPORTED BY THE AGENT")
print("=" * 72)

for index, item in enumerate(predictions[0].metadata["verified_evidence"]):
    print(f"\nEvidence index: {index}")
    print(f"Title: {item['title']}")
    print(f"Source URL: {item['source_url']}")
    print(f"Claim: {item['claim']}")
    print(f"Forecast effect: {item['forecast_effect']}")


# ---------------------------------------------------------------------------
# 14. Metadata serialization and trace link
# ---------------------------------------------------------------------------

for prediction in predictions:
    json.dumps(prediction.metadata)  # raises on failure
print("\nPrediction metadata serialization passed.")

trace_id = predictions[0].metadata.get("langfuse_trace_id")
trace_url = predictions[0].metadata.get("langfuse_trace_url")
print(f"\nTrace ID: {trace_id}")
print(f"Langfuse trace: {trace_url}")


# ---------------------------------------------------------------------------
# 15. Closing summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("MARCH 2, 2026 FORECAST VALIDATION PASSED")
print("=" * 72)
print("Forecast origin:", pd.Timestamp(as_of).date())
print("Predictions returned:", len(predictions))
print("Temporal verifier attempts:", len(verifier_attempts))
print("Accepted verifier attempt:", accepted_verifier_attempt["attempt"])
print("Accepted verifier confidence:", accepted_verifier_attempt["confidence"])
print("Verifier threshold:", config.context_retrieval.verifier_confidence_threshold)
print("Verifier passed:", True)
print("Metadata serialization passed:", True)
print("End-to-end forecast passed:", True)