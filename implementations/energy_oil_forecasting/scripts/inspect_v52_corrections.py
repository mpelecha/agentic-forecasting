"""Check whether v5.2's structured-output correction changed the forecast or only its format.

When the agent's first JSON answer fails validation, ``StructuredAssessmentController``
(``cfm_agent_v_5_2/response_control.py``) sends that answer, the validation errors,
and the schema to a separate LLM call and accepts whatever corrected JSON comes
back. Nothing checks that the corrected answer keeps the original judgment, so
this script compares, per origin, the fields that reach the forecast:

* ``center_action`` and ``uncertainty_action`` for each horizon;
* ``incremental_novelty`` — the evidence policy refuses any adjustment unless it is
  ``likely_new_relative_to_model_data`` or ``possibly_partly_reflected``, the forecast
  engine halves the centre move for the latter, and the strongest evidence tier
  requires the former;
* ``physical_status`` — shown for completeness; the policy and engine do not read it.

Both answers are stored in each prediction's ``structured_output_control_audit``
(``initial_response`` and ``retry_response``). The initial answer failed
validation, so it is parsed leniently: full JSON first, then the outermost
``{...}`` block, then regex extraction.

Usage (repo root, on the branch that has the weekly cache)::

    uv run python implementations/energy_oil_forecasting/scripts/inspect_v52_corrections.py
    uv run python implementations/energy_oil_forecasting/scripts/inspect_v52_corrections.py --model advanced

Reads cached predictions only: no LLM calls, no data fetch.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


_PKG = Path(__file__).resolve().parents[1]
STORE = _PKG / "data" / "predictions"
RESULTS_DIR = _PKG / "results"
MODELS = {"lite": "gemini-3.1-flash-lite-preview", "advanced": "gemini-3.5-flash"}
TASK_ID = "wti_oil_price_forecast"
TOP_FIELDS = ("incremental_novelty", "physical_status")
ACTION_FIELDS = ("center_action", "uncertainty_action")
# Values that let the evidence policy adjust the forecast at all.
NOVELTY_ELIGIBLE = {"likely_new_relative_to_model_data", "possibly_partly_reflected"}

_ACTION_RE = re.compile(r'\{[^{}]*?"horizon"\s*:\s*(\d+)[^{}]*?\}', re.DOTALL)


def _field_re(name: str) -> re.Pattern[str]:
    return re.compile(rf'"{name}"\s*:\s*"([^"]*)"')


def _strip_fence(text: str) -> str:
    text = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    return match.group(1) if match else text


def parse_answer(raw: str | None) -> tuple[dict[str, Any], dict[int, dict[str, Any]], str]:
    """Return (top-level fields, {horizon: action fields}, parse method) from a raw answer."""
    if not raw:
        return {}, {}, "missing"
    text = _strip_fence(raw)
    candidates = [("json", text)]
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(("json_block", text[start : end + 1]))
    for method, candidate in candidates:
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(data, dict):
            continue
        actions = {}
        for item in data.get("horizon_actions") or []:
            if isinstance(item, dict):
                try:
                    actions[int(item["horizon"])] = {f: item.get(f) for f in ACTION_FIELDS}
                except (KeyError, TypeError, ValueError):
                    continue
        return {f: data.get(f) for f in TOP_FIELDS}, actions, method
    # Last resort: regex. Top-level fields are taken from outside the action objects.
    outside = _ACTION_RE.sub("", text)
    top = {f: (m.group(1) if (m := _field_re(f).search(outside)) else None) for f in TOP_FIELDS}
    actions = {}
    for match in _ACTION_RE.finditer(text):
        block = match.group(0)
        actions[int(match.group(1))] = {
            f: (m.group(1) if (m := _field_re(f).search(block)) else None) for f in ACTION_FIELDS
        }
    return top, actions, "regex" if (actions or any(top.values())) else "unparseable"


def _short_errors(errors: list) -> str:
    parts = []
    for err in errors or []:
        if isinstance(err, dict):
            loc = ".".join(str(x) for x in err.get("loc", []) if x != "")
            given = err.get("input")
            given = f" (got {given!r})" if isinstance(given, str) else ""
            parts.append(f"{loc}{given}" if loc else str(err.get("msg", "")))
        else:
            parts.append(str(err))
    return " | ".join(parts)[:300]


def collect(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-origin and per-(origin, horizon) comparisons of initial vs corrected answers."""
    with open(path) as f:
        predictions = yaml.safe_load(f)["predictions"]

    origins, actions = [], []
    seen: set[str] = set()
    for pred in predictions:
        origin = pred["as_of"][:10]
        if origin in seen:  # every horizon of one origin carries the same audit
            continue
        seen.add(origin)
        meta = pred["metadata"]
        audit = meta.get("structured_output_control_audit") or {}
        final = meta.get("llm_context_assessment") or {}
        retried = bool(audit.get("correction_retry_attempted"))
        init_top, init_act, how = parse_answer(audit.get("initial_response"))
        corr_top, corr_act, _ = parse_answer(audit.get("retry_response")) if retried else (init_top, init_act, "")

        row = {
            "origin": origin,
            "disposition": audit.get("final_disposition"),
            "initial_parse": how,
            "validation_errors": _short_errors(audit.get("initial_validation_errors")),
        }
        for f in TOP_FIELDS:
            row[f"initial_{f}"] = init_top.get(f)
            row[f"corrected_{f}"] = corr_top.get(f)
            row[f"final_{f}"] = final.get(f)
            row[f"{f}_changed"] = retried and init_top.get(f) != corr_top.get(f)
        origins.append(row)

        final_actions = {int(a["horizon"]): a for a in final.get("horizon_actions", [])}
        for horizon in sorted(final_actions):
            before, after = init_act.get(horizon, {}), corr_act.get(horizon, {})
            arow = {"origin": origin, "horizon": horizon}
            for f in ACTION_FIELDS:
                arow[f"initial_{f}"] = before.get(f)
                arow[f"corrected_{f}"] = after.get(f)
                arow[f"final_{f}"] = final_actions[horizon].get(f)
                arow[f"{f}_changed"] = retried and before.get(f) != after.get(f)
            actions.append(arow)
    return pd.DataFrame(origins), pd.DataFrame(actions)


def report(per_origin: pd.DataFrame, per_action: pd.DataFrame, model: str) -> None:
    print(f"{model}: {len(per_origin)} origins, {len(per_action)} horizon actions")
    print(per_origin["disposition"].value_counts().to_string())
    print(f"Initial answer parsed via: {per_origin['initial_parse'].value_counts().to_dict()}")

    print("\nPer origin — fields the corrector rewrote (validation errors show the invalid value):")
    for _, r in per_origin.iterrows():
        flag = "*" if r["incremental_novelty_changed"] else " "
        print(
            f"  {r['origin']}  novelty {flag} {r['initial_incremental_novelty']!s} -> "
            f"{r['corrected_incremental_novelty']!s}  | physical {r['initial_physical_status']!s} -> "
            f"{r['corrected_physical_status']!s}"
        )
        if r["validation_errors"]:
            print(f"      errors: {r['validation_errors']}")

    print("\nPer origin and horizon — initial -> corrected action (* = changed):")
    for _, r in per_action.iterrows():
        c = "*" if r["center_action_changed"] else " "
        u = "*" if r["uncertainty_action_changed"] else " "
        print(
            f"  {r['origin']} h={r['horizon']:>2}  centre {c} {r['initial_center_action']!s:>14} -> "
            f"{r['corrected_center_action']!s:<14}  uncertainty {u} {r['initial_uncertainty_action']!s:>22} -> "
            f"{r['corrected_uncertainty_action']!s}"
        )

    def _count(df: pd.DataFrame, col: str) -> str:
        return f"{int(df[col].sum())}/{len(df)} ({df.loc[df[col], 'origin'].nunique()} origins)"

    print("\nChanged by the correction:")
    print(f"  centre action:        {_count(per_action, 'center_action_changed')}")
    print(f"  uncertainty action:   {_count(per_action, 'uncertainty_action_changed')}")
    print(f"  incremental_novelty:  {_count(per_origin, 'incremental_novelty_changed')}")
    print(f"  physical_status:      {_count(per_origin, 'physical_status_changed')}  (not read by policy or engine)")
    final_novelty = per_origin["final_incremental_novelty"]
    eligible = final_novelty.isin(NOVELTY_ELIGIBLE)
    print(
        f"\nAccepted incremental_novelty: {final_novelty.value_counts().to_dict()}\n"
        f"  -> {int(eligible.sum())}/{len(per_origin)} origins pass the policy's novelty gate; "
        "the rest cannot move the forecast whatever the centre action says."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=sorted(MODELS), default="lite")
    parser.add_argument("--spec-id", default="energy_oil_eval_weekly")
    args = parser.parse_args()

    agent_id = f"agent_predictor_cfm_agent_v_5_2_arima_only_{MODELS[args.model]}_continuous"
    per_origin, per_action = collect(STORE / args.spec_id / f"{agent_id}__{TASK_ID}.yaml")

    pd.set_option("display.width", 220)
    report(per_origin, per_action, args.model)

    RESULTS_DIR.mkdir(exist_ok=True)
    base = RESULTS_DIR / f"v52_arima_only_corrections__{args.model}__{args.spec_id}"
    per_origin.to_csv(f"{base}__by_origin.csv", index=False)
    per_action.to_csv(f"{base}__by_horizon.csv", index=False)
    print(f"\nWrote {base}__by_origin.csv\nWrote {base}__by_horizon.csv")


if __name__ == "__main__":
    main()
