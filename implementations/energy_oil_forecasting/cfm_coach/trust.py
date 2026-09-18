"""R2 -- how much to believe one forecast, in a tier and one line of why.

Computed strictly *after* the forecast is final and structurally incapable of changing
it (R9). That is what lets it consume the material the forecast itself is forbidden to
see: v5.0 runs a Source Validator and a Claim-Support Verifier as audit-only controls,
records `material_evidence_conflict` and `evidence_tier`, and measures how far its
three numerical models disagree -- then discards all of it. A trust score is not the
number, so none of that isolation is violated by reading it here.

The tier is built around **detected problems**, not around confidence in direction.
"The agent had strong evidence" is not evidence that the agent was right; asserting
otherwise would bake in an untested belief. So `high` means "no red flag fired and
today has precedent", and every rule names a specific, checkable failure.

This whole tier is a **hypothesis until R4 can test it** -- until enough forecasts have
resolved to show that low-trust runs really do carry higher error. Every input is kept
in `TrustReport.signals` so that test can be run later, and so a tier that fails it can
be retired rather than quietly kept.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from energy_oil_forecasting.cfm_coach.case_base import MarketCaseBase
from energy_oil_forecasting.cfm_coach.config import DEFAULT_SETTINGS, CoachSettings
from energy_oil_forecasting.cfm_coach.diagnostics import state_from_diagnostics
from energy_oil_forecasting.cfm_coach.schemas import RunRecord, TrustReport, TrustTier


# Worst wins. Ordered so a single genuine red flag cannot be outvoted by clean signals.
_SEVERITY = {"no_basis": 0, "low": 1, "medium": 2, "high": 3}


class TrustReporter:
    reporter_id = "cfm_coach_trust_reporter_v1"

    def __init__(self, case_base: MarketCaseBase | None = None, settings: CoachSettings = DEFAULT_SETTINGS):
        self.case_base = case_base
        self.s = settings

    def assess(self, record: RunRecord, *, horizon: int | None = None) -> TrustReport:
        horizon = horizon or max(record.horizons)
        forecast = record.horizon(horizon)
        signals: dict[str, Any] = {"horizon": horizon}
        findings: list[tuple[TrustTier, str]] = []

        # -- v5.0's own audit-only material (the primary signals) ---------------
        audit = record.audit_signals
        failed = audit.get("failed_models") or []
        signals["failed_models"] = failed
        if failed:
            findings.append(("no_basis", f"numerical models failed: {', '.join(failed)}"))

        if record.assessment.get("material_evidence_conflict"):
            findings.append(("low", "the agent declared its own cited evidence in conflict"))
        signals["material_evidence_conflict"] = bool(record.assessment.get("material_evidence_conflict"))

        unsupported = [
            finding
            for finding in (audit.get("claim_support_findings") or [])
            if str(finding.get("verdict", "")).lower() in {"unsupported", "contradicted", "none"}
        ]
        signals["unsupported_claim_count"] = len(unsupported)
        if unsupported:
            findings.append(("low", f"{len(unsupported)} cited claim(s) not entailed by their sources"))

        warnings = forecast.forecast_transformation.get("warnings") or []
        signals["transformation_warnings"] = warnings
        if warnings:
            findings.append(("low", f"forecast engine warning: {warnings[0]}"))

        width = forecast.ensemble_quantiles[0.9] - forecast.ensemble_quantiles[0.1]
        disagreement = _model_disagreement(audit, horizon)
        signals["ensemble_width"] = width
        signals["model_disagreement_std"] = disagreement
        if disagreement is not None and width > 0:
            ratio = disagreement / width
            signals["model_disagreement_ratio"] = ratio
            if ratio > self.s.trust_model_disagreement_ratio:
                findings.append(("low", f"the three numerical models disagree by {ratio:.0%} of the interval"))

        # -- M3a analogues (secondary; see the caveat in the module docstring) ---
        if self.case_base is not None and record.diagnostics:
            findings.extend(self._analogue_findings(record, forecast, horizon, width, signals))

        return TrustReport(
            run_id=record.run_id,
            tier=_worst(findings),
            driver=_driver(findings),
            signals=signals,
            computed_at=datetime.now(),
        )

    def _analogue_findings(self, record, forecast, horizon, width, signals) -> list[tuple[TrustTier, str]]:  # noqa: ANN001
        findings: list[tuple[TrustTier, str]] = []
        state = state_from_diagnostics(record.diagnostics)
        if any(value is None for value in state.values()):
            signals["analogue_status"] = "incomplete_state"
            return findings

        verdict = self.case_base.assess_novelty(state)
        signals["novelty"] = verdict.as_dict()
        if verdict.is_out_of_distribution:
            findings.append(
                (
                    "no_basis",
                    f"no close analogue in {len(self.case_base):,} days "
                    f"({verdict.most_extreme_feature} at the {verdict.most_extreme_percentile:.0f}th percentile)",
                )
            )

        dispersion = self.case_base.move_dispersion(state, horizon, as_of=record.cutoff)
        signals["analogue_dispersion"] = dispersion
        if dispersion and width > 0:
            ratio = dispersion["implied_width"] / width
            signals["analogue_width_ratio"] = ratio
            if ratio > self.s.trust_width_ratio_low:
                findings.append(
                    (
                        "low",
                        f"comparable days spanned ${dispersion['implied_width']:.2f} over {horizon} days "
                        f"but the interval is only ${width:.2f} wide",
                    )
                )
        return findings

    def summarise(self, report: TrustReport) -> str:
        return f"trust: {report.tier} -- {report.driver}"


def _model_disagreement(audit: dict[str, Any], horizon: int) -> float | None:
    """Pull one horizon's model spread. JSON turns the integer keys into strings."""
    raw = audit.get("model_disagreement_std") or {}
    value = raw.get(str(horizon), raw.get(horizon))
    return None if value is None else float(value)


def _worst(findings: list[tuple[TrustTier, str]]) -> TrustTier:
    if not findings:
        return "high"
    return min((tier for tier, _ in findings), key=lambda tier: _SEVERITY[tier])


def _driver(findings: list[tuple[TrustTier, str]]) -> str:
    if not findings:
        return "no red flags: evidence self-consistent, models agree, today has precedent"
    tier = _worst(findings)
    return next(reason for level, reason in findings if level == tier)


__all__ = ["TrustReporter"]
