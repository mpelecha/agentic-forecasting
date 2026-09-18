"""Re-derive a forecast from a stored run, under any calibration, without an LLM.

    RunRecord + candidate calibration -> final quantiles

This is the core of M1, and it works because of one property of ``cfm_agent_v_5_0``:
**the prompt contains no numeric constants.** The LLM emits categorical judgments --
``large_up``, ``substantially_wider``, ``likely_new_relative_to_model_data`` -- and
Python owns every number that turns those into quantiles. So the LLM's output is
independent of the constants by construction, and re-deciding a stored assessment
under different constants is exact rather than an approximation. Microseconds,
offline, no web search, no tokens.

The arithmetic is not reimplemented here. This module *imports the target agent's
own* ``EvidencePolicy`` and ``PythonForecastEngine`` and feeds them reconstructed
inputs, so the coach's model of the agent cannot drift from the agent.
``verify_fidelity`` is what proves that claim on every cycle: replaying a record
under the calibration that actually produced it must reproduce its recorded
quantiles exactly, or the harness stops rather than fits to a fiction.

*Which* agent's classes matters. v5.0's engine is ``python_forecast_engine_v50``
and v5.2's is ``..._v52``; they agree today, so replaying a v5.2 record through
v5.0's engine would pass fidelity right up until the two implementations diverge,
and then fit to a fiction with no signal that anything had changed. The engine
comes from the record's own :class:`AgentTarget` for that reason.

Three things happen per horizon, in v5.0's own order::

    EvidencePolicy.apply   -> PolicyDecision   (caps the LLM's ask to the evidence)
    PythonForecastEngine   -> ForecastTransformation
    CalibrationLayer.apply -> coach-owned centre gain / width scale (post-engine)

The layer is last and lives here rather than in the agent because the highest-leverage
lever -- base interval width -- has no in-engine control: v5.0's uncertainty multipliers
only fire when the LLM asks for a width change, and a neutral decision bypasses them
entirely (see ``PythonForecastEngine.transform``'s ``neutral`` branch).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from aieng.forecasting.evaluation.prediction import ContinuousForecast, Prediction
from aieng.forecasting.evaluation.task import ForecastingTask
from energy_oil_forecasting.cfm_coach.config import DEFAULT_SETTINGS, CoachSettings
from energy_oil_forecasting.cfm_coach.ledger import CalibrationLedger
from energy_oil_forecasting.cfm_coach.schemas import (
    CalibrationLayer,
    CalibrationVersion,
    HorizonRecord,
    Quantiles,
    RunRecord,
)
from energy_oil_forecasting.cfm_coach.targets import DEFAULT_TARGET, AgentTarget


#: Replaying under the record's own calibration is pure float arithmetic over the
#: same inputs, so it should reproduce the recorded output bit-for-bit. A tolerance
#: this tight only absorbs JSON decimal round-tripping; anything larger would be a
#: real divergence between the coach's model of v5.0 and v5.0 itself.
FIDELITY_TOLERANCE = 1e-9


class FidelityError(RuntimeError):
    """Replay did not reproduce a record's own recorded forecast.

    Raised rather than warned because everything downstream -- every candidate
    comparison, every proposal -- assumes replay is faithful. A drifted replay
    does not produce worse calibration; it produces confident nonsense.
    """


@dataclass(frozen=True)
class ReplayedHorizon:
    """One horizon re-derived under a named calibration."""

    horizon: int
    forecast_date: datetime
    calibration_version: str
    ensemble_point_forecast: float
    ensemble_quantiles: Quantiles
    #: v5.0's output, before the coach's post-engine layer.
    engine_point_forecast: float
    engine_quantiles: Quantiles
    #: What this calibration would actually have published.
    point_forecast: float
    quantiles: Quantiles
    #: The target package's own ``PolicyDecision``.
    decision: Any
    transformation: dict[str, Any]

    @property
    def overlay(self) -> float:
        """Dollars the LLM judgment moved the centre away from the raw ensemble."""
        return self.point_forecast - self.ensemble_quantiles[0.5]

    @property
    def p10_p90_width(self) -> float:
        return self.quantiles[0.9] - self.quantiles[0.1]


@dataclass(frozen=True)
class FidelityReport:
    """Whether replaying a record under its own calibration reproduced it."""

    run_id: str
    faithful: bool
    max_absolute_error: float
    tolerance: float
    #: Populated only on failure -- horizon -> human-readable first divergence.
    divergences: dict[int, str]
    #: Backfilled records carry reconstructed settings, so a mismatch there says
    #: the reconstruction was wrong, not that v5.0 changed. Worth distinguishing.
    settings_source: str

    def describe(self) -> str:
        if self.faithful:
            return f"{self.run_id}: faithful (max |error| {self.max_absolute_error:.2e})"
        detail = "; ".join(f"h={horizon} {text}" for horizon, text in sorted(self.divergences.items()))
        return f"{self.run_id}: DRIFTED (max |error| {self.max_absolute_error:.2e}) -- {detail}"


class ReplayEngine:
    """Re-decide and re-transform stored runs under arbitrary calibrations."""

    engine_id = "cfm_coach_replay_engine_v1"

    def __init__(self, settings: CoachSettings = DEFAULT_SETTINGS, *, target: AgentTarget = DEFAULT_TARGET):
        self.s = settings
        self.target = target
        self.ledger = CalibrationLedger(settings)

    # -- input reconstruction -------------------------------------------------

    def _assessment(self, record: RunRecord) -> Any:
        return self.target.assessment_from(record.assessment)

    def _packet(self, record: RunRecord) -> Any:
        return self.target.packet_from(record.research_packet)

    def _ensemble(self, horizon_record: HorizonRecord) -> Any:
        """Rebuild the engine's numerical input from the stored unadjusted ensemble.

        ``ModelHorizonForecast`` re-validates on construction -- quantiles present,
        finite, non-decreasing, ``point_forecast == quantiles[0.5]`` -- so a record
        whose ensemble was corrupted in storage fails here rather than silently
        replaying as something else.
        """
        return self.target.horizon_forecast(
            horizon=horizon_record.horizon,
            forecast_date=str(horizon_record.forecast_date.date()),
            point_forecast=horizon_record.ensemble_point_forecast,
            quantiles=dict(horizon_record.ensemble_quantiles),
        )

    @staticmethod
    def _latest_price(record: RunRecord) -> float | None:
        """Return the last visible WTI close, which sets the engine's emergency centre cap.

        v5.0 passes ``suite.diagnostics.latest_value`` here. When a record predates
        that field the engine falls back to its absolute USD cap, which is what
        ``None`` selects -- the same branch v5.0 itself takes.
        """
        value = record.diagnostics.get("latest_value")
        return float(value) if value is not None else None

    def agent_settings_for(
        self,
        record: RunRecord,
        calibration: CalibrationVersion,
    ) -> Any:
        """Return the settings this record would have run under, given a calibration.

        The overlay is applied on top of the record's *own* recorded settings rather
        than package defaults, because a record carries choices that are not
        calibration -- whether the audit controls ran, whether code execution was
        enabled, which LLM answered. Replaying a candidate must vary the calibrated
        constants and nothing else, or the comparison is confounded.
        """
        base = self.target.validate_settings(record.settings)
        return self.ledger.to_agent_settings(calibration, base=base, target=self.target)

    def _assert_target_matches(self, record: RunRecord) -> None:
        """Refuse to replay a record through another agent's engine.

        Cheap, and it closes the one failure this refactor could otherwise
        introduce: a `ReplayEngine` left on its v5.0 default, handed a v5.2 record
        by a mis-wired stream. The engines agree today, so the replay would look
        perfectly faithful and start lying the moment they diverge.
        """
        if record.agent_id != self.target.agent_id:
            raise FidelityError(
                f"{record.run_id} was produced by {record.agent_id!r} but this ReplayEngine targets "
                f"{self.target.agent_id!r}. Replaying through the wrong package's engine and policy "
                "can reproduce the recorded numbers by coincidence and diverge silently later."
            )

    # -- replay ---------------------------------------------------------------

    def replay(
        self,
        record: RunRecord,
        calibration: CalibrationVersion,
        *,
        horizons: tuple[int, ...] | None = None,
    ) -> list[ReplayedHorizon]:
        """Re-derive every horizon of `record` under `calibration`.

        No LLM, no network, no market data: the assessment and research packet are
        replayed verbatim, which is exactly why this is valid only for constants the
        prompt never sees.
        """
        self._assert_target_matches(record)
        settings = self.agent_settings_for(record, calibration)
        policy = self.target.policy(settings)
        engine = self.target.engine(settings)
        assessment = self._assessment(record)
        packet = self._packet(record)
        latest_price = self._latest_price(record)
        wanted = set(horizons) if horizons is not None else None

        replayed: list[ReplayedHorizon] = []
        for horizon_record in record.forecasts:
            if wanted is not None and horizon_record.horizon not in wanted:
                continue
            decision = policy.apply(assessment, packet, horizon_record.horizon)
            transformation = engine.transform(
                self._ensemble(horizon_record),
                decision,
                assessment.incremental_novelty,
                latest_price,
            )
            point, quantiles = calibration.layer.apply(
                ensemble_p50=horizon_record.ensemble_quantiles[0.5],
                final_point_forecast=transformation.final_point_forecast,
                final_quantiles=dict(transformation.final_quantiles),
                horizon=horizon_record.horizon,
                last_value=latest_price,
            )
            replayed.append(
                ReplayedHorizon(
                    horizon=horizon_record.horizon,
                    forecast_date=horizon_record.forecast_date,
                    calibration_version=calibration.version,
                    ensemble_point_forecast=horizon_record.ensemble_point_forecast,
                    ensemble_quantiles=dict(horizon_record.ensemble_quantiles),
                    engine_point_forecast=transformation.final_point_forecast,
                    engine_quantiles=dict(transformation.final_quantiles),
                    point_forecast=point,
                    quantiles=quantiles,
                    decision=decision,
                    transformation=transformation.model_dump(mode="json"),
                )
            )
        return replayed

    def replay_as_recorded(self, record: RunRecord) -> list[ReplayedHorizon]:
        """Replay under the calibration that actually produced the record.

        Note this loads by *version name*, not by date: `CalibrationLedger.current`
        answers "what was in force on a date", which is the right question when
        issuing a forecast and the wrong one when reproducing a past run, since a
        later version could since have been backdated.
        """
        return self.replay(record, self.ledger.load(record.calibration_version))

    # -- fidelity -------------------------------------------------------------

    def check_fidelity(self, record: RunRecord, *, tolerance: float = FIDELITY_TOLERANCE) -> FidelityReport:
        """Report whether replay reproduces a record, without raising."""
        divergences: dict[int, str] = {}
        worst = 0.0

        for replayed in self.replay_as_recorded(record):
            recorded = record.horizon(replayed.horizon)
            errors = [abs(replayed.point_forecast - recorded.final_point_forecast)]
            messages: list[str] = []
            if errors[0] > tolerance:
                messages.append(
                    f"point forecast {replayed.point_forecast:.10f} != recorded {recorded.final_point_forecast:.10f}"
                )

            missing = set(recorded.final_quantiles) ^ set(replayed.quantiles)
            if missing:
                messages.append(f"quantile levels differ: {sorted(missing)}")
            for level, value in recorded.final_quantiles.items():
                if level not in replayed.quantiles:
                    continue
                error = abs(replayed.quantiles[level] - value)
                errors.append(error)
                if error > tolerance:
                    messages.append(f"q{level} {replayed.quantiles[level]:.10f} != recorded {value:.10f}")

            worst = max(worst, *errors)
            if messages:
                divergences[replayed.horizon] = messages[0]

        return FidelityReport(
            run_id=record.run_id,
            faithful=not divergences,
            max_absolute_error=worst,
            tolerance=tolerance,
            divergences=divergences,
            settings_source=record.settings_source,
        )

    def verify_fidelity(self, record: RunRecord, *, tolerance: float = FIDELITY_TOLERANCE) -> FidelityReport:
        """Assert that replaying `record` under its own calibration reproduces it.

        This is the check that keeps the coach honest about v5.0. It fails when the
        agent package changed under the corpus, when a calibration version was
        edited after use, or when a record's settings were reconstructed wrongly --
        and every one of those invalidates any fit built on top of it.
        """
        report = self.check_fidelity(record, tolerance=tolerance)
        if not report.faithful:
            hint = (
                " Settings for this record were inferred, not observed, so the likely cause is the "
                "reconstruction rather than a change in the agent."
                if report.settings_source == "inferred"
                else f" Recorded package fingerprint: {record.package_fingerprint}."
            )
            raise FidelityError(report.describe() + hint)
        return report

    def verify_all(self, records: list[RunRecord], *, strict: bool = True) -> list[FidelityReport]:
        """Fidelity-check a corpus. With ``strict``, the first drift stops the cycle."""
        reports = [self.check_fidelity(record) for record in records]
        if strict:
            drifted = [report for report in reports if not report.faithful]
            if drifted:
                raise FidelityError(
                    f"{len(drifted)} of {len(reports)} records did not replay faithfully:\n  "
                    + "\n  ".join(report.describe() for report in drifted)
                )
        return reports


class CoachedCfmPredictor:
    """Wraps a live agent predictor and applies the coach's layer to its output.

    Composition, not modification: the agent runs exactly as shipped and this
    re-centres and re-scales the ``ContinuousForecast`` it hands back. With an
    identity layer -- which is what ``v001`` is -- the predictions pass through
    untouched, so wiring this in is a no-op until a calibration is actually approved.

    The original agent output is preserved under ``metadata["pre_calibration"]`` so a
    record written from these predictions still shows what v5.0 alone produced.
    """

    def __init__(self, predictor: Any, layer: CalibrationLayer, *, calibration_version: str):
        self.predictor = predictor
        self.layer = layer
        self.calibration_version = calibration_version

    @property
    def predictor_id(self) -> str:
        return getattr(self.predictor, "predictor_id", "cfm_coach_coached_predictor")

    def predict(self, task: ForecastingTask, context: Any) -> list[Prediction]:
        predictions = self.predictor.predict(task, context)
        if self.layer.is_identity:
            return predictions

        for prediction in predictions:
            payload = prediction.payload
            if not isinstance(payload, ContinuousForecast):
                continue
            horizon = int(prediction.metadata["forecast_transformation"]["horizon"])
            ensemble_p50 = float(prediction.metadata["unadjusted_ensemble"]["quantiles"][0.5])
            latest = (prediction.metadata.get("market_diagnostics") or {}).get("latest_value")
            point, quantiles = self.layer.apply(
                ensemble_p50=ensemble_p50,
                final_point_forecast=payload.point_forecast,
                final_quantiles=dict(payload.quantiles),
                horizon=horizon,
                last_value=float(latest) if latest is not None else None,
            )
            prediction.metadata["pre_calibration"] = {
                "point_forecast": payload.point_forecast,
                "quantiles": dict(payload.quantiles),
            }
            prediction.metadata["calibration_layer"] = {
                "calibration_version": self.calibration_version,
                "centre_gain": self.layer.gain_for(horizon),
                "width_scale": self.layer.scale_for(horizon),
                "rw_anchor": self.layer.anchor_for(horizon),
            }
            prediction.payload = ContinuousForecast(point_forecast=point, quantiles=quantiles)
        return predictions


__all__ = [
    "FIDELITY_TOLERANCE",
    "CoachedCfmPredictor",
    "FidelityError",
    "FidelityReport",
    "ReplayEngine",
    "ReplayedHorizon",
]
