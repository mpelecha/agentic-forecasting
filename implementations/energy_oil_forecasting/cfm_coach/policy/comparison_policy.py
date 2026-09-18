"""The locked gate: the only component permitted to say a calibration is better.

Everything else in the coach measures. This decides. The separation is deliberate --
a scoreboard that also gets to declare victory will eventually be tuned until it
does, and the failure is silent because every intermediate number still looks right.

Ten conditions, all of which must hold. Seven are about *what the corpus is*, three
are about *what the numbers say*, and the ordering matters: the statistical tests are
the last thing checked, because a p-value computed over a contaminated or pooled
corpus is worse than no p-value at all -- it launders the contamination into a number
that looks like evidence.

===  =============================  ===================================================
 1   ``provenance_clean``           Only ``live_forward`` records. A run re-executed at
                                    a past cutoff re-searches today's web, so its
                                    research is shaped by what turned out to matter (R7).
 2   ``single_prompt_version``      A different prompt is a different agent. Pooling
                                    across prompt versions biases every constant fitted.
 3   ``single_package_fingerprint`` If the agent changed under the corpus, constants
                                    fitted to the old build do not transfer to the new one.
 4   ``single_agent_model``         A different LLM is a different agent. Added when the
                                    v5.2 streams made two models a live possibility rather
                                    than a hypothetical -- see below.
 5   ``sufficient_origins``         Counted in origins, never in (origin, horizon) rows.
 6   ``holdout_respected``          Judged only on origins after the candidate's own
                                    declared ``fitted_through``.
 7   ``paired_on_identical_days``   Candidate and incumbent scored on exactly the same
                                    keys, so a difference cannot come from a difference
                                    in which days each was lucky enough to be scored on (R10).
 8   ``single_tunable``             One hypothesis per cycle. Two knobs moved together
                                    cannot be attributed, and the next cycle inherits
                                    the confusion.
 9   ``effect_size_sufficient``     Statistically detectable is not the same as worth
                                    applying to a live forecast.
10   ``bootstrap_significant``      One-sided, resampling **origins** -- see below.
===  =============================  ===================================================

**Why ``single_agent_model`` is worth a condition when the streams already separate
by directory.** It is defence in depth, and the cheap half of it. The stream layout
is what actually keeps ``gemini-3.5-flash`` records out of the lite corpus; this
condition is what notices if that layout is ever bypassed -- a hand-run with an
overridden model, a merged directory, a future stream that reuses a path. The
package fingerprint would not catch it, because the same package answered both
times; only the model differed, and the model is not in the manifest.

**Why the bootstrap blocks on origin.** The three horizons of one run share an
assessment, a research packet, a market state and overlapping outcome windows. They
are close to one observation, not three. Resampling (origin, horizon) rows would
treat them as independent, shrink the standard error by roughly sqrt(3), and
manufacture significance out of correlation. So a resample draws whole origins and
takes all of their horizons with them.

Fidelity is *not* a condition here. It is a precondition, checked by
``ReplayEngine.verify_all(strict=True)`` before any scoring happens, because a
corpus the coach cannot reproduce does not produce a failed comparison -- it produces
a meaningless one.

**Passing is not applying.** A passed verdict carries a value shrunk halfway toward
the incumbent and goes to a human (R8). Nothing here writes a calibration version.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime

import numpy as np
from energy_oil_forecasting.cfm_coach.config import DEFAULT_SETTINGS, CoachSettings
from energy_oil_forecasting.cfm_coach.outcomes import ResolutionReport
from energy_oil_forecasting.cfm_coach.replay import ReplayEngine
from energy_oil_forecasting.cfm_coach.schemas import (
    CalibrationLayer,
    CalibrationVersion,
    Candidate,
    ComparisonVerdict,
    ConditionResult,
    RunRecord,
    ScoreCard,
)
from energy_oil_forecasting.cfm_coach.scoring import ForecastScorer


#: Fixed so a verdict is reproducible. A gate whose answer moves between runs on the
#: same data cannot be audited, and this one feeds a human review queue.
BOOTSTRAP_SEED = 20260814

#: Lower is better for every metric the gate supports, which is what lets the paired
#: delta be written as (incumbent - candidate) throughout.
SUPPORTED_METRICS = ("pinball", "crps", "absolute_error")

_INCUMBENT = "incumbent"
_CANDIDATE = "candidate"


class ComparisonPolicy:
    """The locked statistical gate. Judges; never fits, never applies."""

    policy_id = "cfm_coach_comparison_policy_v1"

    def __init__(
        self,
        settings: CoachSettings = DEFAULT_SETTINGS,
        *,
        replay: ReplayEngine | None = None,
        scorer: ForecastScorer | None = None,
        seed: int = BOOTSTRAP_SEED,
    ):
        self.s = settings
        self.replay = replay or ReplayEngine(settings)
        self.scorer = scorer or ForecastScorer()
        self.seed = seed

    # -- scoring --------------------------------------------------------------

    def _score_variant(
        self,
        records: dict[str, RunRecord],
        resolution: ResolutionReport,
        version: CalibrationVersion,
        *,
        variant: str,
    ) -> list[ScoreCard]:
        cards: list[ScoreCard] = []
        for resolved in resolution.resolved:
            record = records.get(resolved.run_id)
            if record is None:
                continue
            replayed = self.replay.replay(record, version, horizons=(resolved.horizon,))
            if replayed:
                cards.append(self.scorer.score_replayed(replayed[0], resolved, variant=variant))
        return cards

    # -- the bootstrap --------------------------------------------------------

    def origin_blocked_bootstrap(
        self,
        deltas_by_origin: dict[date, list[float]],
    ) -> tuple[float, float, float]:
        """One-sided p-value and a two-sided interval for the mean paired delta.

        Resamples **origins** with replacement, carrying each origin's horizons along
        with it, because horizons within an origin are not independent observations.

        The p-value is ``(1 + #{replicate <= 0}) / (B + 1)``. The ``+1`` is not
        decoration: without it a candidate can post ``p = 0`` purely because B was
        finite, which reads as certainty the resampling cannot support.
        """
        origins = sorted(deltas_by_origin)
        if not origins:
            return 1.0, float("nan"), float("nan")

        blocks = [np.asarray(deltas_by_origin[origin], dtype=float) for origin in origins]
        rng = np.random.default_rng(self.seed)
        resamples = self.s.bootstrap_resamples

        means = np.empty(resamples, dtype=float)
        picks = rng.integers(0, len(blocks), size=(resamples, len(blocks)))
        for index in range(resamples):
            means[index] = np.concatenate([blocks[choice] for choice in picks[index]]).mean()

        p_value = float((1 + np.sum(means <= 0.0)) / (resamples + 1))
        alpha = self.s.significance_alpha
        ci_low, ci_high = (float(value) for value in np.quantile(means, [alpha / 2, 1 - alpha / 2]))
        return p_value, ci_low, ci_high

    # -- shrinkage ------------------------------------------------------------

    def _shrink_value(self, candidate_value: object, incumbent_value: object, factor: float) -> object:
        """Move a tunable partway from incumbent to candidate.

        The fitted value is a point estimate from a small, autocorrelated sample. Its
        sampling error is large relative to the change being proposed, so applying it
        in full systematically over-corrects. Shrinking trades a little bias for a
        large reduction in the variance of what actually reaches a live forecast.
        Non-numeric tunables cannot be interpolated and are passed through whole.
        """
        if isinstance(candidate_value, bool) or not isinstance(candidate_value, (int, float)):
            return candidate_value
        base = (
            incumbent_value
            if isinstance(incumbent_value, (int, float)) and not isinstance(incumbent_value, bool)
            else 0.0
        )
        return float(base) + factor * (float(candidate_value) - float(base))

    def shrink(self, candidate: Candidate, incumbent: CalibrationVersion) -> tuple[dict, CalibrationLayer, float]:
        """Return the candidate pulled ``shrinkage_factor`` of the way from the incumbent."""
        factor = self.s.shrinkage_factor
        overlay = {
            key: self._shrink_value(value, incumbent.settings_overlay.get(key), factor)
            for key, value in candidate.settings_overlay.items()
        }
        layer = CalibrationLayer(
            centre_gain={
                horizon: float(self._shrink_value(value, incumbent.layer.gain_for(horizon), factor))
                for horizon, value in candidate.layer.centre_gain.items()
            },
            width_scale={
                horizon: float(self._shrink_value(value, incumbent.layer.scale_for(horizon), factor))
                for horizon, value in candidate.layer.width_scale.items()
            },
            rw_anchor={
                horizon: float(self._shrink_value(value, incumbent.layer.anchor_for(horizon), factor))
                for horizon, value in candidate.layer.rw_anchor.items()
            },
        )
        return overlay, layer, factor

    # -- the gate -------------------------------------------------------------

    def evaluate(  # noqa: PLR0912, PLR0915 - the nine conditions are the point; splitting them hides the standard
        self,
        candidate: Candidate,
        incumbent: CalibrationVersion,
        *,
        records: Sequence[RunRecord],
        resolution: ResolutionReport,
        metric: str = "pinball",
    ) -> ComparisonVerdict:
        """Judge one candidate against the incumbent. Every condition must hold."""
        if metric not in SUPPORTED_METRICS:
            raise ValueError(f"metric must be one of {SUPPORTED_METRICS}, got {metric!r}")

        conditions: list[ConditionResult] = []
        by_id = {record.run_id: record for record in records}

        # --- 1. provenance (R7) ---------------------------------------------
        eligible = {
            run_id for run_id, record in by_id.items() if record.provenance in self.s.fitting_eligible_provenance
        }
        excluded = sorted(set(by_id) - eligible)
        conditions.append(
            ConditionResult(
                name="provenance_clean",
                passed=bool(eligible),
                detail=(
                    f"{len(eligible)} of {len(by_id)} record(s) are {'/'.join(self.s.fitting_eligible_provenance)}"
                    + (f"; excluded {len(excluded)} replayed/backfilled" if excluded else "")
                ),
            )
        )
        fitting = {run_id: by_id[run_id] for run_id in eligible}

        # --- 2/3. one agent, one prompt --------------------------------------
        prompts = {record.prompt_version for record in fitting.values()}
        conditions.append(
            ConditionResult(
                name="single_prompt_version",
                passed=len(prompts) == 1,
                detail=(
                    f"prompt_version={sorted(prompts)[0]}"
                    if len(prompts) == 1
                    else f"corpus spans {len(prompts)} prompt versions {sorted(prompts)}; a different prompt is a different agent"
                ),
            )
        )
        fingerprints = {record.package_fingerprint for record in fitting.values()}
        conditions.append(
            ConditionResult(
                name="single_package_fingerprint",
                passed=len(fingerprints) == 1,
                detail=(
                    "one agent build"
                    if len(fingerprints) == 1
                    else f"corpus spans {len(fingerprints)} package fingerprints; the agent changed underneath it"
                ),
            )
        )
        # A different LLM answering the same prompt is a different agent, and the
        # package fingerprint cannot see it: the model is chosen by the run stream,
        # not shipped in the manifest. Streams keep their corpora apart on disk, so
        # this should never fire -- which is exactly why it is worth asserting.
        models = {record.agent_model for record in fitting.values()}
        conditions.append(
            ConditionResult(
                name="single_agent_model",
                passed=len(models) == 1,
                detail=(
                    f"agent_model={sorted(models)[0]}"
                    if len(models) == 1
                    else (
                        f"corpus spans {len(models)} models {sorted(models)}; a different LLM is a "
                        "different agent, and no fingerprint records which one answered"
                    )
                ),
            )
        )

        # --- score both variants on the eligible, resolved horizons ----------
        eligible_resolution = ResolutionReport(
            resolved=[item for item in resolution.resolved if item.run_id in fitting],
            pending=resolution.pending,
            data_through=resolution.data_through,
        )
        incumbent_cards = self._score_variant(fitting, eligible_resolution, incumbent, variant=_INCUMBENT)
        candidate_version = candidate.as_version("candidate", effective_from=incumbent.effective_from)
        candidate_cards = self._score_variant(fitting, eligible_resolution, candidate_version, variant=_CANDIDATE)

        # --- 4. sample size, counted in origins ------------------------------
        origins = {card.cutoff for card in incumbent_cards}
        conditions.append(
            ConditionResult(
                name="sufficient_origins",
                passed=len(origins) >= self.s.min_resolved_origins,
                detail=f"{len(origins)} resolved origin(s), need {self.s.min_resolved_origins}",
            )
        )

        # --- 5. holdout: judged only after what the candidate was fitted on ---
        holdout_origins = (
            {origin for origin in origins if origin > candidate.fitted_through}
            if candidate.fitted_through is not None
            else set(origins)
        )
        conditions.append(
            ConditionResult(
                name="holdout_respected",
                passed=len(holdout_origins) >= self.s.min_holdout_origins,
                detail=(
                    f"{len(holdout_origins)} origin(s) after fitted_through="
                    f"{candidate.fitted_through or 'none (nothing fitted; all origins out-of-sample)'}"
                    f", need {self.s.min_holdout_origins}"
                ),
            )
        )

        holdout_incumbent = [card for card in incumbent_cards if card.cutoff in holdout_origins]
        holdout_candidate = [card for card in candidate_cards if card.cutoff in holdout_origins]

        # --- 6. pairing on identical days (R10) ------------------------------
        incumbent_keys = {card.is_paired_key for card in holdout_incumbent}
        candidate_keys = {card.is_paired_key for card in holdout_candidate}
        conditions.append(
            ConditionResult(
                name="paired_on_identical_days",
                passed=bool(incumbent_keys) and incumbent_keys == candidate_keys,
                detail=(
                    f"{len(incumbent_keys)} (run, horizon) pair(s) scored for both"
                    if incumbent_keys and incumbent_keys == candidate_keys
                    else f"unpaired: {len(incumbent_keys ^ candidate_keys)} key(s) scored for only one variant"
                ),
            )
        )

        # --- 7. one tunable per cycle ----------------------------------------
        tunables = candidate.tunable_names(incumbent)
        conditions.append(
            ConditionResult(
                name="single_tunable",
                passed=0 < len(tunables) <= self.s.max_tunables_per_cycle,
                detail=(
                    f"changes {sorted(tunables)}"
                    if tunables
                    else "candidate is identical to the incumbent; nothing to test"
                ),
            )
        )

        # --- the numbers -----------------------------------------------------
        paired = self._paired_deltas(holdout_incumbent, holdout_candidate, metric=metric)
        if paired:
            deltas_by_origin, incumbent_mean, candidate_mean = paired
            all_deltas = [value for values in deltas_by_origin.values() for value in values]
            mean_delta = float(np.mean(all_deltas))
            effect_pct = float(mean_delta / incumbent_mean * 100.0) if incumbent_mean else 0.0
            p_value, ci_low, ci_high = self.origin_blocked_bootstrap(deltas_by_origin)

            conditions.append(
                ConditionResult(
                    name="effect_size_sufficient",
                    passed=effect_pct >= self.s.min_effect_size_pct,
                    detail=f"{effect_pct:+.2f}% improvement in {metric}, floor {self.s.min_effect_size_pct:.2f}%",
                )
            )
            conditions.append(
                ConditionResult(
                    name="bootstrap_significant",
                    passed=p_value < self.s.significance_alpha,
                    detail=(
                        f"one-sided p={p_value:.4f} (alpha {self.s.significance_alpha}), "
                        f"{self.s.bootstrap_resamples} resamples blocked on {len(deltas_by_origin)} origin(s)"
                    ),
                )
            )
            coverage_incumbent = float(np.mean([card.covered_80 for card in holdout_incumbent]))
            coverage_candidate = float(np.mean([card.covered_80 for card in holdout_candidate]))
            rows = len(all_deltas)
        else:
            for name in ("effect_size_sufficient", "bootstrap_significant"):
                conditions.append(ConditionResult(name=name, passed=False, detail="no paired holdout rows to compare"))
            incumbent_mean = candidate_mean = mean_delta = effect_pct = None
            p_value = ci_low = ci_high = None
            coverage_incumbent = coverage_candidate = None
            rows = 0

        overlay, layer, factor = self.shrink(candidate, incumbent)
        return ComparisonVerdict(
            candidate_id=candidate.candidate_id,
            incumbent_version=incumbent.version,
            passed=all(item.passed for item in conditions),
            conditions=conditions,
            metric=metric,
            n_origins=len(origins),
            n_holdout_origins=len(holdout_origins),
            n_scored_rows=rows,
            incumbent_score=incumbent_mean,
            candidate_score=candidate_mean,
            mean_paired_delta=mean_delta,
            effect_size_pct=effect_pct,
            p_value=p_value,
            ci_low=ci_low,
            ci_high=ci_high,
            coverage_incumbent=coverage_incumbent,
            coverage_candidate=coverage_candidate,
            shrunk_settings_overlay=overlay,
            shrunk_layer=layer,
            shrinkage_factor=factor,
            computed_at=datetime.now(),
        )

    @staticmethod
    def _paired_deltas(
        incumbent_cards: Sequence[ScoreCard],
        candidate_cards: Sequence[ScoreCard],
        *,
        metric: str,
    ) -> tuple[dict[date, list[float]], float, float] | None:
        """Per-origin paired deltas, ``incumbent - candidate`` so positive is better.

        Joined on ``(run_id, horizon)``: the same stored assessment scored against the
        same realized value, differing only in calibration. Any key present for one
        variant and not the other is dropped rather than compared, which is what makes
        the comparison paired rather than merely simultaneous.
        """
        candidate_by_key = {card.is_paired_key: card for card in candidate_cards}
        deltas_by_origin: dict[date, list[float]] = {}
        incumbent_values: list[float] = []
        candidate_values: list[float] = []

        for card in incumbent_cards:
            other = candidate_by_key.get(card.is_paired_key)
            if other is None:
                continue
            incumbent_value = getattr(card, metric)
            candidate_value = getattr(other, metric)
            deltas_by_origin.setdefault(card.cutoff, []).append(incumbent_value - candidate_value)
            incumbent_values.append(incumbent_value)
            candidate_values.append(candidate_value)

        if not incumbent_values:
            return None
        return deltas_by_origin, float(np.mean(incumbent_values)), float(np.mean(candidate_values))


__all__ = ["BOOTSTRAP_SEED", "SUPPORTED_METRICS", "ComparisonPolicy"]
