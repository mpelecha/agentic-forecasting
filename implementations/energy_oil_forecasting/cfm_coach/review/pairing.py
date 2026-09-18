"""Champion vs challenger on identical cutoffs: the comparison the gate cannot make.

`ComparisonPolicy` judges one calibration against another over the *same*
records, and refuses a corpus that mixes prompt versions or package builds. A
challenger is a different build by construction, so it is compared here: both
streams' live_forward rows are scored, joined on (cutoff, horizon), each side's
runs at a cutoff are averaged, and the paired delta is bootstrapped with the
gate's own origin-blocked resampler (`ComparisonPolicy.origin_blocked_bootstrap`,
reused unchanged). Support is counted in independent windows. Nothing is pooled
for fitting.

Usage (from the repo root)::

    uv run python -m energy_oil_forecasting.cfm_coach.review.pairing --champion v52_advanced --challenger v53_x_advanced
"""

from __future__ import annotations

import argparse
from datetime import date
from typing import Any

import numpy as np
from energy_oil_forecasting.cfm_coach.config import CoachSettings
from energy_oil_forecasting.cfm_coach.policy.comparison_policy import ComparisonPolicy
from energy_oil_forecasting.cfm_coach.report import AGENT, RANDOM_WALK
from energy_oil_forecasting.cfm_coach.review.collect import StreamCorpus, collect_stream
from energy_oil_forecasting.cfm_coach.review.settings import DEFAULT_REVIEW_SETTINGS, ReviewSettings
from energy_oil_forecasting.cfm_coach.review.stats import independent_windows
from energy_oil_forecasting.cfm_coach.streams import RunStream, stream_for
from pydantic import BaseModel, ConfigDict, Field


class PairingHorizon(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    horizon: int
    shared_cutoffs: int
    independent_windows: int
    champion_pinball: float
    challenger_pinball: float
    rw_pinball: float
    delta: float
    effect_pct: float
    champion_coverage: float
    challenger_coverage: float
    champion_hit_rate: float | None
    challenger_hit_rate: float | None
    challenger_beats_rw: bool
    champion_p50_spread: float
    challenger_p50_spread: float


class PairingReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    champion: str
    challenger: str
    metric: str = "pinball"
    shared_cutoffs: int
    min_cutoffs: int
    effective_n: float
    mean_delta: float | None
    effect_pct: float | None
    p_value: float | None
    ci_low: float | None
    ci_high: float | None
    verdict: str
    conditions: dict[str, bool]
    by_horizon: list[PairingHorizon] = Field(default_factory=list)

    def describe(self) -> str:
        head = f"{self.champion} vs {self.challenger}: {self.verdict} on {self.shared_cutoffs} shared cutoff(s), effective_n {self.effective_n:.1f}"
        if self.mean_delta is not None:
            head += f"; delta {self.mean_delta:+.4f} ({self.effect_pct:+.1f}%), p={self.p_value:.3f}, 90% CI [{self.ci_low:+.4f}, {self.ci_high:+.4f}]"
        rows = [head, "  conditions: " + ", ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in self.conditions.items())]
        for h in self.by_horizon:
            rows.append(
                f"  h{h.horizon}: cutoffs {h.shared_cutoffs} (windows {h.independent_windows}) pinball {h.champion_pinball:.3f} -> {h.challenger_pinball:.3f} (rw {h.rw_pinball:.3f}); coverage {h.champion_coverage:.0%} -> {h.challenger_coverage:.0%}; challenger beats rw {h.challenger_beats_rw}"
            )
        return "\n".join(rows)


def _per_cutoff(corpus: StreamCorpus, variant: str) -> dict[tuple[date, int], dict[str, float]]:
    """Mean pinball / coverage / p50 spread per (cutoff, horizon) over that cutoff's live_forward runs."""
    out: dict[tuple[date, int], dict[str, Any]] = {}
    live = {r.run_id: r for r in corpus.records if r.provenance == "live_forward"}
    for (run_id, h, v), card in corpus.scores.items():
        if v != variant or run_id not in live:
            continue
        bucket = out.setdefault((card.cutoff, h), {"pinball": [], "covered": [], "p50": [], "hits": []})
        bucket["pinball"].append(card.pinball)
        bucket["covered"].append(float(card.covered_80))
        bucket["p50"].append(card.p50)
        if card.direction_hit is not None:
            bucket["hits"].append(float(card.direction_hit))
    return {
        k: {
            "pinball": float(np.mean(b["pinball"])),
            "covered": float(np.mean(b["covered"])),
            "p50_spread": float(max(b["p50"]) - min(b["p50"])),
            "hit": float(np.mean(b["hits"])) if b["hits"] else None,
        }
        for k, b in out.items()
    }


def compare_corpora(
    champion: StreamCorpus,
    challenger: StreamCorpus,
    *,
    settings: ReviewSettings = DEFAULT_REVIEW_SETTINGS,
    coach_settings: CoachSettings | None = None,
) -> PairingReport:
    coach_settings = coach_settings or CoachSettings(target_agent=champion.agent_id)
    a = _per_cutoff(champion, AGENT)
    b = _per_cutoff(challenger, AGENT)
    rw = _per_cutoff(champion, RANDOM_WALK)
    shared = sorted(set(a) & set(b) & set(rw))
    deltas_by_origin: dict[date, list[float]] = {}
    by_h: dict[int, list[tuple[date, dict, dict, dict]]] = {}
    for key in shared:
        cutoff, h = key
        deltas_by_origin.setdefault(cutoff, []).append(a[key]["pinball"] - b[key]["pinball"])
        by_h.setdefault(h, []).append((cutoff, a[key], b[key], rw[key]))
    cutoffs = sorted(deltas_by_origin)
    n_eff = sum(len({c for c, *_ in rows}) / h for h, rows in by_h.items())

    rows_out = []
    for h, rows in sorted(by_h.items()):
        cs = [c for c, *_ in rows]
        champ = float(np.mean([x["pinball"] for _, x, _, _ in rows]))
        chal = float(np.mean([y["pinball"] for _, _, y, _ in rows]))
        rwp = float(np.mean([z["pinball"] for _, _, _, z in rows]))
        hits_a = [x["hit"] for _, x, _, _ in rows if x["hit"] is not None]
        hits_b = [y["hit"] for _, _, y, _ in rows if y["hit"] is not None]
        rows_out.append(
            PairingHorizon(
                horizon=h,
                shared_cutoffs=len(cs),
                independent_windows=independent_windows(cs, h),
                champion_pinball=champ,
                challenger_pinball=chal,
                rw_pinball=rwp,
                delta=champ - chal,
                effect_pct=(champ - chal) / champ * 100 if champ else 0.0,
                champion_coverage=float(np.mean([x["covered"] for _, x, _, _ in rows])),
                challenger_coverage=float(np.mean([y["covered"] for _, _, y, _ in rows])),
                champion_hit_rate=float(np.mean(hits_a)) if hits_a else None,
                challenger_hit_rate=float(np.mean(hits_b)) if hits_b else None,
                challenger_beats_rw=chal < rwp,
                champion_p50_spread=float(np.mean([x["p50_spread"] for _, x, _, _ in rows])),
                challenger_p50_spread=float(np.mean([y["p50_spread"] for _, _, y, _ in rows])),
            )
        )

    conditions = {
        "same_cutoffs_only": True,
        "both_live_forward": True,
        "min_distinct_cutoffs": len(cutoffs) >= settings.pairing_min_cutoffs,
    }
    if not cutoffs:
        return PairingReport(
            champion=champion.stream_id,
            challenger=challenger.stream_id,
            shared_cutoffs=0,
            min_cutoffs=settings.pairing_min_cutoffs,
            effective_n=0.0,
            mean_delta=None,
            effect_pct=None,
            p_value=None,
            ci_low=None,
            ci_high=None,
            verdict="no shared cutoffs",
            conditions=conditions,
            by_horizon=[],
        )
    policy = ComparisonPolicy(coach_settings)
    p_value, ci_low, ci_high = policy.origin_blocked_bootstrap(deltas_by_origin)
    all_deltas = [d for ds in deltas_by_origin.values() for d in ds]
    mean_delta = float(np.mean(all_deltas))
    champ_mean = float(np.mean([a[k]["pinball"] for k in shared]))
    effect = mean_delta / champ_mean * 100 if champ_mean else 0.0
    conditions["effect_size_sufficient"] = effect >= coach_settings.min_effect_size_pct
    conditions["bootstrap_significant"] = p_value < coach_settings.significance_alpha
    conditions["powered"] = n_eff >= settings.min_effective_n_powered
    if not conditions["min_distinct_cutoffs"]:
        verdict = "withheld (too few shared cutoffs)"
    elif all(conditions.values()):
        verdict = "challenger better"
    elif conditions["effect_size_sufficient"] and conditions["bootstrap_significant"]:
        verdict = "challenger better, underpowered"
    elif mean_delta < 0 and p_value > 1 - coach_settings.significance_alpha:
        verdict = "challenger worse"
    else:
        verdict = "no difference established"
    return PairingReport(
        champion=champion.stream_id,
        challenger=challenger.stream_id,
        shared_cutoffs=len(cutoffs),
        min_cutoffs=settings.pairing_min_cutoffs,
        effective_n=float(n_eff),
        mean_delta=mean_delta,
        effect_pct=effect,
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
        verdict=verdict,
        conditions=conditions,
        by_horizon=rows_out,
    )


def compare_streams(
    champion: RunStream, challenger: RunStream, *, settings: ReviewSettings = DEFAULT_REVIEW_SETTINGS
) -> PairingReport:
    return compare_corpora(
        collect_stream(champion), collect_stream(challenger), settings=settings, coach_settings=champion.coach_settings
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--champion", required=True)
    parser.add_argument("--challenger", required=True)
    args = parser.parse_args(argv)
    print(compare_streams(stream_for(args.champion), stream_for(args.challenger)).describe())


if __name__ == "__main__":
    main()


__all__ = ["PairingHorizon", "PairingReport", "compare_corpora", "compare_streams"]
