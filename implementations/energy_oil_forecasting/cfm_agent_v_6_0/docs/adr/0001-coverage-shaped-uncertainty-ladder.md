# ADR 0001: Coverage-shaped uncertainty ladder

Status: accepted (2026-09-16). Specification Section 3.3; decision DD-10.

## Context

The earlier v6.0 planning documents defined each uncertainty action as a dollar
target: the absolute-deviation percentile of historical horizon-matched moves
(P10 to P90) became the target half-width, and the ensemble band was rescaled to
meet it. Measured on the v6.0 bootstrap at h=21, the P60 and P75 targets ($8.31
and $11.43) were below the baseline half-width ($13.84). The scale factors 0.60
and 0.83 that resulted were turned into no-ops by the rule that a wider action
cannot narrow, so two of the three widening rungs did nothing, while P40, P25,
and P10 cut the width by 62 to 90 percent. The cause is structural: the
ensemble's P10-P90 band sits at the 80th to 84th percentile of historical
absolute deviation, not the 50th, so targets drawn below that percentile can only
narrow.

## Decision

Each uncertainty action names a nominal historical coverage (65, 70, 75, 80, 85,
90, 95 percent; setting `uncertainty_coverage`). The ensemble's P10-P90 band is,
by construction, an 80 percent band, so the `unchanged` action is anchored at the
historical 80 percent band (`uncertainty_anchor_coverage`). The multiplier for a
rung is the ratio of the historical absolute-deviation band at the rung's
coverage to the anchor band:

    Band(h, c)             = percentile(|r_h - median(r_h)|, c)
    Multiplier(h, action)  = Band(h, Coverage[action]) / Band(h, AnchorCoverage)
    AdjustedQuantile(q, h) = AdjustedP50(h) + Multiplier(h, action) * (BaselineQuantile(q, h) - BaselineP50(h))

The multiplier scales the ensemble's own quantile distances around the adjusted
P50. `unchanged` yields exactly 1.0. A wider action yields at least 1.0 and a
narrower action at most 1.0 by construction (percentiles are monotone in the
coverage); the engine asserts this and clamps only on an exact tie, with a
warning. The settings loader requires the ladder to be strictly increasing and
the anchor to equal the `unchanged` rung.

## Consequences

- The ensemble's shape is preserved: every quantile moves by the same factor
  relative to the adjusted P50, so the relative spacing of the eleven levels is
  unchanged and no rung can be a no-op.
- Coverage moves by a known amount. Widening from the 80 percent band to the
  85, 90, or 95 percent band, or narrowing to 75, 70, or 65, is a statement
  about historical coverage rather than about dollars, and the engine recomputes
  the ratios at every cutoff from the same history window the models use.
- WTI's fat tails make the widening rungs larger than v5.2's constants
  (full-history values on 2026-09-14: 1.14, 1.36, 1.74 at h=5 against 1.10,
  1.20, 1.30), while the narrowing rungs are close to them (0.69, 0.77, 0.87
  against 0.70, 0.80, 0.90). The values are recorded in CALIBRATION.md and in
  every transformation record through the `HistoricalReturnProfile`.
- Fewer than `min_historical_returns` eligible horizon returns, or a zero anchor
  band, forces a neutral transformation with the reason recorded, so the ladder
  is never applied to an under-sampled profile.
