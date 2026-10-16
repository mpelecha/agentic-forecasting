# CFM Agent v6.0 calibration note

This note records how the centre and uncertainty transformations are derived from history, the
values they take on the reference history, and the measured consequences of the locked choices
(specification Sections 1.8 and 3; DD-06, DD-08, DD-09, DD-10). The engine recomputes every value
at every cutoff; the numbers below are illustrative measurements on the history to 2026-09-14.

## 1. Centre percentiles

The centre action selects a percentile of the eligible, overlapping, horizon-matched WTI returns
in the history window (setting `center_percentiles`). The adjustment is the baseline P50 times
that return, guarded so that a down action never moves the centre up and an up action never moves
it down, scaled by the novelty multiplier, and capped at
`min(emergency_center_cap_usd, emergency_center_cap_price_fraction × latest WTI)`.

| Centre action | Percentile | h=5 | h=10 | h=21 |
|---|---|---|---|---|
| Large down | P10 | -6.09% | -8.36% | -11.56% |
| Moderate down | P25 | -2.75% | -3.76% | -5.53% |
| Small down | P40 | -0.74% | -0.90% | -1.40% |
| No change | 0 | 0 | 0 | 0 |
| Small up | P60 | +1.43% | +2.19% | +3.58% |
| Moderate up | P75 | +3.25% | +4.62% | +7.37% |
| Large up | P90 | +6.19% | +8.65% | +12.98% |

Because the percentiles straddle a positive median, the down and up steps of the same size differ
in magnitude. The directional guard is needed: a trailing-window P40 was positive in 22% of
520-day windows and would otherwise have moved a small-down action up (DD-09).

## 2. Coverage-shaped uncertainty ladder

Each uncertainty rung names a nominal historical coverage (setting `uncertainty_coverage`). The
band at coverage c is the c-th percentile of the absolute deviations of the eligible horizon
returns from their median. The multiplier is the band at the rung's coverage divided by the band
at the anchor coverage (80%, the coverage of the ensemble's own P10 to P90 interval), and it
scales the ensemble's quantile distances around the adjusted P50 (ADR 0001).

| Uncertainty action | Coverage | h=5 | h=10 | h=21 | v5.2 constant |
|---|---|---|---|---|---|
| Substantially narrower | 65% | 0.69 | 0.71 | 0.73 | 0.70 |
| Moderately narrower | 70% | 0.77 | 0.79 | 0.81 | 0.80 |
| Small narrower | 75% | 0.87 | 0.89 | 0.90 | 0.90 |
| Unchanged | 80% | 1.00 | 1.00 | 1.00 | 1.00 |
| Small wider | 85% | 1.14 | 1.15 | 1.14 | 1.10 |
| Moderately wider | 90% | 1.36 | 1.36 | 1.35 | 1.20 |
| Substantially wider | 95% | 1.74 | 1.73 | 1.69 | 1.30 |

WTI's fat tails make the widening rungs larger than v5.2's constants while the narrowing rungs are
close to them. The earlier dollar-target design was replaced because, measured on the v6.0
bootstrap at h=21, the P60 and P75 absolute-deviation targets ($8.31 and $11.43) sat below the
baseline half-width ($13.84) and turned small and moderate widening into no-ops (DD-10).

## 3. Novelty multipliers

| Novelty | Multiplier |
|---|---|
| likely_new_relative_to_model_data | 1.00 |
| possibly_partly_reflected | 0.75 |
| likely_reflected_in_model_data | 0.50 |
| indeterminate | 0.50 |

All four classes may accompany a non-neutral action; the multiplier applies after the percentile
is selected and before the emergency cap (DD-11).

## 4. Recorded consequences of the locked choices

- **April 2020.** CL=F settled at -$37.63 on 2020-04-20. Floored to $1.00, the daily returns are
  -94.5% on 2020-04-20 and +901.0% on 2020-04-21, and both remain in every full-history fit.
  Measured effect on the Sparse AR fit: residual standard deviation 12.5% per day (2.6% with the
  two returns excluded); intercept +0.224% per day (+0.055% excluded). In the bootstrap, 2 of 1,000
  21-step paths finished above twice the last price (maximum $1,325 from $102.58); P90 is
  unaffected (DD-06).
- **Drift.** The full-history mean daily return is +0.055% per day. The bootstrap median at h=21
  sits about +1.25% above the last price. With the AR intercept above at 25% weight, the ensemble
  P50 is expected to sit roughly $1.50 to $2.00 above the last price at h=21 against a historical
  mean forward move of +$0.18; v5.2 sat about $2.60 below (DD-08).
- **Minimum history.** Fewer than `min_historical_returns` (500) eligible horizon returns force a
  neutral transformation of both levers with the reason recorded.
- **Regime conditioning.** Realised-volatility regimes show a two-fold variation in band width;
  the `historical_conditioning` switch ships off pending scored evidence (DD-23).

## 5. Sensitivity

<!-- sensitivity:begin -->
Run at cutoff 2026-09-15 on 2026-09-16 05:28 UTC. Excluded dates: 2020-04-20, 2020-04-21. Informational only; the production model keeps every observation (DD-06).

| Quantity | With excluded dates | Without excluded dates |
|---|---|---|
| Returns available | 5705 | 5703 |
| Data fingerprint | `9f828b0020bed851` | `f39c94d2850712cb` |
| Model-suite ID | `00059933fb52936a` | `d3c01838942d6252` |
| Sparse AR intercept (per day) | 0.002237 | 0.000567 |
| Sparse AR residual standard deviation (per day) | 0.1253 | 0.0263 |

| Component | Horizon | P10 with | P50 with | P90 with | P10 without | P50 without | P90 without |
|---|---|---|---|---|---|---|---|
| bootstrap | 5 | 94.85 | 101.72 | 107.75 | 95.33 | 101.63 | 107.28 |
| bootstrap | 10 | 92.34 | 102.17 | 111.18 | 93.26 | 101.50 | 110.64 |
| bootstrap | 21 | 88.46 | 102.41 | 117.73 | 89.81 | 101.94 | 116.07 |
| sparse_ar | 5 | 95.20 | 101.73 | 107.37 | 94.42 | 101.45 | 107.06 |
| sparse_ar | 10 | 93.09 | 101.82 | 111.15 | 91.89 | 101.36 | 110.03 |
| sparse_ar | 21 | 88.31 | 101.99 | 115.75 | 87.13 | 100.59 | 114.50 |
| lightgbm | 5 | 94.19 | 101.66 | 107.12 | 94.25 | 100.29 | 105.83 |
| lightgbm | 10 | 89.61 | 100.78 | 109.95 | 90.02 | 99.80 | 108.46 |
| lightgbm | 21 | 82.27 | 99.37 | 112.66 | 86.61 | 99.66 | 113.40 |
| ensemble | 5 | 94.77 | 101.71 | 107.50 | 94.83 | 101.25 | 106.86 |
| ensemble | 10 | 91.84 | 101.73 | 110.87 | 92.11 | 101.04 | 109.94 |
| ensemble | 21 | 86.88 | 101.54 | 115.97 | 88.34 | 101.03 | 115.01 |

| Horizon | Percentile | Return with | Return without |
|---|---|---|---|
| 5 | P10 | -0.0609 | -0.0609 |
| 5 | P25 | -0.0275 | -0.0275 |
| 5 | P40 | -0.0074 | -0.0075 |
| 5 | P60 | 0.0143 | 0.0143 |
| 5 | P75 | 0.0325 | 0.0325 |
| 5 | P90 | 0.0619 | 0.0618 |
| 10 | P10 | -0.0836 | -0.0834 |
| 10 | P25 | -0.0376 | -0.0375 |
| 10 | P40 | -0.0090 | -0.0090 |
| 10 | P60 | 0.0219 | 0.0219 |
| 10 | P75 | 0.0462 | 0.0462 |
| 10 | P90 | 0.0865 | 0.0864 |
| 21 | P10 | -0.1156 | -0.1154 |
| 21 | P25 | -0.0553 | -0.0552 |
| 21 | P40 | -0.0140 | -0.0140 |
| 21 | P60 | 0.0358 | 0.0358 |
| 21 | P75 | 0.0737 | 0.0737 |
| 21 | P90 | 0.1298 | 0.1297 |

| Horizon | Uncertainty rung | Multiplier with | Multiplier without |
|---|---|---|---|
| 5 | substantially_narrower | 0.6920 | 0.6921 |
| 5 | moderately_narrower | 0.7731 | 0.7732 |
| 5 | small_narrower | 0.8711 | 0.8717 |
| 5 | unchanged | 1.0000 | 1.0000 |
| 5 | small_wider | 1.1447 | 1.1442 |
| 5 | moderately_wider | 1.3623 | 1.3632 |
| 5 | substantially_wider | 1.7330 | 1.7331 |
| 10 | substantially_narrower | 0.7066 | 0.7073 |
| 10 | moderately_narrower | 0.7924 | 0.7930 |
| 10 | small_narrower | 0.8901 | 0.8898 |
| 10 | unchanged | 1.0000 | 1.0000 |
| 10 | small_wider | 1.1541 | 1.1543 |
| 10 | moderately_wider | 1.3613 | 1.3622 |
| 10 | substantially_wider | 1.7334 | 1.7320 |
| 21 | substantially_narrower | 0.7278 | 0.7273 |
| 21 | moderately_narrower | 0.8059 | 0.8068 |
| 21 | small_narrower | 0.8957 | 0.8970 |
| 21 | unchanged | 1.0000 | 1.0000 |
| 21 | small_wider | 1.1392 | 1.1409 |
| 21 | moderately_wider | 1.3500 | 1.3517 |
| 21 | substantially_wider | 1.6936 | 1.6916 |
<!-- sensitivity:end -->
