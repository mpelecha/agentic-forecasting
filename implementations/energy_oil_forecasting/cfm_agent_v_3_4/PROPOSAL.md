# Proposal: contract changes for a CFM Agent v3.4

**Status: proposed, not approved. Do not treat this package as production.**

*CFM Agent v3.3 — Prompt and Skill Work Guidelines* separates two lanes. Prompt
and skill wording may be edited freely. The output contract, the numerical
suite, the evidence policy, and the final arithmetic are protected, and
`config.py` changes "only through a separately approved development task."

Everything in this document is in the second lane. It is written for review
before any of it is adopted, not as work already done. A working reference
implementation exists in this package so the proposal can be read as running
code rather than prose, and so the cost of each change is visible.

A separate, already-validated change sits in the first lane and does not depend
on any of this: the pillar-boundary section added to v3.3's
`skills/evidence-assessment/SKILL.md`, recorded in that package's
`PROMPT_CHANGELOG.md`. It ships independently.

---

## Why propose anything at all

v3.3 moved the adjustment decision out of the LLM and into a deterministic,
auditable, versioned config. That is the right direction and this proposal does
not touch it.

But the constants that decision rests on — `small_action_width_fraction = 0.10`,
`moderate_action_width_fraction = 0.20`, `max_center_adjustment_usd = 3.0`, the
uncertainty multipliers, and the persistence-profile coefficients — are
judgment calls, not fitted values. Nothing yet shows that a `moderate_up` should
move the center by 20% of the ensemble's p10-p90 width rather than 5% or 40%.

The four changes below exist to make those constants *learnable*. Three of them
are the machinery for a calibration study; the fourth is a discipline that the
study needs in order to have a clean uncertainty signal.

---

## Change 1: driver categories on claims

**File:** `schemas.py` (protected)

**Proposal:** add `driver_category` to `EvidenceClaim`, one of `geopolitical`,
`weather`, `operational`, `policy`, `demand_expectations`.

**Why:** a calibration regression needs a fixed-width feature row — the same
columns on every origin date. Claim text changes every run, so it cannot be a
feature. A closed category enum can. This is the smallest field that makes
per-category coefficients possible, so a hurricane's price signature can be
fitted separately from a sanctions regime's.

**Relationship to `claim_type`:** orthogonal, not a replacement. `claim_type`
says what kind of fact a claim is; `driver_category` says what kind of driver
produced it. A hurricane that shuts in production is `physical_supply` and
`weather`.

**Contract cost:** one added field. Defaults to `geopolitical`, so an
assessment written against the v3.3 contract still parses.

---

## Change 2: competing scenarios

**Files:** `schemas.py`, `outputs.py` (both protected)

**Proposal:** add a `ContextScenario` model and a `scenarios` list — two to
three storylines, each with a probability, a direction (`up`/`down`/`neutral`),
a magnitude (`small`/`moderate`/`large`), and one required low-probability,
high-impact tail case.

**Why:** scenarios force the assessment to name the alternative before
committing to an action. In sibling-package testing, the discipline that
mattered most was the required tail case: without it the model reliably
produced one storyline and an uncertainty action that did not reflect the
genuine two-sidedness of the situation.

**Why categorical:** for the same reason actions are categorical. The LLM
states structure and direction; it never states a price. This proposal does not
weaken that rule.

**Contract cost:** one added field and one new model. An absent scenario set is
tolerated and reproduces v3.3 behaviour exactly, so this degrades safely rather
than failing.

**Known gap:** v3.3 has no retry loop — `AgentPredictor` raises on validation
failure rather than re-prompting. That is why malformed scenario sets are
rejected but absent ones are not: hard-requiring scenarios without a retry loop
would turn a formatting slip into a failed backtest origin. If the team wants
scenarios mandatory, a bounded retry loop should be proposed alongside.

---

## Change 3: scenario-derived uncertainty floor

**Files:** `policy/constrained.py`, `config.py` (both protected)

**Proposal:** when the assessment's own scenario set places at least
`scenario_disagreement_floor_mass` (default 0.20) of probability on the losing
price direction, Python widens the interval to at least `moderately_wider`,
whatever uncertainty action the LLM proposed.

**Why:** without this, scenarios are decorative — the model can state a
genuinely two-sided situation and still propose `unchanged` uncertainty, and
nothing catches the contradiction. This makes the scenario set load-bearing.

**Why it is independent of the evidence tier:** because it widens only and
never moves the center. Widening cannot manufacture a directional view from
weak evidence; it can only make the forecast less confident. The proposed
division is that the evidence gate governs the center and the scenario set
governs the spread.

**This is the one change that could plausibly hurt scores.** Widening is not
free — CRPS punishes over-dispersion as well as under-dispersion. It should be
measured against the ensemble-locked control before adoption, not assumed.

---

## Change 4: calibration feature row

**File:** `predictor.py` (protected)

**Proposal:** record `calibration_features` in prediction metadata — a
fixed-width numeric row built from the *sanitized* assessment, so
sanitizer-rejected evidence cannot inflate a feature.

**Why:** this is the point of the other three. Paired with the
`unadjusted_ensemble` that v3.3 already records, it supplies both sides of the
study: the features, and the residual of the actual move against the quant
baseline's expectation. Regressing scores against the *residual* rather than the
raw price move is what isolates the part of the move the other pillars did not
already explain.

**Contract cost:** metadata only. No field on the LLM's output contract, no
change to any number the agent produces. This is the cheapest of the four and
could be adopted alone.

---

## Suggested sequence

The four changes are separable, and the order matters more than the content.

1. **Adopt Change 4 alone.** Metadata only, no contract change, no behavioural
   risk. Start recording feature rows on every run.
2. **Run the multi-origin study.** Forty-odd origins with the existing v3.3
   behaviour. Fit features against quant-baseline residuals. Ask the honest
   question first: do the assessment's signals predict anything the quant
   models missed? If not, Changes 1-3 do not matter.
3. **Adopt Changes 1 and 2 if the answer is yes**, and refit with the richer
   features to see whether categories and scenarios improve the fit.
4. **Adopt Change 3 last, and only with a measured comparison** against the
   ensemble-locked control.

This ordering answers the cheapest question first and keeps every later step
gated on evidence rather than judgment.

---

## Validation status of the reference implementation

Run in the repository environment:

| Check | Result |
|---|---|
| `ruff check` | passes |
| `ruff format --check` | 5 files would reformat — 3 pre-existing from v3.3, 2 introduced here |
| `compileall` | passes |
| `pytest` | 33 passed — the 17 inherited from v3.3 unchanged, plus 16 new |

Backward compatibility is tested in both directions: a claim without
`driver_category` parses and defaults; an assessment without `scenarios` parses
and applies no floor, reproducing v3.3 behaviour.

**Not done:** any live run. No multi-date comparison, no evidence-tier
distribution study, and no measurement of whether any of this improves CRPS
against the ensemble-locked control. Nothing here should be adopted on the
strength of the implementation alone.

---

## Open question worth deciding first

An earlier design discussion raised putting per-claim, per-scenario scores in a
matrix — cells rather than a single direction and magnitude per scenario. It
was deliberately deferred, and this proposal keeps the simpler categorical
form.

The reason is worth recording: a matrix keyed by claim text cannot be a
regression input, because the claims change every run. Richer cells only help
if they collapse into stable slots, which is what `driver_category` already
provides. If the team wants the matrix, the aggregation rule should be decided
before the schema, not after.
