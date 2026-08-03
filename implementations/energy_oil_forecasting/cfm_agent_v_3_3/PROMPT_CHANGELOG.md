# Prompt and skill change log

Entries follow the template in *CFM Agent v3.3 — Prompt and Skill Work
Guidelines*, section 3.

---

## 2026-08-02 — Pillar boundary in evidence assessment

**File changed:** `skills/evidence-assessment/SKILL.md`

**Behavioral objective:** Stop the agent recording scheduled statistical
publications as material claims, so that its evidence record covers only
drivers the quantitative and physical-fundamentals pillars cannot already see.

**Instructions added, removed, or rewritten:** Added one section, "Pillar
boundary: record events, not statistics." It names two excluded classes
(structured market data; scheduled statistical levels and revisions), states a
general test — would another pillar eventually read this in its own input
data — and prefers primary event reporting over market-wrap or outlook
articles. Nothing was removed or rewritten.

The section explicitly preserves the existing claim types. A production,
inventory, or strategic-reserve claim stays legitimate when its substance is an
event: a decision, an announced release, a confirmed disruption, or an outage.
The exclusion applies only when the substance is a level or a revision from a
scheduled report. This wording was chosen because `claim_type` in `schemas.py`
includes `inventory`, `production_policy`, and `strategic_reserves`; a blanket
exclusion would have contradicted the protected schema, and search step 3 asks
for exactly those responses.

**Why the structured output contract is unchanged:** No field, type, enum, or
category was touched. No file outside the approved prompt and skill set was
modified. The instruction adds no numerical authorship: it constrains which
claims are recorded, not how any number is produced. Python remains the sole
owner of sanitization, evidence-tier assignment, action approval, adjustment
caps, and final arithmetic.

**Test or repeated-run comparison:** All four repository-native validation
commands were run. Compilation succeeds and the suite remains at **17 passed**,
as the guidelines expect. This change adds no new lint or formatting findings —
it edits a Markdown file only.

Two pre-existing findings are worth flagging separately, because the guidelines
state the expected result as "Ruff formatting and linting pass," and v3.3 does
not currently meet that bar at baseline:

- `ruff format --check` would reformat three files: `policy/constrained.py`,
  `predictor.py`, and `sanitizer.py`.
- `ruff check` reports nine errors: four import-ordering, and five
  branch/statement-count warnings in `sanitizer.py` and `policy/constrained.py`.

All of these are in protected files and predate this change, so they were left
alone. They should be cleared under a separate approved development task, not
folded into prompt work.

The repeated-run comparison is **not yet done**. It requires live search
against fixed historical dates, which needs credentials and network access. See
"Observed result" for what the same change produced in a sibling package.

**Observed result:** Pending for v3.3. The equivalent wording was applied to a
sibling event-scoring agent and compared across runs at the same origin date
(2026-03-02). Before the change, two of four recorded drivers were scheduled
statistics — an IEA demand-forecast revision and a monthly supply figure — and
the dominant event of that date was absent from the record. After the change,
the statistics disappeared from the record and the dominant event appeared as
the leading driver. That is indicative only; it is a different package, and the
v3.3 comparison should be run before this entry is considered closed.

### Suggested reviewer checks
- Only `skills/evidence-assessment/SKILL.md` changed, plus this log.
- No field, type, enum, or category changed.
- No instruction asks for prices, quantiles, evidence tiers, dollar changes, or
  final arithmetic.
- Cutoff, provenance, direct-support, source-quality, and conflict instructions
  are intact.
- Ruff formatting, linting, compilation, and all 17 tests pass.
