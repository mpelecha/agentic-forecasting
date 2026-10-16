## CFM Agent v6.0 live validation

Run on 2026-09-16 (build step 9) through the Vector Institute proxy, on the frozen data snapshot
`outputs/cfm_agent_v_6_0/data_snapshot_20260916T052728Z/` at cutoff 2026-09-15 (latest
cutoff-safe WTI observation 2026-09-14, $101.39). Spend is capped at `live_spend_ceiling_usd`
(3.00 USD) under the accounting of specification Section 10.1 (DD-22). Each run was a fresh
process; the harness set `CFM_V60_DATA_SNAPSHOT` and handed the saved preflight result to the
predictor, so both are in every audit record. Raw run records (predictions, complete audit, token
usage) are under `outputs/cfm_agent_v_6_0/live/` and are not committed.

### Sequence and results
0. **Preflight.** `release.py preflight-models` at 05:28 UTC via the proxy's models endpoint (57
   models listed): `gemini-3.1-flash-lite-preview` available, `gemini-3.5-flash` available.
1. **Run 1** (audit on, E2B enabled): 62 s. Initial assessment failed validation (`physical_status`
   = "tight", outside the enumeration); the one correction call repaired it
   (`corrected_response_accepted`). Confidence 0.90, novelty likely new, four claims; one claim
   rejected on clause `resolved_associated_source`. Tier 3 at every horizon (two independent
   publishers, confidence at or above 0.80). Actions: h5 and h10 large up and substantially wider,
   h21 moderate up and substantially wider. P50 moved from 101.71 to 108.00 at h5 (P90 historical
   5-day return +6.19% × 1.0 novelty, +$6.30) and from 101.54 to 109.02 at h21. The model did not
   call the E2B tool (0 accepted, 0 rejected). Claim-support verifier executed and completed
   (two direct, two partial findings).
   A first attempt at run 1 completed its paid calls and then failed in the harness on a Langfuse
   401 before the record was written; its usage was not captured and is charged below at the
   cost of the successful run 1.
   Projection after run 1: 0.1228 × 2.2 = 0.27 USD; actual plus projection 0.52 USD, under the ceiling, so the sequence continued.
2. **Run 2** (identical, new process): 69 s. Data fingerprint, suite seed, the three component
   seeds, every component and ensemble quantile, the diagnostics, and the model-suite ID
   (`…00059933fb52936a`) equal run 1 exactly. The LLM path differed as expected: a different
   research packet id, the same `physical_status` validation error corrected on the retry, and
   actions moderate up and small wider at h5, moderate up and moderately wider at h10, small up
   and substantially wider at h21 (P50 105.01 at h5).
3. **Correction path**: 17 s. Research inputs replayed from run 1 (search results, verdicts, and
   resolutions stubbed from the recorded packet), the numerical suite recomputed (same suite ID),
   the initial response forced invalid, so the only live call was the structured-output correction
   (`gemini-3.5-flash`, one call, 719 in and 2,710 out). Disposition `corrected_response_accepted`;
   the corrected assessment carried one claim at confidence 0.50 and neutral actions at every
   horizon (tier none), reproducing the ensemble exactly.
4. **Langfuse**: `auth_check` failed with 401 Unauthorized ("Invalid credentials. Confirm that
   you've configured the correct host") for the keys in the repository `.env` against the default
   cloud host; the span exporter logged the same 401 during every run. Trace ids were generated in
   process (run 1 `ed1f1fcc5a559d73bacdf0bff991c321`, run 2 `716407d0cc38be613d125e5168809a3d`)
   but no trace was exported. The coach's RUNNING.md records the same pre-existing Langfuse
   authentication failure. **Gate not met**; nothing in the package depends on it. Because no
   Langfuse trace exists, usage was counted in process from the `usage` field of every LiteLLM
   response (a `litellm` success callback registered by the harness), which covers the same
   generations the trace would.
5. **Effective LLM parameters** (from the `llm_effective_parameters` audit key, identical in both
   runs): main role `openai/gemini-3.1-flash-lite-preview`, timeout 180, top_p 1.0, n 1,
   frequency and presence penalties 0.0, parallel_tool_calls false, drop_params true, with
   temperature 0.1 and 24,576 output tokens through AgentConfig; search role four calls at
   temperature 0.1, max_tokens 4,096, timeout 60 with the `googleSearch` tool; cutoff verifier
   four calls at temperature 0.0, max_tokens 4,096, JSON-schema response format; correction one
   call at temperature 0.0, max_tokens 8,192; claim-support verifier one call at temperature 0.0,
   max_tokens 4,096. `parameters_dropped_by_proxy` was empty for every call and the proxy echoed
   the requested model name in every response.
6. **Sensitivity**: `release.py sensitivity --cutoff 2026-09-15` on the snapshot; table in
   CALIBRATION.md Section 5.
7. **E2B, direct tool exercise** (added because the model made no request in runs 1 and 2): one
   sandbox launched through the hardened tool with the Python-defined preamble; a request importing
   `os` was rejected (`prohibited_module:os`) without a launch, the accepted script ran in 1.0 s and
   printed the horizon-21 ensemble width and the cumulative returns, and a third request was
   refused with `accepted_budget_exhausted`. The accepted script raised a `KeyError` on
   `TASK_BINDING["cutoff_date"]` (the key is `cutoff`); the tool captured it in `error` and kept
   the output, as designed. Sandbox time recorded, not priced.

### Prices
| Model | Input USD per 1M tokens | Output USD per 1M tokens | Source | Retrieved |
|---|---|---|---|---|
| gemini-3.1-flash-lite-preview | 0.25 | 1.50 (thinking tokens included) | Google AI for Developers, Gemini Developer API pricing page (the proxy publishes no price list; its `/model/info` endpoint is unavailable) | 2026-09-16 |
| gemini-3.5-flash | 1.50 | 9.00 (thinking tokens included) | same | 2026-09-16 |

### Spend
| Run | Model | Calls | Input tokens | Output tokens | Cost USD |
|---|---|---|---|---|---|
| Run 1, first attempt (record lost, see 1) | both | — | not captured | not captured | 0.1228 (charged at the run 1 cost) |
| Run 1 | gemini-3.1-flash-lite-preview | 9 | 258,229 | 4,699 | 0.0716 |
| Run 1 | gemini-3.5-flash | 6 | 6,340 | 4,632 | 0.0512 |
| Run 2 | gemini-3.1-flash-lite-preview | 9 | 258,651 | 4,966 | 0.0721 |
| Run 2 | gemini-3.5-flash | 6 | 6,526 | 5,299 | 0.0575 |
| Correction path | gemini-3.5-flash | 1 | 719 | 2,710 | 0.0255 |

Total 0.4007 USD against a ceiling of 3.00 USD. Preflight and E2B sandbox time unpriced.

### Gates
| Gate | Result |
|---|---|
| Two identical runs agree numerically | Met |
| Correction path exercised live | Met (in the dedicated run, and incidentally in runs 1 and 2) |
| Grounded search and cutoff verification live | Met (four accepted summaries with 10 to 15 passages each in every run) |
| Effective parameters recorded | Met |
| Spend under the ceiling with the accounting rule | Met |
| One E2B execution | Met at the tool level (item 7); the model issued no request inside the agent turn |
| Langfuse trace present | Not met: 401 for the configured credentials and host (item 4) |

### Observations for the owner (no code change made)
- Both live assessments emitted `physical_status` = "tight", which the schema rejects; the
  correction call repaired it each time at about 0.03 USD. Adding "tight" to the enumeration or
  tightening the prompt wording would remove one paid call per run.
- Tier 3 was reached from publishers such as blog.caveofx.com and oilreviewmiddleeast.com; the
  source-selection hierarchy admits them as independent publishers. This is the specified policy
  (DD-12), recorded here because it moved the centre by the P90 historical return.
