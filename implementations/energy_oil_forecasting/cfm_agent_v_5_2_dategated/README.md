# cfm_agent_v_5_2_dategated

CFM Agent v5.2 (ARIMA-only ensemble) with a deterministic publication-date gate on
its news research. Everything else — model, persona, skills, tools, numerical suite,
evidence policy, forecast engine, settings — is v5.2's, and the agent's prompt and
tool schema are identical to v5.2's (a test checks this).

Requires `cfm_agent_v_5_2` in the same package; this variant does not modify it.

## Why

v5.2 searches today's web for past cutoffs. It asks the search model for information
"published strictly before" the cutoff and has an LLM verifier remove post-cutoff
claims, but nothing checks page dates. In cached v5.2 backtests, sources whose URL
alone carries a date on or after the origin are common (for example a Statistics
Canada release dated 2026-08-20 cited for a 2025-07-07 origin);
`scripts/scan_source_url_dates.py` reproduces the count. Because each topic summary
is written from all of its grounding pages, dropping a late page from the source list
does not remove its content from the summary.

## The gate

Each research topic runs v5.2's grounded search. Google's grounding metadata lists
the pages the answer drew on and, for each sentence, which of those pages support
it. The variant then:

1. **Dates every page** (`date_gate.py`):

   | Page | Decision |
   |---|---|
   | Any date in the URL or on the page is on or after the cutoff | rejected |
   | Live page whose content changes over time (quote pages, "latest" reports) | rejected |
   | No date in the URL or on the page | rejected (unless `allow_undated=True`) |
   | Redirect does not name a destination | rejected |
   | At least one date, all strictly before the cutoff | accepted |

   Page dates come from v5.2's own audit extractor (HTML metadata, JSON-LD, then
   visible "Published" text). The page address is read from the search redirect's
   `Location` header without fetching the page; v5.2's resolver fetches the page
   itself, so sites that refuse automated requests (HTTP 403) were left unresolved.
2. **Keeps only sentences supported solely by accepted pages.** A sentence that
   cites any rejected page, or no page, is dropped.
3. **Runs v5.2's LLM cutoff verifier on the kept text.**

If no sentence survives or the verifier rejects the text, the search is retried
(naming the rejected pages) within v5.2's `search_verifier_max_attempts`; a topic
with no passing attempt fails closed, as in v5.2. The agent sees only the kept text
and the accepted pages it cites.

Every attempt — page verdicts, sentences kept and dropped, outcome — is recorded in
each prediction's `research_execution_audit.date_gate`, tagged with the gate
version (`GATE_ID`).

## Modules

| Module | Role |
|---|---|
| `date_gate.py` | The per-page rule (table above) |
| `page_gate.py` | Shared mechanics: redirect resolution, page dating, sentence filtering |
| `research.py` | v5.2's research tool with the gate |
| `search_web.py` | A gated drop-in for the shared `aieng` `search_web` tool, used by `wti_scenario_memory_agent_dategated` |
| `agent.py` | v5.2 ARIMA-only builder with the gated research tool |

## What it does not fix

- Searching today's web for a past date still shapes *which* pre-cutoff pages are
  found.
- A page can carry an early date and later edits; only the date is checked.
- Sentences with no supporting page are dropped, which removes most of the search
  model's own knowledge, but a supported sentence is still worded by the model. For
  gemini-3.1-flash-lite-preview (knowledge cutoff January 2025) its memory matters
  only for cutoffs before 2025.
- The strict rule can still leave a topic, or a whole origin, with no news; the agent
  then forecasts from the numerical suite alone. Check how often with
  `scripts/run_v52_weekly_extended.py --gate-report`.

## Build

```python
from energy_oil_forecasting.cfm_agent_v_5_2_dategated import (
    build_cfm_agent_config_arima_only_dategated,
    build_cfm_agent_predictor_arima_only_dategated,
)

config = build_cfm_agent_config_arima_only_dategated()  # same defaults as v5.2
predictor = build_cfm_agent_predictor_arima_only_dategated(config)
# predictor_id: agent_predictor_cfm_agent_v_5_2_arima_only_dategated_<model>_continuous
```

The agent keeps v5.2's name because ADK writes the name into the system instruction;
results are kept apart from v5.2's by the predictor ID alone.

## Tests

```bash
uv run pytest -q implementations/energy_oil_forecasting/cfm_agent_v_5_2_dategated/tests
```
