# Updating the corpus page

The corpus page is a static HTML snapshot of every recorded run across all three
streams (`v50_lite`, `v52_advanced`, `v52_lite`). It does not auto-refresh — after
new runs land, regenerate and republish it.

## 1. Regenerate

```bash
cd /Users/namagarwal/Desktop/CFMv6_0
uv run python -m energy_oil_forecasting.cfm_coach.make_corpus_page
```

Writes `implementations/energy_oil_forecasting/cfm_coach/corpus_page.html` and
prints the per-stream counts, e.g.:

```
corpus_page.html  (2.45 MB, 28 records)
  v50_lite        14 record(s)  cfm_agent_v_5_0  gemini-3.1-flash-lite-preview
  v52_advanced     3 record(s)  cfm_agent_v_5_2  gemini-3.5-flash
  v52_lite        11 record(s)  cfm_agent_v_5_2  gemini-3.1-flash-lite-preview
```

Check the counts match what you expect before sharing.

## 2. Publish it

**Regenerating only writes the local file — it does not update the shared link.**
Ask Claude to publish `corpus_page.html` as an artifact using the **same URL**
below (pass the existing URL, don't create a new one), or send the file directly.

**Live link:** https://claude.ai/code/artifact/0e4a27bf-8333-4f5a-bf9c-23ecf7f85436

Private by default — share from the page's share menu.

## Notes

- One page for all three streams, filterable by the **Stream** dropdown. Each card
  is tagged with the stream/agent/model that produced it.
- The **Performance** tab tracks forecasts vs actuals across all four agents
  (adaptive included): timeline against realized WTI, P10–P90 coverage,
  directional hit-rates (vs last close and vs the ensemble), shift quality, and
  width calibration. Agent checkboxes, 5/10/21B horizon selector, and a
  median-vs-all-runs toggle; live_forward runs only.
- Corpora are never pooled for fitting even though they share a page —
  `ComparisonPolicy` enforces one agent build, one prompt version, one model.
- Size ceiling: warns past 14 MB (16 MB is the artifact hard cap). If ever hit,
  narrow with `--stream=<id>` to publish one corpus at a time.
