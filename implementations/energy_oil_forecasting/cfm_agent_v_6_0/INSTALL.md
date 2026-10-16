## Installation

The package lives inside the repository's `implementations/energy_oil_forecasting` namespace and
uses the repository's locked environment; there is no separate installation step.

```bash
cd /path/to/CFMv6_0
uv sync
uv run python -c "import energy_oil_forecasting.cfm_agent_v_6_0 as pkg; print(pkg.AGENT_NAME)"
```

Runtime dependencies beyond the shared library: `numpy`, `pandas`, `lightgbm`, `pydantic`,
`litellm`, `google-adk`, and, only when `code_execution_enabled` is true, the E2B client through
`aieng.agents.tools.code_interpreter`. TOML settings are read with the standard library
(`tomllib`, Python 3.11 or later; the repository pins Python 3.12).

Environment variables for live runs: `OPENAI_BASE_URL` and `OPENAI_API_KEY` (the proxy),
`E2B_API_KEY` (sandboxes), the Langfuse keys used by the shared library, and optionally
`CFM_V60_SETTINGS_FILE` (alternate settings file, recorded in the audit),
`CFM_V60_DATA_SNAPSHOT` (frozen data snapshot directory, recorded in the audit), and
`CFM_V60_LLM_PREFLIGHT_JSON` (a saved `preflight-models` result, recorded in the audit).
