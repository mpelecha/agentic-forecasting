## CFM Agent v5.2

CFM Agent v5.2 is an isolated sibling derived from v5.1. It preserves the locked v5.1 forecasting design and fixes the package-local optional E2B code-execution adapter by awaiting its asynchronous result before returning serializable output to Google ADK.

See `ARCHITECTURE.md` for the authoritative contract.

### Offline validation

```bash
uv run ruff format --check implementations/energy_oil_forecasting/cfm_agent_v_5_2
uv run ruff check implementations/energy_oil_forecasting/cfm_agent_v_5_2
uv run python -m compileall -q implementations/energy_oil_forecasting/cfm_agent_v_5_2
uv run pytest -q implementations/energy_oil_forecasting/cfm_agent_v_5_2/tests
sha256sum -c implementations/energy_oil_forecasting/cfm_agent_v_5_2/MANIFEST.sha256
```

### Output location
Save live and repeated-run test artifacts directly under `implementations/energy_oil_forecasting/outputs`.
