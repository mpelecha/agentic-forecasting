# Installation and validation

Replace the following four files in the existing Version 2.0 package:

```text
implementations/energy_oil_forecasting/cfm_agent_v_2_0/outputs.py
implementations/energy_oil_forecasting/cfm_agent_v_2_0/tests/test_outputs.py
implementations/energy_oil_forecasting/cfm_agent_v_2_0/README.md
implementations/energy_oil_forecasting/cfm_agent_v_2_0/INSTALL.md
```

Do not replace or modify `cfm_agent_v_1_0`.

From the repository root, run:

```bash
uv run ruff format implementations/energy_oil_forecasting/cfm_agent_v_2_0
uv run ruff check implementations/energy_oil_forecasting/cfm_agent_v_2_0
uv run python -m compileall -q implementations/energy_oil_forecasting/cfm_agent_v_2_0
uv run pytest -q implementations/energy_oil_forecasting/cfm_agent_v_2_0/tests
```

Expected result:

```text
19 passed
```

The shared `BaseAgentConfig` deprecation warnings are non-fatal repository
technical debt.

## Construction smoke test

```python
from energy_oil_forecasting.cfm_agent_v_2_0 import (
    build_cfm_agent_config,
    build_cfm_agent_predictor,
)
from energy_oil_forecasting.data import build_wti_multivariate_service

service = build_wti_multivariate_service()
config = build_cfm_agent_config(data_service=service)
predictor = build_cfm_agent_predictor(config)

print(config.name)
print([tool.name for tool in config.function_tools])
print([path.name for path in config.skills_dirs])
print(predictor.predictor_id)
```

Expected identifiers:

```text
cfm_agent_v_2_0
['query_market_data']
['forecasting', 'model-selection', 'research', 'code-analysis']
```

After validation, restart the notebook kernel before rerunning the historical
March 2, 2026 forecast so Python loads the updated output schema.
