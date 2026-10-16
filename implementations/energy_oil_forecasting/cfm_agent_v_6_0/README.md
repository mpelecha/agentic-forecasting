## CFM Agent v6.0

CFM Agent v6.0 is an isolated sibling of `cfm_agent_v_5_2` that forecasts the WTI front-month price
at 5, 10, and 21 business days. It replaces the v5.2 price-level models with three return-path
models (historical bootstrap, sparse autoregression, LightGBM on returns) that each simulate 1,000
paths, derives the centre and uncertainty transformations from the historical distribution of WTI
moves, expands the deterministic diagnostics, simplifies the evidence tiers, splits the LLM
configuration into five forwarded roles, puts Python-enforced budgets around E2B, and keeps every
tunable value in `settings.toml`.

The normative document is `SPECIFICATION.pdf`; the design decisions are in `DECISIONS.md`; the
requirement-to-code map is in `TRACEABILITY.md`; the vocabulary is in `CONTEXT.md`.

### Running a forecast

```python
from datetime import datetime

from aieng.forecasting.evaluation.task import ForecastingTask
from energy_oil_forecasting.cfm_agent_v_6_0 import build_cfm_agent_config, build_cfm_agent_predictor
from energy_oil_forecasting.data import WTI_SERIES_ID, build_wti_multivariate_service

service = build_wti_multivariate_service()
predictor = build_cfm_agent_predictor(build_cfm_agent_config(data_service=service))
task = ForecastingTask(
    task_id="wti_demo", target_series_id=WTI_SERIES_ID, horizons=[5, 10, 21], frequency="B", description="demo"
)
predictions = predictor.predict(task, service.context(as_of=datetime(2026, 9, 14)))
```

Each prediction carries eleven quantiles, the P50 point forecast, and the complete audit record
(specification Section 8) in its metadata. The run needs `OPENAI_BASE_URL` and `OPENAI_API_KEY`
for the proxy, and `E2B_API_KEY` when `code_execution_enabled` is true.

### Changing a setting

Every tunable value is a key in `settings.toml`. To change one for a run without editing the
packaged file, pass it as a keyword override:

```python
from energy_oil_forecasting.cfm_agent_v_6_0.config import CfmV60Settings

settings = CfmV60Settings(history_window_business_days=520, audit_enabled=True)
```

To experiment with a different file, use `CfmV60Settings.from_file(path, **overrides)` or set the
environment variable `CFM_V60_SETTINGS_FILE`. The audit records the file used, its SHA-256, the
resolved values, and every field that differs from the packaged file. Keys tagged `[identity]`
change the model-suite ID; a new key must be added to both `settings.toml` and
`CfmV60Settings`, and the configuration tests enforce that the two agree.

### Offline validation

```bash
uv run ruff format --check implementations/energy_oil_forecasting/cfm_agent_v_6_0
uv run ruff check implementations/energy_oil_forecasting/cfm_agent_v_6_0
uv run python -m compileall -q implementations/energy_oil_forecasting/cfm_agent_v_6_0
uv run pytest -q implementations/energy_oil_forecasting/cfm_agent_v_6_0/tests
uv run pytest -q implementations/energy_oil_forecasting/cfm_agent_v_5_2/tests
```

### Release commands

```bash
uv run python -m energy_oil_forecasting.cfm_agent_v_6_0.release verify-protected
uv run python -m energy_oil_forecasting.cfm_agent_v_6_0.release snapshot-data
uv run python -m energy_oil_forecasting.cfm_agent_v_6_0.release preflight-models
uv run python -m energy_oil_forecasting.cfm_agent_v_6_0.release sensitivity --cutoff 2026-09-14
uv run python -m energy_oil_forecasting.cfm_agent_v_6_0.release write-manifest
uv run python -m energy_oil_forecasting.cfm_agent_v_6_0.release verify-manifest
uv run python -m energy_oil_forecasting.cfm_agent_v_6_0.release package
uv run python -m energy_oil_forecasting.cfm_agent_v_6_0.release verify-archive
```

`PROTECTED_SCOPE.txt` lists the files outside the package that must not change (shared main,
v5.0, v5.2, the coach's code and documents); `PROTECTED_BASELINE.sha256` holds their hashes at the
start of the build. Archives, data snapshots, and sensitivity outputs go under
`implementations/energy_oil_forecasting/outputs/cfm_agent_v_6_0/` and are not committed.

### Output location

Live and repeated-run artefacts belong under `implementations/energy_oil_forecasting/outputs/cfm_agent_v_6_0/`.
