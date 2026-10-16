"""E2B controls (Section 4.5): static screen, budgets, payload, audit. No sandbox is ever created."""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
import subprocess
import sys
import textwrap

import aieng.agents.tools.code_interpreter as interpreter_module
import pytest
from energy_oil_forecasting.cfm_agent_v_6_0.config import DEFAULT_SETTINGS, PACKAGE_ROOT, CfmV60Settings
from energy_oil_forecasting.cfm_agent_v_6_0.schemas import CodeExecutionRequest
from energy_oil_forecasting.cfm_agent_v_6_0.tools import code_execution as module
from energy_oil_forecasting.cfm_agent_v_6_0.tools.code_execution import (
    PREAMBLE_NAMES,
    REASON_ACCEPTED_BUDGET,
    REASON_CODE_TOO_LONG,
    REASON_DISABLED,
    REASON_PURPOSE_MISSING,
    REASON_REJECTED_BUDGET,
    REASON_SUITE_NOT_EXECUTED,
    REASON_SYNTAX_ERROR,
    AuditedCodeExecutionTool,
    build_preamble,
    static_screen,
)


INTERPRETER_MODULE = "aieng.agents.tools.code_interpreter"
PAYLOAD = {
    "diagnostics": {"returns": {"1": 0.01}, "curve_state": "flat", "flag": True, "missing": None},
    "model_suite_quantiles": {"5": {"0.1": 60.0, "0.5": 65.0, "0.9": 70.0}},
    "task_binding": {"target_series_id": "wti", "horizons": [5, 10, 21], "frequency": "B", "cutoff_date": "2026-09-14"},
}
SCRIPT = "print(TASK_BINDING['cutoff_date'])\n"


class ExplodingInterpreter:
    def __init__(self, *args, **kwargs):
        raise AssertionError("CodeInterpreter must not be constructed for a rejected request")


class StubInterpreter:
    instances: list[StubInterpreter] = []
    output = json.dumps({"stdout": ["line one", "line two"], "stderr": [], "results": None, "error": None})

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.code: str | None = None
        StubInterpreter.instances.append(self)

    async def run_code(self, code: str) -> str:
        self.code = code
        return StubInterpreter.output


@pytest.fixture(autouse=True)
def _reset_stub():
    StubInterpreter.instances = []
    StubInterpreter.output = json.dumps(
        {"stdout": ["line one", "line two"], "stderr": [], "results": None, "error": None}
    )


@pytest.fixture
def exploding(monkeypatch):
    monkeypatch.setattr(interpreter_module, "CodeInterpreter", ExplodingInterpreter)


@pytest.fixture
def stubbed(monkeypatch):
    monkeypatch.setattr(interpreter_module, "CodeInterpreter", StubInterpreter)


def _tool(settings: CfmV60Settings = DEFAULT_SETTINGS, payload=PAYLOAD) -> AuditedCodeExecutionTool:
    return AuditedCodeExecutionTool(settings, payload_provider=lambda: payload)


def _run(tool: AuditedCodeExecutionTool, code: str, purpose: str = "check the cutoff") -> dict:
    return json.loads(asyncio.run(tool.run_code(code, purpose)))


# ------------------------------------------------------------- import hygiene ----


def test_tool_module_has_no_module_level_interpreter_import():
    tree = ast.parse((PACKAGE_ROOT / "tools" / "code_execution.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith("aieng.agents") for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("aieng.agents")


def test_importing_the_tool_module_does_not_import_the_interpreter():
    """Load the module by path (bypassing the package ``__init__``) in a fresh interpreter."""
    package_root = str(PACKAGE_ROOT)
    script = textwrap.dedent(
        f"""
        import importlib.util, sys, types
        import energy_oil_forecasting
        root = {package_root!r}
        for name, path in [
            ("energy_oil_forecasting.cfm_agent_v_6_0", root),
            ("energy_oil_forecasting.cfm_agent_v_6_0.tools", root + "/tools"),
        ]:
            stub = types.ModuleType(name)
            stub.__path__ = [path]
            stub.__package__ = name
            sys.modules[name] = stub
        importlib.import_module("energy_oil_forecasting.cfm_agent_v_6_0.tools.code_execution")
        assert {INTERPRETER_MODULE!r} not in sys.modules, "interpreter imported at module import time"
        print("clean")
        """
    )
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "clean" in completed.stdout


def test_run_code_is_a_coroutine_function():
    assert inspect.iscoroutinefunction(AuditedCodeExecutionTool.run_code)
    assert inspect.iscoroutinefunction(_tool().run_code)


def test_tool_is_constructible_without_a_payload_provider():
    tool = AuditedCodeExecutionTool(DEFAULT_SETTINGS)
    assert tool.audit["accepted_count"] == 0
    assert tool.as_function_tool().func == tool.run_code


# ---------------------------------------------------------- rejection reasons ----


@pytest.mark.parametrize(
    ("code", "purpose", "expected"),
    [
        (SCRIPT, "   ", REASON_PURPOSE_MISSING),
        ("x = (", "p", REASON_SYNTAX_ERROR),
        ("import os\n", "p", "prohibited_module:os"),
        ("import os.path\n", "p", "prohibited_module:os"),
        ("from subprocess import run\n", "p", "prohibited_module:subprocess"),
        ("value = sys.argv\n", "p", "prohibited_module:sys"),
        ("open('x')\n", "p", "prohibited_call:open"),
        ("builtins_ref.eval('1')\n", "p", "prohibited_call:eval"),
        ("print(environ)\n", "p", "prohibited_name:environ"),
        ("print(thing.getenv)\n", "p", "prohibited_name:getenv"),
        ("print(__file__)\n", "p", "prohibited_name:__file__"),
    ],
)
def test_each_static_rejection_reason(exploding, code, purpose, expected):
    tool = _tool()
    result = _run(tool, code, purpose)
    assert result["status"] == "rejected"
    assert result["request_index"] == 1
    assert expected in result["rejection_reasons"]
    assert tool.audit["rejected_count"] == 1
    assert tool.audit["accepted_count"] == 0


def test_code_too_long_is_rejected(exploding):
    settings = CfmV60Settings(code_execution_max_code_chars=20)
    tool = _tool(settings)
    result = _run(tool, "x = 1\n" * 10)
    assert result["rejection_reasons"] == [REASON_CODE_TOO_LONG]


def test_disabled_tool_rejects(exploding):
    tool = _tool(CfmV60Settings(code_execution_enabled=False))
    result = _run(tool, SCRIPT)
    assert result["rejection_reasons"] == [REASON_DISABLED]
    assert tool.audit["code_execution_available"] is False
    assert tool.audit["code_execution_enabled"] is False


def test_suite_not_executed_rejects(exploding):
    for tool in (
        AuditedCodeExecutionTool(DEFAULT_SETTINGS, payload_provider=lambda: None),
        AuditedCodeExecutionTool(DEFAULT_SETTINGS),
    ):
        result = _run(tool, SCRIPT)
        assert result["rejection_reasons"] == [REASON_SUITE_NOT_EXECUTED]


def test_all_reasons_are_collected_not_just_the_first(exploding):
    tool = _tool(CfmV60Settings(code_execution_enabled=False), payload=None)
    result = _run(tool, "import os\nopen(os.environ['x'])\n", "")
    reasons = result["rejection_reasons"]
    assert reasons == [
        REASON_PURPOSE_MISSING,
        "prohibited_module:os",
        "prohibited_call:open",
        "prohibited_name:environ",
        REASON_DISABLED,
        REASON_SUITE_NOT_EXECUTED,
    ]


def test_static_screen_passes_a_clean_script():
    assert static_screen("import math\nprint(math.sqrt(DIAGNOSTICS['returns']['1'] ** 2))\n", DEFAULT_SETTINGS) == []


def test_rejected_request_is_recorded_without_launch(exploding):
    tool = _tool()
    _run(tool, "import os\n")
    (record,) = tool.audit["rejected_requests"]
    parsed = CodeExecutionRequest.model_validate(record)
    assert parsed.accepted is False
    assert parsed.launched is False
    assert parsed.rejection_reasons == ["prohibited_module:os"]
    assert parsed.code == "import os\n"
    assert parsed.code_chars == len("import os\n")
    assert tool.audit["accepted_requests"] == []
    assert tool.audit["code_execution_used"] is False


# -------------------------------------------------------------------- budgets ----


def test_one_accepted_call_then_accepted_budget_exhausted(stubbed):
    tool = _tool()
    first = _run(tool, SCRIPT, "first")
    assert first["status"] == "executed"
    second = _run(tool, SCRIPT, "second")
    assert second["status"] == "rejected"
    assert second["rejection_reasons"] == [REASON_ACCEPTED_BUDGET]
    assert len(StubInterpreter.instances) == 1
    audit = tool.audit
    assert audit["accepted_count"] == 1
    assert audit["rejected_count"] == 1
    assert audit["code_execution_call_count"] == 1
    assert audit["code_execution_purposes"] == ["first"]
    assert audit["budgets"]["max_accepted"] == 1


def test_three_rejected_then_rejected_budget_exhausted_is_still_recorded(exploding):
    tool = _tool()
    for _ in range(3):
        result = _run(tool, "import os\n")
        assert result["rejection_reasons"] == ["prohibited_module:os"]
    fourth = _run(tool, SCRIPT)
    assert fourth["status"] == "rejected"
    assert fourth["rejection_reasons"] == [REASON_REJECTED_BUDGET]
    assert fourth["request_index"] == 4
    assert tool.audit["rejected_count"] == 4
    assert [record["request_index"] for record in tool.audit["rejected_requests"]] == [1, 2, 3, 4]
    assert tool.audit["rejected_requests"][3]["rejection_reasons"] == [REASON_REJECTED_BUDGET]


def test_prepare_workflow_resets_budgets(stubbed):
    tool = _tool()
    _run(tool, SCRIPT)
    assert _run(tool, SCRIPT)["rejection_reasons"] == [REASON_ACCEPTED_BUDGET]
    tool.prepare_workflow()
    assert tool.audit["accepted_count"] == 0
    assert tool.audit["rejected_count"] == 0
    assert _run(tool, SCRIPT)["status"] == "executed"
    assert _run(tool, SCRIPT)["request_index"] == 2


# -------------------------------------------------------------- accepted path ----


def test_accepted_request_builds_the_preamble_and_constructs_the_sandbox_from_settings(stubbed):
    tool = _tool()
    result = _run(tool, SCRIPT, "cutoff check")
    assert result["status"] == "executed"
    assert result["stdout"] == "line one\nline two"
    assert result["stderr"] == ""
    assert result["error"] is None
    assert result["output_truncated"] is False
    (interpreter,) = StubInterpreter.instances
    assert interpreter.kwargs == {
        "code_execution_timeout_seconds": DEFAULT_SETTINGS.code_execution_timeout_seconds,
        "sandbox_timeout_seconds": DEFAULT_SETTINGS.code_execution_sandbox_timeout_seconds,
        "allow_internet_access": False,
    }
    assert interpreter.code.endswith(SCRIPT)
    preamble = interpreter.code[: -len(SCRIPT)]
    assert preamble == build_preamble(PAYLOAD)
    assigned = [
        target.id
        for node in ast.parse(preamble).body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    ]
    assert assigned == [name for name, _ in PREAMBLE_NAMES] == ["DIAGNOSTICS", "MODEL_SUITE_QUANTILES", "TASK_BINDING"]
    namespace: dict = {}
    exec(preamble, namespace)  # noqa: S102
    assert namespace["DIAGNOSTICS"] == PAYLOAD["diagnostics"]
    assert namespace["MODEL_SUITE_QUANTILES"] == PAYLOAD["model_suite_quantiles"]
    assert namespace["TASK_BINDING"] == PAYLOAD["task_binding"]
    assert [key for key in namespace if not key.startswith("__")] == [
        "DIAGNOSTICS",
        "MODEL_SUITE_QUANTILES",
        "TASK_BINDING",
    ]
    for forbidden in ("settings", "OPENAI", "/Users", "cleaned_summary"):
        assert forbidden not in preamble


def test_accepted_request_is_recorded_with_output(stubbed):
    StubInterpreter.output = json.dumps(
        {
            "stdout": ["ok"],
            "stderr": ["warn"],
            "results": None,
            "error": {"name": "ValueError", "value": "bad", "traceback": ""},
        }
    )
    tool = _tool()
    result = _run(tool, SCRIPT, "purpose")
    assert result["error"] == "ValueError: bad"
    (record,) = tool.audit["accepted_requests"]
    parsed = CodeExecutionRequest.model_validate(record)
    assert parsed.accepted is True
    assert parsed.launched is True
    assert parsed.stdout == "ok"
    assert parsed.stderr == "warn"
    assert parsed.error == "ValueError: bad"
    assert parsed.duration_seconds is not None and parsed.duration_seconds >= 0
    assert parsed.purpose == "purpose"
    assert tool.audit["code_execution_used"] is True


def test_output_truncation_is_flagged(stubbed):
    StubInterpreter.output = json.dumps({"stdout": ["x" * 100], "stderr": [], "results": None, "error": None})
    tool = _tool(CfmV60Settings(code_execution_max_output_chars=10))
    result = _run(tool, SCRIPT)
    assert result["output_truncated"] is True
    assert result["stdout"] == "x" * 10
    assert tool.audit["accepted_requests"][0]["output_truncated"] is True
    assert tool.audit["accepted_requests"][0]["stdout"] == "x" * 10


def test_interpreter_failure_is_captured_not_raised(monkeypatch):
    class FailingInterpreter(StubInterpreter):
        async def run_code(self, code: str) -> str:
            raise TimeoutError("sandbox timed out")

    monkeypatch.setattr(interpreter_module, "CodeInterpreter", FailingInterpreter)
    tool = _tool()
    result = _run(tool, SCRIPT)
    assert result["status"] == "executed"
    assert result["error"] == "TimeoutError: sandbox timed out"
    assert tool.audit["accepted_requests"][0]["launched"] is True


def test_audit_lists_accepted_and_rejected_separately(stubbed):
    tool = _tool()
    _run(tool, "import os\n", "bad")
    _run(tool, SCRIPT, "good")
    _run(tool, SCRIPT, "over budget")
    audit = tool.audit
    assert [record["purpose"] for record in audit["accepted_requests"]] == ["good"]
    assert [record["purpose"] for record in audit["rejected_requests"]] == ["bad", "over budget"]
    assert audit["accepted_count"] == 1
    assert audit["rejected_count"] == 2
    assert set(audit["budgets"]) == {
        "max_accepted",
        "max_rejected",
        "max_code_chars",
        "max_output_chars",
        "timeout_seconds",
        "sandbox_timeout_seconds",
        "allow_internet",
    }


def test_module_constants_are_exported():
    assert "AuditedCodeExecutionTool" in module.__all__
    assert "static_screen" in module.__all__


# --------------------------------------------------------------- registration ----


def test_disabled_tool_is_not_registered_by_agent(monkeypatch):
    """Imports ``agent.py``; deselect with ``-m "not agent_import"`` while the tools are in flight."""
    monkeypatch.setenv("OPENAI_BASE_URL", "http://proxy.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    from energy_oil_forecasting.cfm_agent_v_6_0.agent import build_cfm_agent_config  # noqa: PLC0415

    class Service:
        series_ids = ["wti"]

    enabled = build_cfm_agent_config(data_service=Service(), settings=CfmV60Settings(code_execution_enabled=True))
    disabled = build_cfm_agent_config(data_service=Service(), settings=CfmV60Settings(code_execution_enabled=False))
    assert [tool.name for tool in enabled.function_tools] == [
        "run_authoritative_suite",
        "run_research_pipeline",
        "run_code",
    ]
    assert [tool.name for tool in disabled.function_tools] == ["run_authoritative_suite", "run_research_pipeline"]


def test_agent_config_carries_the_main_model_and_coach_overrides(monkeypatch):
    """Imports ``agent.py``; the overrides exist for coach compatibility only and re-validate settings."""
    monkeypatch.setenv("OPENAI_BASE_URL", "http://proxy.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    from energy_oil_forecasting.cfm_agent_v_6_0.agent import (  # noqa: PLC0415
        _CONFIG,
        build_cfm_agent_config,
    )
    from google.adk.models.lite_llm import LiteLlm  # noqa: PLC0415

    class Service:
        series_ids = ["wti"]

    config = build_cfm_agent_config(data_service=Service(), model="override-main", search_model="override-search")
    assert isinstance(config.model, LiteLlm)
    assert config.model.model == "openai/override-main"
    assert config.model._additional_args["timeout"] == DEFAULT_SETTINGS.main_timeout_seconds
    assert config.temperature == DEFAULT_SETTINGS.main_temperature
    assert config.max_output_tokens == DEFAULT_SETTINGS.main_max_tokens
    assert config.seed is None
    assert config.name == "cfm_agent_v_6_0"
    assert [path.name for path in config.skills_dirs] == [
        "source-selection",
        "research-planning",
        "claim-building",
        "action-proposal",
        "code-analysis",
    ]
    _, market, research, code_execution, settings = _CONFIG.pop(id(config))
    assert settings.main_model == "override-main"
    assert settings.search_model == "override-search"
    assert code_execution._payload_provider == market.code_execution_payload
    assert research.settings.search_model == "override-search"
