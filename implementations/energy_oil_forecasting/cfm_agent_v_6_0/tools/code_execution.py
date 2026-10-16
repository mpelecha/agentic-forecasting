"""Audited, statically screened, optional E2B code-execution tool (Section 4.5).

The tool is always constructible; the E2B ``CodeInterpreter`` is imported and
constructed only inside the accepted branch of :meth:`AuditedCodeExecutionTool.run_code`,
so a rejected request, a disabled tool, or a package import never touches the
e2b dependency. Every request, accepted or rejected, is recorded as a
:class:`~energy_oil_forecasting.cfm_agent_v_6_0.schemas.CodeExecutionRequest`.
"""

from __future__ import annotations

import ast
import json
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

from energy_oil_forecasting.cfm_agent_v_6_0.config import CfmV60Settings
from energy_oil_forecasting.cfm_agent_v_6_0.schemas import CodeExecutionRequest
from google.adk.tools.function_tool import FunctionTool


STATUS_REJECTED = "rejected"
STATUS_EXECUTED = "executed"

REASON_PURPOSE_MISSING = "purpose_missing"
REASON_CODE_TOO_LONG = "code_too_long"
REASON_SYNTAX_ERROR = "syntax_error"
REASON_PROHIBITED_MODULE = "prohibited_module"
REASON_PROHIBITED_CALL = "prohibited_call"
REASON_PROHIBITED_NAME = "prohibited_name"
REASON_ACCEPTED_BUDGET = "accepted_budget_exhausted"
REASON_REJECTED_BUDGET = "rejected_budget_exhausted"
REASON_DISABLED = "code_execution_disabled"
REASON_SUITE_NOT_EXECUTED = "numerical_suite_not_executed"

#: Names the preamble assigns, in order, and the payload-provider keys they come from.
PREAMBLE_NAMES: tuple[tuple[str, str], ...] = (
    ("DIAGNOSTICS", "diagnostics"),
    ("MODEL_SUITE_QUANTILES", "model_suite_quantiles"),
    ("TASK_BINDING", "task_binding"),
)
_PREAMBLE_HEADER = "# CFM Agent v6.0 diagnostic preamble: task-bound inputs as JSON literals.\n"
_PREAMBLE_JSON_ALIAS = "_cfm_json"

PayloadProvider = Callable[[], Mapping[str, Any] | None]


def _top_level(module_name: str) -> str:
    return module_name.split(".", 1)[0]


class _Findings:
    """Prohibited modules, calls, and names found by one static screen."""

    def __init__(self, settings: CfmV60Settings) -> None:
        self.modules = set(settings.code_execution_prohibited_modules)
        self.calls = set(settings.code_execution_prohibited_calls)
        self.names = set(settings.code_execution_prohibited_names)
        self.found_modules: set[str] = set()
        self.found_calls: set[str] = set()
        self.found_names: set[str] = set()

    def visit(self, node: ast.AST) -> None:
        if isinstance(node, ast.Import):
            self.found_modules.update(_top_level(alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            self.found_modules.add(_top_level(node.module))
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                self.found_modules.add(node.value.id)
            self.found_names.add(node.attr)
        elif isinstance(node, ast.Name):
            self.found_names.add(node.id)
        if isinstance(node, ast.Call):
            self.found_calls.add(_call_name(node))

    def reasons(self) -> list[str]:
        reasons = [f"{REASON_PROHIBITED_MODULE}:{name}" for name in sorted(self.found_modules & self.modules)]
        reasons.extend(f"{REASON_PROHIBITED_CALL}:{name}" for name in sorted(self.found_calls & self.calls))
        reasons.extend(f"{REASON_PROHIBITED_NAME}:{name}" for name in sorted(self.found_names & self.names))
        return reasons


def _call_name(node: ast.Call) -> str:
    target = node.func
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return ""


def static_screen(code: str, settings: CfmV60Settings) -> list[str]:
    """Return every static-screen reason the code fails; an empty list means it passes.

    The screen is syntactic: it checks imports, attribute access, calls, and names
    against the three prohibited lists in settings. It is a control on the
    diagnostic's shape, not a sandbox; the sandbox is the interpreter's job.
    """
    reasons: list[str] = []
    if len(code) > settings.code_execution_max_code_chars:
        reasons.append(REASON_CODE_TOO_LONG)
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        reasons.append(REASON_SYNTAX_ERROR)
        return reasons
    findings = _Findings(settings)
    for node in ast.walk(tree):
        findings.visit(node)
    reasons.extend(findings.reasons())
    return reasons


def build_preamble(payload: Mapping[str, Any]) -> str:
    """Build the Python preamble assigning the three task-bound inputs as JSON literals.

    ``payload`` maps ``diagnostics``, ``model_suite_quantiles``, and ``task_binding``
    (upper-case keys are accepted too) to JSON-serialisable values. Nothing else is
    read from it, so paths, research text, settings, and secrets cannot leak in.
    """
    lines = [_PREAMBLE_HEADER, f"import json as {_PREAMBLE_JSON_ALIAS}\n"]
    for name, key in PREAMBLE_NAMES:
        value = payload.get(key, payload.get(name))
        literal = json.dumps(value, sort_keys=True, default=str)
        lines.append(f"{name} = {_PREAMBLE_JSON_ALIAS}.loads({literal!r})\n")
    lines.append(f"del {_PREAMBLE_JSON_ALIAS}\n")
    return "".join(lines)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value)


def _error_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        name = value.get("name")
        message = value.get("value")
        text = ": ".join(str(part) for part in (name, message) if part)
        return text or json.dumps(value, default=str)
    return str(value)


class AuditedCodeExecutionTool:
    """Diagnostics-only code execution with a static screen, budgets, and a full audit."""

    def __init__(self, settings: CfmV60Settings, payload_provider: PayloadProvider | None = None) -> None:
        self.settings = settings
        self._payload_provider = payload_provider
        self._lock = threading.Lock()
        self.prepare_workflow()

    def prepare_workflow(self) -> None:
        """Reset counters and request lists at the start of a workflow."""
        with self._lock:
            self._accepted: list[CodeExecutionRequest] = []
            self._rejected: list[CodeExecutionRequest] = []
            self._request_count = 0

    @property
    def audit(self) -> dict[str, Any]:
        with self._lock:
            accepted = [request.model_dump(mode="json") for request in self._accepted]
            rejected = [request.model_dump(mode="json") for request in self._rejected]
        settings = self.settings
        return {
            "code_execution_available": settings.code_execution_enabled,
            "code_execution_enabled": settings.code_execution_enabled,
            "code_execution_used": len(accepted) > 0,
            "code_execution_call_count": len(accepted),
            "code_execution_purposes": [request["purpose"] for request in accepted],
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "accepted_requests": accepted,
            "rejected_requests": rejected,
            "budgets": {
                "max_accepted": settings.code_execution_max_accepted,
                "max_rejected": settings.code_execution_max_rejected,
                "max_code_chars": settings.code_execution_max_code_chars,
                "max_output_chars": settings.code_execution_max_output_chars,
                "timeout_seconds": settings.code_execution_timeout_seconds,
                "sandbox_timeout_seconds": settings.code_execution_sandbox_timeout_seconds,
                "allow_internet": settings.code_execution_allow_internet,
            },
        }

    def as_function_tool(self) -> FunctionTool:
        return FunctionTool(func=self.run_code)

    # ------------------------------------------------------------ screening ----

    def _screen(self, code: str, purpose: str) -> tuple[list[str], Mapping[str, Any] | None]:
        """Collect every rejection reason; return the payload when the request may run."""
        settings = self.settings
        reasons: list[str] = []
        if not purpose.strip():
            reasons.append(REASON_PURPOSE_MISSING)
        reasons.extend(static_screen(code, settings))
        with self._lock:
            accepted_count = len(self._accepted)
            rejected_count = len(self._rejected)
        if accepted_count >= settings.code_execution_max_accepted:
            reasons.append(REASON_ACCEPTED_BUDGET)
        if rejected_count >= settings.code_execution_max_rejected:
            reasons.append(REASON_REJECTED_BUDGET)
        if not settings.code_execution_enabled:
            reasons.append(REASON_DISABLED)
        payload = self._payload_provider() if self._payload_provider is not None else None
        if payload is None:
            reasons.append(REASON_SUITE_NOT_EXECUTED)
        return reasons, payload

    def _next_index(self) -> int:
        with self._lock:
            self._request_count += 1
            return self._request_count

    def _record(self, request: CodeExecutionRequest) -> None:
        with self._lock:
            (self._accepted if request.accepted else self._rejected).append(request)

    # ------------------------------------------------------------ execution ----

    async def run_code(self, code: str, purpose: str) -> str:
        """Run one self-contained diagnostic script in the sandbox.

        ``purpose`` must state why the diagnostic is needed. The script may read the
        preamble names DIAGNOSTICS, MODEL_SUITE_QUANTILES, and TASK_BINDING. It must
        not forecast, recompute task-bound values, decide evidence, transform or
        price the forecast, touch the filesystem, packages, environment, or network.
        Returns JSON: ``{"status": "rejected", ...}`` with every failed rule, or
        ``{"status": "executed", ...}`` with stdout, stderr, and error.
        """
        code = "" if code is None else str(code)
        clean_purpose = "" if purpose is None else str(purpose).strip()
        index = self._next_index()
        reasons, payload = self._screen(code, clean_purpose)
        if reasons or payload is None:
            self._record(
                CodeExecutionRequest(
                    request_index=index,
                    purpose=clean_purpose,
                    code=code,
                    code_chars=len(code),
                    accepted=False,
                    launched=False,
                    rejection_reasons=reasons,
                )
            )
            return json.dumps({"status": STATUS_REJECTED, "request_index": index, "rejection_reasons": reasons})
        return await self._execute(index=index, purpose=clean_purpose, code=code, payload=payload)

    async def _execute(self, *, index: int, purpose: str, code: str, payload: Mapping[str, Any]) -> str:
        from aieng.agents.tools.code_interpreter import CodeInterpreter  # noqa: PLC0415

        settings = self.settings
        script = build_preamble(payload) + code
        interpreter = CodeInterpreter(
            code_execution_timeout_seconds=settings.code_execution_timeout_seconds,
            sandbox_timeout_seconds=settings.code_execution_sandbox_timeout_seconds,
            allow_internet_access=settings.code_execution_allow_internet,
        )
        started = time.monotonic()
        error: str | None = None
        stdout = stderr = ""
        try:
            raw = await interpreter.run_code(code=script)
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(parsed, Mapping):
                stdout = _as_text(parsed.get("stdout"))
                stderr = _as_text(parsed.get("stderr"))
                error = _error_text(parsed.get("error"))
            else:
                stdout = _as_text(raw)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        duration = time.monotonic() - started
        limit = settings.code_execution_max_output_chars
        truncated = len(stdout) > limit or len(stderr) > limit
        stdout, stderr = stdout[:limit], stderr[:limit]
        self._record(
            CodeExecutionRequest(
                request_index=index,
                purpose=purpose,
                code=code,
                code_chars=len(code),
                accepted=True,
                launched=True,
                duration_seconds=duration,
                stdout=stdout,
                stderr=stderr,
                error=error,
                output_truncated=truncated,
            )
        )
        return json.dumps(
            {
                "status": STATUS_EXECUTED,
                "request_index": index,
                "stdout": stdout,
                "stderr": stderr,
                "error": error,
                "output_truncated": truncated,
                "duration_seconds": duration,
            }
        )


__all__ = [
    "PREAMBLE_NAMES",
    "REASON_ACCEPTED_BUDGET",
    "REASON_CODE_TOO_LONG",
    "REASON_DISABLED",
    "REASON_PROHIBITED_CALL",
    "REASON_PROHIBITED_MODULE",
    "REASON_PROHIBITED_NAME",
    "REASON_PURPOSE_MISSING",
    "REASON_REJECTED_BUDGET",
    "REASON_SUITE_NOT_EXECUTED",
    "REASON_SYNTAX_ERROR",
    "STATUS_EXECUTED",
    "STATUS_REJECTED",
    "AuditedCodeExecutionTool",
    "build_preamble",
    "static_screen",
]
