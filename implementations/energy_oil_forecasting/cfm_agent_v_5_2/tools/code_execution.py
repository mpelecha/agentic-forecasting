"""Audited optional E2B code-execution tool."""

from __future__ import annotations

import threading

from aieng.agents.tools.code_interpreter import CodeInterpreter
from energy_oil_forecasting.cfm_agent_v_5_2.config import CfmV52Settings
from google.adk.tools.function_tool import FunctionTool


class AuditedCodeExecutionTool:
    """Expose diagnostics-only code execution with mechanical usage auditing."""

    def __init__(self, settings: CfmV52Settings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._interpreter = CodeInterpreter()
        self.prepare_workflow()

    def prepare_workflow(self) -> None:
        with self._lock:
            self._call_count = 0
            self._purposes: list[str] = []

    @property
    def audit(self) -> dict[str, object]:
        with self._lock:
            return {
                "code_execution_available": self.settings.code_execution_enabled,
                "code_execution_used": self._call_count > 0,
                "code_execution_call_count": self._call_count,
                "code_execution_purposes": list(self._purposes),
            }

    def as_function_tool(self) -> FunctionTool:
        return FunctionTool(func=self.run_code)

    async def run_code(self, code: str, purpose: str) -> str:
        """Run diagnostic code; ``purpose`` must explain why it is needed."""
        clean_purpose = str(purpose).strip()
        if not clean_purpose:
            raise ValueError("purpose is required for diagnostic code execution")
        with self._lock:
            self._call_count += 1
            self._purposes.append(clean_purpose)
        result = await self._interpreter.run_code(code=code)
        return str(result)


__all__ = ["AuditedCodeExecutionTool"]
