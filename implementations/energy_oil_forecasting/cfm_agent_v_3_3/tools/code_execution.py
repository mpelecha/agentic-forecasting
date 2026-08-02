"""E2B code-execution configuration for the CFM agent."""

from __future__ import annotations

from aieng.forecasting.methods.agentic.agent_factory import CodeExecutionConfig


def build_code_execution_config() -> CodeExecutionConfig:
    """Enable the repository's isolated E2B ``run_code`` capability."""
    return CodeExecutionConfig(enabled=True)


__all__ = ["build_code_execution_config"]
