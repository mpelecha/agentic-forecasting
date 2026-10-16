"""Literal-scan test (specification Section 7.1 and 7.4).

Every guarded module is parsed with ``ast``; a numeric literal other than 0, 1, -1,
or 2 fails unless its line carries the comment ``# structural: <reason>``.
"""

from __future__ import annotations

import ast
from pathlib import Path

from energy_oil_forecasting.cfm_agent_v_6_0.config import PACKAGE_ROOT


GUARDED_MODULES: tuple[str, ...] = (
    "models",
    "history.py",
    "diagnostics.py",
    "forecast_engine",
    "policy",
    "llm_roles.py",
    "response_control.py",
    "claim_support_verifier.py",
    "passages.py",
    "tools/code_execution.py",
    "tools/research_pipeline.py",
)
ALLOWED_LITERALS = {0, 1, 2}
STRUCTURAL_TAG = "# structural:"


def guarded_files() -> list[Path]:
    files: list[Path] = []
    for entry in GUARDED_MODULES:
        path = PACKAGE_ROOT / entry
        if path.is_dir():
            files.extend(sorted(path.rglob("*.py")))
        elif path.exists():
            files.append(path)
    return files


def offending_literals(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source, filename=str(path))
    findings: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or isinstance(node.value, bool):
            continue
        value = node.value
        if not isinstance(value, (int, float)):
            continue
        if value in ALLOWED_LITERALS:
            continue
        line = lines[node.lineno - 1]
        if STRUCTURAL_TAG in line:
            continue
        findings.append(f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno}: literal {value!r}")
    return findings


def test_guarded_modules_exist():
    assert guarded_files(), "no guarded module found"


def test_no_tunable_literals_in_guarded_modules():
    findings: list[str] = []
    for path in guarded_files():
        findings.extend(offending_literals(path))
    assert not findings, "numeric literals outside {0, 1, -1, 2} without '# structural:' comment:\n" + "\n".join(
        findings
    )
