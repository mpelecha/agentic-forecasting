"""Skill registration contract tests."""

from pathlib import Path

import yaml
from energy_oil_forecasting.cfm_agent_v_2_0.config import SKILLS_ROOT


EXPECTED = ["forecasting", "model-selection", "research", "code-analysis"]


def test_all_skills_have_exact_adk_compatible_names() -> None:
    assert sorted(path.name for path in SKILLS_ROOT.iterdir() if path.is_dir()) == sorted(EXPECTED)
    for name in EXPECTED:
        path = Path(SKILLS_ROOT) / name / "SKILL.md"
        text = path.read_text(encoding="utf-8")
        frontmatter = text.split("---", 2)[1]
        parsed = yaml.safe_load(frontmatter)
        assert parsed["name"] == name
        assert parsed["description"]
