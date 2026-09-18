"""Draft a challenger package for an accepted text proposal. A human registers it.

The weekly job never edits the live package, `targets.py` or `streams.py`. For a
proposal whose lever is skill, persona or prompt text, it writes everything a
person needs under ``review/challengers/<P-id>/``:

- ``cfm_agent_v_5_3_<slug>/`` -- a copy of ``cfm_agent_v_5_2`` with its identity
  rewritten by exact-string replacement (module path, AGENT_NAME, the persona's
  version line), the proposal's text edits applied (each ``find`` must match
  exactly once), and ``MANIFEST.sha256`` regenerated in the shipped format;
- ``stream.yaml`` -- the challenger stream as data (advanced model, 3 runs/day);
- ``calibration/v001.json`` -- the identity baseline for its own ledger;
- ``REGISTER.md`` -- the `mv`, the `AgentTarget` and `RunStream` snippets, the
  test assertion that must change, the `pyproject` line, and the verification
  commands, as text to be applied by hand.

Usage (from the repo root)::

    uv run python -m energy_oil_forecasting.cfm_coach.review.challenger --proposal P-0007
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import yaml
from energy_oil_forecasting.cfm_coach.review.memory import Proposal, ProposalStore
from energy_oil_forecasting.cfm_coach.review.settings import DEFAULT_REVIEW_SETTINGS, ReviewSettings
from energy_oil_forecasting.cfm_coach.targets import V52
from pydantic import BaseModel, ConfigDict


SOURCE_MODULE = "energy_oil_forecasting.cfm_agent_v_5_2"
SOURCE_NAME = "cfm_agent_v_5_2"
IGNORED = ("__pycache__", "*.pyc", ".DS_Store")
#: Files that may still mention the source package name after the rewrite (history, not code).
DOCS_ALLOWLIST = (
    "README.md",
    "CHANGELOG.md",
    "ARCHITECTURE.md",
    "BUILD_VERIFICATION.md",
    "CALIBRATION.md",
    "LIVE_VALIDATION.md",
    "INSTALL.md",
)
TEXT_LEVERS = ("skill_text", "persona", "prompt_instruction")


class ChallengerDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    proposal_id: str
    slug: str
    package_name: str
    module: str
    stream_id: str
    package_dir: str
    manifest_entries: int
    edits_applied: int
    files_rewritten: list[str]
    register_path: str


class DraftError(RuntimeError):
    pass


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:24] or "x"


# ----------------------------------------------------------- manifest ----


def manifest_lines(package_dir: Path) -> list[str]:
    """``<sha256>  ./<path>`` for every file under `package_dir`, C-locale sorted, manifest itself excluded."""
    lines = []
    for path in package_dir.rglob("*"):
        if not path.is_file() or path.name == "MANIFEST.sha256":
            continue
        rel = path.relative_to(package_dir).as_posix()
        if any(part == "__pycache__" for part in path.parts) or path.suffix == ".pyc" or path.name == ".DS_Store":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  ./{rel}")
    return sorted(lines, key=lambda line: line.split("  ./", 1)[1].encode())


def write_manifest(package_dir: Path) -> int:
    lines = manifest_lines(package_dir)
    (package_dir / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def verify_manifest(package_dir: Path) -> bool:
    return (package_dir / "MANIFEST.sha256").read_text(encoding="utf-8") == "\n".join(
        manifest_lines(package_dir)
    ) + "\n"


# -------------------------------------------------------------- edits ----


def replace_exactly_once(path: Path, find: str, replace: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(find)
    if count != 1:
        raise DraftError(f"{path.name}: expected exactly one match for {find[:60]!r}, found {count}")
    path.write_text(text.replace(find, replace), encoding="utf-8")


def rewrite_identity(package_dir: Path, *, package_name: str, module: str) -> list[str]:
    """Rename the package inside its own files so the copy imports itself, not v5.2."""
    rewritten: list[str] = []
    config_ok = False
    for path in sorted(package_dir.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        # Module path first, then every bare mention (predictor ids, docstrings): the copy is a new agent everywhere.
        new = text.replace(SOURCE_MODULE, module).replace(SOURCE_NAME, package_name)
        if path.name == "config.py":
            config_ok = f'AGENT_NAME = "{package_name}"' in new
        if path.name == "agent.py":
            new = new.replace("You are CFM Agent v5.2", f"You are CFM Agent v5.3 ({package_name})")
        if new != text:
            path.write_text(new, encoding="utf-8")
            rewritten.append(path.relative_to(package_dir).as_posix())
    if not config_ok:
        raise DraftError("config.py: AGENT_NAME line not found after rewrite")
    leftovers = [
        p.relative_to(package_dir).as_posix()
        for p in package_dir.rglob("*.py")
        if SOURCE_NAME in p.read_text(encoding="utf-8", errors="ignore")
    ]
    if leftovers:
        raise DraftError(f"source package name still referenced in {leftovers}")
    return rewritten


# --------------------------------------------------------------- draft ----


def draft_challenger(
    proposal: Proposal,
    *,
    settings: ReviewSettings = DEFAULT_REVIEW_SETTINGS,
    source_root: Path = V52.package_root,
    start: date | None = None,
    weeks: int = 8,
) -> ChallengerDraft:
    kind = proposal.lever.get("kind")
    if kind not in TEXT_LEVERS:
        raise DraftError(f"{proposal.id}: lever kind {kind!r} is not a text lever; code changes ship as briefs")
    edits = proposal.lever.get("text_edits") or []
    if not edits:
        raise DraftError(f"{proposal.id}: no text_edits on the lever")

    slug = slugify(proposal.id.lower() + "_" + proposal.title)
    package_name = f"cfm_agent_v_5_3_{slug}"
    module = f"energy_oil_forecasting.{package_name}"
    out_root = settings.challengers_dir / proposal.id
    package_dir = out_root / package_name
    if package_dir.exists():
        shutil.rmtree(package_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_root, package_dir, ignore=shutil.ignore_patterns(*IGNORED))

    rewritten = rewrite_identity(package_dir, package_name=package_name, module=module)
    for edit in edits:
        target = package_dir / edit["file"]
        if not target.exists() or ".." in edit["file"]:
            raise DraftError(f"{proposal.id}: text edit targets a file not in the package: {edit['file']}")
        replace_exactly_once(target, edit["find"], edit["replace"])
    entries = write_manifest(package_dir)

    stream_id = f"v53_{slug}_advanced"
    start = start or (date.today() + timedelta(days=1))
    stream = {
        "stream_id": stream_id,
        "agent_id": package_name,
        "module": module,
        "model": settings.challenger_model,
        "runs_per_day": settings.challenger_runs_per_day,
        "runs_dirname": f"runs_{stream_id}",
        "calibration_dirname": f"calibration_{stream_id}",
        "run_id_prefix": f"{package_name}__advanced",
        "code_execution_enabled": False,
        "window": [start.isoformat(), (start + timedelta(weeks=weeks)).isoformat()],
        "prompt_version": f"{package_name}_builtin",
        "fingerprint_prefix": f"cfm_v5_3_{slug}_package",
        "champion": "v52_advanced",
        "proposal_id": proposal.id,
    }
    (out_root / "stream.yaml").write_text(yaml.safe_dump(stream, sort_keys=False), encoding="utf-8")
    cal = out_root / "calibration"
    cal.mkdir(exist_ok=True)
    (cal / "v001.json").write_text(
        json.dumps(
            {
                "version": "v001",
                "effective_from": "2004-01-01",
                "parent_version": None,
                "settings_overlay": {},
                "layer": {"centre_gain": {}, "width_scale": {}, "rw_anchor": {}},
                "source_proposal_id": None,
                "notes": f"identity baseline for challenger {stream_id}",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    register = out_root / "REGISTER.md"
    register.write_text(render_register(proposal, stream, package_name, module, package_dir, entries), encoding="utf-8")
    return ChallengerDraft(
        proposal_id=proposal.id,
        slug=slug,
        package_name=package_name,
        module=module,
        stream_id=stream_id,
        package_dir=str(package_dir),
        manifest_entries=entries,
        edits_applied=len(edits),
        files_rewritten=rewritten,
        register_path=str(register),
    )


def render_register(
    proposal: Proposal, stream: dict[str, Any], package_name: str, module: str, package_dir: Path, entries: int
) -> str:
    s = stream
    return f"""# Register challenger {s["stream_id"]} for {proposal.id}

**Drafted by the coach; nothing below has been applied.** Every step is a human edit to a guarded file.

Proposal: {proposal.title}
Lever: {proposal.lever.get("kind")}; edits: {len(proposal.lever.get("text_edits") or [])} (see stream.yaml and the package diff)

## 1. Move the package into place

```bash
mv {package_dir} implementations/energy_oil_forecasting/{package_name}
cp -r {package_dir.parent / "calibration"} implementations/energy_oil_forecasting/cfm_coach/{s["calibration_dirname"]}
```

## 2. targets.py -- add one AgentTarget (imports at the top, entry below V52)

```python
from energy_oil_forecasting.{package_name}.config import AGENT_NAME as V53_{s["agent_id"].upper()}_AGENT_NAME, PACKAGE_ROOT as V53_ROOT, CfmV52Settings as V53Settings

V53 = AgentTarget(
    agent_id="{package_name}",
    module="{module}",
    settings_cls=V53Settings,
    package_root=V53_ROOT,
    fingerprint_prefix="{s["fingerprint_prefix"]}",
    prompt_version="{s["prompt_version"]}",
)
TARGETS = {{target.agent_id: target for target in (V50, V52, V53)}}
```

## 3. streams.py -- add one RunStream and schedule it beside the champion

```python
V53_{s["slug"].upper() if "slug" in s else "X"} = RunStream(
    stream_id="{s["stream_id"]}",
    target=V53,
    model=ADVANCED_MODEL,
    runs_per_day={s["runs_per_day"]},
    runs_dirname="{s["runs_dirname"]}",
    calibration_dirname="{s["calibration_dirname"]}",
    run_id_prefix="{s["run_id_prefix"]}",
    description="Challenger for {proposal.id}: {proposal.title[:60]}",
    window=(date.fromisoformat("{s["window"][0]}"), date.fromisoformat("{s["window"][1]}")),
    code_execution_enabled=False,
)
STREAMS = (V50_LITE, V52_ADVANCED, V52_LITE, <the new stream>)
SCHEDULED_STREAMS = (V52_ADVANCED, V52_LITE, <the new stream>)
```

`tests/test_streams.py` asserts the scheduled set literally; update that assertion.

## 4. pyproject.toml

Add `implementations/energy_oil_forecasting/cfm_coach/review/challengers` to `norecursedirs` under `[tool.pytest.ini_options]` if not already present, so drafts are never collected.

## 5. Verify

```bash
cd implementations/energy_oil_forecasting/{package_name} && shasum -a 256 -c MANIFEST.sha256 | grep -vc OK   # expect 0
uv run pytest -q implementations/energy_oil_forecasting/{package_name}/tests
uv run python -c "from energy_oil_forecasting.cfm_coach.streams import stream_for; print(stream_for('{s["stream_id"]}').model)"
```

## 6. Cost and end

{s["runs_per_day"]} runs/day on {s["model"]} for {len(s["window"])} weeks alongside the champion. The window ends the stream by itself; retire it the `v50_lite` way afterwards (drop from SCHEDULED_STREAMS, keep in STREAMS).

## 7. Judge

After {DEFAULT_REVIEW_SETTINGS.pairing_min_cutoffs} shared cutoffs:

```bash
uv run python -m energy_oil_forecasting.cfm_coach.review.pairing --champion v52_advanced --challenger {s["stream_id"]}
```

Manifest entries: {entries}.
"""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--proposal", required=True)
    args = parser.parse_args(argv)
    store = ProposalStore()
    proposal = next((p for p in store.load_all() if p.id == args.proposal), None)
    if proposal is None:
        raise SystemExit(f"no proposal {args.proposal}")
    if proposal.decision.status != "accepted":
        raise SystemExit(f"{proposal.id} is {proposal.decision.status}, not accepted")
    draft = draft_challenger(proposal)
    print(json.dumps(draft.model_dump(), indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "ChallengerDraft",
    "DraftError",
    "TEXT_LEVERS",
    "draft_challenger",
    "manifest_lines",
    "replace_exactly_once",
    "rewrite_identity",
    "slugify",
    "verify_manifest",
    "write_manifest",
]
