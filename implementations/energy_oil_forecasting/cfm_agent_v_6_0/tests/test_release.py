"""Release tooling tests: scope parsing, protected snapshot, manifest, data snapshot, packaging."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest
from energy_oil_forecasting.cfm_agent_v_6_0 import release
from energy_oil_forecasting.cfm_agent_v_6_0.config import PACKAGE_ROOT, CfmV60Settings
from energy_oil_forecasting.cfm_coach.review.challenger import manifest_lines as coach_manifest_lines


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("pattern", "path", "expected"),
    [
        ("aieng-forecasting/**", "aieng-forecasting/aieng/x.py", True),
        ("aieng-forecasting/**", "aieng-forecasting/x.py", True),
        ("aieng-forecasting/**", "other/aieng-forecasting/x.py", False),
        ("implementations/energy_oil_forecasting/*.py", "implementations/energy_oil_forecasting/data.py", True),
        ("implementations/energy_oil_forecasting/*.py", "implementations/energy_oil_forecasting/sub/data.py", False),
        ("cfm_coach/**/*.md", "cfm_coach/README.md", True),
        ("cfm_coach/**/*.md", "cfm_coach/review/notes.md", True),
        ("cfm_coach/**/*.md", "cfm_coach/review/notes.txt", False),
        ("**/__pycache__/**", "a/b/__pycache__/c.pyc", True),
        ("**/*.pyc", "c.pyc", True),
        ("cfm_coach/runs_*/**", "cfm_coach/runs_v52_lite/2026/x.json", True),
        ("cfm_coach/runs_*/**", "cfm_coach/runs/x.json", False),
    ],
)
def test_glob_to_regex(pattern: str, path: str, expected: bool):
    assert bool(release.glob_to_regex(pattern).match(path)) is expected


def test_read_scope_separates_includes_and_excludes(tmp_path: Path):
    scope = _write(tmp_path / "scope.txt", "# comment\nsrc/**\n\n!src/**/*.pyc\n! src/skip/**\n")
    includes, excludes = release.read_scope(scope)
    assert [pattern.pattern for pattern in includes] == ["^src/.*$"]
    assert len(excludes) == 2


def test_packaged_scope_matches_specification_content():
    includes, excludes = release.read_scope(release.SCOPE_FILE)
    text = release.SCOPE_FILE.read_text(encoding="utf-8")
    for required in (
        "aieng-forecasting/**",
        "implementations/energy_oil_forecasting/*.py",
        "implementations/pyproject.toml",
        "pyproject.toml",
        "uv.lock",
        "Makefile",
        ".pre-commit-config.yaml",
        "implementations/energy_oil_forecasting/cfm_agent_v_5_0/**",
        "implementations/energy_oil_forecasting/cfm_agent_v_5_2/**",
        "implementations/energy_oil_forecasting/cfm_coach/**/*.py",
        "implementations/energy_oil_forecasting/cfm_coach/**/*.md",
        "!**/.venv/**",
        "!**/__pycache__/**",
        "!**/*.pyc",
        "!**/.DS_Store",
    ):
        assert f"\n{required}\n" in text, required
    assert includes and excludes


def test_protected_files_and_snapshot_roundtrip(tmp_path: Path):
    root = tmp_path / "repo"
    _write(root / "lib" / "a.py", "print(1)\n")
    _write(root / "lib" / "b.pyc", "x")
    _write(root / "lib" / "__pycache__" / "c.pyc", "x")
    _write(root / "lib" / "data" / "d.md", "keep\n")
    _write(root / "lib" / "runs" / "e.md", "daily\n")
    _write(root / "other.txt", "no\n")
    scope = _write(tmp_path / "scope.txt", "lib/**\n!**/*.pyc\n!**/__pycache__/**\n!lib/runs/**\n")
    selected = [path.relative_to(root).as_posix() for path in release.protected_files(root, scope)]
    assert selected == ["lib/a.py", "lib/data/d.md"]
    baseline = tmp_path / "baseline.sha256"
    assert release.snapshot_protected(root, scope, baseline) == 0
    lines = baseline.read_text(encoding="utf-8").splitlines()
    expected_digest = hashlib.sha256(b"print(1)\n").hexdigest()
    assert lines[0] == f"{expected_digest}  ./lib/a.py"
    assert release.verify_protected(root, scope, baseline) == 0
    _write(root / "lib" / "a.py", "print(2)\n")
    assert release.verify_protected(root, scope, baseline) == 1
    _write(root / "lib" / "a.py", "print(1)\n")
    _write(root / "lib" / "new.py", "x\n")
    assert release.verify_protected(root, scope, baseline) == 1


def test_packaged_protected_baseline_verifies():
    """The protected files must be unchanged from the step-1 snapshot (repository checkout only)."""
    if not (release.REPO_ROOT / "aieng-forecasting").is_dir():
        pytest.skip("protected files are only present in a repository checkout, not in an extracted archive")
    assert release.verify_protected() == 0


def test_manifest_write_and_verify(tmp_path: Path):
    package_dir = tmp_path / "pkg"
    _write(package_dir / "a.py", "a\n")
    _write(package_dir / "sub" / "b.txt", "b\n")
    _write(package_dir / "__pycache__" / "a.pyc", "x")
    _write(package_dir / ".DS_Store", "x")
    assert release.write_manifest(package_dir) == 0
    lines = (package_dir / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines()
    assert [line.split("  ./")[1] for line in lines] == ["a.py", "sub/b.txt"]
    assert release.verify_manifest(package_dir) == 0
    _write(package_dir / "sub" / "b.txt", "changed\n")
    assert release.verify_manifest(package_dir) == 1


def test_manifest_matches_the_coach_format(tmp_path: Path):
    package_dir = tmp_path / "pkg"
    _write(package_dir / "z.py", "z\n")
    _write(package_dir / "a" / "b.py", "b\n")
    _write(package_dir / "MANIFEST.sha256", "ignored\n")
    assert release.manifest_lines(package_dir) == coach_manifest_lines(package_dir)


def test_clean_caches(tmp_path: Path):
    package_dir = tmp_path / "pkg"
    _write(package_dir / "a.py", "a\n")
    _write(package_dir / "__pycache__" / "a.pyc", "x")
    _write(package_dir / "sub" / "stray.pyc", "x")
    _write(package_dir / ".DS_Store", "x")
    assert release.clean_caches(package_dir) == 0
    remaining = sorted(path.relative_to(package_dir).as_posix() for path in package_dir.rglob("*") if path.is_file())
    assert remaining == ["a.py"]


def test_snapshot_data_copies_and_hashes(tmp_path: Path):
    cache = tmp_path / "cache"
    _write(cache / "CL=F.csv", "timestamp,value\n2026-01-02,60\n")
    _write(cache / "nested" / "BZ=F.csv", "timestamp,value\n2026-01-02,65\n")
    outputs = tmp_path / "outputs"
    snapshot = release.snapshot_data(cache, outputs)
    assert snapshot.parent == outputs
    assert snapshot.name.startswith("data_snapshot_")
    assert (snapshot / "DATA_SNAPSHOT.sha256").exists()
    assert release.verify_data_snapshot(snapshot)
    _write(snapshot / "CL=F.csv", "timestamp,value\n2026-01-02,61\n")
    assert not release.verify_data_snapshot(snapshot)


def test_package_writes_archive_and_hash(tmp_path: Path):
    package_dir = tmp_path / "cfm_agent_v_6_0"
    _write(package_dir / "a.py", "a\n")
    _write(package_dir / "__pycache__" / "a.pyc", "x")
    outputs = tmp_path / "outputs"
    archive = release.package(package_dir, outputs)
    assert archive.parent == outputs
    recorded = archive.with_suffix(".zip.sha256").read_text(encoding="utf-8").split()
    assert recorded == [release.sha256_file(archive), archive.name]
    with zipfile.ZipFile(archive) as handle:
        assert handle.namelist() == ["cfm_agent_v_6_0/a.py"]


def test_verify_archive_rejects_a_tampered_archive(tmp_path: Path):
    package_dir = tmp_path / "cfm_agent_v_6_0"
    _write(package_dir / "a.py", "a\n")
    archive = release.package(package_dir, tmp_path / "outputs")
    archive.write_bytes(archive.read_bytes() + b"\0")
    assert release.verify_archive(archive, run_tests=False) == 1


def test_public_modules_exist():
    for name in release.PUBLIC_MODULES:
        path = PACKAGE_ROOT.joinpath(*name.split("."))
        assert path.with_suffix(".py").exists(), name


def test_sensitivity_on_the_synthetic_service(tmp_path: Path, synthetic_service):
    settings = CfmV60Settings(sensitivity_excluded_dates=["2026-09-10", "2026-09-11"])
    calibration = tmp_path / "CALIBRATION.md"
    calibration.write_text(
        "# note\n\n## 5. Sensitivity\n\n<!-- sensitivity:begin -->\nplaceholder\n<!-- sensitivity:end -->\n",
        encoding="utf-8",
    )
    result = release.run_sensitivity(
        cutoff="2026-09-14", cache_dir=None, service=synthetic_service, settings=settings, calibration_file=calibration
    )
    assert result["cutoff"] == "2026-09-14"
    assert result["with"]["returns_available"] - result["without"]["returns_available"] == 2
    assert result["with"]["data_fingerprint"] != result["without"]["data_fingerprint"]
    assert result["with"]["model_suite_id"] != result["without"]["model_suite_id"]
    assert set(result["with"]["quantiles"]) == {"bootstrap", "sparse_ar", "lightgbm", "ensemble"}
    text = calibration.read_text(encoding="utf-8")
    assert "placeholder" not in text
    assert text.startswith("# note\n\n## 5. Sensitivity\n\n<!-- sensitivity:begin -->")
    assert text.rstrip().endswith("<!-- sensitivity:end -->")
    assert "| ensemble | 21 |" in text
    assert "| 21 | substantially_wider |" in text
    assert "Excluded dates: 2026-09-10, 2026-09-11" in text
