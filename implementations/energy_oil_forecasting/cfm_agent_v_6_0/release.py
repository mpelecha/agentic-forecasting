"""Release tooling for ``cfm_agent_v_6_0`` (specification Sections 9 and 10).

Run from the repository root::

    uv run python -m energy_oil_forecasting.cfm_agent_v_6_0.release <command>

Commands: snapshot-protected, verify-protected, write-manifest, verify-manifest,
clean-caches, snapshot-data, package, verify-archive, preflight-models, sensitivity.
The manifest format and exclusions match ``cfm_coach/review/challenger.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from energy_oil_forecasting.cfm_agent_v_6_0.config import AGENT_NAME, PACKAGE_ROOT


REPO_ROOT = PACKAGE_ROOT.parents[2]
SHARED_ROOT = PACKAGE_ROOT.parent
SCOPE_FILE = PACKAGE_ROOT / "PROTECTED_SCOPE.txt"
PROTECTED_BASELINE = PACKAGE_ROOT / "PROTECTED_BASELINE.sha256"
MANIFEST_NAME = "MANIFEST.sha256"
DATA_SNAPSHOT_HASH_NAME = "DATA_SNAPSHOT.sha256"
OUTPUTS_ROOT = SHARED_ROOT / "outputs" / AGENT_NAME
PUBLIC_MODULES: tuple[str, ...] = (
    "config",
    "schemas",
    "fingerprints",
    "history",
    "diagnostics",
    "passages",
    "outputs",
    "forecast_engine.transformation",
    "policy.evidence_policy",
    "models.ensemble",
    "llm_roles",
    "response_control",
    "claim_support_verifier",
    "tools.market_data",
    "tools.research_pipeline",
    "tools.code_execution",
    "prompts",
    "predictor",
    "agent",
    "release",
)
_PRUNED_DIRS = {".git", ".venv", "node_modules", "__pycache__"}


# ------------------------------------------------------------- hashing ----


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_lines(root: Path, files: Iterable[Path]) -> list[str]:
    """``<sha256>  ./<path>`` lines, sorted by path in C-locale byte order."""
    lines = [f"{sha256_file(path)}  ./{path.relative_to(root).as_posix()}" for path in files]
    return sorted(lines, key=lambda line: line.split("  ./", 1)[1].encode())


def diff_lines(expected: list[str], actual: list[str]) -> list[str]:
    """Human-readable differences between two hash listings (missing, added, changed)."""

    def by_path(lines: list[str]) -> dict[str, str]:
        return {line.split("  ./", 1)[1]: line.split("  ./", 1)[0] for line in lines if "  ./" in line}

    before, after = by_path(expected), by_path(actual)
    report = [f"missing: {path}" for path in sorted(set(before) - set(after))]
    report += [f"added: {path}" for path in sorted(set(after) - set(before))]
    report += [f"changed: {path}" for path in sorted(set(before) & set(after)) if before[path] != after[path]]
    return report


def _walk(root: Path) -> Iterable[Path]:
    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if name not in _PRUNED_DIRS)
        for name in sorted(filenames):
            yield Path(directory) / name


# ------------------------------------------------------ protected scope ----


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Translate a gitignore-style pattern (``**`` spans directories, ``*`` one segment)."""
    out = ""
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if pattern.startswith("**/", index):
            out += "(?:.*/)?"
            index += 3
            continue
        if pattern.startswith("**", index):
            out += ".*"
            index += 2
            continue
        if char == "*":
            out += "[^/]*"
        elif char == "?":
            out += "[^/]"
        else:
            out += re.escape(char)
        index += 1
    return re.compile("^" + out + "$")


def read_scope(path: Path = SCOPE_FILE) -> tuple[list[re.Pattern[str]], list[re.Pattern[str]]]:
    """Return (include, exclude) regexes from a scope file; ``!`` prefixes an exclude."""
    includes: list[re.Pattern[str]] = []
    excludes: list[re.Pattern[str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("!"):
            excludes.append(glob_to_regex(line[1:].strip()))
        else:
            includes.append(glob_to_regex(line))
    return includes, excludes


def protected_files(root: Path = REPO_ROOT, scope: Path = SCOPE_FILE) -> list[Path]:
    includes, excludes = read_scope(scope)
    selected: list[Path] = []
    for path in _walk(root):
        rel = path.relative_to(root).as_posix()
        if any(pattern.match(rel) for pattern in includes) and not any(pattern.match(rel) for pattern in excludes):
            selected.append(path)
    return selected


def protected_lines(root: Path = REPO_ROOT, scope: Path = SCOPE_FILE) -> list[str]:
    return hash_lines(root, protected_files(root, scope))


def snapshot_protected(root: Path = REPO_ROOT, scope: Path = SCOPE_FILE, baseline: Path = PROTECTED_BASELINE) -> int:
    lines = protected_lines(root, scope)
    baseline.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"snapshot-protected: {len(lines)} files hashed into {baseline.name}")
    return 0


def verify_protected(root: Path = REPO_ROOT, scope: Path = SCOPE_FILE, baseline: Path = PROTECTED_BASELINE) -> int:
    expected = baseline.read_text(encoding="utf-8").splitlines()
    report = diff_lines(expected, protected_lines(root, scope))
    if report:
        print("verify-protected: FAILED")
        print("\n".join(report))
        return 1
    print(f"verify-protected: OK ({len(expected)} files unchanged)")
    return 0


# ------------------------------------------------------------- manifest ----


def manifest_lines(package_dir: Path = PACKAGE_ROOT) -> list[str]:
    """``<sha256>  ./<path>`` for every file under `package_dir`, C-locale sorted, manifest itself excluded."""
    files = []
    for path in package_dir.rglob("*"):
        if not path.is_file() or path.name == MANIFEST_NAME:
            continue
        if any(part == "__pycache__" for part in path.parts) or path.suffix == ".pyc" or path.name == ".DS_Store":
            continue
        files.append(path)
    return hash_lines(package_dir, files)


def write_manifest(package_dir: Path = PACKAGE_ROOT) -> int:
    lines = manifest_lines(package_dir)
    (package_dir / MANIFEST_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"write-manifest: {len(lines)} files hashed into {MANIFEST_NAME}")
    return 0


def verify_manifest(package_dir: Path = PACKAGE_ROOT) -> int:
    expected = (package_dir / MANIFEST_NAME).read_text(encoding="utf-8").splitlines()
    report = diff_lines(expected, manifest_lines(package_dir))
    if report:
        print("verify-manifest: FAILED")
        print("\n".join(report))
        return 1
    print(f"verify-manifest: OK ({len(expected)} files)")
    return 0


def clean_caches(package_dir: Path = PACKAGE_ROOT) -> int:
    removed = 0
    for path in sorted(package_dir.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_dir() and path.name == "__pycache__":
            shutil.rmtree(path)
            removed += 1
        elif path.is_file() and (path.suffix == ".pyc" or path.name == ".DS_Store"):
            path.unlink()
            removed += 1
    print(f"clean-caches: removed {removed} entries")
    return 0


# --------------------------------------------------------- data snapshot ----


def utc_stamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def snapshot_data(cache_dir: Path | None = None, outputs_root: Path = OUTPUTS_ROOT) -> Path:
    """Copy the yfinance cache to outputs/<agent>/data_snapshot_<UTC>/ and hash it."""
    if cache_dir is None:
        from energy_oil_forecasting.data import DEFAULT_CACHE_DIR  # noqa: PLC0415

        cache_dir = DEFAULT_CACHE_DIR
    source = Path(cache_dir)
    if not source.is_dir():
        raise SystemExit(f"snapshot-data: cache directory {source} does not exist")
    target = outputs_root / f"data_snapshot_{utc_stamp()}"
    shutil.copytree(source, target)
    files = [path for path in _walk(target) if path.is_file()]
    lines = hash_lines(target, files)
    (target / DATA_SNAPSHOT_HASH_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"snapshot-data: {len(lines)} files copied to {target}")
    return target


def verify_data_snapshot(snapshot: Path) -> bool:
    expected = (snapshot / DATA_SNAPSHOT_HASH_NAME).read_text(encoding="utf-8").splitlines()
    files = [path for path in _walk(snapshot) if path.is_file() and path.name != DATA_SNAPSHOT_HASH_NAME]
    return not diff_lines(expected, hash_lines(snapshot, files))


# ------------------------------------------------------------- packaging ----


def package(package_dir: Path = PACKAGE_ROOT, outputs_root: Path = OUTPUTS_ROOT) -> Path:
    outputs_root.mkdir(parents=True, exist_ok=True)
    clean_caches(package_dir)
    archive = outputs_root / f"{package_dir.name}_{utc_stamp()}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(package_dir.rglob("*")):
            if path.is_file():
                handle.write(path, arcname=f"{package_dir.name}/{path.relative_to(package_dir).as_posix()}")
    digest = sha256_file(archive)
    archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    print(f"package: {archive} ({digest})")
    return archive


def latest_archive(outputs_root: Path = OUTPUTS_ROOT) -> Path:
    candidates = sorted(outputs_root.glob(f"{AGENT_NAME}_*.zip"))
    if not candidates:
        raise SystemExit("verify-archive: no archive found; run package first")
    return candidates[-1]


def verify_archive(archive: Path | None = None, *, shared_root: Path = SHARED_ROOT, run_tests: bool = True) -> int:
    """Extract an archive next to a copy of the shared modules, compile, import, and run its tests.

    The extraction directory is put first on ``PYTHONPATH`` so the extracted copy,
    not the repository checkout, is what the child processes import; the smoke
    test asserts this.
    """
    import compileall  # noqa: PLC0415

    archive = archive or latest_archive()
    recorded = archive.with_suffix(".zip.sha256").read_text(encoding="utf-8").split()[0]
    if recorded != sha256_file(archive):
        print("verify-archive: FAILED (archive hash differs from the recorded hash)")
        return 1
    with tempfile.TemporaryDirectory() as temp:
        namespace = Path(temp) / shared_root.name
        namespace.mkdir()
        for shared in shared_root.glob("*.py"):
            shutil.copy2(shared, namespace / shared.name)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(namespace)
        extracted = namespace / AGENT_NAME
        if not compileall.compile_dir(str(extracted), quiet=1):
            print("verify-archive: FAILED (compileall)")
            return 1
        report = diff_lines(
            (extracted / MANIFEST_NAME).read_text(encoding="utf-8").splitlines(), manifest_lines(extracted)
        )
        if report:
            print("verify-archive: FAILED (manifest)\n" + "\n".join(report))
            return 1
        env = dict(os.environ, PYTHONPATH=str(Path(temp)))
        modules = [f"{shared_root.name}.{AGENT_NAME}.{name}" for name in PUBLIC_MODULES]
        smoke_script = (
            "import importlib, sys\n"
            "root = sys.argv[1]\n"
            "for name in sys.argv[2:]:\n"
            "    module = importlib.import_module(name)\n"
            "    if not module.__file__.startswith(root):\n"
            "        raise SystemExit(f'{name} imported from {module.__file__}, not the extraction')\n"
            "print('imports ok')\n"
        )
        smoke = subprocess.run(
            [sys.executable, "-c", smoke_script, str(Path(temp)), *modules],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if smoke.returncode != 0:
            print("verify-archive: FAILED (import smoke)\n" + smoke.stdout + smoke.stderr)
            return 1
        if run_tests:
            tests = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    "--import-mode=importlib",
                    str(extracted / "tests"),
                ],
                env=env,
                cwd=temp,
                capture_output=True,
                text=True,
                check=False,
            )
            tail = "\n".join(tests.stdout.splitlines()[-3:])
            if tests.returncode != 0:
                print("verify-archive: FAILED (pytest)\n" + tests.stdout[-4000:] + tests.stderr[-2000:])
                return 1
            print(f"verify-archive: OK\n{tail}")
        else:
            print("verify-archive: OK (tests skipped)")
    return 0


# ---------------------------------------------------- model preflight -----


def preflight_models(output: Path | None = None) -> int:
    """Confirm every configured role model through the proxy (Section 10.2)."""
    from energy_oil_forecasting.cfm_agent_v_6_0.config import DEFAULT_SETTINGS  # noqa: PLC0415
    from energy_oil_forecasting.cfm_agent_v_6_0.llm_roles import preflight_models as run_preflight  # noqa: PLC0415

    result: dict[str, Any] = run_preflight(DEFAULT_SETTINGS)
    for model, available in sorted(result["models"].items()):
        print(f"{'available' if available else 'MISSING  '}  {model}")
    print(f"method: {result['method']}")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"preflight-models: result written to {output}")
    missing = [model for model, available in result["models"].items() if not available]
    if missing:
        print(f"preflight-models: FAILED, missing {missing}; substitution is prohibited")
        return 1
    print("preflight-models: OK")
    return 0


# ----------------------------------------------------------- sensitivity ----

CALIBRATION_FILE = PACKAGE_ROOT / "CALIBRATION.md"
SENSITIVITY_BEGIN = "<!-- sensitivity:begin -->"
SENSITIVITY_END = "<!-- sensitivity:end -->"
SENSITIVITY_HORIZONS: tuple[int, ...] = (5, 10, 21)
SENSITIVITY_LEVELS: tuple[float, ...] = (0.1, 0.5, 0.9)


def filtered_service(context: Any, target_series_id: str, excluded_dates: Iterable[str]) -> Any:
    """Rebuild the context's series as a new data service with the target observations on the excluded dates removed.

    Removing an observation removes the return into it and the return out of it and
    replaces them with one bridging return, which is how the excluded-date returns
    leave the history. Covariates are copied as served.
    """
    import pandas as pd  # noqa: PLC0415
    from aieng.forecasting.data import DataService  # noqa: PLC0415
    from aieng.forecasting.data.features import StaticFrameAdapter  # noqa: PLC0415

    excluded = {pd.Timestamp(value).normalize() for value in excluded_dates}
    service = DataService()
    for series_id in context.series_ids:
        frame = context.get_series(series_id).copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        if series_id == target_series_id:
            frame = frame[~frame["timestamp"].dt.normalize().isin(excluded)]
        frame = frame[["timestamp", "value"]].reset_index(drop=True)
        frame["released_at"] = frame["timestamp"]
        service.register(series_id, StaticFrameAdapter(frame), context.get_metadata(series_id))
    return service


def _suite_summary(suite: Any, settings: Any) -> dict[str, Any]:
    quantiles: dict[str, dict[int, dict[float, float] | None]] = {}
    for name, component in suite.components.items():
        quantiles[name] = {
            item.horizon: {level: item.quantiles[level] for level in SENSITIVITY_LEVELS} for item in component.forecasts
        } or dict.fromkeys(SENSITIVITY_HORIZONS)
    quantiles["ensemble"] = {
        item.horizon: {level: item.quantiles[level] for level in SENSITIVITY_LEVELS}
        for item in suite.ensemble.forecasts
    }
    sparse = suite.components["sparse_ar"].training_data
    anchor = settings.uncertainty_anchor_coverage
    profiles = suite.diagnostics.historical_profiles
    return {
        "model_suite_id": suite.model_suite_id,
        "data_fingerprint": suite.data_fingerprint,
        "returns_available": suite.components["bootstrap"].training_data.returns_available,
        "dispositions": {name: item.failure_code for name, item in suite.dispositions.items()},
        "quantiles": quantiles,
        "ar_intercept": sparse.fitted_parameters.get("intercept"),
        "ar_residual_std": sparse.residual_std,
        "center_percentiles": {h: dict(profiles[h].return_percentiles) for h in profiles},
        "ladder": {
            h: {
                rung: (profiles[h].absolute_deviation_bands[cov] / profiles[h].absolute_deviation_bands[anchor])
                if profiles[h].absolute_deviation_bands.get(anchor)
                else None
                for rung, cov in settings.uncertainty_coverage.items()
            }
            for h in profiles
        },
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def sensitivity_markdown(
    cutoff: str, excluded: list[str], base: dict[str, Any], alt: dict[str, Any], settings: Any
) -> str:
    lines = [
        SENSITIVITY_BEGIN,
        f"Run at cutoff {cutoff} on {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. "
        f"Excluded dates: {', '.join(excluded)}. Informational only; the production model keeps every observation (DD-06).",
        "",
        "| Quantity | With excluded dates | Without excluded dates |",
        "|---|---|---|",
        f"| Returns available | {base['returns_available']} | {alt['returns_available']} |",
        f"| Data fingerprint | `{base['data_fingerprint'][-16:]}` | `{alt['data_fingerprint'][-16:]}` |",
        f"| Model-suite ID | `{base['model_suite_id'][-16:]}` | `{alt['model_suite_id'][-16:]}` |",
        f"| Sparse AR intercept (per day) | {_fmt(base['ar_intercept'], 6)} | {_fmt(alt['ar_intercept'], 6)} |",
        f"| Sparse AR residual standard deviation (per day) | {_fmt(base['ar_residual_std'])} | {_fmt(alt['ar_residual_std'])} |",
        "",
        "| Component | Horizon | P10 with | P50 with | P90 with | P10 without | P50 without | P90 without |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in (*settings.ensemble_weights, "ensemble"):
        for h in SENSITIVITY_HORIZONS:
            with_q = base["quantiles"].get(name, {}).get(h)
            without_q = alt["quantiles"].get(name, {}).get(h)
            cells = [
                _fmt(with_q[level], 2) if with_q else f"failed ({base['dispositions'].get(name)})"
                for level in SENSITIVITY_LEVELS
            ]
            cells += [
                _fmt(without_q[level], 2) if without_q else f"failed ({alt['dispositions'].get(name)})"
                for level in SENSITIVITY_LEVELS
            ]
            lines.append(f"| {name} | {h} | " + " | ".join(cells) + " |")
    lines += ["", "| Horizon | Percentile | Return with | Return without |", "|---|---|---|---|"]
    for h in SENSITIVITY_HORIZONS:
        for percentile in sorted(base["center_percentiles"].get(h, {})):
            lines.append(
                f"| {h} | P{percentile} | {_fmt(base['center_percentiles'][h][percentile])} | "
                f"{_fmt(alt['center_percentiles'].get(h, {}).get(percentile))} |"
            )
    lines += ["", "| Horizon | Uncertainty rung | Multiplier with | Multiplier without |", "|---|---|---|---|"]
    for h in SENSITIVITY_HORIZONS:
        for rung in settings.uncertainty_coverage:
            lines.append(
                f"| {h} | {rung} | {_fmt(base['ladder'].get(h, {}).get(rung))} | {_fmt(alt['ladder'].get(h, {}).get(rung))} |"
            )
    lines.append(SENSITIVITY_END)
    return "\n".join(lines)


def replace_sensitivity_section(path: Path, table: str) -> None:
    text = path.read_text(encoding="utf-8")
    start, end = text.index(SENSITIVITY_BEGIN), text.index(SENSITIVITY_END) + len(SENSITIVITY_END)
    path.write_text(text[:start] + table + text[end:], encoding="utf-8")


def run_sensitivity(
    *,
    cutoff: str | None,
    cache_dir: Path | None,
    service: Any = None,
    settings: Any = None,
    calibration_file: Path = CALIBRATION_FILE,
) -> dict[str, Any]:
    """Run the suite with and without the excluded dates and write the comparison into CALIBRATION.md."""
    from energy_oil_forecasting.cfm_agent_v_6_0.config import DEFAULT_SETTINGS  # noqa: PLC0415
    from energy_oil_forecasting.cfm_agent_v_6_0.tools.market_data import AuthoritativeSuiteTool  # noqa: PLC0415
    from energy_oil_forecasting.data import (  # noqa: PLC0415
        WTI_SERIES_ID,
        build_wti_multivariate_service,
        naive_utc_now,
    )

    settings = settings or DEFAULT_SETTINGS
    service = service or build_wti_multivariate_service(cache_dir=cache_dir)
    as_of = datetime.fromisoformat(cutoff) if cutoff else naive_utc_now()
    context = service.context(as_of=as_of)
    cutoff_date = cutoff or str(context.get_series(WTI_SERIES_ID)["timestamp"].iloc[-1])[:10]
    horizons = list(SENSITIVITY_HORIZONS)

    def run(data_service: Any) -> Any:
        tool = AuthoritativeSuiteTool(data_service, settings=settings)
        return tool._run_model_suite(
            context=data_service.context(as_of=as_of),
            target_series_id=WTI_SERIES_ID,
            horizons=horizons,
            frequency="B",
            cutoff_date=cutoff_date,
        )

    base = _suite_summary(run(service), settings)
    alt = _suite_summary(run(filtered_service(context, WTI_SERIES_ID, settings.sensitivity_excluded_dates)), settings)
    table = sensitivity_markdown(cutoff_date, list(settings.sensitivity_excluded_dates), base, alt, settings)
    replace_sensitivity_section(calibration_file, table)
    print(table)
    print(f"sensitivity: table written to {calibration_file}")
    return {"cutoff": cutoff_date, "with": base, "without": alt}


def sensitivity(cutoff: str | None, cache_dir: Path | None) -> int:
    run_sensitivity(cutoff=cutoff, cache_dir=cache_dir)
    return 0


# ------------------------------------------------------------------- CLI ----


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="release", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in (
        "snapshot-protected",
        "verify-protected",
        "write-manifest",
        "verify-manifest",
        "clean-caches",
        "package",
    ):
        sub.add_parser(name)
    preflight = sub.add_parser("preflight-models")
    preflight.add_argument("--output", type=Path, default=None, help="write the JSON result here")
    snapshot = sub.add_parser("snapshot-data")
    snapshot.add_argument("--cache-dir", type=Path, default=None)
    verify = sub.add_parser("verify-archive")
    verify.add_argument("--archive", type=Path, default=None)
    verify.add_argument("--no-tests", action="store_true")
    sens = sub.add_parser("sensitivity")
    sens.add_argument("--cutoff", default=None)
    sens.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    commands = {
        "snapshot-protected": snapshot_protected,
        "verify-protected": verify_protected,
        "write-manifest": write_manifest,
        "verify-manifest": verify_manifest,
        "clean-caches": clean_caches,
        "preflight-models": lambda: preflight_models(args.output),
        "snapshot-data": lambda: (snapshot_data(args.cache_dir), 0)[1],
        "package": lambda: (package(), 0)[1],
        "verify-archive": lambda: verify_archive(args.archive, run_tests=not args.no_tests),
        "sensitivity": lambda: sensitivity(args.cutoff, args.cache_dir),
    }
    return commands[args.command]()


if __name__ == "__main__":
    sys.exit(main())
