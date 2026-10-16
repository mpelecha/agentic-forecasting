"""Configuration tests (specification Section 7.4): settings.toml and CfmV60Settings agree."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
from energy_oil_forecasting.cfm_agent_v_6_0 import config
from energy_oil_forecasting.cfm_agent_v_6_0.config import (
    IDENTITY_FIELDS,
    MAPPING_FIELDS,
    PACKAGED_SETTINGS_FILE,
    CfmV60Settings,
    SettingsFileError,
    load_settings_file,
    resolved_settings_audit,
)
from pydantic import ValidationError


TEXT = PACKAGED_SETTINGS_FILE.read_text(encoding="utf-8")


def _key_occurrences(text: str) -> list[str]:
    """Every key or mapping-table name that defines a settings field, in file order."""
    keys: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        table = re.match(r"^\[([A-Za-z0-9_.]+)\]", stripped)
        if table:
            name = table.group(1).rsplit(".", 1)[-1]
            if name in MAPPING_FIELDS:
                keys.append(name)
            continue
        assignment = re.match(r"^([A-Za-z0-9_]+)\s*=", stripped)
        if assignment:
            keys.append(assignment.group(1))
    return keys


def _identity_tags(text: str) -> list[str]:
    tagged: list[str] = []
    for line in text.splitlines():
        if "[identity]" not in line:
            continue
        stripped = line.strip()
        table = re.match(r"^\[([A-Za-z0-9_.]+)\]", stripped)
        if table:
            tagged.append(table.group(1).rsplit(".", 1)[-1])
            continue
        assignment = re.match(r"^([A-Za-z0-9_]+)\s*=", stripped)
        if assignment:
            tagged.append(assignment.group(1))
    return tagged


def test_file_loads_and_bare_construction_equals_file():
    flat = load_settings_file(PACKAGED_SETTINGS_FILE)
    settings = CfmV60Settings()
    assert settings.model_dump() == flat
    assert config.DEFAULT_SETTINGS.model_dump() == flat


def test_every_field_has_exactly_one_key_and_every_key_one_field():
    keys = _key_occurrences(TEXT)
    fields = set(CfmV60Settings.model_fields)
    mapping_subkeys = {"bootstrap", "sparse_ar", "lightgbm"} | set(config.CENTER_PERCENTILE_ACTIONS)
    mapping_subkeys |= set(config.NOVELTY_CLASSES) | set(config.UNCERTAINTY_RUNGS)
    field_keys = [key for key in keys if key not in mapping_subkeys]
    assert sorted(field_keys) == sorted(fields), "keys and fields differ"
    assert len(field_keys) == len(set(field_keys)), "a key is defined more than once"


def test_duplicate_key_rejected(tmp_path: Path):
    duplicated = TEXT.replace(
        "[audit]\naudit_enabled = false", "[audit]\naudit_enabled = false\n\n[extra]\naudit_enabled = true"
    )
    assert duplicated != TEXT
    path = tmp_path / "settings.toml"
    path.write_text(duplicated, encoding="utf-8")
    with pytest.raises(SettingsFileError, match="more than once"):
        load_settings_file(path)


def test_unknown_key_rejected(tmp_path: Path):
    path = tmp_path / "settings.toml"
    path.write_text(TEXT + "\n[extra]\nnot_a_field = 1\n", encoding="utf-8")
    with pytest.raises(SettingsFileError, match="not_a_field"):
        load_settings_file(path)


def test_missing_key_rejected(tmp_path: Path):
    path = tmp_path / "settings.toml"
    path.write_text(TEXT.replace("audit_enabled = false", "", 1), encoding="utf-8")
    with pytest.raises(SettingsFileError, match="audit_enabled"):
        load_settings_file(path)


def test_model_fields_discoverable_by_suffix():
    discovered = sorted(name for name in CfmV60Settings.model_fields if name.endswith("_model"))
    assert discovered == [
        "claim_support_verifier_model",
        "cutoff_verifier_model",
        "main_model",
        "search_model",
        "structured_output_correction_model",
    ]
    assert set(CfmV60Settings().role_models) == set(config.LLM_ROLES)


def test_identity_tags_equal_identity_fields():
    assert _identity_tags(TEXT) == list(IDENTITY_FIELDS)
    assert set(IDENTITY_FIELDS) <= set(CfmV60Settings.model_fields)


def test_audit_carries_file_hash_and_overrides():
    audit = resolved_settings_audit(CfmV60Settings())
    assert audit["settings_file"] == str(PACKAGED_SETTINGS_FILE)
    assert audit["settings_file_sha256"] == hashlib.sha256(PACKAGED_SETTINGS_FILE.read_bytes()).hexdigest()
    assert audit["overrides"] == {}
    assert audit["values"] == CfmV60Settings().model_dump(mode="json")
    overridden = resolved_settings_audit(CfmV60Settings(main_temperature=0.7, audit_enabled=True))
    assert overridden["overrides"] == {"audit_enabled": True, "main_temperature": 0.7}


def test_from_file_records_source_and_applies_overrides(tmp_path: Path):
    path = tmp_path / "alternate.toml"
    path.write_text(TEXT.replace("ar_ridge_penalty = 1e-5", "ar_ridge_penalty = 0.5"), encoding="utf-8")
    settings = CfmV60Settings.from_file(path, audit_enabled=True)
    assert settings.ar_ridge_penalty == 0.5
    assert settings.audit_enabled is True
    audit = resolved_settings_audit(settings)
    assert audit["settings_file"] == str(path.resolve())
    assert audit["settings_file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert audit["overrides"] == {"ar_ridge_penalty": 0.5, "audit_enabled": True}


def test_unknown_override_rejected():
    with pytest.raises(ValidationError):
        CfmV60Settings(not_a_field=1)


def test_final_floor_above_simulated_floor_rejected():
    with pytest.raises(ValidationError, match="final_price_floor"):
        CfmV60Settings(final_price_floor=2.0, simulated_price_floor=1.0)


def test_non_increasing_ladder_rejected():
    ladder = dict(CfmV60Settings().uncertainty_coverage)
    ladder["small_wider"] = ladder["unchanged"]
    with pytest.raises(ValidationError, match="strictly increasing"):
        CfmV60Settings(uncertainty_coverage=ladder)


def test_anchor_must_equal_unchanged_rung():
    with pytest.raises(ValidationError, match="anchor"):
        CfmV60Settings(uncertainty_anchor_coverage=78)


def test_wrong_key_set_rejected():
    weights = {"bootstrap": 0.5, "sparse_ar": 0.5}
    with pytest.raises(ValidationError, match="ensemble_weights keys"):
        CfmV60Settings(ensemble_weights=weights)
    percentiles = dict(CfmV60Settings().center_percentiles)
    percentiles["extra"] = 55
    with pytest.raises(ValidationError, match="center_percentiles keys"):
        CfmV60Settings(center_percentiles=percentiles)


def test_center_percentiles_must_straddle_the_median():
    percentiles = dict(CfmV60Settings().center_percentiles)
    percentiles["small_down"] = 55
    with pytest.raises(ValidationError):
        CfmV60Settings(center_percentiles=percentiles)


def test_weights_must_be_positive():
    weights = dict(CfmV60Settings().ensemble_weights)
    weights["lightgbm"] = 0.0
    with pytest.raises(ValidationError, match="positive"):
        CfmV60Settings(ensemble_weights=weights)


def test_settings_are_frozen():
    settings = CfmV60Settings()
    with pytest.raises(ValidationError):
        settings.audit_enabled = True  # type: ignore[misc]


def test_identity_settings_follow_identity_field_order():
    values = config.identity_settings(CfmV60Settings())
    assert list(values) == list(IDENTITY_FIELDS)
    assert values["ensemble_weights"] == {"bootstrap": 0.5, "sparse_ar": 0.25, "lightgbm": 0.25}
