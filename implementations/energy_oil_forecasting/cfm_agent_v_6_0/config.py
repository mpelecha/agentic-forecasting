"""Configuration for CFM Agent v6.0.

Every tunable value lives in ``settings.toml`` at the package root (specification
Section 7). This module holds the loader, the pydantic model that validates the
values, the list of fields bound into the model-suite identity, and the settings
audit. Python keeps structural constants only: identifiers, enumerations, and
validation bounds.
"""

from __future__ import annotations

import hashlib
import os
import tomllib
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator


AGENT_NAME = "cfm_agent_v_6_0"
PACKAGE_ROOT = Path(__file__).resolve().parent
SKILLS_ROOT = PACKAGE_ROOT / "skills"
PACKAGED_SETTINGS_FILE = PACKAGE_ROOT / "settings.toml"
SETTINGS_FILE_ENV = "CFM_V60_SETTINGS_FILE"

#: Fields written as their own TOML table named after the field (Section 7.1).
MAPPING_FIELDS: tuple[str, ...] = (
    "ensemble_weights",
    "center_percentiles",
    "novelty_multipliers",
    "uncertainty_coverage",
)

#: Every settings.toml key tagged ``[identity]``; bound into the model-suite ID
#: (Section 1.7). ``tests/test_settings_file.py`` parses the tags from the file
#: and requires the two lists to agree.
IDENTITY_FIELDS: tuple[str, ...] = (
    "model_num_paths",
    "simulation_length_business_days",
    "block_length_business_days",
    "history_window_business_days",
    "input_price_floor",
    "simulated_price_floor",
    "min_training_rows_margin",
    "ensemble_weights",
    "return_lags",
    "ar_ridge_penalty",
    "ar_fit_intercept",
    "lightgbm_n_estimators",
    "lightgbm_learning_rate",
    "lightgbm_num_leaves",
    "lightgbm_max_depth",
    "covariate_forward_fill_limit",
    "covariate_series_ids",
)

#: Structural enumerations (Section 7.1): component names in generator-spawn order,
#: centre actions in increasing-percentile order, uncertainty rungs in increasing-
#: coverage order, and the novelty classes.
ENSEMBLE_COMPONENTS: tuple[str, ...] = ("bootstrap", "sparse_ar", "lightgbm")
CENTER_PERCENTILE_ACTIONS: tuple[str, ...] = (
    "large_down",
    "moderate_down",
    "small_down",
    "small_up",
    "moderate_up",
    "large_up",
)
UNCERTAINTY_RUNGS: tuple[str, ...] = (
    "substantially_narrower",
    "moderately_narrower",
    "small_narrower",
    "unchanged",
    "small_wider",
    "moderately_wider",
    "substantially_wider",
)
NOVELTY_CLASSES: tuple[str, ...] = (
    "likely_new_relative_to_model_data",
    "possibly_partly_reflected",
    "likely_reflected_in_model_data",
    "indeterminate",
)
LLM_ROLES: tuple[str, ...] = (
    "main",
    "search",
    "cutoff_verifier",
    "structured_output_correction",
    "claim_support_verifier",
)


class SettingsFileError(ValueError):
    """The settings file is malformed, or its keys and the model's fields disagree."""


def _flatten(table: dict[str, Any], *, path: tuple[str, ...], into: dict[str, Any]) -> None:
    for key, value in table.items():
        if isinstance(value, dict):
            if key in MAPPING_FIELDS:
                _put(into, key, dict(value), path + (key,))
            else:
                _flatten(value, path=path + (key,), into=into)
        else:
            _put(into, key, value, path)


def _put(into: dict[str, Any], key: str, value: Any, path: tuple[str, ...]) -> None:
    if key in into:
        raise SettingsFileError(
            f"key {key!r} appears more than once (second occurrence under {'.'.join(path) or 'root'})"
        )
    into[key] = value


def load_settings_file(path: Path | str) -> dict[str, Any]:
    """Return the flattened key-to-value mapping of a settings file.

    Tables only group keys; the mapping-valued fields listed in
    :data:`MAPPING_FIELDS` are kept as dictionaries. A key that appears twice, a
    key that matches no :class:`CfmV60Settings` field, or a field with no key
    raises :class:`SettingsFileError`.
    """
    resolved = Path(path)
    with resolved.open("rb") as handle:
        raw = tomllib.load(handle)
    flat: dict[str, Any] = {}
    _flatten(raw, path=(), into=flat)
    fields = set(CfmV60Settings.model_fields)
    unknown = sorted(set(flat) - fields)
    if unknown:
        raise SettingsFileError(f"{resolved.name}: keys match no settings field: {unknown}")
    missing = sorted(fields - set(flat))
    if missing:
        raise SettingsFileError(f"{resolved.name}: settings fields have no key: {missing}")
    return flat


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _distinct_positive_ints(name: str, values: list[int]) -> None:
    if not values:
        raise ValueError(f"{name} must not be empty")
    if any(value < 1 for value in values):
        raise ValueError(f"{name} must contain positive integers")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not repeat a value")


class CfmV60Settings(BaseModel):
    """Validated settings; bare construction reproduces ``settings.toml`` exactly."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    _source_file: str | None = PrivateAttr(default=None)

    # [numerical]
    model_num_paths: int = Field(ge=1, le=100_000)
    simulation_length_business_days: int = Field(ge=1, le=260)
    block_length_business_days: int = Field(ge=1, le=260)
    history_window_business_days: int = Field(ge=0)
    input_price_floor: float = Field(gt=0.0)
    simulated_price_floor: float = Field(gt=0.0)
    final_price_floor: float = Field(ge=0.0)
    require_all_components: bool
    min_training_rows_margin: int = Field(ge=0)
    max_pre_execution_attempts: int = Field(ge=1, le=3)
    max_data_rows_per_series: int = Field(ge=1, le=5_000)
    ensemble_weights: dict[str, float]
    # [numerical.sparse_ar]
    return_lags: list[int]
    ar_ridge_penalty: float = Field(ge=0.0)
    ar_fit_intercept: bool
    # [numerical.lightgbm]
    lightgbm_n_estimators: int = Field(ge=1, le=10_000)
    lightgbm_learning_rate: float = Field(gt=0.0, le=1.0)
    lightgbm_num_leaves: int = Field(ge=2)
    lightgbm_max_depth: int = Field(ge=-1)
    covariate_forward_fill_limit: int = Field(ge=0)
    covariate_series_ids: list[str]
    # [diagnostics]
    diagnostic_return_windows: list[int]
    diagnostic_volatility_windows: list[int]
    diagnostic_jump_windows: list[int]
    trading_days_per_year: int = Field(ge=1, le=366)
    curve_flat_threshold: float = Field(ge=0.0)
    # [transformation]
    emergency_center_cap_usd: float = Field(gt=0.0)
    emergency_center_cap_price_fraction: float = Field(gt=0.0)
    uncertainty_anchor_coverage: int = Field(gt=0, lt=100)
    min_historical_returns: int = Field(ge=1)
    historical_conditioning: Literal["none", "volatility_quintile"]
    conditioning_volatility_window: int = Field(ge=2)
    conditioning_quantile_bins: int = Field(ge=2)
    center_percentiles: dict[str, int]
    novelty_multipliers: dict[str, float]
    uncertainty_coverage: dict[str, int]
    # [evidence]
    tier_1_min_publishers: int = Field(ge=1)
    tier_1_min_confidence: float = Field(ge=0.0, le=1.0)
    tier_2_min_publishers: int = Field(ge=1)
    tier_2_min_confidence: float = Field(ge=0.0, le=1.0)
    tier_3_min_publishers: int = Field(ge=1)
    tier_3_min_confidence: float = Field(ge=0.0, le=1.0)
    policy_mode: Literal["constrained_actions", "ensemble_locked"]
    # [research]
    research_max_results_per_query: int = Field(ge=1, le=50)
    source_resolution_timeout_seconds: float = Field(gt=0.0, le=300.0)
    source_resolution_max_redirects: int = Field(ge=0, le=20)
    search_verifier_max_attempts: int = Field(ge=1, le=5)
    search_verifier_confidence_threshold: int = Field(ge=1, le=10)
    research_max_passages_per_source: int = Field(ge=1, le=100)
    research_passage_chars: int = Field(ge=1)
    passage_min_chars: int = Field(ge=1)
    passage_abbreviations: list[str]
    # [llm.common]
    llm_top_p: float = Field(gt=0.0, le=1.0)
    llm_candidate_count: int = Field(ge=1, le=8)
    llm_frequency_penalty: float = Field(ge=-2.0, le=2.0)
    llm_presence_penalty: float = Field(ge=-2.0, le=2.0)
    llm_parallel_tool_calls: bool
    # [llm.main]
    main_model: str = Field(min_length=1)
    main_temperature: float = Field(ge=0.0, le=2.0)
    main_max_tokens: int = Field(ge=1)
    main_timeout_seconds: float = Field(gt=0.0, le=3_600.0)
    # [llm.search]
    search_model: str = Field(min_length=1)
    search_temperature: float = Field(ge=0.0, le=2.0)
    search_max_tokens: int = Field(ge=1)
    search_timeout_seconds: float = Field(gt=0.0, le=3_600.0)
    # [llm.cutoff_verifier]
    cutoff_verifier_model: str = Field(min_length=1)
    cutoff_verifier_temperature: float = Field(ge=0.0, le=2.0)
    cutoff_verifier_max_tokens: int = Field(ge=1)
    cutoff_verifier_timeout_seconds: float = Field(gt=0.0, le=3_600.0)
    # [llm.structured_output_correction]
    structured_output_correction_model: str = Field(min_length=1)
    structured_output_correction_temperature: float = Field(ge=0.0, le=2.0)
    structured_output_correction_max_tokens: int = Field(ge=1)
    structured_output_correction_timeout_seconds: float = Field(gt=0.0, le=3_600.0)
    # [llm.claim_support_verifier]
    claim_support_verifier_model: str = Field(min_length=1)
    claim_support_verifier_temperature: float = Field(ge=0.0, le=2.0)
    claim_support_verifier_max_tokens: int = Field(ge=1)
    claim_support_verifier_timeout_seconds: float = Field(gt=0.0, le=3_600.0)
    # [code_execution]
    code_execution_enabled: bool
    code_execution_max_accepted: int = Field(ge=0)
    code_execution_max_rejected: int = Field(ge=0)
    code_execution_timeout_seconds: float = Field(gt=0.0, le=3_600.0)
    code_execution_sandbox_timeout_seconds: int = Field(ge=1, le=86_400)
    code_execution_max_code_chars: int = Field(ge=1)
    code_execution_max_output_chars: int = Field(ge=1)
    code_execution_allow_internet: bool
    code_execution_prohibited_modules: list[str]
    code_execution_prohibited_calls: list[str]
    code_execution_prohibited_names: list[str]
    # [audit]
    audit_enabled: bool
    # [validation]
    live_spend_ceiling_usd: float = Field(gt=0.0)
    sensitivity_excluded_dates: list[str]

    def __init__(self, **overrides: Any) -> None:
        """Apply explicit overrides on top of the packaged (or env-named) settings file."""
        super().__init__(**{**_FILE_DEFAULTS, **overrides})

    @classmethod
    def from_file(cls, path: Path | str, **overrides: Any) -> CfmV60Settings:
        """Load an alternate settings file, then apply explicit overrides."""
        settings = cls.model_validate({**load_settings_file(path), **overrides})
        settings._source_file = str(Path(path).resolve())
        return settings

    @property
    def settings_file(self) -> Path:
        """The file whose values this instance was built from."""
        return Path(self._source_file) if self._source_file else SETTINGS_FILE

    @property
    def role_models(self) -> dict[str, str]:
        """Role name to configured model, for the five LLM roles."""
        return {role: getattr(self, f"{role}_model") for role in LLM_ROLES}

    # -- cross-field validation (Section 7.2) --------------------------------

    @model_validator(mode="after")
    def _validate_cross_field(self) -> CfmV60Settings:
        self._validate_weights()
        self._validate_ladder()
        self._validate_center_percentiles()
        self._validate_novelty()
        if self.final_price_floor > self.simulated_price_floor:
            raise ValueError("final_price_floor must be at or below simulated_price_floor (Section 3.5)")
        if not self.covariate_series_ids:
            raise ValueError("covariate_series_ids must not be empty")
        if len(set(self.covariate_series_ids)) != len(self.covariate_series_ids):
            raise ValueError("covariate_series_ids must be distinct")
        _distinct_positive_ints("return_lags", self.return_lags)
        _distinct_positive_ints("diagnostic_return_windows", self.diagnostic_return_windows)
        _distinct_positive_ints("diagnostic_volatility_windows", self.diagnostic_volatility_windows)
        _distinct_positive_ints("diagnostic_jump_windows", self.diagnostic_jump_windows)
        if self.block_length_business_days > self.simulation_length_business_days:
            raise ValueError("block_length_business_days must not exceed simulation_length_business_days")
        for value in self.sensitivity_excluded_dates:
            date.fromisoformat(value)
        return self

    def _validate_weights(self) -> None:
        if set(self.ensemble_weights) != set(ENSEMBLE_COMPONENTS):
            raise ValueError(f"ensemble_weights keys must be exactly {sorted(ENSEMBLE_COMPONENTS)}")
        if any(weight <= 0 for weight in self.ensemble_weights.values()):
            raise ValueError("ensemble_weights must all be positive")

    def _validate_ladder(self) -> None:
        if set(self.uncertainty_coverage) != set(UNCERTAINTY_RUNGS):
            raise ValueError(f"uncertainty_coverage keys must be exactly {sorted(UNCERTAINTY_RUNGS)}")
        ladder = [self.uncertainty_coverage[rung] for rung in UNCERTAINTY_RUNGS]
        if any(earlier >= later for earlier, later in zip(ladder, ladder[1:])):
            raise ValueError(
                "uncertainty_coverage must be strictly increasing from substantially_narrower to substantially_wider"
            )
        if not all(0 < value < 100 for value in ladder):  # structural: coverages are percentages
            raise ValueError("uncertainty_coverage values must lie strictly between 0 and 100")
        anchor = self.uncertainty_anchor_coverage
        if not self.uncertainty_coverage["small_narrower"] < anchor < self.uncertainty_coverage["small_wider"]:
            raise ValueError("uncertainty_anchor_coverage must sit between the narrowing and widening rungs")
        if anchor != self.uncertainty_coverage["unchanged"]:
            raise ValueError(
                "uncertainty_anchor_coverage must equal the unchanged rung so that unchanged yields exactly 1.0"
            )

    def _validate_center_percentiles(self) -> None:
        if set(self.center_percentiles) != set(CENTER_PERCENTILE_ACTIONS):
            raise ValueError(f"center_percentiles keys must be exactly {sorted(CENTER_PERCENTILE_ACTIONS)}")
        values = [self.center_percentiles[action] for action in CENTER_PERCENTILE_ACTIONS]
        if any(earlier >= later for earlier, later in zip(values, values[1:])):
            raise ValueError("center_percentiles must be strictly increasing from large_down to large_up")
        median = 50  # structural: the median percentile separates down from up actions
        if not all(0 < value < median for value in values[:3]):  # structural: three down actions
            raise ValueError("down-action percentiles must lie strictly between 0 and 50")
        if not all(median < value < 100 for value in values[3:]):  # structural: percentiles are percentages
            raise ValueError("up-action percentiles must lie strictly between 50 and 100")

    def _validate_novelty(self) -> None:
        if set(self.novelty_multipliers) != set(NOVELTY_CLASSES):
            raise ValueError(f"novelty_multipliers keys must be exactly {sorted(NOVELTY_CLASSES)}")
        if any(not 0.0 <= value <= 1.0 for value in self.novelty_multipliers.values()):
            raise ValueError("novelty_multipliers must lie in [0, 1]")


def _resolve_settings_file() -> tuple[Path, bool]:
    named = os.environ.get(SETTINGS_FILE_ENV)
    if named:
        return Path(named).resolve(), True
    return PACKAGED_SETTINGS_FILE, False


SETTINGS_FILE, SETTINGS_FILE_FROM_ENV = _resolve_settings_file()
_FILE_DEFAULTS: dict[str, Any] = load_settings_file(SETTINGS_FILE)
_PACKAGED_DEFAULTS: dict[str, Any] = (
    _FILE_DEFAULTS if SETTINGS_FILE == PACKAGED_SETTINGS_FILE else load_settings_file(PACKAGED_SETTINGS_FILE)
)
DEFAULT_SETTINGS = CfmV60Settings()


def resolved_settings_audit(settings: CfmV60Settings) -> dict[str, Any]:
    """Return the settings provenance record carried in every prediction's audit."""
    values = settings.model_dump(mode="json")
    overrides = {name: values[name] for name in sorted(values) if values[name] != _PACKAGED_DEFAULTS.get(name)}
    source = settings.settings_file
    return {
        "settings_file": str(source),
        "settings_file_sha256": _file_sha256(source) if source.exists() else None,
        "settings_file_from_env": SETTINGS_FILE_FROM_ENV and source == SETTINGS_FILE,
        "packaged_settings_file_sha256": _file_sha256(PACKAGED_SETTINGS_FILE),
        "identity_fields": list(IDENTITY_FIELDS),
        "values": values,
        "overrides": overrides,
    }


def identity_settings(settings: CfmV60Settings) -> dict[str, Any]:
    """Return the identity-tagged settings values in :data:`IDENTITY_FIELDS` order."""
    dumped = settings.model_dump(mode="json")
    return {name: dumped[name] for name in IDENTITY_FIELDS}


__all__ = [
    "AGENT_NAME",
    "CENTER_PERCENTILE_ACTIONS",
    "DEFAULT_SETTINGS",
    "ENSEMBLE_COMPONENTS",
    "IDENTITY_FIELDS",
    "LLM_ROLES",
    "MAPPING_FIELDS",
    "NOVELTY_CLASSES",
    "PACKAGED_SETTINGS_FILE",
    "PACKAGE_ROOT",
    "SETTINGS_FILE",
    "SETTINGS_FILE_ENV",
    "SETTINGS_FILE_FROM_ENV",
    "SKILLS_ROOT",
    "UNCERTAINTY_RUNGS",
    "CfmV60Settings",
    "SettingsFileError",
    "identity_settings",
    "load_settings_file",
    "resolved_settings_audit",
]
