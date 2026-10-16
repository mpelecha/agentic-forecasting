"""Return-path numerical suite for ``cfm_agent_v_6_0`` (specification Sections 1.3 to 1.6).

Modules: ``paths`` (block sampling, compounding, validation, failure codes),
``bootstrap``, ``sparse_ar``, ``lightgbm_return``, and ``ensemble``. Imports are
kept lazy so that importing the package does not import LightGBM.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_EXPORTS: dict[str, str] = {
    "ComponentFailure": "paths",
    "ReturnPathComponent": "paths",
    "HistoricalBootstrapReturn": "bootstrap",
    "SparseARReturn": "sparse_ar",
    "LightGBMReturn": "lightgbm_return",
    "ReturnPathEnsemble": "ensemble",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"{__name__}.{module_name}"), name)


__all__ = [
    "ComponentFailure",
    "HistoricalBootstrapReturn",
    "LightGBMReturn",
    "ReturnPathComponent",
    "ReturnPathEnsemble",
    "SparseARReturn",
]
