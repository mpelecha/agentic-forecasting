"""Stable fingerprints for v3 numerical artifacts."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_fingerprint(payload: Any, *, prefix: str) -> str:
    """Return a stable SHA-256 ID for a JSON-serializable payload."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return f"{prefix}:sha256:{hashlib.sha256(encoded).hexdigest()}"


__all__ = ["stable_fingerprint"]
