"""Canonical serialization and identity strings (specification Section 1.7, DD-17).

One encoder, :func:`canonical_json`, produces every identity payload. It accepts
plain JSON values, dates, timestamps, numpy scalars and arrays, tuples, and sets;
anything else raises ``TypeError`` rather than falling back to ``str()``. Floats
are written with Python's shortest round-trip representation, ``-0.0`` becomes
``0.0``, strings are NFC-normalised, and NaN or infinite values are rejected.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import unicodedata
from typing import Any

import numpy as np
import pandas as pd


_DATE_FORMAT = "%Y-%m-%d"
_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S"
_SHA256_HEX_LENGTH = 64
_SEED_HEX_DIGITS = 8


def _encode(value: Any) -> Any:  # noqa: PLR0911 - one branch per supported type
    """Convert a non-JSON value to its canonical JSON form, or raise ``TypeError``."""
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is not None:
            value = value.tz_convert("UTC").tz_localize(None)
        return value.strftime(_TIMESTAMP_FORMAT)
    if isinstance(value, dt.datetime):
        if value.tzinfo is not None:
            value = value.astimezone(dt.timezone.utc).replace(tzinfo=None)
        return value.strftime(_TIMESTAMP_FORMAT)
    if isinstance(value, dt.date):
        return value.strftime(_DATE_FORMAT)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray | tuple):
        return list(value)
    if isinstance(value, set | frozenset):
        return sorted(value)
    raise TypeError(f"canonical_json cannot encode {type(value).__name__}")


def _normalise(value: Any) -> Any:  # noqa: PLR0911 - one branch per supported type
    """Walk a payload: fold ``-0.0`` to ``0.0``, NFC-normalise strings, and expand containers."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return 0.0 if value == 0.0 else value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return _normalise(float(value))
    if isinstance(value, dict):
        return {_normalise_key(key): _normalise(item) for key, item in value.items()}
    if isinstance(value, list | tuple | np.ndarray):
        return [_normalise(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted(_normalise(item) for item in value)
    return value


def _normalise_key(key: Any) -> Any:
    if isinstance(key, str):
        return unicodedata.normalize("NFC", key)
    if isinstance(key, np.integer):
        return int(key)
    if isinstance(key, np.floating):
        return float(key)
    return key


def canonical_json(payload: Any) -> str:
    """Return the canonical JSON text of ``payload`` (Section 1.7).

    Raises ``ValueError`` for NaN or infinite floats and ``TypeError`` for any
    type the encoder does not recognise.
    """
    return json.dumps(
        _normalise(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=_encode,
    )


def identity(payload: Any, *, prefix: str) -> str:
    """Return ``"<prefix>:sha256:<hex>"`` over the canonical JSON of ``payload``."""
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"{prefix}:sha256:{digest}"


def sha256_hex(identity_string: str) -> str:
    """Return the hexadecimal digest carried by an identity string."""
    marker = ":sha256:"
    if marker not in identity_string:
        raise ValueError(f"not an identity string: {identity_string!r}")
    digest = identity_string.rsplit(marker, 1)[1]
    if len(digest) != _SHA256_HEX_LENGTH:
        raise ValueError(f"identity string carries a malformed digest: {identity_string!r}")
    return digest


def seed_from_identity(identity_string: str) -> int:
    """Return the first 32 bits of the digest as an integer (the suite seed rule)."""
    return int(sha256_hex(identity_string)[:_SEED_HEX_DIGITS], 16)


__all__ = ["canonical_json", "identity", "seed_from_identity", "sha256_hex"]
