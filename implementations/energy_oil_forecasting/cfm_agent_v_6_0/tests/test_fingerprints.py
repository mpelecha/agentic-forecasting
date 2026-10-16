"""Canonical serialization rules (specification Section 1.7, DD-17)."""

from __future__ import annotations

import datetime as dt
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from energy_oil_forecasting.cfm_agent_v_6_0.fingerprints import (
    canonical_json,
    identity,
    seed_from_identity,
    sha256_hex,
)


def test_dates_and_naive_timestamps():
    payload = {
        "d": dt.date(2026, 9, 14),
        "t": dt.datetime(2026, 9, 14, 13, 5, 7),
        "p": pd.Timestamp("2026-09-14 01:02:03"),
    }
    assert canonical_json(payload) == '{"d":"2026-09-14","p":"2026-09-14T01:02:03","t":"2026-09-14T13:05:07"}'


def test_tz_aware_timestamps_convert_to_utc_then_naive():
    aware = dt.datetime(2026, 9, 14, 20, 0, 0, tzinfo=dt.timezone(dt.timedelta(hours=-4)))
    stamp = pd.Timestamp("2026-09-14 20:00:00", tz="America/New_York")
    assert canonical_json({"a": aware}) == '{"a":"2026-09-15T00:00:00"}'
    assert canonical_json({"s": stamp}) == '{"s":"2026-09-15T00:00:00"}'


def test_numpy_scalars_arrays_tuples_and_sets():
    payload = {
        "i": np.int64(3),
        "f": np.float32(0.5),
        "b": np.bool_(True),
        "arr": np.array([1.5, 2.0]),
        "tup": (1, 2),
        "set": {"b", "a"},
        "nested": [np.array([[1, 2], [3, 4]])],
    }
    assert canonical_json(payload) == (
        '{"arr":[1.5,2.0],"b":true,"f":0.5,"i":3,"nested":[[[1,2],[3,4]]],"set":["a","b"],"tup":[1,2]}'
    )


def test_negative_zero_is_written_as_zero():
    assert (
        canonical_json({"x": -0.0, "arr": np.array([-0.0]), "n": np.float64(-0.0)}) == '{"arr":[0.0],"n":0.0,"x":0.0}'
    )
    assert identity({"x": -0.0}, prefix="p") == identity({"x": 0.0}, prefix="p")


def test_nan_and_infinity_are_rejected():
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})
    with pytest.raises(ValueError):
        canonical_json({"x": np.float64("inf")})
    with pytest.raises(ValueError):
        canonical_json({"x": np.array([1.0, np.nan])})


def test_unsupported_type_raises_type_error():
    class Opaque:
        pass

    with pytest.raises(TypeError):
        canonical_json({"x": Opaque()})
    with pytest.raises(TypeError):
        canonical_json({"x": object()})


def test_strings_are_nfc_normalised():
    decomposed = "Café"
    composed = "Café"
    assert canonical_json({"s": decomposed}) == canonical_json({"s": composed})
    assert canonical_json({decomposed: 1}) == canonical_json({composed: 1})
    assert "\\u00e9" in canonical_json({"s": decomposed})


def test_key_order_and_separators_are_canonical():
    assert canonical_json({"b": 1, "a": [1, {"z": 0, "y": None}]}) == '{"a":[1,{"y":null,"z":0}],"b":1}'


def test_identity_and_seed_rule():
    value = identity({"k": 1}, prefix="data")
    assert value.startswith("data:sha256:")
    assert len(sha256_hex(value)) == 64
    assert seed_from_identity(value) == int(sha256_hex(value)[:8], 16)
    assert 0 <= seed_from_identity(value) < 2**32


def test_identical_output_across_a_fresh_process():
    payload_code = (
        "{'d': dt.date(2026, 9, 14), 't': pd.Timestamp('2026-09-14 20:00', tz='UTC'), "
        "'f': np.float64(0.1) + np.float64(0.2), 'z': -0.0, 'arr': np.arange(3), 's': {'x', 'y'}, "
        "'u': 'Cafe\\u0301', 'n': None, 'b': True}"
    )
    script = (
        "import datetime as dt, numpy as np, pandas as pd\n"
        "from energy_oil_forecasting.cfm_agent_v_6_0.fingerprints import canonical_json, identity\n"
        f"payload = {payload_code}\n"
        "print(canonical_json(payload)); print(identity(payload, prefix='t'))\n"
    )
    child = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    child_json, child_identity = child.stdout.strip().splitlines()
    payload = eval(payload_code, {"dt": dt, "np": np, "pd": pd})  # noqa: S307 - test-local literal
    assert canonical_json(payload) == child_json
    assert identity(payload, prefix="t") == child_identity
