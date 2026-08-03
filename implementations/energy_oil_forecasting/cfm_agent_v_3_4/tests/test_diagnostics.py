"""Market diagnostics and fingerprint tests."""

from datetime import datetime

import pandas as pd
from energy_oil_forecasting.cfm_agent_v_3_4.diagnostics import compute_market_diagnostics
from energy_oil_forecasting.cfm_agent_v_3_4.fingerprints import stable_fingerprint


def test_diagnostics_are_deterministic() -> None:
    dates = pd.bdate_range("2025-01-01", periods=70)
    frame = pd.DataFrame({"timestamp": dates, "value": range(70)})
    a = compute_market_diagnostics(frame, as_of=datetime(2025, 4, 15))
    b = compute_market_diagnostics(frame, as_of=datetime(2025, 4, 15))
    assert a == b
    assert a.return_5b is not None


def test_fingerprint_is_order_insensitive_for_mapping_keys() -> None:
    assert stable_fingerprint({"a": 1, "b": 2}, prefix="x") == stable_fingerprint({"b": 2, "a": 1}, prefix="x")
