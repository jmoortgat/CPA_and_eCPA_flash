"""Package-level smoke tests: imports and bundled data."""

from pathlib import Path

import pandas as pd


def test_import_core_modules():
    import ecpa  # noqa: F401
    from ecpa import constants, parameters, elv, flash, stability  # noqa: F401


def test_make_params():
    from ecpa.parameters import make_params
    params = make_params()
    assert isinstance(params, dict) and len(params) > 0


def test_bundled_solution_table_loads():
    root = Path(__file__).resolve().parents[1]
    pq = root / "code" / "results" / "CPA_ELV_all.parquet"
    assert pq.exists(), "bundled CPA solution table missing"
    df = pd.read_parquet(pq)
    assert len(df) > 1000
    assert {"T", "P"}.issubset(set(df.columns)) or len(df.columns) >= 4
