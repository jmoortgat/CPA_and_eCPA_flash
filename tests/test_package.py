"""Package-level smoke tests: imports and bundled data."""

from pathlib import Path

import pandas as pd


def test_import_core_modules():
    import ecpa  # noqa: F401
    from ecpa import constants, parameters, elv, flash, stability  # noqa: F401


def test_version_is_consistent():
    """ecpa.__version__, pyproject.toml and CITATION.cff must agree."""
    import re

    import ecpa

    root = Path(__file__).resolve().parents[1]
    pyproj = (root / "pyproject.toml").read_text()
    cff = (root / "CITATION.cff").read_text()

    m = re.search(r'^version\s*=\s*"([^"]+)"', pyproj, re.M)
    assert m, "no version in pyproject.toml"
    assert m.group(1) == ecpa.__version__

    m = re.search(r"^version:\s*(\S+)\s*$", cff, re.M)
    assert m, "no version in CITATION.cff"
    assert m.group(1).strip('"\'') == ecpa.__version__


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
