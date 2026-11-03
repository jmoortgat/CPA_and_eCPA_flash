#!/usr/bin/env python3
"""Build the CH4-CO2-H2O ternary (CCW) experimental-data parquet.

The CCW experimental data are *not* in standalone EXP_*.txt files; they are
embedded as Python literals at the top of each commented-out author block in
``$ECPA_SOURCE_MATERIALS/EoS/CH4/CCW/CPA_ELV_EXP.py``.  Three primary references:

  - DHIMA    (T = 344 K, P = 100, 200, 500, 750, 1000 bar)
  - ALGHAFRI (T = 323 K and 423 K, P = 20, 60, 100, 140, 180 bar)
  - QIN      (T = 376 K, P = 105, 205, 303, 402, 500 bar)

For each (T, P, point) we keep ``Yc_gas`` (water-free CO2 fraction in the gas
phase) and the measured aqueous CH4 (xh), aqueous CO2 (xc), and gas-phase H2O
(xw_G = x1h) mole fractions.  Each author reports a different subset of these
three views; rows are tagged with ``author`` so the figure script can pick
appropriate markers.

Schema:
    T_K, P_bar, Yc_gas,
    x_CH4_aq_exp, x_CO2_aq_exp, y_H2O_gas_exp,
    author, source_file

Missing entries are stored as NaN so callers can plot whichever quantity an
author reports.  Run once:

    python3 scripts/build_ccw.py
"""
from __future__ import annotations
import os

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE / "scripts"))

DATA_DIR = _CODE / "data"
SRC_FILE = (
    Path(os.environ.get("ECPA_SOURCE_MATERIALS", "source_materials")) / "EoS" / "CH4" / "CCW" / "CPA_ELV_EXP.py"
)


# ---------------------------------------------------------------------------
# Inline experimental data, transcribed verbatim from CPA_ELV_EXP.py.
# Each ``Yc_*[iT][iP]`` is a list of duplicate measurements at the same (T, P).
# Authors that do not report a given quantity have ``None`` for that field.
# ---------------------------------------------------------------------------

# DHIMA -- T = 344 K (lines 280-284 of CPA_ELV_EXP.py)
DHIMA = {
    "author": "DHIMA",
    "temps": [344],
    "press": [100.0, 200.0, 500.0, 750.0, 1000.0],
    "Yc": [[[0.4330, 0.1955],
            [0.4300, 0.1898],
            [0.4150, 0.1795],
            [0.4130, 0.1740],
            [0.4060]]],
    "x_CH4_aq": [[[0.000776, 0.0011],
                  [0.00131, 0.00182],
                  [0.002434, 0.00319],
                  [0.003027, 0.003893],
                  [0.00361]]],
    "x_CO2_aq": [[[0.008346, 0.003555],
                  [0.0113, 0.00539],
                  [0.01267, 0.006265],
                  [0.014347, 0.007105],
                  [0.015071]]],
    "y_H2O_gas": None,  # DHIMA does not report x_w in the gas phase
}

# ALGHAFRI -- T = 323 K and 423 K (lines 516-525 of CPA_ELV_EXP.py)
ALGHAFRI = {
    "author": "ALGHAFRI",
    "temps": [323, 423],
    "press": [20.0, 60.0, 100.0, 140.0, 180.0],
    "Yc": [[[0.4965], [0.5113], [0.5067], [0.4957], [0.5011]],
           [[0.5013], [0.5001], [0.4877], [0.4962], [0.4892]]],
    "x_CH4_aq": [[[0.000320], [0.000707], [0.000989], [0.001303], [0.001512]],
                 [[0.000281], [0.001029], [0.001635], [0.001811], [0.001915]]],
    "x_CO2_aq": [[[0.003121], [0.00851], [0.01159], [0.01310], [0.01420]],
                 [[0.001271], [0.00430], [0.00723], [0.00880], [0.01029]]],
    "y_H2O_gas": [[[0.01267], [0.00378], [0.00411], [0.00487], [0.00556]],
                  [[0.3445], [0.1145], [0.0689], [0.0489], [0.04694]]],
}

# QIN -- T = 376 K (lines 800-803 of CPA_ELV_EXP.py)
QIN = {
    "author": "QIN",
    "temps": [376],
    "press": [105.0, 205.0, 303.0, 402.0, 500.0],
    "Yc": [[[0.7436, 0.5765, 0.4450],
            [0.7402, 0.5248, 0.4252],
            [0.7289, 0.5432, 0.4213],
            [0.7237, 0.5260, 0.4103],
            [0.7224, 0.5286, 0.4057]]],
    "x_CH4_aq": [[[0.00038, 0.00074, 0.00084],
                  [0.00077, 0.00127, 0.00136],
                  [0.00104, 0.00166, 0.00188],
                  [0.00127, 0.00199, 0.00222],
                  [0.00152, 0.00215, 0.00258]]],
    "x_CO2_aq": [[[0.01065, 0.00855, 0.00627],
                  [0.01524, 0.01186, 0.0094],
                  [0.01801, 0.01369, 0.01074],
                  [0.01882, 0.01414, 0.01172],
                  [0.01971, 0.01549, 0.01179]]],
    "y_H2O_gas": [[[0.0208, 0.01906, 0.01747],
                   [0.02296, 0.01765, 0.01581],
                   [0.02317, 0.01905, 0.01681],
                   [0.0252, 0.01923, 0.01695],
                   [0.02676, 0.02199, 0.01821]]],
}


AUTHORS = (DHIMA, ALGHAFRI, QIN)


def _flatten_author(entry: dict) -> list[dict]:
    """Flatten one author's [iT][iP][k] nested lists into row dicts."""
    out: list[dict] = []
    temps = entry["temps"]
    press = entry["press"]
    Yc = entry["Yc"]
    xh = entry["x_CH4_aq"]
    xc = entry["x_CO2_aq"]
    xw_G = entry["y_H2O_gas"]   # may be None for authors that do not report it
    for iT, T in enumerate(temps):
        for iP, P in enumerate(press):
            n = len(Yc[iT][iP])
            for k in range(n):
                out.append({
                    "T_K": int(T),
                    "P_bar": float(P),
                    "Yc_gas": float(Yc[iT][iP][k]),
                    "x_CH4_aq_exp": float(xh[iT][iP][k]),
                    "x_CO2_aq_exp": float(xc[iT][iP][k]),
                    "y_H2O_gas_exp": (
                        float(xw_G[iT][iP][k]) if xw_G is not None else np.nan
                    ),
                    "author": entry["author"],
                    "source_file": SRC_FILE.name,
                })
    return out


def build_ccw_ternary(out_path: Path | None = None) -> Path:
    out_path = out_path or (DATA_DIR / "ccw_ternary.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for entry in AUTHORS:
        rows.extend(_flatten_author(entry))
    df = pd.DataFrame(rows)
    df.to_parquet(out_path)
    print(f"  {len(df)} rows from {len(AUTHORS)} authors")
    print(f"  per T:      {dict(df.groupby('T_K').size())}")
    print(f"  per author: {dict(df.groupby('author').size())}")
    print(f"  -> {out_path} ({out_path.stat().st_size/1024:.1f} kB)")
    return out_path


def main() -> int:
    print("Building ccw_ternary.parquet ...")
    build_ccw_ternary()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
