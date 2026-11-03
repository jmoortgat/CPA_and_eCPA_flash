#!/usr/bin/env python3
"""Build the consolidated CH4 Henry's-constant experimental-data parquet.

The Henry experimental data for CH4 in water and in brines lives **inline**
as Python literals at the bottom of
``$ECPA_SOURCE_MATERIALS/EoS/CH4/CH4-Brine/eCPA_HENRY.py`` -- not in a per-T text
file tree like the CH4-brine solubility data. Those literals are reproduced
here verbatim so a downstream reader does not have to re-execute the
monolithic script (which also fits a model).

References (and the lines in ``eCPA_HENRY.py`` that introduce them):

  Pure water (lines 473-501):
    CRAMER(1984), CROVETTO(1982), KIEPE(2003), RETTICH(1981),
    SERRA(2006), WANG(1994).

  KCl brine (lines 578-583):
    KIEPE(2003) at ms = 1, 2.5, 4 mol/kg.

  NaCl brine (lines 663-675):
    Authors at ms = 0.81, 1.95, 3.18, 4.70 mol/kg (per the ``CONCS`` list
    on line 618). Identities of the three brine-salinity overlays are
    untagged in the source; we record them as ``KRICHEVSKY+others`` /
    ``various`` because the original script does not name them. See the
    paper draft for the canonical attribution.

Output schema: ``salt`` ('H2O' or 'NaCl' or 'KCl'),
``m_salt_mol_kg``, ``T_K``, ``H_GPa``, ``author``, ``source_file``.
``H_GPa`` is the Henry's constant of CH4 in the named solvent in GPa
(same units as the original ``HENRY.png`` y-axis).
"""
from __future__ import annotations
import os

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parents[1]
DATA_DIR = _CODE / "data"
SOURCE_FILE = (
    Path(os.environ.get("ECPA_SOURCE_MATERIALS", "source_materials")) / "EoS" / "CH4" / "CH4-Brine" /
    "eCPA_HENRY.py"
)


# ---------------------------------------------------------------------------
# Pure-water Henry experimental data (eCPA_HENRY.py lines 473-501)
# ---------------------------------------------------------------------------
T_Cramer_C = [4, 12.6, 25, 37.8, 42, 61, 90.4, 111, 137.8, 159.3, 171.1,
              184.3, 187.7, 193.3, 210.7, 239.7, 264.4, 269.2, 281.0, 300]
T_Cramer = [T + 273.2 for T in T_Cramer_C]
H_Cramer = [2.58, 3.43, 4.55, 5.36, 5.80, 6.26, 6.61, 6.31, 6.32, 4.89,
            5.04, 4.05, 3.99, 3.12, 3.08, 2.58, 2.04, 2.24, 1.78, 1.13]

T_CROV = [297.5, 333.7, 385.3, 388.4, 430.6, 473.2, 518.3]
H_CROV_ln = [1.419, 1.801, 1.855, 1.846, 1.621, 1.362, 0.959]
H_CROV = [float(np.exp(h)) for h in H_CROV_ln]

T_KIEPE_water = [313.4, 353.2, 373.3]
H_KIEPE_water = [4.595, 6.296, 6.449]

T_RET = [275.5, 278.1, 284.0, 288.1, 293.1, 298.1, 303.2, 308.2, 313.2,
         318.2, 323.2, 328.2]
H_RET = [2.31, 2.50, 2.91, 3.22, 3.58, 3.93, 4.27, 4.61, 4.92, 5.19, 5.44, 5.67]

T_SERRA = [293.15, 295.15, 298.15, 300.15, 303.15, 305.15, 308.15, 310.15,
           313.15, 315.15, 318.15, 320.15, 323.15]
H_SERRA = [3.658, 3.818, 4.001, 4.158, 4.339, 4.476, 4.697, 4.872, 5.022,
           6.149, 5.331, 5.493, 5.658]

T_WANG = [283.15, 288.15, 293.15, 298.15]
H_WANG = [2.878, 3.335, 3.578, 3.955]

WATER_SOURCES = [
    ("CRAMER(1984)",   T_Cramer,        H_Cramer),
    ("CROVETTO(1982)", T_CROV,          H_CROV),
    ("KIEPE(2003)",    T_KIEPE_water,   H_KIEPE_water),
    ("RETTICH(1981)",  T_RET,           H_RET),
    ("SERRA(2006)",    T_SERRA,         H_SERRA),
    ("WANG(1994)",     T_WANG,          H_WANG),
]


# ---------------------------------------------------------------------------
# KCl brine Henry experimental data (eCPA_HENRY.py lines 578-583)
# ---------------------------------------------------------------------------
# At ms = 1, 2.5, 4 mol/kg (three points each, all from Kiepe 2003).
T_KCL_brine = [313.5, 353.3, 373.2]
KCL_BRINE_SOURCES = [
    (1.0,  T_KCL_brine, [5.081, 6.451, 6.657]),
    (2.5,  [313.2, 353.3, 373.2], [5.397, 6.640, 6.856]),
    (4.0,  [313.2, 353.3, 373.2], [6.071, 7.392, 7.641]),
]


# ---------------------------------------------------------------------------
# NaCl brine Henry experimental data (eCPA_HENRY.py lines 663-674)
# CONCS = [0.81, 1.95, 3.18, 4.70] mol/kg (Crovetto-style overlays;
# authors unattributed in the source script).
# ---------------------------------------------------------------------------
T1_C = [3.0, 13.0, 20.0, 41.6, 73.3, 100.4, 124.6, 146.0, 174.5, 191.6,
        240.6, 264.0]
H1   = [3.62, 4.19, 5.64, 7.12, 8.40, 9.26, 7.87, 8.15, 6.44, 5.35, 3.40,
        2.45]
T2_C = [0.5, 5.0, 12.6, 13.0, 26.3, 47.5, 78.0, 102.0, 131.9, 161.8, 205.2,
        224.6, 244.9, 271.4, 301.1]
H2   = [5.53, 5.87, 7.08, 6.06, 8.23, 10.53, 11.77, 12.26, 10.45, 8.62, 6.77,
        5.33, 3.82, 3.18, 2.07]
T3_C = [3.5, 12.5, 12.5, 24.5, 45.7, 60.2, 61.8, 61.9, 75.4, 88.8, 111.9,
        134.4, 162.2, 180.7, 203.7, 205.3, 225.1, 256.0, 269.5, 271.5, 295.6]
H3   = [8.16, 8.86, 9.09, 13.39, 15.59, 17.33, 14.29, 15.35, 17.16, 16.30,
        15.92, 14.87, 12.87, 11.20, 7.43, 10.87, 7.16, 5.67, 5.25, 5.05, 3.41]
T4_C = [3.5, 13.3, 31.7, 61.6, 89.5, 112.6, 115.7, 146.9, 159.4, 204.8,
        241.1, 287.9]
H4   = [17.56, 14.65, 23.56, 23.36, 28.15, 21.11, 27.47, 20.11, 18.71, 15.40,
        10.24, 6.74]

# Source script does not name these authors; record as a generic literature
# tag so downstream users do not over-interpret the attribution.
NACL_BRINE_SOURCES = [
    (0.81, [t + 273.15 for t in T1_C], H1),
    (1.95, [t + 273.15 for t in T2_C], H2),
    (3.18, [t + 273.15 for t in T3_C], H3),
    (4.70, [t + 273.15 for t in T4_C], H4),
]


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_ch4_henry(out_path: Path = None) -> Path:
    out_path = out_path or (DATA_DIR / "ch4_henry.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    src = SOURCE_FILE.name
    for label, Ts, Hs in WATER_SOURCES:
        for T, H in zip(Ts, Hs):
            rows.append({
                "salt": "H2O",
                "m_salt_mol_kg": 0.0,
                "T_K": float(T),
                "H_GPa": float(H),
                "author": label,
                "source_file": src,
            })
    for ms, Ts, Hs in KCL_BRINE_SOURCES:
        for T, H in zip(Ts, Hs):
            rows.append({
                "salt": "KCl",
                "m_salt_mol_kg": float(ms),
                "T_K": float(T),
                "H_GPa": float(H),
                "author": "KIEPE(2003)",
                "source_file": src,
            })
    for ms, Ts, Hs in NACL_BRINE_SOURCES:
        for T, H in zip(Ts, Hs):
            rows.append({
                "salt": "NaCl",
                "m_salt_mol_kg": float(ms),
                "T_K": float(T),
                "H_GPa": float(H),
                "author": "various",
                "source_file": src,
            })
    df = pd.DataFrame(rows)
    df.to_parquet(out_path)
    print(f"  {len(df)} rows")
    print(f"  per salt: {dict(df.groupby('salt').size())}")
    print(f"  -> {out_path} ({out_path.stat().st_size/1024:.1f} kB)")
    return out_path


def main() -> int:
    print("Building ch4_henry.parquet ...")
    build_ch4_henry()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
