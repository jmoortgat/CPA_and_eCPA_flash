#!/usr/bin/env python3
"""Build the CH4-CO2-brine ternary (CCB) experimental-data parquet.

The CCB system (CH4 + CO2 dissolved in brine, salinity 0-4 mol/kg) is
*model-only* in the source materials: ``$ECPA_SOURCE_MATERIALS/EoS/CH4/CCB/eCPA_ELV.py``
generates the published ``MC_SOLUBILITY_*.png`` figures from a sweep over
(salt, salinity, Y_CO2_gas) and no embedded experimental dataset exists in the
upstream tree.

We still build an empty parquet so the figure script's data-loading convention
matches the rest of Phase 1 (one ``load_experimental(...)`` call per figure,
even if it returns zero rows).  Schema includes the ion-molality columns
(``m_Na``, ``m_Cl``, ``m_K``, ``m_Ca``, ``m_Mg``, ``m_SO4``) for forward
compatibility with future experimental data.

Schema:
    salt, T_K, P_bar, ms_mol_kg, Yc_gas,
    x_CH4_aq_exp, x_CO2_aq_exp, y_H2O_gas_exp,
    m_Na, m_Cl, m_K, m_Ca, m_Mg, m_SO4,
    author, source_file

Usage:
    python3 scripts/build_ccb.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE / "scripts"))

DATA_DIR = _CODE / "data"

# Reserved column ordering -- matches the convention used elsewhere in this
# project (cf. build_experimental_parquets.py for CO2 brines).
_COLUMNS = [
    "salt", "T_K", "P_bar", "ms_mol_kg", "Yc_gas",
    "x_CH4_aq_exp", "x_CO2_aq_exp", "y_H2O_gas_exp",
    "m_Na", "m_Cl", "m_K", "m_Ca", "m_Mg", "m_SO4",
    "author", "source_file",
]
_DTYPES = {
    "salt": "object", "T_K": "int64", "P_bar": "float64",
    "ms_mol_kg": "float64", "Yc_gas": "float64",
    "x_CH4_aq_exp": "float64", "x_CO2_aq_exp": "float64",
    "y_H2O_gas_exp": "float64",
    "m_Na": "float64", "m_Cl": "float64", "m_K": "float64",
    "m_Ca": "float64", "m_Mg": "float64", "m_SO4": "float64",
    "author": "object", "source_file": "object",
}


def build_ccb_ternary(out_path: Path | None = None) -> Path:
    out_path = out_path or (DATA_DIR / "ccb_ternary.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Empty frame with the reserved schema.
    df = pd.DataFrame({col: pd.Series(dtype=_DTYPES[col]) for col in _COLUMNS})
    df.to_parquet(out_path)
    print(f"  {len(df)} rows (schema-only -- no embedded experimental data)")
    print(f"  -> {out_path} ({out_path.stat().st_size/1024:.1f} kB)")
    return out_path


def main() -> int:
    print("Building ccb_ternary.parquet ...")
    build_ccb_ternary()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
