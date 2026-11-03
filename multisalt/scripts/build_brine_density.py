#!/usr/bin/env python3
"""Build the consolidated brine-density experimental parquet.

Run **once** (and again only when new experimental data arrives):

    python3 scripts/build_brine_density.py

Reads the per-T text files under
``$ECPA_SOURCE_MATERIALS/EoS/CO2/eCPA/MIXED_SALTS/DENSITY/EXP/EXP_{salt}_T{T}K.txt``
and produces a clean parquet under ``code/data/``:

    data/brine_density.parquet     -- aqueous (no-CO2) density of NaCl, KCl,
                                      CaCl2, MgCl2, Na2SO4 brines

File format (each EXP file): multiple blocks of the form

    #AUTHOR(YEAR)
    ms= <m_salt mol/kg>
    P[MPa] RHO[kg/m3]
    <P_MPa>  <rho_kg_m3>
    ...

The first line is the source line. It may carry the annotated form
``#AUTHOR(YEAR) | T=<value> | note`` to declare an isotherm the integer in
the filename cannot represent; see ``scripts/transcription_meta.py``.

Column schema in the output parquet:
    salt, T_K, P_bar, m_salt_mol_kg, m_Na, m_Cl, m_K, m_Ca, m_Mg, m_SO4,
    rho_exp_kg_m3, author, source_file

Charge balance is verified per row (zero net charge from the dissociated
ions implied by the salt + m_salt; tolerance 1e-9 mol/kg).
"""
from __future__ import annotations
import os

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE / "scripts"))
from transcription_meta import parse_source_line, true_temperature

DATA_DIR = _CODE / "data"
EOS_DENSITY = (
    Path(os.environ.get("ECPA_SOURCE_MATERIALS", "source_materials")) / "EoS" / "CO2" / "eCPA"
    / "MIXED_SALTS" / "DENSITY" / "EXP"
)

# Ion-charge vector matching the (Na, Cl, K, Ca, Mg, SO4) order used by
# build_experimental_parquets and ms_ecpa.parameters.
ZI = np.array([+1, -1, +1, +2, +2, -2], dtype=float)

# Dissociation stoichiometry: m_salt * vion -> molality of each ion in the
# (Na, Cl, K, Ca, Mg, SO4) order. Mirrors vion_SALT in OPT_PENELOUX.py.
SALT_DISSOCIATION = {
    "NaCl": np.array([1, 1, 0, 0, 0, 0], dtype=float),
    "KCl":  np.array([0, 1, 1, 0, 0, 0], dtype=float),
    "CaCl": np.array([0, 2, 0, 1, 0, 0], dtype=float),  # CaCl2
    "MgCl": np.array([0, 2, 0, 0, 1, 0], dtype=float),  # MgCl2
    "NaSO": np.array([2, 0, 0, 0, 0, 1], dtype=float),  # Na2SO4
}

SALT_FOLDERS = tuple(SALT_DISSOCIATION.keys())
# Temperatures the source files exist for; OPT_PENELOUX.py iterates a
# superset and skips missing files.
DENSITY_TEMPERATURES = (283, 298, 323, 348, 373, 398, 423, 448, 473)


def _parse_one_density_file(path: Path, salt: str) -> Optional[pd.DataFrame]:
    """Parse one EXP_{salt}_T{T}K.txt density file into a DataFrame.

    The file has multiple ``ms= <m>`` blocks, each followed by a header
    ``P[MPa] RHO[kg/m3]`` and 2-column rows. Pressures are stored in MPa
    in the source (multiplied by 10 -> bar to match the project convention).
    """
    text = path.read_text().splitlines()
    if len(text) < 4:
        return None
    if not text[0].startswith("#"):
        return None
    # Splitting on the pipe keeps the source key clean when a file carries the
    # annotated ``#AUTHOR(YEAR) | T=<value> | note`` header, which would
    # otherwise land in the author field whole.
    author, _, _ = parse_source_line(text[0])
    # These files sit flat in EXP/, so the directory name carries no isotherm;
    # the integer in the filename is the fallback, and a declared T wins.
    T_file = int(path.stem.split("_T")[-1].rstrip("K"))
    T_K = true_temperature(path, T_file)
    if float(T_K).is_integer():
        # Keep the plain integer the rest of the tree has always seen.
        T_K = int(T_K)
    else:
        # Row selection downstream matches the temperature as an integer
        # (scripts/fig_brine_density.py: ``exp["T_K"] == int(T_K)``), so a
        # fractional isotherm would build but never be plotted.
        print(f"  NOTE: {path.name} declares T={T_K} K; "
              f"scripts/fig_brine_density.py matches T_K on integers and "
              f"will not select these rows until it is updated.",
              file=sys.stderr)
    vion = SALT_DISSOCIATION[salt]

    rows = []
    i = 1
    while i < len(text):
        ln = text[i].strip()
        if not ln:
            i += 1
            continue
        # ms= <value>
        if not ln.lower().startswith("ms"):
            i += 1
            continue
        # parse molality
        try:
            m_salt = float(ln.split("=")[-1].strip())
        except ValueError:
            i += 1
            continue
        # next line is the header
        i += 2  # skip the "ms=" line and the column header
        # read (P, rho) rows until blank line or EOF
        while i < len(text):
            row = text[i].strip()
            if not row:
                i += 1
                break
            parts = row.split()
            if len(parts) < 2:
                i += 1
                break
            try:
                P_MPa = float(parts[0])
                rho = float(parts[1])
            except ValueError:
                i += 1
                continue
            P_bar = 10.0 * P_MPa
            mi = m_salt * vion
            zsum = float(np.sum(mi * ZI))
            if abs(zsum) > 1e-9 * max(1.0, float(np.sum(np.abs(mi)))):
                # Should never happen given fixed dissociation; defensive.
                i += 1
                continue
            rows.append({
                "salt": salt,
                "T_K": T_K,
                "P_bar": P_bar,
                "m_salt_mol_kg": m_salt,
                "m_Na":  mi[0],
                "m_Cl":  mi[1],
                "m_K":   mi[2],
                "m_Ca":  mi[3],
                "m_Mg":  mi[4],
                "m_SO4": mi[5],
                "rho_exp_kg_m3": rho,
                "author": author,
                "source_file": path.name,
            })
            i += 1
    if not rows:
        return None
    return pd.DataFrame(rows)


def build_brine_density(out_path: Path = None) -> Path:
    out_path = out_path or (DATA_DIR / "brine_density.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dfs = []
    file_count = 0
    for salt in SALT_FOLDERS:
        for T in DENSITY_TEMPERATURES:
            f = EOS_DENSITY / f"EXP_{salt}_T{T}K.txt"
            if not f.is_file():
                continue
            df = _parse_one_density_file(f, salt)
            if df is not None:
                dfs.append(df)
                file_count += 1
    if not dfs:
        raise RuntimeError(f"No brine-density .txt files parsed under {EOS_DENSITY}")
    out = pd.concat(dfs, ignore_index=True)
    out.to_parquet(out_path)
    by_salt = out.groupby("salt").size()
    by_T = out.groupby("T_K").size()
    print(f"  {file_count} files parsed -> {len(out)} rows")
    print(f"  per salt: {dict(by_salt)}")
    print(f"  per T:    {dict(by_T)}")
    print(f"  -> {out_path} ({out_path.stat().st_size/1024:.1f} kB)")
    return out_path


def main() -> int:
    print("Building brine_density.parquet ...")
    build_brine_density()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
