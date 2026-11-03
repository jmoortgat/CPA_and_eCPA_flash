#!/usr/bin/env python3
"""Build CO2-H2O and CH4-H2O binary experimental-data parquets.

Reads the per-temperature ``EXP*_T{T}K.txt`` text files compiled by
F. M. Coelho Coelho:

  - CH4-H2O: ``$ECPA_SOURCE_MATERIALS/EoS/CH4/CH4-Water/{T}K/EXP{k}_T{T}K.txt``
    schema (all files): ``P [bar]  xc_W  yw_C`` (xc_W = CH4 in aq;
    yw_C = water in CH4-rich gas).

  - CO2-H2O: ``$ECPA_SOURCE_MATERIALS/EoS/CO2/CPA/SRK/T{T}K/EXP{k}_T{T}K.txt``
    schema (4 variants, but cols 1-3 always = P / xc_W / y-col):

       ``P [bar]  xc_W  yc_C``                       (yc_C = CO2 in gas)
       ``P [bar]  xc_W  yw_C``                       (yw_C = H2O in gas)
       ``P [bar]  xc_W  yw_C  rho_W [kg/m3]``        (4th col ignored)

    The 3rd-column convention is detected from the header line, mirroring
    the legacy ``CPA_ELV_OPT.py``. We normalise to ``y_H2O_gas`` (water in
    gas) so the parquet has a single y-column convention; for ``yc_C``
    files we store ``y_H2O = 1 - yc_C``.

Both files use the literal token ``x`` or ``X`` to mark a missing entry;
those rows are kept (the other column may still be valid) but the missing
field is recorded as NaN.

Writes:
    code/data/binaries_co2_water.parquet
    code/data/binaries_ch4_water.parquet

Columns:
    T_K, P_bar, x_CO2_aq_exp  (or x_CH4_aq_exp), y_H2O_gas_exp,
    author, source_file

Usage:
    python3 scripts/build_binaries.py
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
SRC = Path(os.environ.get("ECPA_SOURCE_MATERIALS", "source_materials")) / "EoS"
CH4_WATER_DIR = SRC / "CH4" / "CH4-Water"
# CO2/CPA holds the same experimental compilation twice, filed under the two
# cubic backbones it was paired with. The data bodies are identical except in
# two files where SRK is lossy, both checked against the original papers on
# 2026-09-18: SRK's T323K/EXP2 carries "X" where Hou 2013 (Table 2) records
# y_CO2 = 0.97189 +/- 0.00050 at 323.15 K / 1.089 MPa, and SRK's T538K/EXP1
# stops at 2500 bar where Todheide & Franck 1963 (Table 1b/1c, 265 C column)
# runs to 3500 bar. PR reproduces both in full, so PR is the canonical tree.
CO2_WATER_DIR = SRC / "CO2" / "CPA" / "PR"

_MISSING_TOKENS = ("x", "X")

# A file's name and its directory usually agree on the temperature. Two files
# disagree, and they disagree for opposite reasons, so each is handled by name:
#
#   CH4-Water/443K/EXP2_T444K.txt -- the FILE is right. Olds et al. 1942 report
#     on a Fahrenheit grid and this is their 340 degF isotherm, 444.26 K; the
#     directory rounded it down to 443. It carries 11 rows that appear nowhere
#     else, and until 2026-09-18 the per-directory glob `EXP*_T443K.txt` never
#     matched it, so it had never been read at all.
#
#   CO2/CPA/PR/T478K/EXP1_T473K.txt -- the FILE is right and the file is in the
#     wrong place. It is byte-identical to T473K/EXP1_T473K.txt (Todheide &
#     Franck 1963, 200 degC), a stray copy left in the 478 K directory. Reading
#     it would duplicate those 8 rows at a temperature nobody measured.
#
# The glob is therefore widened to `EXP*_T*K.txt` -- which admits exactly these
# two files and nothing else -- and the stray is excluded here by name.
MISFILED = frozenset({
    ("T478K", "EXP1_T473K.txt"),
})


def _is_misfiled(path: Path) -> bool:
    return (path.parent.name, path.name) in MISFILED


def _is_missing(tok: str) -> bool:
    return tok in _MISSING_TOKENS


def _parse_t_dirname(dirname: str, prefix: str) -> Optional[int]:
    """Extract integer T from a directory name like ``298K`` or ``T298K``."""
    name = dirname
    if prefix and name.startswith(prefix):
        name = name[len(prefix):]
    if not name.endswith("K"):
        return None
    try:
        return int(name[:-1])
    except ValueError:
        return None


def _parse_ch4_water_file(path: Path, T_K: int) -> Optional[pd.DataFrame]:
    """Parse one CH4-Water EXP file: ``P  xc_W (CH4 in aq)  yw_C (H2O in gas)``."""
    lines = path.read_text().splitlines()
    if len(lines) < 3:
        return None
    author = parse_source_line(lines[0])[0].split()[0]
    declared_T = parse_source_line(lines[0])[1]
    rows = []
    for ln in lines[2:]:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) < 3:
            continue
        try:
            P = float(parts[0])
        except ValueError:
            continue
        x_aq = float("nan")
        y_h2o = float("nan")
        if not _is_missing(parts[1]):
            try:
                x_aq = float(parts[1])
            except ValueError:
                pass
        if not _is_missing(parts[2]):
            try:
                y_h2o = float(parts[2])
            except ValueError:
                pass
        if not (np.isfinite(x_aq) or np.isfinite(y_h2o)):
            continue
        rows.append({
            "T_K": (declared_T if declared_T is not None else float(T_K)),
            "P_bar": float(P),
            "x_CH4_aq_exp": x_aq,
            "y_H2O_gas_exp": y_h2o,
            "author": author,
            "source_file": path.name,
        })
    if not rows:
        return None
    return pd.DataFrame(rows)


def _parse_co2_water_file(path: Path, T_K: int) -> Optional[pd.DataFrame]:
    """Parse one CO2-H2O EXP file.

    The 3rd-column convention (``yw_C`` = H2O in gas vs. ``yc_C`` = CO2
    in gas) is detected from the header line. We store ``y_H2O_gas_exp``
    in both cases (converting ``yc_C`` -> ``1 - yc_C``).
    """
    lines = path.read_text().splitlines()
    if len(lines) < 3:
        return None
    author = parse_source_line(lines[0])[0].split()[0]
    declared_T = parse_source_line(lines[0])[1]
    header_toks = lines[1].lower().split()
    if len(header_toks) < 4:
        return None
    col3 = header_toks[3]
    if col3 == "yw_c":
        col3_is_co2_in_gas = False
    elif col3 == "yc_c":
        col3_is_co2_in_gas = True
    else:
        # Unknown; default to yw_C (H2O in gas).
        col3_is_co2_in_gas = False

    rows = []
    for ln in lines[2:]:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) < 3:
            continue
        try:
            P = float(parts[0])
        except ValueError:
            continue
        x_aq = float("nan")
        y_h2o = float("nan")
        if not _is_missing(parts[1]):
            try:
                x_aq = float(parts[1])
            except ValueError:
                pass
        if not _is_missing(parts[2]):
            try:
                v = float(parts[2])
                y_h2o = (1.0 - v) if col3_is_co2_in_gas else v
            except ValueError:
                pass
        if not (np.isfinite(x_aq) or np.isfinite(y_h2o)):
            continue
        rows.append({
            "T_K": (declared_T if declared_T is not None else float(T_K)),
            "P_bar": float(P),
            "x_CO2_aq_exp": x_aq,
            "y_H2O_gas_exp": y_h2o,
            "author": author,
            "source_file": path.name,
            "col3_orig": "yc_C" if col3_is_co2_in_gas else "yw_C",
        })
    if not rows:
        return None
    return pd.DataFrame(rows)


def build_ch4_water(out_path: Path = None) -> Path:
    out_path = out_path or (DATA_DIR / "binaries_ch4_water.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dfs = []
    file_count = 0
    for Tdir in sorted(CH4_WATER_DIR.iterdir()):
        if not Tdir.is_dir():
            continue
        T_K = _parse_t_dirname(Tdir.name, prefix="")
        if T_K is None:
            continue
        for f in sorted(Tdir.glob("EXP*_T*K.txt")):
            if "(copy)" in f.name or _is_misfiled(f):
                continue
            df = _parse_ch4_water_file(f, T_K)
            if df is not None:
                dfs.append(df)
                file_count += 1
    if not dfs:
        raise RuntimeError(f"No CH4-H2O .txt files parsed under {CH4_WATER_DIR}")
    out = pd.concat(dfs, ignore_index=True)
    out.to_parquet(out_path)
    n_x = int(out["x_CH4_aq_exp"].notna().sum())
    n_y = int(out["y_H2O_gas_exp"].notna().sum())
    print(f"  CH4-H2O: {file_count} files -> {len(out)} rows "
          f"({n_x} x_CH4 / {n_y} y_H2O)")
    print(f"  T range:  {int(out['T_K'].min())} ... {int(out['T_K'].max())} K  "
          f"({out['T_K'].nunique()} isotherms)")
    print(f"  P range:  {out['P_bar'].min():.2f} ... {out['P_bar'].max():.1f} bar")
    print(f"  authors:  {out['author'].nunique()}")
    print(f"  -> {out_path} ({out_path.stat().st_size/1024:.1f} kB)")
    return out_path


def build_co2_water(out_path: Path = None) -> Path:
    out_path = out_path or (DATA_DIR / "binaries_co2_water.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dfs = []
    file_count = 0
    for Tdir in sorted(CO2_WATER_DIR.iterdir()):
        if not Tdir.is_dir():
            continue
        # Skip e.g. "T623K (copy)" or other oddities.
        if " " in Tdir.name:
            continue
        T_K = _parse_t_dirname(Tdir.name, prefix="T")
        if T_K is None:
            continue
        for f in sorted(Tdir.glob("EXP*_T*K.txt")):
            # Skip any "(copy)" variants.
            if "(copy)" in f.name or _is_misfiled(f):
                continue
            df = _parse_co2_water_file(f, T_K)
            if df is not None:
                dfs.append(df)
                file_count += 1
    if not dfs:
        raise RuntimeError(f"No CO2-H2O .txt files parsed under {CO2_WATER_DIR}")
    out = pd.concat(dfs, ignore_index=True)
    out.to_parquet(out_path)
    n_x = int(out["x_CO2_aq_exp"].notna().sum())
    n_y = int(out["y_H2O_gas_exp"].notna().sum())
    print(f"  CO2-H2O: {file_count} files -> {len(out)} rows "
          f"({n_x} x_CO2 / {n_y} y_H2O)")
    print(f"  T range:  {int(out['T_K'].min())} ... {int(out['T_K'].max())} K  "
          f"({out['T_K'].nunique()} isotherms)")
    print(f"  P range:  {out['P_bar'].min():.2f} ... {out['P_bar'].max():.1f} bar")
    print(f"  authors:  {out['author'].nunique()}")
    print(f"  -> {out_path} ({out_path.stat().st_size/1024:.1f} kB)")
    return out_path


def main() -> int:
    print("Building binaries_co2_water.parquet ...")
    build_co2_water()
    print()
    print("Building binaries_ch4_water.parquet ...")
    build_ch4_water()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
