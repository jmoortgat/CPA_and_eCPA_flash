#!/usr/bin/env python3
"""Build consolidated experimental-data parquets from F. M. Coelho's raw .txt files.

Run **once** (and again only when new experimental data arrives):

    python3 scripts/build_experimental_parquets.py

Reads the per-T text files under
``$ECPA_SOURCE_MATERIALS/EoS/CO2/eCPA/MIXED_SALTS/T{T}K/EXP_{salt}{k}_T{T}K.txt``
and produces clean parquet files under ``code/data/``:

    data/co2_brines.parquet     -- CO2 in NaCl/KCl/CaCl/MgCl/NaSO brines

(Subsequent iterations of this script will build CH4 brines, ternaries,
brine density, etc. -- one consolidated parquet per data type.)

The two column conventions in F. M. Coelho's tree (xc_W mole fraction vs mc
molality) are detected from the header line and normalized to xc_W
(salt-free CO2 mole fraction) in the output.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE / "scripts"))
from transcription_meta import parse_source_line, true_temperature

DATA_DIR = _CODE / "data"


def _source_root() -> Path:
    """Locate the tree of original transcribed data files.

    These files are not distributed with the repository; the parquets they
    were built into are (``data/``). Set ``ECPA_SOURCE_MATERIALS`` to the
    tree's root to rebuild them.
    """
    import os
    rel = Path("EoS") / "CO2" / "eCPA" / "MIXED_SALTS"
    return Path(os.environ.get("ECPA_SOURCE_MATERIALS", "source_materials")) / rel


EOS_CO2_MIXED = _source_root()

MW = 0.018  # kg/mol H2O
ZI = np.array([+1, -1, +1, +2, +2, -2], dtype=float)
SALT_FOLDERS = ("NaCl", "KCl", "CaCl", "MgCl", "NaSO", "MgSO")

# Every isotherm in the tree is read. This was previously restricted to
# (323, 373, 423) K -- an undocumented narrowing, since the module docstring
# only ever said it reads "the per-T text files". The tree holds 22 isotherms
# from 283 to 473 K; those three carried 716 of its 1,925 transcribed rows,
# so five sixths of the isotherms and roughly two thirds of the data were
# never loaded. Widened 2026-09-18.
#
# LEGACY_T reproduces the old three-isotherm selection for regression
# comparison: build_co2_brines(temperatures=LEGACY_T) rebuilds the 511-point
# set that Papers II and III were fitted and validated against.
LEGACY_T = (323, 373, 423)


def _isotherm_dirs(temperatures=None):
    """Isotherm directories to read: all of them, or a named subset."""
    if temperatures is None:
        return sorted(
            (p for p in EOS_CO2_MIXED.glob("T*K") if p.is_dir()),
            key=lambda p: int(p.name.strip("TK") or 0))
    return [EOS_CO2_MIXED / f"T{T}K" for T in temperatures
            if (EOS_CO2_MIXED / f"T{T}K").is_dir()]


def _parse_one_co2_brine_file(path: Path, salt: str,
                              dropped: Optional[dict] = None
                              ) -> Optional[pd.DataFrame]:
    """Parse one EXP_{salt}{k}_T{T}K.txt into a DataFrame.

    Skips rows that fail charge-balance or have malformed numerics. Returns
    ``None`` if the file has no usable rows.

    ``dropped`` is an optional ``reason -> line count`` tally the caller
    passes in so the discarded lines stay visible. Reporting only the
    survivors cannot distinguish "the isotherm set grew" from "a shifted or
    mistyped ion column just removed forty points".
    """
    def _drop(reason: str) -> None:
        if dropped is not None:
            dropped[reason] = dropped.get(reason, 0) + 1

    text = path.read_text().splitlines()
    if len(text) < 3:
        return None
    author, declared_T, _ = parse_source_line(text[0])
    header = text[1].lower()
    toks = header.split()
    col3_token = toks[4] if len(toks) >= 6 else ""
    # Convention detection: header reads "P [bar] ms [mol/kg] xc_W [...]" or "P [bar] ms [mol/kg] mc [...]"
    col3_is_molality = col3_token in ("mc", "m_co2", "m_c")

    rows = []
    for ln in text[2:]:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) < 9:
            _drop("short_row")
            continue
        try:
            P = float(parts[0])
            ms = float(parts[1])
            v3 = float(parts[2])
            ions = [float(parts[3 + k]) for k in range(6)]
        except ValueError:
            _drop("non_numeric")
            continue
        ion_arr = np.array(ions, dtype=float)
        # Quick charge-balance sanity check.
        zsum = float(np.sum(ion_arr * ZI))
        if abs(zsum) > 1e-6 * max(1.0, float(np.sum(np.abs(ion_arr)))):
            _drop("charge_balance")
            continue
        if col3_is_molality:
            xcW = v3 * MW / (1.0 + v3 * MW)
        else:
            xcW = v3
        rows.append({
            "salt": salt,
            "T_K": true_temperature(path, declared_T),
            "P_bar": P,
            "ms_mol_kg": ms,
            "xc_W": xcW,
            "m_Na": ion_arr[0],
            "m_Cl": ion_arr[1],
            "m_K":  ion_arr[2],
            "m_Ca": ion_arr[3],
            "m_Mg": ion_arr[4],
            "m_SO4": ion_arr[5],
            "author": author,
            "source_file": path.name,
            "col3_orig_units": "mc" if col3_is_molality else "xc_W",
        })
    if not rows:
        return None
    return pd.DataFrame(rows)


def build_co2_brines(out_path: Path = None, temperatures=None) -> Path:
    out_path = out_path or (DATA_DIR / "co2_brines.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dfs = []
    file_count = 0
    # Rows that fail a check are discarded, so the survivor counts below move
    # for two indistinguishable reasons: the transcribed set changed, or a
    # source file stopped parsing. The tally separates them -- and a file that
    # yields nothing at all (a whole literature source lost) is flagged.
    dropped: dict = defaultdict(int)
    empty_files = []
    for Tdir in _isotherm_dirs(temperatures):
        for salt in SALT_FOLDERS:
            for f in sorted(Tdir.glob(f"EXP_{salt}*_T*K*.txt")):
                # "_X<author>" files were previously skipped as "alternative
                # renderings already covered by the canonical file". Checked
                # 2026-09-18 against the built parquets: that is false for
                # every one of them. The suffix marks a SECOND author at a
                # (salt, T) slot whose number was already taken -- e.g.
                # MIXED_SALTS/T308K holds EXP_MIX2_T308K.txt (Tong 2013) and
                # EXP_MIX2_T308K_Xportier.txt (Portier 2005). None of the 29
                # such files had any row in the parquets at its own
                # temperature, so they are read here.
                #
                # "_ORIGINAL" is a genuine supersession and stays out: the
                # T333K pair differs only in the Li 2004 Na molality
                # (1.267 -> 1.405 in the canonical file), i.e. the canonical
                # one carries a correction.
                if "ORIGINAL" in f.stem:
                    continue
                before = sum(dropped.values())
                df = _parse_one_co2_brine_file(f, salt, dropped)
                if df is not None:
                    dfs.append(df)
                    file_count += 1
                else:
                    empty_files.append((f.relative_to(EOS_CO2_MIXED),
                                        sum(dropped.values()) - before))
    if not dfs:
        raise RuntimeError(f"No CO2-brine .txt files parsed under {EOS_CO2_MIXED}")
    out = pd.concat(dfs, ignore_index=True)
    out.to_parquet(out_path)
    by_salt = out.groupby("salt").size()
    by_T = out.groupby("T_K").size()
    n_dropped = sum(dropped.values())
    print(f"  {file_count} files parsed -> {len(out)} rows")
    if n_dropped:
        print(f"  {n_dropped} data lines dropped: {dict(dropped)}")
    for rel, n in empty_files:
        print(f"  WARNING: no usable row in {rel} ({n} lines dropped)",
              file=sys.stderr)
    print(f"  per salt: {dict(by_salt)}")
    print(f"  per T:    {dict(by_T)}")
    print(f"  -> {out_path} ({out_path.stat().st_size/1024:.1f} kB)")
    return out_path


def main() -> int:
    print("Building co2_brines.parquet ...")
    build_co2_brines()
    # Future: build_ch4_brines(), build_ccw_ternary(), build_brine_density(), ...
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
