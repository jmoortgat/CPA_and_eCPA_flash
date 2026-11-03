#!/usr/bin/env python3
"""Build the consolidated CH4-brine experimental-data parquet.

Run once (and again only when new data arrives):

    python3 scripts/build_ch4_brines.py

Reads per-T text files under
``$ECPA_SOURCE_MATERIALS/EoS/CH4/CH4-Brine/T{T}K/EXP_{salt}{k}_T{T}K.txt``
and produces ``code/data/ch4_brines.parquet``.

Three column conventions in F. M. Coelho's tree are detected from the
``[mol/kg]`` / ``[SALTincl]`` / ``[SALTfree]`` token in the header,
following ``OPT_ALLIONS.py:500``::

    [mol/kg]    -> mc directly in molality (mol CH4 / kg water)
    [SALTincl]  -> mole fraction including salt; mc = x/(1-x)*(1+Nion*ms*Mw)/Mw
    [SALTfree]  -> mole fraction salt-free;     mc = (1/Mw)*(x/(1-x))

The output stores CH4 molality (mol/kg water, salt-free) as ``mh_W`` so it
matches the ``mh`` field returned by :func:`ms_ecpa.solve_elv`.

Schema mirrors ``co2_brines.parquet`` except the solubility column is
``mh_W`` (CH4 molality) instead of ``xc_W`` (CO2 mole fraction).
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
CH4_BRINE = (
    Path(os.environ.get("ECPA_SOURCE_MATERIALS", "source_materials")) / "EoS" / "CH4" / "CH4-Brine"
)

Mw = 0.018  # kg/mol H2O
ZI = np.array([+1, -1, +1, +2, +2, -2], dtype=float)

# Five baseline salts (MIX excluded -- mixed-salt brines go to a future ternary builder).
SALT_FOLDERS = ("NaCl", "KCl", "CaCl", "MgCl", "NaSO", "MgSO")

# Nion (cation+anion stoichiometric coefficients summed) for each salt;
# used by the [SALTincl] conversion path.
NION_FROM_SALT = {
    "NaCl": 2,  # Na+ + Cl-
    "KCl":  2,  # K+ + Cl-
    "CaCl": 3,  # Ca2+ + 2 Cl-
    "MgCl": 3,  # Mg2+ + 2 Cl-
    "NaSO": 3,  # 2 Na+ + SO4^2-
    "MgSO": 2,  # Mg2+ + SO4^2-
}


def _detect_units(header: str) -> str:
    """Return 'mol_kg', 'salt_incl', or 'salt_free' (or '' if unknown)."""
    h = header.lower()
    if "[mol/kg]" in h.split("ms")[-1]:
        # both ms and mc carry [mol/kg]; mc is always the 3rd numeric column;
        # treat it as molality if 'mc' appears in the header.
        if " mc " in h or h.strip().endswith("mc [mol/kg]    na    cl    k    ca    mg    so4"):
            pass
        # The columns are P, ms, X; X has its own units token in slot 5..
        # We tokenise instead for robustness.
    toks = header.split()
    # Layout: P [bar]  ms [mol/kg]  X  [...]  Na Cl K Ca Mg SO4
    # We want the unit/tag token attached to X, which sits at index 5.
    if len(toks) < 9:
        return ""
    third_unit = toks[5]
    if third_unit == "[mol/kg]":
        return "mol_kg"
    if third_unit == "[SALTincl]":
        return "salt_incl"
    if third_unit == "[SALTfree]":
        return "salt_free"
    return ""


def _parse_one_ch4_brine_file(path: Path,
                              salt: str) -> tuple[Optional[pd.DataFrame], str]:
    """Parse one EXP_{salt}{k}_T{T}K.txt into a DataFrame.

    Output ``mh_W`` is always CH4 molality (mol CH4 per kg water) -- matches
    the ``mh`` field of :func:`ms_ecpa.solve_elv`.

    Rows that fail charge-balance or have non-numeric tokens are skipped.

    Returns ``(df, diagnostic)``. ``df`` is ``None`` if the file produced no
    usable rows, and ``diagnostic`` is a non-empty string whenever anything
    was dropped -- file or rows. The caller prints every diagnostic, so that
    a source file (or part of one) cannot leave the calibration set without
    showing up in the build output.
    """
    text = path.read_text().splitlines()
    if len(text) < 3:
        return None, f"only {len(text)} lines (need source line + header + data)"
    author, declared_T, _ = parse_source_line(text[0])
    header = text[1]
    units = _detect_units(header)
    if not units:
        # Unknown header layout: skip the file rather than mis-parse silently.
        # Reported by the caller -- an unread header shifts every ion column
        # and would otherwise drop the file with no trace in the row counts.
        toks = header.split()
        where = (repr(toks[5]) if len(toks) >= 9
                 else f"header has only {len(toks)} tokens")
        return None, f"unrecognised units token ({where})"

    Nion = NION_FROM_SALT[salt]

    rows = []
    n_data = 0   # non-comment data lines seen
    n_drop = 0   # of those, lines rejected below
    for ln in text[2:]:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        n_data += 1
        parts = ln.split()
        # Need at least 3 + 5 = 8 numeric tokens. Some files drop a trailing
        # zero on the SO4 column when no SO4 is present (e.g. KIEPE KCl
        # files at 313/353 K). Pad with zeros to 6 ion columns.
        if len(parts) < 8:
            n_drop += 1
            continue
        try:
            P = float(parts[0])
            ms_raw = parts[1]
            # MIX files use 'MIX' as the ms label -- handled by the MIX
            # parser, not this one. If it leaks in here, skip the row.
            ms = float(ms_raw)
            v3 = float(parts[2])
            n_ions_present = min(len(parts) - 3, 6)
            ions = [float(parts[3 + k]) for k in range(n_ions_present)]
            ions.extend([0.0] * (6 - n_ions_present))
        except ValueError:
            n_drop += 1
            continue
        ion_arr = np.array(ions, dtype=float)
        zsum = float(np.sum(ion_arr * ZI))
        if abs(zsum) > 1e-6 * max(1.0, float(np.sum(np.abs(ion_arr)))):
            n_drop += 1
            continue
        if units == "mol_kg":
            mh = v3
        elif units == "salt_incl":
            # x is mole fraction including salt. Convert to molality.
            # Follows OPT_ALLIONS.py:518: mc = x/(1-x) * (1+Nion*ms*Mw) / Mw
            if v3 >= 1.0:
                n_drop += 1
                continue
            mh = v3 / (1.0 - v3) * (1.0 + Nion * ms * Mw) / Mw
        elif units == "salt_free":
            # x is salt-free mole fraction. mc = (1/Mw)*(x/(1-x))
            if v3 >= 1.0:
                n_drop += 1
                continue
            mh = (1.0 / Mw) * (v3 / (1.0 - v3))
        else:
            n_drop += 1
            continue
        rows.append({
            "salt": salt,
            "T_K": true_temperature(path, declared_T),
            "P_bar": P,
            "ms_mol_kg": ms,
            "mh_W": mh,
            "m_Na": ion_arr[0],
            "m_Cl": ion_arr[1],
            "m_K":  ion_arr[2],
            "m_Ca": ion_arr[3],
            "m_Mg": ion_arr[4],
            "m_SO4": ion_arr[5],
            "author": author,
            "source_file": path.name,
            "col3_orig_units": units,
        })
    diag = ("" if n_drop == 0 else
            f"{n_drop} of {n_data} data rows rejected "
            "(charge balance, non-numeric, or x >= 1)")
    if not rows:
        return None, diag or "no data rows"
    return pd.DataFrame(rows), diag


def build_ch4_brines(out_path: Path = None) -> Path:
    """Walk every ``T{T}K/`` subdir under ``CH4-Brine/`` and consolidate."""
    out_path = out_path or (DATA_DIR / "ch4_brines.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dfs = []
    file_count = 0
    candidate_count = 0
    rejected: list[tuple[Path, str]] = []   # file dropped whole
    partial: list[tuple[Path, str]] = []    # file kept, some rows dropped
    # Find all per-T directories (e.g. 298K, 303K, 313K, ...).
    #
    # 303K was "303K_X" until 2026-09-18, excluded because its divalent rows
    # carried the cation molality one column to the left: EXP_CaCl1_T303K.txt
    # read Na=0, Cl=2.8, K=1.4 for a 1.4 m CaCl2 brine, which fails charge
    # balance at 1.4 against 2.8 and lands exactly with the 1.4 in Ca. Duffy
    # (1961) settles it -- the system studied is CH4-H2O-NaCl-CaCl2 and
    # contains no potassium at all -- so the 28 affected rows were moved into
    # the Ca column and the directory renamed. The "_X" suffix is still
    # honoured here in case another such directory appears.
    Tdirs = sorted(p for p in CH4_BRINE.iterdir()
                   if p.is_dir() and p.name.endswith("K") and not p.name.endswith("_X"))
    for Tdir in Tdirs:
        for salt in SALT_FOLDERS:
            # "_X<author>" FILES are read: they are a second author at a slot
            # whose number was taken, not an alternative rendering. See the
            # note in build_experimental_parquets.py.
            for f in sorted(Tdir.glob(f"EXP_{salt}*_T*K*.txt")):
                if "ORIGINAL" in f.stem:
                    continue
                candidate_count += 1
                df, diag = _parse_one_ch4_brine_file(f, salt)
                if df is None:
                    rejected.append((f, diag))
                    continue
                if diag:
                    partial.append((f, diag))
                dfs.append(df)
                file_count += 1
    if not dfs:
        raise RuntimeError(f"No CH4-brine .txt files parsed under {CH4_BRINE}")
    out = pd.concat(dfs, ignore_index=True)
    out.to_parquet(out_path)
    by_salt = out.groupby("salt").size()
    by_T = out.groupby("T_K").size()
    # Print the rejections before the totals: a file that never reaches the
    # parquet is otherwise invisible, since file_count counts successes only.
    for f, why in rejected:
        print(f"  REJECTED {f.parent.name}/{f.name}: {why}", file=sys.stderr)
    for f, why in partial:
        print(f"  PARTIAL  {f.parent.name}/{f.name}: {why}", file=sys.stderr)
    print(f"  {file_count} of {candidate_count} candidate files parsed "
          f"-> {len(out)} rows")
    if rejected:
        print(f"  {len(rejected)} file(s) rejected -- see REJECTED lines above")
    print(f"  per salt: {dict(by_salt)}")
    print(f"  per T:    {dict(by_T)}")
    print(f"  -> {out_path} ({out_path.stat().st_size/1024:.1f} kB)")
    return out_path


def main() -> int:
    print("Building ch4_brines.parquet ...")
    build_ch4_brines()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
