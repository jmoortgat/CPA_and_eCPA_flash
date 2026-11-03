#!/usr/bin/env python3
"""Build consolidated mixed-salt CO2 brine parquet from F. M. Coelho's EXP_MIX*.txt.

Run **once** (and again only when new mixed-brine data arrives):

    python3 scripts/build_co2_mixed_brines.py

Reads ``$ECPA_SOURCE_MATERIALS/EoS/CO2/eCPA/MIXED_SALTS/T{T}K/EXP_MIX{N}_T{T}K.txt``
for T in {323, 373, 423} K and produces ``code/data/co2_mixed_brines.parquet``.

Each EXP_MIX{N} file is keyed by a single author (e.g. "LIU(2011)") and
typically contains multiple "recipes" (distinct ion-molality vectors). One
recipe == one ``mix_id`` (e.g. ``LIU_2011_B1``). Recipes are detected by
de-duplicating the 6-ion vector within an author cohort and numbered in the
order they first appear (``{AUTHOR}_B{N}``).

Schema
------
T_K, P_bar, xc_W, m_Na, m_Cl, m_K, m_Ca, m_Mg, m_SO4, mix_id, author, source_file
where ``xc_W`` is the salt-free CO2 mole fraction (col-3 in molality is
converted via ``xc = m * Mw / (1 + m * Mw)``).
"""
from __future__ import annotations
import os

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
EOS_CO2_MIXED = (
    Path(os.environ.get("ECPA_SOURCE_MATERIALS", "source_materials")) / "EoS" / "CO2" / "eCPA" / "MIXED_SALTS"
)

MW = 0.018  # kg/mol H2O
ZI = np.array([+1, -1, +1, +2, +2, -2], dtype=float)

# We walk every T-directory that contains an EXP_MIX*.txt file rather than a
# fixed list -- mixed-brine recipes from a given author are scattered across
# many T values (e.g. LIU(2011) spans 308-328 K; WANG(2014) spans 318-348 K).
# Restricting to the {323, 373, 423} set used for the single-salt parquet
# would drop the LIU/WANG/LI/ELMAGHRABY cohorts entirely.


def _author_tag(author_raw: str) -> str:
    """Normalize 'LIU(2011)' -> 'LIU_2011' (matches the PNG-file convention)."""
    s = author_raw.strip()
    return (
        s.replace("(", "_")
         .replace(")", "")
         .replace(" ", "_")
    )


def _parse_one_mix_file(path: Path) -> Optional[pd.DataFrame]:
    """Parse one EXP_MIX{N}_T{T}K.txt into a DataFrame. ``mix_id`` is NOT yet
    assigned -- this happens after concatenation across all temperatures
    because the same recipe may appear at multiple T values for one author.
    """
    text = path.read_text().splitlines()
    if len(text) < 3:
        return None
    author_raw, declared_T, _ = parse_source_line(text[0])
    header = text[1].lower()
    toks = header.split()
    col3_token = toks[4] if len(toks) >= 6 else ""
    col3_is_molality = col3_token in ("mc", "m_co2", "m_c")

    rows = []
    for ln in text[2:]:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) < 9:
            continue
        try:
            P = float(parts[0])
            # parts[1] is "MIX" (literal) for these files; skip
            v3 = float(parts[2])
            ions = [float(parts[3 + k]) for k in range(6)]
        except ValueError:
            continue
        ion_arr = np.array(ions, dtype=float)
        zsum = float(np.sum(ion_arr * ZI))
        # Mixed-brine field recipes round each ion to ~0.001 mol/kg, so the
        # net charge can drift up to ~0.01 mol/kg in absolute terms. We use
        # a 5% relative tolerance (not the 1e-3 used for the cleaner
        # single-salt files) -- this keeps real recipes like WANG(2014)
        # block 2 (Na=0.246, Cl=0.269, Ca=0.015, Mg=0.003, SO4=0.006).
        if abs(zsum) > 0.05 * max(1.0, float(np.sum(np.abs(ion_arr)))):
            continue
        if col3_is_molality:
            xcW = v3 * MW / (1.0 + v3 * MW)
        else:
            xcW = v3
        rows.append({
            "T_K": true_temperature(path, declared_T),
            "P_bar": P,
            "xc_W": xcW,
            "m_Na":  ion_arr[0],
            "m_Cl":  ion_arr[1],
            "m_K":   ion_arr[2],
            "m_Ca":  ion_arr[3],
            "m_Mg":  ion_arr[4],
            "m_SO4": ion_arr[5],
            "author":      _author_tag(author_raw),
            "author_raw":  author_raw,
            "source_file": path.name,
            "col3_orig_units": "mc" if col3_is_molality else "xc_W",
        })
    if not rows:
        return None
    return pd.DataFrame(rows)


def _assign_mix_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Assign ``mix_id = "{AUTHOR}_B{N}"`` per unique ion-composition within
    each author. Numbering follows first appearance (matches the prior-art
    ``LIU_2011_B1.png`` convention).
    """
    ion_cols = ["m_Na", "m_Cl", "m_K", "m_Ca", "m_Mg", "m_SO4"]
    df = df.copy()
    mix_ids = []
    # author -> {ion-key: int}
    seen: dict[str, dict[tuple, int]] = defaultdict(dict)
    for _, r in df.iterrows():
        au = r["author"]
        key = tuple(round(float(r[c]), 6) for c in ion_cols)
        if key not in seen[au]:
            seen[au][key] = len(seen[au]) + 1
        mix_ids.append(f"{au}_B{seen[au][key]}")
    df["mix_id"] = mix_ids
    return df


def build_co2_mixed_brines(out_path: Path = None) -> Path:
    out_path = out_path or (DATA_DIR / "co2_mixed_brines.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dfs = []
    file_count = 0
    # Walk every T-directory at the top of MIXED_SALTS. Skip the OPT/
    # subtree (that's a re-packaging of the same data by the original
    # plotting script -- would double-count if included).
    for Tdir in sorted(EOS_CO2_MIXED.glob("T*K")):
        if not Tdir.is_dir():
            continue
        for f in sorted(Tdir.glob("EXP_MIX*_T*K*.txt")):
            # Skip _ORIGINAL backup duplicates: the T333K pair differs only in
            # the Li 2004 Na molality (1.267 -> 1.405), so the canonical file
            # is the corrected one.
            if "ORIGINAL" in f.stem:
                continue
            # "_X<author>" files are NOT duplicates -- see the note in
            # build_experimental_parquets.py. EXP_MIX2_T308K_Xportier.txt is
            # Portier & Rochelle 2005 (Chem. Geol. 217:187-199) Table 2 at
            # 35 and 37 C, 16 points that appear nowhere else; the canonical
            # EXP_MIX2_T308K.txt at that slot is Tong 2013.
            df = _parse_one_mix_file(f)
            if df is not None and len(df):
                dfs.append(df)
                file_count += 1
    if not dfs:
        raise RuntimeError(f"No EXP_MIX*.txt files found under {EOS_CO2_MIXED}")
    out = pd.concat(dfs, ignore_index=True)
    out = _assign_mix_ids(out)
    # Reorder columns to match the documented schema.
    cols = ["T_K", "P_bar", "xc_W",
            "m_Na", "m_Cl", "m_K", "m_Ca", "m_Mg", "m_SO4",
            "mix_id", "author", "author_raw",
            "source_file", "col3_orig_units"]
    out = out[cols]
    out.to_parquet(out_path)
    print(f"  {file_count} EXP_MIX files parsed -> {len(out)} rows")
    print(f"  unique mix_ids: {out['mix_id'].nunique()}")
    print(f"  per author: {dict(out.groupby('author').size())}")
    print(f"  per T:      {dict(out.groupby('T_K').size())}")
    print(f"  -> {out_path} ({out_path.stat().st_size/1024:.1f} kB)")
    return out_path


def main() -> int:
    print("Building co2_mixed_brines.parquet ...")
    build_co2_mixed_brines()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
