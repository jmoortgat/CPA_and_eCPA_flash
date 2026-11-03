"""Serve the figure/fit scripts from GasBrineBench instead of local parquets.

Until 2026-09-18 every figure and every fit read
``code/data/<kind>.parquet``, built by ``scripts/build_*.py`` straight from
``source_materials``. GasBrineBench now holds the same measurements, curated,
quality-coded and citable -- and holds a good deal more of them, because the
release recovered rows the local builders never read.

This module reshapes GasBrineBench's long form (one measurement per row,
carrying ``gas``/``property``/``value``) back into the wide per-kind schema
the scripts already expect, so ``load_experimental()`` keeps its contract and
none of its 21 callers change.

Why the numbers move
--------------------
This is not a like-for-like swap. Relative to the local parquets,
GasBrineBench v1.1.0:

* corrects rows that were already there -- the Duffy (1961) ion vectors (the
  divalent molality sat in the potassium column), nine isotherms whose
  temperature came from a directory name that could not represent 351.65 K
  or tell 344.26 K from 343 K, and 16 quality codes demoted R -> T once the
  false cross-source corroboration those roundings created was removed;
* adds about 1,400 salt-free binary rows that anchor every salting-out
  comparison, a Susak (1980) block above 573 K, every isotherm directory
  rather than a hard-coded three, and one file no build had ever read.

So every AARE recomputed through this module will differ from the published
one. That is the point of the switch, and the refit is where the difference
gets absorbed.

What is NOT served from here
----------------------------
``ch4_henry`` stays on its local parquet and must not be used as a fit or
validation target. Henry constants are a derived quantity resting on Henry's
law, whose limited validity is the reason for building an equation of state
in the first place; scoring the model against them would be circular. It
remains a figure input only.

``ccb_ternary`` is empty and stays empty: no ternary measurement in brine
exists (see the GasBrineBench SCHEMA.md note on the ternary family).

Selecting the source
--------------------
``ECPA_EXP_SOURCE=legacy`` falls back to the local parquets, for A/B
comparison. ``ECPA_GBB`` overrides where GasBrineBench is found.
"""
from __future__ import annotations

import os
from pathlib import Path

from ms_ecpa import gbb as _gbb

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parents[1]

#: Ion columns, in the frozen order both projects use.
IONS = ["m_Na", "m_Cl", "m_K", "m_Ca", "m_Mg", "m_SO4"]

#: GasBrineBench salt-system labels -> the short names the figure scripts use.
_SALT_NAME = {
    "Na-Cl": "NaCl", "Cl-K": "KCl", "Cl-Ca": "CaCl",
    "Cl-Mg": "MgCl", "Na-SO4": "NaSO", "Mg-SO4": "MgSO",
}


def gbb_root() -> Path:
    """Where GasBrineBench lives -- delegated to :mod:`ms_ecpa.gbb`.

    This used to compute ``_CODE.parents[1] / "GasBrineBench"``, which counts
    directory levels and, in the standard checkout, pointed one level above
    the real location. It then failed silently: :func:`available` returned
    False and every figure and table fell back to the legacy parquets in
    ``data/`` while the fitting set was read from GasBrineBench, so the two
    were scored against different measurements with nothing saying so.

    There is now one resolver for the whole tree, and it fetches the database
    when it is absent instead of substituting whatever is lying around.
    """
    return _gbb.ensure()


def available() -> bool:
    """True once a usable checkout is in hand; fetches one if it has to."""
    try:
        return (gbb_root() / "data" / "solubility.csv").exists()
    except Exception:
        return False


def describe_source() -> str:
    """One line naming the experimental database actually in use."""
    if os.environ.get("ECPA_EXP_SOURCE", "").lower() == "legacy":
        return "experimental data: legacy parquets (ECPA_EXP_SOURCE=legacy)"
    try:
        return "experimental data: " + _gbb.describe()
    except Exception as exc:
        return f"experimental data: legacy parquets in data/ ({exc})"


def _load(family: str) -> pd.DataFrame:
    # exclude_tags left at the package default, which drops the 144
    # `lle-regime` propane rows. The fitting set deliberately passes None and
    # applies its own documented rules instead; see ms_ecpa.gbb.load.
    return _gbb.load(family)


def _salt_label(df: pd.DataFrame) -> pd.Series:
    """Short salt name per row, from the ion columns."""
    present = {}
    for c in IONS:
        present[c] = df[c].fillna(0.0) > 0
    def name(i):
        ions = [c[2:] for c in IONS if present[c].iloc[i]]
        if not ions:
            return "none"
        return _SALT_NAME.get("-".join(ions), "-".join(ions))
    return pd.Series([name(i) for i in range(len(df))], index=df.index)


def _salt_molality(df: pd.DataFrame) -> pd.Series:
    """Moles of SALT per kg water -- not total ions.

    The figure scripts' ``ms_mol_kg`` / ``m_salt_mol_kg`` is the salt
    molality: 6 m NaCl is 6, not the 12 mol of ions it dissociates into.
    For every salt in this database the salt molality is the SMALLEST
    non-zero ion molality, because the ion carrying the higher
    stoichiometric coefficient is the one that is multiplied up:

        NaCl   Na 6,  Cl 6   -> 6     KCl    K 1,  Cl 1   -> 1
        CaCl2  Ca 1,  Cl 2   -> 1     MgCl2  Mg 1, Cl 2   -> 1
        Na2SO4 Na 2,  SO4 1  -> 1     MgSO4  Mg 1, SO4 1  -> 1

    Checked against the local parquets: 1831 of 1832 salt-bearing
    co2_brines rows and all 905 brine_density rows agree. The one
    exception is a legacy inconsistency, not a rule violation -- its ion
    vector says 1.5 m NaCl while its ms_mol_kg says 2.0 -- and the ion
    vector is what GasBrineBench carries and what the model consumes.
    """
    m = df[IONS].fillna(0.0)
    return m.where(m > 0).min(axis=1).fillna(0.0)


def _salt_free(df: pd.DataFrame) -> pd.Series:
    return df[IONS].fillna(0.0).abs().sum(axis=1) == 0


def _single_salt(df: pd.DataFrame) -> pd.Series:
    """True where exactly one cation/anion pair is present."""
    n = (df[IONS].fillna(0.0) > 0).sum(axis=1)
    return n.isin([1, 2])


# --------------------------------------------------------------------------
# per-kind adapters
# --------------------------------------------------------------------------

def _solubility(gas: str, prop: str) -> pd.DataFrame:
    d = _load("solubility")
    d = d[(d["gas"] == gas) & (d["property"] == prop)].copy()
    return d


def co2_brines() -> pd.DataFrame:
    d = _solubility("co2", "xc_saltfree")
    d = d[~_salt_free(d) & _single_salt(d)].copy()
    out = pd.DataFrame({
        "salt": _salt_label(d),
        "T_K": d["T_K"].astype(float),
        "P_bar": d["P_bar"].astype(float),
        "ms_mol_kg": _salt_molality(d),
        "xc_W": d["value"].astype(float),
    })
    for c in IONS:
        out[c] = d[c].fillna(0.0).astype(float).values
    out["author"] = d["source"].values
    out["source_file"] = "GasBrineBench:" + d["dataset_id"].astype(str).values
    out["col3_orig_units"] = "xc_saltfree"
    return out.reset_index(drop=True)


def ch4_brines() -> pd.DataFrame:
    d = _solubility("ch4", "solubility_molality")
    d = d[~_salt_free(d) & _single_salt(d)].copy()
    out = pd.DataFrame({
        "salt": _salt_label(d),
        "T_K": d["T_K"].astype(float),
        "P_bar": d["P_bar"].astype(float),
        "ms_mol_kg": _salt_molality(d),
        "mh_W": d["value"].astype(float),
    })
    for c in IONS:
        out[c] = d[c].fillna(0.0).astype(float).values
    out["author"] = d["source"].values
    out["source_file"] = "GasBrineBench:" + d["dataset_id"].astype(str).values
    out["col3_orig_units"] = "mol_kg"
    return out.reset_index(drop=True)


def co2_mixed_brines() -> pd.DataFrame:
    d = _solubility("co2", "xc_saltfree")
    d = d[~_salt_free(d) & ~_single_salt(d)].copy()
    out = pd.DataFrame({
        "T_K": d["T_K"].astype(float),
        "P_bar": d["P_bar"].astype(float),
        "xc_W": d["value"].astype(float),
    })
    for c in IONS:
        out[c] = d[c].fillna(0.0).astype(float).values
    out["mix_id"] = d["dataset_id"].astype(str).values
    out["author"] = d["source"].values
    out["author_raw"] = d["source"].values
    out["source_file"] = "GasBrineBench:" + d["dataset_id"].astype(str).values
    out["col3_orig_units"] = "xc_saltfree"
    return out.reset_index(drop=True)


def _binary(gas: str, xcol: str) -> pd.DataFrame:
    """Salt-free x(gas) and y(H2O), joined on the measured state.

    The two quantities are separate rows in GasBrineBench -- they are separate
    measurements -- so they are joined on (source, T, P). A state where only
    one was measured keeps NaN in the other column, exactly as the local
    parquets did.
    """
    sol = _solubility(gas, "xc_saltfree")
    sol = sol[_salt_free(sol)]
    yh = _load("y_h2o")
    yh = yh[(yh["gas"] == gas) & _salt_free(yh)]

    key = ["source", "T_K", "P_bar"]
    a = (sol[key + ["value"]].rename(columns={"value": xcol})
         .drop_duplicates(subset=key))
    b = (yh[key + ["value"]].rename(columns={"value": "y_H2O_gas_exp"})
         .drop_duplicates(subset=key))
    m = a.merge(b, on=key, how="outer")
    out = pd.DataFrame({
        "T_K": m["T_K"].astype(float),
        "P_bar": m["P_bar"].astype(float),
        xcol: m[xcol].astype(float),
        "y_H2O_gas_exp": m["y_H2O_gas_exp"].astype(float),
        "author": m["source"].values,
        "source_file": "GasBrineBench",
    })
    return out.sort_values(["T_K", "P_bar"]).reset_index(drop=True)


def binaries_co2_water() -> pd.DataFrame:
    out = _binary("co2", "x_CO2_aq_exp")
    out["col3_orig"] = "GasBrineBench"
    return out


def binaries_ch4_water() -> pd.DataFrame:
    return _binary("ch4", "x_CH4_aq_exp")


def brine_density() -> pd.DataFrame:
    d = _load("rho")
    out = pd.DataFrame({
        "salt": _salt_label(d),
        "T_K": d["T_K"].astype(float),
        "P_bar": d["P_bar"].astype(float),
        "m_salt_mol_kg": _salt_molality(d),
    })
    for c in IONS:
        out[c] = d[c].fillna(0.0).astype(float).values
    out["rho_exp_kg_m3"] = d["value"].astype(float).values
    out["author"] = d["source"].values
    out["source_file"] = "GasBrineBench:" + d["dataset_id"].astype(str).values
    return out.reset_index(drop=True)


def ccw_ternary() -> pd.DataFrame:
    """Pivot the ternary family back to one row per measured state."""
    d = _load("ternary")
    key = ["source", "T_K", "P_bar", "y_co2_dry"]
    piv = (d.pivot_table(index=key, columns=["gas", "property"],
                         values="value", aggfunc="first")
             .reset_index())
    # reset_index leaves the key columns as ("source", "") tuples alongside the
    # real ("ch4", "xc_saltfree") ones; join then strip the empty tail.
    piv.columns = [c if isinstance(c, str) else "|".join(x for x in c if x)
                   for c in piv.columns]
    def col(name):
        return piv[name].astype(float) if name in piv.columns else np.nan
    out = pd.DataFrame({
        "T_K": piv["T_K"].astype(float),
        "P_bar": piv["P_bar"].astype(float),
        "Yc_gas": piv["y_co2_dry"].astype(float),
        "x_CH4_aq_exp": col("ch4|xc_saltfree"),
        "x_CO2_aq_exp": col("co2|xc_saltfree"),
        "y_H2O_gas_exp": col("co2-ch4|y_h2o"),
        "author": piv["source"].values,
        "source_file": "GasBrineBench:ternary",
    })
    return out.sort_values(["author", "T_K", "P_bar"]).reset_index(drop=True)


#: kind -> adapter. Kinds absent here fall through to the local parquet.
ADAPTERS = {
    "co2_brines": co2_brines,
    "ch4_brines": ch4_brines,
    "co2_mixed_brines": co2_mixed_brines,
    "binaries_co2_water": binaries_co2_water,
    "binaries_ch4_water": binaries_ch4_water,
    "brine_density": brine_density,
    "ccw_ternary": ccw_ternary,
    # ch4_henry: deliberately absent -- see the module docstring.
    # ccb_ternary: no experimental data exists; the empty parquet stands.
}
