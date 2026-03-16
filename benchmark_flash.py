# --- Standard library ---
import os
import sys
import re
import time
import pickle
import warnings
import types
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Numerical / scientific ---
import numpy as np
import pandas as pd
from scipy.optimize import fsolve, brentq
from scipy.interpolate import NearestNDInterpolator

# --- Plotting ---
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.patches as mpatches

# --- Other third-party ---
from colour import Color

# --- Local modules ---
import importlib
import CPA2
importlib.reload(CPA2)

os.chdir('/Users/moortgat/Software/2026/eCPA_SALTbasis/Claude_code')
sys.path.insert(0, '.')

# === Cell 3 ===
# Use explicit complex Jacobian or default FD (which may actually be faster)
USE_COMPLEX_JAC = False   # set False to fall back to fsolve finite-difference Jacobian

# === Cell 5 ===
# Consolidate CO2/CPA_ELV_T###K.dat files into one Parquet file (saved in CO2/)

# --- paths (relative to your notebook, matching the existing code) ---
dat_dir = Path("CO2")                 # the folder your code uses: open("CO2/CPA_ELV_T{}K.dat")
out_path = dat_dir / "CPA_ELV_all.parquet"

# Explicit numeric column layout in each .dat row: P + 10 unknowns
_COLS = [
    "P_bar",
    "Z_W", "xw_W", "eps_r",
    "Z_C", "xw_C",
    "chiw_W", "chiw_C",
    "Ndchi1w_dNw", "Ndchi1w_dNc", "Vdchi1w_dV",
]

def read_cpa_elv_dat(path: Path) -> pd.DataFrame:
    # Infer T from filename like CPA_ELV_T288K.dat
    m = re.search(r"_T(\d+)K\.dat$", path.name)
    if not m:
        raise ValueError(f"Could not parse temperature from filename: {path.name}")
    T_K = int(m.group(1))

    # Skip the header line entirely (avoids the "P [bar]" split bug),
    # and read numeric data with explicit column names.
    df = pd.read_csv(
        path,
        sep=r"\s+",
        engine="python",
        skiprows=1,
        names=_COLS,
    )

    # Add temperature metadata
    df.insert(0, "T_K", T_K)

    return df


if out_path.exists():
    print(f"{out_path} already exists — skipping consolidation.")
else:
    # --- batch read ---
    paths = sorted(dat_dir.glob("CPA_ELV_T*K.dat"))
    if not paths:
        raise FileNotFoundError(f"No files matching {dat_dir/'CPA_ELV_T*K.dat'}")

    dfs = [read_cpa_elv_dat(p) for p in paths]
    all_df = pd.concat(dfs, ignore_index=True)

    # Helpful for fast lookup later
    all_df = all_df.sort_values(["T_K", "P_bar"]).reset_index(drop=True)

    # --- write parquet ---
    all_df.to_parquet(out_path, index=False)

    print(f"Wrote: {out_path}")
    print(f"Rows: {len(all_df):,} | Columns: {len(all_df.columns)}")
    print("Temperatures:", sorted(all_df["T_K"].unique())[:10],
          ("..." if all_df["T_K"].nunique() > 10 else ""))
    print("Columns:", list(all_df.columns))



# === Cell 7 ===
def load_cpa_guess_table(parquet_path="CO2/CPA_ELV_all.parquet"):
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"{parquet_path} not found.")

    df = pd.read_parquet(parquet_path)

    # Drop the stray header fragment column if present
    if "[bar]" in df.columns:
        df = df.drop(columns=["[bar]"])

    # Normalize names to what our solver expects
    rename_map = {
        "Z_W": "Zw",
        "Z_C": "Zc",
        "xw_W": "xw_W",   # keep
        "xw_C": "xw_C",   # keep
        "eps_r": "eps_r", # keep
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = [
        "T_K", "P_bar",
        "Zw", "xw_W", "eps_r", "Zc", "xw_C",
        "chiw_W", "chiw_C",
        "Ndchi1w_dNw", "Ndchi1w_dNc", "Vdchi1w_dV",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {parquet_path}: {missing}")

    df = df.sort_values(["T_K", "P_bar"]).reset_index(drop=True)

    groups = {}
    for T, sub in df.groupby("T_K"):
        groups[int(T)] = sub.sort_values("P_bar").reset_index(drop=True)

    temps = np.array(sorted(groups.keys()), dtype=float)
    return groups, temps


# === Cell 9 ===
def guess_from_table_nearest(T, P_bar, groups, temps):
    if len(temps) == 0:
        raise ValueError("No temperatures available in guess table.")

    T_closest = temps[np.argmin(np.abs(temps - T))]
    dfT = groups[int(T_closest)]

    Pvals = dfT["P_bar"].to_numpy(dtype=float)
    idx = int(np.argmin(np.abs(Pvals - P_bar)))
    row = dfT.iloc[idx]

    return np.array([
        row["Zw"],
        row["xw_W"],
        row["eps_r"],
        row["Zc"],
        row["xw_C"],
        row["chiw_W"],
        row["chiw_C"],
        row["Ndchi1w_dNw"],
        row["Ndchi1w_dNc"],
        row["Vdchi1w_dV"],
    ], dtype=np.float64)


# === Cell 11 ===
# Linearly interpolate initial guess instead of nearest
def guess_from_table(T, P_bar, groups, temps):
    if len(temps) == 0:
        raise ValueError("No temperatures available in guess table.")

    # --- Temperature interpolation weight ---
    idx_T = np.searchsorted(temps, T)
    idx_T = np.clip(idx_T, 1, len(temps) - 1)
    T_lo, T_hi = temps[idx_T - 1], temps[idx_T]
    wT = (T - T_lo) / (T_hi - T_lo) if T_hi != T_lo else 0.0

    def interp_at_T(T_key, P_bar):
        dfT = groups[int(T_key)]
        Pvals = dfT["P_bar"].to_numpy(dtype=float)
        idx_P = np.searchsorted(Pvals, P_bar)
        idx_P = np.clip(idx_P, 1, len(Pvals) - 1)
        P_lo, P_hi = Pvals[idx_P - 1], Pvals[idx_P]
        wP = (P_bar - P_lo) / (P_hi - P_lo) if P_hi != P_lo else 0.0
        row_lo = dfT.iloc[idx_P - 1]
        row_hi = dfT.iloc[idx_P]
        cols = ["Zw", "xw_W", "eps_r", "Zc", "xw_C",
                "chiw_W", "chiw_C", "Ndchi1w_dNw", "Ndchi1w_dNc", "Vdchi1w_dV"]
        v_lo = np.array([row_lo[c] for c in cols], dtype=np.float64)
        v_hi = np.array([row_hi[c] for c in cols], dtype=np.float64)
        return v_lo + wP * (v_hi - v_lo)

    guess_lo = interp_at_T(T_lo, P_bar)
    guess_hi = interp_at_T(T_hi, P_bar)
    return guess_lo + wT * (guess_hi - guess_lo)

# === Cell 13 ===
# One-time setup: load guess table into memory
CPA_GROUPS, CPA_TEMPS = load_cpa_guess_table("CO2/CPA_ELV_all.parquet")
print(f"Loaded table: {len(CPA_TEMPS)} temperatures, "
      f"T = {CPA_TEMPS[0]:.0f}–{CPA_TEMPS[-1]:.0f} K")


# === Cell 15 ===
def guess_table_fn(T, P_bar):
    # mimic your earlier integer-T snapping if you want:
    t = int(round(T))
    return guess_from_table(t, P_bar, CPA_GROUPS, CPA_TEMPS)


# === Cell 16 ===
def get_table_p_range(T, groups, temps):
    """Return (P_min, P_max) available in the guess table at this T."""
    T_closest = temps[np.argmin(np.abs(temps - T))]
    dfT = groups[int(T_closest)]
    return float(dfT["P_bar"].min()), float(dfT["P_bar"].max())

# === Cell 20 ===
####### CONSTANTS
R = 8.314                   # J/mol.K
Na = 6.02e23                # 1/mol
kb = 1.38e-23               # J/K
e = 1.6e-19                 # C
eps0 = 8.854e-12            # F/m
Mw = 0.018                  # kg/mol
Ms = 0.0585                 # kg/mol
Mc = 0.044                  # kg/mol
Tc1 = 647.29                # K
Pc1 = 22060000              # Pa
Tc4 = 304.4                 # K
Pc4 = 7380000               # Pa

# === Cell 22 ===
# Physical
b1 = 14.515e-6               # m³/mol
c11 = 0.6736                 
a01 = 1017.3*R*b1            # J.m³/mol²
a02 = 0 
b2 = 16.49e-6                # m³/mol
a03 = 0
b3 = 40.83e-6                # m³/mol
b4 = 27.2e-6                 # m³/mol
c14 = 0.7602                
a04 = 1551.2*R*b4            # J.m³/mol
Tref = 298.15                # K
Akij = -0.49206
Bkij = 2.10136
Ckij = -1.57135
ASij = 0.19173
BSij = -0.17299
CSij = -0.00909

# Association
epsW = 2003.25               # K (/kb)
bettaW = 69.2e-3
kappaW = bettaW*b1           # m³/mol

# Ions
Z2 = 1
Z3 = -1
Sg2 = 2.356e-10
Sg3 = 3.187e-10
Rb2 = 1.665e-10
Rb3 = 1.828e-10
Penelouxs = -53.5e-6
Uref1s = -223.5*R            # J/mol
Talfa1s = 340                # K
alfa1s = 1573                # K
Uref4s = 6056.13852
Talfa4s = 243.79352
alfa4s = 691.85326

# Permissivity
dip01 = 1.8546*3.335e-30     # C.m
pol1 = 1.6133e-40            # Cm²/J
pol2 = 2.221e-40             # Cm²/J
pol3 = 3.557e-40             # Cm²/J
pol4 = 2.6946e-40            # Cm²/J
GAMMA1 = 63.4715*np.pi/180   # rad
THETA1 = 94.7939*np.pi/180   # rad
zww = 4

# === Cell 24 ===
def make_params_from_globals(include_names=None, exclude_names=None):
    """
    Snapshot numeric constants from current module globals
    into a dict suitable for passing as `params` to ELV().

    Parameters
    ----------
    include_names : list[str] or None
        If provided, only include these variable names.
    exclude_names : list[str] or None
        If provided, exclude these variable names.

    Returns
    -------
    params : dict[str -> float64]
    """

    g = globals()
    params = {}

    for name, value in g.items():

        # Skip private/internal names
        if name.startswith("_"):
            continue

        # Optional inclusion filter
        if include_names is not None and name not in include_names:
            continue

        # Optional exclusion filter
        if exclude_names is not None and name in exclude_names:
            continue

        # Skip modules, functions, classes
        if isinstance(value, (types.ModuleType, types.FunctionType, type)):
            continue

        # Keep only scalar numeric values
        if isinstance(value, (int, float, np.floating, np.integer)):
            params[name] = np.float64(value)

        # Allow numpy scalar types
        elif isinstance(value, np.ndarray) and value.shape == ():
            params[name] = np.float64(value)

        # Skip arrays, lists, etc.
        else:
            continue

    return params


# === Cell 25 ===
params = make_params_from_globals()

# === Cell 28 ===
# ─────────────────────────────────────────────────────────────────────────────
# CO2–Water experimental VLE data: parse folder → Parquet + lookup helpers
#
# Dataset columns
# ───────────────
#   T_K         temperature [K]
#   P_bar       pressure [bar]
#   xc_W        CO2 mol-frac in the aqueous (water-rich) phase
#   yw_C        H2O mol-frac in the CO2-rich phase
#   rho_W       aqueous-phase density [kg/m³]  (sparse, 37 pts)
#   exp_id      experiment number within that T folder
#   reference   author/year string from the file's # header line
#   source_file original filename
#
# Missing measurements are NaN (originally 'X' or 'x' in the source files).
# The one '(copy)' file in T423K is excluded.
# Column header is parsed to handle both conventions:
#   yw_C  → H2O mol-frac in CO2-rich phase  (stored as-is)
#   yc_C  → CO2 mol-frac in CO2-rich phase  (converted: yw_C = 1 - yc_C)
#
# Usage
# ─────
#   exp_xc_W(323)   → DataFrame of CO2-solubility points at 323 K
#   exp_yw_C(323)   → DataFrame of H2O-in-CO2 points at 323 K
#   exp_at_T(323)   → all rows at 323 K (both quantities)
# ─────────────────────────────────────────────────────────────────────────────

import re
import numpy as np
import pandas as pd
from pathlib import Path

_DATA_DIR     = Path("EXP/CO2-WATER")
_PARQUET_PATH = Path("CO2/CO2_WATER_exp.parquet")


def _parse_co2water_dir(data_dir: Path) -> pd.DataFrame:
    """Read every EXP*.txt in the extracted CO2-WATER folder."""
    records = []

    for txt_path in sorted(data_dir.rglob("EXP*.txt")):
        if '(copy)' in txt_path.name.lower():
            continue

        m_T = re.search(r'T(\d+)K', str(txt_path))
        if not m_T:
            continue
        T_K = int(m_T.group(1))

        m_exp = re.search(r'EXP(\d+)_', txt_path.name)
        exp_id = int(m_exp.group(1)) if m_exp else 0

        lines = txt_path.read_text(encoding='utf-8', errors='replace').splitlines()

        # First '#' line → reference string
        reference, data_start = '', 0
        for i, line in enumerate(lines):
            if line.strip().startswith('#'):
                reference  = line.strip().lstrip('#').strip()
                data_start = i + 1
                break

        # Next non-empty non-comment line → column header: parse it
        col_xc_W  = 1       # default column indices
        col_yw_C  = 2
        col_rho   = 3
        yc_C_mode = False   # True when col 2 is CO2 mol-frac → invert to get H2O

        for i in range(data_start, len(lines)):
            line = lines[i].strip()
            if line and not line.startswith('#'):
                # join and re-split after removing the "P [bar]" prefix
                # so that bracketed units don't shift column indices
                cleaned = re.sub(r'\[.*?\]', '', line)   # drop [...] tokens
                headers = cleaned.lower().split()
                for j, h in enumerate(headers):
                    if h == 'xc_w':
                        col_xc_W = j
                    if h == 'yw_c':
                        col_yw_C  = j
                        yc_C_mode = False
                    if h == 'yc_c':
                        col_yw_C  = j
                        yc_C_mode = True
                    if 'rho' in h:
                        col_rho = j
                data_start = i + 1
                break

        def _v(s: str) -> float:
            """'X' / 'x' → NaN, else float."""
            return np.nan if s.lower() == 'x' else float(s)

        for line in lines[data_start:]:
            parts = line.strip().split()
            if len(parts) < 2 or parts[0].startswith('#'):
                continue
            try:
                raw_y = _v(parts[col_yw_C]) if len(parts) > col_yw_C else np.nan
                yw_C  = (1.0 - raw_y) if (yc_C_mode and np.isfinite(raw_y)) else raw_y

                records.append(dict(
                    T_K         = T_K,
                    P_bar       = float(parts[0]),
                    xc_W        = _v(parts[col_xc_W]) if len(parts) > col_xc_W else np.nan,
                    yw_C        = yw_C,
                    rho_W       = _v(parts[col_rho])  if len(parts) > col_rho  else np.nan,
                    exp_id      = exp_id,
                    reference   = reference,
                    source_file = txt_path.name,
                ))
            except (ValueError, IndexError):
                continue

    df = pd.DataFrame(records)
    return df.sort_values(['T_K', 'P_bar', 'exp_id']).reset_index(drop=True)


# ── Build or load Parquet ─────────────────────────────────────────────────────
_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)

if _PARQUET_PATH.exists():
    EXP_DF = pd.read_parquet(_PARQUET_PATH)
    print(f"Loaded  {_PARQUET_PATH}  ({len(EXP_DF)} rows)")
else:
    print(f"Parsing {_DATA_DIR} ...")
    EXP_DF = _parse_co2water_dir(_DATA_DIR)
    EXP_DF.to_parquet(_PARQUET_PATH, index=False)
    print(f"Saved   {_PARQUET_PATH}  ({len(EXP_DF)} rows)")

EXP_TEMPS = np.array(sorted(EXP_DF['T_K'].unique()), dtype=float)


# ── Lookup helpers ────────────────────────────────────────────────────────────

def exp_at_T(T_K: float, tol_K: float = 1.0) -> pd.DataFrame:
    """All experimental rows within ±tol_K of T_K."""
    return EXP_DF[np.abs(EXP_DF['T_K'] - T_K) <= tol_K].copy()


def exp_xc_W(T_K: float, tol_K: float = 1.0) -> pd.DataFrame:
    """CO2 solubility rows (xc_W not NaN) near T_K."""
    df = exp_at_T(T_K, tol_K)
    return df[df['xc_W'].notna()][['P_bar', 'xc_W', 'reference']].copy()


def exp_yw_C(T_K: float, tol_K: float = 1.0) -> pd.DataFrame:
    """H2O content in CO2-rich phase (yw_C not NaN) near T_K."""
    df = exp_at_T(T_K, tol_K)
    return df[df['yw_C'].notna()][['P_bar', 'yw_C', 'reference']].copy()


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\nDataset summary")
print(f"  Rows      : {len(EXP_DF)}")
print(f"  T range   : {int(EXP_DF['T_K'].min())}–{int(EXP_DF['T_K'].max())} K  "
      f"({EXP_DF['T_K'].nunique()} isotherms)")
print(f"  P range   : {EXP_DF['P_bar'].min():.1f}–{EXP_DF['P_bar'].max():.0f} bar")
print(f"  xc_W pts  : {EXP_DF['xc_W'].notna().sum()}")
print(f"  yw_C pts  : {EXP_DF['yw_C'].notna().sum()}")
print(f"  rho_W pts : {EXP_DF['rho_W'].notna().sum()}")
print(f"  Sources   : {EXP_DF['reference'].nunique()} references")
print()
print(EXP_DF.groupby('T_K').agg(
    n      = ('P_bar',     'count'),
    xc_W   = ('xc_W',     lambda x: x.notna().sum()),
    yw_C   = ('yw_C',     lambda x: x.notna().sum()),
    P_min  = ('P_bar',    'min'),
    P_max  = ('P_bar',    'max'),
    n_refs = ('reference','nunique'),
).to_string())

# === Cell 42 ===
def ELV(x0, T, P, ms, params=None):
    """
    Complex-safe version of ELV for use with complex-step Jacobian.
    Changes from original:
      - x0 cast to plain array (no forced float64) so complex x0 propagates
      - _denom_sym and _rel_err use .real for scaling denominators
      - return array has no forced dtype (infers complex128 when needed)
      - T, P, ms remain real throughout (they are never perturbed)
    """

    def _denom_sym(a, b, eps=1e-30):
        return max(abs(a.real), abs(b.real), eps)          # was: max(abs(a), abs(b), eps)

    def _rel_err(a, b, scale=1.0, eps=1e-30):
        denom = max(abs(float(scale)), abs(a.real), eps)   # was: max(abs(scale), abs(a), eps)
        return (a - b) / denom

    # x0 may be complex — do NOT force float64
    x0 = np.asarray(x0)                                    # was: np.asarray(x0, dtype=np.float64)
    T  = np.float64(T)
    P  = np.float64(P)
    ms = np.float64(ms)

    _saved = None
    if params is not None:
        _saved = {}
        g = globals()
        for k, v in params.items():
            if k in g:
                _saved[k] = g[k]
            g[k] = v

    try:
        Zw, x1w, epsr, Zc, x1c, chi1w, chi1c, Ndchi1WdNw, Ndchi1WdNc, Vdchi1WdV = x0

        # PARAMETERS
        k14 = (Akij*(T/Tc4)**2 + Bkij*(T/Tc4) + Ckij)
        S14 = (ASij*(T/Tc4)**2 + BSij*(T/Tc4) + CSij)
        U4s = Uref4s + alfa4s*R*((1-T/Talfa4s)**2 - (1-Tref/Talfa4s)**2)

        # AQUEOUS PHASE
        x2w = x1w*ms*Mw
        x3w = x2w
        x4w = 1-x1w-x2w-x3w

        # Physical
        rho = P/Zw/R/T
        a1 = a01*(1 + c11*(1 - (T/Tc1)**0.5))**2
        a2 = 0
        a3 = 0
        a4 = a04*(1 + c14*(1 - (T/Tc4)**0.5))**2
        b = b1*x1w + b2*x2w + b3*x3w + b4*x4w
        U1s = Uref1s + alfa1s*R*((1-T/Talfa1s)**2 - (1-Tref/Talfa1s)**2)
        U14 = np.log(2)*(a4/b4-2*(a1*a4)**0.5*(1-k14)/(b1+b4))
        U41 = np.log(2)*(a1/b1-2*(a4*a1)**0.5*(1-k14)/(b4+b1))
        gE = 1/b*(x1w*x2w*U1s*(b1 + b2) + x1w*x3w*U1s*(b1 + b3) + x4w*x2w*U4s*(b4 + b2) + x4w*x3w*U4s*(b4 + b3) + x1w*x4w*(b1*U14 + b4*U41))
        a = b*(x1w*a1/b1 + x2w*a2/b2 + x3w*a3/b3 + x4w*a4/b4 - gE/np.log(2))
        A = a*P/R**2/T**2
        B = b*P/R/T
        A1 = a1*P/R**2/T**2
        B1 = b1*P/R/T
        B2 = b2*P/R/T
        B3 = b3*P/R/T
        A4 = a4*P/R**2/T**2
        B4 = b4*P/R/T
        Zphys = Zw/(Zw-B) - A/(Zw+B)
        lnPHI1phys = -np.log(Zw-B) + B1/B*(B/(Zw-B) - A/(Zw+B)) - np.log((Zw+B)/Zw)*(A1/B1 - 1/(B*np.log(2))*(x2w*U1s/R/T*(B1+B2) + x3w*U1s/R/T*(B1+B3) + x4w/R/T*(B1*U14+B4*U41) - B1*gE/R/T))
        lnPHI4phys = -np.log(Zw-B) + B4/B*(B/(Zw-B) - A/(Zw+B)) - np.log((Zw+B)/Zw)*(A4/B4 - 1/(B*np.log(2))*(x1w/R/T*(B1*U14+B4*U41) + x2w*U4s/R/T*(B4+B2) + x3w*U4s/R/T*(B4+B3) - B4*gE/R/T))

        # Association
        eta = B/4/Zw
        g_eta = 1/(1-1.9*eta)
        dg_deta = 1.9/(1-1.9*eta)**2
        delta = (g_eta*kappaW)*(np.exp(epsW/T) - 1)
        DELTA = delta*P/R/T
        chi4w = Zw/(Zw + 2*x1w*chi1w*S14*DELTA)
        chi1w_new = Zw/(Zw + 2*x1w*chi1w*DELTA + 2*x4w*chi4w*S14*DELTA)
        Zassoc = -2*(1 + eta/g_eta*dg_deta)*(x1w*(1-chi1w) + x4w*(1-chi4w))
        lnPHI1assoc = 4*np.log(chi1w) + B1/(8*g_eta*Zw)*dg_deta*(x1w*4*(chi1w-1) + x4w*4*(chi4w-1))
        lnPHI4assoc = 4*np.log(chi4w) + B4/(8*g_eta*Zw)*dg_deta*(x1w*4*(chi1w-1) + x4w*4*(chi4w-1))

        # DH
        xiZi = (x2w*Z2**2 + x3w*Z2**2)
        debye = (e**2*Na*rho*xiZi/(kb*T*epsr*eps0))**0.5
        X2 = 1/Sg2**3*(np.log(1+debye*Sg2) - debye*Sg2 + 0.5*(debye*Sg2)**2)
        X3 = 1/Sg3**3*(np.log(1+debye*Sg3) - debye*Sg3 + 0.5*(debye*Sg3)**2)
        lnPHI1dh = 0
        lnPHI4dh = 0
        if ms > 0:                                          # ms is real — branch is safe
            ZDH = 1/(4*np.pi*Na*rho*xiZi) * (x2w*Z2**2*(X2 - 0.5*debye**3/(1+debye*Sg2)) + x3w*Z3**2*(X3 - 0.5*debye**3/(1+debye*Sg3)))
        else:
            ZDH = 0

        # Born
        ZBorn = 0
        lnPHI1born = 0
        lnPHI4born = 0

        # Perm
        rho1 = rho*x1w
        M = Na*rho/(3*eps0)*(x1w*pol1 + x2w*pol2 + x3w*pol3 + x4w*pol4)
        eps_inf = (2*M+1)/(1-M)
        Pww = 2*rho*x1w*delta*chi1w**2
        Pwc = 2*rho*x4w*S14*delta*chi1w*chi4w
        Pw = Pww + Pwc
        gw = 1 + zww*Pww*np.cos(GAMMA1)/(Pw*np.cos(THETA1)+1)
        T1 = (2*epsr + eps_inf)*(epsr - eps_inf)/(epsr*(eps_inf + 2)**2)
        T2 = Na*rho/(9*eps0*kb*T)*(x1w*gw*dip01**2)

        dFdchiw = 1 + 2*x1w*rho*delta*chi1w**2
        dFdchic = 2*x2w*rho*S14*delta*chi4w**2
        dGdchiw = 2*x1w*rho*S14*delta*chi1w**2
        dGdchic = 1
        dFdV = -(2*rho*x1w*delta*chi1w + 2*rho*x2w*S14*delta*chi4w)*chi1w**2
        dGdV = -(2*rho*x1w*S14*delta*chi1w)*chi1w**2
        dFdNw = 2*rho*delta*chi1w**3
        dGdNw = 2*rho*S14*delta*chi1w**3
        dFdNc = 2*rho*S14*delta*chi1w**2*chi4w
        dGdNc = 0
        VddeltadV = -delta*eta*dg_deta/g_eta
        NddeltadN1 = delta*eta*b1*dg_deta/(g_eta*b)
        NddeltadN4 = delta*eta*b4*dg_deta/(g_eta*b)
        dFddelta = -chi1w**2*(1 + (delta-1)*(2*rho*x1w*chi1w + 2*rho*x4w*chi4w*S14))
        dGddelta = -chi4w**2*(1 + (delta-1)*(2*rho*x1w*chi1w*S14))
        Vdchi4WdV = -dGdchic**-1*(dGdchiw*Vdchi1WdV + dGdV + dGddelta*VddeltadV)
        Ndchi4WdNw = -dGdchic**-1*(dGdchiw*Ndchi1WdNw + dGdNw + dGddelta*NddeltadN1)
        Ndchi4WdNc = -dGdchic**-1*(dGdchiw*Ndchi1WdNc + dGdNc + dGddelta*NddeltadN4)
        Vdchi1WdV_new = -dFdchiw**-1*(dFdchic*Vdchi4WdV + dFdV + dFddelta*VddeltadV)
        Ndchi1WdNw_new = -dFdchiw**-1*(dFdchic*Ndchi4WdNw + dFdNw + dFddelta*NddeltadN1)
        Ndchi1WdNc_new = -dFdchiw**-1*(dFdchic*Ndchi4WdNc + dFdNc + dFddelta*NddeltadN4)

        dgwdPw = -(gw-1)*np.cos(THETA1)/(Pw*np.cos(THETA1)+1)
        dgwdPww = (gw-1)/Pww
        VdPwwdV = Pww*(-1 + 1/delta*VddeltadV + 2/chi1w*Vdchi1WdV)
        VdPwcdV = Pwc*(-1 + 1/delta*VddeltadV + 1/chi1w*Vdchi1WdV + 1/chi4w*Vdchi4WdV)
        NdPwwdN1 = Pww*(1/x1w + 1/(delta)*NddeltadN1 + 2/chi1w*Ndchi1WdNw)
        NdPwcdN1 = Pwc*(1/chi1w*Ndchi1WdNw + 1/(delta)*NddeltadN1 + 1/chi4w*Ndchi4WdNw)
        NdPwwdN4 = Pww*(2/chi1w*Ndchi1WdNc + 1/(delta)*NddeltadN4)
        NdPwcdN4 = Pwc*(1/x4w + 1/chi1w*Ndchi1WdNc + 1/(delta)*NddeltadN4 + 1/chi4w*Ndchi4WdNc)
        VdPwdV = VdPwwdV + VdPwcdV
        NdPwdN1 = NdPwwdN1 + NdPwcdN1
        NdPwdN4 = NdPwwdN4 + NdPwcdN4
        VdgwdV = dgwdPw*VdPwdV + dgwdPww*VdPwwdV
        NdgwdN1 = dgwdPw*NdPwdN1 + dgwdPww*NdPwwdN1
        NdgwdN4 = dgwdPw*NdPwdN4 + dgwdPww*NdPwwdN4

        dFder = (2*epsr**2 + eps_inf**2)/(epsr**2*(eps_inf + 2)**2)
        dFdeinf = (epsr*eps_inf - 4*eps_inf - 4*epsr**2 - 2*epsr)/(epsr*(eps_inf + 2)**3)
        VdeinfdV = -3*M/(1-M)**2
        NdeinfdN1 = Na*pol1*rho/(eps0*(1-M)**2)
        NdeinfdN4 = Na*pol4*rho/(eps0*(1-M)**2)
        VdFdV = Na*rho1*dip01**2/(9*eps0*kb*T)*(gw - VdgwdV)
        NdFdN1 = -Na*rho*dip01**2/(9*eps0*kb*T)*(gw + x1w*NdgwdN1)
        NdFdN4 = -Na*rho*dip01**2/(9*eps0*kb*T)*(x1w*NdgwdN4)
        VderdV = -(dFder)**-1*(dFdeinf*VdeinfdV + VdFdV)
        NderdN1 = -(dFder)**-1*(dFdeinf*NdeinfdN1 + NdFdN1)
        NderdN4 = -(dFder)**-1*(dFdeinf*NdeinfdN4 + NdFdN4)

        if ms > 0:                                          # ms is real — branch is safe
            daDHBder = debye**2/(8*np.pi*Na*rho*epsr*xiZi)*(x2w*Z2**2*(debye/(1+debye*Sg2)-1/Rb2) + x3w*Z3**2*(debye/(1+debye*Sg3)-1/Rb3))
        else:
            daDHBder = 0
        ZPerm = -daDHBder*VderdV
        lnPHI1perm = daDHBder*NderdN1
        lnPHI4perm = daDHBder*NderdN4

        # Aqueous phase totals
        Zw_new = Zphys + Zassoc + ZDH + ZBorn + ZPerm
        lnPHI1w = lnPHI1phys + lnPHI1assoc + lnPHI1dh + lnPHI1born + lnPHI1perm
        lnPHI4w = lnPHI4phys + lnPHI4assoc + lnPHI4dh + lnPHI4born + lnPHI4perm
        f1w = np.exp(lnPHI1w)*P*x1w*1e-5
        f4w = np.exp(lnPHI4w)*P*x4w*1e-5

        # CO2-RICH PHASE
        x4c = 1-x1c

        rho = P/Zc/R/T
        a1 = a01*(1 + c11*(1 - (T/Tc1)**0.5))**2
        a4 = a04*(1 + c14*(1 - (T/Tc4)**0.5))**2
        b = b1*x1c + b4*x4c
        a14 = (a1*a4)**0.5*(1-k14)
        a = x1c**2*a1 + 2*x1c*x4c*a14 + x4c**2*a4
        A = a*P/R**2/T**2
        B = b*P/R/T
        A1 = a1*P/R**2/T**2
        B1 = b1*P/R/T
        A4 = a4*P/R**2/T**2
        B4 = b4*P/R/T
        A14 = a14*P/R**2/T**2
        Zphys = Zc/(Zc-B) - A/(Zc+B)
        lnPHI1phys = -np.log(Zc-B) + B1/B*(B/(Zc-B) - A/(Zc+B)) + A/B*(B1/B - 2*(x1c*A1 + x4c*A14)/A)*np.log(1+B/Zc)
        lnPHI4phys = -np.log(Zc-B) + B4/B*(B/(Zc-B) - A/(Zc+B)) + A/B*(B4/B - 2*(x1c*A14 + x4c*A4)/A)*np.log(1+B/Zc)

        eta = B/4/Zc
        g_eta = 1/(1-1.9*eta)
        dg_deta = 1.9/(1-1.9*eta)**2
        delta = (g_eta*kappaW)*(np.exp(epsW/T) - 1)
        DELTA = delta*P/R/T
        chi4c = Zc/(Zc + 2*x1c*chi1c*S14*DELTA)
        chi1c_new = Zc/(Zc + 2*x1c*chi1c*DELTA + 2*x4c*chi4c*S14*DELTA)
        Zassoc = -2*(1 + eta/g_eta*dg_deta)*(x1c*(1-chi1c) + x4c*(1-chi4c))
        lnPHI1assoc = 4*np.log(chi1c) + B1/(8*g_eta*Zc)*dg_deta*(x1c*4*(chi1c-1) + x4c*4*(chi4c-1))
        lnPHI4assoc = 4*np.log(chi4c) + B4/(8*g_eta*Zc)*dg_deta*(x1c*4*(chi1c-1) + x4c*4*(chi4c-1))

        Zc_new = Zphys + Zassoc
        lnPHI1c = lnPHI1phys + lnPHI1assoc
        lnPHI4c = lnPHI4phys + lnPHI4assoc
        f1c = np.exp(lnPHI1c)*P*x1c*1e-5
        f4c = np.exp(lnPHI4c)*P*x4c*1e-5

        # Residuals
        f1_x  = _rel_err(Zw, Zw_new, scale=1.0)
        f2_x  = _rel_err(T1, T2, scale=1.0)
        f3_x  = _rel_err(Zc, Zc_new, scale=1.0)
        f4_x  = (f1w - f1c) / _denom_sym(f1w, f1c)
        f5_x  = (f4w - f4c) / _denom_sym(f4w, f4c)
        f6_x  = _rel_err(chi1w, chi1w_new, scale=1.0)
        f7_x  = _rel_err(chi1c, chi1c_new, scale=1.0)
        f8_x  = _rel_err(Ndchi1WdNw, Ndchi1WdNw_new, scale=1.0)
        f9_x  = _rel_err(Ndchi1WdNc, Ndchi1WdNc_new, scale=1.0)
        f10_x = _rel_err(Vdchi1WdV, Vdchi1WdV_new, scale=1.0)

        return np.array([f1_x, f2_x, f3_x, f4_x, f5_x,          # was: dtype=np.float64
                         f6_x, f7_x, f8_x, f9_x, f10_x])

    finally:
        if _saved is not None:
            g = globals()
            for k in list(params.keys()):
                if k in _saved:
                    g[k] = _saved[k]
                else:
                    del g[k]

# === Cell 44 ===
def ELV_jac(x0, T, P, ms, params=None):
    """
    Exact Jacobian of ELV via complex-step differentiation.
    10x more accurate than finite differences, same cost (10 ELV calls).
    """
    h = 1e-20
    x0 = np.asarray(x0, dtype=complex)
    n = len(x0)
    J = np.zeros((n, n), dtype=float)
    for i in range(n):
        xp = x0.copy()
        xp[i] += 1j * h
        J[:, i] = ELV(xp, T, P, ms, params).imag / h
    return J

# === Cell 46 ===
VERBOSE_ELV = False

# === Cell 49 ===
def build_continuation_cache(T, P_bar, base_guess, ms_max, params=None,
                              dm=0.1, xtol=1e-10, maxfev=2000, verbose=False):
    """
    March ms from 0 to ms_max in steps of dm, storing converged solutions.
    Returns a list of (ms, sol) pairs, in ascending ms order.

    verbose : if True, print nfev, resnorm, and wall time for every fsolve call.
    """
    import time

    cache = []
    guess = np.asarray(base_guess, dtype=np.float64)
    m = np.float64(0.0)

    def _solve_one(m, guess):
        t0 = time.perf_counter()
        sol, info, ier, mesg = fsolve(
            ELV, guess, args=(T, P_bar*1e5, m, params),
            fprime=ELV_jac if USE_COMPLEX_JAC else None,
            full_output=True, xtol=xtol, maxfev=maxfev
        )
        sol = np.asarray(sol, dtype=np.float64)
        if verbose:
            elapsed = time.perf_counter() - t0
            resnorm = float(np.linalg.norm(
                np.asarray(ELV(sol, T, P_bar*1e5, m, params), dtype=np.float64)))
            status = "OK" if ier == 1 else f"FAIL({mesg.strip()})"
            print(f"  [cache] ms={m:.3f}  ier={ier}  nfev={info['nfev']:4d}"
                  f"  njev={info['njev']:4d}  resnorm={resnorm:.2e}"
                  f"  t={elapsed*1e3:.1f}ms  {status}")
        return sol, ier

    # solve at ms=0 first
    sol, ier = _solve_one(m, guess)
    if ier == 1 and np.all(np.isfinite(sol)):
        cache.append((float(m), sol.copy()))
        guess = sol

    # march forward
    m += dm
    while m <= ms_max + 1e-12:
        sol, ier = _solve_one(m, guess)
        if ier == 1 and np.all(np.isfinite(sol)):
            cache.append((float(m), sol.copy()))
            guess = sol
        else:
            break   # stop marching if solver fails — ms_max may be too high
        m = np.float64(m + dm)

    if verbose:
        print(f"  [cache] done: {len(cache)} points, "
              f"ms = 0.000 → {cache[-1][0]:.3f}" if cache else
              f"  [cache] done: empty")

    return cache

# === Cell 51 ===
def solve_elv(T, P_bar, ms_target, cache, params=None,
              xtol=1e-10, maxfev=2000, verbose=False):
    """
    Solves the 10-variable ELV system at a single target molality using a warm
    start from the continuation cache. Tries multiple perturbations of x1c if
    the first attempt fails.

    verbose : if True, print nfev, njev, resnorm, and wall time for every attempt.
    """
    import time

    T         = np.float64(T)
    P_bar     = np.float64(P_bar)
    ms_target = np.float64(ms_target)

    # find nearest cached solution as base
    ms_vals    = np.array([c[0] for c in cache])
    idx        = int(np.argmin(np.abs(ms_vals - ms_target)))
    base_guess = cache[idx][1].copy()

    x1c_multipliers = [1.0, 2.0, 5.0, 0.5, 0.1, 10.0, 0.01]

    sol = base_guess.copy()
    ier = 0
    resnorm = np.inf

    for i, mult in enumerate(x1c_multipliers):
        guess    = base_guess.copy()
        guess[4] = np.clip(base_guess[4] * mult, 1e-6, 1.0 - 1e-6)

        t0 = time.perf_counter()
        sol, info, ier, mesg = fsolve(
            ELV, guess, args=(T, P_bar*1e5, ms_target, params),
            fprime=ELV_jac if USE_COMPLEX_JAC else None,
            full_output=True, xtol=xtol, maxfev=maxfev
        )
        sol = np.asarray(sol, dtype=np.float64)

        # only compute resnorm if needed
        converged_flag = (ier == 1 and np.all(np.isfinite(sol)))
        if converged_flag or verbose:
            res     = np.asarray(ELV(sol, T, P_bar*1e5, ms_target, params),
                                 dtype=np.float64)
            resnorm = float(np.linalg.norm(res))

        if verbose:
            elapsed = (time.perf_counter() - t0) * 1e3
            status  = "OK" if (converged_flag and resnorm < 1e-6) else "fail"
            print(f"  [solve_elv] T={T:.1f}K P={P_bar:.2f}bar ms={ms_target:.3f}"
                  f"  attempt={i+1}/{len(x1c_multipliers)} mult={mult}"
                  f"  ier={ier}  nfev={info['nfev']:4d}  njev={info['njev']:4d}"
                  f"  resnorm={resnorm:.2e}  t={elapsed:.1f}ms  {status}")

        if converged_flag and resnorm < 1e-6:
            if i > 0 and verbose:
                print(f"  [solve_elv] converged on attempt {i+1} (mult={mult})")
            return sol, ier, resnorm, mesg

    # all attempts failed
    if verbose:
        print(f"  [solve_elv] ALL {len(x1c_multipliers)} attempts failed"
              f"  T={T:.1f}K P={P_bar:.2f}bar ms={ms_target:.3f}"
              f"  final resnorm={resnorm:.2e}")
    return sol, ier, resnorm, mesg

# === Cell 57 ===
def _cpa2_phase_check(T, P_bar, z_co2, params):
    """
    Quick CPA2 salt-free flash to classify likely phase state.
    Returns: 'two_phase', 'single_phase_liquid', 'single_phase_gas', or 'unknown'

    Classification for single-phase results uses the CPA2 phase compositions:
    - If x_CO2 in the liquid < 0.01: essentially no CO2 dissolves → single-phase liquid
    - If x_H2O in the gas < 0.01: essentially no water evaporates → single-phase gas
    CPA2 is salt-free, so these are conservative outer bounds — with salt, the
    two-phase window only shrinks further.
    """
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = CPA2.flash_co2_h2o_tpz(
                T=float(T), P_bar=float(P_bar), z_co2=float(z_co2))

        if out['phase'] == 'two_phase':
            return 'two_phase'

        # Single-phase: CPA2 convention is index 0=CO2, index 1=H2O
        # out['x'] is liquid composition, out['y'] is vapour composition
        x_co2_liq = out['x'][0]   # CO2 in liquid phase
        x_h2o_vap = out['y'][1]   # H2O in vapour phase

        if x_co2_liq < 0.01:
            return 'single_phase_liquid'
        if x_h2o_vap < 0.01:
            return 'single_phase_gas'

        # compositions look two-phase-like but CPA2 said single-phase —
        # near-critical or solver edge case, don't skip the eCPA flash
        return 'unknown'

    except Exception:
        return 'unknown'

# === Cell 59 ===
def flash_co2_h2o_salt_1d(
    T, P_bar, z_co2, m_tot,
    guess_table_fn,
    params=None,
    dm=0.1,
    ms_min=1e-8,
    ms_max=20.0,
    xtol_ms=1e-10,
    elv_res_tol=1e-6,
    scan_multipliers=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0),
    scan_absolutes=(1.0, 2.0, 5.0, 10.0),
    verbose=False,
    stability_check=False,      # ← set True for single-point diagnostics only
):
    T     = float(T)
    P_bar = float(P_bar)
    z_co2 = float(z_co2)
    m_tot = float(m_tot)

    if not (0.0 < z_co2 < 1.0):
        raise ValueError("z_co2 must be between 0 and 1 (exclusive).")
    if m_tot < 0.0:
        raise ValueError("m_tot must be >= 0.")
    if m_tot == 0.0:
        raise NotImplementedError("m_tot=0 (salt-free) branch not implemented yet.")

    # ── optional CPA2 stability pre-check (disable for bulk scans) ────────
    if stability_check:
        _phase_hint = _cpa2_phase_check(T, P_bar, z_co2, params)
        if _phase_hint in ('single_phase_liquid', 'single_phase_gas'):
            raise RuntimeError(
                f"CPA2 salt-free pre-check indicates {_phase_hint} at "
                f"T={T:.1f}K P={P_bar:.2f}bar z_co2={z_co2:.4f}. "
                f"With salt, two-phase region only shrinks — eCPA flash skipped."
            )

    # -------------------------
    # Basis: 1 mol (CO2 + H2O)
    # -------------------------
    n_co2_tot = z_co2
    n_h2o_tot = 1.0 - z_co2
    n_salt = m_tot * n_h2o_tot * Mw

    if n_salt <= 0.0:
        raise ValueError("Computed n_salt <= 0. Check Mw units and m_tot definition.")

    # -------------------------
    # Get base guess from table
    # -------------------------
    base_guess = np.asarray(guess_table_fn(T, P_bar), dtype=np.float64)

    # -------------------------
    # Build continuation cache ONCE before Brent loop
    # -------------------------
    cache = build_continuation_cache(
        T, P_bar, base_guess, ms_max,
        params=params, dm=dm, verbose=verbose    # ← added verbose
    )

    if len(cache) == 0:
        raise RuntimeError("Continuation cache is empty — ELV failed even at ms=0.")

    if verbose:
        ms_cached = [c[0] for c in cache]
        print(f"Cache built: {len(cache)} points, ms in [{ms_cached[0]:.3f}, {ms_cached[-1]:.3f}]")

    # -------------------------
    # elv_at_ms now uses cache
    # -------------------------
    def elv_at_ms(ms):
        sol, ier, resnorm, mesg = solve_elv(
            T, P_bar, ms, cache, params=params, verbose=verbose    # ← added verbose
        )
        ok = (ier == 1) and np.isfinite(resnorm) and (resnorm <= elv_res_tol) and np.all(np.isfinite(sol))
        return ok, np.asarray(sol, dtype=np.float64), float(resnorm), mesg

    def water_residual(ms):
        ms = float(ms)
        if ms <= 0.0:
            return np.nan

        ok, sol, resnorm, mesg = elv_at_ms(ms)
        if not ok:
            return np.nan

        Zw, x1w, epsr, Zc, x1c, chi1w, chi1c, Ndchi1WdNw, Ndchi1WdNc, Vdchi1WdV = sol

        x2w = x1w * ms * Mw
        x3w = x2w
        x4w = 1.0 - x1w - x2w - x3w
        x4c = 1.0 - x1c

        if not (0.0 < x1w < 1.0 and 0.0 < x1c < 1.0):
            return np.nan
        if x2w <= 0.0 or x4c <= 0.0 or x4w <= 0.0:
            return np.nan

        N_aq = n_salt / x2w
        N_c  = (n_co2_tot - N_aq * x4w) / x4c

        if N_aq <= 0.0 or N_c <= 0.0:
            return np.nan

        return N_aq * x1w + N_c * x1c - n_h2o_tot

    # -------------------------
    # Bracketing scan
    # -------------------------
    ms_list = []
    for mult in scan_multipliers:
        ms_list.append(min(ms_max, max(ms_min, mult * m_tot)))
    for v in scan_absolutes:
        ms_list.append(min(ms_max, max(ms_min, v)))
    ms_list.append(ms_min)
    ms_list.append(ms_max)
    ms_list = sorted(set(ms_list))

    finite_scan = []
    for ms in ms_list:
        r = water_residual(ms)
        if np.isfinite(r):
            finite_scan.append((ms, float(r)))
            if verbose:
                print(f"scan ms={ms:10.6g}  residual={r: .6e}")

    if len(finite_scan) < 2:
        raise RuntimeError(
            "Could not evaluate water residual at any ms in scan. "
            "ELV likely failing for all trial ms."
        )

    a = b = fa = fb = None
    for (m1, r1), (m2, r2) in zip(finite_scan[:-1], finite_scan[1:]):
        if r1 == 0.0:
            a, fa, b, fb = m1, r1, m1, r1
            break
        if r1 * r2 < 0.0:
            a, fa, b, fb = m1, r1, m2, r2
            break

    if a is None:
        raise RuntimeError(
            "No sign change found in water residual scan. "
            "This may indicate a single-phase state at this T, P."
        )

    if verbose:
        print(f"bracket: a={a} fa={fa:.3e} | b={b} fb={fb:.3e}")

    # -------------------------
    # Brent root find
    # -------------------------
    ms_aq = float(brentq(lambda x: water_residual(x), a, b, xtol=xtol_ms))

    if ms_aq < 1e-8:
        ms_aq = 0.0

    # -------------------------
    # Final ELV at ms_aq
    # -------------------------
    ok, sol, resnorm, mesg = elv_at_ms(max(ms_aq, ms_min) if ms_aq == 0.0 else ms_aq)
    if not ok:
        raise RuntimeError(f"Final ELV failed at ms={ms_aq}: {mesg} (resnorm={resnorm})")

    Zw, x1w, epsr, Zc, x1c, *_ = sol
    x2w = x1w * ms_aq * Mw
    x3w = x2w
    x4w = 1.0 - x1w - x2w - x3w
    x4c = 1.0 - x1c

    N_aq = n_salt / x2w
    N_c  = (n_co2_tot - N_aq * x4w) / x4c

    x_aq = {"x1w": float(x1w), "x2w": float(x2w), "x3w": float(x3w), "x4w": float(x4w)}
    x_c  = {"x1c": float(x1c), "x4c": float(x4c)}

    return {
        "T": float(T), "P_bar": float(P_bar),
        "z_co2": float(z_co2), "m_tot": float(m_tot),
        "n_totals": {
            "n_co2_tot": float(z_co2),
            "n_h2o_tot": float(1.0 - z_co2),
            "n_salt_tot": float(m_tot * (1.0 - z_co2) * Mw),
        },
        "ms_aq": float(ms_aq),
        "N_aq": float(N_aq), "N_c": float(N_c),
        "beta": float(N_c / (N_aq + N_c)),
        "x_aq": x_aq, "x_c": x_c,
        "Z_aq": float(sol[0]), "Z_c": float(sol[3]),
        "sol": np.asarray(sol, dtype=np.float64),
    }

# === Cell 60 ===
def flash_co2_h2o_salt_ssi(
    T, P_bar, z_co2, m_tot,
    guess_table_fn,
    params=None,
    maxiter_ms=40,
    tol_ms=1e-8,
    omega=0.7,               # SSI damping (1 = none, <1 = more conservative)
    elv_xtol=1e-10,
    elv_maxfev=2000,
    elv_res_tol=1e-6,
    verbose=False,
    stability_check=False,
):
    """
    Two-phase flash for CO2 + H2O + NaCl using folded SSI on ms_aq.

    Instead of a Brent bisection outer loop over ms_aq with a full continuation
    cache, ms_aq is updated directly from the 2x2 material balance after each
    ELV Newton solve -- no continuation, no bracketing scan.

    Algorithm
    ---------
    1. Solve ELV at ms = 0 (salt-free, table guess).
    2. Estimate initial ms_aq from the salt-free compositions via H2O/CO2
       material balance + salt balance.
    3. Iterate:
       a. Solve ELV at current ms_aq (warm-started from previous solution).
       b. Extract compositions; solve 2x2 (H2O + CO2) mass balance for N_aq, N_c.
       c. ms_aq_new = n_salt / (N_aq * x_H2O_aq * Mw)   [salt balance]
       d. Damp: ms_aq <- ms_aq + omega*(ms_aq_new - ms_aq)
       e. Stop when |delta ms_aq| < tol_ms.

    Returns the same dict structure as flash_co2_h2o_salt_1d.
    """
    T     = float(T)
    P_bar = float(P_bar)
    z_co2 = float(z_co2)
    m_tot = float(m_tot)

    if not (0.0 < z_co2 < 1.0):
        raise ValueError("z_co2 must be strictly between 0 and 1.")
    if m_tot < 0.0:
        raise ValueError("m_tot must be >= 0.")
    if m_tot == 0.0:
        raise NotImplementedError("m_tot=0 (salt-free) not implemented; use CPA2 flash.")

    if stability_check:
        hint = _cpa2_phase_check(T, P_bar, z_co2, params)
        if hint in ("single_phase_liquid", "single_phase_gas"):
            raise RuntimeError(
                f"CPA2 pre-check: {hint} at T={T:.1f}K P={P_bar:.2f}bar. "
                "Two-phase window only shrinks with salt -- eCPA flash skipped."
            )

    # ── Basis: 1 mol (CO2 + H2O) ─────────────────────────────────────────────
    n_co2_tot = z_co2
    n_h2o_tot = 1.0 - z_co2
    n_salt    = m_tot * n_h2o_tot * Mw      # mol NaCl in feed

    if n_salt <= 0.0:
        raise ValueError("n_salt <= 0. Check Mw units and m_tot.")

    # ── Internal helpers ──────────────────────────────────────────────────────
    x1c_retry_mults = [1.0, 2.0, 0.5, 5.0, 0.1, 10.0, 0.02]

    def _solve_elv(ms_aq, guess):
        """Solve 10-var ELV at ms_aq; try x1c multipliers on failure."""
        best_sol, best_rn, best_ok = guess.copy(), np.inf, False
        for mult in x1c_retry_mults:
            g = guess.copy()
            g[4] = np.clip(guess[4] * mult, 1e-6, 1.0 - 1e-6)
            sol, info, ier, mesg = fsolve(
                ELV, g, args=(T, P_bar * 1e5, ms_aq, params),
                fprime=ELV_jac if USE_COMPLEX_JAC else None,
                full_output=True, xtol=elv_xtol, maxfev=elv_maxfev,
            )
            sol = np.asarray(sol, dtype=np.float64)
            res = np.asarray(ELV(sol, T, P_bar * 1e5, ms_aq, params), dtype=np.float64)
            rn  = float(np.linalg.norm(res))
            ok  = (ier == 1) and np.all(np.isfinite(sol)) and (rn < elv_res_tol)
            if ok:
                return sol, True, rn
            if rn < best_rn:
                best_sol, best_rn, best_ok = sol.copy(), rn, ok
            if mult == 1.0 and ok:   # fast-exit on first try
                break
        return best_sol, best_ok, best_rn

    def _mass_balance(x1w, x4w, x1c, x4c):
        """
        Solve 2x2 H2O/CO2 material balance for (N_aq, N_c).
        Returns (N_aq, N_c, ms_aq_new).
        """
        det = x1w * x4c - x4w * x1c
        if abs(det) < 1e-14:
            raise ValueError(f"Degenerate phase compositions (det={det:.2e})")
        N_aq = (n_h2o_tot * x4c - n_co2_tot * x1c) / det
        N_c  = (n_co2_tot * x1w - n_h2o_tot * x4w) / det
        if N_aq <= 0.0 or N_c <= 0.0:
            raise ValueError(f"Non-physical phase amounts N_aq={N_aq:.4g} N_c={N_c:.4g}")
        ms_new = n_salt / (N_aq * x1w * Mw)
        if ms_new <= 0.0:
            raise ValueError(f"Non-physical ms_new={ms_new:.4g}")
        return float(N_aq), float(N_c), float(ms_new)

    # ── Step 0: salt-free ELV for initial ms_aq estimate ─────────────────────
    base_guess = np.asarray(guess_table_fn(T, P_bar), dtype=np.float64)
    sol0, ok0, rn0 = _solve_elv(0.0, base_guess)

    # Use salt-free compositions (ms=0 => x4w0 = 1 - x1w0, no ion terms)
    _, x1w0, _, _, x1c0, *_ = sol0
    x4w0 = 1.0 - x1w0
    x4c0 = 1.0 - x1c0

    try:
        _, _, ms_aq = _mass_balance(x1w0, x4w0, x1c0, x4c0)
        ms_aq = float(np.clip(ms_aq, m_tot * 0.5, m_tot * 100.0))
    except ValueError:
        ms_aq = m_tot   # fallback to feed molality

    if verbose:
        print(f"[SSI] ms=0 ELV ok={ok0} rn={rn0:.2e}  initial ms_aq={ms_aq:.6f}")

    guess    = sol0.copy()
    sol      = sol0.copy()
    delta    = np.inf
    converged = False

    # ── SSI loop ──────────────────────────────────────────────────────────────
    for it in range(maxiter_ms):
        sol, ok, rn = _solve_elv(ms_aq, guess)

        if not ok:
            if verbose:
                print(f"[SSI] iter {it+1}: ELV failed ms_aq={ms_aq:.6f} rn={rn:.2e}")
            break

        _, x1w, _, _, x1c, *_ = sol
        x2w = x1w * ms_aq * Mw
        x4w = 1.0 - x1w - 2.0 * x2w
        x4c = 1.0 - x1c

        if verbose:
            print(f"[SSI] iter {it+1}: ms_aq={ms_aq:.8f}  x1w={x1w:.6f}"
                  f"  x1c={x1c:.3e}  x4w={x4w:.6f}  rn={rn:.2e}")

        try:
            N_aq, N_c, ms_new = _mass_balance(x1w, x4w, x1c, x4c)
        except ValueError as e:
            if verbose:
                print(f"[SSI] iter {it+1}: mass balance failed: {e}")
            break

        # damped update
        delta  = abs(ms_new - ms_aq)
        ms_aq  = ms_aq + omega * (ms_new - ms_aq)
        guess  = sol.copy()

        if delta < tol_ms:
            converged = True
            break

    if not converged:
        raise RuntimeError(
            f"SSI ms_aq did not converge in {maxiter_ms} iterations "
            f"(T={T:.1f}K P={P_bar:.2f}bar m_tot={m_tot:.4f}, last delta={delta:.2e}). "
            "Consider increasing maxiter_ms or reducing omega."
        )

    # ── Final composition bookkeeping ─────────────────────────────────────────
    _, x1w, _, _, x1c, *_ = sol
    x2w = x1w * ms_aq * Mw
    x3w = x2w
    x4w = 1.0 - x1w - 2.0 * x2w
    x4c = 1.0 - x1c
    N_aq, N_c, _ = _mass_balance(x1w, x4w, x1c, x4c)

    if verbose:
        print(f"[SSI] converged: ms_aq={ms_aq:.8f}  beta={N_c/(N_aq+N_c):.6f}"
              f"  iters={it+1}")

    return {
        "T": T, "P_bar": P_bar,
        "z_co2": z_co2, "m_tot": m_tot,
        "n_totals": {
            "n_co2_tot": n_co2_tot,
            "n_h2o_tot": n_h2o_tot,
            "n_salt_tot": n_salt,
        },
        "ms_aq": ms_aq,
        "N_aq": N_aq, "N_c": N_c,
        "beta": N_c / (N_aq + N_c),
        "x_aq": {"x1w": x1w, "x2w": x2w, "x3w": x3w, "x4w": x4w},
        "x_c":  {"x1c": x1c, "x4c": x4c},
        "Z_aq": float(sol[0]), "Z_c": float(sol[3]),
        "sol":  np.asarray(sol, dtype=np.float64),
        "n_iter_ms": it + 1,
    }


# === Cell 62 ===
def print_flash_report(out):
    """
    Nicely formatted flash result summary, including:
      - molar densities (aq + CO2-rich)
      - mass densities (aq + CO2-rich)
      - phase volumes and saturations (volume fractions)

    Mole fractions, saturations, and beta printed as percentages.
    """

    # ---- Unpack state ----
    T     = float(out["T"])
    P_bar = float(out["P_bar"])
    z_co2 = float(out["z_co2"])
    m_tot = float(out["m_tot"])

    ms_aq = float(out["ms_aq"])
    N_aq  = float(out["N_aq"])
    N_c   = float(out["N_c"])
    beta  = float(out["beta"])

    x_aq  = out["x_aq"]
    x_c   = out["x_c"]
    sol   = out["sol"]

    Z_aq = float(sol[0])
    Z_c  = float(sol[3])

    # ---- Totals on basis n_CO2 + n_H2O = 1 ----
    n_co2_tot  = z_co2
    n_h2o_tot  = 1.0 - z_co2
    n_salt_tot = m_tot * n_h2o_tot * Mw

    # ---- Species moles in each phase ----
    n_h2o_a = N_aq * x_aq["x1w"]
    n_na_a  = N_aq * x_aq["x2w"]
    n_cl_a  = N_aq * x_aq["x3w"]
    n_co2_a = N_aq * x_aq["x4w"]

    n_h2o_c = N_c * x_c["x1c"]
    n_co2_c = N_c * x_c["x4c"]

    # ---- Molar densities via Z ----
    R    = 8.314462618
    P_Pa = P_bar * 1e5

    rho_mol_aq = P_Pa / (Z_aq * R * T)
    rho_mol_c  = P_Pa / (Z_c  * R * T)

    # ---- Phase-average molecular weights (kg/mol) ----
    try:
        M_H2O = float(M1);  M_NA = float(M2)
        M_CL  = float(M3);  M_CO2 = float(M4)
    except NameError:
        M_H2O = 18.01528e-3;  M_NA  = 22.98977e-3
        M_CL  = 35.453e-3;    M_CO2 = 44.0095e-3

    Mbar_aq = (x_aq["x1w"]*M_H2O + x_aq["x2w"]*M_NA +
               x_aq["x3w"]*M_CL  + x_aq["x4w"]*M_CO2)
    Mbar_c  =  x_c["x1c"]*M_H2O  + x_c["x4c"]*M_CO2

    # ---- Mass densities (kg/m^3) ----
    rho_mass_aq = rho_mol_aq * Mbar_aq
    rho_mass_c  = rho_mol_c  * Mbar_c

    # ---- Phase volumes on this basis (m^3) ----
    V_aq  = N_aq / rho_mol_aq
    V_c   = N_c  / rho_mol_c
    V_tot = V_aq + V_c

    # ---- Saturations / volume fractions ----
    S_aq = V_aq / V_tot
    S_c  = V_c  / V_tot

    # ---- Masses per phase (kg) ----
    m_aq = N_aq * Mbar_aq
    m_c  = N_c  * Mbar_c

    # ---- helpers ----
    def pct(x):  return f"{100*x:6.2f}%"
    def mol(x):  return f"{x:.4f} mol"
    def sci(x):  return f"{x:.4e}"

    # ---- Print report ----
    print("\n================ FLASH RESULT ================\n")
    print(f"  T = {T:.2f} K          P = {P_bar:.2f} bar")
    print(f"  z_CO2 (feed)  = {pct(z_co2)}    m_tot = {m_tot:.4f} mol/kg")
    print(f"  n_salt (basis)= {sci(n_salt_tot)} mol")

    print("\n-------- Phase Split --------")
    print(f"  beta  (CO2-rich fraction) = {pct(beta)}")
    print(f"  N_aq  = {sci(N_aq)} mol     N_c = {sci(N_c)} mol")

    print("\n-------- Aqueous Phase --------")
    print(f"  ms_aq = {ms_aq:.4f} mol/kg    Z_aq = {Z_aq:.6f}")
    print(f"  rho   = {rho_mass_aq:.1f} kg/m³    ({sci(rho_mol_aq)} mol/m³)")
    print(f"  V_aq  = {sci(V_aq)} m³    S_aq = {pct(S_aq)}")
    print(f"  x_H2O = {pct(x_aq['x1w'])}   x_CO2 = {pct(x_aq['x4w'])}")
    print(f"  x_Na+ = {pct(x_aq['x2w'])}   x_Cl- = {pct(x_aq['x3w'])}")
    print(f"  n_H2O = {mol(n_h2o_a)}   n_CO2 = {mol(n_co2_a)}")
    print(f"  n_Na+ = {mol(n_na_a)}   n_Cl- = {mol(n_cl_a)}")

    print("\n-------- CO2-Rich Phase --------")
    print(f"  Z_c   = {Z_c:.6f}")
    print(f"  rho   = {rho_mass_c:.1f} kg/m³    ({sci(rho_mol_c)} mol/m³)")
    print(f"  V_c   = {sci(V_c)} m³    S_c  = {pct(S_c)}")
    print(f"  x_H2O = {pct(x_c['x1c'])}   x_CO2 = {pct(x_c['x4c'])}")
    print(f"  n_H2O = {mol(n_h2o_c)}   n_CO2 = {mol(n_co2_c)}")

    print("\n-------- Mass Balance --------")
    print(f"  H2O : {sci(n_h2o_a+n_h2o_c)} mol  (target {sci(n_h2o_tot)})")
    print(f"  CO2 : {sci(n_co2_a+n_co2_c)} mol  (target {sci(n_co2_tot)})")
    print(f"  salt: {sci(n_na_a)}  mol  (target {sci(n_salt_tot)})")

    print("\n-------- Totals --------")
    print(f"  mass: {sci(m_aq)} kg (aq) + {sci(m_c)} kg (c) = {sci(m_aq+m_c)} kg")
    print(f"  vol : {sci(V_aq)} m³ (aq) + {sci(V_c)} m³ (c) = {sci(V_tot)} m³")

    print("\n=============================================\n")

# === Cell 64 ===
out = flash_co2_h2o_salt_1d(
    T=398.0,
    P_bar=100.0,
    z_co2=0.5,
    m_tot=0.5,
    guess_table_fn=guess_table_fn,
    params=params,
)
print_flash_report(out)

# ── SSI version for comparison ────────────────────────────────────────────────
print("\n" + "="*60)
print("flash_co2_h2o_salt_ssi (folded SSI, no continuation):")
print("="*60)
out_ssi = flash_co2_h2o_salt_ssi(
    T=398.0,
    P_bar=100.0,
    z_co2=0.5,
    m_tot=0.5,
    guess_table_fn=guess_table_fn,
    params=params,
    verbose=True,
)
print_flash_report(out_ssi)


# === Cell 66 ===
import time, warnings, traceback
import pandas as pd

# ── Benchmark grid ────────────────────────────────────────────────────────
T_bench  = [308.0, 323.0, 348.0, 373.0, 398.0, 423.0, 448.0, 473.0, 498.0, 523.0]
P_bench  = [10.0, 20.0, 50.0, 100.0, 200.0, 400.0, 800.0]
ms_bench = [0.5, 1.0, 2.0]
z_bench  = [0.3, 0.5, 0.7]

results = []
total = len(T_bench) * len(P_bench) * len(ms_bench) * len(z_bench)
print(f'Benchmark: {total} points')
print(f'  T: {T_bench}')
print(f'  P: {P_bench}')
print(f'  ms: {ms_bench}')
print(f'  z_co2: {z_bench}')
print()

for T_i in T_bench:
    P_min_tab, P_max_tab = get_table_p_range(float(T_i), CPA_GROUPS, CPA_TEMPS)
    for P_i in P_bench:
        if not (P_min_tab - 1e-6 <= P_i <= P_max_tab + 1e-6):
            continue
        for ms_i in ms_bench:
            for z_i in z_bench:
                rec = dict(T=T_i, P=P_i, ms=ms_i, z_co2=z_i)

                # ── Brent ──────────────────────────────────────────────────
                t0 = time.perf_counter()
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        out_b = flash_co2_h2o_salt_1d(
                            T=T_i, P_bar=P_i, z_co2=z_i, m_tot=ms_i,
                            guess_table_fn=guess_table_fn, params=params,
                        )
                    rec['brent_ok']   = True
                    rec['brent_beta'] = out_b['beta']
                    rec['brent_ms_aq']= out_b['ms_aq']
                except Exception as e:
                    rec['brent_ok']   = False
                    rec['brent_beta'] = float('nan')
                    rec['brent_ms_aq']= float('nan')
                rec['brent_time_s'] = time.perf_counter() - t0

                # ── SSI ────────────────────────────────────────────────────
                t0 = time.perf_counter()
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        out_s = flash_co2_h2o_salt_ssi(
                            T=T_i, P_bar=P_i, z_co2=z_i, m_tot=ms_i,
                            guess_table_fn=guess_table_fn, params=params,
                        )
                    rec['ssi_ok']     = True
                    rec['ssi_beta']   = out_s['beta']
                    rec['ssi_ms_aq']  = out_s['ms_aq']
                    rec['ssi_niters'] = out_s['n_iter_ms']
                except Exception as e:
                    rec['ssi_ok']     = False
                    rec['ssi_beta']   = float('nan')
                    rec['ssi_ms_aq']  = float('nan')
                    rec['ssi_niters'] = float('nan')
                rec['ssi_time_s'] = time.perf_counter() - t0

                results.append(rec)

df = pd.DataFrame(results)

# ── Summary ───────────────────────────────────────────────────────────────
both_ok  = df['brent_ok'] & df['ssi_ok']
brent_only_ok = df['brent_ok'] & ~df['ssi_ok']
ssi_only_ok   = ~df['brent_ok'] & df['ssi_ok']
both_fail     = ~df['brent_ok'] & ~df['ssi_ok']

print(f'Total benchmark points: {len(df)}')
print(f'  Both converged   : {both_ok.sum()}')
print(f'  Brent only       : {brent_only_ok.sum()}')
print(f'  SSI only         : {ssi_only_ok.sum()}')
print(f'  Both failed      : {both_fail.sum()}')
print()

if both_ok.any():
    sub = df[both_ok]
    print('--- Timing (converged points only) ---')
    print(f'  Brent mean/median/max: {sub["brent_time_s"].mean():.3f}s / '
          f'{sub["brent_time_s"].median():.3f}s / {sub["brent_time_s"].max():.3f}s')
    print(f'  SSI   mean/median/max: {sub["ssi_time_s"].mean():.3f}s / '
          f'{sub["ssi_time_s"].median():.3f}s / {sub["ssi_time_s"].max():.3f}s')
    speedup = sub['brent_time_s'] / sub['ssi_time_s']
    print(f'  Speedup SSI/Brent mean/median: {speedup.mean():.2f}x / {speedup.median():.2f}x')
    print()

    # Beta agreement
    dbeta = (sub['brent_beta'] - sub['ssi_beta']).abs()
    dms   = (sub['brent_ms_aq'] - sub['ssi_ms_aq']).abs()
    print('--- Accuracy (both converged) ---')
    print(f'  |beta_brent - beta_ssi|  max={dbeta.max():.2e}  mean={dbeta.mean():.2e}')
    print(f'  |ms_aq_brent - ms_aq_ssi| max={dms.max():.2e}  mean={dms.mean():.2e}')
    print()
    print(f'--- SSI iteration count (converged) ---')
    print(sub['ssi_niters'].describe().to_string())
    print()

# ── Failure cases ─────────────────────────────────────────────────────────
if brent_only_ok.any():
    print('--- Points where Brent converged but SSI failed ---')
    cols = ['T','P','ms','z_co2','brent_time_s','ssi_time_s','brent_beta']
    print(df[brent_only_ok][cols].to_string(index=False))
    print()

if ssi_only_ok.any():
    print('--- Points where SSI converged but Brent failed ---')
    cols = ['T','P','ms','z_co2','brent_time_s','ssi_time_s','ssi_beta']
    print(df[ssi_only_ok][cols].to_string(index=False))

df_bench = df  # keep for further inspection


