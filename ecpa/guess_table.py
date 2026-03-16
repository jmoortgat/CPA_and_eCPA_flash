"""
CPA initial-guess table: load, interpolate, query.

The table is a Parquet file consolidating the salt-free CPA_ELV_T###K.dat
files.  Each row stores (T_K, P_bar, Zw, xw_W, eps_r, Zc, xw_C,
chiw_W, chiw_C, Ndchi1w_dNw, Ndchi1w_dNc, Vdchi1w_dV).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ── Column names in the .dat files ────────────────────────────────────────────
_COLS = [
    "P_bar",
    "Z_W", "xw_W", "eps_r",
    "Z_C", "xw_C",
    "chiw_W", "chiw_C",
    "Ndchi1w_dNw", "Ndchi1w_dNc", "Vdchi1w_dV",
]

# Ordered list of variable names returned by the lookup functions
GUESS_COLS = [
    "Zw", "xw_W", "eps_r", "Zc", "xw_C",
    "chiw_W", "chiw_C", "Ndchi1w_dNw", "Ndchi1w_dNc", "Vdchi1w_dV",
]


def read_cpa_elv_dat(path: Path) -> pd.DataFrame:
    """Read one CPA_ELV_T###K.dat file; infers T from filename."""
    m = re.search(r"_T(\d+)K\.dat$", path.name)
    if not m:
        raise ValueError(f"Could not parse temperature from filename: {path.name}")
    T_K = int(m.group(1))
    df = pd.read_csv(path, sep=r"\s+", engine="python", skiprows=1, names=_COLS)
    df.insert(0, "T_K", T_K)
    return df


def build_parquet(dat_dir: Path = Path("CO2"),
                  out_path: Path = Path("CO2/CPA_ELV_all.parquet")) -> None:
    """Consolidate all .dat files into a single Parquet (runs once)."""
    if out_path.exists():
        print(f"{out_path} already exists — skipping.")
        return
    paths = sorted(dat_dir.glob("CPA_ELV_T*K.dat"))
    if not paths:
        raise FileNotFoundError(f"No files matching {dat_dir / 'CPA_ELV_T*K.dat'}")
    dfs = [read_cpa_elv_dat(p) for p in paths]
    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df.sort_values(["T_K", "P_bar"]).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_parquet(out_path, index=False)
    print(f"Wrote {out_path}  ({len(all_df):,} rows)")


def load_cpa_guess_table(parquet_path: str | Path = "CO2/CPA_ELV_all.parquet"):
    """
    Load the CPA guess table from Parquet.

    Returns
    -------
    groups : dict[int -> DataFrame]
        Keyed by integer T_K; each DataFrame sorted by P_bar.
    temps : np.ndarray
        Sorted array of available temperatures [K].
    """
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"{parquet_path} not found.")

    df = pd.read_parquet(parquet_path)

    if "[bar]" in df.columns:
        df = df.drop(columns=["[bar]"])

    rename_map = {"Z_W": "Zw", "Z_C": "Zc"}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = ["T_K", "P_bar"] + GUESS_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {parquet_path}: {missing}")

    df = df.sort_values(["T_K", "P_bar"]).reset_index(drop=True)
    groups = {int(T): sub.sort_values("P_bar").reset_index(drop=True)
              for T, sub in df.groupby("T_K")}
    temps = np.array(sorted(groups.keys()), dtype=float)
    return groups, temps


def guess_from_table_nearest(T: float, P_bar: float, groups: dict,
                              temps: np.ndarray) -> np.ndarray:
    """Nearest-neighbour lookup (legacy; prefer guess_from_table)."""
    T_closest = temps[np.argmin(np.abs(temps - T))]
    dfT = groups[int(T_closest)]
    Pvals = dfT["P_bar"].to_numpy(dtype=float)
    idx = int(np.argmin(np.abs(Pvals - P_bar)))
    row = dfT.iloc[idx]
    return np.array([row[c] for c in GUESS_COLS], dtype=np.float64)


def guess_from_table(T: float, P_bar: float, groups: dict,
                     temps: np.ndarray) -> np.ndarray:
    """Bilinear interpolation in T and P (preferred)."""
    if len(temps) == 0:
        raise ValueError("No temperatures available in guess table.")

    idx_T = int(np.clip(np.searchsorted(temps, T), 1, len(temps) - 1))
    T_lo, T_hi = temps[idx_T - 1], temps[idx_T]
    wT = (T - T_lo) / (T_hi - T_lo) if T_hi != T_lo else 0.0

    def _at_T(T_key: float) -> np.ndarray:
        dfT = groups[int(T_key)]
        Pvals = dfT["P_bar"].to_numpy(dtype=float)
        idx_P = int(np.clip(np.searchsorted(Pvals, P_bar), 1, len(Pvals) - 1))
        P_lo_p, P_hi_p = Pvals[idx_P - 1], Pvals[idx_P]
        wP = (P_bar - P_lo_p) / (P_hi_p - P_lo_p) if P_hi_p != P_lo_p else 0.0
        v_lo = np.array([dfT.iloc[idx_P - 1][c] for c in GUESS_COLS], dtype=np.float64)
        v_hi = np.array([dfT.iloc[idx_P    ][c] for c in GUESS_COLS], dtype=np.float64)
        return v_lo + wP * (v_hi - v_lo)

    g_lo = _at_T(T_lo)
    g_hi = _at_T(T_hi)
    return g_lo + wT * (g_hi - g_lo)


def make_guess_fn(groups: dict, temps: np.ndarray):
    """
    Return a callable  guess_fn(T, P_bar) -> np.ndarray  that closes over the
    loaded table.  Pass this as `guess_table_fn` to the flash functions.
    """
    def _fn(T: float, P_bar: float) -> np.ndarray:
        return guess_from_table(int(round(T)), P_bar, groups, temps)
    return _fn


def get_table_p_range(T: float, groups: dict, temps: np.ndarray):
    """Return (P_min, P_max) available in the table at the closest T."""
    T_closest = temps[np.argmin(np.abs(temps - T))]
    dfT = groups[int(T_closest)]
    return float(dfT["P_bar"].min()), float(dfT["P_bar"].max())
