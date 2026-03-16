"""
Experimental CO₂–H₂O VLE data: load from EXP folder and provide lookup helpers.

Dataset columns
---------------
T_K         temperature [K]
P_bar       pressure [bar]
xc_W        CO₂ mol-frac in the aqueous (water-rich) phase
yw_C        H₂O mol-frac in the CO₂-rich phase
rho_W       aqueous-phase density [kg/m³]  (sparse)
exp_id      experiment number within that T folder
reference   author/year string
source_file original filename

Usage
-----
    from ecpa.exp_data import load_exp_data, exp_at_T, exp_xc_W, exp_yw_C
    EXP_DF, EXP_TEMPS = load_exp_data()
    df = exp_xc_W(323, EXP_DF)
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd


_DEFAULT_DATA_DIR    = Path("EXP/CO2-WATER")
_DEFAULT_PARQUET     = Path("CO2/CO2_WATER_exp.parquet")


def _parse_co2water_dir(data_dir: Path) -> pd.DataFrame:
    """Read every EXP*.txt in the CO2-WATER folder tree."""
    records = []
    for txt_path in sorted(data_dir.rglob("EXP*.txt")):
        if "(copy)" in txt_path.name.lower():
            continue
        m_T = re.search(r"T(\d+)K", str(txt_path))
        if not m_T:
            continue
        T_K = int(m_T.group(1))
        m_exp = re.search(r"EXP(\d+)_", txt_path.name)
        exp_id = int(m_exp.group(1)) if m_exp else 0

        lines = txt_path.read_text(encoding="utf-8", errors="replace").splitlines()

        reference, data_start = "", 0
        for i, line in enumerate(lines):
            if line.strip().startswith("#"):
                reference = line.strip().lstrip("#").strip()
                data_start = i + 1
                break

        col_xc_W, col_yw_C, col_rho = 1, 2, 3
        yc_C_mode = False

        for i in range(data_start, len(lines)):
            line = lines[i].strip()
            if line and not line.startswith("#"):
                cleaned = re.sub(r"\[.*?\]", "", line)
                headers = cleaned.lower().split()
                for j, h in enumerate(headers):
                    if h == "xc_w":
                        col_xc_W = j
                    if h == "yw_c":
                        col_yw_C = j
                        yc_C_mode = False
                    if h == "yc_c":
                        col_yw_C = j
                        yc_C_mode = True
                    if "rho" in h:
                        col_rho = j
                data_start = i + 1
                break

        def _v(s: str) -> float:
            return np.nan if s.lower() == "x" else float(s)

        for line in lines[data_start:]:
            parts = line.strip().split()
            if len(parts) < 2 or parts[0].startswith("#"):
                continue
            try:
                raw_y = _v(parts[col_yw_C]) if len(parts) > col_yw_C else np.nan
                yw_C = (1.0 - raw_y) if (yc_C_mode and np.isfinite(raw_y)) else raw_y
                records.append(dict(
                    T_K=T_K,
                    P_bar=float(parts[0]),
                    xc_W=_v(parts[col_xc_W]) if len(parts) > col_xc_W else np.nan,
                    yw_C=yw_C,
                    rho_W=_v(parts[col_rho]) if len(parts) > col_rho else np.nan,
                    exp_id=exp_id,
                    reference=reference,
                    source_file=txt_path.name,
                ))
            except (ValueError, IndexError):
                continue

    df = pd.DataFrame(records)
    return df.sort_values(["T_K", "P_bar", "exp_id"]).reset_index(drop=True)


def load_exp_data(data_dir: Path | str = _DEFAULT_DATA_DIR,
                  parquet_path: Path | str = _DEFAULT_PARQUET):
    """
    Load experimental data from Parquet (or build it from raw text files).

    Returns
    -------
    df    : pd.DataFrame  (all experimental rows)
    temps : np.ndarray    (sorted unique T values [K])
    """
    data_dir     = Path(data_dir)
    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        print(f"Loaded {parquet_path}  ({len(df)} rows)")
    else:
        print(f"Parsing {data_dir} ...")
        df = _parse_co2water_dir(data_dir)
        df.to_parquet(parquet_path, index=False)
        print(f"Saved {parquet_path}  ({len(df)} rows)")

    temps = np.array(sorted(df["T_K"].unique()), dtype=float)
    return df, temps


# ── Lookup helpers ─────────────────────────────────────────────────────────────

def exp_at_T(T_K: float, df: pd.DataFrame, tol_K: float = 1.0) -> pd.DataFrame:
    """All experimental rows within ±tol_K of T_K."""
    return df[np.abs(df["T_K"] - T_K) <= tol_K].copy()


def exp_xc_W(T_K: float, df: pd.DataFrame, tol_K: float = 1.0) -> pd.DataFrame:
    """CO₂ solubility rows (xc_W not NaN) near T_K."""
    sub = exp_at_T(T_K, df, tol_K)
    return sub[sub["xc_W"].notna()][["P_bar", "xc_W", "reference"]].copy()


def exp_yw_C(T_K: float, df: pd.DataFrame, tol_K: float = 1.0) -> pd.DataFrame:
    """H₂O content in CO₂-rich phase (yw_C not NaN) near T_K."""
    sub = exp_at_T(T_K, df, tol_K)
    return sub[sub["yw_C"].notna()][["P_bar", "yw_C", "reference"]].copy()
