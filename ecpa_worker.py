"""
ecpa_worker.py — shared implementation for parallel eCPA flash workers.

Do NOT call this module directly from ProcessPoolExecutor.
Use the algorithm-specific wrappers instead:

    ecpa_worker_ssi.py   — SSI flash  (preferred / default)
    ecpa_worker_brent.py — Brent flash

Each wrapper calls `_compute(task, flash_algo)` defined here.

Task tuple (shared format):
    (T, P_bar, z_co2, ms, parquet_path, params)

    T             : float — temperature [K]
    P_bar         : float — pressure [bar]
    z_co2         : float — feed CO₂ mole fraction
    ms            : float — total salt molality [mol/kg]
    parquet_path  : str   — path to CPA_ELV_all.parquet
    params        : dict  — EoS parameter overrides (may be empty / {})

Result dict keys:
    T, P_bar, z_co2, ms, flash_algo,
    converged, beta, ms_aq, Z_aq, Z_c,
    error, error_type
"""

import warnings

import numpy as np

from ecpa.flash import get_flash_fn
from ecpa.guess_table import load_cpa_guess_table, make_guess_fn

# ── Per-process guess-table cache ─────────────────────────────────────────────
_GUESS_TABLE_CACHE: dict = {}   # parquet_path → guess_fn


def _get_guess_fn(parquet_path: str):
    if parquet_path not in _GUESS_TABLE_CACHE:
        groups, temps = load_cpa_guess_table(parquet_path)
        _GUESS_TABLE_CACHE[parquet_path] = make_guess_fn(groups, temps)
    return _GUESS_TABLE_CACHE[parquet_path]


# ── Shared implementation ──────────────────────────────────────────────────────

def _compute(task: tuple, flash_algo: str) -> dict:
    """
    Run one eCPA flash point with the specified algorithm.
    Called by the algorithm-specific worker modules.
    """
    T, P_bar, z_co2, ms, parquet_path, params = task

    result = dict(
        T=T, P_bar=P_bar, z_co2=z_co2, ms=ms, flash_algo=flash_algo,
        converged=False, beta=np.nan, ms_aq=np.nan,
        Z_aq=np.nan, Z_c=np.nan, error="", error_type="",
    )

    guess_fn = _get_guess_fn(parquet_path)
    flash_fn = get_flash_fn(flash_algo)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out = flash_fn(
                T=float(T), P_bar=float(P_bar),
                z_co2=float(z_co2), m_tot=float(ms),
                guess_table_fn=guess_fn, params=params,
            )
        result["converged"]  = True
        result["error_type"] = "none"
        result["beta"]       = out["beta"]
        result["ms_aq"]      = out["ms_aq"]
        result["Z_aq"]       = out["Z_aq"]
        result["Z_c"]        = out["Z_c"]

    except RuntimeError as exc:
        msg = str(exc)
        result["error"] = msg[:80]
        if "sign change" in msg:
            result["error_type"] = "no_sign_change"
        elif "ELV likely failing" in msg:
            result["error_type"] = "elv_solver"
        elif "cache is empty" in msg:
            result["error_type"] = "cache_empty"
        elif "did not converge" in msg:
            result["error_type"] = "ssi_no_converge"
        else:
            result["error_type"] = "runtime_other"

    except Exception as exc:
        result["error"]      = f"{type(exc).__name__}: {str(exc)[:60]}"
        result["error_type"] = "exception"

    return result
