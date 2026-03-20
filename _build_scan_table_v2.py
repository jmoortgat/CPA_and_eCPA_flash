"""
_build_scan_table_v2.py — Comprehensive eCPA/CPA solution table + performance scan.

Grid
----
  T   = 273–633 K  (5 K steps, 73 points)
  P   = 1–1500 bar (log-spaced, 30 points)
  z   = 0.05–0.90  (uniform, 18 points)
  ms  = [0, 1e-5, 0.5, 1, 2, 3, 4, 5, 6] mol/kg  (9 salinities)

  ms = 0    → CPA2.flash_co2_h2o_tpz_robust   (pure CO₂–H₂O, no salt/electrostatics)
  ms > 0    → ecpa_stability_flash             (eCPA hierarchical: stability K-init
                                               → Wilson K fallback)

  Total: 73 × 50 × 25 × 14 = 1,277,500 grid points.

Solution table (NPZ, new path — does not overwrite the existing table)
-----------------------------------------------------------------------
  T_grid, P_grid, z_grid, ms_grid   — 1-D axis arrays
  sol        (nT, nP, nz, nms, 10)  — ELV solution vector (NaN when single-phase / failed)
  Z_aq       (nT, nP, nz, nms)      — aqueous compressibility
  Z_c        (nT, nP, nz, nms)      — CO₂-rich compressibility
  epsr       (nT, nP, nz, nms)      — rel. permittivity (NaN for CPA ms=0 rows)
  chi1w      (nT, nP, nz, nms)      — H₂O association fraction (aqueous)
  chi1c      (nT, nP, nz, nms)      — H₂O association fraction (CO₂-rich)
  x1w        (nT, nP, nz, nms)      — H₂O mol-frac (aqueous)
  x4w        (nT, nP, nz, nms)      — CO₂ mol-frac (aqueous)
  x1c        (nT, nP, nz, nms)      — H₂O mol-frac (CO₂-rich)
  x4c        (nT, nP, nz, nms)      — CO₂ mol-frac (CO₂-rich)
  ms_aq      (nT, nP, nz, nms)      — equilibrium aqueous molality
  beta       (nT, nP, nz, nms)      — CO₂-rich phase mole fraction
  is_two_phase (nT, nP, nz, nms)    — bool: True = two-phase (converged)

Performance metrics (Parquet)
------------------------------
  One row per grid point. Columns: T, P, z, ms_feed, eos_type,
  converged, is_two_phase, error_type,
  n_ssi_iters, n_elv_nfev (eCPA) / n_cpa_ssi_iters, n_cpa_newton_iters (CPA),
  n_lnphi_aq, n_lnphi_c, n_newton_aq, n_newton_aq_ok, n_newton_aq_iters,
  n_fsolve_aq, n_fsolve_aq_nfev, n_fsolve_c, n_fsolve_c_nfev,
  wall_time_ms.

Usage
-----
  python _build_scan_table_v2.py [--workers N] [--out-prefix PREFIX]
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# ── Grid definition ────────────────────────────────────────────────────────────

T_GRID  = np.arange(273.0, 634.0, 5.0)            # 73 temperatures: 273…633 K  (5 K steps)
P_GRID  = np.logspace(0.0, np.log10(1500.0), 50)  # 50 log-spaced pressures
Z_GRID  = np.linspace(0.05, 0.90, 25)             # 25 feed CO₂ fractions
MS_GRID = np.array([0.0, 1e-5, 0.1, 0.25, 0.5, 0.75,
                    1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0])  # 14 salinities

SOL_DIM = 10   # ELV solution vector length

SCRIPT_DIR = str(Path(__file__).resolve().parent)

# ── Module-level worker globals (populated by _worker_init) ───────────────────

_W_params    = None
_W_guess_fn  = None


def _worker_init(params, CPA_GROUPS, CPA_TEMPS):
    """Initialise per-process globals for spawned workers."""
    global _W_params, _W_guess_fn
    import sys as _sys
    _sys.path.insert(0, SCRIPT_DIR)
    import warnings as _w; _w.filterwarnings("ignore")
    from ecpa.guess_table import make_guess_fn
    _W_params   = params
    _W_guess_fn = make_guess_fn(CPA_GROUPS, CPA_TEMPS)


# ── CPA2 path (ms = 0) ────────────────────────────────────────────────────────

def _cpa2_flash_one(T, P, z_co2):
    """Run CPA2 hierarchical flash. Returns result dict or None on failure."""
    import CPA2
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = CPA2.flash_co2_h2o_tpz_robust(T=float(T), P_bar=float(P),
                                                 z_co2=float(z_co2))
    except Exception:
        return None
    return res


def _parse_cpa2_result(res, z_co2):
    """
    Extract standardised fields from CPA2 result.

    Returns (is_two_phase, sol10, Z_aq, Z_c, epsr, chi1w, chi1c,
             x1w, x4w, x1c, x4c, ms_aq, beta, metrics_dict).
    CPA2 convention: comp 0 = CO₂, comp 1 = H₂O.
    eCPA sol convention: [Zw, x1w, epsr, Zc, x1c, chi1w, chi1c, Ndc1wdNw, Ndc1wdNc, Vdc1wdV].
    """
    nan = np.nan
    _fail = (False, np.full(SOL_DIM, nan), nan, nan, nan, nan, nan,
             nan, nan, nan, nan, 0.0, nan,
             {"converged": False, "is_two_phase": False,
              "n_cpa_ssi_iters": 0, "n_cpa_newton_iters": 0,
              "error_type": "no_result"})

    if res is None:
        return _fail

    phase = res.get("phase", "failed")
    tie   = res.get("tie")

    if phase == "two_phase" and tie is not None and tie.get("converged"):
        x   = tie["x"]   # [x_CO2_aq, x_H2O_aq]
        y   = tie["y"]   # [y_CO2_c,  y_H2O_c ]
        Z   = tie["Z"]   # [Zx_aq,    Zy_c    ]
        chi = tie.get("chi", {})
        liq = chi.get("liq", (nan, nan))
        vap = chi.get("vap", (nan, nan))

        x1w_v = float(x[1]);   x4w_v = float(x[0])
        x1c_v = float(y[1]);   x4c_v = float(y[0])
        Z_aq  = float(Z[0]);   Z_c   = float(Z[1])
        chi1w = float(liq[0]); chi1c = float(vap[0])
        beta_v = float(res.get("beta", nan))
        ms_aq  = 0.0

        # Build partial sol (epsr and cross-derivs unavailable from CPA2)
        sol = np.full(SOL_DIM, nan)
        sol[0] = Z_aq;  sol[1] = x1w_v
        # sol[2] = epsr — NaN (not computed by CPA2)
        sol[3] = Z_c;   sol[4] = x1c_v
        sol[5] = chi1w; sol[6] = chi1c
        # sol[7..9] = cross-derivatives — NaN

        metrics = {
            "converged":            True,
            "is_two_phase":         True,
            "n_cpa_ssi_iters":      int(tie.get("ssi_iterations", 0)),
            "n_cpa_newton_iters":   int(tie.get("newton_iterations", 0)),
            "error_type":           "",
        }
        return (True, sol, Z_aq, Z_c, nan, chi1w, chi1c,
                x1w_v, x4w_v, x1c_v, x4c_v, ms_aq, beta_v, metrics)

    elif phase in ("single_phase", "single_liquid", "single_vapor",
                   "single_phase_liquid", "single_phase_gas"):
        metrics = {
            "converged":          True,
            "is_two_phase":       False,
            "n_cpa_ssi_iters":    0,
            "n_cpa_newton_iters": 0,
            "error_type":         "single_phase",
        }
        return (False, np.full(SOL_DIM, nan), nan, nan, nan, nan, nan,
                nan, nan, nan, nan, 0.0, nan, metrics)
    else:
        metrics = {
            "converged":          False,
            "is_two_phase":       False,
            "n_cpa_ssi_iters":    0,
            "n_cpa_newton_iters": 0,
            "error_type":         f"cpa_failed:{phase}",
        }
        return _fail[:-1] + (metrics,)


# ── eCPA path (ms > 0) ────────────────────────────────────────────────────────

def _ecpa_flash_warm(T, P, z_co2, ms, K_prev, sol_aq_prev, sol_c_prev):
    """
    Warm-started K-value flash.  Passes K-values and inner warm-starts from
    the previous converged point.  Raises RuntimeError on failure.
    Returns (out_dict, K_new, sol_aq_new, sol_c_new).
    """
    from ecpa.flash import flash_co2_h2o_salt_kv
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = flash_co2_h2o_salt_kv(
            T=float(T), P_bar=float(P), z_co2=float(z_co2), m_tot=float(ms),
            K_init=K_prev,
            sol_aq_x0=sol_aq_prev, sol_c_x0=sol_c_prev,
            params=_W_params, maxiter=80,
        )
    return out, out["K_vals"], out["sol_aq_x0"], out["sol_c_x0"]


def _ecpa_stability_flash(T, P, z_co2, ms, co2_ref_x0, aq_ref_x0):
    """
    Hierarchical stability + flash (Jex et al. 2024):
      1. TPD stability (6 guesses, accel SSI, Newton ZChi).
      2. If two-phase: flash with ELV guess from stability trial K-values.
      3. Fallback: Wilson K initialization.

    Returns the ecpa_stability_flash result dict, which includes a 'stability'
    sub-dict containing updated co2_ref_x0 / aq_ref_x0 for the next P call.
    Raises RuntimeError on convergence failure.
    """
    from ecpa.stability import ecpa_stability_flash
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ecpa_stability_flash(
            z_co2=float(z_co2), ms=float(ms), T=float(T), P=float(P),
            params=_W_params,
            co2_ref_x0=co2_ref_x0,
            aq_ref_x0=aq_ref_x0,
        )


# ── Row worker ────────────────────────────────────────────────────────────────

def _scan_row_worker(task):
    """
    Process one (T, z, ms) row — all P values, warm-started along P.

    task = (iT, iz, ims, T_val, z_val, ms_val, P_arr_sorted)
    Returns (iT, iz, ims, row_data_list)
      where each entry of row_data_list is a dict with all fields for one P point.
    """
    iT, iz, ims, T_val, z_val, ms_val, P_arr = task

    from ecpa import stability as _stab

    # Flash warm-start state (reset on genuine failure, kept across single-phase)
    # K-value warm-start state (K-value flash; reset on genuine failure)
    K_prev      = None   # (K1, K4) from previous converged point
    sol_aq_prev = None   # [Zw, epsr, chi1w]
    sol_c_prev  = None   # [Zc, chi1c]
    # Stability inner-fsolve warm-start refs (carried across all P in the row)
    co2_ref_x0 = None
    aq_ref_x0  = None

    row_data = []

    for iP, P_val in enumerate(P_arr):
        t0 = time.perf_counter()
        _stab.reset_call_stats()

        # ── CPA2 path (ms = 0) ────────────────────────────────────────────────
        if ms_val == 0.0:
            res = _cpa2_flash_one(T_val, P_val, z_val)
            (is_2ph, sol10, Z_aq, Z_c, epsr_v, chi1w_v, chi1c_v,
             x1w_v, x4w_v, x1c_v, x4c_v, ms_aq_v, beta_v,
             cpa_m) = _parse_cpa2_result(res, z_val)
            wall_ms = (time.perf_counter() - t0) * 1000.0
            rec = {
                "iT": iT, "iP": iP, "iz": iz, "ims": ims,
                "T": T_val, "P": P_val, "z": z_val, "ms_feed": ms_val,
                "eos_type": "CPA",
                "converged": cpa_m["converged"], "is_two_phase": is_2ph,
                "error_type": cpa_m["error_type"],
                "sol": sol10.tolist(),
                "Z_aq": Z_aq, "Z_c": Z_c, "epsr": epsr_v,
                "chi1w": chi1w_v, "chi1c": chi1c_v,
                "x1w": x1w_v, "x4w": x4w_v, "x1c": x1c_v, "x4c": x4c_v,
                "ms_aq": ms_aq_v, "beta": beta_v,
                "n_ssi_iters": cpa_m["n_cpa_ssi_iters"], "n_elv_nfev": 0,
                "n_cpa_ssi_iters": cpa_m["n_cpa_ssi_iters"],
                "n_cpa_newton_iters": cpa_m["n_cpa_newton_iters"],
                "n_lnphi_aq": 0, "n_lnphi_c": 0,
                "n_newton_aq": 0, "n_newton_aq_ok": 0, "n_newton_aq_iters": 0,
                "n_fsolve_aq": 0, "n_fsolve_aq_nfev": 0,
                "n_fsolve_c": 0, "n_fsolve_c_nfev": 0,
                "wall_time_ms": wall_ms,
            }
            row_data.append(rec)
            continue

        # ── eCPA path (ms > 0) ────────────────────────────────────────────────
        # Strategy:
        #   1. If warm start available: K-value flash from previous K_vals
        #      (fast in two-phase region; fails fast outside it).
        #   2. If no warm start OR warm start failed: hierarchical stability +
        #      flash (stability K-init → Wilson K fallback).

        out      = None
        is_2ph   = False
        err_type = ""
        n_ssi = n_elv = 0

        # Attempt 1: warm-started K-value flash
        if K_prev is not None:
            try:
                out, K_new, sol_aq_new, sol_c_new = _ecpa_flash_warm(
                    T_val, P_val, z_val, ms_val, K_prev, sol_aq_prev, sol_c_prev)
                is_2ph      = True
                K_prev      = K_new
                sol_aq_prev = sol_aq_new
                sol_c_prev  = sol_c_new
                n_ssi = int(out.get("n_iter_ms", 0))
                n_elv = int(out.get("n_elv_nfev", 0))
            except Exception:
                out = None   # fall through to stability check

        # Attempt 2: hierarchical stability + flash
        #   (stability K-init → Wilson K fallback)
        if out is None:
            try:
                sf = _ecpa_stability_flash(
                    T_val, P_val, z_val, ms_val, co2_ref_x0, aq_ref_x0)
                co2_ref_x0 = sf["stability"]["co2_ref_x0"]
                aq_ref_x0  = sf["stability"]["aq_ref_x0"]

                if sf.get("phase") == "single_phase":
                    err_type = "single_phase"
                    # Keep K_prev for possible re-entry into two-phase region.
                else:
                    out         = sf
                    is_2ph      = True
                    K_prev      = sf["K_vals"]
                    sol_aq_prev = sf["sol_aq_x0"]
                    sol_c_prev  = sf["sol_c_x0"]
                    n_ssi = int(out.get("n_iter_ms", 0))
                    n_elv = int(out.get("n_elv_nfev", 0))
            except RuntimeError as exc:
                err_type = ("ssi_no_convergence"
                            if "did not converge" in str(exc)
                            else f"runtime:{str(exc)[:60]}")
                K_prev = None; sol_aq_prev = None; sol_c_prev = None
            except Exception as exc:
                err_type = f"error:{type(exc).__name__}"
                K_prev = None; sol_aq_prev = None; sol_c_prev = None

        # Collect instrumentation stats
        istats = _stab.get_call_stats()
        wall_ms = (time.perf_counter() - t0) * 1000.0

        if out is not None and is_2ph:
            sol10   = out["sol"]
            ms_aq_new_v = float(out["ms_aq"])
            Z_aq    = float(sol10[0]); Z_c     = float(sol10[3])
            x1w_v   = float(sol10[1]); epsr_v  = float(sol10[2])
            x1c_v   = float(sol10[4]); chi1w_v = float(sol10[5])
            chi1c_v = float(sol10[6])
            x2w_v   = x1w_v * ms_aq_new_v * 0.018015
            x4w_v   = 1.0 - x1w_v - 2.0 * x2w_v
            x4c_v   = 1.0 - x1c_v
            beta_v  = float(out["beta"])
            conv    = True
        else:
            sol10   = np.full(SOL_DIM, np.nan)
            Z_aq = Z_c = epsr_v = chi1w_v = chi1c_v = np.nan
            x1w_v = x4w_v = x1c_v = x4c_v = beta_v = np.nan
            ms_aq_new_v = np.nan
            conv = (err_type == "single_phase")   # single-phase IS a resolved result

        rec = {
            "iT": iT, "iP": iP, "iz": iz, "ims": ims,
            "T": T_val, "P": P_val, "z": z_val, "ms_feed": ms_val,
            "eos_type": "eCPA",
            "converged": conv, "is_two_phase": is_2ph, "error_type": err_type,
            "sol": sol10.tolist() if hasattr(sol10, "tolist") else list(sol10),
            "Z_aq": Z_aq, "Z_c": Z_c, "epsr": epsr_v,
            "chi1w": chi1w_v, "chi1c": chi1c_v,
            "x1w": x1w_v, "x4w": x4w_v, "x1c": x1c_v, "x4c": x4c_v,
            "ms_aq": ms_aq_new_v, "beta": beta_v,
            "n_ssi_iters": n_ssi, "n_elv_nfev": n_elv,
            "n_cpa_ssi_iters": 0, "n_cpa_newton_iters": 0,
            "n_lnphi_aq":        istats.get("n_lnphi_aq",        0),
            "n_lnphi_c":         istats.get("n_lnphi_c",         0),
            "n_newton_aq":       istats.get("n_newton_aq",       0),
            "n_newton_aq_ok":    istats.get("n_newton_aq_ok",    0),
            "n_newton_aq_iters": istats.get("n_newton_aq_iters", 0),
            "n_fsolve_aq":       istats.get("n_fsolve_aq",       0),
            "n_fsolve_aq_nfev":  istats.get("n_fsolve_aq_nfev",  0),
            "n_fsolve_c":        istats.get("n_fsolve_c",        0),
            "n_fsolve_c_nfev":   istats.get("n_fsolve_c_nfev",   0),
            "wall_time_ms": wall_ms,
        }
        row_data.append(rec)

    return iT, iz, ims, row_data


# ── Main scan ──────────────────────────────────────────────────────────────────

def run_scan(n_workers=None, out_prefix="results/scan_v2"):
    """Build the solution table and performance metrics."""
    sys.path.insert(0, SCRIPT_DIR)

    from ecpa.parameters import make_params
    from ecpa.guess_table import load_cpa_guess_table

    params     = make_params()
    CPA_GROUPS, CPA_TEMPS = load_cpa_guess_table()

    nT  = len(T_GRID)
    nP  = len(P_GRID)
    nz  = len(Z_GRID)
    nms = len(MS_GRID)
    total_rows   = nT * nz * nms   # (T, z, ms) rows
    total_points = total_rows * nP

    print(f"Grid: {nT}T × {nP}P × {nz}z × {nms}ms = {total_points:,} points "
          f"({total_rows:,} rows × {nP} P-values each)")
    print(f"T = {T_GRID[0]:.0f}–{T_GRID[-1]:.0f} K, "
          f"P = {P_GRID[0]:.1f}–{P_GRID[-1]:.0f} bar, "
          f"ms = {MS_GRID}")

    # ── Allocate result arrays ─────────────────────────────────────────────────
    sol_arr    = np.full((nT, nP, nz, nms, SOL_DIM), np.nan)
    Z_aq_arr   = np.full((nT, nP, nz, nms), np.nan)
    Z_c_arr    = np.full((nT, nP, nz, nms), np.nan)
    epsr_arr   = np.full((nT, nP, nz, nms), np.nan)
    chi1w_arr  = np.full((nT, nP, nz, nms), np.nan)
    chi1c_arr  = np.full((nT, nP, nz, nms), np.nan)
    x1w_arr    = np.full((nT, nP, nz, nms), np.nan)
    x4w_arr    = np.full((nT, nP, nz, nms), np.nan)
    x1c_arr    = np.full((nT, nP, nz, nms), np.nan)
    x4c_arr    = np.full((nT, nP, nz, nms), np.nan)
    ms_aq_arr  = np.full((nT, nP, nz, nms), np.nan)
    beta_arr   = np.full((nT, nP, nz, nms), np.nan)
    is2ph_arr  = np.zeros((nT, nP, nz, nms), dtype=bool)

    all_metrics = []

    # ── Build task list ────────────────────────────────────────────────────────
    P_sorted = np.sort(P_GRID)   # ascending for warm-start continuity
    P_order  = np.argsort(P_GRID)   # map from sorted-P index → original P index

    tasks = []
    for iT, T_val in enumerate(T_GRID):
        for iz, z_val in enumerate(Z_GRID):
            for ims, ms_val in enumerate(MS_GRID):
                tasks.append((iT, iz, ims, T_val, z_val, ms_val, P_sorted))

    assert len(tasks) == total_rows

    # ── Run in parallel ────────────────────────────────────────────────────────
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    n_workers = min(n_workers, total_rows)
    print(f"Launching {n_workers} workers for {total_rows:,} row tasks …")

    ctx = mp.get_context("spawn")
    t_start = time.perf_counter()
    rows_done = 0

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx,
                              initializer=_worker_init,
                              initargs=(params, CPA_GROUPS, CPA_TEMPS)) as pool:
        futures = {pool.submit(_scan_row_worker, task): task for task in tasks}

        for fut in as_completed(futures):
            try:
                iT, iz, ims, row_data = fut.result()
            except Exception as exc:
                task = futures[fut]
                print(f"  [ERROR] task iT={task[0]} iz={task[2]} ims={task[3]}: {exc}")
                rows_done += 1
                continue

            # Store results — row_data is indexed by sorted-P order
            for rec in row_data:
                iP_sorted = rec["iP"]
                iP_orig   = P_order[iP_sorted]   # original P index

                sol10 = np.array(rec["sol"])
                sol_arr  [iT, iP_orig, iz, ims] = sol10
                Z_aq_arr [iT, iP_orig, iz, ims] = rec["Z_aq"]
                Z_c_arr  [iT, iP_orig, iz, ims] = rec["Z_c"]
                epsr_arr [iT, iP_orig, iz, ims] = rec["epsr"]
                chi1w_arr[iT, iP_orig, iz, ims] = rec["chi1w"]
                chi1c_arr[iT, iP_orig, iz, ims] = rec["chi1c"]
                x1w_arr  [iT, iP_orig, iz, ims] = rec["x1w"]
                x4w_arr  [iT, iP_orig, iz, ims] = rec["x4w"]
                x1c_arr  [iT, iP_orig, iz, ims] = rec["x1c"]
                x4c_arr  [iT, iP_orig, iz, ims] = rec["x4c"]
                ms_aq_arr[iT, iP_orig, iz, ims] = rec["ms_aq"]
                beta_arr [iT, iP_orig, iz, ims] = rec["beta"]
                is2ph_arr[iT, iP_orig, iz, ims] = bool(rec["is_two_phase"])

                # Flatten metrics for Parquet
                m = {k: rec[k] for k in [
                    "T", "P", "z", "ms_feed", "eos_type",
                    "converged", "is_two_phase", "error_type",
                    "n_ssi_iters", "n_elv_nfev",
                    "n_cpa_ssi_iters", "n_cpa_newton_iters",
                    "n_lnphi_aq", "n_lnphi_c",
                    "n_newton_aq", "n_newton_aq_ok", "n_newton_aq_iters",
                    "n_fsolve_aq", "n_fsolve_aq_nfev",
                    "n_fsolve_c", "n_fsolve_c_nfev",
                    "wall_time_ms",
                ]}
                all_metrics.append(m)

            rows_done += 1
            if rows_done % max(1, total_rows // 50) == 0 or rows_done == total_rows:
                elapsed = time.perf_counter() - t_start
                rate = rows_done / elapsed if elapsed > 0 else 1.0
                eta  = (total_rows - rows_done) / rate
                print(f"  {rows_done:5d}/{total_rows} rows done  "
                      f"({100*rows_done/total_rows:.1f}%)  "
                      f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s")

    elapsed_total = time.perf_counter() - t_start
    print(f"\nScan finished in {elapsed_total:.1f} s")

    # ── Summary statistics ──────────────────────────────────────────────────────
    n_two_phase  = int(is2ph_arr.sum())
    n_total      = is2ph_arr.size
    print(f"Two-phase:  {n_two_phase:,} / {n_total:,} "
          f"({100*n_two_phase/n_total:.1f}%)")
    for ims, ms_val in enumerate(MS_GRID):
        frac = float(is2ph_arr[:, :, :, ims].mean())
        print(f"  ms={ms_val:.4g}:  {100*frac:.1f}% two-phase")

    # ── Save solution table (NPZ) ──────────────────────────────────────────────
    npz_path = Path(f"{out_prefix}_table.npz")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        T_grid=T_GRID, P_grid=P_GRID, z_grid=Z_GRID, ms_grid=MS_GRID,
        sol=sol_arr,
        Z_aq=Z_aq_arr, Z_c=Z_c_arr,
        epsr=epsr_arr,
        chi1w=chi1w_arr, chi1c=chi1c_arr,
        x1w=x1w_arr, x4w=x4w_arr,
        x1c=x1c_arr, x4c=x4c_arr,
        ms_aq=ms_aq_arr, beta=beta_arr,
        is_two_phase=is2ph_arr,
    )
    print(f"Saved solution table → {npz_path}  "
          f"({npz_path.stat().st_size/1e6:.1f} MB)")

    # ── Save performance metrics (Parquet) ─────────────────────────────────────
    pq_path = Path(f"{out_prefix}_metrics.parquet")
    df = pd.DataFrame(all_metrics)
    df.to_parquet(pq_path, index=False)
    print(f"Saved metrics        → {pq_path}  ({len(df):,} rows, "
          f"{pq_path.stat().st_size/1e6:.1f} MB)")

    return npz_path, pq_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build eCPA/CPA solution table v2")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: cpu_count-1)")
    parser.add_argument("--out-prefix", type=str, default="results/scan_v2",
                        help="Output file prefix (default: results/scan_v2)")
    args = parser.parse_args()

    run_scan(n_workers=args.workers, out_prefix=args.out_prefix)
