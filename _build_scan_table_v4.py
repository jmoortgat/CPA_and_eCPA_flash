"""
_build_scan_table_v4.py — Robust 3-D eCPA/CPA solution table (z-free design).

Grid design
-----------
  T  = 273–633 K, 1 K steps       →  361 points
  P  = 1–2000 bar, log-spaced     →  100 points
  ms = [0, 1e-5, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
     = 14 salinities (same as scan_v3)

Why no z dimension
------------------
K-values and Newton inner states (Z_aq, ε_r, χ₁w, Z_c, χ₁c) depend only on
(T, P, ms_aq) — the equilibrium conditions — not on the feed split.  The feed
z_CO₂ only determines the phase fraction β via the lever rule.  Empirically,
the z-spread of K4 across z=0.05–0.90 is < 0.2% at subsurface conditions
(T=323K, P=100bar, ms=1) and < 1% for most of the validated range (see
scan_v3 analysis).  The 25 z-points in scan_v3 were therefore wasted — those
slots are now used for finer T (5K → 1K) and P (50 → 100 points) coverage.

A single representative z = Z_REP = 0.5 is used for all stability calls.

Total: 361 × 100 × 14 = 505,400 points  (~2× scan_v3 in T×P coverage,
       25× fewer in z, net ~5× less data — fits in ~120 MB vs ~350 MB).

Changes from scan_v3
---------------------
* No z dimension → 3-D table (nT, nP, nms)
* T: 1 K steps from 273 K (was 5 K from 288 K)
* P: 100 points to 2000 bar (was 50 points to 1500 bar)
* Uses current code: B1/B4 bugfix, S14_eff≥0, Newton aqueous inner solve

Output
------
  results/scan_v4_table.npz    — solution table
  results/scan_v4_metrics.parq — per-point performance metrics

Compatibility: ScanTableWarmStart auto-detects 3-D vs 4-D tables by ndim.

Usage
-----
  python _build_scan_table_v4.py [--workers N] [--out results/scan_v4]
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

# ── Grid ──────────────────────────────────────────────────────────────────────

T_GRID  = np.arange(273.0, 634.0, 1.0)             # 361 points, 1 K steps
P_GRID  = np.logspace(0.0, np.log10(2000.0), 100)  # 100 log-spaced, 1–2000 bar
MS_GRID = np.array([0.0, 1e-5, 0.1, 0.25, 0.5, 0.75,
                    1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0])  # 14 salinities

Z_REP   = 0.5    # representative feed CO₂ fraction for all stability calls

SOL_DIM = 10     # ELV solution vector length

SCRIPT_DIR = str(Path(__file__).resolve().parent)

# ── Worker globals ─────────────────────────────────────────────────────────────

_W_params = None


def _worker_init(params):
    global _W_params
    import sys as _sys
    _sys.path.insert(0, SCRIPT_DIR)
    import warnings as _w; _w.filterwarnings("ignore")
    _W_params = params


# ── CPA2 path (ms = 0) ────────────────────────────────────────────────────────

def _cpa2_flash_one(T, P, z_co2=Z_REP):
    import CPA2
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = CPA2.flash_co2_h2o_tpz_robust(
                T=float(T), P_bar=float(P), z_co2=float(z_co2))
    except Exception:
        return None
    return res


def _parse_cpa2_result(res):
    nan = np.nan
    _fail = (False, np.full(SOL_DIM, nan), nan, nan, nan, nan, nan,
             nan, nan, nan, nan, 0.0, nan, False)
    if res is None:
        return _fail

    phase = res.get("phase", "failed")
    tie   = res.get("tie")

    if phase == "two_phase" and tie is not None and tie.get("converged"):
        x   = tie["x"]   # [x_CO2_aq, x_H2O_aq]
        y   = tie["y"]   # [y_CO2_c,  y_H2O_c ]
        Z   = tie["Z"]
        chi = tie.get("chi", {})
        liq = chi.get("liq", (nan, nan))
        vap = chi.get("vap", (nan, nan))

        x1w_v = float(x[1]);   x4w_v = float(x[0])
        x1c_v = float(y[1]);   x4c_v = float(y[0])
        Z_aq  = float(Z[0]);   Z_c   = float(Z[1])
        chi1w = float(liq[0]); chi1c = float(vap[0])
        beta_v = float(res.get("beta", nan))

        sol = np.full(SOL_DIM, nan)
        sol[0] = Z_aq;  sol[1] = x1w_v
        sol[3] = Z_c;   sol[4] = x1c_v
        sol[5] = chi1w; sol[6] = chi1c

        return (True, sol, Z_aq, Z_c, nan, chi1w, chi1c,
                x1w_v, x4w_v, x1c_v, x4c_v, 0.0, beta_v, True)

    elif "single" in phase:
        return (False, np.full(SOL_DIM, nan), nan, nan, nan, nan, nan,
                nan, nan, nan, nan, 0.0, nan, True)   # converged=True, single-phase
    else:
        return _fail


# ── eCPA row worker ────────────────────────────────────────────────────────────

def _scan_row_worker(task):
    """
    Process one (T, ms) row — all 100 P values warm-started along P.

    task = (iT, ims, T_val, ms_val, P_arr_sorted)
    Returns (iT, ims, row_data_list)
    """
    iT, ims, T_val, ms_val, P_arr = task

    from ecpa import stability as _stab
    from ecpa.flash import flash_co2_h2o_salt_kv
    from ecpa.stability import ecpa_stability_flash

    K_prev      = None
    sol_aq_prev = None
    sol_c_prev  = None
    co2_ref_x0  = None
    aq_ref_x0   = None

    row_data = []

    for iP, P_val in enumerate(P_arr):
        t0 = time.perf_counter()
        _stab.reset_call_stats()

        # ── CPA2 path ─────────────────────────────────────────────────────────
        if ms_val == 0.0:
            res = _cpa2_flash_one(T_val, P_val, Z_REP)
            (is_2ph, sol10, Z_aq, Z_c, epsr_v, chi1w_v, chi1c_v,
             x1w_v, x4w_v, x1c_v, x4c_v, ms_aq_v, beta_v,
             conv) = _parse_cpa2_result(res)
            wall_ms = (time.perf_counter() - t0) * 1000.0
            row_data.append({
                "iT": iT, "iP": iP, "ims": ims,
                "T": T_val, "P": P_val, "ms_feed": ms_val,
                "eos_type": "CPA",
                "converged": conv, "is_two_phase": is_2ph,
                "sol": sol10.tolist(),
                "Z_aq": Z_aq, "Z_c": Z_c, "epsr": epsr_v,
                "chi1w": chi1w_v, "chi1c": chi1c_v,
                "x1w": x1w_v, "x4w": x4w_v,
                "x1c": x1c_v, "x4c": x4c_v,
                "ms_aq": ms_aq_v, "beta": beta_v,
                "wall_time_ms": wall_ms,
            })
            continue

        # ── eCPA path ─────────────────────────────────────────────────────────
        out      = None
        is_2ph   = False
        conv     = False
        err_type = ""

        # Attempt 1: warm-started K-value flash (fast path)
        if K_prev is not None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    out = flash_co2_h2o_salt_kv(
                        T=float(T_val), P_bar=float(P_val),
                        z_co2=Z_REP, m_tot=float(ms_val),
                        K_init=K_prev,
                        sol_aq_x0=sol_aq_prev,
                        sol_c_x0=sol_c_prev,
                        params=_W_params, maxiter=80,
                    )
                is_2ph      = True
                K_prev      = out["K_vals"]
                sol_aq_prev = out["sol_aq_x0"]
                sol_c_prev  = out["sol_c_x0"]
                conv        = True
            except Exception:
                out = None

        # Attempt 2: full stability + flash (slow path, 6 guesses)
        if out is None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sf = ecpa_stability_flash(
                        z_co2=Z_REP, ms=float(ms_val),
                        T=float(T_val), P=float(P_val),
                        params=_W_params,
                        co2_ref_x0=co2_ref_x0,
                        aq_ref_x0=aq_ref_x0,
                    )
                co2_ref_x0 = sf["stability"]["co2_ref_x0"]
                aq_ref_x0  = sf["stability"]["aq_ref_x0"]

                if sf.get("phase") == "single_phase":
                    err_type = "single_phase"
                    conv     = True
                else:
                    out         = sf
                    is_2ph      = True
                    K_prev      = sf["K_vals"]
                    sol_aq_prev = sf["sol_aq_x0"]
                    sol_c_prev  = sf["sol_c_x0"]
                    conv        = True
            except Exception as exc:
                err_type = f"error:{type(exc).__name__}"
                K_prev = sol_aq_prev = sol_c_prev = None

        wall_ms = (time.perf_counter() - t0) * 1000.0

        if out is not None and is_2ph:
            sol10   = out["sol"]
            Z_aq    = float(sol10[0]); Z_c     = float(sol10[3])
            x1w_v   = float(sol10[1]); epsr_v  = float(sol10[2])
            x1c_v   = float(sol10[4]); chi1w_v = float(sol10[5])
            chi1c_v = float(sol10[6])
            ms_aq_v = float(out["ms_aq"])
            x2w_v   = x1w_v * ms_aq_v * 0.018015
            x4w_v   = 1.0 - x1w_v - 2.0 * x2w_v
            x4c_v   = 1.0 - x1c_v
            beta_v  = float(out["beta"])
        else:
            sol10 = np.full(SOL_DIM, np.nan)
            Z_aq = Z_c = epsr_v = chi1w_v = chi1c_v = np.nan
            x1w_v = x4w_v = x1c_v = x4c_v = beta_v = np.nan
            ms_aq_v = np.nan

        row_data.append({
            "iT": iT, "iP": iP, "ims": ims,
            "T": T_val, "P": P_val, "ms_feed": ms_val,
            "eos_type": "eCPA",
            "converged": conv, "is_two_phase": is_2ph,
            "sol": sol10.tolist() if hasattr(sol10, "tolist") else list(sol10),
            "Z_aq": Z_aq, "Z_c": Z_c, "epsr": epsr_v,
            "chi1w": chi1w_v, "chi1c": chi1c_v,
            "x1w": x1w_v, "x4w": x4w_v,
            "x1c": x1c_v, "x4c": x4c_v,
            "ms_aq": ms_aq_v, "beta": beta_v,
            "wall_time_ms": wall_ms,
        })

    return iT, ims, row_data


# ── Main scan ──────────────────────────────────────────────────────────────────

def run_scan(n_workers=None, out_prefix="results/scan_v4"):
    sys.path.insert(0, SCRIPT_DIR)

    from ecpa.parameters import make_params
    params = make_params()

    nT  = len(T_GRID)
    nP  = len(P_GRID)
    nms = len(MS_GRID)
    total_rows   = nT * nms
    total_points = nT * nP * nms

    print(f"Scan v4 — z-free 3-D solution table")
    print(f"Grid: {nT}T × {nP}P × {nms}ms = {total_points:,} points "
          f"(z-free; z_rep={Z_REP})")
    print(f"T = {T_GRID[0]:.0f}–{T_GRID[-1]:.0f} K ({nT} pts, 1 K steps)")
    print(f"P = {P_GRID[0]:.1f}–{P_GRID[-1]:.0f} bar ({nP} pts, log-spaced)")
    print(f"ms = {MS_GRID}")

    # Allocate result arrays
    Z_aq_arr   = np.full((nT, nP, nms), np.nan)
    Z_c_arr    = np.full((nT, nP, nms), np.nan)
    epsr_arr   = np.full((nT, nP, nms), np.nan)
    chi1w_arr  = np.full((nT, nP, nms), np.nan)
    chi1c_arr  = np.full((nT, nP, nms), np.nan)
    x1w_arr    = np.full((nT, nP, nms), np.nan)
    x4w_arr    = np.full((nT, nP, nms), np.nan)
    x1c_arr    = np.full((nT, nP, nms), np.nan)
    x4c_arr    = np.full((nT, nP, nms), np.nan)
    ms_aq_arr  = np.full((nT, nP, nms), np.nan)
    beta_arr   = np.full((nT, nP, nms), np.nan)
    is2ph_arr  = np.zeros((nT, nP, nms), dtype=bool)

    all_metrics = []

    # Build tasks — sweep P ascending for warm-start continuity
    P_sorted = np.sort(P_GRID)
    P_order  = np.argsort(P_GRID)   # sorted-P index → original P index

    tasks = []
    for iT, T_val in enumerate(T_GRID):
        for ims, ms_val in enumerate(MS_GRID):
            tasks.append((iT, ims, T_val, ms_val, P_sorted))

    assert len(tasks) == total_rows

    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    n_workers = min(n_workers, total_rows)
    print(f"\nLaunching {n_workers} workers for {total_rows:,} row-tasks …")

    ctx = mp.get_context("spawn")
    t_start = time.perf_counter()
    rows_done = 0

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx,
                              initializer=_worker_init,
                              initargs=(params,)) as pool:
        futures = {pool.submit(_scan_row_worker, task): task for task in tasks}

        for fut in as_completed(futures):
            try:
                iT, ims, row_data = fut.result()
            except Exception as exc:
                task = futures[fut]
                print(f"  [ERROR] iT={task[0]} ims={task[1]}: {exc}")
                rows_done += 1
                continue

            for rec in row_data:
                iP_sorted = rec["iP"]
                iP_orig   = P_order[iP_sorted]

                sol10 = np.array(rec["sol"])
                Z_aq_arr  [iT, iP_orig, ims] = rec["Z_aq"]
                Z_c_arr   [iT, iP_orig, ims] = rec["Z_c"]
                epsr_arr  [iT, iP_orig, ims] = rec["epsr"]
                chi1w_arr [iT, iP_orig, ims] = rec["chi1w"]
                chi1c_arr [iT, iP_orig, ims] = rec["chi1c"]
                x1w_arr   [iT, iP_orig, ims] = rec["x1w"]
                x4w_arr   [iT, iP_orig, ims] = rec["x4w"]
                x1c_arr   [iT, iP_orig, ims] = rec["x1c"]
                x4c_arr   [iT, iP_orig, ims] = rec["x4c"]
                ms_aq_arr [iT, iP_orig, ims] = rec["ms_aq"]
                beta_arr  [iT, iP_orig, ims] = rec["beta"]
                is2ph_arr [iT, iP_orig, ims] = bool(rec["is_two_phase"])

                all_metrics.append({
                    "T": rec["T"], "P": rec["P"], "ms_feed": rec["ms_feed"],
                    "eos_type": rec["eos_type"],
                    "converged": rec["converged"],
                    "is_two_phase": rec["is_two_phase"],
                    "wall_time_ms": rec["wall_time_ms"],
                })

            rows_done += 1
            if rows_done % max(1, total_rows // 50) == 0 or rows_done == total_rows:
                elapsed = time.perf_counter() - t_start
                rate = rows_done / elapsed if elapsed > 0 else 1.0
                eta  = (total_rows - rows_done) / rate
                print(f"  {rows_done:5d}/{total_rows} rows  "
                      f"({100*rows_done/total_rows:.1f}%)  "
                      f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s", flush=True)

    elapsed_total = time.perf_counter() - t_start
    print(f"\nScan finished in {elapsed_total:.1f} s  "
          f"({elapsed_total/60:.1f} min)")

    n_two_phase = int(is2ph_arr.sum())
    n_total     = is2ph_arr.size
    print(f"Two-phase: {n_two_phase:,} / {n_total:,} "
          f"({100*n_two_phase/n_total:.1f}%)")
    for ims, ms_val in enumerate(MS_GRID):
        frac = float(is2ph_arr[:, :, ims].mean())
        print(f"  ms={ms_val:.4g}:  {100*frac:.1f}% two-phase")

    # Save NPZ — 3-D arrays, no z dimension
    npz_path = Path(f"{out_prefix}_table.npz")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        T_grid=T_GRID, P_grid=P_GRID, ms_grid=MS_GRID,
        # no z_grid — ScanTableWarmStart detects 3-D format by its absence
        Z_aq=Z_aq_arr, Z_c=Z_c_arr,
        epsr=epsr_arr,
        chi1w=chi1w_arr, chi1c=chi1c_arr,
        x1w=x1w_arr, x4w=x4w_arr,
        x1c=x1c_arr, x4c=x4c_arr,
        ms_aq=ms_aq_arr, beta=beta_arr,
        is_two_phase=is2ph_arr,
    )
    print(f"Saved → {npz_path}  ({npz_path.stat().st_size/1e6:.1f} MB)")

    # Save metrics parquet
    pq_path = Path(f"{out_prefix}_metrics.parquet")
    pd.DataFrame(all_metrics).to_parquet(pq_path, index=False)
    print(f"Saved → {pq_path}  ({len(all_metrics):,} rows)")

    return npz_path, pq_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build eCPA/CPA solution table v4 (3-D, z-free)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: cpu_count-1)")
    parser.add_argument("--out", type=str, default="results/scan_v4",
                        help="Output prefix (default: results/scan_v4)")
    args = parser.parse_args()

    run_scan(n_workers=args.workers, out_prefix=args.out)
