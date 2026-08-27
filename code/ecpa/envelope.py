"""
Phase-envelope computation: eCPA (from scan bisection) and CPA (parallel).
"""
import multiprocessing as mp
import os
import pickle
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from .flash import get_flash_fn
from .guess_table import make_guess_fn
from .scan import _cpu_count


# ── Envelope worker state ─────────────────────────────────────────────────────
# Populated once per spawned process by _envelope_worker_init.

_EW_flash_fn       = None
_EW_guess_table_fn = None
_EW_params         = None
_EW_n_bisect       = 25


def _envelope_worker_init(flash_algo, params, CPA_GROUPS, CPA_TEMPS, n_bisect):
    """Called once per worker process to set up shared state."""
    global _EW_flash_fn, _EW_guess_table_fn, _EW_params, _EW_n_bisect
    _EW_flash_fn       = get_flash_fn(flash_algo)
    _EW_guess_table_fn = make_guess_fn(CPA_GROUPS, CPA_TEMPS)
    _EW_params         = params
    _EW_n_bisect       = n_bisect


def _envelope_is_two_phase(T, P, z_co2, ms):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            _EW_flash_fn(T=float(T), P_bar=float(P),
                         z_co2=float(z_co2), m_tot=float(ms),
                         guess_table_fn=_EW_guess_table_fn, params=_EW_params)
        return True
    except Exception:
        return False


def _envelope_bisect(T, P_two, P_one, z_co2, ms):
    a, b = P_two, P_one
    for _ in range(_EW_n_bisect):
        mid = 0.5 * (a + b)
        if _envelope_is_two_phase(T, mid, z_co2, ms):
            a = mid
        else:
            b = mid
        if abs(b - a) < 0.01:
            break
    return 0.5 * (a + b)


def _envelope_worker_task(task):
    """
    Bisect both boundaries for one (z_co2, ms, T) point.

    task = (z_co2, ms, T, P_conv_min, P_conv_max, P_nc_below, P_nc_above)

    P_nc_below = nan  →  no non-converged point below → use P_conv_min directly
    P_nc_above = nan  →  no non-converged point above → P_hi = nan
    """
    z_co2, ms, T, P_conv_min, P_conv_max, P_nc_below, P_nc_above = task

    n_bisect_calls = 0

    if np.isfinite(P_nc_below):
        P_lo = _envelope_bisect(T, P_conv_min, P_nc_below, z_co2, ms)
        n_bisect_calls += _EW_n_bisect
    else:
        P_lo = P_conv_min

    if np.isfinite(P_nc_above):
        P_hi = _envelope_bisect(T, P_conv_max, P_nc_above, z_co2, ms)
        n_bisect_calls += _EW_n_bisect
    else:
        P_hi = float("nan")

    return dict(z_co2=z_co2, ms=ms, T=T, P_lo=P_lo, P_hi=P_hi,
                n_bisect_calls=n_bisect_calls)


# ── Public API ────────────────────────────────────────────────────────────────

def find_envelope_from_scan(
    df_results,
    z_co2_values,
    ms_values,
    T_values,
    guess_table_fn,
    params,
    CPA_GROUPS,
    CPA_TEMPS,
    flash_algo: str = "ssi",
    n_bisect: int = 25,
    cache_file: str = "results/envelope_ecpa.pkl",
    force_recompute: bool = False,
    n_workers: int = None,
):
    """
    Extract eCPA phase-boundary pressures from a scan DataFrame using bisection.

    For each (z_co2, ms, T), finds P_lo (bubble) and P_hi (dew/miscibility)
    by bisecting between converged and non-converged scan points.

    Parameters
    ----------
    flash_algo : 'ssi' or 'brent' — which flash to use for bisection calls
    n_workers  : number of parallel workers (None = auto-detect; 1 = sequential)

    Returns
    -------
    envelopes_df : dict[z_key -> dict[ms -> DataFrame(T, P_lo, P_hi)]]
    """
    os.makedirs(os.path.dirname(cache_file) or ".", exist_ok=True)

    if os.path.exists(cache_file) and not force_recompute:
        with open(cache_file, "rb") as f:
            envelopes = pickle.load(f)
        print(f"Resuming eCPA envelope from {cache_file}")
    else:
        if force_recompute and os.path.exists(cache_file):
            os.remove(cache_file)
            print(f"force_recompute=True — deleted {cache_file}")
        envelopes = {}

    if n_workers is None:
        n_workers = _cpu_count()

    n_z   = len(z_co2_values)
    n_ms  = len(ms_values)
    n_T   = len(T_values)
    total = n_z * n_ms * n_T
    t_total = time.time()

    # Single-phase error types that indicate the upper boundary of the two-phase
    # window — used to identify P_nc_above for the high-pressure bisection.
    _above_types = {
        "no_sign_change", "ssi_no_converge",
        "single_phase_stable", "single_phase_gas", "single_phase_liquid",
        "runtime_other",
    }

    # ── Pre-extract bisection tasks from df_results ───────────────────────────
    # For each uncached (z_co2, ms, T) point, determine:
    #   • P_nc_below: highest non-converged P below the two-phase window
    #   • P_nc_above: lowest single-phase P above the two-phase window
    # Points with no scan data or no converged scan points get NaN immediately.

    all_tasks    = []   # (z_co2, ms, T, P_conv_min, P_conv_max, P_nc_below, P_nc_above)
    direct_nans  = []   # (z_key, ms, T) points to store NaN without bisection

    for z_co2 in z_co2_values:
        z_key = round(z_co2, 2)
        sub   = df_results[
            np.isclose(df_results["z_co2"], z_key) & True
        ].copy()

        for ms in ms_values:
            existing_T = set(envelopes.get(z_key, {}).get(ms, {}).keys())
            sub_ms     = sub[sub["ms"] == ms]

            for T in T_values:
                if T in existing_T:
                    continue

                row_T = sub_ms[sub_ms["T"] == T].sort_values("P")

                if len(row_T) == 0:
                    direct_nans.append((z_key, ms, T, "no scan data"))
                    continue

                conv = row_T[row_T["error_type"] == "none"]
                if len(conv) == 0:
                    direct_nans.append((z_key, ms, T, "no converged points"))
                    continue

                P_conv_min = conv["P"].min()
                P_conv_max = conv["P"].max()

                below = row_T[row_T["P"] < P_conv_min]
                P_nc_below = float(below["P"].max()) if len(below) > 0 else float("nan")

                above = row_T[
                    (row_T["P"] > P_conv_max) &
                    (row_T["error_type"].isin(_above_types))
                ]
                P_nc_above = float(above["P"].min()) if len(above) > 0 else float("nan")

                all_tasks.append((
                    float(z_co2), float(ms), float(T),
                    float(P_conv_min), float(P_conv_max),
                    P_nc_below, P_nc_above,
                ))

    # ── Apply direct-NaN records ──────────────────────────────────────────────
    for z_key, ms, T, reason in direct_nans:
        envelopes.setdefault(z_key, {}).setdefault(ms, {})[T] = dict(
            T=T, P_lo=np.nan, P_hi=np.nan)

    n_cached   = total - len(all_tasks) - len(direct_nans)
    n_nan_only = len(direct_nans)
    print(f"\neCPA envelope: {total} points total | "
          f"{n_cached} cached | {n_nan_only} NaN (no data) | "
          f"{len(all_tasks)} to bisect | {n_workers} workers")

    # ── Parallel bisection ────────────────────────────────────────────────────
    if all_tasks:
        initargs = (flash_algo, params, CPA_GROUPS, CPA_TEMPS, n_bisect)
        n_use    = min(n_workers, len(all_tasks))
        completed = 0
        save_interval = max(1, min(20, len(all_tasks) // 10))
        unsaved = 0

        if n_use > 1:
            ctx = mp.get_context("spawn")
            pool_kw = dict(
                max_workers=n_use,
                mp_context=ctx,
                initializer=_envelope_worker_init,
                initargs=initargs,
            )
            print(f"Launching {n_use} workers ...\n")
            print(f"  {'z_co2':>6}  {'ms':>4}  {'T (K)':>6}  "
                  f"{'P_lo (bar)':>12}  {'P_hi (bar)':>12}  "
                  f"{'bisect':>6}  {'elapsed':>8}  {'progress':>10}")

            t_start = time.time()
            with ProcessPoolExecutor(**pool_kw) as pool:
                futures = {pool.submit(_envelope_worker_task, task): task
                           for task in all_tasks}
                for future in as_completed(futures):
                    res      = future.result()
                    z_key    = round(res["z_co2"], 2)
                    ms_res   = res["ms"]
                    T_res    = res["T"]
                    envelopes.setdefault(z_key, {}).setdefault(ms_res, {})[T_res] = dict(
                        T=T_res, P_lo=res["P_lo"], P_hi=res["P_hi"])
                    completed += 1
                    unsaved   += 1

                    if unsaved >= save_interval:
                        with open(cache_file, "wb") as f:
                            pickle.dump(envelopes, f)
                        unsaved = 0

                    elapsed  = time.time() - t_start
                    P_hi_str = (f"{res['P_hi']:12.3f}" if np.isfinite(res["P_hi"])
                                else f"{'not detected':>12}")
                    pct = 100 * completed / len(all_tasks)
                    print(f"  {res['z_co2']:6.2f}  {ms_res:4.1f}  {T_res:6.0f}  "
                          f"{res['P_lo']:12.3f}  {P_hi_str}  "
                          f"{res['n_bisect_calls']:6d}  {elapsed:7.1f}s  "
                          f"[{pct:5.1f}%]")

            # Final cache save
            with open(cache_file, "wb") as f:
                pickle.dump(envelopes, f)

        else:
            # Sequential fallback (n_workers == 1)
            flash_fn = get_flash_fn(flash_algo)

            # Temporarily inject globals so the worker functions work sequentially
            global _EW_flash_fn, _EW_guess_table_fn, _EW_params, _EW_n_bisect
            _EW_flash_fn       = flash_fn
            _EW_guess_table_fn = guess_table_fn
            _EW_params         = params
            _EW_n_bisect       = n_bisect

            print(f"  {'z_co2':>6}  {'ms':>4}  {'T (K)':>6}  "
                  f"{'P_lo (bar)':>12}  {'P_hi (bar)':>12}  "
                  f"{'bisect':>6}  {'time':>8}  {'progress':>10}")

            for task in all_tasks:
                t_T  = time.time()
                res  = _envelope_worker_task(task)
                z_key = round(res["z_co2"], 2)
                ms_res = res["ms"]
                T_res  = res["T"]
                envelopes.setdefault(z_key, {}).setdefault(ms_res, {})[T_res] = dict(
                    T=T_res, P_lo=res["P_lo"], P_hi=res["P_hi"])
                completed += 1
                unsaved   += 1

                if unsaved >= save_interval:
                    with open(cache_file, "wb") as f:
                        pickle.dump(envelopes, f)
                    unsaved = 0

                dt       = time.time() - t_T
                P_hi_str = (f"{res['P_hi']:12.3f}" if np.isfinite(res["P_hi"])
                            else f"{'not detected':>12}")
                pct = 100 * completed / len(all_tasks)
                print(f"  {res['z_co2']:6.2f}  {ms_res:4.1f}  {T_res:6.0f}  "
                      f"{res['P_lo']:12.3f}  {P_hi_str}  "
                      f"{res['n_bisect_calls']:6d}  {dt:7.2f}s  "
                      f"[{pct:5.1f}%]")

            # Reset globals
            _EW_flash_fn = _EW_guess_table_fn = _EW_params = None
            _EW_n_bisect = 25

            with open(cache_file, "wb") as f:
                pickle.dump(envelopes, f)

    # ── Convert inner dicts to DataFrames ─────────────────────────────────────
    envelopes_df = {}
    for z_key, ms_dict in envelopes.items():
        envelopes_df[z_key] = {}
        for ms, T_dict in ms_dict.items():
            records = list(T_dict.values())
            envelopes_df[z_key][ms] = (pd.DataFrame(records)
                                        .sort_values("T")
                                        .reset_index(drop=True))

    print(f"\neCPA envelope complete in {time.time()-t_total:.1f}s")
    return envelopes_df


def build_cpa2_envelope(
    T_values,
    z_co2_values,
    params_cpa2,
    n_workers: int = 48,
    cache_file: str = "results/envelope_cpa2.pkl",
    force_recompute: bool = False,
):
    """
    Build CPA salt-free phase envelopes in parallel using cpa_worker.

    Returns
    -------
    envelopes_df : dict[z_key -> DataFrame(T, P_lo, P_hi)]
    """
    import cpa_worker  # top-level module

    os.makedirs(os.path.dirname(cache_file) or ".", exist_ok=True)

    if os.path.exists(cache_file) and not force_recompute:
        with open(cache_file, "rb") as f:
            envelopes = pickle.load(f)
        n_cached = sum(len(v) for v in envelopes.values())
        print(f"Resuming CPA envelope from {cache_file}  ({n_cached} pts cached)")
    else:
        if force_recompute and os.path.exists(cache_file):
            os.remove(cache_file)
            print(f"force_recompute=True — deleted {cache_file}")
        envelopes = {}

    P_scan    = list(np.logspace(0, np.log10(1500), 20))
    all_tasks = []
    for z_co2 in z_co2_values:
        z_key  = round(z_co2, 2)
        done_T = set(envelopes.get(z_key, {}).keys())
        for T in T_values:
            if T not in done_T:
                all_tasks.append((float(T), float(z_co2), P_scan))

    total    = len(T_values) * len(z_co2_values)
    n_cached = total - len(all_tasks)
    print(f"Total: {total}  cached: {n_cached}  remaining: {len(all_tasks)}")

    if all_tasks:
        n_use   = min(n_workers, len(all_tasks))
        t_start = time.time()
        completed = 0

        print(f"Launching {n_use} workers ...\n")
        with ProcessPoolExecutor(max_workers=n_use) as executor:
            futures = {executor.submit(cpa_worker.compute_one_point, task): task
                       for task in all_tasks}
            for future in as_completed(futures):
                result = future.result()
                T      = result["T"]
                z_co2  = result["z_co2"]
                z_key  = round(z_co2, 2)
                if z_key not in envelopes:
                    envelopes[z_key] = {}
                envelopes[z_key][T] = result
                with open(cache_file, "wb") as f:
                    pickle.dump(envelopes, f)
                completed += 1
                elapsed  = time.time() - t_start
                P_hi     = result["P_hi"]
                P_hi_str = f"{P_hi:12.3f}" if np.isfinite(P_hi) else f"{'> scan range':>12s}"
                print(f"  z={z_co2:.2f}  T={T:.0f}K  "
                      f"P_lo={result['P_lo']:10.3f}  P_hi={P_hi_str}  "
                      f"{completed}/{len(all_tasks)}  {elapsed:.1f}s")

        print(f"\nDone in {time.time()-t_start:.1f}s")

    envelopes_df = {}
    for z_key, T_dict in envelopes.items():
        records = list(T_dict.values())
        envelopes_df[z_key] = (pd.DataFrame(records)
                                .sort_values("T")
                                .reset_index(drop=True))
    return envelopes_df
