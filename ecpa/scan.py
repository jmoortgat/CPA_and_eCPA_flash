"""
LEGACY / OLD INFRASTRUCTURE — do not use for new scans.

This module uses flash_co2_h2o_salt_ssi (outer ms_aq loop) which has been
superseded by flash_co2_h2o_salt_kv + ecpa_stability_flash.  For new grid
scans, use _build_scan_table_v2.py (hierarchical stability + KV flash).

run_flash_scan — grid scan over (T, P, z_co2, ms) using either flash algorithm.

Results are cached to a Parquet file and reloaded on subsequent calls.
Parallel execution is supported via n_workers > 1.
"""
import multiprocessing as mp
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from .flash import get_flash_fn, _cpa2_phase_check
from .guess_table import get_table_p_range, make_guess_fn
from .constants import Mw


def _cpu_count() -> int:
    """
    Return the number of usable CPUs.

    Uses os.sched_getaffinity on Linux (respects cgroups / taskset / SLURM),
    falls back to os.cpu_count() on macOS and other platforms.
    Subtracts one to leave a core free for the main process.
    """
    try:
        n = len(os.sched_getaffinity(0))   # Linux only
    except AttributeError:
        n = os.cpu_count() or 1             # macOS / Windows
    return max(1, n - 1)


# ── Parallel worker state ──────────────────────────────────────────────────────
# These module-level globals are populated by the worker initializer in each
# spawned process.  They are never set in the main process.

_W_flash_fn           = None
_W_flash_extra_kw     = {}
_W_params             = None
_W_guess_table_fn     = None
_W_use_cpa2           = False
_W_use_stab           = False
_W_flash_algo         = "ssi"
_W_ssi_maxiter        = 200
_W_P_min_tab_map      = {}
_W_P_max_tab_map      = {}


def _cpa2_label(z_co2: float, T: float, P: float, params,
                fallback: str) -> str:
    """
    Run a fast CPA2 salt-free phase check and return a specific single-phase
    label if it confirms single-phase, otherwise return `fallback`.

    CPA2 is salt-free; salt only shrinks the two-phase window, so if CPA2
    says single-phase the eCPA (with salt) system is also single-phase.
    """
    hint = _cpa2_phase_check(float(z_co2), float(T), float(P), params)
    if hint == "single_phase_gas":
        return "single_phase_gas"
    if hint == "single_phase_liquid":
        return "single_phase_liquid"
    return fallback   # CPA2 uncertain → keep the algorithmic failure label


def _scan_worker_init(flash_algo, ssi_maxiter, params,
                      CPA_GROUPS, CPA_TEMPS,
                      use_cpa2_precheck, use_stability_check,
                      P_min_tab_map, P_max_tab_map):
    """Called once per worker process to set up shared state."""
    global _W_flash_fn, _W_flash_extra_kw, _W_params
    global _W_guess_table_fn, _W_use_cpa2, _W_use_stab
    global _W_flash_algo, _W_ssi_maxiter, _W_P_min_tab_map, _W_P_max_tab_map

    _W_flash_algo     = flash_algo
    _W_flash_fn       = get_flash_fn(flash_algo)
    _W_flash_extra_kw = {"maxiter_ms": ssi_maxiter} if flash_algo == "ssi" else {}
    _W_params         = params
    _W_guess_table_fn = make_guess_fn(CPA_GROUPS, CPA_TEMPS)
    _W_use_cpa2       = use_cpa2_precheck
    _W_use_stab       = use_stability_check
    _W_ssi_maxiter    = ssi_maxiter
    _W_P_min_tab_map  = P_min_tab_map
    _W_P_max_tab_map  = P_max_tab_map


def _scan_worker_task(task):
    """Process one (z_co2, T, P, ms) grid point and return a result record."""
    z_co2, T_i, P_i, ms_i = task

    record = dict(
        z_co2=round(z_co2, 2), T=T_i, P=round(P_i, 2), ms=ms_i,
        flash_algo=_W_flash_algo,
        converged=False, beta=np.nan,
        ms_aq=np.nan, Z_aq=np.nan, Z_c=np.nan,
        resnorm=np.nan, error="", error_type="",
    )

    # Filter 1: outside table P range
    P_min_tab = _W_P_min_tab_map.get(T_i, -np.inf)
    P_max_tab = _W_P_max_tab_map.get(T_i,  np.inf)
    if not (P_min_tab - 1e-6 <= P_i <= P_max_tab + 1e-6):
        record["error"]      = "outside table P range"
        record["error_type"] = "out_of_range"
        return record

    # Filter 2: salting-out feasibility
    x1w_approx = 1.0 / (1.0 + 2.0 * ms_i * Mw)
    x2w_approx = x1w_approx * ms_i * Mw
    x4w_approx = 1.0 - x1w_approx - 2 * x2w_approx
    if x4w_approx < -0.05:
        record["error"]      = f"x4w<0 ({x4w_approx:.4f}) — salting-out"
        record["error_type"] = "salting_out"
        return record

    # Filter 3: CPA2 single-phase pre-check (optional)
    if _W_use_cpa2:
        hint = _cpa2_phase_check(float(T_i), float(P_i), float(z_co2), _W_params)
        if hint in ("single_phase_liquid", "single_phase_gas"):
            record["error"]      = f"CPA2 pre-check: {hint}"
            record["error_type"] = "out_of_range"
            return record

    # Filter 4: eCPA stability pre-check (optional).
    # NOTE: we do NOT skip the flash if stability says stable.  The stability
    # SSI can produce false positives at intermediate z_co2 where the reference
    # composition is outside the intended EOS regime.  Instead we run the flash
    # regardless and use the stability result only to label flash failures:
    # a flash that fails AND whose stability check says single-phase is labelled
    # "single_phase_stable" (high confidence); one that fails with uncertain
    # stability is labelled "no_sign_change" / "ssi_no_converge".
    stability_stable = False
    if _W_use_stab:
        try:
            from .stability import ecpa_stability
            stab = ecpa_stability(float(z_co2), float(ms_i),
                                  float(T_i), float(P_i), _W_params)
            stability_stable = bool(stab["stable"])
        except Exception:
            pass  # stability check failed → flash decides

    # Flash
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out = _W_flash_fn(
                T=float(T_i), P_bar=float(P_i),
                z_co2=float(z_co2), m_tot=ms_i,
                guess_table_fn=_W_guess_table_fn, params=_W_params,
                **_W_flash_extra_kw,
            )
        record["converged"]  = True
        record["error_type"] = "none"
        record["beta"]       = out["beta"]
        record["ms_aq"]      = out["ms_aq"]
        record["Z_aq"]       = out["Z_aq"]
        record["Z_c"]        = out["Z_c"]
        return record

    except RuntimeError as exc:
        msg = str(exc)
        record["error"] = msg[:80]

        # For Brent no_sign_change, try SSI as a fallback — it uses a different
        # root-finding strategy that can succeed where the bracketing scan fails.
        if "sign change" in msg and _W_flash_algo == "brent":
            try:
                from .flash import flash_co2_h2o_salt_ssi
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    out = flash_co2_h2o_salt_ssi(
                        T=float(T_i), P_bar=float(P_i),
                        z_co2=float(z_co2), m_tot=ms_i,
                        guess_table_fn=_W_guess_table_fn, params=_W_params,
                        maxiter_ms=_W_ssi_maxiter,
                    )
                record["converged"]  = True
                record["error_type"] = "none"
                record["error"]      = ""
                record["beta"]       = out["beta"]
                record["ms_aq"]      = out["ms_aq"]
                record["Z_aq"]       = out["Z_aq"]
                record["Z_c"]        = out["Z_c"]
                return record
            except Exception:
                pass  # SSI also failed

            # Both Brent and SSI failed.  Use CPA2 (salt-free, fast) to decide:
            # if CPA2 says single-phase, salt only tightens the window, so eCPA
            # is also single-phase.  stability_stable upgrades the fallback label.
            fallback = "single_phase_stable" if stability_stable else "no_sign_change"
            record["error_type"] = _cpa2_label(z_co2, T_i, P_i, _W_params, fallback)

        elif "ELV likely failing" in msg:
            record["error_type"] = "elv_solver"
        elif "cache is empty" in msg:
            record["error_type"] = "cache_empty"
        elif "did not converge" in msg:
            fallback = "single_phase_stable" if stability_stable else "ssi_no_converge"
            record["error_type"] = _cpa2_label(z_co2, T_i, P_i, _W_params, fallback)
        else:
            record["error_type"] = "runtime_other"

    except Exception as exc:
        record["error"]      = f"{type(exc).__name__}: {str(exc)[:60]}"
        record["error_type"] = "exception"

    return record


# ── Public API ─────────────────────────────────────────────────────────────────

def run_flash_scan(
    guess_table_fn,
    params,
    CPA_GROUPS,
    CPA_TEMPS,
    T_values=None,
    P_values=None,
    ms_values=None,
    z_co2_values=None,
    flash_algo: str = "ssi",
    ssi_maxiter: int = 40,
    scan_file: str = "results/scan_results.parquet",
    use_cpa2_precheck: bool = False,
    use_stability_check: bool = False,
    force_recompute: bool = False,
    n_workers: int = None,
    executor: str = "process",
):
    """
    Run a 4-D flash scan and cache results to Parquet.

    Parameters
    ----------
    guess_table_fn  : callable  T, P_bar → np.ndarray
    params          : dict
    CPA_GROUPS, CPA_TEMPS : output of load_cpa_guess_table()
    T_values        : array-like of temperatures [K]   (default: arange(283,534,25))
    P_values        : array-like of pressures [bar]    (default: logspace(0,log10(1500),20))
    ms_values       : list of salt molalities [mol/kg] (default: [0.5, 1.0, 2.0])
    z_co2_values    : list of feed CO₂ mole fractions  (default: linspace(0.1,0.9,5))
    flash_algo      : 'ssi' or 'brent'
    ssi_maxiter     : max outer iterations for SSI flash, and for the SSI fallback
                      when Brent gives no_sign_change
    scan_file       : path for Parquet cache
    use_cpa2_precheck   : if True, skip points identified as single-phase by CPA2
    use_stability_check : if True, run ecpa_stability to enrich failure labels.
                          The stability result is used ONLY to label flash failures
                          (not to skip the flash), so false positives cannot cause
                          two-phase points to be missed.
    force_recompute : if True, ignore existing cache and recompute
    n_workers       : number of parallel workers (None = auto-detect from CPU count;
                      1 = sequential; >1 = that many workers)
    executor        : 'process' (default, safest for CPU-bound work) or 'thread'
                      (shares memory, no spawn overhead; may help if scipy releases
                      the GIL enough, but risks BLAS oversubscription)

    Returns
    -------
    df_results : pd.DataFrame  (one row per grid point)
    """
    os.makedirs(os.path.dirname(scan_file) or ".", exist_ok=True)

    # ── Load cache ─────────────────────────────────────────────────────────────
    if os.path.exists(scan_file) and not force_recompute:
        df_results = pd.read_parquet(scan_file)
        total  = len(df_results)
        n_conv = df_results["converged"].sum()
        print(f"Loaded cached scan: {scan_file}")
        print(f"  {total} records | {n_conv} converged ({100*n_conv/total:.1f}%)")
        return df_results

    # ── Grid defaults ──────────────────────────────────────────────────────────
    if T_values     is None: T_values     = np.arange(283, 534, 25)
    if P_values     is None: P_values     = np.logspace(0, np.log10(1500), 20)
    if ms_values    is None: ms_values    = [0.5, 1.0, 2.0]
    if z_co2_values is None: z_co2_values = np.linspace(0.1, 0.9, 5)

    if n_workers is None:
        n_workers = _cpu_count()

    total = len(z_co2_values) * len(T_values) * len(P_values) * len(ms_values)
    print(f"Running {total} flash calculations "
          f"(algo='{flash_algo}', n_workers={n_workers}, executor='{executor}') ...\n")

    # Pre-compute table P range for each T (avoids repeated lookups in workers)
    P_min_tab_map = {}
    P_max_tab_map = {}
    for T_i in T_values:
        P_min_tab_map[float(T_i)], P_max_tab_map[float(T_i)] = get_table_p_range(
            float(T_i), CPA_GROUPS, CPA_TEMPS)

    # Build flat task list in the same order as the original sequential loop
    all_tasks = [
        (float(z_co2), float(T_i), float(P_i), float(ms_i))
        for z_co2 in z_co2_values
        for T_i   in T_values
        for ms_i  in ms_values
        for P_i   in P_values
    ]

    t0 = time.time()

    if n_workers > 1:
        records = _run_parallel(
            all_tasks, total, t0,
            flash_algo, ssi_maxiter, params, CPA_GROUPS, CPA_TEMPS,
            use_cpa2_precheck, use_stability_check,
            P_min_tab_map, P_max_tab_map,
            n_workers, executor,
        )
    else:
        flash_fn       = get_flash_fn(flash_algo)
        flash_extra_kw = {"maxiter_ms": ssi_maxiter} if flash_algo == "ssi" else {}
        records = _run_sequential(
            all_tasks, total, t0,
            flash_algo, flash_fn, flash_extra_kw, ssi_maxiter, params,
            guess_table_fn, use_cpa2_precheck, use_stability_check,
            P_min_tab_map, P_max_tab_map,
        )

    elapsed = time.time() - t0
    print(f"\nFinished {total} calculations in {elapsed:.1f}s")

    df_results = pd.DataFrame(records)
    df_results.to_parquet(scan_file, index=False)
    print(f"Saved → {scan_file}")
    return df_results


def _run_parallel(all_tasks, total, t0,
                  flash_algo, ssi_maxiter, params, CPA_GROUPS, CPA_TEMPS,
                  use_cpa2_precheck, use_stability_check,
                  P_min_tab_map, P_max_tab_map, n_workers, executor_type):
    initargs = (
        flash_algo, ssi_maxiter, params,
        CPA_GROUPS, CPA_TEMPS,
        use_cpa2_precheck, use_stability_check,
        P_min_tab_map, P_max_tab_map,
    )
    records   = [None] * len(all_tasks)
    task_idx  = {id(task): i for i, task in enumerate(all_tasks)}
    completed = 0

    if executor_type == "thread":
        # ThreadPoolExecutor shares memory with the main process — no spawn
        # overhead, and scipy/numpy release the GIL during computation.
        Executor = ThreadPoolExecutor
        ctx_kw   = {}
        init_kw  = {"initializer": _scan_worker_init, "initargs": initargs}
    else:
        # ProcessPoolExecutor with explicit 'spawn' context works identically
        # on macOS (default spawn) and Linux (default fork), and avoids the
        # deadlock risk from forking after numpy/BLAS has started threads.
        Executor = ProcessPoolExecutor
        ctx_kw   = {"mp_context": mp.get_context("spawn")}
        init_kw  = {"initializer": _scan_worker_init, "initargs": initargs}

    with Executor(max_workers=n_workers, **ctx_kw, **init_kw) as pool:
        futures = {pool.submit(_scan_worker_task, task): task
                   for task in all_tasks}
        for future in as_completed(futures):
            record = future.result()
            task   = futures[future]
            records[task_idx[id(task)]] = record
            completed += 1
            _maybe_print_progress(completed, total, t0)

    return records


def _run_sequential(all_tasks, total, t0,
                    flash_algo, flash_fn, flash_extra_kw, ssi_maxiter, params,
                    guess_table_fn, use_cpa2_precheck, use_stability_check,
                    P_min_tab_map, P_max_tab_map):
    records = []
    done    = 0

    for (z_co2, T_i, P_i, ms_i) in all_tasks:
        done += 1
        record = dict(
            z_co2=round(z_co2, 2), T=T_i, P=round(P_i, 2), ms=ms_i,
            flash_algo=flash_algo,
            converged=False, beta=np.nan,
            ms_aq=np.nan, Z_aq=np.nan, Z_c=np.nan,
            resnorm=np.nan, error="", error_type="",
        )

        # Filter 1: outside table P range
        P_min_tab = P_min_tab_map.get(T_i, -np.inf)
        P_max_tab = P_max_tab_map.get(T_i,  np.inf)
        if not (P_min_tab - 1e-6 <= P_i <= P_max_tab + 1e-6):
            record["error"]      = "outside table P range"
            record["error_type"] = "out_of_range"
            records.append(record)
            _maybe_print_progress(done, total, t0)
            continue

        # Filter 2: salting-out feasibility
        x1w_approx = 1.0 / (1.0 + 2.0 * ms_i * Mw)
        x2w_approx = x1w_approx * ms_i * Mw
        x4w_approx = 1.0 - x1w_approx - 2 * x2w_approx
        if x4w_approx < -0.05:
            record["error"]      = f"x4w<0 ({x4w_approx:.4f}) — salting-out"
            record["error_type"] = "salting_out"
            records.append(record)
            _maybe_print_progress(done, total, t0)
            continue

        # Filter 3: CPA2 single-phase pre-check (optional)
        if use_cpa2_precheck:
            hint = _cpa2_phase_check(float(T_i), float(P_i), float(z_co2), params)
            if hint in ("single_phase_liquid", "single_phase_gas"):
                record["error"]      = f"CPA2 pre-check: {hint}"
                record["error_type"] = "out_of_range"
                records.append(record)
                _maybe_print_progress(done, total, t0)
                continue

        # Filter 4: stability pre-check — sets label only, never skips the flash.
        stability_stable = False
        if use_stability_check:
            try:
                from .stability import ecpa_stability
                stab = ecpa_stability(float(z_co2), float(ms_i),
                                      float(T_i), float(P_i), params)
                stability_stable = bool(stab["stable"])
            except Exception:
                pass

        # Flash
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out = flash_fn(
                    T=float(T_i), P_bar=float(P_i),
                    z_co2=float(z_co2), m_tot=ms_i,
                    guess_table_fn=guess_table_fn, params=params,
                    **flash_extra_kw,
                )
            record["converged"]  = True
            record["error_type"] = "none"
            record["beta"]       = out["beta"]
            record["ms_aq"]      = out["ms_aq"]
            record["Z_aq"]       = out["Z_aq"]
            record["Z_c"]        = out["Z_c"]

        except RuntimeError as exc:
            msg = str(exc)
            record["error"] = msg[:80]

            if "sign change" in msg and flash_algo == "brent":
                # Brent bracketing failed — try SSI as fallback
                try:
                    from .flash import flash_co2_h2o_salt_ssi
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        out = flash_co2_h2o_salt_ssi(
                            T=float(T_i), P_bar=float(P_i),
                            z_co2=float(z_co2), m_tot=ms_i,
                            guess_table_fn=guess_table_fn, params=params,
                            maxiter_ms=ssi_maxiter,
                        )
                    record["converged"]  = True
                    record["error_type"] = "none"
                    record["error"]      = ""
                    record["beta"]       = out["beta"]
                    record["ms_aq"]      = out["ms_aq"]
                    record["Z_aq"]       = out["Z_aq"]
                    record["Z_c"]        = out["Z_c"]
                except Exception:
                    fallback = "single_phase_stable" if stability_stable else "no_sign_change"
                    record["error_type"] = _cpa2_label(z_co2, T_i, P_i, params, fallback)

            elif "ELV likely failing" in msg:
                record["error_type"] = "elv_solver"
            elif "cache is empty" in msg:
                record["error_type"] = "cache_empty"
            elif "did not converge" in msg:
                fallback = "single_phase_stable" if stability_stable else "ssi_no_converge"
                record["error_type"] = _cpa2_label(z_co2, T_i, P_i, params, fallback)
            else:
                record["error_type"] = "runtime_other"

        except Exception as exc:
            record["error"]      = f"{type(exc).__name__}: {str(exc)[:60]}"
            record["error_type"] = "exception"

        records.append(record)
        _maybe_print_progress(done, total, t0)

    return records


def print_scan_summary(df_results: pd.DataFrame) -> None:
    """Print a breakdown of scan results by error_type."""
    interp = {
        "none":               "converged successfully",
        "out_of_range":       "outside table P range or CPA2 single-phase",
        "salting_out":        "salt displaces CO2 (x4w<0)",
        "single_phase_stable":"flash failed + ecpa_stability confirmed single-phase",
        "single_phase_gas":   "flash failed + CPA2 confirmed single-phase gas",
        "single_phase_liquid":"flash failed + CPA2 confirmed single-phase liquid",
        "no_sign_change":     "Brent + SSI + CPA2 all uncertain — genuine flash failure",
        "elv_solver":         "ELV Newton failed",
        "cache_empty":        "continuation cache empty at ms=0",
        "ssi_no_converge":    "SSI did not converge — CPA2 also uncertain",
        "runtime_other":      "other RuntimeError",
        "exception":          "unexpected exception",
    }
    type_counts = df_results["error_type"].value_counts()
    total = len(df_results)
    print(f"\n{'Error type':<30s} {'Count':>6s} {'%':>7s}  Interpretation")
    print("-" * 80)
    for etype, count in type_counts.items():
        pct = 100 * count / total
        print(f"  {etype:<28s} {count:>6d} {pct:>6.1f}%  {interp.get(etype, '')}")

    expected_ok = {"out_of_range", "salting_out", "no_sign_change",
                   "single_phase_stable", "single_phase_gas", "single_phase_liquid",
                   "none", "ssi_no_converge"}
    concerning = df_results[~df_results["error_type"].isin(expected_ok)]
    print(f"\nGenuinely concerning failures: {len(concerning)}")
    if len(concerning) > 0:
        print(concerning[["z_co2", "T", "P", "ms", "error_type", "error"]]
              .to_string(index=False))

    boundary_types = {"no_sign_change", "single_phase_stable", "single_phase_gas",
                      "single_phase_liquid", "ssi_no_converge"}
    boundary = df_results[df_results["error_type"].isin(boundary_types)]
    if len(boundary) > 0:
        print(f"\nBoundary / single-phase-classified points ({len(boundary)} total):")
        print(f"  {'z_co2':>6} {'T':>6} {'P':>8} {'ms':>4}  error_type")
        for _, row in boundary.sort_values(["z_co2", "ms", "T", "P"]).iterrows():
            print(f"  {row['z_co2']:6.2f} {row['T']:6.0f} {row['P']:8.2f} {row['ms']:4.1f}"
                  f"  {row['error_type']}")


def _maybe_print_progress(done: int, total: int, t0: float,
                           interval: int = 200) -> None:
    if done % interval == 0:
        elapsed   = time.time() - t0
        remaining = (total - done) / (done / elapsed) if done > 0 else 0
        print(f"  {done}/{total} | {elapsed:.0f}s elapsed | ~{remaining:.0f}s remaining")
