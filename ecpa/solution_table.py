"""
LEGACY / OLD INFRASTRUCTURE — build_solution_table() uses flash_co2_h2o_salt_ssi
(outer ms_aq loop) which has been superseded by flash_co2_h2o_salt_kv.
For new solution tables, use _build_scan_table_v2.py.

The load_solution_table() and make_solution_guess_fn() helpers remain valid
for reading existing NPZ tables.

Solution lookup table for fast eCPA flash.

Offline (once):
    build_solution_table() runs a dense flash scan on a regular (T, P, z_co2, ms)
    grid, stores the full 10-element ELV solution vector and ms_aq at each
    converged point, fills unconverged cells by nearest-neighbour from valid
    neighbours, and saves everything to a compressed NPZ file.

Online (per simulation call):
    grid_data = load_solution_table(path)
    guess_fn  = make_solution_guess_fn(grid_data)

    # scalar query
    sol_guess, ms_aq_guess, is_two_phase = guess_fn(T, P_bar, z_co2, ms)

    # batch query (N points at once — fast for reservoir simulator use)
    sol_arr, ms_aq_arr, is_2ph_arr = guess_fn(T_arr, P_arr, z_arr, ms_arr)

With a good initial guess, flash_co2_h2o_salt_ssi (via flash_co2_h2o_salt_fast)
typically converges in 1-3 SSI iterations instead of 20-40 from a cold start.

Grid axes
---------
T    [K]         : uniform spacing (DEFAULT_T_GRID)
logP [log10 bar] : uniform in log (DEFAULT_P_GRID)
z    [mol/mol]   : uniform (DEFAULT_Z_GRID)
ms   [mol/kg]    : non-uniform sparse (DEFAULT_MS_GRID); ms=0 excluded
                   because eCPA flash is undefined for salt-free.

Stored arrays (in NPZ)
----------------------
T_grid, logP_grid, z_grid, ms_grid  — 1-D axis arrays
sol_filled      (nT, nP, nz, nms, 10)  — ELV solution vector, NaN-filled
ms_aq_filled    (nT, nP, nz, nms)      — equilibrium ms_aq, NaN-filled
stable          (nT, nP, nz, nms)      — bool: True = two-phase (converged)
"""
import multiprocessing as mp
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt

from .flash import flash_co2_h2o_salt_ssi
from .guess_table import make_guess_fn
from .scan import _cpu_count


# ── Default grid ───────────────────────────────────────────────────────────────

# Temperature [K]: 283–523 K in 15 K steps → 17 points
# Upper limit 523 K: convergence below 20% above that (mostly single-phase at
# supercritical CO2 conditions), so including those points gives a poor table.
DEFAULT_T_GRID = np.arange(283.0, 524.0, 15.0)

# Pressure [bar]: 1–1500 bar, log-spaced → 30 points
DEFAULT_P_GRID = np.logspace(0.0, np.log10(1500.0), 30)

# Feed CO₂ mole fraction: 0.05–0.90 → 18 points
# Upper limit 0.90: z=0.95 has only ~26% convergence (near-pure CO2, single-phase)
DEFAULT_Z_GRID = np.linspace(0.05, 0.90, 18)

# Salt molality [mol/kg H₂O]: ms=0 excluded (eCPA flash not defined for ms=0).
# 2.5 replaces 5.0: ms=5.0 had only 24% convergence (unrealistically high for
# most saline aquifers); better resolution in the physically relevant 0–3 range.
DEFAULT_MS_GRID = np.array([0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

SOL_DIM = 10   # length of ELV solution vector from flash_co2_h2o_salt_ssi


# ── Parallel scan worker (row-parallel, warm-started along P) ──────────────────
# Each worker handles all P values for one (T, z, ms) combination, scanning P
# in ascending order and warm-starting each flash from the previous P solution.
# This dramatically improves convergence near phase boundaries compared to
# independent cold-start calls.
#
# Module-level globals — populated by _sol_scan_init in each spawned process.

_ST_params   = None
_ST_guess_fn = None


def _sol_scan_init(params, CPA_GROUPS, CPA_TEMPS):
    """Initialise worker process globals."""
    global _ST_params, _ST_guess_fn
    _ST_params   = params
    _ST_guess_fn = make_guess_fn(CPA_GROUPS, CPA_TEMPS)


def _sol_row_worker(task):
    """
    Run a warm-started P-scan for one (T, z, ms) row.

    task = (iT, iz, ims, T_val, P_arr_sorted, z_val, ms_val)

    P_arr_sorted must be in ascending order (for warm-starting continuity).

    For each P:
      1. If a warm-start solution is available from the previous P, try it first.
      2. If the warm start fails (or no warm start available), fall back to a
         cold start from the CPA2 guess table.
      3. Only record failure if both attempts fail.

    Returns (iT, iz, ims, row_results) where row_results is a list of
    (iP_sorted, sol_list, ms_aq, converged) for each pressure point.
    """
    iT, iz, ims, T_i, P_arr, z_i, ms_i = task
    sol_prev   = None
    ms_aq_prev = None
    row_results = []

    for iP, P_i in enumerate(P_arr):
        sol_out   = [np.nan] * SOL_DIM
        ms_aq_out = np.nan
        conv      = False

        # ── Attempt 1: warm start from previous P ────────────────────────────
        if sol_prev is not None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    out = flash_co2_h2o_salt_ssi(
                        T=float(T_i), P_bar=float(P_i),
                        z_co2=float(z_i), m_tot=float(ms_i),
                        params=_ST_params,
                        initial_sol=sol_prev,
                        initial_ms_aq=ms_aq_prev,
                        maxiter_ms=20,
                    )
                sol_prev   = out["sol"].copy()
                ms_aq_prev = float(out["ms_aq"])
                sol_out    = sol_prev.tolist()
                ms_aq_out  = ms_aq_prev
                conv       = True
            except Exception:
                pass   # warm start failed → try cold start below

        # ── Attempt 2: cold start from CPA2 guess table ───────────────────────
        if not conv:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    out = flash_co2_h2o_salt_ssi(
                        T=float(T_i), P_bar=float(P_i),
                        z_co2=float(z_i), m_tot=float(ms_i),
                        guess_table_fn=_ST_guess_fn,
                        params=_ST_params,
                    )
                sol_prev   = out["sol"].copy()
                ms_aq_prev = float(out["ms_aq"])
                sol_out    = sol_prev.tolist()
                ms_aq_out  = ms_aq_prev
                conv       = True
            except Exception:
                # Both attempts failed — reset warm start, record failure
                sol_prev   = None
                ms_aq_prev = None

        row_results.append((iP, sol_out, ms_aq_out, conv))

    return iT, iz, ims, row_results


# ── NaN fill ───────────────────────────────────────────────────────────────────

def _fill_nan_nearest(arr_4d, arr_5d, valid_mask):
    """
    Replace entries at invalid (not converged) grid points with values from
    the nearest valid (converged) grid point, using Euclidean distance in
    grid-index space.

    Parameters
    ----------
    arr_4d     : ndarray (nT, nP, nz, nms)       — scalar field (ms_aq)
    arr_5d     : ndarray (nT, nP, nz, nms, k)    — vector field (sol)
    valid_mask : bool ndarray (nT, nP, nz, nms)  — True where converged

    Returns
    -------
    arr_4d_filled, arr_5d_filled : copies with NaN replaced
    """
    invalid = ~valid_mask
    if not invalid.any():
        return arr_4d.copy(), arr_5d.copy()

    _, nearest_idx = distance_transform_edt(invalid, return_indices=True)
    # nearest_idx: shape (4, nT, nP, nz, nms) — integer grid coords of nearest valid cell
    iT_n, iP_n, iz_n, ims_n = nearest_idx

    arr_4d = arr_4d.copy()
    arr_5d = arr_5d.copy()

    arr_4d[invalid] = arr_4d[iT_n[invalid], iP_n[invalid],
                              iz_n[invalid], ims_n[invalid]]
    arr_5d[invalid] = arr_5d[iT_n[invalid], iP_n[invalid],
                              iz_n[invalid], ims_n[invalid]]
    return arr_4d, arr_5d


# ── Build ──────────────────────────────────────────────────────────────────────

def build_solution_table(
    params,
    CPA_GROUPS,
    CPA_TEMPS,
    T_grid=None,
    P_grid=None,
    z_grid=None,
    ms_grid=None,
    save_path="results/solution_table.npz",
    n_workers=None,
    force_recompute=False,
):
    """
    Run a dense flash scan on a regular (T, P, z_co2, ms) grid and save
    full ELV solution vectors to a compressed NPZ file.

    If the file already exists and force_recompute=False, the file is loaded
    and returned without re-running the scan.

    Parameters
    ----------
    params          : dict of EoS parameter overrides (passed to flash)
    CPA_GROUPS, CPA_TEMPS : output of load_cpa_guess_table()
    T_grid          : 1-D array of temperatures [K]  (default: DEFAULT_T_GRID)
    P_grid          : 1-D array of pressures [bar]   (default: DEFAULT_P_GRID)
    z_grid          : 1-D array of z_CO₂             (default: DEFAULT_Z_GRID)
    ms_grid         : 1-D array of molalities [mol/kg](default: DEFAULT_MS_GRID)
    save_path       : output NPZ path
    n_workers       : parallel workers (None = cpu_count − 1)
    force_recompute : re-run even if save_path exists

    Returns
    -------
    grid_data : dict with keys T_grid, logP_grid, z_grid, ms_grid,
                sol_filled, ms_aq_filled, stable
    """
    if os.path.exists(save_path) and not force_recompute:
        print(f"Loading existing solution table: {save_path}")
        return load_solution_table(save_path)

    T_grid  = np.asarray(T_grid  if T_grid  is not None else DEFAULT_T_GRID,  dtype=float)
    P_grid  = np.asarray(P_grid  if P_grid  is not None else DEFAULT_P_GRID,  dtype=float)
    z_grid  = np.asarray(z_grid  if z_grid  is not None else DEFAULT_Z_GRID,  dtype=float)
    ms_grid = np.asarray(ms_grid if ms_grid is not None else DEFAULT_MS_GRID, dtype=float)

    nT, nP, nz, nms = len(T_grid), len(P_grid), len(z_grid), len(ms_grid)
    logP_grid = np.log10(P_grid)
    total = nT * nP * nz * nms

    # Row tasks: one per (T, z, ms) combination; P scanned serially with warm-start.
    # P is sorted ascending within each row so warm-starting traces the two-phase
    # window continuously from low to high pressure.
    P_sorted   = np.sort(P_grid)
    P_sort_idx = np.argsort(P_grid)   # maps sorted position → original iP

    tasks = [
        (iT, iz, ims,
         float(T_grid[iT]), P_sorted.tolist(),
         float(z_grid[iz]), float(ms_grid[ims]))
        for iT  in range(nT)
        for iz  in range(nz)
        for ims in range(nms)
    ]
    n_rows = len(tasks)

    n_workers = n_workers or _cpu_count()
    print(f"Building solution table: {total:,} points "
          f"({nT}T × {nP}P × {nz}z × {nms}ms)  "
          f"n_rows={n_rows}  n_workers={n_workers}")
    t0 = time.time()

    # Allocate output arrays (NaN = not yet converged)
    sol_raw   = np.full((nT, nP, nz, nms, SOL_DIM), np.nan)
    ms_aq_raw = np.full((nT, nP, nz, nms),           np.nan)
    stable    = np.zeros((nT, nP, nz, nms),           dtype=bool)

    rows_done = 0
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx,
                             initializer=_sol_scan_init,
                             initargs=(params, CPA_GROUPS, CPA_TEMPS)) as ex:
        for iT, iz, ims, row_results in ex.map(
                _sol_row_worker, tasks, chunksize=4):
            for iP_sorted, sol_list, ms_aq_val, conv in row_results:
                iP = int(P_sort_idx[iP_sorted])   # restore original P ordering
                if conv:
                    sol_raw[iT, iP, iz, ims]   = sol_list
                    ms_aq_raw[iT, iP, iz, ims] = ms_aq_val
                    stable[iT, iP, iz, ims]    = True
            rows_done += 1
            if rows_done % 100 == 0 or rows_done == n_rows:
                elapsed  = time.time() - t0
                rate     = rows_done / elapsed if elapsed > 0 else 0
                rem      = (n_rows - rows_done) / rate if rate > 0 else 0
                n_conv   = int(stable.sum())
                pts_done = rows_done * nP
                print(f"  {rows_done}/{n_rows} rows  {elapsed:.0f}s elapsed"
                      f"  ~{rem:.0f}s left  "
                      f"{n_conv}/{pts_done} pts converged "
                      f"({100*n_conv/pts_done:.1f}%)")

    elapsed = time.time() - t0
    n_conv  = int(stable.sum())
    print(f"Done: {n_conv}/{total} converged ({100*n_conv/total:.1f}%)  {elapsed:.1f}s")

    # Fill NaN cells with nearest-neighbour so the interpolator has no gaps
    ms_aq_filled, sol_filled = _fill_nan_nearest(ms_aq_raw, sol_raw, stable)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    np.savez_compressed(
        save_path,
        T_grid=T_grid, logP_grid=logP_grid, z_grid=z_grid, ms_grid=ms_grid,
        sol_filled=sol_filled,
        ms_aq_filled=ms_aq_filled,
        stable=stable.astype(np.uint8),   # bool arrays don't compress well
    )
    print(f"Saved → {save_path}")

    return dict(T_grid=T_grid, logP_grid=logP_grid, z_grid=z_grid,
                ms_grid=ms_grid, sol_filled=sol_filled,
                ms_aq_filled=ms_aq_filled, stable=stable)


# ── Load ───────────────────────────────────────────────────────────────────────

def load_solution_table(path="results/solution_table.npz"):
    """
    Load a solution table saved by build_solution_table().

    Returns
    -------
    grid_data : dict with keys T_grid, logP_grid, z_grid, ms_grid,
                sol_filled, ms_aq_filled, stable
    """
    data = np.load(path)
    return dict(
        T_grid       = data["T_grid"].astype(float),
        logP_grid    = data["logP_grid"].astype(float),
        z_grid       = data["z_grid"].astype(float),
        ms_grid      = data["ms_grid"].astype(float),
        sol_filled   = data["sol_filled"].astype(float),
        ms_aq_filled = data["ms_aq_filled"].astype(float),
        stable       = data["stable"].astype(bool),
    )


# ── Interpolator ───────────────────────────────────────────────────────────────

def make_solution_guess_fn(grid_data):
    """
    Build a fast interpolated-guess function from a loaded solution table.

    Constructs one RegularGridInterpolator per output variable (ms_aq + 10
    ELV components) plus a nearest-neighbour interpolator for the stability
    flag.  All interpolators use linear interpolation in (T, log₁₀P, z, ms)
    space; queries outside the grid are handled by clamping to the boundary
    (no extrapolation error).

    Parameters
    ----------
    grid_data : dict returned by build_solution_table() or load_solution_table()

    Returns
    -------
    guess_fn : callable
        Signature: guess_fn(T, P_bar, z_co2, ms)
            → (sol_guess, ms_aq_guess, is_two_phase)

        All arguments may be scalars or 1-D arrays of the same length N.
        For scalar inputs returns (ndarray(10,), float, bool).
        For array inputs returns (ndarray(N,10), ndarray(N,), ndarray(N, bool)).

    Notes
    -----
    is_two_phase uses nearest-neighbour from the raw stable mask, so it
    reflects whether the nearest scanned grid point converged as two-phase.
    Points near the phase boundary may be misclassified; the caller should
    always run a flash attempt and fall back to the full SSI on failure.
    """
    T_grid       = grid_data["T_grid"]
    logP_grid    = grid_data["logP_grid"]
    z_grid       = grid_data["z_grid"]
    ms_grid      = grid_data["ms_grid"]
    sol_filled   = grid_data["sol_filled"]    # (nT, nP, nz, nms, 10)
    ms_aq_filled = grid_data["ms_aq_filled"]  # (nT, nP, nz, nms)
    stable       = grid_data["stable"]         # (nT, nP, nz, nms) bool

    axes = (T_grid, logP_grid, z_grid, ms_grid)

    # Stack ms_aq (scalar) + sol (10 components) into (nT, nP, nz, nms, 11)
    values_all = np.concatenate(
        [ms_aq_filled[:, :, :, :, np.newaxis], sol_filled], axis=-1
    )  # shape (nT, nP, nz, nms, 11)

    # One linear interpolator per output component
    interps = [
        RegularGridInterpolator(
            axes, values_all[:, :, :, :, k],
            method="linear", bounds_error=False, fill_value=None,
        )
        for k in range(11)
    ]

    # Nearest-neighbour for the stability flag (bool → float → threshold)
    stable_interp = RegularGridInterpolator(
        axes, stable.astype(float),
        method="nearest", bounds_error=False, fill_value=0.0,
    )

    def guess_fn(T, P_bar, z_co2, ms):
        """
        Interpolated initial guess for flash_co2_h2o_salt_fast.

        Returns (sol_guess, ms_aq_guess, is_two_phase).
        """
        scalar = np.ndim(T) == 0
        T_a    = np.atleast_1d(np.asarray(T,     dtype=float))
        P_a    = np.atleast_1d(np.asarray(P_bar, dtype=float))
        z_a    = np.atleast_1d(np.asarray(z_co2, dtype=float))
        ms_a   = np.atleast_1d(np.asarray(ms,    dtype=float))

        pts = np.column_stack([T_a, np.log10(P_a), z_a, ms_a])  # (N, 4)

        # Evaluate all 11 interpolators; stack into (N, 11)
        out = np.column_stack([interp(pts) for interp in interps])
        ms_aq_guess  = out[:, 0]          # (N,)
        sol_guess    = out[:, 1:]         # (N, 10)
        is_two_phase = stable_interp(pts) > 0.5  # (N,)

        if scalar:
            return sol_guess[0], float(ms_aq_guess[0]), bool(is_two_phase[0])
        return sol_guess, ms_aq_guess, is_two_phase.astype(bool)

    return guess_fn


# ── Convenience: print grid info ───────────────────────────────────────────────

def print_table_summary(grid_data):
    """Print coverage statistics for a loaded solution table."""
    stable    = grid_data["stable"]
    T_grid    = grid_data["T_grid"]
    logP_grid = grid_data["logP_grid"]
    z_grid    = grid_data["z_grid"]
    ms_grid   = grid_data["ms_grid"]

    total  = stable.size
    n_conv = int(stable.sum())
    P_grid = 10.0 ** logP_grid

    print(f"Solution table summary")
    print(f"  Grid   : {len(T_grid)}T × {len(P_grid)}P × {len(z_grid)}z × {len(ms_grid)}ms"
          f"  = {total:,} points")
    print(f"  T      : {T_grid[0]:.0f} – {T_grid[-1]:.0f} K")
    print(f"  P      : {P_grid[0]:.2f} – {P_grid[-1]:.1f} bar")
    print(f"  z_CO2  : {z_grid[0]:.2f} – {z_grid[-1]:.2f}")
    print(f"  ms     : {ms_grid[0]:.1f} – {ms_grid[-1]:.1f} mol/kg")
    print(f"  Converged: {n_conv}/{total} ({100*n_conv/total:.1f}%)")
