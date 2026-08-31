"""
Extend the solution table with a ms=0 (salt-free) slice.

The existing table covers ms = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0].
For ms < 0.1, the RegularGridInterpolator extrapolates to the ms=0.1
boundary (fill_value=None) but the stability nearest-neighbour interpolator
uses fill_value=0.0, so it returns 'single-phase' for every out-of-bounds
query and forces the expensive Michelsen TPD test every call.

Adding ms=0 gives the interpolator a proper salt-free anchor so that small
ms values (e.g. ms=1e-4 used for binary validation) receive an accurate
warm-start and a correct phase hint without triggering the stability test.

CPA_ELV_all.parquet was generated with the eCPA code at ms=0, so the
10-element solution vectors are already exact eCPA solutions at ms=0 and
can be used directly — no re-solving needed.

Algorithm
---------
1. Load the existing solution table and CPA_ELV_all.parquet.
2. For each (T, P) in the solution-table grid:
   - Find the two nearest T values in CPA_ELV_all and linearly interpolate
     in P (clamping at boundaries).
   - Mark as converged if the interpolation succeeded and the solution is
     physically sensible (xw_W, xw_C in (0,1) and two distinct phases).
3. Apply 2-D nearest-neighbour fill for any missing (T, P) cells.
4. Replicate the (nT, nP) ms=0 slice across all nz z-values
   (binary flash result is z-independent — same phase compositions for any
   feed as long as we are in the two-phase region).
5. Set ms_aq = 0 for the ms=0 slice (no salt => equilibrium molality = 0).
6. Prepend the ms=0 level and save as results/solution_table.npz
   (old table backed up to results/solution_table_bak.npz).
"""

import warnings; warnings.filterwarnings('ignore')

if __name__ == '__main__':
    import shutil
    import numpy as np
    import pandas as pd
    from scipy.ndimage import distance_transform_edt

    from ecpa.solution_table import load_solution_table, make_solution_guess_fn

    SOL_COLS = ['Z_W', 'xw_W', 'eps_r', 'Z_C', 'xw_C',
                'chiw_W', 'chiw_C', 'Ndchi1w_dNw', 'Ndchi1w_dNc', 'Vdchi1w_dV']

    # ── Load existing solution table ──────────────────────────────────────────
    print("Loading existing solution table …")
    table_path = 'results/solution_table.npz'
    gd = load_solution_table(table_path)

    T_grid    = gd['T_grid']       # (17,)
    logP_grid = gd['logP_grid']    # (30,)
    z_grid    = gd['z_grid']       # (18,)
    ms_grid   = gd['ms_grid']      # (7,)  [0.1 … 3.0]
    sol_old   = gd['sol_filled']   # (17,30,18,7,10)
    msaq_old  = gd['ms_aq_filled'] # (17,30,18,7)
    stab_old  = gd['stable']       # (17,30,18,7) bool

    nT, nP, nz, nms = len(T_grid), len(logP_grid), len(z_grid), len(ms_grid)
    P_grid = 10.0 ** logP_grid

    print(f"  Grid: {nT}T × {nP}P × {nz}z × {nms}ms  "
          f"  converged: {stab_old.sum()}/{stab_old.size}")

    # ── Load CPA_ELV_all.parquet ──────────────────────────────────────────────
    print("Loading CPA_ELV_all.parquet …")
    cpa_df = pd.read_parquet('results/CPA_ELV_all.parquet')

    # Build per-T lookup: T_K → dict(P_arr, sol_mat)
    cpa_cache = {}
    for T_K, grp in cpa_df.groupby('T_K'):
        grp = grp.sort_values('P_bar')
        cpa_cache[float(T_K)] = {
            'P': grp['P_bar'].values,
            'sol': grp[SOL_COLS].values,   # (n, 10)
        }

    available_Ts = sorted(cpa_cache.keys())
    print(f"  T values in CPA_ELV_all: {len(available_Ts)}  "
          f"({min(available_Ts):.0f}–{max(available_Ts):.0f} K)")

    def interp_cpa(T, P_bar):
        """
        Interpolate eCPA ms=0 solution at (T, P_bar).
        Uses the two nearest T levels in cpa_cache and linear P interpolation.
        Returns (sol_10, ok).
        """
        # Find bracketing or nearest T values
        T_arr = np.array(available_Ts)
        idx_T = np.searchsorted(T_arr, T)
        if idx_T == 0:
            T_lo = T_hi = T_arr[0]
            w_hi = 0.0
        elif idx_T >= len(T_arr):
            T_lo = T_hi = T_arr[-1]
            w_hi = 0.0
        else:
            T_lo, T_hi = T_arr[idx_T - 1], T_arr[idx_T]
            w_hi = (T - T_lo) / (T_hi - T_lo)

        def _interp_P(T_key, P_bar):
            entry = cpa_cache[T_key]
            P_arr = entry['P']
            sol_mat = entry['sol']
            if P_bar <= P_arr[0]:
                return sol_mat[0].copy()
            if P_bar >= P_arr[-1]:
                return sol_mat[-1].copy()
            i = int(np.searchsorted(P_arr, P_bar))
            i = max(1, min(i, len(P_arr) - 1))
            f = (P_bar - P_arr[i-1]) / (P_arr[i] - P_arr[i-1])
            return (1 - f) * sol_mat[i-1] + f * sol_mat[i]

        sol_lo = _interp_P(T_lo, P_bar)
        if T_lo == T_hi:
            sol = sol_lo
        else:
            sol_hi = _interp_P(T_hi, P_bar)
            sol = (1 - w_hi) * sol_lo + w_hi * sol_hi

        ok = (np.all(np.isfinite(sol))
              and 0.0 < sol[1] < 1.0    # xw_W (H2O in aqueous)
              and 0.0 < sol[4] < 1.0    # xw_C (H2O in CO2-rich)
              and sol[1] - sol[4] > 0.01)  # two distinct phases

        return sol, ok

    # ── Interpolate onto solution-table (T, P) grid ───────────────────────────
    print(f"Interpolating CPA_ELV_all onto {nT}×{nP} = {nT*nP} (T,P) grid points …")

    sol_ms0  = np.full((nT, nP, 10), np.nan)
    conv_ms0 = np.zeros((nT, nP), dtype=bool)

    n_conv = 0
    for iT, T_i in enumerate(T_grid):
        for iP, P_i in enumerate(P_grid):
            sol, ok = interp_cpa(float(T_i), float(P_i))
            if ok:
                sol_ms0[iT, iP]  = sol
                conv_ms0[iT, iP] = True
                n_conv += 1
        print(f"  T={T_i:.0f}K  ok {conv_ms0[iT].sum()}/{nP}")

    print(f"\nTotal mapped at ms=0: {n_conv}/{nT*nP} "
          f"({100*n_conv/(nT*nP):.1f}%)")

    # ── 2-D nearest-neighbour fill for missing (T, P) cells ──────────────────
    invalid_2d = ~conv_ms0
    if invalid_2d.any():
        print(f"Filling {invalid_2d.sum()} missing cells with nearest-neighbour …")
        _, nn_idx = distance_transform_edt(invalid_2d, return_indices=True)
        iT_nn, iP_nn = nn_idx
        sol_ms0_filled = sol_ms0.copy()
        sol_ms0_filled[invalid_2d] = sol_ms0[iT_nn[invalid_2d], iP_nn[invalid_2d]]
    else:
        sol_ms0_filled = sol_ms0.copy()

    # ── Replicate across z (binary ELV is z-independent) ─────────────────────
    sol_ms0_4d  = np.broadcast_to(
        sol_ms0_filled[:, :, np.newaxis, :],
        (nT, nP, nz, 10)
    ).copy()
    msaq_ms0_3d = np.zeros((nT, nP, nz), dtype=float)
    stab_ms0_3d = np.broadcast_to(
        conv_ms0[:, :, np.newaxis],
        (nT, nP, nz)
    ).copy()

    # ── Build expanded table ──────────────────────────────────────────────────
    ms_grid_new = np.concatenate([[0.0], ms_grid])   # (8,)

    sol_new  = np.concatenate(
        [sol_ms0_4d[:, :, :, np.newaxis, :], sol_old],   # (nT,nP,nz,8,10)
        axis=3
    )
    msaq_new = np.concatenate(
        [msaq_ms0_3d[:, :, :, np.newaxis], msaq_old],    # (nT,nP,nz,8)
        axis=3
    )
    stab_new = np.concatenate(
        [stab_ms0_3d[:, :, :, np.newaxis], stab_old],    # (nT,nP,nz,8)
        axis=3
    )

    print(f"\nExpanded table shape: {sol_new.shape}")
    print(f"ms_grid: {ms_grid_new}")
    total_new = stab_new.size
    conv_new  = int(stab_new.sum())
    print(f"Total two-phase cells: {conv_new}/{total_new} ({100*conv_new/total_new:.1f}%)")

    # ── Back up and save ──────────────────────────────────────────────────────
    backup_path = table_path.replace('.npz', '_bak.npz')
    shutil.copy2(table_path, backup_path)
    print(f"\nBacked up original table → {backup_path}")

    np.savez_compressed(
        table_path,
        T_grid       = T_grid,
        logP_grid    = logP_grid,
        z_grid       = z_grid,
        ms_grid      = ms_grid_new,
        sol_filled   = sol_new,
        ms_aq_filled = msaq_new,
        stable       = stab_new.astype(np.uint8),
    )
    print(f"Saved expanded table → {table_path}")

    # ── Sanity check: compare new table at ms=0 vs CPA_ELV_all directly ──────
    print("\n── Sanity check: table ms=0 vs CPA_ELV_all (should be ~0%) ─────────")
    print(f"{'T':>6} {'P':>8}  {'ELV xc_W':>10} {'table xc_W':>10} {'diff%':>7}  "
          f"{'ELV yw_C':>10} {'table yw_C':>10} {'diff%':>7}")

    gd_new = load_solution_table(table_path)
    guess_fn_new = make_solution_guess_fn(gd_new)

    check_cases = [(323, 100), (373, 200), (448, 300), (498, 500), (523, 800)]
    for T_c, P_c in check_cases:
        sol_t, _, _ = guess_fn_new(T_c, P_c, 0.5, 0.0)
        xc_W_tab = 1.0 - float(sol_t[1])
        yw_C_tab = float(sol_t[4])

        sol_d, ok = interp_cpa(T_c, P_c)
        if ok:
            xc_W_elv = 1.0 - float(sol_d[1])
            yw_C_elv = float(sol_d[4])
            diff_xc = abs(xc_W_tab - xc_W_elv) / xc_W_elv * 100 if xc_W_elv > 0 else float('nan')
            diff_yw = abs(yw_C_tab - yw_C_elv) / yw_C_elv * 100 if yw_C_elv > 0 else float('nan')
        else:
            xc_W_elv = yw_C_elv = diff_xc = diff_yw = float('nan')

        print(f"{T_c:>6.0f} {P_c:>8.1f}  "
              f"{xc_W_elv:>10.5f} {xc_W_tab:>10.5f} {diff_xc:>7.3f}%  "
              f"{yw_C_elv:>10.5f} {yw_C_tab:>10.5f} {diff_yw:>7.3f}%")

    print("\nDone.")
