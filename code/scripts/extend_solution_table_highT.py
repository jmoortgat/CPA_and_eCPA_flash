"""
Extend the solution table to T = 538–623 K.

The existing table covers T = 283–523 K.  Experimental data extends to 623 K,
so we add 7 new T levels: 538, 553, 568, 583, 598, 613, 623 K (continuing
the 15 K grid spacing), covering all ms levels [0.0, 0.1, 0.5, …, 3.0].

Strategy
--------
ELV direct solves are used instead of full stability+flash because:
  1. ELV is cheaper (one 10-var fsolve vs SSI outer loop + optional Michelsen test).
  2. ELV solutions are z-independent — solve once per (T, P, ms), replicate
     across all 18 z values.
  3. The resulting table entries are identical in format to the existing ones
     and serve as warm-start initial guesses for the flash at inference time.

Warm-start hierarchy
--------------------
  ms=0, T=538 K : CPA_ELV_all.parquet at T=533 K (exact eCPA ms=0 solutions).
  ms=0, T>538 K : T-continuation — use converged ms=0 solution from T_prev.
  ms>0 at any T : ms-continuation — step 0→0.1→0.5→1.0→1.5→2.0→2.5→3.0,
                  each step using the previous ms solution as warm start.

Non-converged cells are filled by 2-D nearest-neighbour in the (T, P) plane
per ms level, consistent with how the existing table was built.

The new T slices are appended along the T axis and the expanded table is
saved (original backed up first).
"""

import warnings; warnings.filterwarnings('ignore')

if __name__ == '__main__':
    import shutil
    import numpy as np
    import pandas as pd
    from scipy.optimize import fsolve
    from scipy.ndimage import distance_transform_edt

    from ecpa.parameters import make_params
    from ecpa.elv import ELV, ELV_jac, USE_COMPLEX_JAC
    from ecpa.solution_table import load_solution_table, make_solution_guess_fn

    params = make_params()

    SOL_COLS = ['Z_W', 'xw_W', 'eps_r', 'Z_C', 'xw_C',
                'chiw_W', 'chiw_C', 'Ndchi1w_dNw', 'Ndchi1w_dNc', 'Vdchi1w_dV']

    # ── Load existing solution table ──────────────────────────────────────────
    print("Loading existing solution table …")
    table_path = 'results/solution_table.npz'
    gd = load_solution_table(table_path)

    T_grid    = gd['T_grid']       # (17,)  283–523 K
    logP_grid = gd['logP_grid']    # (30,)
    z_grid    = gd['z_grid']       # (18,)
    ms_grid   = gd['ms_grid']      # (8,)   [0.0, 0.1, 0.5, …, 3.0]
    sol_old   = gd['sol_filled']   # (17,30,18,8,10)
    msaq_old  = gd['ms_aq_filled'] # (17,30,18,8)
    stab_old  = gd['stable']       # (17,30,18,8) bool

    nT, nP, nz, nms = sol_old.shape[:4]
    P_grid = 10.0 ** logP_grid

    print(f"  Existing grid: {nT}T × {nP}P × {nz}z × {nms}ms")
    print(f"  T range: {T_grid[0]:.0f}–{T_grid[-1]:.0f} K")
    print(f"  ms_grid: {ms_grid}")

    # ── New T values ──────────────────────────────────────────────────────────
    T_new = np.array([538., 553., 568., 583., 598., 613., 623.])
    nT_new = len(T_new)
    print(f"\nNew T values: {T_new}")

    # ── Load CPA_ELV_all.parquet — warm starts for ms=0 at T≤533 K ───────────
    print("\nLoading CPA_ELV_all.parquet …")
    cpa_df = pd.read_parquet('results/CPA_ELV_all.parquet')
    cpa_cache = {}
    for T_K, grp in cpa_df.groupby('T_K'):
        grp = grp.sort_values('P_bar')
        cpa_cache[float(T_K)] = {
            'P':   grp['P_bar'].values,
            'sol': grp[SOL_COLS].values,
        }
    cpa_T_arr = np.array(sorted(cpa_cache.keys()))
    print(f"  Available: T={cpa_T_arr.min():.0f}–{cpa_T_arr.max():.0f} K  "
          f"({len(cpa_T_arr)} levels)")

    def interp_cpa_ms0(T, P_bar):
        """Bilinear (T,P) interpolation of CPA_ELV_all. Returns (sol, ok)."""
        idx_T = np.searchsorted(cpa_T_arr, T)
        if idx_T == 0:
            T_lo = T_hi = cpa_T_arr[0]; w_hi = 0.0
        elif idx_T >= len(cpa_T_arr):
            T_lo = T_hi = cpa_T_arr[-1]; w_hi = 0.0
        else:
            T_lo, T_hi = cpa_T_arr[idx_T-1], cpa_T_arr[idx_T]
            w_hi = (T - T_lo) / (T_hi - T_lo)

        def _interp_P(Tk, P_bar):
            e = cpa_cache[Tk]
            Pa = e['P']; Sa = e['sol']
            if P_bar <= Pa[0]:  return Sa[0].copy()
            if P_bar >= Pa[-1]: return Sa[-1].copy()
            i = max(1, min(int(np.searchsorted(Pa, P_bar)), len(Pa)-1))
            f = (P_bar - Pa[i-1]) / (Pa[i] - Pa[i-1])
            return (1-f)*Sa[i-1] + f*Sa[i]

        sol_lo = _interp_P(T_lo, P_bar)
        sol = sol_lo if T_lo == T_hi else (1-w_hi)*sol_lo + w_hi*_interp_P(T_hi, P_bar)
        ok = (np.all(np.isfinite(sol))
              and 0.0 < sol[1] < 1.0
              and 0.0 < sol[4] < 1.0
              and sol[1] - sol[4] > 0.01)
        return sol, ok

    # ── Solver helper ─────────────────────────────────────────────────────────
    def solve_elv(T, P_bar, ms, guess):
        """Run fsolve on ELV. Returns (sol, converged)."""
        try:
            sol, info, ier, _ = fsolve(
                ELV, guess,
                args=(float(T), float(P_bar)*1e5, float(ms), params),
                fprime=ELV_jac if USE_COMPLEX_JAC else None,
                full_output=True, xtol=1e-10, maxfev=2000,
            )
            sol = np.asarray(sol, dtype=np.float64)
            res = np.asarray(ELV(sol, float(T), float(P_bar)*1e5, float(ms), params),
                             dtype=np.float64)
            rn  = float(np.linalg.norm(res))
            ok  = (ier == 1
                   and np.all(np.isfinite(sol))
                   and rn < 1e-6
                   and 0.0 < sol[1] < 1.0
                   and 0.0 < sol[4] < 1.0
                   and sol[1] - sol[4] > 0.01)
        except Exception:
            sol = guess.copy(); ok = False
        return sol, ok

    # ── Main loop: solve ELV at all new (T, P, ms) points ────────────────────
    # Storage: (nT_new, nP, nms, 10) for solutions, (nT_new, nP, nms) for flags
    sol_new_raw  = np.full((nT_new, nP, nms, 10), np.nan)
    conv_new     = np.zeros((nT_new, nP, nms), dtype=bool)

    # T-continuation: keep previous-T ms=0 solutions for warm start when
    # CPA_ELV_all doesn't reach the target T.
    prev_T_ms0 = np.full((nP, 10), np.nan)  # will be filled after first T

    for iT, T_i in enumerate(T_new):
        print(f"\n── T = {T_i:.0f} K ──────────────────────────────────────────────")

        for iP, P_i in enumerate(P_grid):

            # ── ms=0 warm start ────────────────────────────────────────────
            if T_i <= cpa_T_arr[-1] + 8:
                # CPA_ELV_all covers this T (or is within 8 K)
                guess_ms0, cpa_ok = interp_cpa_ms0(T_i, P_i)
            else:
                # Use T-continuation from previous T
                cpa_ok = np.all(np.isfinite(prev_T_ms0[iP]))
                guess_ms0 = prev_T_ms0[iP].copy() if cpa_ok else np.full(10, np.nan)

            if not cpa_ok:
                continue

            # ── Solve ms=0 ────────────────────────────────────────────────
            if T_i <= cpa_T_arr[-1] + 8:
                # CPA_ELV_all is already exact eCPA ms=0: check residual
                res0 = np.linalg.norm(
                    ELV(guess_ms0, float(T_i), float(P_i)*1e5, 0.0, params))
                if res0 < 1e-6 and 0.0 < guess_ms0[1] < 1.0 and 0.0 < guess_ms0[4] < 1.0:
                    sol_ms0 = guess_ms0
                    ok_ms0  = True
                else:
                    sol_ms0, ok_ms0 = solve_elv(T_i, P_i, 0.0, guess_ms0)
            else:
                sol_ms0, ok_ms0 = solve_elv(T_i, P_i, 0.0, guess_ms0)

            if ok_ms0:
                sol_new_raw[iT, iP, 0] = sol_ms0
                conv_new[iT, iP, 0]    = True

            # ── ms continuation: 0 → 0.1 → 0.5 → … → 3.0 ───────────────
            prev_sol = sol_ms0 if ok_ms0 else guess_ms0
            for ims, ms_k in enumerate(ms_grid[1:], start=1):
                sol_k, ok_k = solve_elv(T_i, P_i, ms_k, prev_sol)
                if ok_k:
                    sol_new_raw[iT, iP, ims] = sol_k
                    conv_new[iT, iP, ims]    = True
                    prev_sol = sol_k
                else:
                    # Don't advance continuation from a failed point
                    pass

        # Store ms=0 solutions for T-continuation to next T level
        prev_T_ms0 = sol_new_raw[iT, :, 0].copy()

        # Print convergence summary for this T
        for ims, ms_k in enumerate(ms_grid):
            n_ok = int(conv_new[iT, :, ims].sum())
            print(f"  ms={ms_k:.1f}  converged {n_ok}/{nP}")

    # ── 2-D nearest-neighbour fill per ms level ───────────────────────────────
    print("\nNearest-neighbour fill for non-converged (T, P) cells …")
    sol_new_filled  = sol_new_raw.copy()
    conv_new_filled = conv_new.copy()

    for ims in range(nms):
        invalid = ~conv_new[:, :, ims]
        if invalid.any():
            _, nn_idx = distance_transform_edt(invalid, return_indices=True)
            iT_nn, iP_nn = nn_idx
            sol_new_filled[invalid, ims, :] = sol_new_raw[
                iT_nn[invalid], iP_nn[invalid], ims, :]
            print(f"  ms={ms_grid[ims]:.1f}  filled {invalid.sum()} cells")

    # ── Replicate across z (ELV solutions are z-independent) ─────────────────
    # sol_new_filled : (nT_new, nP, nms, 10) → (nT_new, nP, nz, nms, 10)
    sol_new_5d = np.broadcast_to(
        sol_new_filled[:, :, np.newaxis, :, :],
        (nT_new, nP, nz, nms, 10)
    ).copy()

    # ms_aq = ms (ELV input molality IS the aqueous molality)
    msaq_new_4d = np.broadcast_to(
        ms_grid[np.newaxis, np.newaxis, np.newaxis, :],   # (1,1,1,nms)
        (nT_new, nP, nz, nms)
    ).copy()

    # stable flag: True only where ELV genuinely converged (not NN-filled)
    stab_new_4d = np.broadcast_to(
        conv_new[:, :, np.newaxis, :],                    # (nT_new,nP,1,nms)
        (nT_new, nP, nz, nms)
    ).copy()

    # ── Concatenate with existing table along T axis ──────────────────────────
    T_grid_new  = np.concatenate([T_grid, T_new])
    sol_cat     = np.concatenate([sol_old,  sol_new_5d],   axis=0)  # (24,30,18,8,10)
    msaq_cat    = np.concatenate([msaq_old, msaq_new_4d],  axis=0)
    stab_cat    = np.concatenate([stab_old, stab_new_4d],  axis=0)

    print(f"\nExpanded table shape: {sol_cat.shape}")
    print(f"T_grid ({len(T_grid_new)} values): {T_grid_new}")
    total = stab_cat.size
    conv  = int(stab_cat.sum())
    print(f"Total two-phase cells: {conv}/{total} ({100*conv/total:.1f}%)")

    # ── Back up and save ──────────────────────────────────────────────────────
    backup_path = table_path.replace('.npz', '_pre_highT_bak.npz')
    shutil.copy2(table_path, backup_path)
    print(f"\nBacked up → {backup_path}")

    np.savez_compressed(
        table_path,
        T_grid       = T_grid_new,
        logP_grid    = logP_grid,
        z_grid       = z_grid,
        ms_grid      = ms_grid,
        sol_filled   = sol_cat,
        ms_aq_filled = msaq_cat,
        stable       = stab_cat.astype(np.uint8),
    )
    print(f"Saved expanded table → {table_path}")

    # ── Sanity check: compare ELV ms=0 at a few high-T points ────────────────
    print("\n── Sanity check: ELV ms=0 at new T levels vs CPA_ELV_all ─────────")
    print(f"{'T':>6} {'P':>8}  {'ELV xc_W':>10} {'CPA xc_W':>10} {'diff%':>7}  "
          f"{'ELV yw_C':>10} {'CPA yw_C':>10} {'diff%':>7}")

    check_cases = [(538, 100), (553, 200), (568, 300), (583, 500), (598, 800), (623, 1000)]
    gd_new = load_solution_table(table_path)
    guess_fn = make_solution_guess_fn(gd_new)

    for T_c, P_c in check_cases:
        sol_t, _, _ = guess_fn(T_c, P_c, 0.5, 0.0)
        xc_W_tab = 1.0 - float(sol_t[1])
        yw_C_tab = float(sol_t[4])

        sol_c, ok_c = interp_cpa_ms0(T_c, P_c)
        if ok_c:
            xc_W_cpa = 1.0 - float(sol_c[1])
            yw_C_cpa = float(sol_c[4])
            diff_xc  = abs(xc_W_tab - xc_W_cpa) / max(xc_W_cpa, 1e-10) * 100
            diff_yw  = abs(yw_C_tab - yw_C_cpa) / max(yw_C_cpa, 1e-10) * 100
        else:
            xc_W_cpa = yw_C_cpa = diff_xc = diff_yw = float('nan')

        print(f"{T_c:>6} {P_c:>8}  "
              f"{xc_W_cpa:>10.5f} {xc_W_tab:>10.5f} {diff_xc:>7.3f}%  "
              f"{yw_C_cpa:>10.5f} {yw_C_tab:>10.5f} {diff_yw:>7.3f}%")

    print("\nDone.")
