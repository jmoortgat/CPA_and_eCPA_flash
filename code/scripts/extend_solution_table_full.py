"""
Extend the solution table to cover all experimental conditions:

  Phase 1 — add ms levels [3.5, 4.0, 4.5, 5.0, 5.5, 6.0] for existing
             T = 283–623 K (using ms-continuation from ms=3.0)

  Phase 2 — add T levels [638, 653, 668, 683, 698, 713, 728 K] for ALL
             ms levels (0–6), using T-continuation from T=623 K

Result
------
  T_grid  : 31 values, 283–728 K (15 K steps)
  ms_grid : 14 values [0, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0,
                        3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
  shape   : (31, 30, 18, 14, 10)  = ~2.3 M elements

Strategy
--------
ELV direct solves (10-variable Newton via fsolve) throughout:
  - cheaper than full stability+flash per cell
  - ELV solutions are z-independent → solve once per (T, P, ms),
    replicate across the 18 z-values

Warm-start hierarchy
--------------------
Phase 1 (new ms for existing T):
  ms-continuation: 3.0 → 3.5 → 4.0 → 4.5 → 5.0 → 5.5 → 6.0
  Anchor: ms=3.0 solution already in the existing table.

Phase 2 (new T, all ms):
  ms=0 : T-continuation from the previous T level.
  ms>0 : ms-continuation at each T (0 → 0.1 → … → 6.0).

Non-converged cells filled by 2-D nearest-neighbour per ms slice,
consistent with the existing table build strategy.
"""

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    import shutil
    import numpy as np
    from scipy.optimize import fsolve
    from scipy.ndimage import distance_transform_edt

    from ecpa.parameters import make_params
    from ecpa.elv import ELV, ELV_jac, USE_COMPLEX_JAC
    from ecpa.solution_table import load_solution_table

    params = make_params()

    # ── Load existing solution table ──────────────────────────────────────────
    print("Loading existing solution table …")
    table_path = 'results/solution_table.npz'
    gd = load_solution_table(table_path)

    T_grid    = gd['T_grid']                    # (24,)  283–623 K
    logP_grid = gd['logP_grid']                 # (30,)
    z_grid    = gd['z_grid']                    # (18,)
    ms_grid   = gd['ms_grid']                   # (8,)  [0, 0.1, …, 3.0]
    sol_old   = gd['sol_filled']                # (24,30,18,8,10)
    msaq_old  = gd['ms_aq_filled']              # (24,30,18,8)
    stab_old  = gd['stable'].astype(bool)       # (24,30,18,8)

    nT, nP, nz, nms = sol_old.shape[:4]
    P_grid = 10.0 ** logP_grid

    print(f"  Existing grid : {nT}T × {nP}P × {nz}z × {nms}ms")
    print(f"  T range       : {T_grid[0]:.0f}–{T_grid[-1]:.0f} K")
    print(f"  ms_grid       : {ms_grid}")

    # ── Grid extensions ───────────────────────────────────────────────────────
    MS_NEW  = np.array([3.5, 4.0, 4.5, 5.0, 5.5, 6.0])
    T_NEW   = np.arange(638.0, 729.0, 15.0)   # 638, 653, 668, 683, 698, 713, 728
    ms_all  = np.concatenate([ms_grid, MS_NEW])  # (14,)

    nms_new = len(MS_NEW)
    nT_new  = len(T_NEW)
    nms_all = len(ms_all)

    print(f"\nNew ms levels   : {MS_NEW}")
    print(f"New T levels    : {T_NEW}")
    print(f"Full ms_grid    : {ms_all}")
    print(f"Final shape will be ({nT + nT_new}, {nP}, {nz}, {nms_all}, 10)")

    # ── ELV solver helper ─────────────────────────────────────────────────────
    def solve_elv(T, P_bar, ms, guess):
        """Run fsolve on ELV. Returns (sol, converged)."""
        try:
            sol, info, ier, _ = fsolve(
                ELV, guess,
                args=(float(T), float(P_bar) * 1e5, float(ms), params),
                fprime=ELV_jac if USE_COMPLEX_JAC else None,
                full_output=True, xtol=1e-10, maxfev=2000,
            )
            sol = np.asarray(sol, dtype=np.float64)
            res = np.asarray(
                ELV(sol, float(T), float(P_bar) * 1e5, float(ms), params),
                dtype=np.float64,
            )
            rn  = float(np.linalg.norm(res))
            ok  = (ier == 1
                   and np.all(np.isfinite(sol))
                   and rn < 1e-6
                   and 0.0 < sol[1] < 1.0
                   and 0.0 < sol[4] < 1.0
                   and sol[1] - sol[4] > 0.01)
        except Exception:
            sol = guess.copy()
            ok  = False
        return sol, ok

    def nn_fill_2d(sol_raw, conv):
        """Nearest-neighbour fill of (nT_or_P, nP_or_P, 10) for non-converged cells."""
        sol_filled = sol_raw.copy()
        invalid = ~conv
        if invalid.any():
            _, nn_idx = distance_transform_edt(invalid, return_indices=True)
            i0, i1   = nn_idx
            sol_filled[invalid] = sol_raw[i0[invalid], i1[invalid]]
        return sol_filled

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1: Add ms = [3.5 … 6.0] for existing T levels (283–623 K)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("PHASE 1: Extending ms to 6.0 for existing T levels (283–623 K)")
    print("=" * 65)

    # ms=3.0 anchor: index 7 in ms_grid (0, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    ims_3   = int(np.where(ms_grid == 3.0)[0][0])
    sol_ms3 = sol_old[:, :, 0, ims_3, :]    # (nT, nP, 10); z-independent → use iz=0

    sol_p1_raw  = np.full((nT, nP, nms_new, 10), np.nan)
    conv_p1     = np.zeros((nT, nP, nms_new), dtype=bool)

    for iT, T_i in enumerate(T_grid):
        n_anchor = np.sum(np.all(np.isfinite(sol_ms3[iT]), axis=-1))
        print(f"\n  T = {T_i:.0f} K  (ms=3.0 anchors: {n_anchor}/{nP})")
        for iP, P_i in enumerate(P_grid):
            prev_sol = sol_ms3[iT, iP].copy()
            if not np.all(np.isfinite(prev_sol)):
                continue
            for ims_n, ms_k in enumerate(MS_NEW):
                sol_k, ok_k = solve_elv(T_i, P_i, ms_k, prev_sol)
                if ok_k:
                    sol_p1_raw[iT, iP, ims_n] = sol_k
                    conv_p1[iT, iP, ims_n]    = True
                    prev_sol = sol_k

        for ims_n, ms_k in enumerate(MS_NEW):
            n_ok = int(conv_p1[iT, :, ims_n].sum())
            print(f"    ms={ms_k:.1f}  converged {n_ok}/{nP}")

    # NN fill per new-ms slice (2-D in T × P space)
    print("\nPhase-1 nearest-neighbour fill …")
    sol_p1_filled  = np.full_like(sol_p1_raw, np.nan)
    conv_p1_filled = np.zeros((nT, nP, nms_new), dtype=bool)

    for ims_n, ms_k in enumerate(MS_NEW):
        raw  = sol_p1_raw[:, :, ims_n, :]   # (nT, nP, 10)
        conv = conv_p1[:, :, ims_n]          # (nT, nP)
        sol_p1_filled[:, :, ims_n, :] = nn_fill_2d(raw, conv)
        conv_p1_filled[:, :, ims_n]   = conv
        filled_extra = int((~conv).sum())
        print(f"  ms={ms_k:.1f}  orig {conv.sum()}/{nT*nP}  "
              f"(NN-filled {filled_extra} cells)")

    # Replicate across z and stack on ms axis
    sol_p1_5d = np.broadcast_to(
        sol_p1_filled[:, :, np.newaxis, :, :],
        (nT, nP, nz, nms_new, 10),
    ).copy()
    msaq_p1 = np.broadcast_to(
        MS_NEW[np.newaxis, np.newaxis, np.newaxis, :],
        (nT, nP, nz, nms_new),
    ).copy()
    stab_p1 = np.broadcast_to(
        conv_p1_filled[:, :, np.newaxis, :],
        (nT, nP, nz, nms_new),
    ).copy()

    # Intermediate table: (nT, nP, nz, nms_all, 10)
    sol_int  = np.concatenate([sol_old,  sol_p1_5d], axis=3)
    msaq_int = np.concatenate([msaq_old, msaq_p1],   axis=3)
    stab_int = np.concatenate([stab_old, stab_p1],   axis=3)

    print(f"\nIntermediate table shape : {sol_int.shape}")
    print(f"  ms_all : {ms_all}")
    print(f"  Two-phase cells: {stab_int.sum()}/{stab_int.size}")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2: Add T = [638 … 728 K] for ALL ms levels (0–6)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("PHASE 2: Extending T to 728 K for all ms levels (0–6)")
    print("=" * 65)

    sol_p2_raw  = np.full((nT_new, nP, nms_all, 10), np.nan)
    conv_p2     = np.zeros((nT_new, nP, nms_all), dtype=bool)

    # T-continuation anchor: ms=0 solutions at T=623 K (last row of intermediate)
    iT_last    = nT - 1          # index of T=623 K in intermediate table
    ims_0      = 0               # ms=0 is index 0 in ms_all
    prev_T_ms0 = sol_int[iT_last, :, 0, ims_0, :].copy()   # (nP, 10)

    for iT_n, T_i in enumerate(T_NEW):
        print(f"\n── T = {T_i:.0f} K ──────────────────────────────────────────────")
        anchor_ok = np.sum(np.all(np.isfinite(prev_T_ms0), axis=-1))
        print(f"   T-continuation anchors (ms=0): {anchor_ok}/{nP}")

        for iP, P_i in enumerate(P_grid):
            # ms=0: T-continuation
            guess_ms0 = prev_T_ms0[iP].copy()
            if not np.all(np.isfinite(guess_ms0)):
                continue
            sol_ms0, ok_ms0 = solve_elv(T_i, P_i, 0.0, guess_ms0)
            if ok_ms0:
                sol_p2_raw[iT_n, iP, 0] = sol_ms0
                conv_p2[iT_n, iP, 0]    = True

            # ms>0: ms-continuation
            prev_sol = sol_ms0 if ok_ms0 else guess_ms0
            for ims_k, ms_k in enumerate(ms_all[1:], start=1):
                sol_k, ok_k = solve_elv(T_i, P_i, ms_k, prev_sol)
                if ok_k:
                    sol_p2_raw[iT_n, iP, ims_k] = sol_k
                    conv_p2[iT_n, iP, ims_k]    = True
                    prev_sol = sol_k

        # Update T-continuation anchor for the next T
        prev_T_ms0 = sol_p2_raw[iT_n, :, 0].copy()

        # Summary
        for ims_k, ms_k in enumerate(ms_all):
            n_ok = int(conv_p2[iT_n, :, ims_k].sum())
            print(f"  ms={ms_k:.1f}  converged {n_ok}/{nP}")

    # NN fill per ms slice (2-D in T × P space for new T rows)
    print("\nPhase-2 nearest-neighbour fill …")
    sol_p2_filled  = np.full_like(sol_p2_raw, np.nan)
    conv_p2_filled = np.zeros((nT_new, nP, nms_all), dtype=bool)

    for ims_k, ms_k in enumerate(ms_all):
        raw  = sol_p2_raw[:, :, ims_k, :]   # (nT_new, nP, 10)
        conv = conv_p2[:, :, ims_k]          # (nT_new, nP)
        sol_p2_filled[:, :, ims_k, :] = nn_fill_2d(raw, conv)
        conv_p2_filled[:, :, ims_k]   = conv
        filled_extra = int((~conv).sum())
        print(f"  ms={ms_k:.1f}  orig {conv.sum()}/{nT_new*nP}  "
              f"(NN-filled {filled_extra} cells)")

    # Replicate across z
    sol_p2_5d = np.broadcast_to(
        sol_p2_filled[:, :, np.newaxis, :, :],
        (nT_new, nP, nz, nms_all, 10),
    ).copy()
    msaq_p2 = np.broadcast_to(
        ms_all[np.newaxis, np.newaxis, np.newaxis, :],
        (nT_new, nP, nz, nms_all),
    ).copy()
    stab_p2 = np.broadcast_to(
        conv_p2_filled[:, :, np.newaxis, :],
        (nT_new, nP, nz, nms_all),
    ).copy()

    # ── Assemble final table ──────────────────────────────────────────────────
    T_grid_final = np.concatenate([T_grid, T_NEW])
    sol_final    = np.concatenate([sol_int,  sol_p2_5d], axis=0)
    msaq_final   = np.concatenate([msaq_int, msaq_p2],   axis=0)
    stab_final   = np.concatenate([stab_int, stab_p2],   axis=0)

    nT_f, nP_f, nz_f, nms_f = stab_final.shape
    total  = stab_final.size
    conv_f = int(stab_final.sum())

    print(f"\n{'='*65}")
    print(f"Final table shape : {sol_final.shape}")
    print(f"T_grid  ({nT_f} values): {T_grid_final}")
    print(f"ms_grid ({nms_f} values): {ms_all}")
    print(f"Total cells          : {total:,}")
    print(f"Two-phase (converged): {conv_f:,}  ({100*conv_f/total:.1f}%)")

    # ── Back up and save ──────────────────────────────────────────────────────
    backup_path = table_path.replace('.npz', '_pre_fullext_bak.npz')
    shutil.copy2(table_path, backup_path)
    print(f"\nBacked up original → {backup_path}")

    np.savez_compressed(
        table_path,
        T_grid       = T_grid_final,
        logP_grid    = logP_grid,
        z_grid       = z_grid,
        ms_grid      = ms_all,
        sol_filled   = sol_final,
        ms_aq_filled = msaq_final,
        stable       = stab_final.astype(np.uint8),
    )
    print(f"Saved expanded table → {table_path}")
    print("\nDone. Run _run_validation_full.py next.")
