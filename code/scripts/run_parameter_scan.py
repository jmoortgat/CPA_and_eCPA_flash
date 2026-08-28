"""
_scan_parameter_space_extended.py — Extended CPA flash parameter-space scan.

Same logic as _scan_parameter_space.py but with expanded grid:
  T: 273–700 K (step 5 K, 86 points)
  P: 1–1500 bar (20 points, semi-log)
  z: same 19 points

Saves to results/scan_results_extended.npz
"""
import os
import time
import sys
import numpy as np

import CPA

# ═══════════════════════════════════════════════════════════════════════════════
# Extended grid definition
# ═══════════════════════════════════════════════════════════════════════════════
T_grid = np.arange(273, 701, 5, dtype=float)       # 273–700 K, step 5 K  (86 pts)
P_grid = np.array([1, 2, 5, 10, 20, 50, 75, 100,
                   150, 200, 300, 400, 500, 600,
                   800, 1000, 1200, 1500], dtype=float)  # 18 pts
z_grid = np.array([0.001, 0.005, 0.01, 0.02, 0.05,
                   0.1, 0.2, 0.3, 0.4, 0.5,
                   0.6, 0.7, 0.8, 0.9, 0.95,
                   0.98, 0.99, 0.995, 0.999],
                  dtype=float)                       # 19 pts

nT, nP, nz = len(T_grid), len(P_grid), len(z_grid)
N_TRIALS = 6

print(f"Grid: {nT} T x {nP} P x {nz} z = {nT*nP*nz:,} points")
print(f"T: {T_grid[0]}–{T_grid[-1]} K   P: {P_grid[0]}–{P_grid[-1]} bar   "
      f"z_CO2: {z_grid[0]}–{z_grid[-1]}")

# ═══════════════════════════════════════════════════════════════════════════════
# Pre-allocate output arrays
# ═══════════════════════════════════════════════════════════════════════════════
shape = (nT, nP, nz)
shape_trial = (nT, nP, nz, N_TRIALS)

stab_stable      = np.full(shape, False)
stab_tpd_min     = np.full(shape, np.nan)
stab_best_trial  = np.full(shape, -1, dtype=np.int8)
stab_n_unstable  = np.zeros(shape, dtype=np.int8)
stab_trial_tpd   = np.full(shape_trial, np.nan)
stab_trial_conv  = np.full(shape_trial, False)
stab_trial_iter  = np.zeros(shape_trial, dtype=np.int16)
stab_K           = np.full((*shape, 2), np.nan)

STRAT_NAMES = ["std_wilson", "acc_wilson", "acc_stabK", "robust"]
N_STRAT = len(STRAT_NAMES)

flash_conv       = np.full((*shape, N_STRAT), False)
flash_iter       = np.zeros((*shape, N_STRAT), dtype=np.int16)
flash_resid      = np.full((*shape, N_STRAT), np.nan)
flash_m          = np.full((*shape, N_STRAT), np.nan)

phase_id         = np.zeros(shape, dtype=np.int8)
_PHASE_MAP = {"failed": 0, "single_phase": 1, "single_liquid": 2,
              "single_vapor": 3, "two_phase": 4}
beta             = np.full(shape, np.nan)
x_co2            = np.full(shape, np.nan)
y_co2            = np.full(shape, np.nan)
K_final          = np.full((*shape, 2), np.nan)
Z_phase          = np.full((*shape, 2), np.nan)
rho_phase        = np.full((*shape, 2), np.nan)
chi_aq           = np.full((*shape, 2), np.nan)
chi_vap          = np.full((*shape, 2), np.nan)
assoc_t          = np.full((*shape, 2), np.nan)

swc_fallback     = np.full(shape, False)
robust_attempt   = np.zeros(shape, dtype=np.int8)
kij_arr          = np.full(shape[:1], np.nan)
swc_arr          = np.full(shape[:1], np.nan)
wall_time        = np.full(shape, np.nan)

# ═══════════════════════════════════════════════════════════════════════════════
# Component data
# ═══════════════════════════════════════════════════════════════════════════════
comps = CPA.make_components_co2_h2o()
Omega, Tc, Pc, Mw = comps["Omega"], comps["Tc"], comps["Pc"], comps["Mw"]
stab_trial_labels = None

# ═══════════════════════════════════════════════════════════════════════════════
# Main scan loop
# ═══════════════════════════════════════════════════════════════════════════════
t_start_global = time.time()
n_done = 0
n_total = nT * nP * nz

for iT, T in enumerate(T_grid):
    kij = CPA.kij_ecpa(T)
    swc = CPA.s14_ecpa(T)
    kij_arr[iT] = kij
    swc_arr[iT] = swc

    for iP, P in enumerate(P_grid):
        for iz, zc in enumerate(z_grid):
            t0 = time.time()
            z = np.array([zc, 1.0 - zc])

            kw_base = dict(Omega=Omega, Tc=Tc, Pc=Pc, Mw=Mw,
                           kij12=kij, swc=swc, tol=1e-10, maxiter=1000)

            # ── Stability test ──────────────────────────────────────────
            stab = CPA.stability_test(T, P, z, accelerated=True,
                                       Omega=Omega, Tc=Tc, Pc=Pc, Mw=Mw,
                                       kij12=kij, swc=swc, maxiter=200)

            stab_stable[iT, iP, iz] = stab["stable"]
            stab_tpd_min[iT, iP, iz] = stab["tpd_min"]
            if stab["K_unstable"] is not None:
                stab_K[iT, iP, iz] = stab["K_unstable"]

            trials = stab["trials"]
            if stab_trial_labels is None:
                stab_trial_labels = [t[0] for t in trials]

            best_idx = 0; best_tpd = 0.0
            n_unst = 0
            for jt, (lbl, tpd_val, conv, nit) in enumerate(trials):
                if jt >= N_TRIALS:
                    break
                stab_trial_tpd[iT, iP, iz, jt] = tpd_val
                stab_trial_conv[iT, iP, iz, jt] = conv
                stab_trial_iter[iT, iP, iz, jt] = nit
                if tpd_val < best_tpd:
                    best_tpd = tpd_val; best_idx = jt
                if tpd_val < -1e-7:
                    n_unst += 1

            stab_best_trial[iT, iP, iz] = best_idx
            stab_n_unstable[iT, iP, iz] = n_unst

            # ── Flash strategy 0: standard SSI + Wilson K ───────────────
            tie0 = CPA.tie_line_two_comp(T=T, P_bar=P, accelerated=False,
                                          **kw_base)
            flash_conv[iT, iP, iz, 0] = tie0["converged"]
            flash_iter[iT, iP, iz, 0] = tie0["iterations"]
            flash_resid[iT, iP, iz, 0] = tie0["residual_norm"]
            flash_m[iT, iP, iz, 0] = tie0["final_m"]

            # ── Flash strategy 1: accelerated SSI + Wilson K ────────────
            tie1 = CPA.tie_line_two_comp(T=T, P_bar=P, accelerated=True,
                                          **kw_base)
            flash_conv[iT, iP, iz, 1] = tie1["converged"]
            flash_iter[iT, iP, iz, 1] = tie1["iterations"]
            flash_resid[iT, iP, iz, 1] = tie1["residual_norm"]
            flash_m[iT, iP, iz, 1] = tie1["final_m"]

            # ── Flash strategy 2: accelerated SSI + stability K ─────────
            K_s = stab["K_unstable"]
            if (K_s is not None and np.all(np.isfinite(K_s))
                    and np.all(K_s > 0)):
                tie2 = CPA.tie_line_two_comp(
                    T=T, P_bar=P, K_init=K_s, accelerated=True, **kw_base)
            else:
                tie2 = tie1
            flash_conv[iT, iP, iz, 2] = tie2["converged"]
            flash_iter[iT, iP, iz, 2] = tie2["iterations"]
            flash_resid[iT, iP, iz, 2] = tie2["residual_norm"]
            flash_m[iT, iP, iz, 2] = tie2["final_m"]

            # ── Strategy 3: "robust" ────────────────────────────────────
            has_stab_K = (K_s is not None and np.all(np.isfinite(K_s))
                          and np.all(K_s > 0))

            if stab["stable"]:
                ph = "single_phase"
                flash_conv[iT, iP, iz, 3] = True
                flash_iter[iT, iP, iz, 3] = 0
                flash_resid[iT, iP, iz, 3] = 0.0
                x_co2[iT, iP, iz] = zc
                y_co2[iT, iP, iz] = zc
                robust_attempt[iT, iP, iz] = 0
            else:
                best_tie = None; attempt_code = 5
                if has_stab_K and tie2["converged"]:
                    best_tie = tie2; attempt_code = 1
                elif tie1["converged"]:
                    best_tie = tie1; attempt_code = 2
                elif 0 < abs(swc) < 0.005:
                    kw_s0 = dict(kw_base, swc=0.0)
                    if has_stab_K:
                        tie_s0 = CPA.tie_line_two_comp(
                            T=T, P_bar=P, K_init=K_s,
                            accelerated=True, **kw_s0)
                        if tie_s0["converged"]:
                            best_tie = tie_s0; attempt_code = 3
                    if best_tie is None:
                        tie_s0w = CPA.tie_line_two_comp(
                            T=T, P_bar=P, accelerated=True, **kw_s0)
                        if tie_s0w["converged"]:
                            best_tie = tie_s0w; attempt_code = 4
                    if best_tie is not None:
                        swc_fallback[iT, iP, iz] = True

                robust_attempt[iT, iP, iz] = attempt_code

                if best_tie is not None and best_tie["converged"]:
                    flash_conv[iT, iP, iz, 3] = True
                    flash_iter[iT, iP, iz, 3] = best_tie["iterations"]
                    flash_resid[iT, iP, iz, 3] = best_tie["residual_norm"]
                    flash_m[iT, iP, iz, 3] = best_tie["final_m"]

                    x0 = float(best_tie["x"][0])
                    y0 = float(best_tie["y"][0])
                    denom_b = y0 - x0
                    if abs(denom_b) > 1e-14:
                        beta_val = (zc - x0) / denom_b
                        if beta_val <= 0:
                            ph = "single_liquid"; beta_val = 0.0
                        elif beta_val >= 1:
                            ph = "single_vapor"; beta_val = 1.0
                        else:
                            ph = "two_phase"
                        beta[iT, iP, iz] = beta_val
                    else:
                        ph = "failed"

                    x_co2[iT, iP, iz] = x0
                    y_co2[iT, iP, iz] = y0
                    K_final[iT, iP, iz] = best_tie["K"]
                    if "Z" in best_tie:
                        Z_phase[iT, iP, iz] = best_tie["Z"]
                    if "rho_mass" in best_tie:
                        rho_phase[iT, iP, iz] = best_tie["rho_mass"]
                    if "assoc_t" in best_tie:
                        assoc_t[iT, iP, iz] = best_tie["assoc_t"]
                    if "chi" in best_tie:
                        chi_aq[iT, iP, iz] = best_tie["chi"]["liq"]
                        chi_vap[iT, iP, iz] = best_tie["chi"]["vap"]
                else:
                    ph = "failed"

            phase_id[iT, iP, iz] = _PHASE_MAP.get(ph, 0)
            wall_time[iT, iP, iz] = time.time() - t0
            n_done += 1

        elapsed = time.time() - t_start_global
        rate = n_done / elapsed if elapsed > 0 else 0
        eta = (n_total - n_done) / rate if rate > 0 else 0
        pct = 100 * n_done / n_total
        sys.stdout.write(
            f"\r  T={T:6.0f}K  P={P:6.0f}bar  "
            f"[{n_done:>6d}/{n_total}  {pct:5.1f}%]  "
            f"{rate:.0f} pts/s  ETA {eta/60:.1f} min   ")
        sys.stdout.flush()

print()
t_total = time.time() - t_start_global
print(f"\nScan complete: {n_total:,} points in {t_total:.1f}s "
      f"({n_total/t_total:.0f} pts/s)")

# ═══════════════════════════════════════════════════════════════════════════════
# Summary statistics
# ═══════════════════════════════════════════════════════════════════════════════
n_failed = int(np.sum(phase_id == 0))
n_single = int(np.sum(np.isin(phase_id, [1, 2, 3])))
n_twoph  = int(np.sum(phase_id == 4))
print(f"\nPhase identification:")
print(f"  Two-phase:    {n_twoph:>6d}  ({100*n_twoph/n_total:.1f}%)")
print(f"  Single-phase: {n_single:>6d}  ({100*n_single/n_total:.1f}%)")
print(f"  Failed:       {n_failed:>6d}  ({100*n_failed/n_total:.1f}%)")

mask_2ph = (phase_id == 4)
print(f"\nFlash convergence by strategy (two-phase points only, n={n_twoph}):")
for s, name in enumerate(STRAT_NAMES):
    nc = int(np.sum(flash_conv[..., s][mask_2ph]))
    conv_mask = mask_2ph & flash_conv[..., s]
    avg_it = np.nanmean(flash_iter[..., s][conv_mask]) if np.any(conv_mask) else 0
    print(f"  {name:15s}: {nc:>6d}/{n_twoph}  "
          f"({100*nc/max(n_twoph,1):.1f}%)  avg iter = {avg_it:.1f}")

n_unstable = int(np.sum(~stab_stable))
print(f"\nStability initial guesses needed (unstable points, n={n_unstable}):")
for ng in range(1, N_TRIALS + 1):
    n_first = int(np.sum(stab_best_trial[~stab_stable] < ng))
    print(f"  <={ng} guesses sufficient: {n_first} "
          f"({100*n_first/max(n_unstable,1):.1f}%)")

print(f"\nRobust wrapper attempt distribution:")
for code, label in [(0, "stability->single"), (1, "acc+stabK"),
                     (2, "acc+wilson"), (3, "swc0+stabK"),
                     (4, "swc0+wilson"), (5, "failed")]:
    nc = int(np.sum(robust_attempt == code))
    print(f"  {label:20s}: {nc:>6d}  ({100*nc/n_total:.1f}%)")

n_swc = int(np.sum(swc_fallback))
print(f"\nswc->0 fallback needed: {n_swc} ({100*n_swc/n_total:.2f}%)")

mask_both = flash_conv[..., 0] & flash_conv[..., 1]
if np.any(mask_both):
    it_std = flash_iter[..., 0][mask_both].astype(float)
    it_acc = flash_iter[..., 1][mask_both].astype(float)
    ratio = it_std / np.maximum(it_acc, 1)
    print(f"\nAcceleration speedup (std vs acc Wilson, n={int(np.sum(mask_both))}):")
    print(f"  Mean iter:  std={np.mean(it_std):.1f}  acc={np.mean(it_acc):.1f}  "
          f"ratio={np.mean(ratio):.2f}")
    print(f"  Median iter: std={np.median(it_std):.0f}  acc={np.median(it_acc):.0f}")
    print(f"  Accelerated faster: {100*np.mean(it_acc < it_std):.1f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════════════════════════════
os.makedirs("results", exist_ok=True)
outfile = "results/scan_results_extended.npz"
np.savez_compressed(
    outfile,
    T_grid=T_grid, P_grid=P_grid, z_grid=z_grid,
    kij_arr=kij_arr, swc_arr=swc_arr,
    stab_stable=stab_stable, stab_tpd_min=stab_tpd_min,
    stab_best_trial=stab_best_trial, stab_n_unstable=stab_n_unstable,
    stab_trial_tpd=stab_trial_tpd, stab_trial_conv=stab_trial_conv,
    stab_trial_iter=stab_trial_iter, stab_K=stab_K,
    stab_trial_labels=np.array(stab_trial_labels if stab_trial_labels else []),
    flash_strategy_names=np.array(STRAT_NAMES),
    flash_conv=flash_conv, flash_iter=flash_iter,
    flash_resid=flash_resid, flash_m=flash_m,
    phase_id=phase_id, phase_map=np.array(list(_PHASE_MAP.keys())),
    beta=beta, x_co2=x_co2, y_co2=y_co2,
    K_final=K_final, Z_phase=Z_phase, rho_phase=rho_phase,
    chi_aq=chi_aq, chi_vap=chi_vap, assoc_t=assoc_t,
    swc_fallback=swc_fallback, robust_attempt=robust_attempt,
    wall_time=wall_time,
)
print(f"\nResults saved to {outfile}")
print(f"Array shapes: main={shape}, trials={shape_trial}, "
      f"strategies={(*shape, N_STRAT)}")
