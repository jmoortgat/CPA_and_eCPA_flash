"""
_scan_exp_points.py — Run CPA stability + flash at every experimental CO2-H2O data point.

Loads CO2_WATER_exp.parquet (631 points, T=273–623 K, P=5–3500 bar),
runs the full CPA hierarchical stability+flash at each (T, P) with z_CO2=0.5,
and reports convergence/iteration statistics.

Saves detailed per-point results to scan_exp_results.npz.
"""
import time
import sys
import os
import numpy as np
import pandas as pd

import CPA

# ═══════════════════════════════════════════════════════════════════════════════
# Load experimental data
# ═══════════════════════════════════════════════════════════════════════════════
parquet_paths = [
    "CO2_WATER_exp.parquet",
    "CO2/CO2_WATER_exp.parquet",
]
df = None
for p in parquet_paths:
    if os.path.exists(p):
        df = pd.read_parquet(p)
        print(f"Loaded {p}: {len(df)} rows")
        break

if df is None:
    raise FileNotFoundError("Cannot find CO2_WATER_exp.parquet")

# Extract unique (T, P) pairs
tp_pairs = df[["T_K", "P_bar"]].drop_duplicates().sort_values(["T_K", "P_bar"])
T_vals = tp_pairs["T_K"].values
P_vals = tp_pairs["P_bar"].values
N = len(T_vals)
print(f"Unique (T, P) conditions: {N}")
print(f"T range: {T_vals.min():.1f}–{T_vals.max():.1f} K")
print(f"P range: {P_vals.min():.1f}–{P_vals.max():.1f} bar")

# Also run at each raw data point (may have duplicate T,P with different exp values)
T_all = df["T_K"].values
P_all = df["P_bar"].values
N_all = len(T_all)
print(f"Total data points (including duplicates): {N_all}")

# ═══════════════════════════════════════════════════════════════════════════════
# Component data
# ═══════════════════════════════════════════════════════════════════════════════
comps = CPA.make_components_co2_h2o()
Omega, Tc, Pc, Mw = comps["Omega"], comps["Tc"], comps["Pc"], comps["Mw"]

N_TRIALS = 6
STRAT_NAMES = ["std_wilson", "acc_wilson", "acc_stabK", "robust"]
N_STRAT = len(STRAT_NAMES)
z_co2_feed = 0.5  # standard feed composition for flash

# ═══════════════════════════════════════════════════════════════════════════════
# Pre-allocate arrays (one per data point)
# ═══════════════════════════════════════════════════════════════════════════════
stab_stable      = np.full(N_all, False)
stab_tpd_min     = np.full(N_all, np.nan)
stab_best_trial  = np.full(N_all, -1, dtype=np.int8)
stab_n_unstable  = np.zeros(N_all, dtype=np.int8)
stab_trial_tpd   = np.full((N_all, N_TRIALS), np.nan)
stab_trial_conv  = np.full((N_all, N_TRIALS), False)
stab_trial_iter  = np.zeros((N_all, N_TRIALS), dtype=np.int16)

flash_conv       = np.full((N_all, N_STRAT), False)
flash_iter       = np.zeros((N_all, N_STRAT), dtype=np.int16)
flash_resid      = np.full((N_all, N_STRAT), np.nan)
flash_m          = np.full((N_all, N_STRAT), np.nan)

phase_label      = np.full(N_all, "", dtype="U15")
robust_attempt   = np.zeros(N_all, dtype=np.int8)
swc_fallback     = np.full(N_all, False)
wall_time_arr    = np.full(N_all, np.nan)
x_co2_arr        = np.full(N_all, np.nan)
y_co2_arr        = np.full(N_all, np.nan)
beta_arr         = np.full(N_all, np.nan)

stab_trial_labels = None

# ═══════════════════════════════════════════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════════════════════════════════════════
t_start = time.time()

for i in range(N_all):
    T = float(T_all[i])
    P = float(P_all[i])
    t0 = time.time()

    kij = CPA.kij_ecpa(T)
    swc = CPA.s14_ecpa(T)
    z = np.array([z_co2_feed, 1.0 - z_co2_feed])

    kw_base = dict(Omega=Omega, Tc=Tc, Pc=Pc, Mw=Mw,
                   kij12=kij, swc=swc, tol=1e-10, maxiter=1000)

    # ── Stability test ──────────────────────────────────────────
    stab = CPA.stability_test(T, P, z, accelerated=True,
                               Omega=Omega, Tc=Tc, Pc=Pc, Mw=Mw,
                               kij12=kij, swc=swc, maxiter=200)

    stab_stable[i] = stab["stable"]
    stab_tpd_min[i] = stab["tpd_min"]

    trials = stab["trials"]
    if stab_trial_labels is None:
        stab_trial_labels = [t[0] for t in trials]

    best_idx = 0; best_tpd = 0.0; n_unst = 0
    for jt, (lbl, tpd_val, conv, nit) in enumerate(trials):
        if jt >= N_TRIALS:
            break
        stab_trial_tpd[i, jt] = tpd_val
        stab_trial_conv[i, jt] = conv
        stab_trial_iter[i, jt] = nit
        if tpd_val < best_tpd:
            best_tpd = tpd_val; best_idx = jt
        if tpd_val < -1e-7:
            n_unst += 1

    stab_best_trial[i] = best_idx
    stab_n_unstable[i] = n_unst

    # ── Flash strategy 0: standard SSI + Wilson K ───────────────
    tie0 = CPA.tie_line_two_comp(T=T, P_bar=P, accelerated=False, **kw_base)
    flash_conv[i, 0] = tie0["converged"]
    flash_iter[i, 0] = tie0["iterations"]
    flash_resid[i, 0] = tie0["residual_norm"]
    flash_m[i, 0] = tie0["final_m"]

    # ── Flash strategy 1: accelerated SSI + Wilson K ────────────
    tie1 = CPA.tie_line_two_comp(T=T, P_bar=P, accelerated=True, **kw_base)
    flash_conv[i, 1] = tie1["converged"]
    flash_iter[i, 1] = tie1["iterations"]
    flash_resid[i, 1] = tie1["residual_norm"]
    flash_m[i, 1] = tie1["final_m"]

    # ── Flash strategy 2: accelerated SSI + stability K ─────────
    K_s = stab["K_unstable"]
    if K_s is not None and np.all(np.isfinite(K_s)) and np.all(K_s > 0):
        tie2 = CPA.tie_line_two_comp(T=T, P_bar=P, K_init=K_s,
                                      accelerated=True, **kw_base)
    else:
        tie2 = tie1
    flash_conv[i, 2] = tie2["converged"]
    flash_iter[i, 2] = tie2["iterations"]
    flash_resid[i, 2] = tie2["residual_norm"]
    flash_m[i, 2] = tie2["final_m"]

    # ── Strategy 3: "robust" ────────────────────────────────────
    has_stab_K = (K_s is not None and np.all(np.isfinite(K_s))
                  and np.all(K_s > 0))

    if stab["stable"]:
        ph = "single_phase"
        flash_conv[i, 3] = True
        flash_iter[i, 3] = 0
        flash_resid[i, 3] = 0.0
        x_co2_arr[i] = z_co2_feed
        y_co2_arr[i] = z_co2_feed
        robust_attempt[i] = 0
    else:
        best_tie = None; attempt_code = 5
        if has_stab_K and tie2["converged"]:
            best_tie = tie2; attempt_code = 1
        elif tie1["converged"]:
            best_tie = tie1; attempt_code = 2
        elif 0 < abs(swc) < 0.005:
            kw_s0 = dict(kw_base, swc=0.0)
            if has_stab_K:
                tie_s0 = CPA.tie_line_two_comp(T=T, P_bar=P, K_init=K_s,
                                                accelerated=True, **kw_s0)
                if tie_s0["converged"]:
                    best_tie = tie_s0; attempt_code = 3
            if best_tie is None:
                tie_s0w = CPA.tie_line_two_comp(T=T, P_bar=P,
                                                 accelerated=True, **kw_s0)
                if tie_s0w["converged"]:
                    best_tie = tie_s0w; attempt_code = 4
            if best_tie is not None:
                swc_fallback[i] = True

        robust_attempt[i] = attempt_code

        if best_tie is not None and best_tie["converged"]:
            flash_conv[i, 3] = True
            flash_iter[i, 3] = best_tie["iterations"]
            flash_resid[i, 3] = best_tie["residual_norm"]
            flash_m[i, 3] = best_tie["final_m"]

            x0 = float(best_tie["x"][0])
            y0 = float(best_tie["y"][0])
            x_co2_arr[i] = x0
            y_co2_arr[i] = y0
            denom_b = y0 - x0
            if abs(denom_b) > 1e-14:
                beta_val = (z_co2_feed - x0) / denom_b
                beta_arr[i] = np.clip(beta_val, 0, 1)
                if beta_val <= 0:
                    ph = "single_liquid"
                elif beta_val >= 1:
                    ph = "single_vapor"
                else:
                    ph = "two_phase"
            else:
                ph = "failed"
        else:
            ph = "failed"

    phase_label[i] = ph
    wall_time_arr[i] = time.time() - t0

    if (i + 1) % 50 == 0 or i == N_all - 1:
        elapsed = time.time() - t_start
        rate = (i + 1) / elapsed
        eta = (N_all - i - 1) / rate
        sys.stdout.write(
            f"\r  [{i+1:>4d}/{N_all}]  T={T:.0f}K  P={P:.0f}bar  "
            f"{rate:.0f} pts/s  ETA {eta:.0f}s   ")
        sys.stdout.flush()

print()
t_total = time.time() - t_start
print(f"\nComplete: {N_all} points in {t_total:.1f}s ({N_all/t_total:.0f} pts/s)")

# ═══════════════════════════════════════════════════════════════════════════════
# Summary statistics
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY: CPA stability+flash at experimental CO2-H2O conditions")
print("=" * 70)
print(f"Total points: {N_all}")
print(f"T range: {T_all.min():.1f}–{T_all.max():.1f} K")
print(f"P range: {P_all.min():.1f}–{P_all.max():.1f} bar")

# Phase breakdown
for ph in ["single_phase", "two_phase", "single_liquid", "single_vapor", "failed"]:
    n = np.sum(phase_label == ph)
    if n > 0:
        print(f"  {ph:15s}: {n:>4d} ({100*n/N_all:.1f}%)")

# Stability
n_stable = np.sum(stab_stable)
n_unstable = np.sum(~stab_stable)
print(f"\nStability test: {n_stable} stable, {n_unstable} unstable")
if n_unstable > 0:
    for k in range(N_TRIALS):
        n_first = int(np.sum(stab_best_trial[~stab_stable] < k + 1))
        print(f"  <={k+1} guesses: {n_first}/{n_unstable} "
              f"({100*n_first/n_unstable:.1f}%)")

# Flash convergence
print(f"\nFlash convergence (all {N_all} points):")
for s, name in enumerate(STRAT_NAMES):
    nc = int(np.sum(flash_conv[:, s]))
    conv_mask = flash_conv[:, s]
    avg_it = np.mean(flash_iter[:, s][conv_mask]) if np.any(conv_mask) else 0
    med_it = np.median(flash_iter[:, s][conv_mask]) if np.any(conv_mask) else 0
    max_it = int(np.max(flash_iter[:, s][conv_mask])) if np.any(conv_mask) else 0
    print(f"  {name:15s}: {nc:>4d}/{N_all} ({100*nc/N_all:.1f}%)  "
          f"mean={avg_it:.1f}  med={med_it:.0f}  max={max_it}")

# Flash convergence (two-phase only)
mask_2ph = phase_label == "two_phase"
n2 = np.sum(mask_2ph)
if n2 > 0:
    print(f"\nFlash convergence (two-phase only, n={n2}):")
    for s, name in enumerate(STRAT_NAMES):
        nc = int(np.sum(flash_conv[:, s][mask_2ph]))
        conv_mask = flash_conv[:, s] & mask_2ph
        avg_it = np.mean(flash_iter[:, s][conv_mask]) if np.any(conv_mask) else 0
        print(f"  {name:15s}: {nc:>4d}/{n2} ({100*nc/n2:.1f}%)  "
              f"mean iter={avg_it:.1f}")

# Speedup
both_conv = flash_conv[:, 0] & flash_conv[:, 1]
if np.any(both_conv):
    it_std = flash_iter[:, 0][both_conv].astype(float)
    it_acc = flash_iter[:, 1][both_conv].astype(float)
    ratio = it_std / np.maximum(it_acc, 1)
    print(f"\nSpeedup (std vs acc Wilson, n={np.sum(both_conv)}):")
    print(f"  Mean: {np.mean(ratio):.2f}x  "
          f"({np.mean(it_std):.1f} -> {np.mean(it_acc):.1f} iters)")

# Robust attempt distribution
print(f"\nRobust attempt distribution:")
for code, label in [(0, "stable->single"), (1, "acc+stabK"),
                     (2, "acc+wilson"), (3, "swc0+stabK"),
                     (4, "swc0+wilson"), (5, "FAILED")]:
    nc = int(np.sum(robust_attempt == code))
    if nc > 0:
        print(f"  {label:20s}: {nc:>4d} ({100*nc/N_all:.1f}%)")

n_swc = int(np.sum(swc_fallback))
if n_swc > 0:
    print(f"\nswc->0 fallback needed: {n_swc}")

# Any failures?
n_failed = int(np.sum(phase_label == "failed"))
if n_failed > 0:
    print(f"\n*** {n_failed} FAILED POINTS ***")
    fail_idx = np.where(phase_label == "failed")[0]
    for idx in fail_idx[:20]:
        print(f"  T={T_all[idx]:.1f}K  P={P_all[idx]:.1f}bar  "
              f"attempt={robust_attempt[idx]}  stab_stable={stab_stable[idx]}")
else:
    print(f"\n*** ALL {N_all} POINTS CONVERGED ***")

# ═══════════════════════════════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════════════════════════════
outfile = "scan_exp_results.npz"
np.savez_compressed(
    outfile,
    T_K=T_all, P_bar=P_all,
    z_co2_feed=z_co2_feed,
    stab_stable=stab_stable, stab_tpd_min=stab_tpd_min,
    stab_best_trial=stab_best_trial, stab_n_unstable=stab_n_unstable,
    stab_trial_tpd=stab_trial_tpd, stab_trial_conv=stab_trial_conv,
    stab_trial_iter=stab_trial_iter,
    stab_trial_labels=np.array(stab_trial_labels if stab_trial_labels else []),
    flash_strategy_names=np.array(STRAT_NAMES),
    flash_conv=flash_conv, flash_iter=flash_iter,
    flash_resid=flash_resid, flash_m=flash_m,
    phase_label=phase_label,
    robust_attempt=robust_attempt, swc_fallback=swc_fallback,
    wall_time=wall_time_arr,
    x_co2=x_co2_arr, y_co2=y_co2_arr, beta=beta_arr,
)
print(f"\nResults saved to {outfile}")
