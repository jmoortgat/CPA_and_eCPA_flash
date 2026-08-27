"""
_bench_cold_start.py — Cold-start stability + flash benchmark.

No solution table.  For each grid point:
  1. Run ecpa_stability (cold) → get tpd, trial compositions (x1c_trial, x1w_trial)
  2. If unstable, run four flash strategies:
       A. SSI cold    : flash_co2_h2o_salt_ssi via guess_table_fn (ms=0 cold start)
       B. SSI+stab    : flash_co2_h2o_salt_ssi, warm-started from stability trial comps
       C. KV cold     : flash_co2_h2o_salt_kv,  K_init=None (K1=0.005, K4=30)
       D. KV+stab     : flash_co2_h2o_salt_kv,  K-values derived from stability trials

Goal: compare convergence rate, iteration count, and speed across methods.

Usage:
    cd /Users/moortgat/Software/2026/eCPA_SALTbasis/eCPA_improvements
    python _bench_cold_start.py
"""
import sys, time, warnings
import numpy as np

sys.path.insert(0, ".")

from ecpa.parameters import make_params
from ecpa.constants import Mw as Mw_H2O
from ecpa.guess_table import load_cpa_guess_table, make_guess_fn
from ecpa.stability import ecpa_stability
from ecpa.flash import flash_co2_h2o_salt_ssi, flash_co2_h2o_salt_kv

params = make_params()

# Load the salt-free CPA guess table (needed for SSI cold start)
PARQUET = "CO2/CPA_ELV_all.parquet"  # local data dir (see REPRODUCING_FIGURES.md)
print(f"Loading CPA guess table: {PARQUET}")
_grps, _temps = load_cpa_guess_table(PARQUET)
guess_table_fn = make_guess_fn(_grps, _temps)
print("Done.\n")

# ── Grid ──────────────────────────────────────────────────────────────────────
T_grid  = np.array([313., 343., 373., 403., 433., 463., 493.])
P_grid  = np.array([10., 25., 50., 75., 100., 150., 200., 300., 500., 800.])
z_grid  = np.array([0.10, 0.20, 0.35, 0.50, 0.65, 0.80])
ms_grid = np.array([0.5, 1.5, 3.0])

N_total = len(T_grid)*len(P_grid)*len(z_grid)*len(ms_grid)
print(f"Grid: {len(T_grid)}T × {len(P_grid)}P × {len(z_grid)}z × {len(ms_grid)}ms "
      f"= {N_total} total points")
print()

# ── Step 1: cold stability ─────────────────────────────────────────────────────
print("Step 1: Running cold stability analysis …")
t0 = time.perf_counter()

two_phase_pts    = []   # (T, P, z, ms, x1c_trial, x1w_trial)
single_phase_pts = 0
stab_fail_pts    = 0

for T in T_grid:
    for P in P_grid:
        for z in z_grid:
            for ms in ms_grid:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        stab = ecpa_stability(z, ms, T, P, params)
                        if not stab["stable"]:
                            two_phase_pts.append(
                                (T, P, z, ms,
                                 float(stab["x1c_trial"]),
                                 float(stab["x1w_trial"])))
                        else:
                            single_phase_pts += 1
                    except Exception:
                        stab_fail_pts += 1

t_stab = time.perf_counter() - t0
N2 = len(two_phase_pts)
print(f"  Done in {t_stab:.1f}s  ({1000*t_stab/N_total:.2f} ms/point)")
print(f"  Two-phase   : {N2} / {N_total} ({100*N2/N_total:.1f}%)")
print(f"  Single-phase: {single_phase_pts}  |  Stability failed: {stab_fail_pts}")
print()

if N2 == 0:
    print("No two-phase points found — check grid.")
    sys.exit(1)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt(n, iters, t, N):
    if len(iters) == 0:
        return (f"  Conv: {n}/{N} ({100*n/N:.1f}%)  "
                f"mean_iter=—  median=—  max=—  "
                f"time={t:.2f}s  ms/pt=—")
    return (f"  Conv: {n}/{N} ({100*n/N:.1f}%)  "
            f"mean_iter={iters.mean():.2f}  median={np.median(iters):.0f}  "
            f"max={iters.max()}"
            f"  time={t:.2f}s  ms/pt={1000*t/max(n,1):.2f}")


def stab_to_K(x1c_trial, x1w_trial, ms):
    """Convert stability trial compositions to K-values for flash seed."""
    x2w = x1w_trial * ms * Mw_H2O
    x4w = max(1.0 - x1w_trial - 2.0*x2w, 1e-6)
    x4c = max(1.0 - x1c_trial, 1e-6)
    K1  = x1c_trial / max(x1w_trial, 1e-9)
    K4  = x4c / x4w
    K1  = float(np.clip(K1, 1e-9, 1.0 - 1e-9))
    K4  = float(np.clip(K4, 1.0 + 1e-9, 1e6))
    return K1, K4


def run_ssi(pts, warm_from_stab=False):
    n_conv = 0; n_fail = 0; iters = []
    t0 = time.perf_counter()
    for T, P, z, ms, x1c_tr, x1w_tr in pts:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                if warm_from_stab:
                    # Seed from stability trial: build partial sol vector
                    # [Zw, x1w, epsr, Zc, x1c, chi1w, chi1c, d1, d2, d3]
                    # Use rough guesses for EOS variables; x1w/x1c come from stability
                    sol0 = np.array([0.08, x1w_tr, 70.0, 0.8, x1c_tr,
                                     0.4, 0.99, 0.0, 0.0, 0.0])
                    out = flash_co2_h2o_salt_ssi(
                        T=T, P_bar=P, z_co2=z, m_tot=ms, params=params,
                        initial_sol=sol0, initial_ms_aq=ms)
                else:
                    out = flash_co2_h2o_salt_ssi(
                        T=T, P_bar=P, z_co2=z, m_tot=ms, params=params,
                        guess_table_fn=guess_table_fn)
                n_conv += 1
                iters.append(int(out.get("n_iter_ms", 0)))
            except Exception:
                n_fail += 1
    elapsed = time.perf_counter() - t0
    return n_conv, np.array(iters, dtype=float), elapsed


def run_kv(pts, warm_from_stab=False):
    n_conv = 0; n_fail = 0; iters = []
    t0 = time.perf_counter()
    for T, P, z, ms, x1c_tr, x1w_tr in pts:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                K_init = None
                if warm_from_stab:
                    K_init = stab_to_K(x1c_tr, x1w_tr, ms)
                out = flash_co2_h2o_salt_kv(
                    T=T, P_bar=P, z_co2=z, m_tot=ms, params=params,
                    K_init=K_init)
                n_conv += 1
                iters.append(int(out.get("n_iter_ms", 0)))
            except Exception:
                n_fail += 1
    elapsed = time.perf_counter() - t0
    return n_conv, np.array(iters, dtype=float), elapsed


# ── Step 2: Flash comparison ──────────────────────────────────────────────────
print(f"Step 2: Flash on {N2} two-phase points\n")

labels  = []
results = {}

print("A. SSI cold  (ms=0 ELV via guess_table + ms_aq SSI) …")
n, it, t = run_ssi(two_phase_pts, warm_from_stab=False)
labels.append("SSI cold"); results["SSI cold"] = (n, it, t)
print(_fmt(n, it, t, N2), "\n")

print("B. SSI+stab  (stability x1c/x1w as starting ELV guess + ms_aq SSI) …")
n, it, t = run_ssi(two_phase_pts, warm_from_stab=True)
labels.append("SSI+stab"); results["SSI+stab"] = (n, it, t)
print(_fmt(n, it, t, N2), "\n")

print("C. KV cold   (K₁=0.005, K₄=30 default) …")
n, it, t = run_kv(two_phase_pts, warm_from_stab=False)
labels.append("KV cold"); results["KV cold"] = (n, it, t)
print(_fmt(n, it, t, N2), "\n")

print("D. KV+stab   (K-values from stability trial compositions) …")
n, it, t = run_kv(two_phase_pts, warm_from_stab=True)
labels.append("KV+stab"); results["KV+stab"] = (n, it, t)
print(_fmt(n, it, t, N2), "\n")

# ── Summary table ─────────────────────────────────────────────────────────────
print("=" * 72)
print(f"COLD-START SUMMARY  (stability: {1000*t_stab/N_total:.2f} ms/pt)")
print(f"{'Method':<16}  {'Conv%':>6}  {'Mean it':>8}  {'Flash ms/pt':>12}  "
      f"{'Total ms/pt':>12}")
print("-" * 72)
stab_ms = 1000 * t_stab / N_total
for label in labels:
    n, it, t = results[label]
    flash_ms = 1000*t/max(n,1) if n > 0 else float('nan')
    total_ms = (flash_ms if np.isfinite(flash_ms) else 0.0) + stab_ms
    mean_it  = it.mean() if len(it) > 0 else float('nan')
    print(f"  {label:<14}  {100*n/N2:>5.1f}%  {mean_it:>8.2f}  "
          f"{flash_ms:>12.2f}  {total_ms:>12.2f}")
print()
print("'Total ms/pt' = flash cost per converged point + stability cost per point.")
print("Stability is always required; its cost is shared regardless of flash method.")
