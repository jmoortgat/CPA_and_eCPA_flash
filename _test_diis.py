"""
Benchmark DIIS vs Jex vs standard SSI for CPA2 flash (CO2-H2O binary).

Runs all 631 experimental (T, P) conditions with each acceleration method
and compares convergence, iteration count, wall time, and solution accuracy.
"""
import time
import os
import numpy as np
import pandas as pd

import CPA2

# ── Load experimental data ─────────────────────────────────────────────────
df = pd.read_parquet('CO2_WATER_exp.parquet')
tp = df[['T_K', 'P_bar']].drop_duplicates().sort_values(['T_K', 'P_bar'])
T_all = tp['T_K'].values
P_all = tp['P_bar'].values
N = len(T_all)
print(f"Loaded {N} unique (T, P) conditions")

comps = CPA2.make_components_co2_h2o()
kw_base = dict(Omega=comps["Omega"], Tc=comps["Tc"], Pc=comps["Pc"],
               Mw=comps["Mw"], tol=1e-10, maxiter=1000)

# ── Strategies ─────────────────────────────────────────────────────────────
STRATEGIES = [
    ("standard",  False, "none"),
    ("jex",       True,  "jex"),
    ("diis",      True,  "diis"),
]
N_STRAT = len(STRATEGIES)

# Pre-allocate result arrays
conv  = np.zeros((N, N_STRAT), dtype=bool)
iters = np.zeros((N, N_STRAT), dtype=int)
resid = np.full((N, N_STRAT), np.nan)
t_ms  = np.full((N, N_STRAT), np.nan)
x_co2 = np.full((N, N_STRAT), np.nan)
y_co2 = np.full((N, N_STRAT), np.nan)

# ── Main loop ──────────────────────────────────────────────────────────────
t_start = time.time()
for i in range(N):
    T, P = float(T_all[i]), float(P_all[i])
    kij = CPA2.kij_ecpa(T)
    swc = CPA2.s14_ecpa(T)
    kw = {**kw_base, 'kij12': kij, 'swc': swc}

    for j, (name, acc, method) in enumerate(STRATEGIES):
        t0 = time.perf_counter()
        r = CPA2.tie_line_two_comp(T=T, P_bar=P, accelerated=acc,
                                   accel_method=method, **kw)
        t_ms[i, j] = (time.perf_counter() - t0) * 1000
        conv[i, j]  = r['converged']
        iters[i, j] = r['iterations']
        resid[i, j] = r['residual_norm']
        if r['converged']:
            x_co2[i, j] = r['x'][0]
            y_co2[i, j] = r['y'][0]

    if (i + 1) % 100 == 0:
        elapsed = time.time() - t_start
        print(f"  {i+1}/{N} done  ({elapsed:.1f}s)")

elapsed = time.time() - t_start
print(f"\nTotal time: {elapsed:.1f}s")

# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

names = [s[0] for s in STRATEGIES]
print(f"\n{'Method':>10}  {'Conv':>5}  {'Conv%':>6}  "
      f"{'Mean It':>8}  {'Med It':>7}  {'Max It':>7}  "
      f"{'Mean ms':>8}  {'Med ms':>7}")

for j, name in enumerate(names):
    c = conv[:, j]
    it_c = iters[c, j]
    tm_c = t_ms[c, j]
    print(f"{name:>10}  {c.sum():>5d}  {c.mean()*100:>5.1f}%  "
          f"{it_c.mean():>8.1f}  {np.median(it_c):>7.0f}  {it_c.max():>7d}  "
          f"{tm_c.mean():>8.2f}  {np.median(tm_c):>7.2f}")

# ── Robustness: cross-method comparison ────────────────────────────────────
print("\n" + "─"*80)
print("ROBUSTNESS: Points where methods disagree on convergence")
print("─"*80)

jex_idx = names.index("jex")
diis_idx = names.index("diis")
std_idx = names.index("standard")

jex_only = conv[:, jex_idx] & ~conv[:, diis_idx]
diis_only = conv[:, diis_idx] & ~conv[:, jex_idx]
neither = ~conv[:, jex_idx] & ~conv[:, diis_idx] & conv[:, std_idx]

print(f"  Jex converges but DIIS fails: {jex_only.sum()} points")
if jex_only.sum() > 0:
    for i in np.where(jex_only)[0]:
        print(f"    T={T_all[i]:.0f}K  P={P_all[i]:.0f}bar  "
              f"jex_iter={iters[i,jex_idx]}  diis_iter={iters[i,diis_idx]}")

print(f"  DIIS converges but Jex fails: {diis_only.sum()} points")
if diis_only.sum() > 0:
    for i in np.where(diis_only)[0]:
        print(f"    T={T_all[i]:.0f}K  P={P_all[i]:.0f}bar  "
              f"diis_iter={iters[i,diis_idx]}  jex_iter={iters[i,jex_idx]}")

print(f"  Standard converges but both Jex+DIIS fail: {neither.sum()} points")

# ── Solution consistency ───────────────────────────────────────────────────
print("\n" + "─"*80)
print("SOLUTION CONSISTENCY (where both converge)")
print("─"*80)

both_conv = conv[:, jex_idx] & conv[:, diis_idx]
if both_conv.sum() > 0:
    dx = np.abs(x_co2[both_conv, jex_idx] - x_co2[both_conv, diis_idx])
    dy = np.abs(y_co2[both_conv, jex_idx] - y_co2[both_conv, diis_idx])
    print(f"  N both converge: {both_conv.sum()}")
    print(f"  max |Δx_CO2|: {dx.max():.2e}")
    print(f"  max |Δy_CO2|: {dy.max():.2e}")
    n_diff = ((dx > 1e-6) | (dy > 1e-6)).sum()
    print(f"  Points with |Δ| > 1e-6: {n_diff}")
    if n_diff > 0:
        idx_diff = np.where(both_conv)[0][(dx > 1e-6) | (dy > 1e-6)]
        for i in idx_diff[:10]:
            print(f"    T={T_all[i]:.0f}K P={P_all[i]:.0f}bar  "
                  f"jex: x={x_co2[i,jex_idx]:.6f}  diis: x={x_co2[i,diis_idx]:.6f}")

# ── Iteration comparison: DIIS vs Jex ─────────────────────────────────────
print("\n" + "─"*80)
print("ITERATION COMPARISON (where both converge)")
print("─"*80)

if both_conv.sum() > 0:
    it_jex  = iters[both_conv, jex_idx]
    it_diis = iters[both_conv, diis_idx]

    diis_faster = (it_diis < it_jex).sum()
    diis_same   = (it_diis == it_jex).sum()
    diis_slower = (it_diis > it_jex).sum()

    print(f"  DIIS faster:  {diis_faster} ({diis_faster/both_conv.sum()*100:.0f}%)")
    print(f"  Same:         {diis_same} ({diis_same/both_conv.sum()*100:.0f}%)")
    print(f"  DIIS slower:  {diis_slower} ({diis_slower/both_conv.sum()*100:.0f}%)")
    print(f"  Mean ratio (DIIS/Jex): {it_diis.mean()/it_jex.mean():.2f}")
    print(f"  Median DIIS: {np.median(it_diis):.0f}  Median Jex: {np.median(it_jex):.0f}")

    # Worst DIIS regressions
    ratio = it_diis / np.maximum(it_jex, 1)
    worst = np.argsort(-ratio)[:5]
    print("\n  Worst DIIS regressions:")
    bc_idx = np.where(both_conv)[0]
    for w in worst:
        i = bc_idx[w]
        print(f"    T={T_all[i]:.0f}K P={P_all[i]:.0f}bar  "
              f"jex={iters[i,jex_idx]}  diis={iters[i,diis_idx]}  "
              f"ratio={ratio[w]:.1f}x")

# ── Per-temperature summary ────────────────────────────────────────────────
print("\n" + "─"*80)
print("PER-TEMPERATURE SUMMARY (mean iterations, converged only)")
print("─"*80)

print(f"{'T [K]':>6}  {'N':>3}  {'std':>5}  {'jex':>5}  {'diis':>5}  "
      f"{'std_conv':>8}  {'jex_conv':>8}  {'diis_conv':>9}")

for T in sorted(set(T_all)):
    mask = T_all == T
    n = mask.sum()
    vals = []
    conv_vals = []
    for j in range(N_STRAT):
        c_mask = mask & conv[:, j]
        if c_mask.sum() > 0:
            vals.append(f"{iters[c_mask, j].mean():>5.1f}")
        else:
            vals.append("  ---")
        conv_vals.append(f"{c_mask.sum():>3d}/{n}")
    print(f"{T:>6.0f}  {n:>3d}  {vals[0]}  {vals[1]}  {vals[2]}  "
          f"{conv_vals[0]:>8}  {conv_vals[1]:>8}  {conv_vals[2]:>9}")

print("\nDone.")
