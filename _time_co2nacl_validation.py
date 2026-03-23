"""
Re-run the CO2-NaCl validation flash loop with wall-time measurement.
Prints timing statistics (median, mean, percentiles) for the 689 converged calls.
"""
import warnings; warnings.filterwarnings('ignore')
import time
import numpy as np
import pandas as pd

from ecpa.parameters import make_params
from ecpa.solution_table import load_solution_table, make_solution_guess_fn
from ecpa.flash import flash_co2_h2o_salt_fast_kv
from ecpa.validate_nacl import load_co2nacl_exp
from ecpa.constants import Mw

params   = make_params()
gd       = load_solution_table('results/solution_table.npz')
guess_fn = make_solution_guess_fn(gd)

exp_df = load_co2nacl_exp(force_reparse=True)
rows   = exp_df.to_dict('records')

def _z_candidates(qty, val_exp):
    if qty == 'mc':
        x4w = val_exp * Mw / (1.0 + val_exp * Mw)
        z_mid = float(np.clip(x4w * 1.5, 0.05, 0.85))
        return [z_mid, 0.3, 0.5, 0.15, 0.65, 0.8]
    elif qty in ('xc_W_SALTfree', 'xc_W_SALTincl'):
        x4w = float(val_exp)
        z_mid = float(np.clip(x4w * 1.5, 0.05, 0.85))
        return [z_mid, 0.3, 0.5, 0.15, 0.65]
    elif qty == 'xc_C':
        return [0.7, 0.5, 0.85, 0.3]
    return [0.3, 0.5, 0.15, 0.65]

wall_times = []   # ms, for converged two-phase calls
n_iter_all = []

print(f"Timing {len(rows)} experimental conditions …")
t_start = time.perf_counter()

for k, row in enumerate(rows):
    T   = float(row['T_K'])
    P   = float(row['P_bar'])
    ms  = float(row['ms'])
    qty = row['qty']
    val = float(row['value'])

    candidates = _z_candidates(qty, val)
    t0 = time.perf_counter()
    converged = False

    for z in candidates:
        try:
            out = flash_co2_h2o_salt_fast_kv(
                T=T, P_bar=P, z_co2=z, m_tot=ms,
                solution_guess_fn=guess_fn, params=params,
            )
        except Exception:
            continue

        if out.get('phase') == 'single_phase':
            continue

        ms_aq = float(out.get('ms_aq', -1))
        if ms_aq <= 0 or abs(ms_aq - ms) / max(ms, 0.1) > 0.6:
            continue

        x_aq = out['x_aq']
        if x_aq['x1w'] <= 0:
            continue

        t_ms = (time.perf_counter() - t0) * 1e3
        wall_times.append(t_ms)
        n_iter_all.append(int(out.get('n_iter_ms', -1)))
        converged = True
        break

    if (k + 1) % 100 == 0:
        print(f"  {k+1}/{len(rows)}  converged so far: {len(wall_times)}", flush=True)

elapsed = time.perf_counter() - t_start
print(f"\nDone in {elapsed:.1f}s")
print(f"Converged two-phase: {len(wall_times)}/{len(rows)}")

wt = np.array(wall_times)
print(f"\nWall time per call (ms):")
print(f"  median = {np.median(wt):.1f}")
print(f"  mean   = {np.mean(wt):.1f}")
print(f"  p75    = {np.percentile(wt,75):.1f}")
print(f"  p90    = {np.percentile(wt,90):.1f}")
print(f"  p95    = {np.percentile(wt,95):.1f}")
print(f"  max    = {np.max(wt):.1f}")

ni = np.array(n_iter_all)
print(f"\nSSI iterations:")
print(f"  median = {np.median(ni):.1f}")
print(f"  mean   = {np.mean(ni):.2f}")
print(f"  max    = {np.max(ni):.0f}")
