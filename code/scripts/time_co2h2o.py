"""
Re-run the CO2-H2O validation flash loop with wall-time measurement,
using flash_co2_h2o_salt_fast_kv at ms=1e-5 (consistent with NaCl timing script).
"""
import warnings; warnings.filterwarnings('ignore')
import time
import numpy as np
import pandas as pd

from ecpa.parameters import make_params
from ecpa.solution_table import load_solution_table, make_solution_guess_fn
from ecpa.flash import flash_co2_h2o_salt_fast_kv
from ecpa.validate_co2h2o import Z_CO2_RETRY

MS     = 1e-5
Z_DEF  = 0.5
Z_CANDS = [Z_DEF] + list(Z_CO2_RETRY)   # [0.5, 0.3, 0.1, 0.01]

params   = make_params()
gd       = load_solution_table()
guess_fn = make_solution_guess_fn(gd)

exp_df = pd.read_parquet('CO2_WATER_exp.parquet')
rows   = exp_df.to_dict('records')

wall_times = []
n_iter_all = []

print(f"Timing {len(rows)} CO2+H2O experimental conditions at ms={MS} …")
t_start = time.perf_counter()

for k, row in enumerate(rows):
    T = float(row['T_K'])
    P = float(row['P_bar'])

    t0 = time.perf_counter()
    converged = False

    for z in Z_CANDS:
        try:
            out = flash_co2_h2o_salt_fast_kv(
                T=T, P_bar=P, z_co2=z, m_tot=MS,
                solution_guess_fn=guess_fn, params=params,
            )
        except Exception:
            continue

        if out.get('phase') == 'single_phase':
            continue

        x_aq = out.get('x_aq', {})
        if x_aq.get('x1w', 0) <= 0:
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
