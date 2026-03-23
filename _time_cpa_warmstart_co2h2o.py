"""
Run the 631 CO2+H2O experimental conditions with warm-started CPA
(flash_co2_h2o_tpz_warmstart) and report timing + iteration statistics.
"""
import warnings; warnings.filterwarnings('ignore')
import time
import numpy as np
import pandas as pd

import CPA
from ecpa.solution_table import load_solution_table, make_solution_guess_fn
from ecpa.validate_co2h2o import Z_CO2_RETRY

Z_DEF   = 0.5
Z_CANDS = [Z_DEF] + list(Z_CO2_RETRY)   # [0.5, 0.3, 0.1, 0.01]

print("Loading solution table …")
gd       = load_solution_table()
guess_fn = make_solution_guess_fn(gd)

exp_df = pd.read_parquet('CO2_WATER_exp.parquet')
rows   = exp_df.to_dict('records')

wall_times_ws  = []   # warm-started
n_iter_ws      = []
wall_times_fb  = []   # fallback to robust
n_iter_fb      = []

print(f"Timing {len(rows)} CO2+H2O conditions with warm-started CPA …")
t_start = time.perf_counter()

for k, row in enumerate(rows):
    T = float(row['T_K'])
    P = float(row['P_bar'])

    t0 = time.perf_counter()
    converged = False

    for z in Z_CANDS:
        try:
            out = CPA.flash_co2_h2o_tpz_warmstart(
                T=T, P_bar=P, z_co2=z,
                solution_guess_fn=guess_fn,
            )
        except Exception:
            continue

        if out.get('phase') != 'two_phase':
            continue

        t_ms = (time.perf_counter() - t0) * 1e3
        if out.get('warmstarted'):
            wall_times_ws.append(t_ms)
            n_iter_ws.append(int(out.get('n_iter', -1)))
        else:
            wall_times_fb.append(t_ms)
            n_iter_fb.append(int(out.get('n_iter', -1)))
        converged = True
        break

    if (k + 1) % 100 == 0:
        print(f"  {k+1}/{len(rows)}  converged so far: {len(wall_times_ws)+len(wall_times_fb)}", flush=True)

elapsed = time.perf_counter() - t_start
n_total = len(wall_times_ws) + len(wall_times_fb)
print(f"\nDone in {elapsed:.1f}s")
print(f"Converged two-phase: {n_total}/{len(rows)}")
print(f"  warm-started: {len(wall_times_ws)}  |  fell back to robust: {len(wall_times_fb)}")

def stats(label, wt, ni):
    wt = np.array(wt); ni = np.array(ni)
    print(f"\n{label} (N={len(wt)}):")
    print(f"  Wall time (ms): median={np.median(wt):.1f}  mean={np.mean(wt):.1f}"
          f"  p95={np.percentile(wt,95):.1f}  max={np.max(wt):.1f}")
    print(f"  Iterations:     median={np.median(ni):.1f}  mean={np.mean(ni):.2f}"
          f"  max={np.max(ni):.0f}")

stats("Warm-started", wall_times_ws, n_iter_ws)
if wall_times_fb:
    stats("Fallback (robust)", wall_times_fb, n_iter_fb)

all_wt = np.array(wall_times_ws + wall_times_fb)
all_ni = np.array(n_iter_ws + n_iter_fb)
stats("Overall", all_wt, all_ni)
