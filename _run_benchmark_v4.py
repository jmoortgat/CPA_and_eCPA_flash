"""Re-run the flash benchmark (tab:benchmark) with post-fix code and scan_v4 table.

Conditions: T=125°C, z_CO2=0.5, ms=1.0 mol/kg, 30 pressure points 1-1500 bar.
Strategies:
  1. Cold SSI, Wilson K init (omega=0.7)
  2. Table warm-start (scan_v4), stability skipped
  3. Table warm-start + forced stability check on every call
"""
import warnings; warnings.filterwarnings('ignore')
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from ecpa.parameters import make_params
from ecpa.warmstart import ScanTableWarmStart
from ecpa.flash import flash_co2_h2o_salt_kv
from ecpa.stability import ecpa_stability_flash

T = 125.0 + 273.15   # 125°C in K
z = 0.5
ms = 1.0
P_arr = np.logspace(0, np.log10(1500), 30)

params = make_params()
ws = ScanTableWarmStart.load('results/scan_v4_table.npz')

N_rep = 5   # repetitions for stable timing

# ── Strategy 1: Cold SSI with Wilson K, omega=0.7 ────────────────────────────
times1 = []
iters1 = []
for _ in range(N_rep):
    t0 = time.perf_counter()
    for P in P_arr:
        try:
            out = flash_co2_h2o_salt_kv(T=T, P_bar=float(P), z_co2=z, m_tot=ms,
                                        params=params, maxiter=80, omega=0.7,
                                        K_init=None, sol_aq_x0=None, sol_c_x0=None)
            if out and out.get('n_iter') is not None:
                iters1.append(out['n_iter'])
        except Exception:
            pass
    times1.append(time.perf_counter() - t0)

t1_total = np.mean(times1)
print(f'Strategy 1 (cold SSI, Wilson, omega=0.7):')
print(f'  Total: {t1_total:.3f} s  |  per call: {1000*t1_total/len(P_arr):.1f} ms')
if iters1:
    print(f'  Mean iters: {np.mean(iters1):.1f}  median: {np.median(iters1):.0f}')

# ── Strategy 2: Table warm-start, skip stability ─────────────────────────────
times2 = []
iters2 = []
for _ in range(N_rep):
    t0 = time.perf_counter()
    for P in P_arr:
        guess = ws(T, float(P), z, ms)
        try:
            out = flash_co2_h2o_salt_kv(T=T, P_bar=float(P), z_co2=z, m_tot=ms,
                                        params=params, maxiter=80,
                                        K_init=guess.K_init if guess else None,
                                        sol_aq_x0=guess.sol_aq_x0 if guess else None,
                                        sol_c_x0=guess.sol_c_x0 if guess else None)
            if out and out.get('n_iter') is not None:
                iters2.append(out['n_iter'])
        except Exception:
            pass
    times2.append(time.perf_counter() - t0)

t2_total = np.mean(times2)
speedup = t1_total / t2_total if t2_total > 0 else float('nan')
print(f'\nStrategy 2 (table warm-start, stability skipped):')
print(f'  Total: {t2_total:.3f} s  |  per call: {1000*t2_total/len(P_arr):.1f} ms')
print(f'  Speedup vs cold: {speedup:.1f}x')
if iters2:
    print(f'  Mean iters: {np.mean(iters2):.1f}  median: {np.median(iters2):.0f}')

# ── Strategy 3: Table warm-start + forced stability check ────────────────────
times3 = []
for _ in range(N_rep):
    t0 = time.perf_counter()
    for P in P_arr:
        try:
            sf = ecpa_stability_flash(z_co2=z, ms=ms, T=T, P=float(P), params=params)
            if sf.get('phase') != 'single_phase':
                guess = ws(T, float(P), z, ms)
                flash_co2_h2o_salt_kv(T=T, P_bar=float(P), z_co2=z, m_tot=ms,
                                      params=params, maxiter=80,
                                      K_init=guess.K_init if guess else sf.get('K_vals'),
                                      sol_aq_x0=guess.sol_aq_x0 if guess else sf.get('sol_aq_x0'),
                                      sol_c_x0=guess.sol_c_x0 if guess else sf.get('sol_c_x0'))
        except Exception:
            pass
    times3.append(time.perf_counter() - t0)

t3_total = np.mean(times3)
speedup3 = t1_total / t3_total if t3_total > 0 else float('nan')
print(f'\nStrategy 3 (table warm-start + forced stability check):')
print(f'  Total: {t3_total:.3f} s  |  per call: {1000*t3_total/len(P_arr):.1f} ms')
print(f'  Speedup vs cold: {speedup3:.1f}x')

print(f'\n=== SUMMARY (for tab:benchmark) ===')
print(f'Cold SSI (Wilson, omega=0.7):   {t1_total:.2f} s total, {1000*t1_total/len(P_arr):.0f} ms/call'
      + (f', mean iters {np.mean(iters1):.1f}' if iters1 else ''))
print(f'Table warm-start, no stability: {t2_total:.2f} s total, {1000*t2_total/len(P_arr):.0f} ms/call, {speedup:.1f}x'
      + (f', mean iters {np.mean(iters2):.1f}' if iters2 else ''))
print(f'Table warm-start + stability:   {t3_total:.2f} s total, {1000*t3_total/len(P_arr):.0f} ms/call, {speedup3:.1f}x')
