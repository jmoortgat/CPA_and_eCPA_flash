"""
Benchmark: eCPA outer K-value Newton polish — compare newton_tol=1e-4 vs 1e-3.

Tests three configurations:
  A. SSI only             (use_newton=False)
  B. SSI + Newton 1e-4   (current default)
  C. SSI + Newton 1e-3   (earlier switch — new test)

For both warm-start (K_init provided from converged solution) and
cold-start (K_init=None) conditions.

Reports: total iterations (SSI + Newton), wall time per call,
Newton trigger rate, fallback rate.
"""
import sys, time, warnings
import numpy as np
sys.path.insert(0, '/Users/moortgat/Software/2026/eCPA_SALTbasis/Claude_code')
warnings.filterwarnings('ignore')
import ecpa.constants
from ecpa.flash import flash_co2_h2o_salt_kv

# ── Condition grid ──────────────────────────────────────────────────────────────
Ts     = [310.0, 330.0, 350.0, 370.0, 390.0, 410.0, 430.0]
Ps     = [ 60.0, 100.0, 150.0, 200.0, 300.0, 400.0]
z_co2s = [0.15, 0.30, 0.50, 0.70]
ms_vals = [0.5, 1.5, 3.0, 4.5]

conditions = [(T, P, z, ms)
              for T in Ts for P in Ps for z in z_co2s for ms in ms_vals]
print(f"Grid: {len(conditions)} conditions  "
      f"(T×P×z×ms = {len(Ts)}×{len(Ps)}×{len(z_co2s)}×{len(ms_vals)})\n")

N_REP = 6   # repetitions for timing

# ── Step 1: collect converged solutions for warm-start ─────────────────────────
print("Collecting cold-start solutions (SSI only, for warm-start baseline) ...")
warm_starts = {}   # (T, P, z, ms) → (K1, K4, sol_aq_x0, sol_c_x0)
ok_cold = fail_cold = 0
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for T, P, z, ms in conditions:
        try:
            res = flash_co2_h2o_salt_kv(T, P, z, ms,
                                         use_newton=False, tol=1e-8, maxiter=120)
            K1, K4 = res['K_vals']
            warm_starts[(T, P, z, ms)] = (K1, K4, res['sol_aq_x0'], res['sol_c_x0'])
            ok_cold += 1
        except Exception:
            fail_cold += 1

print(f"  Converged: {ok_cold}/{len(conditions)}  (failed: {fail_cold})\n")


# ── Benchmark helper ───────────────────────────────────────────────────────────
def run_batch(warm, use_newton, newton_tol):
    """
    warm=True:  provide K_init + sol_{aq,c}_x0 from warm_starts.
    warm=False: cold-start only.
    Returns list of result dicts (one per condition × rep), or None for failures.
    """
    results = []
    for T, P, z, ms in conditions:
        for _ in range(N_REP):
            kw = dict(use_newton=use_newton, newton_tol=newton_tol,
                      max_newton=10, tol=1e-8, maxiter=120, accelerated=True)
            if warm and (T, P, z, ms) in warm_starts:
                K1, K4, sol_aq, sol_c = warm_starts[(T, P, z, ms)]
                kw.update(K_init=(K1, K4), sol_aq_x0=sol_aq, sol_c_x0=sol_c)
            try:
                r = flash_co2_h2o_salt_kv(T, P, z, ms, **kw)
                results.append(r)
            except Exception:
                results.append(None)
    return results


def summarise(label, results):
    ok      = sum(1 for r in results if r is not None)
    fail    = len(results) - ok
    n_cond  = len(results) // N_REP

    # Per-call counts
    n_ssi    = [r.get('n_ssi_iter',  r.get('n_iter_ms', 0)) for r in results if r]
    n_newt   = [r.get('n_newton_iter', 0)                   for r in results if r]
    n_total  = [s + n for s, n in zip(n_ssi, n_newt)]
    n_trig   = sum(1 for n in n_newt if n > 0)
    n_fb     = sum(1 for r in results if r and not r.get('converged', True) and
                                          r.get('n_newton_iter', 0) > 0)

    avg_ssi   = np.mean(n_ssi)   if n_ssi   else float('nan')
    avg_newt  = np.mean(n_newt)  if n_newt  else float('nan')
    avg_tot   = np.mean(n_total) if n_total else float('nan')
    frac_trig = n_trig / ok * 100 if ok > 0 else 0.0

    print(f"\n  {label}")
    print(f"    Converged / failed      : {ok//N_REP:4d} / {fail//N_REP}")
    print(f"    Avg SSI iters           : {avg_ssi:.2f}")
    print(f"    Avg Newton iters        : {avg_newt:.2f}"
          f"  ({frac_trig/N_REP:.0f}% of calls triggered Newton)")
    print(f"    Avg total iters (SSI+NR): {avg_tot:.2f}")
    return avg_tot


# ── Run benchmarks ─────────────────────────────────────────────────────────────
configs = [
    ("A  SSI only                   (baseline)", False, 1e-4),
    ("B  SSI + Newton  tol=1e-4     (default) ", True,  1e-4),
    ("C  SSI + Newton  tol=1e-3     (new test)", True,  1e-3),
]

for mode_label, warm in [("WARM-START", True), ("COLD-START", False)]:
    print("=" * 70)
    print(f"  {mode_label}")
    print("=" * 70)

    totals = {}
    times  = {}

    for label, use_nw, ntol in configs:
        t0 = time.perf_counter()
        results = run_batch(warm=warm, use_newton=use_nw, newton_tol=ntol)
        wall = time.perf_counter() - t0
        avg_tot = summarise(label, results)
        totals[label] = avg_tot
        n_ok = sum(1 for r in results if r is not None)
        times[label] = wall / n_ok * 1e3 if n_ok > 0 else float('nan')
        print(f"    Wall time per call       : {times[label]:.3f} ms")

    # Speed-ups relative to A
    base_t = times[configs[0][0]]
    base_i = totals[configs[0][0]]
    print(f"\n  Speed-ups vs baseline (A):")
    for label, _, _ in configs[1:]:
        speedup_t = base_t / times[label] if times[label] > 0 else float('nan')
        iter_red  = (base_i - totals[label]) / base_i * 100 if base_i > 0 else float('nan')
        print(f"    {label.split('(')[0].strip():<38}  "
              f"{speedup_t:.2f}×  ({iter_red:+.1f}% iters)")
    print()
