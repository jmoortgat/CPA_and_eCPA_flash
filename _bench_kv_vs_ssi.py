"""
_bench_kv_vs_ssi.py
-------------------
Head-to-head benchmark: flash_co2_h2o_salt_fast (ms_aq SSI)
vs flash_co2_h2o_salt_fast_kv (K-value SSI) using the actual
solution table as warm-start.

Loads the solution table from ../Claude_code/results/solution_table.npz
and runs both flash functions on every two-phase grid point.

Usage:
    cd /Users/moortgat/Software/2026/eCPA_SALTbasis/eCPA_improvements
    python _bench_kv_vs_ssi.py
"""
import sys, time, warnings
import numpy as np

sys.path.insert(0, ".")

from ecpa.parameters import make_params
from ecpa.solution_table import load_solution_table, make_solution_guess_fn
from ecpa.flash import flash_co2_h2o_salt_fast, flash_co2_h2o_salt_fast_kv

TABLE_PATH = "../Claude_code/results/solution_table.npz"
params     = make_params()

# ── Load table ────────────────────────────────────────────────────────────────
print(f"Loading solution table: {TABLE_PATH}")
grid_data = load_solution_table(TABLE_PATH)
guess_fn  = make_solution_guess_fn(grid_data)

T_grid  = grid_data["T_grid"]
P_grid  = 10.0 ** grid_data["logP_grid"]
z_grid  = grid_data["z_grid"]
ms_grid = grid_data["ms_grid"]
stable  = grid_data["stable"]      # (nT, nP, nz, nms) bool

nT, nP, nz, nms = stable.shape
total = int(stable.sum())
print(f"Table: {nT}T × {nP}P × {nz}z × {nms}ms  → {total} two-phase points")

# Collect all two-phase (T, P, z, ms) combinations
pts = []
for iT in range(nT):
    for iP in range(nP):
        for iz in range(nz):
            for ims in range(nms):
                if stable[iT, iP, iz, ims]:
                    pts.append((T_grid[iT], P_grid[iP],
                                z_grid[iz], ms_grid[ims]))

print(f"Running benchmark on {len(pts)} two-phase points …\n")

# ── Benchmark helper ──────────────────────────────────────────────────────────

def run_flash(flash_fn, pts, label):
    n_conv  = 0
    n_fail  = 0
    iters   = []
    t_start = time.perf_counter()
    for T, P, z, ms in pts:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                out = flash_fn(T, P, z, ms, solution_guess_fn=guess_fn,
                               params=params)
                n_conv += 1
                iters.append(int(out.get("n_iter_ms", 0)))
            except Exception:
                n_fail += 1
    elapsed = time.perf_counter() - t_start

    iters = np.array(iters)
    print(f"{label}")
    print(f"  Converged : {n_conv}/{len(pts)}  ({100*n_conv/len(pts):.1f}%)")
    if n_conv > 0:
        print(f"  Iters     : mean={iters.mean():.2f}  "
              f"median={np.median(iters):.0f}  "
              f"max={iters.max()}")
        print(f"  Time      : {elapsed:.3f}s  "
              f"({1000*elapsed/n_conv:.2f} ms/point)")
    print()
    return n_conv, iters, elapsed


# ── Run both ──────────────────────────────────────────────────────────────────

n1, it1, t1 = run_flash(flash_co2_h2o_salt_fast,
                         pts, "flash_co2_h2o_salt_fast    [ms_aq SSI]")
n2, it2, t2 = run_flash(flash_co2_h2o_salt_fast_kv,
                         pts, "flash_co2_h2o_salt_fast_kv [K-value SSI]")

# ── Summary ───────────────────────────────────────────────────────────────────
print("=" * 60)
if n1 > 0 and n2 > 0:
    print(f"Iteration ratio  KV/SSI : {it2.mean()/it1.mean():.3f}")
    print(f"Wall-time ratio  KV/SSI : {t2/t1:.3f}  "
          f"({'faster' if t2 < t1 else 'slower'})")
    print(f"Speed-up (SSI/KV)       : {t1/t2:.2f}×")
