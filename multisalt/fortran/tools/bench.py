#!/usr/bin/env python3
"""Timing comparison: Fortran engine vs Python reference, same machine,
single-threaded, same states.

Measures:
  * lnphi_aq evaluations/second (cold-start driver, representative
    brine state), Python vs Fortran
  * lnphi_gas evaluations/second
  * flash_kv per-point wall time (two-phase state, Wilson K init)
  * stability_flash per-point wall time

Usage: python3 bench.py [n_lnphi] [n_flash]
"""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(HERE))

from ecpa_ctypes import Engine                       # noqa: E402
from ms_ecpa.lnphi import lnphi_aq, lnphi_gas        # noqa: E402
from ms_ecpa import flash_kv, stability_flash        # noqa: E402

N_LNPHI = int(sys.argv[1]) if len(sys.argv) > 1 else 200
N_FLASH = int(sys.argv[2]) if len(sys.argv) > 2 else 20

T, Pbar = 373.15, 200.0
P_Pa = Pbar * 1e5
x_aq = np.array([0.97, 0.01, 0.02])
mi = np.array([1.0, 1.0, 0, 0, 0, 0])
y_gas = np.array([0.02, 0.30, 0.68])
z = np.array([0.85, 0.05, 0.10])

eng = Engine(parfile=HERE.parent / "params" / "paperII_III.par")


def timeit(fn, n):
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


rows = []

t_py = timeit(lambda: lnphi_aq(x_aq, mi, T, P_Pa), N_LNPHI)
t_f = timeit(lambda: eng.lnphi_aq(x_aq, mi, T, P_Pa), N_LNPHI)
rows.append(("lnphi_aq", t_py, t_f))

t_py = timeit(lambda: lnphi_gas(y_gas, T, P_Pa), N_LNPHI)
t_f = timeit(lambda: eng.lnphi_gas(y_gas, T, P_Pa), N_LNPHI)
rows.append(("lnphi_gas", t_py, t_f))

t_py = timeit(lambda: flash_kv(T, Pbar, z, mi), N_FLASH)
t_f = timeit(lambda: eng.flash_kv(T, Pbar, z, mi), N_FLASH)
rows.append(("flash_kv", t_py, t_f))

t_py = timeit(lambda: stability_flash(T, Pbar, z, mi), max(N_FLASH // 2, 5))
t_f = timeit(lambda: eng.stability_flash(T, Pbar, z, mi), max(N_FLASH // 2, 5))
rows.append(("stability_flash", t_py, t_f))

print(f"{'kernel':18s} {'python':>12s} {'fortran':>12s} {'speedup':>9s} "
      f"{'py evals/s':>11s} {'f90 evals/s':>12s}")
for name, tp, tf in rows:
    print(f"{name:18s} {tp*1e3:9.3f} ms {tf*1e3:9.3f} ms {tp/tf:8.1f}x "
          f"{1/tp:11.1f} {1/tf:12.1f}")
