#!/usr/bin/env python3
"""Fortran-only timing: evals/s for lnphi_aq, lnphi_gas, flash_kv,
stability_flash for ONE build variant (no Python reference needed).

Select the build with the ECPA_LIB env var, e.g.:
    ECPA_LIB=$PWD/build-ifx-sapphire/libecpa.so python3 tools/bench_fortran.py

Same states and protocol as tools/bench.py (single thread, cold-start
calls via ctypes -- numbers include the ~2-3 us ctypes crossing).

Usage: python3 bench_fortran.py [n_lnphi] [n_flash]
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ecpa_ctypes import Engine, DEFAULT_LIB  # noqa: E402

N_LNPHI = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
N_FLASH = int(sys.argv[2]) if len(sys.argv) > 2 else 200

T, Pbar = 373.15, 200.0
P_Pa = Pbar * 1e5
x_aq = np.array([0.97, 0.01, 0.02])
mi = np.array([1.0, 1.0, 0, 0, 0, 0])
y_gas = np.array([0.02, 0.30, 0.68])
z = np.array([0.85, 0.05, 0.10])

eng = Engine(parfile=HERE.parent / "params" / "paperII_III.par")


def timeit(fn, n):
    fn()  # warm up
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


print(f"library: {os.environ.get('ECPA_LIB', str(DEFAULT_LIB))}")
for name, fn, n in [
    ("lnphi_aq", lambda: eng.lnphi_aq(x_aq, mi, T, P_Pa), N_LNPHI),
    ("lnphi_gas", lambda: eng.lnphi_gas(y_gas, T, P_Pa), N_LNPHI),
    ("flash_kv", lambda: eng.flash_kv(T, Pbar, z, mi), N_FLASH),
    ("stability_flash", lambda: eng.stability_flash(T, Pbar, z, mi), N_FLASH),
]:
    t = timeit(fn, n)
    print(f"{name:16s} {t*1e3:10.4f} ms/eval   {1.0/t:12,.0f} evals/s")
