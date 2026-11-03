#!/usr/bin/env python3
"""Benchmark the Step 1 K-value flash against the legacy ``solve_elv``.

For each of the 5 baseline cases, runs ``solve_elv`` (the legacy coupled
12-variable Newton/fsolve) and ``flash_kv`` (the new SSI + Newton-polish
K-value flash on the partition basis), reports wall time, SSI / Newton
iteration counts, max-|g| at solution, and the worst-case composition
difference between the two.

The aim is purely diagnostic — they solve different problems (solve_elv
takes ``YC`` directly, flash_kv takes a partition feed ``z``) — so the
"speedup" column is best read as an indicator of where the new flash
spends its time, not as a head-to-head efficiency claim.

The Δ columns are printed only when both paths converged, and are ``nan``
otherwise, with the reason listed under the table.

    python scripts/benchmark_flash_kv.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ms_ecpa.eos import solve_elv  # noqa: E402
from ms_ecpa.flash import flash_kv  # noqa: E402
from ms_ecpa.parameters import Mw  # noqa: E402


CASES = [
    ("323K 500bar YC=0.8 mixed brine", 323, 500, 0.8, [1, 2, 0.6, 0.2, 0.2, 0.2]),
    ("350K 100bar YC=1 pure CO2",      350, 100, 1.0, [0, 0, 0, 0, 0, 0]),
    ("350K 100bar YC=0 pure CH4",      350, 100, 0.0, [0, 0, 0, 0, 0, 0]),
    ("400K 300bar YC=0.5 NaCl 2m",     400, 300, 0.5, [2, 2, 0, 0, 0, 0]),
    ("300K 600bar YC=1 CaCl2 1m",      300, 600, 1.0, [0, 2, 0, 1, 0, 0]),
]
N = 5  # runs per timing (median)


def _build_z_from_solve_elv(r, mi):
    """Build a partition feed z that flashes to the equilibrium of r.

    ``r`` is used unconditionally: on a failed solve its compositions are
    the last fsolve iterate, so the feed is then the partition basis of a
    non-solution.  `main` tests ``r.converged`` before it reports the
    comparison rather than here, so the timing columns are still produced.
    """
    x_aq = np.array([r.x1w, r.x2w, r.x3w])
    y_gas = np.array([r.x1h, r.x2h, r.x3h])
    m_tot = float(np.sum(mi))
    scale = 1.0 - Mw * r.x1w * m_tot
    x_part = x_aq / scale
    beta = 0.5  # arbitrary feed split; gives a feed that re-flashes to (x_aq, y_gas)
    z = (1.0 - beta) * x_part + beta * y_gas
    return z / z.sum()


def _time(fn, *args, **kw):
    ts = []
    out = None
    for _ in range(N):
        t0 = time.perf_counter()
        out = fn(*args, **kw)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), out


def main():
    hdr = (f"{'case':<32}"
           f"{'solve_elv [ms]':>16}"
           f"{'flash_kv [ms]':>16}"
           f"{'SSI':>5}"
           f"{'Nwt':>5}"
           f"{'max|g|':>11}"
           f"{'Δx max':>11}"
           f"{'Δy max':>11}")
    print(hdr)
    print("-" * len(hdr))
    notes = []
    for name, T, P, YC, mi in CASES:
        mi_arr = np.asarray(mi, dtype=float)
        t_elv, r_elv = _time(solve_elv, T, P, YC, mi_arr)
        z = _build_z_from_solve_elv(r_elv, mi_arr)
        t_kv, r_kv = _time(flash_kv, T, P, z, mi_arr)

        # Δx / Δy certify that the two solve paths land on the same
        # equilibrium, so they mean that only when both paths reached one.
        # solve_elv reports ier != 1 in `converged` rather than raising, and
        # the feed z is built from whatever compositions it returned; without
        # this test a failed reference solve still prints a finite number
        # under a header that reads as agreement, and the flash_kv-side
        # max|g| column gives the reader no way to tell which side broke.
        if r_elv.converged and r_kv.converged:
            dx = max(abs(r_kv.x_aq[0] - r_elv.x1w),
                     abs(r_kv.x_aq[1] - r_elv.x2w),
                     abs(r_kv.x_aq[2] - r_elv.x3w))
            dy = max(abs(r_kv.y_gas[0] - r_elv.x1h),
                     abs(r_kv.y_gas[1] - r_elv.x2h),
                     abs(r_kv.y_gas[2] - r_elv.x3h))
        else:
            dx = float("nan")
            dy = float("nan")
            notes.append(
                f"{name}: Δ not comparable -- solve_elv converged="
                f"{r_elv.converged} (ier={r_elv.ier}, "
                f"max|F|={r_elv.residual:.3e}); "
                f"flash_kv converged={r_kv.converged}")
        print(f"{name:<32}"
              f"{t_elv*1e3:>16.2f}"
              f"{t_kv*1e3:>16.2f}"
              f"{r_kv.n_iter_ssi:>5d}"
              f"{r_kv.n_iter_newton:>5d}"
              f"{r_kv.max_residual:>11.2e}"
              f"{dx:>11.2e}"
              f"{dy:>11.2e}")
    for note in notes:
        print(note)


if __name__ == "__main__":
    main()
