#!/usr/bin/env python3
"""Benchmark the Step 2 stability_flash driver against the Step 1 flash_kv.

For each of the 5 baseline cases:

* Run ``solve_elv`` to obtain the reference equilibrium and reconstruct
  the partition-basis feed ``z``.
* Time ``flash_kv`` with default (Wilson) ``K_init``.
* Time ``stability_flash`` (Michelsen TPD → trial-K flash → Wilson fallback).
* Report wall time, TPD trial iterations, inner SSI + Newton iterations,
  the ladder rung (``route``) and flash residual that produced the answer,
  and the worst-case composition difference between the two paths.  The
  composition difference is printed only when both paths converged to a
  two-phase root, and is ``nan`` otherwise.

    python3 scripts/benchmark_stability_flash.py
    python3 scripts/benchmark_stability_flash.py > tests/baselines/v0.3_step2_stability_flash_benchmark.txt
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ms_ecpa.eos import solve_elv  # noqa: E402
from ms_ecpa.flash import flash_kv  # noqa: E402
from ms_ecpa.stability import stability_flash, michelsen_tpd  # noqa: E402
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
    """Build a partition feed z that flashes to the equilibrium of r."""
    x_aq = np.array([r.x1w, r.x2w, r.x3w])
    y_gas = np.array([r.x1h, r.x2h, r.x3h])
    m_tot = float(np.sum(mi))
    scale = 1.0 - Mw * r.x1w * m_tot
    x_part = x_aq / scale
    beta = 0.5
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


def _total_trial_iters(stab):
    return int(sum(t.n_iter for t in stab.trials))


def main():
    hdr = (f"{'case':<32}"
           f"{'flash_kv [ms]':>15}"
           f"{'stab_flash [ms]':>17}"
           f"{'TPD iters':>11}"
           f"{'flash SSI':>11}"
           f"{'flash Nwt':>11}"
           f"{'route':>7}"
           f"{'max|g|':>11}"
           f"{'tpd_min':>11}"
           f"{'Δx max':>11}")
    print(hdr)
    print("-" * len(hdr))
    notes = []
    for name, T, P, YC, mi in CASES:
        mi_arr = np.asarray(mi, dtype=float)
        r_elv = solve_elv(T, P, YC, mi_arr)
        z = _build_z_from_solve_elv(r_elv, mi_arr)

        t_kv, r_kv = _time(flash_kv, T, P, z, mi_arr)
        t_sf, r_sf = _time(stability_flash, T, P, z, mi_arr)
        # Iteration breakdown: TPD trial iters + inner flash iters
        n_tpd = _total_trial_iters(r_sf.stability)
        fr = r_sf.flash_result
        if fr is not None:
            n_ssi = fr.n_iter_ssi
            n_nwt = fr.n_iter_newton
            res = fr.max_residual
        else:
            n_ssi = 0
            n_nwt = 0
            res = float("nan")
        # Δx is the column that certifies the two solve paths land on the
        # same aqueous composition, so it is only meaningful when both
        # paths reached a converged two-phase root.  A non-None
        # flash_result is not that: stability_flash also attaches one to
        # its β-degenerate return (route 7, phases == 1) and to its hard
        # failure (route 9, flash_result.converged False), and solve_elv
        # reports ier != 1 in `converged` rather than raising.  Without
        # these tests such a case still prints a finite number under a
        # header that reads as agreement.
        two_phase = (r_sf.phases == 2 and fr is not None and fr.converged)
        if two_phase and r_elv.converged:
            x = fr.x_aq
            dx = max(abs(x[0] - r_elv.x1w),
                     abs(x[1] - r_elv.x2w),
                     abs(x[2] - r_elv.x3w))
        else:
            dx = float("nan")
            notes.append(
                f"{name}: Δx not comparable -- stability_flash phases="
                f"{r_sf.phases}, route={r_sf.route}, flash converged="
                f"{None if fr is None else fr.converged}; "
                f"solve_elv converged={r_elv.converged} (ier={r_elv.ier})")
        print(f"{name:<32}"
              f"{t_kv*1e3:>15.2f}"
              f"{t_sf*1e3:>17.2f}"
              f"{n_tpd:>11d}"
              f"{n_ssi:>11d}"
              f"{n_nwt:>11d}"
              f"{r_sf.route:>7d}"
              f"{res:>11.2e}"
              f"{r_sf.tpd_min:>11.2e}"
              f"{dx:>11.2e}")
    for note in notes:
        print(note)


if __name__ == "__main__":
    main()
