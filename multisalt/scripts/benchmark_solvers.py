#!/usr/bin/env python3
"""Benchmark the Newton (complex-step Jacobian) solver against legacy fsolve.

For each test condition, reports end-to-end wall time (median of N runs),
the residual infinity-norm at each solver's returned solution, each solver's
status flag, and the max absolute difference between the two solvers' raw
solution vectors. A speedup and a solution difference only mean anything
when both solvers actually solved, so rows where either one did not are
marked and those two columns are blanked.

    python scripts/benchmark_solvers.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ms_ecpa.eos import solve_elv  # noqa: E402

CASES = [
    ("323K 500bar YC=0.8 mixed brine", 323, 500, 0.8, [1, 2, 0.6, 0.2, 0.2, 0.2]),
    ("350K 100bar YC=1 pure CO2",      350, 100, 1.0, [0, 0, 0, 0, 0, 0]),
    ("350K 100bar YC=0 pure CH4",      350, 100, 0.0, [0, 0, 0, 0, 0, 0]),
    ("400K 300bar YC=0.5 NaCl 2m",     400, 300, 0.5, [2, 2, 0, 0, 0, 0]),
    ("300K 600bar YC=1 CaCl2 1m",      300, 600, 1.0, [0, 2, 0, 1, 0, 0]),
]
N = 7  # runs per timing (median)


def _time(solver, T, P, YC, mi):
    ts = []
    res = None
    for _ in range(N):
        t0 = time.perf_counter()
        res = solve_elv(T, P, YC, mi, solver=solver)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), res


def main():
    hdr = (f"{'case':<32}{'fsolve [ms]':>12}{'newton [ms]':>12}{'speedup':>9}"
           f"{'max|F| fsolve':>15}{'max|F| newton':>15}{'sol Δ':>11}{'status':>16}")
    print(hdr)
    print("-" * len(hdr))
    n_bad = 0
    for name, T, P, YC, mi in CASES:
        t_f, r_f = _time("fsolve", T, P, YC, mi)
        t_n, r_n = _time("newton", T, P, YC, mi)
        # A returned vector is the solver's last iterate whether or not it
        # converged; without this test a solver that bailed at its iteration
        # cap is timed and differenced exactly like one that found the root.
        bad = [tag for tag, r in (("f", r_f), ("n", r_n)) if not r.converged]
        if bad:
            n_bad += 1
            status = "FAIL " + ",".join(
                f"{tag}:ier={r.ier}"
                for tag, r in (("f", r_f), ("n", r_n)) if tag in bad)
            speedup, sol_diff = f"{'--':>9}", f"{'--':>11}"
        else:
            status = "ok"
            sol = np.max(np.abs(np.asarray(r_n.solution) - np.asarray(r_f.solution)))
            speedup, sol_diff = f"{t_f/t_n:>8.2f}x", f"{sol:>11.2e}"
        print(f"{name:<32}{t_f*1e3:>12.2f}{t_n*1e3:>12.2f}{speedup}"
              f"{r_f.residual:>15.2e}{r_n.residual:>15.2e}{sol_diff}{status:>16}")
    if n_bad:
        print(f"\n{n_bad} case(s) had a solver that did not converge; their "
              f"timings are for a solve that never reached a root, so the "
              f"speedup and sol Δ columns are blanked there.")
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
