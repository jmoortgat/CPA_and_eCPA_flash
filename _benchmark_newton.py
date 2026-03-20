"""
Benchmark: analytical vs FD-Jacobian Newton for the eCPA aqueous inner solve.

Architecture
------------
`_newton_aq` is reached on the warm-start path of `_lnphi_aq_inner` (x0≠None).
For the cold-start (x0=None) the current code runs scipy fsolve to full
convergence.

This benchmark tests three solve strategies for the cold-start case:

  A. fsolve-only        — current production code (baseline).
  B. heuristic → Newton — try Newton directly from the 3 heuristic starting
                          points; fall back to fsolve if all fail.
  C. 1-fsolve → Newton  — run fsolve once with a loose tolerance (≈1e-3) to
                          get a rough x0, then switch to Newton for the tight
                          solve.  The user's suggested approach.

All three are then compared for:
  · success rate
  · avg Newton iterations (strategies B/C) or fsolve function evaluations (A)
  · wall time per condition

Additionally a warm-start benchmark (current production path) compares
analytical vs FD Newton at several perturbation levels.

Run:
    python _benchmark_newton.py
"""

import time, warnings
import numpy as np
from scipy.optimize import fsolve
import ecpa.constants          # ensure globals set
from ecpa.stability import (
    _lnphi_aq_inner, _newton_aq,
    _eval_aq_all, _eval_aq_all_with_jac,
    b1, b2, b3, b4, Mw, R,
)

# ── Newton helpers ─────────────────────────────────────────────────────────────

H_FD = 1e-6


def _residual(v, x1w, ms, T, P):
    Zw, epsr, chi1w = float(v[0]), float(v[1]), float(v[2])
    if Zw <= 0 or epsr <= 1 or chi1w <= 0 or chi1w >= 2.0:
        return np.array([1e6, 1e6, 1e6])
    chi1w_e = min(chi1w, 1.0 - 1e-12)
    Zw_new, T1, T2, chi1w_new, _, _ = _eval_aq_all(Zw, epsr, chi1w_e, x1w, ms, T, P)
    return np.array([Zw - Zw_new, T1 - T2, chi1w - chi1w_new])


def _newton_an_count(v0, x1w, ms, T, P, tol=1e-10, maxiter=20):
    """Analytical Newton; returns (solution | None, iterations)."""
    v = np.array(v0, dtype=float)
    for k in range(maxiter):
        Zw, epsr, chi1w = v[0], v[1], v[2]
        if Zw <= 0 or epsr <= 1 or chi1w <= 0 or chi1w >= 2.0:
            return None, k
        chi1w_e = min(chi1w, 1.0 - 1e-12)
        Zw_new, T1, T2, chi1w_new, _, _, J = _eval_aq_all_with_jac(
            Zw, epsr, chi1w_e, x1w, ms, T, P)
        F = np.array([Zw - Zw_new, T1 - T2, chi1w - chi1w_new])
        if np.max(np.abs(F)) < tol:
            return v, k
        try:
            dv = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            return None, k
        alpha = 1.0
        for _ in range(6):
            vt = v + alpha * dv
            if vt[0] > 0 and vt[1] > 1 and 0 < vt[2] < 2.0:
                break
            alpha *= 0.5
        else:
            return None, k
        v = vt
    return None, maxiter


def _newton_fd_count(v0, x1w, ms, T, P, tol=1e-10, maxiter=20):
    """FD Newton; returns (solution | None, iterations)."""
    v = np.array(v0, dtype=float)
    for k in range(maxiter):
        Zw, epsr, chi1w = v[0], v[1], v[2]
        if Zw <= 0 or epsr <= 1 or chi1w <= 0 or chi1w >= 2.0:
            return None, k
        chi1w_e = min(chi1w, 1.0 - 1e-12)
        Zw_new, T1, T2, chi1w_new, _, _ = _eval_aq_all(
            Zw, epsr, chi1w_e, x1w, ms, T, P)
        F = np.array([Zw - Zw_new, T1 - T2, chi1w - chi1w_new])
        if np.max(np.abs(F)) < tol:
            return v, k
        J = np.zeros((3, 3))
        for j in range(3):
            h = max(abs(v[j]) * H_FD, 1e-8)
            vp = v.copy(); vp[j] += h
            chi1w_p = min(vp[2], 1.0 - 1e-12)
            Zw_p, T1_p, T2_p, chi1w_p_new, _, _ = _eval_aq_all(
                vp[0], vp[1], chi1w_p, x1w, ms, T, P)
            Fp = np.array([vp[0] - Zw_p, T1_p - T2_p, vp[2] - chi1w_p_new])
            J[:, j] = (Fp - F) / h
        try:
            dv = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            return None, k
        alpha = 1.0
        for _ in range(6):
            vt = v + alpha * dv
            if vt[0] > 0 and vt[1] > 1 and 0 < vt[2] < 2.0:
                break
            alpha *= 0.5
        else:
            return None, k
        v = vt
    return None, maxiter


def _heuristic_starts(x1w, ms, T, P):
    x2w = x1w * ms * Mw
    b_est = b1*x1w + b2*x2w + b3*x2w + b4*(1 - x1w - 2*x2w)
    Zw0 = max(b_est * (P * 1e5) / R / T * 1.2, 1e-4)
    return [
        np.array([Zw0,       60.0, 0.4]),
        np.array([Zw0 * 1.3, 70.0, 0.6]),
        np.array([Zw0,       60.0, 0.99]),
    ]


# ── Strategy implementations ───────────────────────────────────────────────────

def solve_fsolve_only(x1w, ms, T, P):
    """Strategy A: fsolve to full convergence (current code)."""
    def residual(v):
        return _residual(v, x1w, ms, T, P)
    for start in _heuristic_starts(x1w, ms, T, P):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sol, info, ier, _ = fsolve(residual, start, full_output=True)
        if ier == 1 and sol[0] > 0 and sol[1] > 1 and 0 < sol[2] < 2.0:
            return sol, info['nfev'], 0   # sol, fsolve_fevals, newton_iters
    return None, 0, 0


def solve_heuristic_newton(x1w, ms, T, P, newton_fn=_newton_an_count):
    """Strategy B: try Newton from each heuristic start; fall back to fsolve."""
    for v0 in _heuristic_starts(x1w, ms, T, P):
        sol, iters = newton_fn(v0, x1w, ms, T, P)
        if sol is not None:
            return sol, 0, iters
    # Newton failed from all starts: fall back to fsolve
    sol, fevals, _ = solve_fsolve_only(x1w, ms, T, P)
    return sol, fevals, -1   # -1 Newton iters = fsolve fallback


def solve_1fsolve_then_newton(x1w, ms, T, P, maxfev_loose=20,
                               newton_fn=_newton_an_count):
    """Strategy C: budget-limited fsolve (≈5 quasi-Newton steps) → Newton polish.
    maxfev_loose limits the rough phase; Newton then polishes quadratically."""
    def residual(v):
        return _residual(v, x1w, ms, T, P)
    x0 = None; fevals = 0
    for start in _heuristic_starts(x1w, ms, T, P):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sol, info, ier, _ = fsolve(residual, start, full_output=True,
                                       maxfev=maxfev_loose)
        fevals += info['nfev']
        v = sol
        # Accept if physically valid AND residual already reduced somewhat
        Fv = _residual(v, x1w, ms, T, P)
        if v[0] > 0 and v[1] > 1 and 0 < v[2] < 2.0 and np.max(np.abs(Fv)) < 1e2:
            x0 = v; break
    if x0 is None:
        # Loose fsolve didn't land anywhere useful — try tight fsolve fallback
        sol, fevals2, _ = solve_fsolve_only(x1w, ms, T, P)
        return sol, fevals + fevals2, -1 if sol is not None else 0
    # Newton polish from the rough estimate
    sol, iters = newton_fn(x0, x1w, ms, T, P)
    if sol is not None:
        return sol, fevals, iters
    # Newton failed: tight fsolve from x0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol2, info2, ier2, _ = fsolve(residual, x0, full_output=True)
    fevals += info2['nfev']
    if ier2 == 1 and sol2[0] > 0 and sol2[1] > 1 and 0 < sol2[2] < 2.0:
        return sol2, fevals, -1
    return None, fevals, 0


# ── Condition grid ─────────────────────────────────────────────────────────────

Ts   = [323.0, 348.0, 373.0, 398.0, 423.0]
Ps   = [ 50.0, 100.0, 200.0, 300.0, 500.0]
MSs  = [0.0, 1.0, 2.0, 3.0, 4.0]
x1w_vals = [0.95, 0.90]   # two aqueous compositions

conditions_all = [(T, P, ms, x1w)
                  for T in Ts for P in Ps for ms in MSs for x1w in x1w_vals]

print(f"Grid: {len(conditions_all)} conditions  "
      f"(T×P×ms×x1w = {len(Ts)}×{len(Ps)}×{len(MSs)}×{len(x1w_vals)})\n")

N_REPEAT = 5


# ── Cold-start strategies comparison ──────────────────────────────────────────

print("=" * 70)
print("  COLD-START COMPARISON")
print("  A. fsolve only (current)  B. heuristic→Newton  C. 1-fsolve→Newton")
print("=" * 70)

strategies = [
    ("A  fsolve only       ", solve_fsolve_only,
     lambda x1w, ms, T, P: solve_fsolve_only(x1w, ms, T, P)),
    ("B  heuristic→Newton  ", None,
     lambda x1w, ms, T, P: solve_heuristic_newton(x1w, ms, T, P, _newton_an_count)),
    ("C  1-fsolve→Newton   ", None,
     lambda x1w, ms, T, P: solve_1fsolve_then_newton(
         x1w, ms, T, P, newton_fn=_newton_an_count)),
    ("B' heuristic→FD-Newt ", None,
     lambda x1w, ms, T, P: solve_heuristic_newton(x1w, ms, T, P, _newton_fd_count)),
    ("C' 1-fsolve→FD-Newt  ", None,
     lambda x1w, ms, T, P: solve_1fsolve_then_newton(
         x1w, ms, T, P, newton_fn=_newton_fd_count)),
]

for label, _, fn in strategies:
    ok = fail = 0
    tot_fevals = tot_newton = n_fallback = 0
    t0 = time.perf_counter()
    for _ in range(N_REPEAT):
        for T, P, ms, x1w in conditions_all:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    sol, fevals, niters = fn(x1w, ms, T, P)
                    if sol is not None:
                        ok += 1
                        tot_fevals  += fevals
                        if niters >= 0:
                            tot_newton += niters
                        else:
                            n_fallback += 1
                    else:
                        fail += 1
                except Exception:
                    fail += 1
    wall = time.perf_counter() - t0
    n_total = len(conditions_all) * N_REPEAT
    avg_fev = tot_fevals / ok if ok > 0 else float("nan")
    avg_nit = tot_newton / (ok - n_fallback) if (ok - n_fallback) > 0 else float("nan")
    ms_per = wall / n_total * 1e3

    print(f"\n  {label}")
    print(f"    Converged / failed      : {ok//N_REPEAT:4d} / {fail//N_REPEAT}")
    print(f"    fsolve fevals (avg/call): {avg_fev:7.1f}"
          f"  |  Newton iters (avg/call): {avg_nit:.2f}")
    if n_fallback > 0:
        print(f"    Newton→fsolve fallbacks : {n_fallback//N_REPEAT}")
    print(f"    Time per condition      : {ms_per:.3f} ms")

print()

# ── Warm-start Newton comparison (analytical vs FD) ───────────────────────────

print("=" * 70)
print("  WARM-START COMPARISON  (analytical vs FD Newton)")
print("=" * 70)

# Collect converged solutions via fsolve cold-start
print("\nCollecting cold-start solutions for warm-start benchmark ...")
conditions_with_sol = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for T, P, ms, x1w in conditions_all:
        try:
            _, _, sol = _lnphi_aq_inner(x1w, ms, T, P, x0=None)
            conditions_with_sol.append((T, P, ms, x1w, sol))
        except Exception:
            pass
print(f"  {len(conditions_with_sol)} / {len(conditions_all)} converged\n")

PERTURB_LEVELS = [0.001, 0.005, 0.01, 0.02, 0.05]

def make_perturbed(sol, delta):
    return sol * np.array([1 + delta, 1 + 0.5*delta, 1 - delta])

for label, newton_fn in [("Analytical (1 eval/step)", _newton_an_count),
                          ("FD        (4 eval/step) ", _newton_fd_count)]:
    ok = fail = 0
    tot_iters = 0
    t0 = time.perf_counter()
    for _ in range(N_REPEAT):
        for T, P, ms, x1w, sol in conditions_with_sol:
            for delta in PERTURB_LEVELS:
                v0 = make_perturbed(sol, delta)
                result, iters = newton_fn(v0, x1w, ms, T, P)
                if result is not None:
                    ok += 1; tot_iters += iters
                else:
                    fail += 1
    wall = time.perf_counter() - t0
    n_total = len(conditions_with_sol) * len(PERTURB_LEVELS) * N_REPEAT
    avg_it = tot_iters / ok if ok > 0 else float("nan")
    print(f"  {label}")
    print(f"    Calls / converged / failed : {n_total:6d} / {ok:6d} / {fail}")
    print(f"    Avg iterations (success)   : {avg_it:.2f}")
    print(f"    Time per call              : {wall/n_total*1e3:.4f} ms")
    print()

speedup_str = ""
# Quick timing ratio
t_an = t_fd = 0
for _ in range(N_REPEAT):
    for T, P, ms, x1w, sol in conditions_with_sol:
        for delta in PERTURB_LEVELS:
            v0 = make_perturbed(sol, delta)
_t0 = time.perf_counter()
for _ in range(N_REPEAT):
    for T, P, ms, x1w, sol in conditions_with_sol:
        for delta in PERTURB_LEVELS:
            v0 = make_perturbed(sol, delta)
            _newton_an_count(v0, x1w, ms, T, P)
t_an = time.perf_counter() - _t0
_t0 = time.perf_counter()
for _ in range(N_REPEAT):
    for T, P, ms, x1w, sol in conditions_with_sol:
        for delta in PERTURB_LEVELS:
            v0 = make_perturbed(sol, delta)
            _newton_fd_count(v0, x1w, ms, T, P)
t_fd = time.perf_counter() - _t0
print(f"  Warm-start speedup (FD / Analytical): {t_fd/t_an:.2f}×\n")

# ── Per-perturbation-level detail ─────────────────────────────────────────────

print("Per-perturbation breakdown (analytical, warm-start):")
print(f"  {'delta':>7}  {'N_ok':>5}  {'Avg iter':>9}  {'t [ms/call]':>12}")
for delta in PERTURB_LEVELS:
    ok = iters = 0
    t0 = time.perf_counter()
    for _ in range(N_REPEAT):
        for T, P, ms, x1w, sol in conditions_with_sol:
            v0 = make_perturbed(sol, delta)
            result, k = _newton_an_count(v0, x1w, ms, T, P)
            if result is not None:
                ok += 1; iters += k
    wall = time.perf_counter() - t0
    n_total = len(conditions_with_sol) * N_REPEAT
    avg_it = iters / ok if ok > 0 else float("nan")
    print(f"  {delta:7.1%}  {ok//N_REPEAT:5d}  {avg_it:9.2f}  "
          f"{wall/n_total*1e3:12.4f}")

print()
