"""
eCPA flash algorithms for CO₂ + H₂O + NaCl.

Public API
----------
build_continuation_cache(T, P_bar, base_guess, ms_max, ...)
    March ms from 0 to ms_max, storing converged ELV solutions.

solve_elv(T, P_bar, ms_target, cache, ...)
    Solve ELV at a target molality using a warm start from the cache.

flash_co2_h2o_salt_1d(T, P_bar, z_co2, m_tot, guess_table_fn, ...)  [Brent]
    Full-cache + bracketing-scan + Brent root-find on the water residual.

flash_co2_h2o_salt_ssi(T, P_bar, z_co2, m_tot, guess_table_fn=None, ...)  [SSI]
    Damped fixed-point (SSI) iteration on ms_aq.  Accepts optional
    initial_sol / initial_ms_aq for a fast warm-start (skips ms=0 solve).

flash_co2_h2o_salt_fast(T, P_bar, z_co2, m_tot, solution_guess_fn, ...)  [fast]
    Production-oriented flash using a solution_table interpolant as the
    starting point.  Typically 1–3 SSI iterations instead of 20–40.

flash_co2_h2o_salt_kv(T, P_bar, z_co2, m_tot, K_init, ...)  [K-value SSI]
    Single-level iteration on K-values (K₁, K₄); ms_aq is computed
    analytically at each step — no nested ms_aq loop.  Jex-accelerated.

flash_co2_h2o_salt_fast_kv(T, P_bar, z_co2, m_tot, solution_guess_fn, ...)
    Like flash_co2_h2o_salt_fast but uses K-value SSI internally.

Result dict keys
----------------
T, P_bar, z_co2, m_tot
n_totals    dict: n_co2_tot, n_h2o_tot, n_salt_tot
ms_aq       float: equilibrium aqueous molality [mol/kg]
N_aq, N_c   float: molar amounts of each phase (basis: n_CO2+n_H2O = 1)
beta        float: CO₂-rich phase mole fraction
x_aq        dict: x1w, x2w, x3w, x4w
x_c         dict: x1c, x4c
Z_aq, Z_c   float: phase compressibilities
sol         np.ndarray: converged 10-element ELV solution vector
n_iter_ms   int: number of outer iterations (SSI only)
"""
from __future__ import annotations
import time
import warnings

import numpy as np
from scipy.optimize import fsolve, brentq

from .constants import Mw
from .elv import ELV, ELV_jac, USE_COMPLEX_JAC


# ── Helpers ────────────────────────────────────────────────────────────────────

def _cpa2_phase_check(T: float, P_bar: float, z_co2: float, params) -> str:
    """
    Quick CPA salt-free flash to classify the likely phase state.

    Returns: 'two_phase' | 'single_phase_liquid' | 'single_phase_gas' | 'unknown'
    CPA is salt-free, so the two-phase window only shrinks with salt —
    this provides a conservative pre-filter for bulk scans.
    """
    try:
        import CPA
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = CPA.flash_co2_h2o_tpz(T=float(T), P_bar=float(P_bar),
                                          z_co2=float(z_co2))
        if out["phase"] == "two_phase":
            return "two_phase"
        x_co2_liq = out["x"][0]
        x_h2o_vap = out["y"][1]
        if x_co2_liq < 0.01:
            return "single_phase_liquid"
        if x_h2o_vap < 0.01:
            return "single_phase_gas"
        return "unknown"
    except Exception:
        return "unknown"


# ── Brent-loop helpers ─────────────────────────────────────────────────────────

def build_continuation_cache(T, P_bar, base_guess, ms_max,
                              params=None, dm=0.1,
                              xtol=1e-10, maxfev=2000,
                              verbose=False):
    """
    March ms from 0 to ms_max in steps of dm, storing converged ELV solutions.

    Returns a list of (ms, sol) pairs in ascending ms order.
    Marching stops as soon as fsolve fails.
    """
    cache = []
    guess = np.asarray(base_guess, dtype=np.float64)
    m = np.float64(0.0)

    _total_nfev = [0]

    def _solve_one(m, guess):
        t0 = time.perf_counter()
        sol, info, ier, mesg = fsolve(
            ELV, guess, args=(T, P_bar * 1e5, m, params),
            fprime=ELV_jac if USE_COMPLEX_JAC else None,
            full_output=True, xtol=xtol, maxfev=maxfev,
        )
        sol = np.asarray(sol, dtype=np.float64)
        _total_nfev[0] += info['nfev']
        return sol, ier

    sol, ier = _solve_one(m, guess)
    if ier == 1 and np.all(np.isfinite(sol)):
        cache.append((float(m), sol.copy()))
        guess = sol

    m += dm
    while m <= ms_max + 1e-12:
        # Clip x1w so x4w = 1 - x1w - 2*x1w*ms*Mw > 0 (physical feasibility).
        # At ms=0 the solution can have x1w very close to 1; at higher ms the
        # maximum allowed x1w is 1/(1 + 2*ms*Mw).  Without this clip the
        # warm-start guess is often unphysical and fsolve diverges.
        x1w_max = 0.9999 / (1.0 + 2.0 * float(m) * Mw)
        guess_safe = guess.copy()
        guess_safe[1] = float(min(guess[1], x1w_max))
        sol, ier = _solve_one(m, guess_safe)
        if ier == 1 and np.all(np.isfinite(sol)):
            cache.append((float(m), sol.copy()))
            guess = sol
        else:
            break
        m = np.float64(m + dm)

    if verbose:
        if cache:
            avg_nfev = _total_nfev[0] / len(cache)
            print(f"  cache : {len(cache)} pts  ms = 0.000 → {cache[-1][0]:.3f}"
                  f"  avg nfev = {avg_nfev:.1f}")
        else:
            print("  cache : empty")

    return cache


def solve_elv(T, P_bar, ms_target, cache,
              params=None, xtol=1e-10, maxfev=2000,
              verbose=False):
    """
    Solve ELV at ms_target using a warm start from the continuation cache.
    Tries several x1c multipliers on failure.

    Returns (sol, ier, resnorm, mesg).
    """
    T         = np.float64(T)
    P_bar     = np.float64(P_bar)
    ms_target = np.float64(ms_target)

    ms_vals    = np.array([c[0] for c in cache])
    idx        = int(np.argmin(np.abs(ms_vals - ms_target)))
    base_guess = cache[idx][1].copy()

    x1c_multipliers = [1.0, 2.0, 5.0, 0.5, 0.1, 10.0, 0.01]
    sol = base_guess.copy()
    ier = 0
    resnorm = np.inf
    total_nfev = 0

    # Clip x1w in warm-start so that x4w = 1 - x1w - 2*x1w*ms*Mw > 0.
    x1w_max = 0.9999 / (1.0 + 2.0 * float(ms_target) * Mw)
    base_guess = base_guess.copy()
    base_guess[1] = float(min(base_guess[1], x1w_max))

    for i, mult in enumerate(x1c_multipliers):
        guess    = base_guess.copy()
        guess[4] = np.clip(base_guess[4] * mult, 1e-6, 1.0 - 1e-6)

        t0 = time.perf_counter()
        sol, info, ier, mesg = fsolve(
            ELV, guess, args=(T, P_bar * 1e5, ms_target, params),
            fprime=ELV_jac if USE_COMPLEX_JAC else None,
            full_output=True, xtol=xtol, maxfev=maxfev,
        )
        sol = np.asarray(sol, dtype=np.float64)
        total_nfev += info['nfev']

        converged_flag = (ier == 1 and np.all(np.isfinite(sol)))
        if converged_flag:
            res     = np.asarray(ELV(sol, T, P_bar * 1e5, ms_target, params),
                                 dtype=np.float64)
            resnorm = float(np.linalg.norm(res))

        if converged_flag and resnorm < 1e-6:
            return sol, ier, resnorm, mesg, total_nfev

    return sol, ier, resnorm, mesg, total_nfev


# ── Brent-loop flash ───────────────────────────────────────────────────────────

def flash_co2_h2o_salt_1d(
    T, P_bar, z_co2, m_tot,
    guess_table_fn,
    params=None,
    dm=0.1,
    ms_min=1e-8,
    ms_max=20.0,
    xtol_ms=1e-10,
    elv_res_tol=1e-6,
    scan_multipliers=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0),
    scan_absolutes=(1.0, 2.0, 5.0, 10.0),
    verbose=False,
    stability_check=False,
):
    """
    Two-phase flash using a full molality-continuation cache + Brent root-find.

    Algorithm
    ---------
    1. Build a continuation cache from ms=0 to ms_max (in steps of dm).
    2. Evaluate a water-balance residual at several trial ms values to bracket
       the root.
    3. Refine with scipy.optimize.brentq.
    """
    T     = float(T);  P_bar = float(P_bar)
    z_co2 = float(z_co2);  m_tot = float(m_tot)

    if not (0.0 < z_co2 < 1.0):
        raise ValueError("z_co2 must be strictly between 0 and 1.")
    if m_tot < 0.0:
        raise ValueError("m_tot must be >= 0.")
    if m_tot == 0.0:
        raise NotImplementedError("m_tot=0 (salt-free): use CPA flash instead.")

    if stability_check:
        hint = _cpa2_phase_check(T, P_bar, z_co2, params)
        if hint in ("single_phase_liquid", "single_phase_gas"):
            raise RuntimeError(
                f"CPA pre-check: {hint} at T={T:.1f}K P={P_bar:.2f}bar. "
                "Two-phase window only shrinks with salt — eCPA flash skipped.")

    n_co2_tot = z_co2
    n_h2o_tot = 1.0 - z_co2
    n_salt    = m_tot * n_h2o_tot * Mw
    if n_salt <= 0.0:
        raise ValueError("Computed n_salt <= 0.")

    base_guess = np.asarray(guess_table_fn(T, P_bar), dtype=np.float64)
    cache = build_continuation_cache(T, P_bar, base_guess, ms_max,
                                     params=params, dm=dm, verbose=verbose)

    if len(cache) == 0:
        raise RuntimeError("Continuation cache is empty — ELV failed even at ms=0.")

    _scan_nfev = []
    _scan_resnorm = []

    def elv_at_ms(ms):
        sol, ier, resnorm, mesg, nfev = solve_elv(T, P_bar, ms, cache,
                                                   params=params, verbose=False)
        ok = (ier == 1 and np.isfinite(resnorm)
              and resnorm <= elv_res_tol and np.all(np.isfinite(sol)))
        _scan_nfev.append(nfev)
        if ok:
            _scan_resnorm.append(resnorm)
        return ok, np.asarray(sol, dtype=np.float64), float(resnorm), mesg

    def water_residual(ms):
        ms = float(ms)
        if ms <= 0.0:
            return np.nan
        ok, sol, resnorm, mesg = elv_at_ms(ms)
        if not ok:
            return np.nan
        Zw, x1w, epsr, Zc, x1c, *_ = sol
        x2w = x1w * ms * Mw
        x3w = x2w
        x4w = 1.0 - x1w - x2w - x3w
        x4c = 1.0 - x1c
        if not (0.0 < x1w < 1.0 and 0.0 < x1c < 1.0):
            return np.nan
        if x2w <= 0.0 or x4c <= 0.0 or x4w <= 0.0:
            return np.nan
        N_aq = n_salt / x2w
        N_c  = (n_co2_tot - N_aq * x4w) / x4c
        if N_aq <= 0.0 or N_c <= 0.0:
            return np.nan
        return N_aq * x1w + N_c * x1c - n_h2o_tot

    # Bracketing scan — include every cached ms value so the sign change
    # is never missed when the cache stops short of ms_max.
    cache_ms_pts = [c[0] for c in cache if ms_min < c[0] < ms_max]
    ms_list = sorted(set(
        [min(ms_max, max(ms_min, mult * m_tot)) for mult in scan_multipliers]
        + [min(ms_max, max(ms_min, v))          for v in scan_absolutes]
        + [ms_min, ms_max]
        + cache_ms_pts
    ))
    finite_scan = []
    for ms in ms_list:
        r = water_residual(ms)
        if np.isfinite(r):
            finite_scan.append((ms, float(r)))

    if len(finite_scan) < 2:
        raise RuntimeError(
            "Could not evaluate water residual at ≥2 ms values. "
            "ELV likely failing for all trial ms.")

    a = b = fa = fb = None
    for (m1, r1), (m2, r2) in zip(finite_scan[:-1], finite_scan[1:]):
        if r1 == 0.0:
            a, fa, b, fb = m1, r1, m1, r1; break
        if r1 * r2 < 0.0:
            a, fa, b, fb = m1, r1, m2, r2; break

    if a is None:
        raise RuntimeError(
            "No sign change found in water residual scan. "
            "This may indicate a single-phase state at this T, P.")

    if verbose:
        avg_nfev   = sum(_scan_nfev)   / len(_scan_nfev)   if _scan_nfev   else float('nan')
        avg_resnorm = sum(_scan_resnorm) / len(_scan_resnorm) if _scan_resnorm else float('nan')
        print(f"  scan  : {len(finite_scan)} pts  avg nfev = {avg_nfev:.1f}"
              f"  avg resnorm = {avg_resnorm:.2e}"
              f"  bracket ms = [{a:.4f}, {b:.4f}]")

    ms_aq = float(brentq(lambda x: water_residual(x), a, b, xtol=xtol_ms))
    if ms_aq < 1e-8:
        ms_aq = 0.0

    ok, sol, resnorm, mesg = elv_at_ms(max(ms_aq, ms_min) if ms_aq == 0.0 else ms_aq)
    if not ok:
        raise RuntimeError(f"Final ELV failed at ms={ms_aq}: {mesg} resnorm={resnorm}")
    if verbose:
        print(f"  result: ms_aq = {ms_aq:.4f}  resnorm = {resnorm:.2e}  OK")

    Zw, x1w, epsr, Zc, x1c, *_ = sol
    x2w = x1w * ms_aq * Mw
    x3w = x2w
    x4w = 1.0 - x1w - x2w - x3w
    x4c = 1.0 - x1c
    N_aq = n_salt / x2w
    N_c  = (n_co2_tot - N_aq * x4w) / x4c

    return {
        "T": float(T), "P_bar": float(P_bar),
        "z_co2": float(z_co2), "m_tot": float(m_tot),
        "n_totals": dict(n_co2_tot=float(z_co2),
                         n_h2o_tot=float(1.0 - z_co2),
                         n_salt_tot=float(m_tot * (1.0 - z_co2) * Mw)),
        "ms_aq": float(ms_aq),
        "N_aq": float(N_aq), "N_c": float(N_c),
        "beta": float(N_c / (N_aq + N_c)),
        "x_aq": dict(x1w=float(x1w), x2w=float(x2w),
                     x3w=float(x3w), x4w=float(x4w)),
        "x_c":  dict(x1c=float(x1c), x4c=float(x4c)),
        "Z_aq": float(sol[0]), "Z_c": float(sol[3]),
        "sol":  np.asarray(sol, dtype=np.float64),
    }


# ── SSI flash (DEPRECATED — legacy outer ms_aq loop) ──────────────────────────
# DEPRECATED: flash_co2_h2o_salt_ssi uses an outer damped SSI loop over ms_aq
# and an inner 10-variable ELV fsolve.  It is retained only for benchmarking
# and historical comparison.  Do NOT use in production code.
#
# Use instead:
#   flash_co2_h2o_salt_kv   — K-value SSI (ms_aq as algebraic constraint, no
#                              outer loop); ~3–5× faster and more robust.
#   ecpa_stability_flash    — full hierarchical flash (stability → K-init →
#                              flash_co2_h2o_salt_kv; from ecpa.stability).
# ──────────────────────────────────────────────────────────────────────────────

def flash_co2_h2o_salt_ssi(
    T, P_bar, z_co2, m_tot,
    guess_table_fn=None,
    params=None,
    maxiter_ms=40,
    tol_ms=1e-8,
    omega=0.7,
    elv_xtol=1e-10,
    elv_maxfev=2000,
    elv_res_tol=1e-6,
    verbose=False,
    stability_check=False,
    initial_sol=None,
    initial_ms_aq=None,
):
    """
    DEPRECATED — use flash_co2_h2o_salt_kv or ecpa_stability_flash instead.

    Two-phase flash using a damped SSI (successive substitution) on ms_aq.

    Algorithm
    ---------
    1. Solve ELV at ms=0 (salt-free, table guess) — OR skip to step 2 if
       initial_sol / initial_ms_aq are provided (fast path from solution table).
    2. Estimate initial ms_aq from compositions + salt balance (or use initial_ms_aq).
    3. Iterate:
       a. Solve ELV at current ms_aq (warm-started from previous solution).
       b. Compute 2×2 H₂O/CO₂ material balance → N_aq, N_c.
       c. ms_aq_new = n_salt / (N_aq · x_H2O_aq · Mw)   [salt balance]
       d. Damped update: ms_aq += ω·(ms_aq_new − ms_aq)
       e. Converge when |Δms_aq| < tol_ms.

    No continuation cache or bracketing scan required.

    Fast-path parameters
    --------------------
    initial_sol    : ndarray (10,), optional
        Pre-computed ELV solution vector (e.g. from solution_table).  When
        provided together with initial_ms_aq the ms=0 cold-start is skipped
        entirely and SSI starts from this guess.  Requires initial_ms_aq.
    initial_ms_aq  : float, optional
        Initial aqueous molality estimate paired with initial_sol.
        Requires initial_sol.  guess_table_fn may be None when both are given.
    """
    T     = float(T);  P_bar = float(P_bar)
    z_co2 = float(z_co2);  m_tot = float(m_tot)

    if not (0.0 < z_co2 < 1.0):
        raise ValueError("z_co2 must be strictly between 0 and 1.")
    if m_tot < 0.0:
        raise ValueError("m_tot must be >= 0.")
    if m_tot == 0.0:
        raise NotImplementedError("m_tot=0 (salt-free): use CPA flash instead.")

    if stability_check:
        hint = _cpa2_phase_check(T, P_bar, z_co2, params)
        if hint in ("single_phase_liquid", "single_phase_gas"):
            raise RuntimeError(
                f"CPA pre-check: {hint} at T={T:.1f}K P={P_bar:.2f}bar. "
                "Two-phase window only shrinks with salt — eCPA flash skipped.")

    n_co2_tot = z_co2
    n_h2o_tot = 1.0 - z_co2
    n_salt    = m_tot * n_h2o_tot * Mw
    if n_salt <= 0.0:
        raise ValueError("n_salt <= 0. Check Mw units and m_tot.")

    x1c_retry_mults = [1.0, 2.0, 0.5, 5.0, 0.1, 10.0, 0.02]

    # Track total ELV function evaluations across all SSI iterations and retries.
    _total_elv_nfev = 0

    def _solve_elv(ms_aq, guess):
        nonlocal _total_elv_nfev
        best_sol, best_rn, best_ok = guess.copy(), np.inf, False
        # Clip x1w so x4w = 1 - x1w - 2*x1w*ms*Mw > 0 (physical feasibility).
        x1w_max = 0.9999 / (1.0 + 2.0 * float(ms_aq) * Mw) if ms_aq > 0 else 1.0
        guess = guess.copy()
        guess[1] = float(min(guess[1], x1w_max))
        for mult in x1c_retry_mults:
            g    = guess.copy()
            g[4] = np.clip(guess[4] * mult, 1e-6, 1.0 - 1e-6)
            sol, info, ier, mesg = fsolve(
                ELV, g, args=(T, P_bar * 1e5, ms_aq, params),
                fprime=ELV_jac if USE_COMPLEX_JAC else None,
                full_output=True, xtol=elv_xtol, maxfev=elv_maxfev,
            )
            _total_elv_nfev += info["nfev"]
            sol = np.asarray(sol, dtype=np.float64)
            res = np.asarray(ELV(sol, T, P_bar * 1e5, ms_aq, params), dtype=np.float64)
            rn  = float(np.linalg.norm(res))
            ok  = (ier == 1) and np.all(np.isfinite(sol)) and (rn < elv_res_tol)
            if ok:
                return sol, True, rn
            if rn < best_rn:
                best_sol, best_rn, best_ok = sol.copy(), rn, ok
        return best_sol, best_ok, best_rn

    def _mass_balance(x1w, x4w, x1c, x4c):
        det = x1w * x4c - x4w * x1c
        if abs(det) < 1e-14:
            raise ValueError(f"Degenerate phase compositions (det={det:.2e})")
        N_aq = (n_h2o_tot * x4c - n_co2_tot * x1c) / det
        N_c  = (n_co2_tot * x1w - n_h2o_tot * x4w) / det
        if N_aq <= 0.0 or N_c <= 0.0:
            raise ValueError(f"Non-physical amounts N_aq={N_aq:.4g} N_c={N_c:.4g}")
        ms_new = n_salt / (N_aq * x1w * Mw)
        if ms_new <= 0.0:
            raise ValueError(f"Non-physical ms_new={ms_new:.4g}")
        return float(N_aq), float(N_c), float(ms_new)

    # Step 0: get initial (sol, ms_aq) — either from provided guess or cold ms=0 solve
    use_fast_path = (initial_sol is not None) and (initial_ms_aq is not None)
    if use_fast_path:
        # Fast path: skip the ms=0 ELV solve entirely.
        # The caller provides an interpolated solution from the solution table.
        sol   = np.asarray(initial_sol, dtype=np.float64)
        ms_aq = float(np.clip(initial_ms_aq, m_tot * 0.1, m_tot * 100.0))
        if verbose:
            print(f"[SSI] fast path: initial ms_aq={ms_aq:.6f} (from table)")
    else:
        if guess_table_fn is None:
            raise ValueError(
                "guess_table_fn must be provided when initial_sol / initial_ms_aq "
                "are not given.  Pass a function from make_guess_fn() or use "
                "flash_co2_h2o_salt_fast() with a solution_guess_fn.")
        # Cold start: solve ELV at ms=0 (salt-free) to get an initial estimate
        base_guess = np.asarray(guess_table_fn(T, P_bar), dtype=np.float64)
        sol0, ok0, rn0 = _solve_elv(0.0, base_guess)
        _, x1w0, _, _, x1c0, *_ = sol0
        x4w0 = 1.0 - x1w0
        x4c0 = 1.0 - x1c0
        try:
            _, _, ms_aq = _mass_balance(x1w0, x4w0, x1c0, x4c0)
            ms_aq = float(np.clip(ms_aq, m_tot * 0.5, m_tot * 100.0))
        except ValueError:
            ms_aq = m_tot
        if verbose:
            print(f"[SSI] ms=0 ELV ok={ok0} rn={rn0:.2e}  initial ms_aq={ms_aq:.6f}")
        sol = sol0.copy()

    guess = sol.copy()
    delta = np.inf
    converged = False

    for it in range(maxiter_ms):
        sol, ok, rn = _solve_elv(ms_aq, guess)
        if not ok:
            if verbose:
                print(f"[SSI] iter {it+1}: ELV failed ms_aq={ms_aq:.6f} rn={rn:.2e}")
            break

        _, x1w, _, _, x1c, *_ = sol
        x2w = x1w * ms_aq * Mw
        x4w = 1.0 - x1w - 2.0 * x2w
        x4c = 1.0 - x1c

        if verbose:
            print(f"[SSI] iter {it+1}: ms_aq={ms_aq:.8f}  x1w={x1w:.6f}"
                  f"  x1c={x1c:.3e}  x4w={x4w:.6f}  rn={rn:.2e}")

        try:
            N_aq, N_c, ms_new = _mass_balance(x1w, x4w, x1c, x4c)
        except ValueError as exc:
            if verbose:
                print(f"[SSI] iter {it+1}: mass balance failed: {exc}")
            break

        delta  = abs(ms_new - ms_aq)
        ms_aq  = ms_aq + omega * (ms_new - ms_aq)
        guess  = sol.copy()
        if delta < tol_ms:
            converged = True
            break

    if not converged:
        raise RuntimeError(
            f"SSI ms_aq did not converge in {maxiter_ms} iterations "
            f"(T={T:.1f}K P={P_bar:.2f}bar m_tot={m_tot:.4f}, "
            f"last delta={delta:.2e}). "
            "Try increasing maxiter_ms or reducing omega.")

    _, x1w, _, _, x1c, *_ = sol
    x2w = x1w * ms_aq * Mw
    x3w = x2w
    x4w = 1.0 - x1w - 2.0 * x2w
    x4c = 1.0 - x1c
    N_aq, N_c, _ = _mass_balance(x1w, x4w, x1c, x4c)

    if verbose:
        print(f"[SSI] converged: ms_aq={ms_aq:.8f}  "
              f"beta={N_c/(N_aq+N_c):.6f}  iters={it+1}")

    return {
        "T": T, "P_bar": P_bar, "z_co2": z_co2, "m_tot": m_tot,
        "n_totals": dict(n_co2_tot=n_co2_tot,
                         n_h2o_tot=n_h2o_tot,
                         n_salt_tot=n_salt),
        "ms_aq": ms_aq,
        "N_aq": N_aq, "N_c": N_c,
        "beta": N_c / (N_aq + N_c),
        "x_aq": dict(x1w=x1w, x2w=x2w, x3w=x3w, x4w=x4w),
        "x_c":  dict(x1c=x1c, x4c=x4c),
        "Z_aq": float(sol[0]), "Z_c": float(sol[3]),
        "sol":  np.asarray(sol, dtype=np.float64),
        "n_iter_ms":   it + 1,
        "n_elv_nfev":  _total_elv_nfev,
    }


# ── Fast flash (solution-table warm start) ─────────────────────────────────────

def flash_co2_h2o_salt_fast(
    T, P_bar, z_co2, m_tot,
    solution_guess_fn,
    params=None,
    max_ssi_iter=10,
    tol_ms=1e-8,
    omega_warm=1.0,
    fallback_guess_table_fn=None,
    force_stability_check=False,
):
    """
    Two-phase flash with table warm-start and selective phase identification.

    Algorithm
    ---------
    1. Interpolate solution table → (sol_guess, ms_aq_guess, is_two_phase_hint).
    2. Phase identification:
       a. If is_two_phase_hint=True AND not force_stability_check:
          Trust the table (built from converged two-phase solutions) and skip the
          Michelsen TPD test.  Benchmarks show the stability test costs ~5× more
          than the warm-started flash itself.
       b. If is_two_phase_hint=False OR force_stability_check=True:
          Run ecpa_stability for a rigorous Michelsen TPD test.  This covers
          single-phase regions, near-boundary cells, and unconverged table entries
          (whose hint comes from nearest-neighbour fill and is less reliable).
    3. If confirmed single-phase → return immediately without flash.
    4. If two-phase → run undamped warm-started SSI (omega=1.0 by default).
       Near the solution, undamped SSI converges in ~3 iterations (vs ~12 for the
       damped cold-start).
    5. On failure → fall back to full cold-start SSI (if fallback provided).

    Parameters
    ----------
    solution_guess_fn : callable returned by make_solution_guess_fn()
        Signature: (T, P_bar, z_co2, ms) → (sol_10, ms_aq, is_two_phase_hint)
    max_ssi_iter : int
        Max SSI iterations for the warm-start flash (10 is safe; 5 is usually
        enough when starting from the table guess).
    omega_warm : float
        SSI damping for the warm-started flash.  Default 1.0 (no damping) is
        safe near the solution and gives fastest convergence.
    fallback_guess_table_fn : callable (T, P_bar) → np.ndarray, optional
        CPA guess table for a full cold-start SSI if the fast path fails.
        If None and the fast SSI fails, the RuntimeError propagates.
    force_stability_check : bool
        If True, always run ecpa_stability regardless of the table hint.
        Use for validation or near-boundary conditions where extra safety is
        needed.  Adds ~30 ms/call overhead.

    Returns
    -------
    dict with the same keys as flash_co2_h2o_salt_ssi, plus:
        "phase"   : "two_phase" or "single_phase"
        "stable"  : bool or None — True/False if stability test was run, None if
                    skipped (table hint used directly)
        "tpd_min" : float or None — TPD minimum if stability test was run
    """
    # Lazy import avoids circular dependency (stability.py imports from flash.py)
    from .stability import ecpa_stability

    T     = float(T);  P_bar = float(P_bar)
    z_co2 = float(z_co2);  m_tot = float(m_tot)

    # Step 1: interpolated guess + phase hint from solution table
    sol_guess, ms_aq_guess, is_two_phase_hint = solution_guess_fn(T, P_bar, z_co2, m_tot)

    # Step 2: decide whether to run the Michelsen TPD stability test
    stab_result = None
    if is_two_phase_hint and not force_stability_check:
        # Table confirms two-phase → skip stability test (5× cheaper)
        is_two_phase = True
        stable       = None   # not evaluated
        tpd_min      = None
    else:
        # Uncertain region (single-phase hint, unconverged cell, or forced check):
        # run the rigorous Michelsen TPD test
        stab_result  = ecpa_stability(z_co2, m_tot, T, P_bar, params)
        is_two_phase = not stab_result["stable"]
        stable       = stab_result["stable"]
        tpd_min      = float(stab_result["tpd_min"])

    if not is_two_phase:
        return {
            "T": T, "P_bar": P_bar, "z_co2": z_co2, "m_tot": m_tot,
            "phase":   "single_phase",
            "stable":  True,
            "tpd_min": tpd_min,
        }

    # Step 3: two-phase — run undamped warm-started SSI
    try:
        out = flash_co2_h2o_salt_ssi(
            T=T, P_bar=P_bar, z_co2=z_co2, m_tot=m_tot,
            params=params,
            maxiter_ms=max_ssi_iter,
            tol_ms=tol_ms,
            omega=omega_warm,
            initial_sol=np.asarray(sol_guess, dtype=np.float64),
            initial_ms_aq=float(ms_aq_guess),
        )
        out["phase"]   = "two_phase"
        out["stable"]  = stable
        out["tpd_min"] = tpd_min
        return out

    except RuntimeError:
        if fallback_guess_table_fn is not None:
            # Fast SSI failed — full cold-start SSI as fallback
            out = flash_co2_h2o_salt_ssi(
                T=T, P_bar=P_bar, z_co2=z_co2, m_tot=m_tot,
                guess_table_fn=fallback_guess_table_fn,
                params=params,
            )
            out["phase"]   = "two_phase"
            out["stable"]  = stable
            out["tpd_min"] = tpd_min
            return out
        raise


# ── K-value SSI flash ──────────────────────────────────────────────────────────

# Cold-start K-value candidates tried in sequence when K_init=None.
# (K1, K4) = (x1c/x1w, x4c/x4w): K1<1 (H₂O aqueous), K4>1 (CO₂ CO₂-rich).
# Ordered from typical reservoir conditions toward near-critical.
_KV_COLD_STARTS = [
    (0.005, 30.0),   # typical: deep reservoir, P > 50 bar
    (0.05,  5.0),    # near-critical: high T, moderate P
    (0.3,   2.0),    # very near-critical: K-values approaching 1
    (0.01,  15.0),   # intermediate: moderate T/P
]


def _flash_kv_core(
    T, P_bar, z_co2, m_tot, n_h2o, n_co2, n_salt,
    K1, K4, sol_aq_x0, sol_c_x0,
    maxiter, tol, accelerated, verbose,
    _lnphi_aq_inner, _lnphi_c_inner,
):
    """
    One K-value SSI attempt from a single (K1, K4) starting point.
    Returns the result dict on success, raises RuntimeError on failure.
    Called by flash_co2_h2o_salt_kv for each cold-start candidate.
    """
    K1 = float(np.clip(K1, 1e-9, 1.0 - 1e-9))
    K4 = float(np.clip(K4, 1.0 + 1e-9, 1e9))

    lnK    = np.array([np.log(K1), np.log(K4)])
    _m_jex = 1.0
    _g_prev = None
    sol_aq = np.asarray(sol_aq_x0, dtype=float) if sol_aq_x0 is not None else None
    sol_c  = np.asarray(sol_c_x0,  dtype=float) if sol_c_x0  is not None else None

    converged = False
    n_iter = 0
    x1w = x1c = x4w = x4c = x2w = x3w = float('nan')
    N_aq = N_c = ms_aq = float('nan')
    g_vec = np.zeros(2)

    for it in range(maxiter):
        K1, K4 = float(np.exp(lnK[0])), float(np.exp(lnK[1]))
        dK = K4 - K1
        if dK < 1e-9:
            raise RuntimeError(
                f"K4 ≈ K1 ({K4:.4g}) at iter {it+1} — near-critical.")

        # ── Step 1: scalar equation for x₁w ──────────────────────────────
        x1w_max = (n_h2o - 1e-10) / K1
        x1w_min = n_h2o / (n_co2 * K4 + n_h2o * K1) + 1e-10
        if x1w_min >= x1w_max:
            break

        def _eq_x1w(f):
            u = 1.0 - K1 * f
            v = n_h2o - K1 * f
            if u <= 0.0 or v <= 0.0 or f <= 0.0:
                return 1e10
            Naq_f = K4 * v / (u * f * dK)
            if Naq_f <= 0.0:
                return 1e10
            return f + 2.0 * n_salt / Naq_f + u / K4 - 1.0

        r_lo = _eq_x1w(x1w_min)
        r_hi = _eq_x1w(x1w_max)
        if not (np.isfinite(r_lo) and np.isfinite(r_hi)):
            break
        if r_lo * r_hi > 0.0:
            break

        try:
            x1w_new = brentq(_eq_x1w, x1w_min, x1w_max,
                             xtol=1e-12, rtol=1e-12, maxiter=100)
        except ValueError:
            break

        # ── Step 2: compositions and ms_aq analytically ───────────────────
        x1w = float(x1w_new)
        x1c = K1 * x1w
        x4c = 1.0 - x1c
        x4w = x4c / K4

        det = x1w * x4c - x4w * x1c
        if abs(det) < 1e-15:
            break
        N_aq = (n_h2o * x4c - n_co2 * x1c) / det
        N_c  = (n_co2 * x1w - n_h2o * x4w) / det
        if N_aq <= 0.0 or N_c <= 0.0:
            break

        ms_aq = n_salt / (N_aq * x1w * Mw)
        if ms_aq <= 0.0:
            break
        x2w = x3w = x1w * ms_aq * Mw

        # ── Step 3: lnφ via warm-started inner solves ─────────────────────
        try:
            lnphi1_aq, lnphi4_aq, sol_aq = _lnphi_aq_inner(
                x1w, ms_aq, T, P_bar, x0=sol_aq)
            lnphi1_c,  lnphi4_c,  sol_c  = _lnphi_c_inner(
                x1c, T, P_bar, x0=sol_c)
        except RuntimeError:
            break

        # ── Step 4: K-value residual ──────────────────────────────────────
        lnK_new = np.array([lnphi1_aq - lnphi1_c,
                             lnphi4_aq - lnphi4_c])
        g_vec = lnK_new - lnK

        if verbose:
            print(f"[KV] iter {it+1}: "
                  f"lnK=[{lnK[0]:.5f},{lnK[1]:.5f}]  "
                  f"x1w={x1w:.6f}  ms_aq={ms_aq:.5f}  "
                  f"|g|={np.linalg.norm(g_vec):.2e}")

        if np.linalg.norm(g_vec) < tol:
            converged = True
            lnK = lnK_new
            n_iter = it + 1
            break

        # ── Step 5: Jex et al. 2024 acceleration ─────────────────────────
        if accelerated and _g_prev is not None:
            num_a   = np.dot(_g_prev, _g_prev)
            denom_a = np.dot(_g_prev, _g_prev - g_vec)
            if abs(denom_a) > 1e-30:
                _m_jex = abs(num_a / denom_a * _m_jex)
                _m_jex = float(np.clip(_m_jex, 1.0, 10.0))
            else:
                _m_jex = 1.0
        _g_prev = g_vec.copy()

        lnK = lnK + _m_jex * g_vec
        n_iter = it + 1

    if not converged:
        raise RuntimeError(
            f"K-value SSI did not converge in {maxiter} iterations "
            f"(T={T:.1f}K P={P_bar:.2f}bar z={z_co2:.3f} ms={m_tot:.3f}, "
            f"last ‖g‖={np.linalg.norm(g_vec):.2e}).")

    beta  = N_c / (N_aq + N_c)
    Zw    = float(sol_aq[0]) if sol_aq is not None else float('nan')
    epsr  = float(sol_aq[1]) if sol_aq is not None else float('nan')
    chi1w = float(sol_aq[2]) if sol_aq is not None else float('nan')
    Zc    = float(sol_c[0])  if sol_c  is not None else float('nan')
    chi1c = float(sol_c[1])  if sol_c  is not None else float('nan')

    sol_vec = np.array([Zw, x1w, epsr, Zc, x1c,
                        chi1w, chi1c, float('nan'), float('nan'), float('nan')])
    return {
        "T": T, "P_bar": P_bar, "z_co2": z_co2, "m_tot": m_tot,
        "n_totals": dict(n_co2_tot=n_co2, n_h2o_tot=n_h2o, n_salt_tot=n_salt),
        "ms_aq":  ms_aq,
        "N_aq":   N_aq,  "N_c":   N_c,
        "beta":   beta,
        "x_aq":   dict(x1w=x1w, x2w=x2w, x3w=x3w, x4w=x4w),
        "x_c":    dict(x1c=x1c, x4c=x4c),
        "Z_aq":   Zw,    "Z_c":   Zc,
        "sol":    sol_vec,
        "n_iter_ms":  n_iter,
        "K_vals":     (float(np.exp(lnK[0])), float(np.exp(lnK[1]))),
        "sol_aq_x0":  np.asarray(sol_aq) if sol_aq is not None else None,
        "sol_c_x0":   np.asarray(sol_c)  if sol_c  is not None else None,
    }


def flash_co2_h2o_salt_kv(
    T, P_bar, z_co2, m_tot,
    K_init=None,
    sol_aq_x0=None,
    sol_c_x0=None,
    warm_start=None,
    params=None,
    maxiter=80,
    tol=1e-8,
    accelerated=True,
    verbose=False,
    use_newton=True,
    newton_tol=1e-4,
    max_newton=5,
):
    """
    Two-phase flash via K-value SSI with Jex et al. (2024) acceleration.

    Replaces the ms_aq outer-loop approach with a single-level iteration on
    K-values (K₁ = x1c/x1w for H₂O, K₄ = x4c/x4w for CO₂).  NaCl is
    non-volatile (K_NaCl = 0), so ms_aq is computed algebraically at every
    step from the current K-values — it is never an independent variable.

    Each iteration:
      1. Solve one scalar equation for x₁w given (K₁, K₄) and feed
         constraints (including salt).  All other compositions and ms_aq
         follow analytically from x₁w.
      2. Evaluate lnφ for each phase via warm-started inner Newton/fsolve.
      3. Update lnK_i ← lnφ_i^aq − lnφ_i^c with Jex acceleration.

    Convergence criterion: ‖lnK_new − lnK‖ < tol, which equals the
    fugacity-equality residual ‖ln(f_i^aq) − ln(f_i^c)‖.

    Parameters
    ----------
    K_init : (float, float) or None
        Initial (K₁, K₄).  From a solution table: K₁ = x1c/x1w,
        K₄ = (1−x1c)/x4w.  If None, all candidates in _KV_COLD_STARTS
        are tried in sequence; the first to converge is returned.
    sol_aq_x0 : array-like [Zw, epsr, chi1w] or None
        Warm-start for the aqueous phase inner solve.
    sol_c_x0 : array-like [Zc, chi1c] or None
        Warm-start for the CO₂-rich phase inner solve.
    warm_start : callable or None
        Optional warm-start provider with signature
        ``warm_start(T, P_bar, z_co2, m_tot) -> WarmStartGuess | None``.
        If provided and ``K_init`` is None, it is called first to obtain
        K_init, sol_aq_x0, and sol_c_x0.  On failure (returns None or raises)
        the solver falls back to its own cold-start candidates.
        Typical provider: ``ScanTableWarmStart`` (from ``ecpa.warmstart``).
    tol : float
        Convergence tolerance on ‖lnK_new − lnK‖ (= fugacity residual).
    accelerated : bool
        Apply Jex et al. 2024 step-size acceleration on the lnK update.

    Returns
    -------
    Same dict format as flash_co2_h2o_salt_ssi, plus:
        "K_vals"     : (K1, K4) at convergence
        "sol_aq_x0"  : [Zw, epsr, chi1w] for warm-starting the next call
        "sol_c_x0"   : [Zc, chi1c] for warm-starting the next call
    """
    # Lazy import avoids circular dependency (stability.py imports from flash.py)
    from .stability import (_lnphi_aq_inner, _lnphi_c_inner,
                            _apply_params, _restore_params, _wilson_K)

    T     = float(T);   P_bar = float(P_bar)
    z_co2 = float(z_co2);  m_tot = float(m_tot)

    if not (0.0 < z_co2 < 1.0):
        raise ValueError("z_co2 must be strictly between 0 and 1.")
    if m_tot <= 0.0:
        raise ValueError("m_tot must be > 0.")

    # ── Warm-start provider ───────────────────────────────────────────────────
    # Apply if K_init not already supplied explicitly.
    if warm_start is not None and K_init is None:
        try:
            guess = warm_start(T, P_bar, z_co2, m_tot)
            if guess is not None:
                K_init    = guess.K_init
                if sol_aq_x0 is None:
                    sol_aq_x0 = guess.sol_aq_x0
                if sol_c_x0 is None:
                    sol_c_x0  = guess.sol_c_x0
        except Exception:
            pass   # fall through to cold-start candidates below

    # Feed (basis: n_H2O + n_CO2 = 1)
    n_h2o  = 1.0 - z_co2
    n_co2  = z_co2
    n_salt = m_tot * n_h2o * Mw   # moles NaCl per basis

    saved = _apply_params(params)
    try:
        # ── Build list of (K1, K4) starting points to try ────────────────────
        # When a warm-start K_init is available, try it first; if it fails,
        # fall through to the standard cold-start candidates so robustness is
        # unchanged relative to calling without a warm-start.
        if K_init is not None:
            cold_starts = [(float(K_init[0]), float(K_init[1]))] + list(_KV_COLD_STARTS)
        else:
            cold_starts = list(_KV_COLD_STARTS)

        for i_cs, (_cs_K1, _cs_K4) in enumerate(cold_starts):
            # Only pass Newton-state warm-start on the first (warm-start) attempt;
            # cold-start candidates should initialise their own inner solves.
            _sol_aq = sol_aq_x0 if i_cs == 0 else None
            _sol_c  = sol_c_x0  if i_cs == 0 else None
            try:
                return _flash_kv_single(
                    T, P_bar, z_co2, m_tot, n_h2o, n_co2, n_salt,
                    K1=_cs_K1, K4=_cs_K4,
                    sol_aq_x0=_sol_aq, sol_c_x0=_sol_c,
                    maxiter=maxiter, tol=tol, accelerated=accelerated,
                    verbose=verbose,
                    use_newton=use_newton, newton_tol=newton_tol,
                    max_newton=max_newton,
                    _lnphi_aq_inner=_lnphi_aq_inner,
                    _lnphi_c_inner=_lnphi_c_inner,
                )
            except RuntimeError:
                continue

        raise RuntimeError(
            f"K-value SSI did not converge with any of {len(cold_starts)} "
            f"starts (T={T:.1f}K P={P_bar:.2f}bar "
            f"z={z_co2:.3f} ms={m_tot:.3f}).")

    finally:
        _restore_params(saved, params)


def _one_kv_eval(lnK, n_h2o, n_co2, n_salt, T, P_bar,
                 sol_aq_ws, sol_c_ws,
                 _lnphi_aq_inner, _lnphi_c_inner):
    """Evaluate the eCPA K-value residual at a given lnK.

    Solves for x1w (Brent), computes all compositions, evaluates lnφ for both
    phases, and returns the fugacity-residual vector g = lnφ^aq − lnφ^c − lnK
    together with the full composition/phase state.

    Used by the outer Newton polish to build the forward-difference Jacobian.
    Raises RuntimeError on any numerical failure.
    """
    K1, K4 = float(np.exp(lnK[0])), float(np.exp(lnK[1]))
    dK = K4 - K1
    if dK < 1e-9:
        raise RuntimeError("K4 ≈ K1 in _one_kv_eval")
    x1w_max = (n_h2o - 1e-10) / K1
    x1w_min = n_h2o / (n_co2 * K4 + n_h2o * K1) + 1e-10
    if x1w_min >= x1w_max:
        raise RuntimeError("degenerate bracket in _one_kv_eval")

    def _eq(f):
        u = 1.0 - K1 * f;  v = n_h2o - K1 * f
        if u <= 0.0 or v <= 0.0 or f <= 0.0:
            return 1e10
        Naq_f = K4 * v / (u * f * dK)
        if Naq_f <= 0.0:
            return 1e10
        return f + 2.0 * n_salt / Naq_f + u / K4 - 1.0

    r_lo, r_hi = _eq(x1w_min), _eq(x1w_max)
    if not (np.isfinite(r_lo) and np.isfinite(r_hi)) or r_lo * r_hi > 0.0:
        raise RuntimeError("no sign change in _one_kv_eval")

    x1w = float(brentq(_eq, x1w_min, x1w_max, xtol=1e-12, rtol=1e-12, maxiter=100))
    x1c = K1 * x1w;  x4c = 1.0 - x1c;  x4w = x4c / K4
    det = x1w * x4c - x4w * x1c
    if abs(det) < 1e-15:
        raise RuntimeError("singular det in _one_kv_eval")
    N_aq = (n_h2o * x4c - n_co2 * x1c) / det
    N_c  = (n_co2 * x1w - n_h2o * x4w) / det
    if N_aq <= 0.0 or N_c <= 0.0:
        raise RuntimeError("negative phase amounts in _one_kv_eval")
    ms_aq = n_salt / (N_aq * x1w * Mw)
    if ms_aq <= 0.0:
        raise RuntimeError("non-positive ms_aq in _one_kv_eval")
    x2w = x3w = x1w * ms_aq * Mw

    lnphi1_aq, lnphi4_aq, sol_aq = _lnphi_aq_inner(x1w, ms_aq, T, P_bar, x0=sol_aq_ws)
    lnphi1_c,  lnphi4_c,  sol_c  = _lnphi_c_inner(x1c, T, P_bar, x0=sol_c_ws)

    lnK_new = np.array([lnphi1_aq - lnphi1_c, lnphi4_aq - lnphi4_c])
    return dict(
        g=lnK_new - lnK,
        x1w=x1w, x1c=x1c, x4c=x4c, x4w=x4w, x2w=x2w, x3w=x3w,
        N_aq=N_aq, N_c=N_c, ms_aq=ms_aq, sol_aq=sol_aq, sol_c=sol_c,
    )


def _flash_kv_single(
    T, P_bar, z_co2, m_tot, n_h2o, n_co2, n_salt,
    K1, K4,
    sol_aq_x0, sol_c_x0,
    maxiter, tol, accelerated, verbose,
    use_newton, newton_tol, max_newton,
    _lnphi_aq_inner, _lnphi_c_inner,
):
    """One K-value SSI attempt from a single (K1, K4) starting point.
    Returns the result dict on convergence; raises RuntimeError on failure."""
    # Safety clips: K1 (H₂O) should be < 1, K4 (CO₂) should be > 1
    K1 = float(np.clip(K1, 1e-9, 1.0 - 1e-9))
    K4 = float(np.clip(K4, 1.0 + 1e-9, 1e9))

    lnK       = np.array([np.log(K1), np.log(K4)])
    _m_jex    = 1.0
    _g_prev   = None
    sol_aq    = np.asarray(sol_aq_x0, dtype=float) if sol_aq_x0 is not None else None
    sol_c     = np.asarray(sol_c_x0,  dtype=float) if sol_c_x0  is not None else None

    converged    = False
    n_iter       = 0
    _resid_norm  = np.inf
    # Working variables (populated in loop, needed for result dict)
    x1w = x1c = x4w = x4c = x2w = x3w = float('nan')
    N_aq = N_c = ms_aq = float('nan')
    g_vec = np.zeros(2)

    for it in range(maxiter):
        K1, K4 = float(np.exp(lnK[0])), float(np.exp(lnK[1]))
        dK = K4 - K1
        if dK < 1e-9:
            raise RuntimeError(
                f"K4 ≈ K1 ({K4:.4g} ≈ {K1:.4g}) at iter {it+1} — "
                "near-critical; K-value SSI cannot proceed.")

        # ── Step 1: scalar equation for x₁w ─────────────────────────────
        #
        # Aqueous mole-fraction normalisation, with salt and CO₂ expressed
        # via K-values and the analytical N_aq formula:
        #
        #   x₁w  +  2·n_salt / N_aq(x₁w)  +  (1 − K₁·x₁w)/K₄  =  1
        #
        # where N_aq = K₄·(n_h2o − K₁·x₁w) / [(1 − K₁·x₁w)·x₁w·(K₄ − K₁)]
        #
        # Valid domain: x₁w ∈ (x1w_min, x1w_max) so that N_aq > 0, N_c > 0.
        #   x1w_max = n_h2o / K₁               (N_aq numerator > 0)
        #   x1w_min = n_h2o / (n_co2·K₄ + n_h2o·K₁)  (N_c numerator > 0)

        x1w_max = (n_h2o - 1e-10) / K1
        x1w_min = n_h2o / (n_co2 * K4 + n_h2o * K1) + 1e-10
        if x1w_min >= x1w_max:
            break   # degenerate bracket — K-values not meaningful

        def _eq_x1w(f):
            u = 1.0 - K1 * f        # = x4c  (must be > 0)
            v = n_h2o - K1 * f      # N_aq numerator (must be > 0)
            if u <= 0.0 or v <= 0.0 or f <= 0.0:
                return 1e10
            Naq_f = K4 * v / (u * f * dK)
            if Naq_f <= 0.0:
                return 1e10
            return f + 2.0 * n_salt / Naq_f + u / K4 - 1.0

        r_lo = _eq_x1w(x1w_min)
        r_hi = _eq_x1w(x1w_max)
        if not (np.isfinite(r_lo) and np.isfinite(r_hi)):
            break
        if r_lo * r_hi > 0.0:
            # No sign change — can occur during the first few iterations
            # from a poor start.
            break

        try:
            x1w_new = brentq(_eq_x1w, x1w_min, x1w_max,
                              xtol=1e-12, rtol=1e-12, maxiter=100)
        except ValueError:
            break

        # ── Step 2: all compositions and ms_aq, analytically ─────────────
        x1w = float(x1w_new)
        x1c = K1 * x1w
        x4c = 1.0 - x1c
        x4w = x4c / K4

        # N_aq, N_c from 2×2 H₂O / CO₂ material balance (Cramer's rule)
        det = x1w * x4c - x4w * x1c
        if abs(det) < 1e-15:
            break
        N_aq = (n_h2o * x4c - n_co2 * x1c) / det
        N_c  = (n_co2 * x1w - n_h2o * x4w) / det
        if N_aq <= 0.0 or N_c <= 0.0:
            break

        ms_aq = n_salt / (N_aq * x1w * Mw)
        if ms_aq <= 0.0:
            break
        x2w = x3w = x1w * ms_aq * Mw

        # ── Step 3: lnφ evaluation (warm-started inner solves) ────────────
        try:
            lnphi1_aq, lnphi4_aq, sol_aq = _lnphi_aq_inner(
                x1w, ms_aq, T, P_bar, x0=sol_aq)
            lnphi1_c,  lnphi4_c,  sol_c  = _lnphi_c_inner(
                x1c, T, P_bar, x0=sol_c)
        except RuntimeError:
            break

        # ── Step 4: K-value residual = fugacity equality residual ─────────
        # g_i = lnφ_i^aq − lnφ_i^c − lnK_i = ln(f_i^aq) − ln(f_i^c)
        lnK_new = np.array([lnphi1_aq - lnphi1_c,
                             lnphi4_aq - lnphi4_c])
        g_vec   = lnK_new - lnK

        if verbose:
            print(f"[KV] iter {it+1}: "
                  f"lnK=[{lnK[0]:.5f},{lnK[1]:.5f}]  "
                  f"x1w={x1w:.6f}  ms_aq={ms_aq:.5f}  "
                  f"|g|={np.linalg.norm(g_vec):.2e}")

        _resid_norm = float(np.linalg.norm(g_vec))
        if _resid_norm < tol:
            converged = True
            lnK   = lnK_new
            n_iter = it + 1
            break
        if use_newton and _resid_norm < newton_tol:
            n_iter = it + 1
            break   # hand off to outer Newton polish

        # ── Step 5: Jex et al. 2024 acceleration ─────────────────────────
        if accelerated and _g_prev is not None:
            num_a   = np.dot(_g_prev, _g_prev)
            denom_a = np.dot(_g_prev, _g_prev - g_vec)
            if abs(denom_a) > 1e-30:
                _m_jex = abs(num_a / denom_a * _m_jex)
                _m_jex = float(np.clip(_m_jex, 1.0, 10.0))
            else:
                _m_jex = 1.0
        _g_prev = g_vec.copy()

        lnK = lnK + _m_jex * g_vec
        n_iter = it + 1

    n_ssi_iters    = n_iter
    n_newton_iters = 0

    # ── Outer Newton polish ───────────────────────────────────────────────────
    # Once the K-value SSI residual falls below newton_tol, switch to
    # Newton–Raphson with a forward-difference 2×2 Jacobian ∂g/∂lnK for
    # quadratic convergence to the final tolerance.
    # NOTE: this is the *outer* Newton on K-values, distinct from the inner
    # per-phase EoS Newton solves for (Z, εr, χ) that run inside every
    # lnφ evaluation.
    if use_newton and not converged and np.isfinite(_resid_norm) and _resid_norm < newton_tol:
        # Save state so we can revert if Newton diverges or fails
        _lnK_pre    = lnK.copy();        _g_pre      = g_vec.copy()
        _x1w_pre    = x1w;               _x1c_pre    = x1c
        _x4c_pre    = x4c;               _x4w_pre    = x4w
        _x2w_pre    = x2w;               _x3w_pre    = x3w
        _N_aq_pre   = N_aq;              _N_c_pre    = N_c
        _ms_aq_pre  = ms_aq
        _sol_aq_pre = np.asarray(sol_aq).copy() if sol_aq is not None else None
        _sol_c_pre  = np.asarray(sol_c).copy()  if sol_c  is not None else None
        _resid_pre  = _resid_norm

        _newton_ok = False
        _h_fd = 1e-5   # forward-difference step in lnK space

        for _nit in range(max_newton):
            # Build 2×2 Jacobian by forward differences on lnK
            _J = np.zeros((2, 2))
            _jac_ok = True
            for _j in range(2):
                _lnK_p = lnK.copy();  _lnK_p[_j] += _h_fd
                try:
                    _ev = _one_kv_eval(_lnK_p, n_h2o, n_co2, n_salt, T, P_bar,
                                       sol_aq, sol_c,
                                       _lnphi_aq_inner, _lnphi_c_inner)
                    _J[:, _j] = (_ev['g'] - g_vec) / _h_fd
                except RuntimeError:
                    _jac_ok = False;  break
            if not _jac_ok:
                break

            # Newton step: J · dlnK = −g
            try:
                _dlnK = np.linalg.solve(_J, -g_vec)
            except np.linalg.LinAlgError:
                break
            _dlnK = np.clip(_dlnK, -5.0, 5.0)

            # Evaluate residual at lnK + dlnK
            try:
                _ev = _one_kv_eval(lnK + _dlnK, n_h2o, n_co2, n_salt, T, P_bar,
                                   sol_aq, sol_c,
                                   _lnphi_aq_inner, _lnphi_c_inner)
            except RuntimeError:
                break
            if not np.all(np.isfinite(_ev['g'])):
                break

            lnK   = lnK + _dlnK
            g_vec = _ev['g']
            x1w   = _ev['x1w'];  x1c   = _ev['x1c'];  x4c   = _ev['x4c']
            x4w   = _ev['x4w'];  x2w   = _ev['x2w'];  x3w   = _ev['x3w']
            N_aq  = _ev['N_aq']; N_c   = _ev['N_c'];  ms_aq = _ev['ms_aq']
            sol_aq = _ev['sol_aq'];  sol_c = _ev['sol_c']

            n_newton_iters += 1
            _resid_norm = float(np.linalg.norm(g_vec))
            if _resid_norm < tol:
                converged = True;  _newton_ok = True;  break

        # Revert if Newton failed to converge or made things worse
        if not converged:
            if not _newton_ok or _resid_norm > _resid_pre:
                lnK   = _lnK_pre;   g_vec = _g_pre
                x1w   = _x1w_pre;   x1c   = _x1c_pre;  x4c   = _x4c_pre
                x4w   = _x4w_pre;   x2w   = _x2w_pre;  x3w   = _x3w_pre
                N_aq  = _N_aq_pre;  N_c   = _N_c_pre;  ms_aq = _ms_aq_pre
                sol_aq = _sol_aq_pre;  sol_c = _sol_c_pre
                n_newton_iters = 0

    if not converged:
        raise RuntimeError(
            f"K-value SSI did not converge in {maxiter} iterations "
            f"(T={T:.1f}K P={P_bar:.2f}bar z={z_co2:.3f} ms={m_tot:.3f}, "
            f"last ‖g‖={np.linalg.norm(g_vec):.2e}).")

    # ── Build result dict ─────────────────────────────────────────────────
    beta = N_c / (N_aq + N_c)
    Zw   = float(sol_aq[0]) if sol_aq is not None else float('nan')
    epsr = float(sol_aq[1]) if sol_aq is not None else float('nan')
    chi1w = float(sol_aq[2]) if sol_aq is not None else float('nan')
    Zc   = float(sol_c[0])  if sol_c  is not None else float('nan')
    chi1c = float(sol_c[1]) if sol_c  is not None else float('nan')

    # 10-element sol array compatible with ELV layout for warm-starting;
    # the chi-derivative slots [7-9] are NaN (not computed here).
    sol_vec = np.array([Zw, x1w, epsr, Zc, x1c,
                        chi1w, chi1c, float('nan'), float('nan'), float('nan')])

    return {
        "T": T, "P_bar": P_bar, "z_co2": z_co2, "m_tot": m_tot,
        "n_totals": dict(n_co2_tot=n_co2, n_h2o_tot=n_h2o,
                         n_salt_tot=n_salt),
        "ms_aq":  ms_aq,
        "N_aq":   N_aq,  "N_c":  N_c,
        "beta":   beta,
        "x_aq":   dict(x1w=x1w, x2w=x2w, x3w=x3w, x4w=x4w),
        "x_c":    dict(x1c=x1c, x4c=x4c),
        "Z_aq":   Zw,    "Z_c":  Zc,
        "sol":    sol_vec,
        "n_iter_ms":      n_ssi_iters + n_newton_iters,  # backward-compatible total
        "n_ssi_iter":     n_ssi_iters,
        "n_newton_iter":  n_newton_iters,
        "K_vals":    (float(np.exp(lnK[0])), float(np.exp(lnK[1]))),
        "sol_aq_x0": np.asarray(sol_aq) if sol_aq is not None else None,
        "sol_c_x0":  np.asarray(sol_c)  if sol_c  is not None else None,
    }


def flash_co2_h2o_salt_fast_kv(
    T, P_bar, z_co2, m_tot,
    solution_guess_fn,
    params=None,
    maxiter=10,
    tol=1e-8,
    accelerated=True,
    fallback_to_ssi=True,
    force_stability_check=False,
):
    """
    Fast two-phase flash: solution-table warm start + K-value SSI.

    Replaces the ms_aq SSI in flash_co2_h2o_salt_fast with K-value SSI.
    The solution table provides (K₁, K₄) and inner ZChi warm-starts so that
    only ~3–5 outer K-value iterations are needed.

    Parameters
    ----------
    solution_guess_fn : callable
        (T, P_bar, z_co2, ms) → (sol_10, ms_aq, is_two_phase_hint)
    fallback_to_ssi : bool
        If K-value SSI fails, fall back to flash_co2_h2o_salt_ssi with the
        same table warm-start.
    force_stability_check : bool
        Always run Michelsen TPD even when the table says two-phase.
    """
    from .stability import ecpa_stability

    T     = float(T);  P_bar = float(P_bar)
    z_co2 = float(z_co2);  m_tot = float(m_tot)

    # Step 1: table interpolation
    sol_guess, ms_aq_guess, is_two_phase_hint = solution_guess_fn(
        T, P_bar, z_co2, m_tot)

    # Step 2: phase identification
    stab_result = None
    if is_two_phase_hint and not force_stability_check:
        is_two_phase = True
        stable = None
        tpd_min = None
    else:
        stab_result  = ecpa_stability(z_co2, m_tot, T, P_bar, params)
        is_two_phase = not stab_result["stable"]
        stable       = stab_result["stable"]
        tpd_min      = float(stab_result["tpd_min"])

    if not is_two_phase:
        return {
            "T": T, "P_bar": P_bar, "z_co2": z_co2, "m_tot": m_tot,
            "phase": "single_phase", "stable": True, "tpd_min": tpd_min,
        }

    # Step 3: extract K-values and ZChi warm-starts from the table solution
    sol_g   = np.asarray(sol_guess, dtype=float)
    x1w_g   = float(sol_g[1])
    x1c_g   = float(sol_g[4])
    x4c_g   = 1.0 - x1c_g
    x2w_g   = x1w_g * float(ms_aq_guess) * Mw
    x4w_g   = max(1.0 - x1w_g - 2.0 * x2w_g, 1e-9)

    K1_g = x1c_g / max(x1w_g, 1e-9)
    K4_g = x4c_g / x4w_g

    sol_aq_x0 = sol_g[[0, 2, 5]]   # [Zw, epsr, chi1w]
    sol_c_x0  = sol_g[[3, 6]]      # [Zc, chi1c]

    # Step 4: K-value SSI
    try:
        out = flash_co2_h2o_salt_kv(
            T=T, P_bar=P_bar, z_co2=z_co2, m_tot=m_tot,
            K_init=(K1_g, K4_g),
            sol_aq_x0=sol_aq_x0,
            sol_c_x0=sol_c_x0,
            params=params,
            maxiter=maxiter,
            tol=tol,
            accelerated=accelerated,
        )
        out["phase"]   = "two_phase"
        out["stable"]  = stable
        out["tpd_min"] = tpd_min
        return out

    except RuntimeError:
        if not fallback_to_ssi:
            raise
        # Fall back to ms_aq SSI with same table warm-start
        out = flash_co2_h2o_salt_ssi(
            T=T, P_bar=P_bar, z_co2=z_co2, m_tot=m_tot,
            params=params,
            initial_sol=sol_g,
            initial_ms_aq=float(ms_aq_guess),
        )
        out["phase"]   = "two_phase"
        out["stable"]  = stable
        out["tpd_min"] = tpd_min
        return out


# Registry — maps algo name to function.  Remove 'brent' entry to drop Brent.
FLASH_ALGORITHMS = {
    "ssi":   flash_co2_h2o_salt_ssi,
    "brent": flash_co2_h2o_salt_1d,
}


def get_flash_fn(algo: str = "ssi"):
    """Return the flash function for the given algorithm name."""
    if algo not in FLASH_ALGORITHMS:
        raise ValueError(f"Unknown flash algorithm '{algo}'. "
                         f"Available: {list(FLASH_ALGORITHMS)}")
    return FLASH_ALGORITHMS[algo]
