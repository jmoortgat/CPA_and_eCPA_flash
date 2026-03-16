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
import time
import warnings

import numpy as np
from scipy.optimize import fsolve, brentq

from .constants import Mw
from .elv import ELV, ELV_jac, USE_COMPLEX_JAC


# ── Helpers ────────────────────────────────────────────────────────────────────

def _cpa2_phase_check(T: float, P_bar: float, z_co2: float, params) -> str:
    """
    Quick CPA2 salt-free flash to classify the likely phase state.

    Returns: 'two_phase' | 'single_phase_liquid' | 'single_phase_gas' | 'unknown'
    CPA2 is salt-free, so the two-phase window only shrinks with salt —
    this provides a conservative pre-filter for bulk scans.
    """
    try:
        import CPA2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = CPA2.flash_co2_h2o_tpz(T=float(T), P_bar=float(P_bar),
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
        raise NotImplementedError("m_tot=0 (salt-free): use CPA2 flash instead.")

    if stability_check:
        hint = _cpa2_phase_check(T, P_bar, z_co2, params)
        if hint in ("single_phase_liquid", "single_phase_gas"):
            raise RuntimeError(
                f"CPA2 pre-check: {hint} at T={T:.1f}K P={P_bar:.2f}bar. "
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


# ── SSI flash ──────────────────────────────────────────────────────────────────

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
        raise NotImplementedError("m_tot=0 (salt-free): use CPA2 flash instead.")

    if stability_check:
        hint = _cpa2_phase_check(T, P_bar, z_co2, params)
        if hint in ("single_phase_liquid", "single_phase_gas"):
            raise RuntimeError(
                f"CPA2 pre-check: {hint} at T={T:.1f}K P={P_bar:.2f}bar. "
                "Two-phase window only shrinks with salt — eCPA flash skipped.")

    n_co2_tot = z_co2
    n_h2o_tot = 1.0 - z_co2
    n_salt    = m_tot * n_h2o_tot * Mw
    if n_salt <= 0.0:
        raise ValueError("n_salt <= 0. Check Mw units and m_tot.")

    x1c_retry_mults = [1.0, 2.0, 0.5, 5.0, 0.1, 10.0, 0.02]

    def _solve_elv(ms_aq, guess):
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
        "n_iter_ms": it + 1,
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
        CPA2 guess table for a full cold-start SSI if the fast path fails.
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
