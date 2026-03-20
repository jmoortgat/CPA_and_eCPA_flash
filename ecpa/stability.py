"""
eCPA phase stability analysis — Michelsen tangent-plane distance (TPD).

Generalises the CPA2 stability approach to the eCPA CO₂ + H₂O + NaCl system.

Public API
----------
ecpa_lnphi_aq(x1w, ms, T, P, params)  -> (lnphi_H2O, lnphi_CO2)
ecpa_lnphi_c(x1c, T, P, params)       -> (lnphi_H2O, lnphi_CO2)

ecpa_stability(z_co2, ms, T, P, params, guess_table_fn=None)
    -> dict: stable, tpd_min, trial_type, trial_composition, message

ecpa_stability_flash(z_co2, ms, T, P, params, guess_table_fn)
    -> flash result dict (same format as flash_co2_h2o_salt_ssi) if two-phase,
       or dict with phase='single_phase' if stable.

Algorithm overview
------------------
Phase stability is checked via the Michelsen tangent-plane distance (TPD).
For a feed (z_co2, ms) at (T, P), the system is single-phase stable if and
only if tpd(w) >= 0 for every trial composition w.  The stationary conditions
of tpd are the same as the fugacity-equality conditions of the flash — so the
SSI update rule is identical to the flash SSI, just without the mass-balance
constraint coupling the two phases.

Two trials are tested:
  1. CO₂-rich trial   — start from a nearly-pure-CO₂ composition (x1c ≈ 0)
  2. Aqueous trial    — start from a nearly-pure-brine composition (x1w ≈ 1)

For each trial, SSI iterates:
    w_i^new = exp(d_i - lnφ_i(w_normalised))
until Σ wᵢ converges.  If Σ wᵢ > 1 at convergence → tpd < 0 → unstable.

Salt constraint
---------------
Na⁺ and Cl⁻ are non-partitioning: all salt remains in the aqueous phase.
For the aqueous trial, ms_trial = ms (feed molality) is kept fixed —
only the CO₂/H₂O split in the aqueous mole fractions varies.
The CO₂-rich trial has ms = 0 (no salt).

Within-phase lnφ evaluation
----------------------------
CO₂-rich phase: simple 2-variable SSI on (Zc, chi1c).

Aqueous phase: scipy.fsolve on 3 variables (Zw, epsr, chi1w).
  The chi cross-derivatives (Ndchi1WdNw, Ndchi1WdNc, Vdchi1WdV) are solved
  analytically as a 2×2 linear system given (Zw, chi1w, chi4w) — they are
  NOT additional free variables (ELV includes them as explicit self-consistency
  checks only to allow the complex-step Jacobian to propagate through them).
"""

import multiprocessing as mp
import threading
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import fsolve

from .constants import *   # noqa: F401,F403  — same pattern as elv.py
from .flash import flash_co2_h2o_salt_ssi
from .scan import _cpu_count


# ── Per-thread call statistics (opt-in instrumentation) ───────────────────────
# Call reset_call_stats() before a flash/stability call and get_call_stats()
# after to collect metrics without affecting any other thread.

_tls = threading.local()


def reset_call_stats():
    """Reset per-thread call statistics.  Call before each flash for instrumentation."""
    s = _tls
    s.n_lnphi_aq        = 0
    s.n_lnphi_c         = 0
    s.n_newton_aq       = 0   # times Newton was attempted
    s.n_newton_aq_ok    = 0   # times Newton converged
    s.n_newton_aq_iters = 0   # total Newton iterations (success + failure)
    s.n_fsolve_aq       = 0   # times fsolve path taken (aq)
    s.n_fsolve_aq_nfev  = 0   # total fsolve function evals (aq, all starts)
    s.n_fsolve_c        = 0   # times fsolve path taken (CO₂-rich)
    s.n_fsolve_c_nfev   = 0   # total fsolve function evals (CO₂-rich, all starts)


def get_call_stats() -> dict:
    """Return current per-thread call statistics as a plain dict."""
    s = _tls
    return {
        "n_lnphi_aq":        getattr(s, "n_lnphi_aq",        0),
        "n_lnphi_c":         getattr(s, "n_lnphi_c",         0),
        "n_newton_aq":       getattr(s, "n_newton_aq",       0),
        "n_newton_aq_ok":    getattr(s, "n_newton_aq_ok",    0),
        "n_newton_aq_iters": getattr(s, "n_newton_aq_iters", 0),
        "n_fsolve_aq":       getattr(s, "n_fsolve_aq",       0),
        "n_fsolve_aq_nfev":  getattr(s, "n_fsolve_aq_nfev",  0),
        "n_fsolve_c":        getattr(s, "n_fsolve_c",        0),
        "n_fsolve_c_nfev":   getattr(s, "n_fsolve_c_nfev",   0),
    }


# ── params-override context (mirrors ELV mechanism) ────────────────────────────

def _apply_params(params: dict | None) -> dict:
    """Inject params into this module's globals, return {key: old_val}."""
    if not params:
        return {}
    saved = {}
    g = globals()
    for k, v in params.items():
        if k in g:
            saved[k] = g[k]
        g[k] = v
    return saved


def _restore_params(saved: dict, params: dict | None) -> None:
    if not params:
        return
    g = globals()
    for k in params:
        if k in saved:
            g[k] = saved[k]
        elif k in g:
            del g[k]


# ── CO₂-rich within-phase lnφ ─────────────────────────────────────────────────

def _eval_c_residual(vars_, x1c, T, P):
    """
    2-variable residual for CO₂-rich within-phase self-consistency.
    vars_ = [Zc, chi1c].  Uses module globals (call inside params override).
    P in bar (converted to Pa internally).
    """
    P_Pa = P * 1e5
    Zc, chi1c = float(vars_[0]), float(vars_[1])
    if Zc <= 0 or chi1c <= 0 or chi1c >= 2.0:
        return [1e6, 1e6]
    x4c = 1.0 - x1c

    k14 = Akij*(T/Tc4)**2 + Bkij*(T/Tc4) + Ckij
    S14 = ASij*(T/Tc4)**2 + BSij*(T/Tc4) + CSij
    a1  = a01*(1 + c11*(1 - (T/Tc1)**0.5))**2
    a4  = a04*(1 + c14*(1 - (T/Tc4)**0.5))**2
    a14 = (a1*a4)**0.5*(1 - k14)
    a   = x1c**2*a1 + 2*x1c*x4c*a14 + x4c**2*a4
    b   = b1*x1c + b4*x4c
    A   = a*P_Pa/R**2/T**2
    B   = b*P_Pa/R/T

    eta     = B / (4*Zc)
    g_eta   = 1.0 / (1.0 - 1.9*eta)
    dg_deta = 1.9 / (1.0 - 1.9*eta)**2
    delta   = g_eta * kappaW * (np.exp(epsW/T) - 1.0)
    DELTA   = delta * P_Pa / R / T

    chi4c     = Zc / (Zc + 2*x1c*chi1c*S14*DELTA)
    chi1c_new = Zc / (Zc + 2*x1c*chi1c*DELTA + 2*x4c*chi4c*S14*DELTA)

    Zphys  = Zc/(Zc-B) - A/(Zc+B)
    Zassoc = -2*(1 + eta/g_eta*dg_deta)*(x1c*(1-chi1c) + x4c*(1-chi4c))
    Zc_new = Zphys + Zassoc

    return [Zc - Zc_new, chi1c - chi1c_new]


def _lnphi_c_inner(x1c: float, T: float, P: float,
                   x0=None) -> tuple[float, float, np.ndarray]:
    """
    Core CO₂-rich lnφ computation (no params handling — call within params context).

    x0 : warm-start guess [Zc, chi1c].  If None, tries multiple starting points
         (both liquid-like and gas-like) and returns the thermodynamically stable
         root (minimum departure Gibbs energy Σᵢ xᵢ lnφᵢ).  Providing x0 keeps
         the solver on the same branch across SSI iterations (warm-starting).

    Returns (lnphi_H2O, lnphi_CO2, sol_array) where sol_array = [Zc, chi1c].
    """
    P_Pa = P * 1e5
    x1c  = float(x1c)
    x4c  = 1.0 - x1c

    k14 = Akij*(T/Tc4)**2 + Bkij*(T/Tc4) + Ckij
    S14 = ASij*(T/Tc4)**2 + BSij*(T/Tc4) + CSij
    a1  = a01*(1 + c11*(1 - (T/Tc1)**0.5))**2
    a4  = a04*(1 + c14*(1 - (T/Tc4)**0.5))**2
    a14 = (a1*a4)**0.5*(1 - k14)
    b   = b1*x1c + b4*x4c
    a   = x1c**2*a1 + 2*x1c*x4c*a14 + x4c**2*a4
    A   = a*P_Pa/R**2/T**2
    B   = b*P_Pa/R/T
    A1  = a1*P_Pa/R**2/T**2; B1 = b1*P_Pa/R/T
    A4  = a4*P_Pa/R**2/T**2; B4 = b4*P_Pa/R/T
    A14 = a14*P_Pa/R**2/T**2

    def _lnphi_from_zc_chi(Zc, chi1c):
        """Compute (lnphi1, lnphi4) from a converged (Zc, chi1c)."""
        eta     = B / (4*Zc)
        g_eta   = 1.0 / (1.0 - 1.9*eta)
        dg_deta = 1.9 / (1.0 - 1.9*eta)**2
        delta   = g_eta * kappaW * (np.exp(epsW/T) - 1.0)
        DELTA   = delta * P_Pa / R / T
        chi4c   = Zc / (Zc + 2*x1c*chi1c*S14*DELTA)

        lp1phys = (-np.log(Zc-B) + B1/B*(B/(Zc-B) - A/(Zc+B))
                   + A/B*(B1/B - 2*(x1c*A1 + x4c*A14)/A)*np.log(1 + B/Zc))
        lp4phys = (-np.log(Zc-B) + B4/B*(B/(Zc-B) - A/(Zc+B))
                   + A/B*(B4/B - 2*(x1c*A14 + x4c*A4)/A)*np.log(1 + B/Zc))
        assoc   = B1/(8*g_eta*Zc)*dg_deta*(x1c*4*(chi1c-1) + x4c*4*(chi4c-1))
        return (lp1phys + 4*np.log(chi1c) + assoc,
                lp4phys + 4*np.log(chi4c) + assoc)

    # Use warm-start if provided; otherwise start from the liquid-like root
    # (covolume estimate Z ≈ b·P/RT).  The liquid root is the physically correct
    # branch for the CO₂-rich reference used in stability analysis — it gives
    # the strong-interaction lnφ values that correctly detect instability.
    # Warm-starting across SSI iterations (or across a P-scan) keeps fsolve on
    # the same branch and prevents discontinuous jumps.
    caller_x0 = x0   # preserve original to detect warm-start vs cold-start
    if x0 is None:
        Z_liq = max(P_Pa * b / R / T + 0.01, 0.05)
        cold_starts = [
            [Z_liq, 0.9],    # liquid-like (dense CO₂-rich, high P)
            [0.9,   0.99],   # gas-like (P < Pc or high T)
            [0.5,   0.95],   # intermediate
        ]
    else:
        cold_starts = [list(x0)]

    # Collect all distinct valid roots, then return the thermodynamically stable
    # one (minimum residual Gibbs energy G^res/RT = Σᵢ xᵢ lnφᵢ).
    # Taking the first valid root without comparing is wrong when two roots exist
    # (e.g. at P < P_sat(H₂O) with H₂O-rich composition): the liquid root is
    # found first but the gas root is the stable state, and using the liquid root
    # for the stability reference while the trial uses the gas root produces a
    # spurious tpd < 0 (root-mixing artefact).
    _s = _tls
    _s.n_lnphi_c = getattr(_s, "n_lnphi_c", 0) + 1
    _nfev_c = 0
    valid_roots = []
    for x0_try in cold_starts:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sol, info, ier, _ = fsolve(
                _eval_c_residual, x0_try, args=(x1c, T, P), full_output=True)
        _nfev_c += info["nfev"]
        Zc_t, chi1c_t = float(sol[0]), float(sol[1])
        if ier != 1 or Zc_t <= 0 or not (0 < chi1c_t < 2.0):
            continue
        # Deduplicate: skip if within 1% of an already-found root
        duplicate = any(abs(Zc_t - Zc_p) / max(abs(Zc_p), 1e-10) < 0.01
                        for Zc_p, _ in valid_roots)
        if not duplicate:
            valid_roots.append((Zc_t, chi1c_t))

    _s.n_fsolve_c     = getattr(_s, "n_fsolve_c",     0) + 1
    _s.n_fsolve_c_nfev = getattr(_s, "n_fsolve_c_nfev", 0) + _nfev_c

    if not valid_roots:
        # All cold starts failed — retry cold if we were warm-started
        if caller_x0 is not None:
            return _lnphi_c_inner(x1c, T, P, x0=None)
        raise RuntimeError(f"ecpa_lnphi_c: no valid root (x1c={x1c:.4f}, P={P:.2f})")

    # Pick the root with the lowest residual Gibbs energy Σᵢ xᵢ lnφᵢ
    best_G = np.inf
    Zc_best = chi1c_best = -1.0
    for Zc_t, chi1c_t in valid_roots:
        lp1_t, lp4_t = _lnphi_from_zc_chi(Zc_t, chi1c_t)
        G_res = x1c * lp1_t + x4c * lp4_t
        if G_res < best_G:
            best_G = G_res
            Zc_best, chi1c_best = Zc_t, chi1c_t

    lp1, lp4 = _lnphi_from_zc_chi(Zc_best, chi1c_best)
    return lp1, lp4, np.array([Zc_best, chi1c_best])


def ecpa_lnphi_c(x1c: float, T: float, P: float,
                 params: dict | None = None) -> tuple[float, float]:
    """
    Compute lnφ for H₂O (component 1) and CO₂ (component 4) in the CO₂-rich phase.

    Parameters
    ----------
    x1c : H₂O mole fraction in the CO₂-rich trial phase  (x4c = 1 - x1c)
    T   : temperature [K]
    P   : pressure [bar]
    params : optional EoS parameter overrides

    Returns
    -------
    (lnphi_H2O, lnphi_CO2)
    """
    saved = _apply_params(params)
    try:
        lnphi1, lnphi4, _ = _lnphi_c_inner(x1c, T, P)
        return lnphi1, lnphi4
    finally:
        _restore_params(saved, params)


# ── Aqueous within-phase lnφ ──────────────────────────────────────────────────


def _eval_aq_all(Zw: float, epsr: float, chi1w: float,
                 x1w: float, ms: float, T: float, P: float):
    """
    Evaluate all aqueous-phase quantities given (Zw, epsr, chi1w).
    Uses module-level globals for EoS constants.
    P in bar (converted to Pa internally).
    Returns (Zw_new, T1, T2, chi1w_new, lnphi1w, lnphi4w).
    """
    P_Pa = P * 1e5
    x2w = x1w * ms * Mw
    x3w = x2w
    x4w = 1.0 - x1w - x2w - x3w

    k14 = Akij*(T/Tc4)**2 + Bkij*(T/Tc4) + Ckij
    S14 = ASij*(T/Tc4)**2 + BSij*(T/Tc4) + CSij
    U4s = Uref4s + alfa4s*R*((1-T/Talfa4s)**2 - (1-Tref/Talfa4s)**2)
    U1s = Uref1s + alfa1s*R*((1-T/Talfa1s)**2 - (1-Tref/Talfa1s)**2)

    rho = P_Pa / Zw / R / T
    a1  = a01*(1 + c11*(1 - (T/Tc1)**0.5))**2
    a4  = a04*(1 + c14*(1 - (T/Tc4)**0.5))**2

    b   = b1*x1w + b2*x2w + b3*x3w + b4*x4w
    U14 = np.log(2)*(a4/b4 - 2*(a1*a4)**0.5*(1-k14)/(b1+b4))
    U41 = np.log(2)*(a1/b1 - 2*(a4*a1)**0.5*(1-k14)/(b4+b1))
    gE  = (1/b)*(x1w*x2w*U1s*(b1+b2) + x1w*x3w*U1s*(b1+b3)
                 + x4w*x2w*U4s*(b4+b2) + x4w*x3w*U4s*(b4+b3)
                 + x1w*x4w*(b1*U14 + b4*U41))
    a   = b*(x1w*a1/b1 + x2w*a02/b2 + x3w*a03/b3 + x4w*a4/b4 - gE/np.log(2))

    A  = a*P_Pa/R**2/T**2;  B  = b*P_Pa/R/T
    A1 = a1*P_Pa/R**2/T**2; B1 = b1*P_Pa/R/T
    B2 = b2*P_Pa/R/T;        B3 = b3*P_Pa/R/T
    A4 = a4*P_Pa/R**2/T**2; B4 = b4*P_Pa/R/T

    Zphys      = Zw/(Zw-B) - A/(Zw+B)
    lnPHI1phys = (-np.log(Zw-B) + B1/B*(B/(Zw-B) - A/(Zw+B))
                  - np.log((Zw+B)/Zw)*(A1/B1
                    - 1/(B*np.log(2))*(x2w*U1s/R/T*(B1+B2)
                                       + x3w*U1s/R/T*(B1+B3)
                                       + x4w/R/T*(B1*U14+B4*U41)
                                       - B1*gE/R/T)))
    lnPHI4phys = (-np.log(Zw-B) + B4/B*(B/(Zw-B) - A/(Zw+B))
                  - np.log((Zw+B)/Zw)*(A4/B4
                    - 1/(B*np.log(2))*(x1w/R/T*(B1*U14+B4*U41)
                                       + x2w*U4s/R/T*(B4+B2)
                                       + x3w*U4s/R/T*(B4+B3)
                                       - B4*gE/R/T)))

    # Association
    eta     = B / (4*Zw)
    g_eta   = 1.0 / (1.0 - 1.9*eta)
    dg_deta = 1.9 / (1.0 - 1.9*eta)**2
    delta   = g_eta * kappaW * (np.exp(epsW/T) - 1.0)
    DELTA   = delta * P_Pa / R / T
    chi4w      = Zw / (Zw + 2*x1w*chi1w*S14*DELTA)
    chi1w_new  = Zw / (Zw + 2*x1w*chi1w*DELTA + 2*x4w*chi4w*S14*DELTA)
    Zassoc     = -2*(1 + eta/g_eta*dg_deta)*(x1w*(1-chi1w) + x4w*(1-chi4w))
    a_sum      = x1w*4*(chi1w-1) + x4w*4*(chi4w-1)
    lnPHI1assoc = 4*np.log(chi1w) + B1/(8*g_eta*Zw)*dg_deta*a_sum
    lnPHI4assoc = 4*np.log(chi4w) + B4/(8*g_eta*Zw)*dg_deta*a_sum

    # Debye–Hückel
    xiZi  = x2w*Z2**2 + x3w*Z2**2
    if ms > 0 and xiZi > 0:
        debye = (e**2*Na*rho*xiZi / (kb*T*epsr*eps0))**0.5
        X2 = 1/Sg2**3*(np.log(1+debye*Sg2) - debye*Sg2 + 0.5*(debye*Sg2)**2)
        X3 = 1/Sg3**3*(np.log(1+debye*Sg3) - debye*Sg3 + 0.5*(debye*Sg3)**2)
        ZDH = (1/(4*np.pi*Na*rho*xiZi)
               * (x2w*Z2**2*(X2 - 0.5*debye**3/(1+debye*Sg2))
                  + x3w*Z3**2*(X3 - 0.5*debye**3/(1+debye*Sg3))))
    else:
        debye = 0.0; ZDH = 0.0

    # Permittivity
    M       = Na*rho/(3*eps0)*(x1w*pol1 + x2w*pol2 + x3w*pol3 + x4w*pol4)
    eps_inf = (2*M + 1)/(1 - M)
    Pww     = 2*rho*x1w*delta*chi1w**2
    Pwc     = 2*rho*x4w*S14*delta*chi1w*chi4w
    Pw      = Pww + Pwc
    gw      = 1 + zww*Pww*np.cos(GAMMA1)/(Pw*np.cos(THETA1) + 1)
    T1      = (2*epsr + eps_inf)*(epsr - eps_inf)/(epsr*(eps_inf + 2)**2)
    T2      = Na*rho/(9*eps0*kb*T)*(x1w*gw*dip01**2)

    # Chi cross-derivatives (linear in chi1w, chi4w → analytical)
    VddeltadV  = -delta*eta*dg_deta/g_eta
    NddeltadN1 =  delta*eta*b1*dg_deta/(g_eta*b)
    NddeltadN4 =  delta*eta*b4*dg_deta/(g_eta*b)
    dFdchiw  = 1 + 2*x1w*rho*delta*chi1w**2
    dFdchic  = 2*x2w*rho*S14*delta*chi4w**2
    dGdchiw  = 2*x1w*rho*S14*delta*chi1w**2
    dFdV     = -(2*rho*x1w*delta*chi1w + 2*rho*x2w*S14*delta*chi4w)*chi1w**2
    dGdV     = -(2*rho*x1w*S14*delta*chi1w)*chi1w**2
    dFdNw    = 2*rho*delta*chi1w**3
    dGdNw    = 2*rho*S14*delta*chi1w**3
    dFdNc    = 2*rho*S14*delta*chi1w**2*chi4w
    dFddelta = -chi1w**2*(1 + (delta-1)*(2*rho*x1w*chi1w + 2*rho*x4w*chi4w*S14))
    dGddelta = -chi4w**2*(1 + (delta-1)*(2*rho*x1w*chi1w*S14))
    det      = dFdchiw - dFdchic*dGdchiw

    Vdchi1WdV  = ((-(dFdV + dFddelta*VddeltadV)
                   + dFdchic*(dGdV + dGddelta*VddeltadV)) / det)
    Vdchi4WdV  = -(dGdchiw*Vdchi1WdV + dGdV + dGddelta*VddeltadV)
    Ndchi1WdNw = ((-(dFdNw + dFddelta*NddeltadN1)
                   + dFdchic*(dGdNw + dGddelta*NddeltadN1)) / det)
    Ndchi4WdNw = -(dGdchiw*Ndchi1WdNw + dGdNw + dGddelta*NddeltadN1)
    Ndchi1WdNc = ((-(dFdNc + dFddelta*NddeltadN4)
                   + dFdchic*(-dGddelta*NddeltadN4)) / det)  # dGdNc=0
    Ndchi4WdNc = -(dGdchiw*Ndchi1WdNc + dGddelta*NddeltadN4)

    # Permittivity lnφ contributions
    dgwdPw   = -(gw-1)*np.cos(THETA1)/(Pw*np.cos(THETA1) + 1)
    dgwdPww  =  (gw-1)/Pww if Pww > 1e-30 else 0.0
    NdPwwdN1 = Pww*(1/x1w + 1/delta*NddeltadN1 + 2/chi1w*Ndchi1WdNw)
    NdPwcdN1 = Pwc*(1/chi1w*Ndchi1WdNw + 1/delta*NddeltadN1 + 1/chi4w*Ndchi4WdNw)
    NdPwwdN4 = Pww*(2/chi1w*Ndchi1WdNc + 1/delta*NddeltadN4)
    NdPwcdN4 = Pwc*(1/x4w*float(x4w>1e-30) + 1/chi1w*Ndchi1WdNc
                    + 1/delta*NddeltadN4 + 1/chi4w*Ndchi4WdNc)
    NdgwdN1  = dgwdPw*(NdPwwdN1+NdPwcdN1) + dgwdPww*NdPwwdN1
    NdgwdN4  = dgwdPw*(NdPwwdN4+NdPwcdN4) + dgwdPww*NdPwwdN4

    dFder    = (2*epsr**2 + eps_inf**2)/(epsr**2*(eps_inf+2)**2)
    dFdeinf  = (epsr*eps_inf - 4*eps_inf - 4*epsr**2 - 2*epsr)/(epsr*(eps_inf+2)**3)
    NdeinfdN1 = Na*pol1*rho/(eps0*(1-M)**2)
    NdeinfdN4 = Na*pol4*rho/(eps0*(1-M)**2)
    NdFdN1   = -Na*rho*dip01**2/(9*eps0*kb*T)*(gw + x1w*NdgwdN1)
    NdFdN4   = -Na*rho*dip01**2/(9*eps0*kb*T)*(x1w*NdgwdN4)
    NderdN1  = -(dFder)**-1*(dFdeinf*NdeinfdN1 + NdFdN1)
    NderdN4  = -(dFder)**-1*(dFdeinf*NdeinfdN4 + NdFdN4)

    VdeinfdV = -3*M/(1-M)**2
    VdPwwdV  = Pww*(-1 + 1/delta*VddeltadV + 2/chi1w*Vdchi1WdV)
    VdPwcdV  = Pwc*(-1 + 1/delta*VddeltadV + 1/chi1w*Vdchi1WdV + 1/chi4w*Vdchi4WdV)
    VdgwdV   = dgwdPw*(VdPwwdV+VdPwcdV) + dgwdPww*VdPwwdV
    VdFdV    = Na*rho*x1w*dip01**2/(9*eps0*kb*T)*(gw - VdgwdV)
    VderdV   = -(dFder)**-1*(dFdeinf*VdeinfdV + VdFdV)

    if ms > 0:
        daDHBder = (debye**2/(8*np.pi*Na*rho*epsr*xiZi)
                    * (x2w*Z2**2*(debye/(1+debye*Sg2) - 1/Rb2)
                       + x3w*Z3**2*(debye/(1+debye*Sg3) - 1/Rb3)))
    else:
        daDHBder = 0.0

    ZPerm       = -daDHBder * VderdV
    lnPHI1perm  =  daDHBder * NderdN1
    lnPHI4perm  =  daDHBder * NderdN4

    Zw_new  = Zphys + Zassoc + ZDH + ZPerm
    lnphi1w = lnPHI1phys + lnPHI1assoc + lnPHI1perm
    lnphi4w = lnPHI4phys + lnPHI4assoc + lnPHI4perm

    return Zw_new, T1, T2, chi1w_new, lnphi1w, lnphi4w


def _eval_aq_all_with_jac(Zw: float, epsr: float, chi1w: float,
                           x1w: float, ms: float, T: float, P: float):
    """
    Forward pass identical to _eval_aq_all, plus the analytical 3×3 Jacobian.

    J[i, j] = ∂F_i / ∂v_j  where
        F = [Zw − Zw_new,  T1 − T2,  chi1w − chi1w_new]
        v = [Zw,           epsr,      chi1w            ]

    Returns (Zw_new, T1, T2, chi1w_new, lnphi1w, lnphi4w, J).

    Approximation: J[0, 1] (∂F0/∂epsr) and J[0, 2] (∂F0/∂chi1w) are exact.
    J[0, 0] (∂F0/∂Zw) omits the d0_VdFdV term in d0_VderdV (would require
    differentiating the chi cross-derivative system w.r.t. Zw); the residual
    error is ~1–2 % at typical conditions and does not impair Newton
    convergence from warm starts.
    """
    # ── Forward pass (identical to _eval_aq_all) ─────────────────────────────
    P_Pa = P * 1e5
    x2w = x1w * ms * Mw
    x3w = x2w
    x4w = 1.0 - x1w - x2w - x3w

    k14 = Akij*(T/Tc4)**2 + Bkij*(T/Tc4) + Ckij
    S14 = ASij*(T/Tc4)**2 + BSij*(T/Tc4) + CSij
    U4s = Uref4s + alfa4s*R*((1-T/Talfa4s)**2 - (1-Tref/Talfa4s)**2)
    U1s = Uref1s + alfa1s*R*((1-T/Talfa1s)**2 - (1-Tref/Talfa1s)**2)

    rho = P_Pa / Zw / R / T
    a1  = a01*(1 + c11*(1 - (T/Tc1)**0.5))**2
    a4  = a04*(1 + c14*(1 - (T/Tc4)**0.5))**2

    b   = b1*x1w + b2*x2w + b3*x3w + b4*x4w
    U14 = np.log(2)*(a4/b4 - 2*(a1*a4)**0.5*(1-k14)/(b1+b4))
    U41 = np.log(2)*(a1/b1 - 2*(a4*a1)**0.5*(1-k14)/(b4+b1))
    gE  = (1/b)*(x1w*x2w*U1s*(b1+b2) + x1w*x3w*U1s*(b1+b3)
                 + x4w*x2w*U4s*(b4+b2) + x4w*x3w*U4s*(b4+b3)
                 + x1w*x4w*(b1*U14 + b4*U41))
    a   = b*(x1w*a1/b1 + x2w*a02/b2 + x3w*a03/b3 + x4w*a4/b4 - gE/np.log(2))

    A  = a*P_Pa/R**2/T**2;  B  = b*P_Pa/R/T
    A1 = a1*P_Pa/R**2/T**2; B1 = b1*P_Pa/R/T
    B2 = b2*P_Pa/R/T;        B3 = b3*P_Pa/R/T
    A4 = a4*P_Pa/R**2/T**2; B4 = b4*P_Pa/R/T

    Zphys      = Zw/(Zw-B) - A/(Zw+B)
    lnPHI1phys = (-np.log(Zw-B) + B1/B*(B/(Zw-B) - A/(Zw+B))
                  - np.log((Zw+B)/Zw)*(A1/B1
                    - 1/(B*np.log(2))*(x2w*U1s/R/T*(B1+B2)
                                       + x3w*U1s/R/T*(B1+B3)
                                       + x4w/R/T*(B1*U14+B4*U41)
                                       - B1*gE/R/T)))
    lnPHI4phys = (-np.log(Zw-B) + B4/B*(B/(Zw-B) - A/(Zw+B))
                  - np.log((Zw+B)/Zw)*(A4/B4
                    - 1/(B*np.log(2))*(x1w/R/T*(B1*U14+B4*U41)
                                       + x2w*U4s/R/T*(B4+B2)
                                       + x3w*U4s/R/T*(B4+B3)
                                       - B4*gE/R/T)))

    eta     = B / (4*Zw)
    g_eta   = 1.0 / (1.0 - 1.9*eta)
    dg_deta = 1.9 / (1.0 - 1.9*eta)**2
    delta   = g_eta * kappaW * (np.exp(epsW/T) - 1.0)
    DELTA   = delta * P_Pa / R / T
    chi4w      = Zw / (Zw + 2*x1w*chi1w*S14*DELTA)
    chi1w_new  = Zw / (Zw + 2*x1w*chi1w*DELTA + 2*x4w*chi4w*S14*DELTA)
    Zassoc     = -2*(1 + eta/g_eta*dg_deta)*(x1w*(1-chi1w) + x4w*(1-chi4w))
    a_sum      = x1w*4*(chi1w-1) + x4w*4*(chi4w-1)
    lnPHI1assoc = 4*np.log(chi1w) + B1/(8*g_eta*Zw)*dg_deta*a_sum
    lnPHI4assoc = 4*np.log(chi4w) + B4/(8*g_eta*Zw)*dg_deta*a_sum

    xiZi  = x2w*Z2**2 + x3w*Z2**2
    if ms > 0 and xiZi > 0:
        debye = (e**2*Na*rho*xiZi / (kb*T*epsr*eps0))**0.5
        X2 = 1/Sg2**3*(np.log(1+debye*Sg2) - debye*Sg2 + 0.5*(debye*Sg2)**2)
        X3 = 1/Sg3**3*(np.log(1+debye*Sg3) - debye*Sg3 + 0.5*(debye*Sg3)**2)
        ZDH = (1/(4*np.pi*Na*rho*xiZi)
               * (x2w*Z2**2*(X2 - 0.5*debye**3/(1+debye*Sg2))
                  + x3w*Z3**2*(X3 - 0.5*debye**3/(1+debye*Sg3))))
    else:
        debye = 0.0; ZDH = 0.0

    M       = Na*rho/(3*eps0)*(x1w*pol1 + x2w*pol2 + x3w*pol3 + x4w*pol4)
    eps_inf = (2*M + 1)/(1 - M)
    Pww     = 2*rho*x1w*delta*chi1w**2
    Pwc     = 2*rho*x4w*S14*delta*chi1w*chi4w
    Pw      = Pww + Pwc
    gw      = 1 + zww*Pww*np.cos(GAMMA1)/(Pw*np.cos(THETA1) + 1)
    T1      = (2*epsr + eps_inf)*(epsr - eps_inf)/(epsr*(eps_inf + 2)**2)
    T2      = Na*rho/(9*eps0*kb*T)*(x1w*gw*dip01**2)

    VddeltadV  = -delta*eta*dg_deta/g_eta
    NddeltadN1 =  delta*eta*b1*dg_deta/(g_eta*b)
    NddeltadN4 =  delta*eta*b4*dg_deta/(g_eta*b)
    dFdchiw  = 1 + 2*x1w*rho*delta*chi1w**2
    dFdchic  = 2*x2w*rho*S14*delta*chi4w**2
    dGdchiw  = 2*x1w*rho*S14*delta*chi1w**2
    dFdV     = -(2*rho*x1w*delta*chi1w + 2*rho*x2w*S14*delta*chi4w)*chi1w**2
    dGdV     = -(2*rho*x1w*S14*delta*chi1w)*chi1w**2
    dFdNw    = 2*rho*delta*chi1w**3
    dGdNw    = 2*rho*S14*delta*chi1w**3
    dFdNc    = 2*rho*S14*delta*chi1w**2*chi4w
    dFddelta = -chi1w**2*(1 + (delta-1)*(2*rho*x1w*chi1w + 2*rho*x4w*chi4w*S14))
    dGddelta = -chi4w**2*(1 + (delta-1)*(2*rho*x1w*chi1w*S14))
    det      = dFdchiw - dFdchic*dGdchiw

    Vdchi1WdV  = ((-(dFdV + dFddelta*VddeltadV)
                   + dFdchic*(dGdV + dGddelta*VddeltadV)) / det)
    Vdchi4WdV  = -(dGdchiw*Vdchi1WdV + dGdV + dGddelta*VddeltadV)
    Ndchi1WdNw = ((-(dFdNw + dFddelta*NddeltadN1)
                   + dFdchic*(dGdNw + dGddelta*NddeltadN1)) / det)
    Ndchi4WdNw = -(dGdchiw*Ndchi1WdNw + dGdNw + dGddelta*NddeltadN1)
    Ndchi1WdNc = ((-(dFdNc + dFddelta*NddeltadN4)
                   + dFdchic*(-dGddelta*NddeltadN4)) / det)  # dGdNc=0
    Ndchi4WdNc = -(dGdchiw*Ndchi1WdNc + dGddelta*NddeltadN4)

    dgwdPw   = -(gw-1)*np.cos(THETA1)/(Pw*np.cos(THETA1) + 1)
    dgwdPww  =  (gw-1)/Pww if Pww > 1e-30 else 0.0
    NdPwwdN1 = Pww*(1/x1w + 1/delta*NddeltadN1 + 2/chi1w*Ndchi1WdNw)
    NdPwcdN1 = Pwc*(1/chi1w*Ndchi1WdNw + 1/delta*NddeltadN1 + 1/chi4w*Ndchi4WdNw)
    NdPwwdN4 = Pww*(2/chi1w*Ndchi1WdNc + 1/delta*NddeltadN4)
    NdPwcdN4 = Pwc*(1/x4w*float(x4w>1e-30) + 1/chi1w*Ndchi1WdNc
                    + 1/delta*NddeltadN4 + 1/chi4w*Ndchi4WdNc)
    NdgwdN1  = dgwdPw*(NdPwwdN1+NdPwcdN1) + dgwdPww*NdPwwdN1
    NdgwdN4  = dgwdPw*(NdPwwdN4+NdPwcdN4) + dgwdPww*NdPwwdN4

    dFder    = (2*epsr**2 + eps_inf**2)/(epsr**2*(eps_inf+2)**2)
    dFdeinf  = (epsr*eps_inf - 4*eps_inf - 4*epsr**2 - 2*epsr)/(epsr*(eps_inf+2)**3)
    NdeinfdN1 = Na*pol1*rho/(eps0*(1-M)**2)
    NdeinfdN4 = Na*pol4*rho/(eps0*(1-M)**2)
    NdFdN1   = -Na*rho*dip01**2/(9*eps0*kb*T)*(gw + x1w*NdgwdN1)
    NdFdN4   = -Na*rho*dip01**2/(9*eps0*kb*T)*(x1w*NdgwdN4)
    NderdN1  = -(dFder)**-1*(dFdeinf*NdeinfdN1 + NdFdN1)
    NderdN4  = -(dFder)**-1*(dFdeinf*NdeinfdN4 + NdFdN4)

    VdeinfdV = -3*M/(1-M)**2
    VdPwwdV  = Pww*(-1 + 1/delta*VddeltadV + 2/chi1w*Vdchi1WdV)
    VdPwcdV  = Pwc*(-1 + 1/delta*VddeltadV + 1/chi1w*Vdchi1WdV + 1/chi4w*Vdchi4WdV)
    VdgwdV   = dgwdPw*(VdPwwdV+VdPwcdV) + dgwdPww*VdPwwdV
    VdFdV    = Na*rho*x1w*dip01**2/(9*eps0*kb*T)*(gw - VdgwdV)
    VderdV   = -(dFder)**-1*(dFdeinf*VdeinfdV + VdFdV)

    if ms > 0:
        daDHBder = (debye**2/(8*np.pi*Na*rho*epsr*xiZi)
                    * (x2w*Z2**2*(debye/(1+debye*Sg2) - 1/Rb2)
                       + x3w*Z3**2*(debye/(1+debye*Sg3) - 1/Rb3)))
    else:
        daDHBder = 0.0

    ZPerm       = -daDHBder * VderdV
    lnPHI1perm  =  daDHBder * NderdN1
    lnPHI4perm  =  daDHBder * NderdN4

    Zw_new  = Zphys + Zassoc + ZDH + ZPerm
    lnphi1w = lnPHI1phys + lnPHI1assoc + lnPHI1perm
    lnphi4w = lnPHI4phys + lnPHI4assoc + lnPHI4perm

    # ── Analytical Jacobian ───────────────────────────────────────────────────
    J = np.zeros((3, 3))

    # Shared first-order derivatives w.r.t. Zw
    # (all at fixed epsr, chi1w — Newton independent variables)
    d0_rho   = -rho / Zw                               # ∂ρ/∂Zw
    d0_eta   = -eta / Zw                               # ∂η/∂Zw
    # ∂δ/∂Zw  (= VddeltadV/Zw, with V∝Zw at fixed T,P)
    d0_delta = VddeltadV / Zw
    d0_DELTA = d0_delta * P_Pa / (R * T)               # ∂Δ/∂Zw
    d0_eps_inf = VdeinfdV / Zw                         # ∂ε_inf/∂Zw
    C2 = Na*x1w*dip01**2 / (9*eps0*kb*T)              # T2 prefactor (const)

    # ── J[2, :] — F2 = chi1w − chi1w_new ────────────────────────────────────
    # chi4w = Zw / Den4
    Den4     = Zw + 2*x1w*chi1w*S14*DELTA
    d0_Den4  = 1.0 + 2*x1w*chi1w*S14*d0_DELTA
    d2_Den4  = 2*x1w*S14*DELTA                          # ∂Den4/∂chi1w
    d0_chi4w = (Den4 - Zw*d0_Den4) / Den4**2
    d2_chi4w = -Zw*d2_Den4 / Den4**2

    # chi1w_new = Zw / Den1
    Den1      = Zw + 2*x1w*chi1w*DELTA + 2*x4w*chi4w*S14*DELTA
    d0_Den1   = (1.0 + 2*x1w*chi1w*d0_DELTA
                 + 2*x4w*S14*(d0_chi4w*DELTA + chi4w*d0_DELTA))
    d2_Den1   = 2*x1w*DELTA + 2*x4w*S14*d2_chi4w*DELTA
    d0_chi1w_new = (Den1 - Zw*d0_Den1) / Den1**2
    d2_chi1w_new = -Zw*d2_Den1 / Den1**2

    J[2, 0] = -d0_chi1w_new
    J[2, 1] = 0.0
    J[2, 2] = 1.0 - d2_chi1w_new

    # ── Shared Pw/gw derivatives (used by both J[0,:] and J[1,:]) ────────────
    # ∂Pww/∂Zw at fixed chi1w (no Vdchi1WdV — chi1w is independent here)
    d0_Pww = -Pww/Zw * (1.0 + dg_deta*eta/g_eta)
    # ∂Pwc/∂Zw: ρ·δ·chi1w·chi4w, with chi4w also varying with Zw
    d0_Pwc = (Pwc*(-1.0/Zw - dg_deta*eta/(g_eta*Zw))
              + 2*rho*x4w*S14*delta*chi1w*d0_chi4w)
    d0_gw = dgwdPww*d0_Pww + dgwdPw*(d0_Pww + d0_Pwc)
    # ∂Pww/∂chi1w and ∂Pwc/∂chi1w (at fixed Zw; chi4w depends on chi1w)
    d2_Pww = 2*Pww / chi1w
    d2_Pwc = Pwc/chi1w + 2*rho*x4w*S14*delta*chi1w*d2_chi4w
    d2_gw = dgwdPww*d2_Pww + dgwdPw*(d2_Pww + d2_Pwc)

    # ── J[0, :] — F0 = Zw − Zw_new ──────────────────────────────────────────
    # Zphys = Zw/(Zw-B) - A/(Zw+B)
    d0_Zphys = -B/(Zw-B)**2 + A/(Zw+B)**2

    # Zassoc = -2*k_assoc*s_assoc
    k_assoc  = 1.0 + eta*dg_deta/g_eta
    s_assoc  = x1w*(1-chi1w) + x4w*(1-chi4w)
    d0_k_assoc = -1.9*eta*(g_eta + eta*dg_deta) / Zw    # ∂k_assoc/∂Zw
    d0_s_assoc = -x4w*d0_chi4w
    d2_s_assoc = -x1w - x4w*d2_chi4w
    d0_Zassoc = -2.0*(d0_k_assoc*s_assoc + k_assoc*d0_s_assoc)
    d2_Zassoc = -2.0*k_assoc*d2_s_assoc

    if ms > 0 and debye > 0:
        d0_debye_v = -debye / (2*Zw)
        d1_debye_v = -debye / (2*epsr)

        # df_DH/ddebye: analytic differentiation of X2-Y2 and X3-Y3
        dX2_dd = (1.0/(1+debye*Sg2) - 1.0 + debye*Sg2) / Sg2**2
        dX3_dd = (1.0/(1+debye*Sg3) - 1.0 + debye*Sg3) / Sg3**2
        dY2_dd = 0.5*debye**2*(3 + 2*debye*Sg2) / (1+debye*Sg2)**2
        dY3_dd = 0.5*debye**2*(3 + 2*debye*Sg3) / (1+debye*Sg3)**2
        df_dd  = (x2w*Z2**2*(dX2_dd - dY2_dd) + x3w*Z3**2*(dX3_dd - dY3_dd))

        norm4pi = 4*np.pi*Na*rho*xiZi
        # d0_ZDH: df/dd * d0_debye / (4πNaρξ) + ZDH/Zw  (1/ρ factor picks up d(1/Zw))
        d0_ZDH = df_dd*d0_debye_v / norm4pi + ZDH/Zw
        d1_ZDH = df_dd*d1_debye_v / norm4pi

        # ── ZPerm Jacobian ────────────────────────────────────────────────────
        # ZPerm = −daDHBder · VderdV
        # daDHBder depends on (Zw, epsr) through debye; VderdV depends on
        # (Zw, epsr, chi1w) through dFder, dFdeinf, VdFdV.
        H_DH = (x2w*Z2**2*(debye/(1+debye*Sg2) - 1/Rb2)
                + x3w*Z3**2*(debye/(1+debye*Sg3) - 1/Rb3))
        dH_dd = (x2w*Z2**2/(1+debye*Sg2)**2
                 + x3w*Z3**2/(1+debye*Sg3)**2)
        if abs(H_DH) > 1e-60:
            d0_daDHBder = daDHBder/H_DH * dH_dd * d0_debye_v
            d1_daDHBder = (daDHBder/H_DH * dH_dd * d1_debye_v
                           - 2*daDHBder/epsr)
        else:
            d0_daDHBder = 0.0
            d1_daDHBder = 0.0

        # ∂VderdV/∂epsr: VderdV = −(dFder)⁻¹·(dFdeinf·VdeinfdV + VdFdV);
        # dFder and dFdeinf both depend on epsr.
        d1_dFder   = -2*eps_inf**2 / (epsr**3*(eps_inf+2)**2)
        d1_dFdeinf = 4*(eps_inf - epsr**2) / (epsr**2*(eps_inf+2)**3)
        d1_VderdV  = (-VderdV * d1_dFder / dFder
                      - d1_dFdeinf * VdeinfdV / dFder)
        d1_ZPerm   = -(VderdV*d1_daDHBder + daDHBder*d1_VderdV)

        # ∂VderdV/∂Zw: ε_inf(Zw) variation in dFder, dFdeinf, VdeinfdV captured
        # below.  The d0_VdFdV term (chain through chi cross-deriv w.r.t. Zw) is
        # omitted; residual ≈ 1–2 % in J[0,0] — sufficient for Newton.
        ddFder_deinf   = 4*(eps_inf - epsr**2) / (epsr**2*(eps_inf+2)**3)
        ddFdeinf_deinf = ((12*epsr**2 - 2*epsr*eps_inf + 8*epsr + 8*eps_inf - 8)
                          / (epsr*(eps_inf+2)**4))
        d0_VdeinfdV = 3*M*(1+M) / (Zw*(1-M)**3)
        d0_VderdV_eps = ((-VderdV/dFder) * ddFder_deinf * d0_eps_inf
                         - (ddFdeinf_deinf * d0_eps_inf * VdeinfdV
                            + dFdeinf * d0_VdeinfdV) / dFder)
        d0_ZPerm = -(VderdV*d0_daDHBder + daDHBder*d0_VderdV_eps)

        # ── ∂ZPerm/∂chi1w: exact via chi cross-derivative differentiation ────
        # Differentiate M_cd·[Vdchi1WdV, Vdchi4WdV]=RHS_V w.r.t. chi1w.
        # VddeltadV does not depend on chi1w.
        d2_dFdchiw_xd = 4*x1w*rho*delta*chi1w
        d2_dFdchic_xd = 4*x2w*rho*S14*delta*chi4w*d2_chi4w
        d2_dGdchiw_xd = 4*x1w*rho*S14*delta*chi1w
        d2_dFdV_xd    = (-6*rho*x1w*delta*chi1w**2
                          - 2*rho*x2w*S14*delta*(d2_chi4w*chi1w**2 + 2*chi4w*chi1w))
        d2_dGdV_xd    = -6*rho*x1w*S14*delta*chi1w**2
        P_F_xd = 2*rho*x1w*chi1w + 2*rho*x4w*chi4w*S14
        P_G_xd = 2*rho*x1w*chi1w*S14
        d2_P_F_xd = 2*rho*x1w + 2*rho*x4w*S14*d2_chi4w
        d2_dFddelta_xd = (-2*chi1w*(1 + (delta-1)*P_F_xd)
                           - chi1w**2*(delta-1)*d2_P_F_xd)
        d2_dGddelta_xd = (-2*chi4w*d2_chi4w*(1 + (delta-1)*P_G_xd)
                           - chi4w**2*(delta-1)*2*rho*x1w*S14)
        new_RHS_F = (-(d2_dFdV_xd + d2_dFddelta_xd*VddeltadV)
                     - d2_dFdchiw_xd*Vdchi1WdV - d2_dFdchic_xd*Vdchi4WdV)
        new_RHS_G = (-(d2_dGdV_xd + d2_dGddelta_xd*VddeltadV)
                     - d2_dGdchiw_xd*Vdchi1WdV)
        d2_Vdchi1WdV = (new_RHS_F - dFdchic*new_RHS_G) / det
        d2_Vdchi4WdV = -(dGdchiw*d2_Vdchi1WdV + new_RHS_G)
        # Propagate to VdPwwdV, VdPwcdV, VdgwdV, VdFdV
        C_pw = -1 + VddeltadV/delta + 2*Vdchi1WdV/chi1w
        C_pc = (-1 + VddeltadV/delta + Vdchi1WdV/chi1w
                + (Vdchi4WdV/chi4w if chi4w > 1e-30 else 0.0))
        d2_VdPwwdV = (d2_Pww*C_pw
                      + Pww*2*(chi1w*d2_Vdchi1WdV - Vdchi1WdV)/chi1w**2)
        if chi4w > 1e-30:
            d2_VdPwcdV = (d2_Pwc*C_pc
                          + Pwc*((chi1w*d2_Vdchi1WdV - Vdchi1WdV)/chi1w**2
                                 + (chi4w*d2_Vdchi4WdV
                                    - Vdchi4WdV*d2_chi4w)/chi4w**2))
        else:
            d2_VdPwcdV = 0.0
        denom_gw   = np.cos(THETA1)*Pw + 1
        d2_dgwdPw  = (-np.cos(THETA1)*d2_gw/denom_gw
                      + (gw-1)*np.cos(THETA1)**2*(d2_Pww+d2_Pwc)/denom_gw**2)
        d2_dgwdPww = ((d2_gw*Pww - (gw-1)*d2_Pww)/Pww**2
                      if Pww > 1e-30 else 0.0)
        d2_VdgwdV  = (d2_dgwdPww*VdPwwdV + dgwdPww*d2_VdPwwdV
                      + d2_dgwdPw*(VdPwwdV+VdPwcdV)
                      + dgwdPw*(d2_VdPwwdV+d2_VdPwcdV))
        d2_VdFdV   = C2*rho*(d2_gw - d2_VdgwdV)
        d2_ZPerm   = daDHBder * d2_VdFdV / dFder   # = -daDHBder*(-(1/dFder)*d2_VdFdV)
    else:
        d0_ZDH = d1_ZDH = d0_ZPerm = d1_ZPerm = d2_ZPerm = 0.0

    J[0, 0] = 1.0 - (d0_Zphys + d0_Zassoc + d0_ZDH + d0_ZPerm)
    J[0, 1] = -(d1_ZDH + d1_ZPerm)
    J[0, 2] = -(d2_Zassoc + d2_ZPerm)

    # ── J[1, :] — F1 = T1 − T2 ──────────────────────────────────────────────
    # T1 depends on epsr (direct) and eps_inf (through rho → Zw)
    d0_T1 = dFdeinf * d0_eps_inf
    # d1_T1 = dFder (∂T1/∂ε_r at fixed ε_inf)

    # T2 = C2 * rho * gw; gw derivatives computed in shared section above.
    d0_T2 = C2*(d0_rho*gw + rho*d0_gw)
    d2_T2 = C2*rho*d2_gw

    J[1, 0] = d0_T1 - d0_T2
    J[1, 1] = dFder                # ∂T1/∂ε_r; T2 has no direct ε_r dependence
    J[1, 2] = -d2_T2

    return Zw_new, T1, T2, chi1w_new, lnphi1w, lnphi4w, J


def _newton_aq(v0, x1w, ms, T, P, tol=1e-10, maxiter=20):
    """Newton iteration for the aqueous inner solve using analytical Jacobian.

    Solves F(Zw, epsr, chi1w) = 0 where
        F = [Zw - Zw_new(Zw, epsr, chi1w),
             T1(epsr)  - T2(Zw, chi1w),
             chi1w     - chi1w_new(Zw, chi1w)].

    Uses the analytical 3×3 Jacobian from _eval_aq_all_with_jac (1 eval/step).
    Returns np.array([Zw, epsr, chi1w]) on convergence, None otherwise.
    """
    _s = _tls
    _s.n_newton_aq = getattr(_s, "n_newton_aq", 0) + 1
    v = np.array(v0, dtype=float)

    for _iter in range(maxiter):
        Zw, epsr, chi1w = v[0], v[1], v[2]
        if Zw <= 0 or epsr <= 1 or chi1w <= 0 or chi1w >= 2.0:
            _s.n_newton_aq_iters = getattr(_s, "n_newton_aq_iters", 0) + _iter
            return None
        chi1w_eval = min(chi1w, 1.0 - 1e-12)
        Zw_new, T1, T2, chi1w_new, _, _, J = _eval_aq_all_with_jac(
            Zw, epsr, chi1w_eval, x1w, ms, T, P)
        F = np.array([Zw - Zw_new, T1 - T2, chi1w - chi1w_new])
        if np.max(np.abs(F)) < tol:
            _s.n_newton_aq_ok    = getattr(_s, "n_newton_aq_ok",    0) + 1
            _s.n_newton_aq_iters = getattr(_s, "n_newton_aq_iters", 0) + _iter + 1
            return v
        try:
            dv = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            _s.n_newton_aq_iters = getattr(_s, "n_newton_aq_iters", 0) + _iter + 1
            return None
        # Backtracking line search to stay in physical domain
        alpha = 1.0
        for _ls in range(6):
            vt = v + alpha * dv
            if vt[0] > 0 and vt[1] > 1 and 0 < vt[2] < 2.0:
                break
            alpha *= 0.5
        else:
            _s.n_newton_aq_iters = getattr(_s, "n_newton_aq_iters", 0) + _iter + 1
            return None
        v = vt
    _s.n_newton_aq_iters = getattr(_s, "n_newton_aq_iters", 0) + maxiter
    return None  # maxiter reached without convergence


def _lnphi_aq_inner(x1w: float, ms: float, T: float, P: float,
                    x0=None) -> tuple[float, float, np.ndarray]:
    """
    Core aqueous lnφ computation (no params handling — call within params context).

    x0 : warm-start guess [Zw, epsr, chi1w]; if None, uses default heuristic.

    Returns (lnphi_H2O, lnphi_CO2, sol_array) where sol_array = [Zw, epsr, chi1w].
    """
    _s = _tls
    _s.n_lnphi_aq = getattr(_s, "n_lnphi_aq", 0) + 1
    x1w = float(x1w)

    x2w = x1w * ms * Mw
    b_est = b1*x1w + b2*x2w + b3*x2w + b4*(1 - x1w - 2*x2w)

    if x0 is None:
        # b*P/RT is the liquid co-volume estimate — scales correctly with P.
        # The old floor of 0.02 caused Zw0 to be 3–4× too large at P < ~37 bar,
        # placing the start outside the basin of attraction for the liquid root.
        Zw0 = max(b_est * (P * 1e5) / R / T * 1.2, 1e-4)
        starts = [
            [Zw0,       60.0, 0.4],   # dense/liquid-like
            [Zw0 * 1.3, 70.0, 0.6],   # slightly higher Z / epsr
            [Zw0,       60.0, 0.99],  # near-unity chi1w (low-P limit)
            [Zw0,       50.0, 0.2],   # low chi1w (strongly bonded, T > ~300 K)
            [Zw0,       50.0, 0.1],   # very low chi1w (high-T / high-P liquid)
        ]
    else:
        # Warm start: try Newton first (fast when close to solution).
        v_newton = _newton_aq(x0, x1w, ms, T, P)
        if v_newton is not None:
            Zw, epsr, chi1w = v_newton
            chi1w_eval = min(chi1w, 1.0 - 1e-12)
            _, _, _, _, lnphi1w, lnphi4w = _eval_aq_all(
                Zw, epsr, chi1w_eval, x1w, ms, T, P)
            return lnphi1w, lnphi4w, np.array([Zw, epsr, chi1w])
        # Newton failed — fall back to fsolve warm-started at x0
        starts = [list(x0)]

    def residual(v):
        Zw, epsr, chi1w = float(v[0]), float(v[1]), float(v[2])
        if Zw <= 0 or epsr <= 1 or chi1w <= 0 or chi1w >= 2.0:
            return [1e6, 1e6, 1e6]
        # At low P, chi1w → 1 (all H₂O associated).  Clamp for physics so
        # fsolve's finite-difference Jacobian stays smooth across chi1w = 1.
        chi1w_eval = min(chi1w, 1.0 - 1e-12)
        Zw_new, T1, T2, chi1w_new, _, _ = _eval_aq_all(
            Zw, epsr, chi1w_eval, x1w, ms, T, P)
        return [Zw - Zw_new, T1 - T2, chi1w - chi1w_new]

    best_sol = None
    _nfev_aq = 0
    for start in starts:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sol, info, ier, _ = fsolve(residual, start, full_output=True)
        _nfev_aq += info["nfev"]
        Zw_s, epsr_s, chi1w_s = float(sol[0]), float(sol[1]), float(sol[2])
        if ier == 1 and Zw_s > 0 and epsr_s > 1 and 0 < chi1w_s < 2.0:
            best_sol = sol
            break   # aqueous EOS has one physical root; first success is fine

    _s.n_fsolve_aq     = getattr(_s, "n_fsolve_aq",     0) + 1
    _s.n_fsolve_aq_nfev = getattr(_s, "n_fsolve_aq_nfev", 0) + _nfev_aq

    if best_sol is None:
        if x0 is not None:
            return _lnphi_aq_inner(x1w, ms, T, P, x0=None)
        raise RuntimeError(
            f"ecpa_lnphi_aq: fsolve did not converge "
            f"(x1w={x1w:.4f}, ms={ms:.3f}, T={T:.1f}, P={P:.2f})")

    Zw, epsr, chi1w = float(best_sol[0]), float(best_sol[1]), float(best_sol[2])
    chi1w_eval = min(chi1w, 1.0 - 1e-12)
    _, _, _, _, lnphi1w, lnphi4w = _eval_aq_all(Zw, epsr, chi1w_eval, x1w, ms, T, P)
    return lnphi1w, lnphi4w, np.array([Zw, epsr, chi1w])


def ecpa_lnphi_aq(x1w: float, ms: float, T: float, P: float,
                  params: dict | None = None,
                  x0: np.ndarray | None = None) -> tuple[float, float]:
    """
    Compute lnφ for H₂O and CO₂ in the aqueous phase via within-phase fsolve.

    Parameters
    ----------
    x1w  : H₂O mole fraction (x2w=x3w=x1w*ms*Mw, x4w=1-x1w-2*x2w)
    ms   : salt molality [mol/kg H₂O] — fixes x2w, x3w relative to x1w
    T    : temperature [K]
    P    : pressure [bar]
    params : EoS parameter overrides
    x0   : initial guess [Zw, epsr, chi1w]; if None, uses defaults

    Returns
    -------
    (lnphi_H2O, lnphi_CO2)
    """
    saved = _apply_params(params)
    try:
        lnphi1w, lnphi4w, _ = _lnphi_aq_inner(x1w, ms, T, P, x0=x0)
        return lnphi1w, lnphi4w
    finally:
        _restore_params(saved, params)


# ── Stability SSI ──────────────────────────────────────────────────────────────

def _stability_ssi(z_co2: float, ms: float, T: float, P: float,
                   params: dict | None,
                   trial: str,
                   x_init: float,
                   max_iter: int = 50,
                   tol: float = 1e-8,
                   ref_x0=None,
                   accelerated: bool = True) -> dict:
    """
    Run one Michelsen stability SSI trial with optional Jex et al. acceleration.

    trial   : 'co2_rich' or 'aqueous'
    x_init  : initial x1c (CO₂-rich) or x1w (aqueous)
    ref_x0  : warm-start [Zc, chi1c] for the reference fsolve call.  Pass the
              'ref_sol' from the previous P-value to prevent the reference from
              jumping between solution branches across a pressure scan.
    accelerated : use Jex et al. 2024 accelerated SSI (extrapolation in log-space).

    Reference: CO₂-rich EOS at x1c_feed = z_h2o (valid for full composition
    range; avoids aqueous EOS breakdown at large x4w = z_co2).

    Returns dict with keys:
        converged, sum_W, tpd_negative, x1_final, lnphi_h2o, lnphi_co2,
        ref_sol  ← [Zc, chi1c] of the reference fsolve solution for warm-starting
    """
    saved = _apply_params(params)
    try:
        z_h2o    = 1.0 - z_co2
        x1c_feed = z_h2o
        x4c_feed = z_co2

        # ── Reference d_i = ln(x_i) + lnφ_i  (CO₂-rich EOS, warm-started) ──
        ref_sol = ref_x0   # carry through on failure
        try:
            d1_ref, d4_ref, ref_sol = _lnphi_c_inner(x1c_feed, T, P, x0=ref_x0)
        except Exception:
            return dict(converged=False, sum_W=np.nan, tpd_negative=True,
                        x1_final=np.nan, lnphi_h2o=np.nan, lnphi_co2=np.nan,
                        ref_sol=ref_sol)

        d1 = np.log(max(x1c_feed, 1e-300)) + d1_ref
        d4 = np.log(max(x4c_feed, 1e-300)) + d4_ref

        # ── SSI loop (warm-started within iterations, accelerated) ────────────
        x1_t       = float(x_init)
        sol_prev   = None
        converged  = False
        sum_W_prev = np.nan
        lnphi1 = lnphi4 = np.nan
        _m = 1.0          # acceleration factor
        _g_prev = None     # previous residual vector for acceleration

        for _ in range(max_iter):
            try:
                if trial == "co2_rich":
                    lnphi1, lnphi4, sol_prev = _lnphi_c_inner(x1_t, T, P,
                                                                x0=sol_prev)
                else:
                    lnphi1, lnphi4, sol_prev = _lnphi_aq_inner(x1_t, ms, T, P,
                                                                 x0=sol_prev)
            except Exception:
                sol_prev = None
                break

            # Current W in log-space
            lnW1 = np.clip(d1 - lnphi1, -500, 500)
            lnW4 = np.clip(d4 - lnphi4, -500, 500)

            W1    = np.exp(lnW1)
            W4    = np.exp(lnW4)
            sum_W = W1 + W4
            if not np.isfinite(sum_W) or sum_W <= 0:
                break

            # Stationary-condition residual: g_i = ln(W_i) + lnphi_i - d_i
            lnW_old = np.array([np.log(max(W1, 1e-300)), np.log(max(W4, 1e-300))])
            g_vec   = np.array([lnW_old[0] + lnphi1 - d1,
                                lnW_old[1] + lnphi4 - d4])

            # Accelerated step-size (Jex et al. 2024, Eq. 7)
            if accelerated and _g_prev is not None:
                num_a   = np.dot(_g_prev, _g_prev)
                denom_a = np.dot(_g_prev, _g_prev - g_vec)
                if abs(denom_a) > 1e-30:
                    _m = abs(num_a / denom_a * _m)
                    _m = np.clip(_m, 1.0, 10.0)
                else:
                    _m = 1.0
            _g_prev = g_vec.copy()

            # Direct substitution update in log-space
            lnW_new = np.array([d1 - lnphi1, d4 - lnphi4])
            lnW_step = np.clip(lnW_new - lnW_old, -5.0, 5.0)

            # Apply acceleration
            W_acc = np.exp(lnW_old + _m * lnW_step)
            W_acc = np.clip(W_acc, 1e-300, 1e10)
            sum_W = W_acc[0] + W_acc[1]
            if not np.isfinite(sum_W) or sum_W <= 0:
                break

            x1_new = W_acc[0] / sum_W

            # Convergence on step norm (not sum_W difference)
            if np.linalg.norm(lnW_step) < tol:
                converged  = True
                x1_t       = x1_new
                sum_W_prev = sum_W
                break

            x1_t       = x1_new
            sum_W_prev = sum_W

        tpd_neg = bool(sum_W_prev > 1.0 + 1e-8) if converged else True
        return dict(
            converged=converged,
            sum_W=float(sum_W_prev),
            tpd_negative=tpd_neg,
            x1_final=float(x1_t),
            lnphi_h2o=float(lnphi1) if converged else np.nan,
            lnphi_co2=float(lnphi4) if converged else np.nan,
            ref_sol=ref_sol,
        )
    finally:
        _restore_params(saved, params)


# ── Public stability API ───────────────────────────────────────────────────────

def _wilson_K(T: float, P_bar: float) -> tuple[float, float]:
    """Wilson K-values for H₂O (1) and CO₂ (4): K_i = (Pc_i/P)*exp(5.373*(1+ω_i)*(1-Tc_i/T))."""
    # Acentric factors (same values as CPA2)
    omega_h2o, omega_co2 = 0.34400, 0.22394
    # Pc in bar (convert from Pa constants)
    Pc1_bar = Pc1 / 1e5   # H₂O
    Pc4_bar = Pc4 / 1e5   # CO₂
    K_h2o = (Pc1_bar / P_bar) * np.exp(5.373 * (1.0 + omega_h2o) * (1.0 - Tc1 / T))
    K_co2 = (Pc4_bar / P_bar) * np.exp(5.373 * (1.0 + omega_co2) * (1.0 - Tc4 / T))
    return float(np.clip(K_h2o, 1e-12, 1e12)), float(np.clip(K_co2, 1e-12, 1e12))


def ecpa_stability(
    z_co2: float,
    ms: float,
    T: float,
    P: float,
    params: dict | None = None,
    guess_table_fn=None,
    co2_ref_x0=None,
    aq_ref_x0=None,
) -> dict:
    """
    Michelsen phase stability test for the eCPA CO₂ + H₂O + NaCl system.

    Uses 6 trial guesses (2 original + 4 Wilson-K variants) with early
    termination when a strongly negative TPD is found (TPD < −0.01).

    Parameters
    ----------
    z_co2          : overall CO₂ mole fraction (feed)
    ms             : salt molality [mol/kg H₂O]
    T              : temperature [K]
    P              : pressure [bar]
    params         : EoS parameter overrides
    guess_table_fn : optional; not used for stability (reserved for future)
    co2_ref_x0     : warm-start [Zc, chi1c] for the CO₂-rich trial reference fsolve.
                     Pass result['co2_ref_x0'] from the previous P call to prevent
                     the reference from jumping between EOS roots across a P-scan.
    aq_ref_x0      : same for the aqueous trial reference fsolve.

    Returns
    -------
    dict with keys:
        stable, tpd_min, trial_type, x1c_trial, x1w_trial,
        sum_W_c, sum_W_aq, message, trials,
        co2_ref_x0, aq_ref_x0  ← pass to next call for warm-starting
    """
    z_h2o = 1.0 - z_co2

    # Wilson K-values for trial generation
    K_h2o, K_co2 = _wilson_K(T, P)

    # Wilson CO₂-rich trial: x1c = W_h2o / sum(W)
    W_h2o_wil = K_h2o * z_h2o
    W_co2_wil = K_co2 * z_co2
    x1c_wilson = W_h2o_wil / (W_h2o_wil + W_co2_wil)
    x1c_wilson = float(np.clip(x1c_wilson, 1e-6, 1.0 - 1e-6))

    # Wilson aqueous trial: x1w = W_h2o / sum(W) using 1/K
    W_h2o_inv = z_h2o / max(K_h2o, 1e-30)
    W_co2_inv = z_co2 / max(K_co2, 1e-30)
    x1w_wilson = W_h2o_inv / (W_h2o_inv + W_co2_inv)
    x1w_wilson = float(np.clip(x1w_wilson, 0.5, 0.9999))

    # Moderate aqueous: x4w ≈ 0.05 → x1w = (1 - 0.05) / (1 + 2*ms*Mw)
    x1w_mod = 0.95 / (1.0 + 2.0*ms*Mw) if ms > 0 else 0.95

    # Build trial list: (label, trial_type, x_init)
    x1w_default = 0.999 / (1.0 + 2.0*ms*Mw) if ms > 0 else 0.999
    trials_spec = [
        ("CO2-rich (x1c=0.01)",     "co2_rich", 0.01),
        ("Aqueous (default)",       "aqueous",  x1w_default),
        ("CO2-rich (x1c=0.10)",     "co2_rich", 0.10),
        ("Aqueous (x4w≈0.05)",      "aqueous",  x1w_mod),
        ("Wilson CO2-rich",         "co2_rich", x1c_wilson),
        ("Wilson aqueous",          "aqueous",  x1w_wilson),
    ]

    best_tpd = 0.0
    best_result_c  = None
    best_result_aq = None
    all_trials = []
    ref_x0_c  = co2_ref_x0
    ref_x0_aq = aq_ref_x0

    for label, trial_type, x_init in trials_spec:
        ref = ref_x0_c if trial_type == "co2_rich" else ref_x0_aq
        result = _stability_ssi(z_co2, ms, T, P, params,
                                trial=trial_type, x_init=x_init,
                                ref_x0=ref, accelerated=True)

        # Propagate warm-start references
        if trial_type == "co2_rich" and result.get("ref_sol") is not None:
            ref_x0_c = result["ref_sol"]
        elif trial_type == "aqueous" and result.get("ref_sol") is not None:
            ref_x0_aq = result["ref_sol"]

        sum_W = result["sum_W"]
        tpd = (1.0 - sum_W) if np.isfinite(sum_W) else 0.0
        all_trials.append((label, trial_type, float(tpd),
                           result["converged"], result["x1_final"]))

        if tpd < best_tpd:
            best_tpd = tpd
            if trial_type == "co2_rich":
                best_result_c = result
            else:
                best_result_aq = result

        # Early termination: strongly unstable
        if tpd < -0.01:
            break

    # Collect best results for each trial type
    if best_result_c is None:
        # Use the first CO₂-rich trial result (trial 0)
        best_result_c = _stability_ssi(z_co2, ms, T, P, params,
                                        trial="co2_rich", x_init=0.01,
                                        ref_x0=co2_ref_x0, accelerated=True)
        # This shouldn't normally happen since trial 0 is CO₂-rich
    if best_result_aq is None:
        best_result_aq = _stability_ssi(z_co2, ms, T, P, params,
                                         trial="aqueous", x_init=x1w_default,
                                         ref_x0=aq_ref_x0, accelerated=True)

    # Determine instability from all trials
    stable = best_tpd >= -1e-8
    sum_c  = best_result_c["sum_W"]
    sum_aq = best_result_aq["sum_W"]

    if not stable:
        # Identify which trial type found the instability
        unstable_c  = any(tpd < -1e-8 for _, tt, tpd, conv, _ in all_trials
                          if tt == "co2_rich")
        unstable_aq = any(tpd < -1e-8 for _, tt, tpd, conv, _ in all_trials
                          if tt == "aqueous")
        if unstable_c and unstable_aq:
            trial_type = "both"
            msg = "Both CO₂-rich and aqueous trials indicate instability"
        elif unstable_c:
            trial_type = "co2_rich"
            msg = "CO₂-rich trial indicates instability"
        else:
            trial_type = "aqueous"
            msg = "Aqueous trial indicates instability"
    else:
        trial_type = None
        msg = "Stable (all trials converged with tpd ≥ 0)"

    return dict(
        stable=stable,
        tpd_min=float(best_tpd),
        trial_type=trial_type,
        x1c_trial=best_result_c["x1_final"],
        x1w_trial=best_result_aq["x1_final"],
        sum_W_c=float(sum_c),
        sum_W_aq=float(sum_aq),
        message=msg,
        trials=all_trials,
        co2_ref_x0=ref_x0_c,
        aq_ref_x0=ref_x0_aq,
    )


# ── Parallel stability map ─────────────────────────────────────────────────────

_SM_params: dict | None = None
_SM_ms:     float       = 0.0
_SM_T:      float       = 0.0


def _stability_map_init(params, ms, T):
    global _SM_params, _SM_ms, _SM_T
    _SM_params, _SM_ms, _SM_T = params, float(ms), float(T)


def _stability_map_worker(task):
    i, j, z, P = task
    try:
        r = ecpa_stability(z, _SM_ms, _SM_T, P, _SM_params)
        return i, j, bool(r['stable'])
    except Exception:
        return i, j, False   # treat convergence failure as unstable


def stability_map(
    z_vals,
    P_vals,
    ms: float,
    T: float,
    params: dict | None = None,
    n_workers: int | None = None,
) -> np.ndarray:
    """
    Compute a 2-D phase-stability map over a grid of (z_CO2, P) values.

    Parameters
    ----------
    z_vals   : 1-D array of feed CO₂ mole fractions
    P_vals   : 1-D array of pressures [bar]
    ms       : salt molality [mol/kg H₂O]
    T        : temperature [K]
    params   : EoS parameter overrides
    n_workers: number of parallel workers (None = cpu_count - 1)

    Returns
    -------
    stable_map : bool array of shape (len(z_vals), len(P_vals))
    """
    z_vals = np.asarray(z_vals, dtype=float)
    P_vals = np.asarray(P_vals, dtype=float)
    tasks  = [(i, j, float(z), float(P))
              for i, z in enumerate(z_vals)
              for j, P in enumerate(P_vals)]

    n = n_workers or _cpu_count()
    result = np.zeros((len(z_vals), len(P_vals)), dtype=bool)
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n, mp_context=ctx,
                             initializer=_stability_map_init,
                             initargs=(params, ms, T)) as ex:
        for i, j, stable in ex.map(_stability_map_worker, tasks, chunksize=10):
            result[i, j] = stable
    return result


# ── Parallel salinity scan ─────────────────────────────────────────────────────

_SS_params: dict | None = None
_SS_z:      float       = 0.0
_SS_T:      float       = 0.0


def _stability_scan_init(params, z, T):
    global _SS_params, _SS_z, _SS_T
    _SS_params, _SS_z, _SS_T = params, float(z), float(T)


def _stability_ms_row_worker(task):
    """Run one ms value as a warm-started serial P-scan (P in ascending order)."""
    i, ms, P_arr = task
    co2_ref_x0 = None
    aq_ref_x0  = None
    row_c  = []
    row_aq = []
    for P in P_arr:
        try:
            r = ecpa_stability(_SS_z, float(ms), _SS_T, float(P), _SS_params,
                               co2_ref_x0=co2_ref_x0, aq_ref_x0=aq_ref_x0)
            co2_ref_x0 = r.get('co2_ref_x0')
            aq_ref_x0  = r.get('aq_ref_x0')
            row_c.append(r.get('sum_W_c',  float('nan')))
            row_aq.append(r.get('sum_W_aq', float('nan')))
        except Exception:
            co2_ref_x0 = None
            aq_ref_x0  = None
            row_c.append(float('nan'))
            row_aq.append(float('nan'))
    return i, row_c, row_aq


def stability_ms_scan(
    ms_vals,
    P_vals,
    z: float,
    T: float,
    params: dict | None = None,
    n_workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute sum_W_c and sum_W_aq over a (ms, P) grid at fixed (z, T).

    Each ms row is run as a serial warm-started P-scan (ascending P order) so
    the EOS solver stays on the same root branch across the pressure range.
    Rows are parallelised across ms values.

    Parameters
    ----------
    ms_vals  : 1-D array of salt molalities [mol/kg H₂O]
    P_vals   : 1-D array of pressures [bar]
    z        : feed CO₂ mole fraction
    T        : temperature [K]
    params   : EoS parameter overrides
    n_workers: number of parallel workers (None = cpu_count - 1)

    Returns
    -------
    sum_W_c  : float array of shape (len(ms_vals), len(P_vals))
    sum_W_aq : float array of shape (len(ms_vals), len(P_vals))

    Notes
    -----
    sum_W_c (CO₂-rich trial) is independent of ms because both the reference
    feed fugacity and the trial lnφ use the CO₂-rich EOS only.  To visualise
    the salinity effect, plot sum_W_aq (aqueous trial), which uses the
    salt-dependent aqueous EOS for the trial phase.
    """
    ms_vals = np.asarray(ms_vals, dtype=float)
    P_vals  = np.asarray(P_vals,  dtype=float)

    # Sort P ascending for warm-starting continuity; un-sort results after
    sort_idx = np.argsort(P_vals)
    P_sorted = P_vals[sort_idx].tolist()

    tasks = [(i, float(ms), P_sorted) for i, ms in enumerate(ms_vals)]

    n = min(n_workers or _cpu_count(), len(ms_vals))
    sw_c  = np.full((len(ms_vals), len(P_vals)), float('nan'))
    sw_aq = np.full((len(ms_vals), len(P_vals)), float('nan'))
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n, mp_context=ctx,
                             initializer=_stability_scan_init,
                             initargs=(params, z, T)) as ex:
        for i, row_c, row_aq in ex.map(_stability_ms_row_worker, tasks):
            sw_c[i,  sort_idx] = row_c
            sw_aq[i, sort_idx] = row_aq
    return sw_c, sw_aq


def ecpa_stability_flash(
    z_co2: float,
    ms: float,
    T: float,
    P: float,
    params: dict | None = None,
    guess_table_fn=None,
) -> dict:
    """
    Combined phase stability + flash for the eCPA system.

    If the stability test indicates a single phase, returns immediately without
    running the flash.  If unstable, runs flash_co2_h2o_salt_ssi (the TPD
    trial compositions are available as warm-start seeds in future work).

    Parameters
    ----------
    z_co2          : overall CO₂ mole fraction
    ms             : salt molality [mol/kg H₂O]
    T, P           : temperature [K], pressure [bar]
    params         : EoS parameter overrides
    guess_table_fn : required for the flash step

    Returns
    -------
    dict — same format as flash_co2_h2o_salt_ssi on success, or:
        {'phase': 'single_phase', 'stable': True, 'T': T, 'P_bar': P,
         'z_co2': z_co2, 'ms': ms, 'stability': stab_result}
    """
    stab = ecpa_stability(z_co2, ms, T, P, params, guess_table_fn)

    if stab["stable"]:
        return dict(phase="single_phase", stable=True,
                    T=T, P_bar=P, z_co2=z_co2, ms=ms,
                    stability=stab)

    # Unstable → run flash
    if guess_table_fn is None:
        raise ValueError("guess_table_fn is required for the flash step")

    result = flash_co2_h2o_salt_ssi(
        T=T, P_bar=P, z_co2=z_co2, m_tot=ms,
        guess_table_fn=guess_table_fn,
        params=params,
    )
    result["stability"] = stab
    return result
