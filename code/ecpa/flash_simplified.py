"""
flash_simplified.py — Simplified phase-split computation exploiting y_{H₂O} = 0.

Mathematical basis
------------------
For the CO₂ + H₂O (+ NaCl) system we assume H₂O is absent from the CO₂-rich phase:

    y_{H₂O} = 0  →  CO₂-rich phase is pure CO₂  (y_{CO₂} = 1)

Under this assumption (good for T ≲ 100 °C where y_{H₂O} < 1 mol%):

1. The CO₂-rich-phase fugacity collapses to the *pure-CO₂* fugacity:
       f_{CO₂}^{vap}(T, P) = P · φ_{CO₂}^{pure}(T, P)

2. The salt molality in the aqueous phase equals the *feed* molality:
       m_s^{aq} = m_{feed}   (exact under y_{H₂O} = 0, proven below)

3. The phase split reduces to a single scalar equation in x_{H₂O}^{aq}:
       f_{CO₂}^{aq}(T, P, x_{H₂O}, m_{feed}) = f_{CO₂}^{pure}(T, P)
   where  x_{CO₂}^{aq} = 1 − x_{H₂O} · (1 + 2 · m_{feed} · M_w)

4. β and all compositions follow analytically from H₂O material balance:
       β = 1 − z_{H₂O} / x_{H₂O}^{aq}
       z_{H₂O} = (1 − z_{CO₂}) / (1 + 2 · m_{feed} · M_w)

Proof that m_s^{aq} = m_feed
-----------------------------
Salt balance (NaCl non-volatile):  z_{NaCl} = (1-β) · x_{H₂O} · m_s^{aq} · M_w
H₂O balance:                       z_{H₂O}  = (1-β) · x_{H₂O}
Dividing:  m_s^{aq} = z_{NaCl} / (z_{H₂O} · M_w) = m_{feed}  (QED)

This converts the standard two-component K-value flash into a one-dimensional
CO₂-solubility calculation: simpler, equally accurate in the applicable T range,
and often faster because (a) no K-value outer loop, (b) no salt iteration,
(c) the CO₂-rich-phase inner solve is done only once (pure CO₂).

Public API
----------
flash_co2_h2o_simplified(T, P_bar, z_co2, m_tot=0.0, ...)
    Phase split for CPA (m_tot=0) or eCPA (m_tot>0).
    Returns a dict compatible with flash_co2_h2o_salt_kv.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .constants import Mw
from .stability import _lnphi_aq_inner, _lnphi_c_inner


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pure_co2_fugacity(T: float, P_bar: float) -> tuple[float, np.ndarray]:
    """
    Fugacity of pure CO₂ at (T, P_bar) [Pa] and converged [Zc, chi1c] warm-start.

    Pure CO₂ is the x_{H₂O} → 0 limit of the CO₂-rich-phase EoS.
    """
    _lnphi1c, lnphi4c, sol_c = _lnphi_c_inner(x1c=0.0, T=T, P=P_bar)
    f_pa = P_bar * 1e5 * float(np.exp(lnphi4c))   # x4c = 1 (pure)
    return f_pa, np.asarray(sol_c, dtype=float)


def _aq_co2_fugacity(x1w: float, ms: float, T: float, P_bar: float,
                     x0=None) -> tuple[float, np.ndarray]:
    """
    Fugacity of CO₂ in the aqueous phase [Pa] at given x_{H₂O} and m_s.

    Uses the eCPA aqueous lnφ inner Newton solver (3×3 system).
    x0 = [Zw, epsr, chi1w] warm-start (None → cold start).

    Returns (f_pa, sol_aq) where sol_aq = [Zw, epsr, chi1w].
    """
    x4w = 1.0 - x1w * (1.0 + 2.0 * ms * Mw)
    if x4w <= 0.0:
        return np.inf, np.full(3, np.nan)
    _lnphi1w, lnphi4w, sol_aq = _lnphi_aq_inner(x1w=x1w, ms=ms, T=T, P=P_bar, x0=x0)
    f_pa = P_bar * 1e5 * float(np.exp(lnphi4w)) * x4w
    return f_pa, np.asarray(sol_aq, dtype=float)


# ---------------------------------------------------------------------------
# Public flash function
# ---------------------------------------------------------------------------

def flash_co2_h2o_simplified(
    T: float,
    P_bar: float,
    z_co2: float,
    m_tot: float = 0.0,
    params=None,          # accepted for API compatibility; not yet propagated
    xtol: float = 1e-10,
    verbose: bool = False,
) -> dict:
    """
    Simplified CO₂ + H₂O (+ NaCl) flash assuming y_{H₂O} = 0.

    The CO₂-rich phase is treated as pure CO₂ throughout.  The only unknown is
    x_{H₂O}^{aq}, found by a single Brent root-finding call on the CO₂ fugacity
    balance.  For ternary mixtures the salt molality in the aqueous phase is
    exactly m_{feed} (no iteration), so the only inner work is the 3×3 eCPA
    Newton solve for lnφ_CO₂^{aq}.

    Valid approximation for T ≲ 100 °C (y_{H₂O} < 1 mol%).

    Parameters
    ----------
    T : float
        Temperature [K].
    P_bar : float
        Pressure [bar].
    z_co2 : float
        Overall CO₂ mole fraction (feed).
    m_tot : float
        Feed NaCl molality [mol kg⁻¹ H₂O]; 0 for salt-free CPA.
    params : dict, optional
        Reserved for parameter overrides (API compatibility).
    xtol : float
        Absolute tolerance on x_{H₂O}^{aq} for Brent solver.
    verbose : bool
        Print convergence diagnostics.

    Returns
    -------
    dict
        Keys compatible with ``flash_co2_h2o_salt_kv``:
        T, P_bar, z_co2, m_tot, phase, ms_aq, beta, N_aq, N_c,
        x_aq {x1w, x2w, x3w, x4w}, x_c {x1c, x4c},
        Z_aq, Z_c, sol (10-element), n_iter_ms, K_vals,
        sol_aq_x0, sol_c_x0.
        Additionally: ``simplified=True``, ``y_h2o_assumed_zero=True``.
    """
    T     = float(T);     P_bar = float(P_bar)
    z_co2 = float(z_co2); m_tot = float(m_tot)
    ms    = m_tot          # m_s^{aq} = m_{feed} under y_{H₂O} = 0

    # ── Feed H₂O mole fraction ────────────────────────────────────────────────
    # From z1*(1 + 2*ms*Mw) + z4 = 1  (salt mole fractions absorbed into z1):
    z4 = z_co2
    z1 = (1.0 - z4) / (1.0 + 2.0 * ms * Mw)

    # Maximum x1w consistent with x4w ≥ 0
    x1w_max = 1.0 / (1.0 + 2.0 * ms * Mw)

    # ── Pure CO₂ fugacity (one-time solve for CO₂-rich phase) ─────────────────
    f_pure, sol_c0 = _pure_co2_fugacity(T, P_bar)

    if verbose:
        print(f"[simplified] T={T:.1f}K P={P_bar:.2f}bar z4={z4:.4f} ms={ms:.4f}")
        print(f"[simplified] f_CO2_pure = {f_pure:.6e} Pa")

    # ── Residual: CO₂ fugacity balance ────────────────────────────────────────
    # g(x1w) = f_CO2^{aq}(x1w) - f_CO2^{pure}
    # Monotonically decreasing in x1w (as x1w↑, x4w↓, f_CO2^aq↓).
    # Root → two-phase equilibrium x1w.

    _sol_aq_cache = [None]   # warm-start carrier across Brent evaluations
    _n_calls = [0]

    def residual(x1w: float) -> float:
        f_aq, sol_aq = _aq_co2_fugacity(x1w, ms, T, P_bar, x0=_sol_aq_cache[0])
        if np.isfinite(sol_aq[0]):
            _sol_aq_cache[0] = sol_aq   # update warm-start only on success
        _n_calls[0] += 1
        return f_aq - f_pure

    # ── Phase identification ───────────────────────────────────────────────────
    # At x1w = z1 (β = 0): all CO₂ is dissolved, x4w = z4.
    # g(z1) > 0 → CO₂ over-saturated → two-phase
    # g(z1) ≤ 0 → CO₂ under-saturated → single-phase aqueous
    g_lo = residual(z1)

    def _single_phase_aq_result(x1w_sp: float) -> dict:
        """Build single-phase aqueous result dict."""
        x4w_sp = max(1.0 - x1w_sp * (1.0 + 2.0 * ms * Mw), 0.0)
        x2w_sp = x3w_sp = x1w_sp * ms * Mw
        sol_aq = _sol_aq_cache[0]
        Zw = float(sol_aq[0]) if (sol_aq is not None and np.isfinite(sol_aq[0])) else float('nan')
        epsr = float(sol_aq[1]) if sol_aq is not None else float('nan')
        chi1w = float(sol_aq[2]) if sol_aq is not None else float('nan')
        Zc = float(sol_c0[0])
        chi1c = float(sol_c0[1])
        sol_vec = np.array([Zw, x1w_sp, epsr, Zc, 0.0,
                            chi1w, chi1c, np.nan, np.nan, np.nan])
        return {
            "T": T, "P_bar": P_bar, "z_co2": z_co2, "m_tot": m_tot,
            "phase": "single_phase",
            "ms_aq": ms,
            "beta": 0.0, "N_aq": 1.0, "N_c": 0.0,
            "x_aq": dict(x1w=x1w_sp, x2w=x2w_sp, x3w=x3w_sp, x4w=x4w_sp),
            "x_c":  dict(x1c=0.0, x4c=1.0),
            "Z_aq": Zw, "Z_c": Zc,
            "sol": sol_vec,
            "n_iter_ms": _n_calls[0],
            "K_vals": (0.0, 1.0 / max(x4w_sp, 1e-15)),
            "sol_aq_x0": sol_aq,
            "sol_c_x0":  sol_c0,
            "simplified": True, "y_h2o_assumed_zero": True,
        }

    if g_lo <= 0.0:
        if verbose:
            print(f"[simplified] Single-phase aqueous (g_lo={g_lo:.3e})")
        return _single_phase_aq_result(z1)

    # ── Two-phase: root in (z1, x1w_max) ──────────────────────────────────────
    # Evaluate upper bracket (x4w → 0, f_CO2^aq → 0 → g < 0 always).
    x1w_hi = x1w_max * (1.0 - 1e-8)
    g_hi   = residual(x1w_hi)

    if g_hi >= 0.0:
        # Shouldn't occur for well-behaved EoS; treat as single-phase CO₂.
        if verbose:
            print(f"[simplified] No sign change: g_lo={g_lo:.3e} g_hi={g_hi:.3e}. "
                  "Returning single-phase CO₂.")
        return {
            "T": T, "P_bar": P_bar, "z_co2": z_co2, "m_tot": m_tot,
            "phase": "single_phase",
            "ms_aq": ms,
            "beta": 1.0, "N_aq": 0.0, "N_c": 1.0,
            "x_aq": dict(x1w=z1, x2w=z1*ms*Mw, x3w=z1*ms*Mw, x4w=z4),
            "x_c":  dict(x1c=0.0, x4c=1.0),
            "Z_aq": float('nan'), "Z_c": float(sol_c0[0]),
            "sol": np.full(10, np.nan),
            "n_iter_ms": _n_calls[0],
            "K_vals": (0.0, float('nan')),
            "sol_aq_x0": None, "sol_c_x0": sol_c0,
            "simplified": True, "y_h2o_assumed_zero": True,
        }

    x1w_sol = brentq(residual, z1, x1w_hi, xtol=xtol, full_output=False)

    # ── Recover equilibrium phase properties ──────────────────────────────────
    ms_aq = ms
    x1w   = float(x1w_sol)
    x2w   = x3w = x1w * ms_aq * Mw
    x4w   = 1.0 - x1w * (1.0 + 2.0 * ms_aq * Mw)
    x4w   = max(x4w, 0.0)

    # Phase fraction from H₂O balance: β = 1 - z1/x1w
    beta  = float(np.clip(1.0 - z1 / x1w, 0.0, 1.0))
    N_aq  = 1.0 - beta
    N_c   = beta

    # CO₂-rich phase is pure CO₂
    x1c, x4c = 0.0, 1.0

    # Z factors and inner state from converged Brent evaluations
    sol_aq = _sol_aq_cache[0]
    Zw     = float(sol_aq[0]) if (sol_aq is not None and np.isfinite(sol_aq[0])) else float('nan')
    epsr   = float(sol_aq[1]) if sol_aq is not None else float('nan')
    chi1w  = float(sol_aq[2]) if sol_aq is not None else float('nan')
    Zc     = float(sol_c0[0])
    chi1c  = float(sol_c0[1])

    sol_vec = np.array([Zw, x1w, epsr, Zc, x1c,
                        chi1w, chi1c, np.nan, np.nan, np.nan])

    # K-values under y_{H₂O} = 0 assumption
    K1 = 0.0                          # y_H2O / x_H2O = 0
    K4 = 1.0 / max(x4w, 1e-15)       # y_CO2 / x_CO2 = 1 / x4w

    if verbose:
        print(f"[simplified] x1w={x1w:.6f}  x4w={x4w:.6f}  beta={beta:.6f}  "
              f"ms_aq={ms_aq:.4f}  n_calls={_n_calls[0]}")

    return {
        "T": T, "P_bar": P_bar, "z_co2": z_co2, "m_tot": m_tot,
        "phase": "two_phase",
        "ms_aq":  ms_aq,
        "N_aq":   N_aq,  "N_c":  N_c,
        "beta":   beta,
        "x_aq":   dict(x1w=x1w, x2w=x2w, x3w=x3w, x4w=x4w),
        "x_c":    dict(x1c=x1c, x4c=x4c),
        "Z_aq":   Zw,    "Z_c":  Zc,
        "sol":    sol_vec,
        "n_iter_ms":  _n_calls[0],
        "K_vals":     (K1, K4),
        "sol_aq_x0":  sol_aq,
        "sol_c_x0":   sol_c0,
        "simplified": True, "y_h2o_assumed_zero": True,
    }
