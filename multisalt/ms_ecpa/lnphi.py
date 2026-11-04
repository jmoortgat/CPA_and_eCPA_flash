"""Per-phase log-fugacity-coefficient kernels for the multi-salt eCPA.

These are the *self-consistent* aqueous and gas-phase ``lnφ`` evaluations that
``ms_ecpa.eos.residual_elv`` performs internally.  Extracting them as
black-box callables lets the K-value SSI flash (Step 1) and the Michelsen TPD
stability search (Step 2) reuse the verbatim physics of the original code
without re-running the full coupled equilibrium solver.

Inputs and units
----------------
* aqueous: ``x_aq = (x1w, x2w, x3w)`` — water, methane, CO2 mole fractions in
  the *aqueous* phase (the ion mole fractions ``xiw = x1w * Mw * mi`` are
  reconstructed inside the kernel; the caller does not pass them).
* gas: ``y_gas = (y1, y2, y3)`` — water, methane, CO2 mole fractions in the
  gas phase (ions are non-partitioning, so the gas phase has only these three
  components).
* ``T`` in Kelvin, ``P`` in **Pa** (SI, matching the internal convention of
  ``residual_elv``; multiply ``P_bar`` by ``1e5`` when calling).

Design choices (per the Phase A spec)
-------------------------------------
* ``lnphi_aq`` solves internally for ``(Zw, epsr, chi1w)`` by a small Newton
  iteration on the three EoS-closure equations (volume root, association
  fixed-point, dielectric self-consistency).  The four χ-derivatives needed
  by the permittivity-coupled lnφ then follow analytically: each one is a
  linear-in-itself fixed-point of the form ``x = α x + β``, with solution
  ``x = β / (1 − α)``.  This makes ``lnphi_aq`` a true black box.
* ``lnphi_gas`` solves internally for ``(Zh, chi1h)``.  The cubic SRK
  equation in ``Zh`` can have 1 or 3 real roots; the function picks the
  root closest to the hint ``Z_init`` (default 1.0 — the gas-like root).
* All inner Newton iterations use ``ms_ecpa.newton.newton_solve``
  (complex-step Jacobian, damped, with backtracking line search).

The exact algebra is copy-faithful to the aqueous-phase block in
``ms_ecpa.eos.residual_elv`` (lines 28 – 202) and to the gas-phase block
(lines 208 – 242).  Any deviation would break the consistency test in
``tests/test_lnphi_consistency.py``.
"""
from __future__ import annotations

import numpy as np

from .parameters import *  # noqa: F401,F403  (model constants by name)
from . import parameters as _prm  # live module ref (runtime-togglable flags)
from . import extensions as _ext
from .newton import newton_solve


# ---------------------------------------------------------------------------
# Aqueous phase
# ---------------------------------------------------------------------------

def _aq_core_state(Zw, epsr, chi1w, x_aq, mi, T, P):
    """Compute every cached intermediate used by the aqueous residual and lnφ.

    Returns a dict so the residual, the χ-derivative solver, and the lnφ
    evaluator all share one symbolic state (no duplication of physics).
    The inputs may be complex-valued (for complex-step differentiation).
    """
    x1w, x2w, x3w = x_aq
    xiw = x1w * Mw * mi
    ai_loc = a0i * (1 + c1i * (1 - (T / Tref) ** 0.5)) ** 2

    a1 = a01 * (1 + c11 * (1 - (T / Tc1) ** 0.5)) ** 2
    a2 = a02 * (1 + c12 * (1 - (T / Tc2) ** 0.5)) ** 2
    a3 = a03 * (1 + c13 * (1 - (T / Tc3) ** 0.5)) ** 2
    k13 = Ak13 * (T / Tc3) ** 2 + Bk13 * (T / Tc3) + Ck13
    S13 = _prm.s13_of_T(T)          # one definition; see parameters.s13_of_T
    k12 = Ak12 * (T / Tc1) ** 2 + Bk12 * (T / Tc1) + Ck12
    U12 = np.log(2) * (a2 / b2 - 2 * (a1 * a2) ** 0.5 * (1 - k12) / (b1 + b2))
    U21 = np.log(2) * (a1 / b1 - 2 * (a2 * a1) ** 0.5 * (1 - k12) / (b2 + b1))
    U13 = np.log(2) * (a3 / b3 - 2 * (a1 * a3) ** 0.5 * (1 - k13) / (b1 + b3))
    U31 = np.log(2) * (a1 / b1 - 2 * (a3 * a1) ** 0.5 * (1 - k13) / (b3 + b1))
    U23 = np.log(2) * (a3 / b3 - 2 * (a2 * a3) ** 0.5 * (1 - k23) / (b2 + b3))
    U32 = np.log(2) * (a2 / b2 - 2 * (a3 * a2) ** 0.5 * (1 - k23) / (b3 + b2))
    U1i = Uref1i + alfa1i * R * ((1 - T / Talfa1i) ** 2 - (1 - Tref / Talfa1i) ** 2)
    U2i = A_U2i * (T / Tc2) + B_U2i
    U3i = A_U3i * (T / Tc3) + B_U3i

    A1 = a1 * P / R ** 2 / T ** 2
    B1 = b1 * P / R / T
    A2 = a2 * P / R ** 2 / T ** 2
    B2 = b2 * P / R / T
    A3 = a3 * P / R ** 2 / T ** 2
    B3 = b3 * P / R / T
    Bi = bi * P / R / T

    rho = P / Zw / R / T
    b = b1 * x1w + b2 * x2w + b3 * x3w + np.sum(bi * xiw)
    gE = 1 / b * (x1w * x2w * (b1 * U12 + b2 * U21)
                  + x1w * x3w * (b1 * U13 + b3 * U31)
                  + x2w * x3w * (b2 * U23 + b3 * U32)
                  + x1w * np.sum(xiw * U1i * (b1 + bi))
                  + x2w * np.sum(xiw * U2i * (b2 + bi))
                  + x3w * np.sum(xiw * U3i * (b3 + bi)))
    # Ion-ion Huron-Vidal energies (production = all zeros).
    # The published double sum g^E = (1/b) sum_ij x_i x_j b_j DU_ij extended
    # to the ion block: a symmetric DU_ij adds x_i x_j (b_i + b_j) DU_ij / b
    # per unordered ion pair -- the same convention as the water-ion terms
    # above.  The lnphi
    # composition/volume chains need NO new explicit terms: DU_kj = 0 for
    # every partitioning species k against ions, so the ion-ion term enters
    # lnPHI{1,2,3} only through gE itself (the -B_k*gE/(R T) entries below)
    # and through a/A -- both of which already use this gE variable.  With
    # Uref_ii all zeros the quadratic form is analytically zero and the
    # branch is skipped: the baseline is bit-exact.  Complex-step safe: the
    # gate tests only the (real) parameter matrix, never a perturbed value.
    Uii = _prm.Uref_ii
    if np.any(Uii):
        gE = gE + np.dot(xiw, Uii @ (bi * xiw)) / b
    a = b * (x1w * a1 / b1 + x2w * a2 / b2 + x3w * a3 / b3
             + np.sum(xiw * ai_loc / bi) - gE / np.log(2))
    A = a * P / R ** 2 / T ** 2
    B = b * P / R / T

    Zphys = Zw / (Zw - B) - A / (Zw + B)
    # Physical lnφ (same algebra as residual_elv)
    lnPHI1phys = (-np.log(Zw - B) + B1 / B * (B / (Zw - B) - A / (Zw + B))
                  - np.log((Zw + B) / Zw)
                  * (A1 / B1 - 1 / (B * np.log(2))
                     * (x2w / R / T * (B1 * U12 + B2 * U21)
                        + x3w / R / T * (B1 * U13 + B3 * U31)
                        + np.sum(xiw * U1i / R / T * (B1 + Bi))
                        - B1 * gE / R / T)))
    lnPHI2phys = (-np.log(Zw - B) + B2 / B * (B / (Zw - B) - A / (Zw + B))
                  - np.log((Zw + B) / Zw)
                  * (A2 / B2 - 1 / (B * np.log(2))
                     * (x1w / R / T * (B1 * U12 + B2 * U21)
                        + x3w / R / T * (B2 * U23 + B3 * U32)
                        + np.sum(xiw * U2i / R / T * (B2 + Bi))
                        - B2 * gE / R / T)))
    lnPHI3phys = (-np.log(Zw - B) + B3 / B * (B / (Zw - B) - A / (Zw + B))
                  - np.log((Zw + B) / Zw)
                  * (A3 / B3 - 1 / (B * np.log(2))
                     * (x1w / R / T * (B1 * U13 + B3 * U31)
                        + x2w / R / T * (B3 * U32 + B2 * U23)
                        + np.sum(xiw * U3i / R / T * (B3 + Bi))
                        - B3 * gE / R / T)))

    # Association.  Slot-2 water cross-association (S12A, production
    # 0.0) mirrors the CO2 solvation pattern exactly: with S12A = 0,
    # chi2w = Zw/Zw = 1 identically and every added term is a bit-exact
    # zero (see parameters.S12A).
    S12A_ = _prm.S12A
    eta = B / 4 / Zw
    g_eta = 1 / (1 - 1.9 * eta)
    dg_deta = 1.9 / (1 - 1.9 * eta) ** 2
    delta = (g_eta * kappaW) * (np.exp(epsW / T) - 1)
    DELTA = delta * P / R / T
    chi3w = Zw / (Zw + 2 * x1w * chi1w * S13 * DELTA)
    chi2w = Zw / (Zw + 2 * x1w * chi1w * S12A_ * DELTA)
    chi1w_new = Zw / (Zw + 2 * x1w * chi1w * DELTA + 2 * x3w * chi3w * S13 * DELTA
                      + 2 * x2w * chi2w * S12A_ * DELTA)
    Zassoc = -2 * (1 + eta / g_eta * dg_deta) * (x1w * (1 - chi1w) + x3w * (1 - chi3w)
                                                 + x2w * (1 - chi2w))
    _assoc_sum = (x1w * (chi1w - 1) + x3w * (chi3w - 1)
                  + x2w * (chi2w - 1))
    lnPHI1assoc = (4 * np.log(chi1w)
                   + B1 / (2 * g_eta * Zw) * dg_deta * _assoc_sum)
    lnPHI2assoc = (4 * np.log(chi2w)
                   + B2 / (2 * g_eta * Zw) * dg_deta * _assoc_sum)
    lnPHI3assoc = (4 * np.log(chi3w)
                   + B3 / (2 * g_eta * Zw) * dg_deta * _assoc_sum)

    # Debye-Hückel
    xiZi = np.sum(xiw * Zi ** 2)
    has_ions = (np.sum(mi) > 0)
    if has_ions:
        debye = (e ** 2 * Na * rho * xiZi / (kb * T * epsr * eps0)) ** 0.5
        Xi = 1 / Sgi ** 3 * (np.log(1 + debye * Sgi) - debye * Sgi
                             + 0.5 * (debye * Sgi) ** 2)
        ZDH = (1 / (4 * np.pi * Na * rho * xiZi)
               * np.sum(xiw * Zi ** 2 * (Xi - 0.5 * debye ** 3 / (1 + debye * Sgi))))
    else:
        debye = 0.0
        ZDH = 0.0

    # Permittivity (Maribo-Mogensen)
    rho1 = rho * x1w
    # Extension terms (ms_ecpa.extensions). With no extension registered
    # these are the neutral constants r_d = 1.0, w = 1.0, V dw/dV = 0.0,
    # N dw/dN_w = 0.0, Z_ext = 0.0, lnphi_ext = 0.0, and every expression
    # below reduces to the unextended closure.
    ext = _ext.aqueous_terms(T, rho, x1w, x2w, x3w, xiw) if has_ions \
        else _ext.NEUTRAL
    r_d = ext.density_factor
    M = (Na * rho * r_d / (3 * eps0)
         * (x1w * pol1 + x2w * pol2 + x3w * pol3 + np.sum(xiw * poli)))
    eps_inf = (2 * M + 1) / (1 - M)
    Pww = 2 * rho * x1w * delta * chi1w ** 2
    Pwc = 2 * rho * x3w * S13 * delta * chi1w * chi3w
    Pwh = 2 * rho * x2w * S12A_ * delta * chi1w * chi2w
    Pw = Pww + Pwc + Pwh
    gw = 1 + zww * Pww * np.cos(GAMMA1) / (Pw * np.cos(THETA1) + 1)
    w_d = ext.dipole_weight
    T1 = (2 * epsr + eps_inf) * (epsr - eps_inf) / (epsr * (eps_inf + 2) ** 2)
    T2 = Na * rho * r_d / (9 * eps0 * kb * T) * (x1w * w_d * gw * dip01 ** 2)

    return dict(
        x1w=x1w, x2w=x2w, x3w=x3w, xiw=xiw,
        B1=B1, B2=B2, B3=B3, Bi=Bi, B=B, A=A,
        rho=rho, rho1=rho1, b=b, gE=gE, S13=S13,
        Zphys=Zphys, Zassoc=Zassoc, ZDH=ZDH,
        lnPHI1phys=lnPHI1phys, lnPHI2phys=lnPHI2phys, lnPHI3phys=lnPHI3phys,
        lnPHI1assoc=lnPHI1assoc, lnPHI2assoc=lnPHI2assoc, lnPHI3assoc=lnPHI3assoc,
        eta=eta, g_eta=g_eta, dg_deta=dg_deta,
        delta=delta, DELTA=DELTA,
        chi3w=chi3w, chi2w=chi2w, chi1w_new=chi1w_new, S12A=S12A_,
        M=M, eps_inf=eps_inf, r_d=r_d,
        Pww=Pww, Pwc=Pwc, Pwh=Pwh, Pw=Pw, gw=gw,
        w_d=w_d, V_dw_dV=ext.V_dw_dV, N_dw_dNw=ext.N_dw_dNw,
        Z_ext=ext.Z, lnphi_ext=ext.lnphi,
        T1=T1, T2=T2,
        xiZi=xiZi, debye=debye, has_ions=has_ions,
    )


def _aq_chi_derivatives(state, Zw, epsr, chi1w, x_aq, mi, T, P):
    """Solve the four linear χ-derivative fixed points analytically.

    Each derivative ``x ∈ {Vdchi1WdV, Ndchi1WdNw, Ndchi1WdNc, Ndchi1WdNh}``
    satisfies ``x_new(x) = α·x + β`` with constants determined by
    ``state`` (a snapshot of EoS intermediates at a converged
    ``(Zw, epsr, chi1w)``).  The fixed-point ``x = x_new`` is solved by
    ``x = β / (1 − α)`` — no Newton.

    Returns the four scalars as ``(Vdchi1WdV, Ndchi1WdNw, Ndchi1WdNc,
    Ndchi1WdNh)``.  Slots corresponding to absent partitioning species
    (``x2w == 0`` or ``x3w == 0``) are returned as 1.0 placeholders that
    are unused by the lnφ evaluator in those branches.
    """
    x1w, x2w, x3w = x_aq
    delta = state["delta"]; chi3w = state["chi3w"]; g_eta = state["g_eta"]
    dg_deta = state["dg_deta"]; eta = state["eta"]; b = state["b"]
    rho = state["rho"]; S13 = state["S13"]
    chi2w = state["chi2w"]; S12A_ = state["S12A"]

    # Common dχ_w / dφ helpers (constants given Zw, epsr, chi1w).
    # The slot-2 solvation (S12A, production 0.0) adds an H-system for
    # chi2w that mirrors the CO2 G-system exactly; every added term
    # carries a factor S12A (or chi2w - 1 = 0) and vanishes bit-exactly
    # at S12A = 0.
    # Every partial of a row carries chi_new**2 of THAT row's own residual
    # (= chi**2 at the fixed point): the F row is chi1w**2 throughout, the
    # G row chi3w**2, the H row chi2w**2.  Corrected 2026-09; see
    # tests/test_chi_chain_derivatives.py for the complex-step ground truth.
    dFdchiw = 1 + 2 * x1w * rho * delta * chi1w ** 2
    dFdchic = 2 * x3w * rho * S13 * delta * chi1w ** 2
    dFdchih = 2 * x2w * rho * S12A_ * delta * chi1w ** 2
    dGdchiw = 2 * x1w * rho * S13 * delta * chi3w ** 2
    dGdchic = 1.0
    dHdchiw = 2 * x1w * rho * S12A_ * delta * chi2w ** 2
    VddeltadV = -delta * eta * dg_deta / g_eta
    # d/ddelta of chi - 1/(1 + Q(delta)) is +chi**2 * dQ/ddelta, with
    # dQ/ddelta = Q/delta.  The legacy -chi**2*(1 + (delta-1)*dQ/ddelta)
    # grouping is dimensionally incoherent (delta is m^3/mol, so delta-1
    # mixes units) and equals the truth minus exactly chi.
    dFddelta = chi1w ** 2 * (2 * rho * x1w * chi1w
                             + 2 * rho * x3w * chi3w * S13
                             + 2 * rho * x2w * chi2w * S12A_)
    dGddelta = chi3w ** 2 * (2 * rho * x1w * chi1w * S13)
    dHddelta = chi2w ** 2 * (2 * rho * x1w * chi1w * S12A_)

    # General fixed-point coefficients: each derivative x satisfies
    #   x = β / (1 − α)
    # where α = (dFdchic·dGdchiw + dFdchih·dHdchiw)/dFdchiw (shared) and
    # β depends on the specific φ (V, Nw, Nc, Nh).
    alpha = (dFdchic * dGdchiw + dFdchih * dHdchiw) / dFdchiw

    # --- Vdchi1WdV --------------------------------------------------------
    dFdV = -(2 * rho * x1w * delta * chi1w + 2 * rho * x3w * S13 * delta * chi3w
             + 2 * rho * x2w * S12A_ * delta * chi2w) * chi1w ** 2
    dGdV = -(2 * rho * x1w * S13 * delta * chi1w) * chi3w ** 2
    dHdV = -(2 * rho * x1w * S12A_ * delta * chi1w) * chi2w ** 2
    beta_V = (dFdchic * (dGdV + dGddelta * VddeltadV)
              + dFdchih * (dHdV + dHddelta * VddeltadV)
              - dFdV - dFddelta * VddeltadV) / dFdchiw
    Vdchi1WdV = beta_V / (1.0 - alpha)

    # --- Ndchi1WdNw -------------------------------------------------------
    NddeltadN1 = delta * eta * b1 * dg_deta / (g_eta * b)
    dFdNw = 2 * rho * delta * chi1w ** 3
    dGdNw = 2 * rho * S13 * delta * chi1w * chi3w ** 2
    dHdNw = 2 * rho * S12A_ * delta * chi1w * chi2w ** 2
    beta_Nw = (dFdchic * (dGdNw + dGddelta * NddeltadN1)
               + dFdchih * (dHdNw + dHddelta * NddeltadN1)
               - dFdNw - dFddelta * NddeltadN1) / dFdchiw
    Ndchi1WdNw = beta_Nw / (1.0 - alpha)

    # --- Ndchi1WdNc -------------------------------------------------------
    if x3w > 0:
        NddeltadN3 = delta * eta * b3 * dg_deta / (g_eta * b)
        dFdNc = 2 * rho * S13 * delta * chi1w ** 2 * chi3w
        dGdNc = 0.0
        dHdNc = 0.0
        beta_Nc = (dFdchic * (dGdNc + dGddelta * NddeltadN3)
                   + dFdchih * (dHdNc + dHddelta * NddeltadN3)
                   - dFdNc - dFddelta * NddeltadN3) / dFdchiw
        Ndchi1WdNc = beta_Nc / (1.0 - alpha)
    else:
        Ndchi1WdNc = 1.0  # unused: x3w=0 zeroes lnφ_CO2 in the aqueous phase

    # --- Ndchi1WdNh -------------------------------------------------------
    if x2w > 0:
        NddeltadN2 = delta * eta * b2 * dg_deta / (g_eta * b)
        dFdNh = 2 * rho * S12A_ * delta * chi1w ** 2 * chi2w
        dGdNh = 0.0
        dHdNh = 0.0
        beta_Nh = (dFdchic * (dGdNh + dGddelta * NddeltadN2)
                   + dFdchih * (dHdNh + dHddelta * NddeltadN2)
                   - dFdNh - dFddelta * NddeltadN2) / dFdchiw
        Ndchi1WdNh = beta_Nh / (1.0 - alpha)
    else:
        Ndchi1WdNh = 1.0

    return Vdchi1WdV, Ndchi1WdNw, Ndchi1WdNc, Ndchi1WdNh


def _aq_residual_3var(v3, x_aq, mi, T, P):
    """Three-equation aqueous residual in (Zw, epsr, chi1w).

    Residuals:
      * f1 = (Zw − Zw_new)/Zw          — physical+association+DH+permittivity
      * f2 = (chi1w − chi1w_new)/chi1w — association closure
      * f3 = (T1 − T2)/T2              — Maribo-Mogensen permittivity closure

    The χ-derivatives are solved analytically inside (``_aq_chi_derivatives``)
    and substituted into ZPerm.  This eliminates the four extra unknowns
    that the full ``residual_elv`` carries.
    """
    Zw, epsr, chi1w = v3

    state = _aq_core_state(Zw, epsr, chi1w, x_aq, mi, T, P)
    Vdchi1WdV, Ndchi1WdNw, Ndchi1WdNc, Ndchi1WdNh = _aq_chi_derivatives(
        state, Zw, epsr, chi1w, x_aq, mi, T, P,
    )

    # ZPerm closure: needs VderdV, which needs (gw, Pw, …) — recompute the
    # bits not stored in ``state`` (could be pre-cached too, but this keeps
    # the algebra readable).
    x1w, x2w, x3w = x_aq
    M = state["M"]; eps_inf = state["eps_inf"]
    rho = state["rho"]; rho1 = state["rho1"]
    Pw = state["Pw"]; Pww = state["Pww"]; Pwc = state["Pwc"]
    Pwh = state["Pwh"]
    gw = state["gw"]; delta = state["delta"]
    chi3w = state["chi3w"]; g_eta = state["g_eta"]; dg_deta = state["dg_deta"]
    eta = state["eta"]; b = state["b"]; S13 = state["S13"]
    chi2w = state["chi2w"]; S12A_ = state["S12A"]
    debye = state["debye"]; xiZi = state["xiZi"]

    # dPw* w.r.t. V using analytically-solved Vdchi1WdV
    VddeltadV = -delta * eta * dg_deta / g_eta
    # Vdchi3WdV from the same linear system
    dGdchiw = 2 * x1w * rho * S13 * delta * chi3w ** 2
    dGddelta = chi3w ** 2 * (2 * rho * x1w * chi1w * S13)
    dGdV = -(2 * rho * x1w * S13 * delta * chi1w) * chi3w ** 2
    Vdchi3WdV = -(dGdchiw * Vdchi1WdV + dGdV + dGddelta * VddeltadV)
    # Vdchi2WdV from the mirrored slot-2 system (0-value terms at S12A=0)
    dHdchiw = 2 * x1w * rho * S12A_ * delta * chi2w ** 2
    dHddelta = chi2w ** 2 * (2 * rho * x1w * chi1w * S12A_)
    dHdV = -(2 * rho * x1w * S12A_ * delta * chi1w) * chi2w ** 2
    Vdchi2WdV = -(dHdchiw * Vdchi1WdV + dHdV + dHddelta * VddeltadV)

    dgwdPw = -(gw - 1) * np.cos(THETA1) / (Pw * np.cos(THETA1) + 1)
    if abs(Pww) < 1e-30:  # no water self-association (shouldn't happen here)
        dgwdPww = 0.0
    else:
        dgwdPww = (gw - 1) / Pww
    VdPwwdV = Pww * (-1 + 1 / delta * VddeltadV + 2 / chi1w * Vdchi1WdV)
    if x3w > 0:
        VdPwcdV = Pwc * (-1 + 1 / delta * VddeltadV + 1 / chi1w * Vdchi1WdV
                          + 1 / chi3w * Vdchi3WdV)
    else:
        VdPwcdV = 0.0
    if x2w > 0 and S12A_ != 0.0:
        VdPwhdV = Pwh * (-1 + 1 / delta * VddeltadV + 1 / chi1w * Vdchi1WdV
                          + 1 / chi2w * Vdchi2WdV)
    else:
        VdPwhdV = 0.0
    VdPwdV = VdPwwdV + VdPwcdV + VdPwhdV
    VdgwdV = dgwdPw * VdPwdV + dgwdPww * VdPwwdV

    dFder = (2 * epsr ** 2 + eps_inf ** 2) / (epsr ** 2 * (eps_inf + 2) ** 2)
    dFdeinf = (epsr * eps_inf - 4 * eps_inf - 4 * epsr ** 2 - 2 * epsr) / (epsr * (eps_inf + 2) ** 3)
    # Density-factor chain: M and the dipole term are evaluated at density
    # rho*r_d with d(V/r_d)/dV = 1, so the V-derivative of every such factor
    # gains a factor r_d (1.0 without an extension).
    r_d = state["r_d"]
    VdeinfdV = -3 * M * r_d / (1 - M) ** 2
    # Dipole-weight chain (w_d = 1, V dw/dV = 0 without an extension).
    w_d = state["w_d"]
    VdwdV = state["V_dw_dV"]
    # T2 = C*(rho1*r_d)*w_d*gw with V d(rho1*r_d)/dV = -rho1*r_d^2: the
    # dilution part of the V-chain carries r_d, the gw/w_d parts do not.
    VdFdV = (Na * rho1 * r_d * dip01 ** 2 / (9 * eps0 * kb * T)
             * (w_d * (gw * r_d - VdgwdV) - gw * VdwdV))
    VderdV = -(dFder) ** -1 * (dFdeinf * VdeinfdV + VdFdV)

    if state["has_ions"]:
        daDHBder = (debye ** 2 / (8 * np.pi * Na * rho * epsr * xiZi)
                    * np.sum(state["xiw"] * Zi ** 2 *
                             (debye / (1 + debye * Sgi) - 1 / Rbi)))
    else:
        daDHBder = 0.0
    ZPerm = -daDHBder * VderdV

    # Extension Z contribution (0.0 without an extension).
    Zw_new = state["Zphys"] + state["Zassoc"] + state["ZDH"] + ZPerm + state["Z_ext"]

    f1 = (Zw - Zw_new) / Zw
    f2 = (chi1w - state["chi1w_new"]) / chi1w
    f3 = (state["T1"] - state["T2"]) / state["T2"]
    return [f1, f2, f3]


def _aq_lnphi_evaluate_3var(v3, x_aq, mi, T, P):
    """Aqueous lnφ given the converged ``(Zw, epsr, chi1w)`` triple.

    Computes the χ-derivatives analytically and then assembles
    ``(lnφ_h2o, lnφ_ch4, lnφ_co2)`` from the same five contributions used by
    ``residual_elv``: physical, association, DH (zero), Born (zero in this
    implementation), permittivity.
    """
    Zw, epsr, chi1w = v3
    x1w, x2w, x3w = x_aq

    state = _aq_core_state(Zw, epsr, chi1w, x_aq, mi, T, P)
    Vdchi1WdV, Ndchi1WdNw, Ndchi1WdNc, Ndchi1WdNh = _aq_chi_derivatives(
        state, Zw, epsr, chi1w, x_aq, mi, T, P,
    )

    # Permittivity lnφ — need NderdN_k for each partitioning species
    M = state["M"]; eps_inf = state["eps_inf"]
    rho = state["rho"]; Pw = state["Pw"]; Pww = state["Pww"]; Pwc = state["Pwc"]
    Pwh = state["Pwh"]
    gw = state["gw"]; delta = state["delta"]; chi3w = state["chi3w"]
    g_eta = state["g_eta"]; dg_deta = state["dg_deta"]; eta = state["eta"]
    b = state["b"]; S13 = state["S13"]; debye = state["debye"]
    chi2w = state["chi2w"]; S12A_ = state["S12A"]
    xiZi = state["xiZi"]; xiw = state["xiw"]
    _s12_on = (S12A_ != 0.0)

    NddeltadN1 = delta * eta * b1 * dg_deta / (g_eta * b)
    NddeltadN2 = delta * eta * b2 * dg_deta / (g_eta * b)
    NddeltadN3 = delta * eta * b3 * dg_deta / (g_eta * b)
    dGdchiw = 2 * x1w * rho * S13 * delta * chi3w ** 2
    dGdchic = 1.0
    dGddelta = chi3w ** 2 * (2 * rho * x1w * chi1w * S13)
    dGdNw = 2 * rho * S13 * delta * chi1w * chi3w ** 2
    # Mirrored slot-2 (chi2w) system; all terms vanish at S12A = 0.
    dHdchiw = 2 * x1w * rho * S12A_ * delta * chi2w ** 2
    dHddelta = chi2w ** 2 * (2 * rho * x1w * chi1w * S12A_)
    dHdNw = 2 * rho * S12A_ * delta * chi1w * chi2w ** 2

    Ndchi3WdNw = -dGdchic ** -1 * (dGdchiw * Ndchi1WdNw + dGdNw + dGddelta * NddeltadN1)
    Ndchi2WdNw = -(dHdchiw * Ndchi1WdNw + dHdNw + dHddelta * NddeltadN1)
    if x3w > 0:
        Ndchi3WdNc = -dGdchic ** -1 * (dGdchiw * Ndchi1WdNc + 0.0 + dGddelta * NddeltadN3)
        Ndchi2WdNc = -(dHdchiw * Ndchi1WdNc + 0.0 + dHddelta * NddeltadN3)
    if x2w > 0:
        Ndchi3WdNh = -dGdchic ** -1 * (dGdchiw * Ndchi1WdNh + 0.0 + dGddelta * NddeltadN2)
        Ndchi2WdNh = -(dHdchiw * Ndchi1WdNh + 0.0 + dHddelta * NddeltadN2)

    dgwdPw = -(gw - 1) * np.cos(THETA1) / (Pw * np.cos(THETA1) + 1)
    if abs(Pww) < 1e-30:
        dgwdPww = 0.0
    else:
        dgwdPww = (gw - 1) / Pww

    NdPwwdN1 = Pww * (1 / x1w + 1 / delta * NddeltadN1 + 2 / chi1w * Ndchi1WdNw)
    if x3w > 0:
        NdPwcdN1 = Pwc * (1 / chi1w * Ndchi1WdNw + 1 / delta * NddeltadN1
                          + 1 / chi3w * Ndchi3WdNw)
    else:
        NdPwcdN1 = 0.0
    if x2w > 0 and _s12_on:
        NdPwhdN1 = Pwh * (1 / chi1w * Ndchi1WdNw + 1 / delta * NddeltadN1
                          + 1 / chi2w * Ndchi2WdNw)
    else:
        NdPwhdN1 = 0.0
    NdgwdN1 = dgwdPw * (NdPwwdN1 + NdPwcdN1 + NdPwhdN1) + dgwdPww * NdPwwdN1

    if x3w > 0:
        NdPwwdN3 = Pww * (2 / chi1w * Ndchi1WdNc + 1 / delta * NddeltadN3)
        NdPwcdN3 = Pwc * (1 / x3w + 1 / chi1w * Ndchi1WdNc
                          + 1 / delta * NddeltadN3 + 1 / chi3w * Ndchi3WdNc)
        if x2w > 0 and _s12_on:
            NdPwhdN3 = Pwh * (1 / chi1w * Ndchi1WdNc + 1 / delta * NddeltadN3
                              + 1 / chi2w * Ndchi2WdNc)
        else:
            NdPwhdN3 = 0.0
        NdgwdN3 = dgwdPw * (NdPwwdN3 + NdPwcdN3 + NdPwhdN3) + dgwdPww * NdPwwdN3
    else:
        NdgwdN3 = 0.0

    if x2w > 0:
        NdPwwdN2 = Pww * (2 / chi1w * Ndchi1WdNh + 1 / delta * NddeltadN2)
        if x3w > 0:
            NdPwcdN2 = Pwc * (1 / chi1w * Ndchi1WdNh + 1 / delta * NddeltadN2
                              + 1 / chi3w * Ndchi3WdNh)
        else:
            NdPwcdN2 = 0.0
        if _s12_on:
            NdPwhdN2 = Pwh * (1 / x2w + 1 / chi1w * Ndchi1WdNh
                              + 1 / delta * NddeltadN2 + 1 / chi2w * Ndchi2WdNh)
        else:
            NdPwhdN2 = 0.0
        NdgwdN2 = dgwdPw * (NdPwwdN2 + NdPwcdN2 + NdPwhdN2) + dgwdPww * NdPwwdN2
    else:
        NdgwdN2 = 0.0

    dFder = (2 * epsr ** 2 + eps_inf ** 2) / (epsr ** 2 * (eps_inf + 2) ** 2)
    dFdeinf = (epsr * eps_inf - 4 * eps_inf - 4 * epsr ** 2 - 2 * epsr) / (epsr * (eps_inf + 2) ** 3)
    # Density-factor chain: d(V/r_d)/dN_k = 0 for the three partitioning
    # species, so each N-chain of a scaled density factor is scaled by r_d.
    r_d = state["r_d"]
    NdeinfdN1 = Na * pol1 * rho * r_d / (eps0 * (1 - M) ** 2)
    NdeinfdN2 = Na * pol2 * rho * r_d / (eps0 * (1 - M) ** 2)
    NdeinfdN3 = Na * pol3 * rho * r_d / (eps0 * (1 - M) ** 2)
    # Dipole-weight chain: w depends on N_w only (N dw/dN_w from the
    # extension; 0 without one).
    w_d = state["w_d"]
    NdwdN1 = state["N_dw_dNw"]
    NdFdN1 = (-Na * rho * r_d * dip01 ** 2 / (9 * eps0 * kb * T)
              * (gw * w_d + x1w * (w_d * NdgwdN1 + gw * NdwdN1)))
    NdFdN2 = -Na * rho * r_d * dip01 ** 2 / (9 * eps0 * kb * T) * (x1w * w_d * NdgwdN2)
    NdFdN3 = -Na * rho * r_d * dip01 ** 2 / (9 * eps0 * kb * T) * (x1w * w_d * NdgwdN3)
    NderdN1 = -(dFder) ** -1 * (dFdeinf * NdeinfdN1 + NdFdN1)
    NderdN2 = -(dFder) ** -1 * (dFdeinf * NdeinfdN2 + NdFdN2)
    NderdN3 = -(dFder) ** -1 * (dFdeinf * NdeinfdN3 + NdFdN3)

    if state["has_ions"]:
        daDHBder = (debye ** 2 / (8 * np.pi * Na * rho * epsr * xiZi)
                    * np.sum(xiw * Zi ** 2 *
                             (debye / (1 + debye * Sgi) - 1 / Rbi)))
    else:
        daDHBder = 0.0
    lnPHI1perm = daDHBder * NderdN1
    lnPHI2perm = daDHBder * NderdN2
    lnPHI3perm = daDHBder * NderdN3

    # Extension contributions (0.0 without an extension).
    l1x, l2x, l3x = state["lnphi_ext"]
    lnPHI1w = state["lnPHI1phys"] + state["lnPHI1assoc"] + lnPHI1perm + l1x
    lnPHI2w = state["lnPHI2phys"] + state["lnPHI2assoc"] + lnPHI2perm + l2x
    lnPHI3w = state["lnPHI3phys"] + state["lnPHI3assoc"] + lnPHI3perm + l3x

    return lnPHI1w, lnPHI2w, lnPHI3w


def lnphi_aq(x_aq, mi, T, P, *, v_init=None, tol=1e-10, maxiter=80):
    """Return ln of the fugacity coefficients of H2O, CH4 and CO2 in the aqueous phase.

    The aqueous state is found by a Newton solve for the compressibility
    factor ``Zw``, the relative permittivity ``epsr`` and the H2O monomer
    fraction ``chi1w``.  If the first solve fails, or gives
    ``chi1w <= 0`` or a permittivity below ``LIQUID_EPSR_MIN``, a fixed
    list of other starting points is tried (plus any seeds added through
    ``ms_ecpa.extensions``).

    Parameters
    ----------
    x_aq : array_like, shape (3,)
        Aqueous mole fractions of (H2O, CH4, CO2), with the ions included
        in the total (so they sum to ``1 - sum(x_ion)``).
    mi : array_like, shape (6,)
        Ion molalities (Na+, Cl-, K+, Ca2+, Mg2+, SO4 2-) [mol/kg water].
    T : float
        Temperature [K].
    P : float
        Pressure [Pa] (not bar; multiply a pressure in bar by 1e5).
    v_init : array_like, shape (3,), optional
        Starting values ``[Zw, epsr, chi1w]``.  The default is
        ``[P * 18.015e-6 / (8.314 T), 50.0, 0.10]``, i.e. the compressibility
        factor of liquid water at a molar volume of 18.015 cm3/mol.
    tol : float
        Newton tolerance on ``max|residual|``.
    maxiter : int
        Maximum number of Newton iterations per starting point.

    Returns
    -------
    tuple of float, length 3
        ``(lnphi_H2O, lnphi_CH4, lnphi_CO2)``, natural logarithms of the
        fugacity coefficients, defined so that
        ``f_i = x_aq,i * exp(lnphi_i) * P``.

    Raises
    ------
    RuntimeError
        If no starting point gives a converged, physical root.
    ValueError
        If ``parameters.Uref_ii`` is not symmetric.
    """
    x_aq = np.asarray(x_aq, dtype=float)
    mi = np.asarray(mi, dtype=float)
    _Uii = _prm.Uref_ii
    if np.any(_Uii) and not np.array_equal(_Uii, _Uii.T):
        raise ValueError(
            "parameters.Uref_ii must be symmetric (DU_ij = DU_ji); use "
            "parameters.set_ion_ion_du() to set entries"
        )
    if v_init is None:
        # Physics-scaled compressibility seed for the aqueous (liquid)
        # branch: Z ≈ P·V_m/(R·T) with V_m = 18.015 cm^3/mol (pure liquid
        # water at ~room T; a good order-of-magnitude for the seed at
        # every T–P in the sweep window).  The prior fixed seed Z=0.05
        # was wrong for near-pure water at low P — at, e.g., T=337 K,
        # P=1 bar the true liquid Z is ~6e-4, and the Newton with Z=0.05
        # converged to a spurious non-liquid root that gave f_water ~4 bar
        # instead of the physical ~0.23 bar (see IAPWS).  That misled TPD
        # into declaring the aqueous phase unstable at conditions where
        # water condensation was actually thermodynamically favoured,
        # producing the discontinuous phase boundary at low T low P
        # across every §S12 panel.  The Zw_liq seed lands in the correct
        # basin at every T–P; the fallback ladder below is unchanged.
        Zw_liq = float(P) * 18.015e-6 / (8.314 * float(T))
        v_init = np.array([Zw_liq, 50.0, 0.10])

    v, ok, _ = newton_solve(
        _aq_residual_3var, v_init, args=(x_aq, mi, T, P),
        tol=tol, maxiter=maxiter,
    )
    # Physical guard: chi1w (v[2]) must be positive.  At high dissolved-gas
    # fractions the Newton can converge to a spurious fixed point with chi1w<0,
    # causing log(chi1w)=NaN in the lnφ expression.  Reject and try next seed.
    #
    # The relative permittivity (v[1]) is guarded on the same footing.  A
    # relative permittivity is bounded below by that of vacuum, so a negative
    # value is not a state of any medium -- it is the Onsager-Kirkwood closure
    # solved outside its domain.  The 3x3 Newton can settle there and satisfy
    # the residual to machine precision, so the solver reports success and the
    # value is consumed without complaint.  Rejecting it sends the seed ladder
    # below looking for a real root, exactly as a negative chi1w does.
    if ok and (v[2] <= 0 or v[1] <= LIQUID_EPSR_MIN):
        ok = False
    if not ok:
        # Fallback ladder: spans the (Zw, epsr, chi1w) range observed across
        # the standard-sweep grid.  The leading seed is physics-based:
        # Zw_liq ≈ P·Vm_liq/(R·T) with Vm_liq = 18.015 cm³/mol (liquid water)
        # scales correctly from P=1 bar (Zw≈0.0008) to 2000 bar (Zw≈1.6) and
        # covers all T=283–623 K.  This single seed fixes the dominant failure
        # class found in sweep iteration 1 (inner Newton missed the liquid
        # basin at low P / low T because [0.05, 50, 0.10] was too large).
        # Seeds with high chi1w (0.80–0.90) catch cases where the low-chi1w
        # seeds converge to a spurious negative-chi1w fixed point.
        _Zw_liq = float(P) * 18.015e-6 / (8.314 * float(T))
        _alts = (np.array([_Zw_liq, 60.0, 0.07]),
                    np.array([_Zw_liq, 50.0, 0.80]),
                    np.array([_Zw_liq, 50.0, 0.90]),
                    np.array([0.07,  60.0, 0.13]),
                    np.array([0.10,  30.0, 0.05]),
                    np.array([0.30,  70.0, 0.20]),
                    np.array([0.20,  40.0, 0.10]),
                    np.array([0.50,  50.0, 0.10]),
                    np.array([0.50,  80.0, 0.50]),
                    np.array([_Zw_liq, 78.0, 0.85]),
                    np.array([_Zw_liq, 78.0, 0.95]),
                    # ── Iter 6 Fix 1: high-P eps_r seeds.
                    # At P=1341-2000 bar the literature relative
                    # permittivity of water reaches ~100-130; the
                    # earlier rungs top out at eps_r=78-80, leaving
                    # the Newton outside the basin in standard-sweep
                    # v5 (regime A1: ~50 failures).
                    np.array([_Zw_liq,  90.0, 0.07]),
                    np.array([_Zw_liq, 100.0, 0.07]),
                    np.array([_Zw_liq, 110.0, 0.07]),
                    np.array([_Zw_liq,  90.0, 0.85]),
                    np.array([_Zw_liq, 100.0, 0.90]),
                    # ── Iter 6 Fix 3: chi1w≈0.97 seeds for 3 m brines.
                    # At m_NaCl/KCl = 3 mol/kg, ionic shielding
                    # suppresses Wertheim association more strongly
                    # than the chi1w=0.85/0.95 rungs cover (regime B1:
                    # ~21 failures in standard-sweep v5).
                    np.array([_Zw_liq, 80.0, 0.97]),
                    np.array([_Zw_liq, 85.0, 0.95]),
                    # ── Iter 7 Fix B: water-poor aqueous seeds.
                    # When the aqueous phase carries 50-75% dissolved
                    # gas (e.g. x_aq=[0.25, 0.74, 0]) Wertheim
                    # association is weak from H2O sparsity, not from
                    # ionic shielding -- so the physical root has low
                    # chi1w AND moderate eps_r (the mixture permittivity
                    # is dragged down by the gas content).  These rungs
                    # target standard-sweep v6 Bucket B (~27 failures
                    # at z=(0.25,0.75,0), (0.50,0.50,0), etc.).
                    np.array([_Zw_liq, 30.0, 0.10]),
                    np.array([_Zw_liq, 40.0, 0.20]),
                    np.array([_Zw_liq, 50.0, 0.30]),
                    # ── Iter 8 Fix E: targeted seeds from observed v7
                    # failure x_aq vectors.  Three patterns:
                    # (1) Water-rich + 3 m salt (x_H2O~0.82, m_tot~6):
                    #     chi1w needs to reach 0.98+ to clear ionic
                    #     shielding at the 3 m level.
                    np.array([_Zw_liq, 75.0, 0.98]),
                    np.array([_Zw_liq, 70.0, 0.99]),
                    # (2) Water-rich + moderate salt (x_H2O~0.78,
                    #     m_tot~1.5-2): the mid-shielding regime
                    #     between the existing 0.85/0.95 and 0.99
                    #     rungs.
                    np.array([_Zw_liq, 70.0, 0.93]),
                    np.array([_Zw_liq, 65.0, 0.95]),
                    # (3) Very water-poor (x_H2O<=0.10): when the
                    #     aqueous phase is mostly dissolved gas, the
                    #     mixture permittivity is dragged toward the
                    #     gas value (~5-15), much lower than the
                    #     30-50 rungs of Fix B.
                    np.array([_Zw_liq,  5.0, 0.05]),
                    np.array([_Zw_liq, 10.0, 0.05]),
                    np.array([_Zw_liq, 15.0, 0.10]),
                    # ── Low-permittivity basin of concentrated 2:1 brines
                    # (eps_r ~ 25-30, chi1w ~ 0.09), between the 15/30
                    # rungs. Reached only after every earlier rung fails.
                    np.array([_Zw_liq, 25.0, 0.12]),
                    # ── Dense concentrated-brine basin (eps_r ~ 40,
                    # chi1w ~ 0.09, Zw ~ 40 % above the pure-water seed).
                    # Reached only after every earlier rung fails.
                    np.array([1.4 * _Zw_liq, 42.0, 0.09]))
        # Seeds an extension adds for its own terms (none by default).
        _alts = _alts + tuple(np.asarray(v, dtype=float) for v in
                              _ext.extra_aqueous_seeds(_Zw_liq))
        for alt in _alts:
            v, ok, _ = newton_solve(_aq_residual_3var, alt,
                                    args=(x_aq, mi, T, P),
                                    tol=tol, maxiter=maxiter)
            if ok and (v[2] <= 0 or v[1] <= LIQUID_EPSR_MIN):
                ok = False  # spurious chi1w<0, or a negative permittivity
            if ok:
                break
    if not ok:
        raise RuntimeError(
            f"lnphi_aq: inner Newton failed to converge (x_aq={x_aq}, "
            f"T={T:.2f}K, P={P/1e5:.2f}bar, mi={mi})"
        )

    lnphi1, lnphi2, lnphi3 = _aq_lnphi_evaluate_3var(v, x_aq, mi, T, P)
    return float(np.real(lnphi1)), float(np.real(lnphi2)), float(np.real(lnphi3))


# ---------------------------------------------------------------------------
# Gas phase
# ---------------------------------------------------------------------------

#: Slack on the monomer-fraction upper bound, to absorb the rounding of a
#: converged Newton root that sits exactly on chi = 1 (the no-association
#: limit).
#: Least relative permittivity that can belong to an AQUEOUS phase.
#:
#: The acceptance test used to be eps_r > 0, which rejects a negative
#: permittivity but not a vapour: the seed ladder can converge on
#: Zw ~ 0.5, eps_r ~ 1.0003, chi ~ 0.998 -- a gas -- and that passes
#: "positive". Water and brine sit between about 25 and 80 over the range
#: this model covers, so 5.0 is far below any real aqueous state and far
#: above any vapour.
#:
#: This is not hypothetical. At 6 mol/kg and 1778 Pa the brine root exists
#: (Zw = 1.71e-05, eps_r = 46.9) and holds about 4 % of the converged seed
#: space, but rung 0 stalls there and rung 4 reaches the vapour first; with
#: only eps_r > 0 to stop it, the vapour was accepted and the brine
#: vapour-pressure curve returned P/P0 = 1.012 -- above pure water, which
#: salt cannot do. With this bound the ladder keeps going and rung 14
#: reaches the brine root.
LIQUID_EPSR_MIN = 5.0

_CHI_UPPER_SLACK = 1e-12


def _gas_assoc_state(Zh, y_gas, T, P):
    """Association coefficients of the gas-phase Wertheim closure at a
    GIVEN ``Zh``.

    Returns ``(B, DELTA, S13, S12A)``, or ``None`` when ``Zh`` is outside
    the domain on which the closure is defined:

    * ``Zh <= B`` — inside the mixture co-volume, where the physical part
      of lnφ takes ``log(Zh - B)``;
    * ``1.9*eta >= 1`` — at or past the pole of the Carnahan-Starling-like
      radial distribution function ``g(eta) = 1/(1 - 1.9 eta)``.

    The algebra is the verbatim gas-phase block of ``_gas_residual``.
    It is repeated here rather than factored out of the residual so that
    ``_gas_residual`` itself stays byte-for-byte unchanged — its Newton
    trajectory is what the Fortran port's golden vectors pin.
    """
    y1, y2, y3 = float(y_gas[0]), float(y_gas[1]), float(y_gas[2])
    Zh = float(Zh)
    b = b1 * y1 + b2 * y2 + b3 * y3
    B = b * float(P) / R / float(T)
    if not np.isfinite(Zh) or Zh <= B:
        return None
    eta = B / 4.0 / Zh
    if 1.0 - 1.9 * eta <= 0.0:
        return None
    g_eta = 1.0 / (1.0 - 1.9 * eta)
    delta = (g_eta * kappaW) * (np.exp(epsW / float(T)) - 1.0)
    DELTA = delta * float(P) / R / float(T)
    S13 = _prm.s13_of_T(float(T))   # one definition; see parameters.s13_of_T
    return B, float(DELTA), float(S13), float(_prm.S12A)


def _gas_chi_triple(Zh, chi1h, y_gas, T, P, state=None):
    """The three monomer fractions ``(chi1h, chi2h, chi3h)`` implied by the
    pair ``(Zh, chi1h)``.  ``None`` outside the closure domain."""
    st = _gas_assoc_state(Zh, y_gas, T, P) if state is None else state
    if st is None:
        return None
    _B, DELTA, S13, S12A_ = st
    y1 = float(y_gas[0])
    Zh, chi1h = float(Zh), float(chi1h)
    den3 = Zh + 2 * y1 * chi1h * S13 * DELTA
    den2 = Zh + 2 * y1 * chi1h * S12A_ * DELTA
    if den3 == 0.0 or den2 == 0.0:
        return None
    return chi1h, Zh / den2, Zh / den3


def _gas_root_is_physical(v, y_gas, T, P):
    """Whether a converged ``(Zh, chi1h)`` pair lies on the physical branch.

    Each of the three species' monomer fractions enters lnφ through
    ``4 log(chi)``, so every one of them must be **strictly positive**.
    The association closure ``chi (Z + 2 y1 chi Delta) = Z`` is a quadratic
    with one positive and one negative root, and at liquid-like ``Zh`` the
    unconstrained 2x2 Newton can settle on the negative one: it satisfies
    the residual to machine precision, so the solver reports success, but
    it is not a physical state and its logarithm is NaN.  This predicate is
    the gas-side mirror of the ``chi1w > 0`` guard ``lnphi_aq`` has
    carried on the aqueous side since the same failure was found there.

    Standard Wertheim theory also bounds a monomer fraction above by 1 —
    but only when every association strength is non-negative.  This model
    lets the H2O-CO2 solvation strength ``S13(T)`` go negative below about
    290 K, and ``chi > 1`` there is the documented extended domain that the
    production three-phase kernel already accepts
    (``ms_ecpa.flash3._solve_gas_kernel``).  So the upper bound is
    asserted only where it is actually a theorem.

    Scope, and what this predicate deliberately does NOT decide.  It tests
    the association closure, not mechanical stability: it asks whether a
    converged pair is *evaluable* (every chi positive, Zh inside the
    closure domain), not whether the root it sits on is the one the caller
    wanted.  Branch selection in ``lnphi_gas`` is by seed --
    ``Z_init=1.0`` for the vapour branch, ``Z_init ~ B_mix`` for a dense
    liquid-CO2 phase -- and that contract is intentional, because the two
    branches are both legitimate answers at the same ``(T, P, y)``.

    One consequence is known and open: inside the near-critical CO2 fold
    (roughly 274-312 K and 45-85 bar at CO2-rich ``y``) the Zh equation has
    three roots, and the middle one has ``dP/dV > 0`` -- a mechanically
    unstable state that is not realisable.  It satisfies every test above,
    so it is accepted here, and at some of those conditions the default
    ``Z_init=1.0`` Newton lands on it.  Rejecting it needs a ``dP/dV``
    test, which changes the returned lnphi at those conditions (measured:
    phi_H2O by up to ~23 %, phi_CO2 by up to ~0.4 %) and therefore moves
    published AADs; it must also be mirrored in ``fortran/src/ecpa_lnphi.f90``
    with the golden vectors regenerated in the same commit.  Deferred to
    the author as a deliberate refit decision rather than applied here.
    """
    Zh, chi1h = float(v[0]), float(v[1])
    if not (np.isfinite(Zh) and np.isfinite(chi1h)):
        return False
    st = _gas_assoc_state(Zh, y_gas, T, P)
    if st is None:
        return False
    _B, DELTA, S13, S12A_ = st
    tri = _gas_chi_triple(Zh, chi1h, y_gas, T, P, state=st)
    if tri is None:
        return False
    if not all(np.isfinite(c) and c > 0.0 for c in tri):
        return False
    if DELTA >= 0.0 and S13 >= 0.0 and S12A_ >= 0.0:
        if not all(c <= 1.0 + _CHI_UPPER_SLACK for c in tri):
            return False
    return True


def _gas_chi_positive_branch(Zh, y_gas, T, P, maxiter=200, tol=1e-14):
    """``chi1h`` on the POSITIVE branch of the association closure at a
    given ``Zh`` — the seed used to pull a mis-branched Newton back.

    With ``Zh`` held fixed the closure collapses to one scalar equation.
    Dropping the two solvation sinks leaves the water-water quadratic
    ``2 y1 Delta c^2 + Z c - Z = 0``, whose positive root is available in
    closed form and, written as ``2Z / (Z + sqrt(Z^2 + 8 y1 Delta Z))``,
    is free of subtractive cancellation and automatically lands in
    ``(0, 1]``.  That root is *exact* whenever ``y_CH4 = y_CO2 = 0`` — the
    regime in which the mis-branching is observed — and a good starting
    point otherwise, so it is refined by damped substitution on the full
    three-species closure, which cannot cross zero.

    Returns ``None`` when ``Zh`` is outside the closure domain or the
    iteration leaves the positive branch.
    """
    st = _gas_assoc_state(Zh, y_gas, T, P)
    if st is None:
        return None
    _B, DELTA, S13, S12A_ = st
    y1, y2, y3 = float(y_gas[0]), float(y_gas[1]), float(y_gas[2])
    Zh = float(Zh)
    u = 2.0 * y1 * DELTA
    if u <= 0.0:
        return 1.0                      # no water-water association
    disc = Zh * Zh + 4.0 * u * Zh          # = Z^2 + 8 y1 Delta Z
    if disc < 0.0:
        return None
    c = 2.0 * Zh / (Zh + np.sqrt(disc))
    if not (np.isfinite(c) and c > 0.0):
        return None
    for _ in range(maxiter):
        tri = _gas_chi_triple(Zh, c, y_gas, T, P, state=st)
        if tri is None:
            break
        _c1, chi2, chi3 = tri
        den = (Zh + u * c + 2.0 * y3 * chi3 * S13 * DELTA
               + 2.0 * y2 * chi2 * S12A_ * DELTA)
        if not (den > 0.0):
            break
        c_new = Zh / den
        if not (np.isfinite(c_new) and c_new > 0.0):
            break
        c_next = 0.5 * (c + c_new)
        done = abs(c_next - c) <= tol * max(1.0, abs(c))
        c = c_next
        if done:
            break
    return float(c) if np.isfinite(c) and c > 0.0 else None


def _gas_accept_or_repair(v, ok, y_gas, T, P, tol, maxiter):
    """Accept a converged gas root only if it is physical; if it is not,
    re-solve once from the SAME ``Zh`` with ``chi1h`` re-seeded on the
    positive branch.  Returns ``(v, ok)``.

    A non-converged input is passed straight through, so a state whose
    Newton merely failed takes exactly the seed ladder it took before this
    guard existed.  The only behaviour that changes is at roots the solver
    previously reported as converged while they sat on the non-physical
    chi branch — precisely the states whose water lnφ was silently NaN.
    """
    if not ok:
        return v, False
    if _gas_root_is_physical(v, y_gas, T, P):
        return v, True
    Zh = float(v[0])
    c0 = _gas_chi_positive_branch(Zh, y_gas, T, P)
    if c0 is not None:
        v2, ok2, _ = newton_solve(
            _gas_residual, np.array([Zh, c0]), args=(y_gas, T, P),
            tol=tol, maxiter=maxiter,
        )
        if ok2 and _gas_root_is_physical(v2, y_gas, T, P):
            return v2, True
    return v, False


def _gas_residual(v, y_gas, T, P):
    """Gas-phase fixed-point residual.

    Variables: ``[Zh, chi1h]`` (2 unknowns).
    Residuals: ``(Zh - Zh_new)/Zh``, ``(chi1h - chi1h_new)/chi1h``.
    """
    Zh, chi1h = v
    y1, y2, y3 = y_gas

    a1 = a01 * (1 + c11 * (1 - (T / Tc1) ** 0.5)) ** 2
    a2 = a02 * (1 + c12 * (1 - (T / Tc2) ** 0.5)) ** 2
    a3 = a03 * (1 + c13 * (1 - (T / Tc3) ** 0.5)) ** 2
    k13 = Ak13 * (T / Tc3) ** 2 + Bk13 * (T / Tc3) + Ck13
    S13 = _prm.s13_of_T(T)          # one definition; see parameters.s13_of_T
    k12 = Ak12 * (T / Tc1) ** 2 + Bk12 * (T / Tc1) + Ck12
    a12 = (a1 * a2) ** 0.5 * (1 - k12)
    a13 = (a1 * a3) ** 0.5 * (1 - k13)
    a23 = (a2 * a3) ** 0.5 * (1 - k23)

    b = b1 * y1 + b2 * y2 + b3 * y3
    a = (a1 * y1 ** 2 + 2 * a12 * y1 * y2 + a2 * y2 ** 2
         + 2 * a13 * y1 * y3 + a3 * y3 ** 2 + 2 * a23 * y2 * y3)
    A = a * P / R ** 2 / T ** 2
    B = b * P / R / T
    Zphys = Zh / (Zh - B) - A / (Zh + B)

    eta = B / 4 / Zh
    g_eta = 1 / (1 - 1.9 * eta)
    dg_deta = 1.9 / (1 - 1.9 * eta) ** 2
    delta = (g_eta * kappaW) * (np.exp(epsW / T) - 1)
    DELTA = delta * P / R / T
    S12A_ = _prm.S12A
    chi3h = Zh / (Zh + 2 * y1 * chi1h * S13 * DELTA)
    chi2h = Zh / (Zh + 2 * y1 * chi1h * S12A_ * DELTA)
    chi1h_new = Zh / (Zh + 2 * y1 * chi1h * DELTA + 2 * y3 * chi3h * S13 * DELTA
                      + 2 * y2 * chi2h * S12A_ * DELTA)
    Zassoc = -2 * (1 + eta / g_eta * dg_deta) * (y1 * (1 - chi1h) + y3 * (1 - chi3h)
                                                 + y2 * (1 - chi2h))
    Zh_new = Zphys + Zassoc

    f1 = (Zh - Zh_new) / Zh
    f2 = (chi1h - chi1h_new) / chi1h
    return [f1, f2]


def _gas_lnphi_evaluate(v, y_gas, T, P):
    """Evaluate gas-phase lnφ for the three partitioning species.

    Mirrors the gas-phase block of ``residual_elv`` exactly.
    """
    Zh, chi1h = v
    y1, y2, y3 = y_gas

    a1 = a01 * (1 + c11 * (1 - (T / Tc1) ** 0.5)) ** 2
    a2 = a02 * (1 + c12 * (1 - (T / Tc2) ** 0.5)) ** 2
    a3 = a03 * (1 + c13 * (1 - (T / Tc3) ** 0.5)) ** 2
    k13 = Ak13 * (T / Tc3) ** 2 + Bk13 * (T / Tc3) + Ck13
    S13 = _prm.s13_of_T(T)          # one definition; see parameters.s13_of_T
    k12 = Ak12 * (T / Tc1) ** 2 + Bk12 * (T / Tc1) + Ck12
    a12 = (a1 * a2) ** 0.5 * (1 - k12)
    a13 = (a1 * a3) ** 0.5 * (1 - k13)
    a23 = (a2 * a3) ** 0.5 * (1 - k23)
    A1 = a1 * P / R ** 2 / T ** 2
    B1 = b1 * P / R / T
    A2 = a2 * P / R ** 2 / T ** 2
    B2 = b2 * P / R / T
    A3 = a3 * P / R ** 2 / T ** 2
    B3 = b3 * P / R / T
    A12 = a12 * P / R ** 2 / T ** 2
    A13 = a13 * P / R ** 2 / T ** 2
    A23 = a23 * P / R ** 2 / T ** 2

    b = b1 * y1 + b2 * y2 + b3 * y3
    a = (a1 * y1 ** 2 + 2 * a12 * y1 * y2 + a2 * y2 ** 2
         + 2 * a13 * y1 * y3 + a3 * y3 ** 2 + 2 * a23 * y2 * y3)
    A = a * P / R ** 2 / T ** 2
    B = b * P / R / T

    lnPHI1phys = (-np.log(Zh - B) + B1 / B * (B / (Zh - B) - A / (Zh + B))
                  + A / B * (B1 / B - 2 * (y1 * A1 + y2 * A12 + y3 * A13) / A)
                  * np.log(1 + B / Zh))
    lnPHI2phys = (-np.log(Zh - B) + B2 / B * (B / (Zh - B) - A / (Zh + B))
                  + A / B * (B2 / B - 2 * (y1 * A12 + y2 * A2 + y3 * A23) / A)
                  * np.log(1 + B / Zh))
    lnPHI3phys = (-np.log(Zh - B) + B3 / B * (B / (Zh - B) - A / (Zh + B))
                  + A / B * (B3 / B - 2 * (y1 * A13 + y2 * A23 + y3 * A3) / A)
                  * np.log(1 + B / Zh))

    eta = B / 4 / Zh
    g_eta = 1 / (1 - 1.9 * eta)
    dg_deta = 1.9 / (1 - 1.9 * eta) ** 2
    delta = (g_eta * kappaW) * (np.exp(epsW / T) - 1)
    DELTA = delta * P / R / T
    S12A_ = _prm.S12A
    chi3h = Zh / (Zh + 2 * y1 * chi1h * S13 * DELTA)
    chi2h = Zh / (Zh + 2 * y1 * chi1h * S12A_ * DELTA)
    # Last line of defence: every chi below feeds a logarithm.  A
    # non-physical (non-positive) monomer fraction must never be turned
    # into a silent NaN lnφ — every caller already screens its root with
    # _gas_root_is_physical, so reaching this raise means a new call site
    # skipped that screen.
    for _name, _c in (("chi1h", chi1h), ("chi2h", chi2h), ("chi3h", chi3h)):
        if not (np.isfinite(np.real(_c)) and np.real(_c) > 0.0):
            raise ValueError(
                f"_gas_lnphi_evaluate: non-physical {_name}="
                f"{np.real(_c)!r} at Zh={np.real(Zh)!r}, y_gas="
                f"{np.real(np.asarray(y_gas)).tolist()}, T={float(T):.2f}K, "
                f"P={float(P)/1e5:.2f}bar; 4*log({_name}) is undefined"
            )
    _assoc_sum = (y1 * (chi1h - 1) + y3 * (chi3h - 1)
                  + y2 * (chi2h - 1))
    lnPHI1assoc = (4 * np.log(chi1h)
                   + B1 / (2 * g_eta * Zh) * dg_deta * _assoc_sum)
    lnPHI2assoc = (4 * np.log(chi2h)
                   + B2 / (2 * g_eta * Zh) * dg_deta * _assoc_sum)
    lnPHI3assoc = (4 * np.log(chi3h)
                   + B3 / (2 * g_eta * Zh) * dg_deta * _assoc_sum)

    return (lnPHI1phys + lnPHI1assoc,
            lnPHI2phys + lnPHI2assoc,
            lnPHI3phys + lnPHI3assoc)


_GAS_NESTED_NGRID = 240
_GAS_NESTED_ZMAX = 3.0
_GAS_NESTED_BISECT = 80


def _gas_nested_residual(Zh, y_gas, T, P):
    """Scalar EOS residual with ``chi1h`` eliminated on the positive branch.

    At fixed ``Zh`` the association closure is one scalar equation whose
    positive root ``_gas_chi_positive_branch`` returns in closed form
    (refined by a substitution that cannot cross zero).  Substituting it
    leaves a single equation in ``Zh`` alone.  Returns NaN where ``Zh`` is
    outside the closure domain.
    """
    c = _gas_chi_positive_branch(float(Zh), y_gas, T, P)
    if c is None:
        return float("nan")
    r = _gas_residual(np.array([float(Zh), float(c)]), y_gas, T, P)[0]
    r = float(np.real(r))
    return r if np.isfinite(r) else float("nan")


def _gas_nested_root(y_gas, T, P):
    """Last-resort gas root: bracket-and-bisect on ``Zh`` alone.

    Why this exists.  The production path is a 2x2 Newton in
    ``(Zh, chi1h)``.  As ``y_H2O -> 0`` the association closure degenerates,
    because ``chi1h`` enters it only multiplied by ``y_H2O``: the Jacobian
    column for ``chi1h`` vanishes, the step in that direction blows up, and
    the Newton either diverges or lands on the negative branch.  The Zh
    direction is perfectly well conditioned throughout -- only the chi
    direction is sick.  Eliminating ``chi1h`` analytically therefore removes
    the whole failure mode rather than papering over it, and in the limit
    ``y_H2O = 0`` it returns the exact answer ``chi1h = 1`` (no water, hence
    no association) instead of a singular system.

    This matters at ``T < 290 K``, where ``S13(T) < 0`` puts the closure on
    its extended (anti-solvation) domain and folds the root locus, so no
    fixed seed ladder in ``(Zh, chi1h)`` covers every composition.

    Root selection.  Over ``Zh`` in ``(B_mix, 3]`` the nested residual is
    normally single-rooted, but inside the fold window it admits three.
    When that happens the root of lowest molar Gibbs energy
    ``sum_i y_i ln phi_i`` is returned -- the standard criterion, the same
    one used to choose between cubic roots anywhere else.

    Returns ``(v, True)`` with ``v = [Zh, chi1h]``, or ``(None, False)``.
    """
    y = np.asarray(y_gas, dtype=float)
    b_mix = b1 * float(y[0]) + b2 * float(y[1]) + b3 * float(y[2])
    B = b_mix * float(P) / (R * float(T))
    lo = B * 1.0001 + 1e-9
    hi = _GAS_NESTED_ZMAX
    if not (np.isfinite(lo) and lo < hi):
        return None, False

    xs = np.linspace(lo, hi, _GAS_NESTED_NGRID)
    fs = np.array([_gas_nested_residual(x, y, T, P) for x in xs])

    roots = []
    for a, b, fa, fb in zip(xs[:-1], xs[1:], fs[:-1], fs[1:]):
        if not (np.isfinite(fa) and np.isfinite(fb)):
            continue
        if fa == 0.0:
            roots.append(float(a))
            continue
        if fa * fb >= 0.0:
            continue
        lo_i, hi_i, f_lo = float(a), float(b), float(fa)
        for _ in range(_GAS_NESTED_BISECT):
            mid = 0.5 * (lo_i + hi_i)
            f_mid = _gas_nested_residual(mid, y, T, P)
            if not np.isfinite(f_mid):
                break
            if f_lo * f_mid <= 0.0:
                hi_i = mid
            else:
                lo_i, f_lo = mid, f_mid
        roots.append(0.5 * (lo_i + hi_i))

    best, best_g = None, None
    for Zr in roots:
        c = _gas_chi_positive_branch(Zr, y, T, P)
        if c is None:
            continue
        v = np.array([float(Zr), float(c)])
        if not _gas_root_is_physical(v, y, T, P):
            continue
        try:
            ls = _gas_lnphi_evaluate(v, y, T, P)
        except Exception:
            continue
        g = sum(float(y[i]) * float(np.real(ls[i])) for i in range(3))
        if not np.isfinite(g):
            continue
        if best_g is None or g < best_g:
            best, best_g = v, g
    if best is None:
        return None, False
    return best, True


def lnphi_gas(y_gas, T, P, *, Z_init=None, chi_init=0.99, tol=1e-10, maxiter=80):
    """Return ln of the fugacity coefficients of H2O, CH4 and CO2 in the gas phase.

    The gas state is found by a Newton solve for the compressibility factor
    ``Zh`` and the H2O monomer fraction ``chi1h``.  If that solve fails or
    gives a non-physical root, further starting values of ``Zh`` are tried,
    and finally a bisection on ``Zh`` alone.

    Parameters
    ----------
    y_gas : array_like, shape (3,)
        Gas-phase mole fractions of (H2O, CH4, CO2).
    T : float
        Temperature [K].
    P : float
        Pressure [Pa] (not bar).
    Z_init : float, optional
        Starting compressibility factor.  The default 1.0 selects the
        gas-like root.  For a dense (liquid-CO2) phase, start near the
        reduced co-volume, ``Z_init ≈ B_mix``.
    chi_init : float
        Starting value of ``chi1h``.
    tol : float
        Newton tolerance on ``max|residual|``.
    maxiter : int
        Maximum number of Newton iterations per starting point.

    Returns
    -------
    tuple of float, length 3
        ``(lnphi_H2O, lnphi_CH4, lnphi_CO2)``, natural logarithms of the
        fugacity coefficients, defined so that
        ``f_i = y_i * exp(lnphi_i) * P``.

    Raises
    ------
    RuntimeError
        If no physical root (``chi > 0``) is found.

    Notes
    -----
    The physical (SRK) term gives one or three real roots in ``Zh``.  The
    root found is normally the one nearest ``Z_init``.  Ions are not present
    in the gas phase.
    """
    y_gas = np.asarray(y_gas, dtype=float)
    if Z_init is None:
        Z_init = 1.0
    v_init = np.array([float(Z_init), float(chi_init)])
    v, ok, _ = newton_solve(
        _gas_residual, v_init, args=(y_gas, T, P),
        tol=tol, maxiter=maxiter,
    )
    # Physical guard, mirroring the chi1w>0 guard in lnphi_aq: a root the
    # Newton reports as converged is accepted only when it lies on the
    # physical chi branch.  If it converged onto the negative branch the
    # root is repaired in place (same Zh, chi1h re-seeded positive);
    # otherwise the seed ladder below runs exactly as it did before.
    v, ok = _gas_accept_or_repair(v, ok, y_gas, T, P, tol, maxiter)
    if not ok:
        # Fallback ladder: at high P the reduced co-volume B_mix can exceed 1
        # (dense / supercritical gas), making the seed Zh=1.0 fall inside the
        # unphysical region Zh < B_mix and causing the Newton to blow up.
        # Pre-compute the B-scaled seed Zh_B = B_mix * 1.2 + 0.3 so the
        # starting point is safely above B_mix for any P.
        y1, y2, y3 = float(y_gas[0]), float(y_gas[1]), float(y_gas[2])
        b_mix = b1 * y1 + b2 * y2 + b3 * y3
        Zh_B = b_mix * float(P) / (R * float(T)) * 1.2 + 0.3
        # Liquid-CO2 seed: at T < T_c,CO2 (304 K) and P > P_sat the "gas"
        # phase is actually liquid CO2; its Z ≈ 2·B_mix.  Without this seed
        # all gas-like starting points (Z=0.3–3.0) miss the liquid root.
        Zh_liq_co2 = b_mix * float(P) / (R * float(T)) * 2.0
        for alt_Zh in (max(Zh_B, 1.5), max(Zh_B * 1.3, 2.0), 3.0, 0.7, 0.5, 0.3,
                       Zh_liq_co2, b_mix * float(P) / (R * float(T)) * 1.5):
            v, ok, _ = newton_solve(
                _gas_residual, np.array([alt_Zh, float(chi_init)]),
                args=(y_gas, T, P), tol=tol, maxiter=maxiter,
            )
            v, ok = _gas_accept_or_repair(v, ok, y_gas, T, P, tol, maxiter)
            if ok:
                break
    if not ok:
        # Last resort: eliminate chi1h analytically and bisect on Zh alone.
        # Purely additive -- reached only where the code above would have
        # raised, so no state that already solved can change.
        v_n, ok_n = _gas_nested_root(y_gas, T, P)
        if ok_n:
            v, ok = v_n, True
    if not ok:
        raise RuntimeError(
            f"lnphi_gas: inner Newton failed to converge on a physical "
            f"root (y_gas={y_gas}, T={T:.2f}K, P={P/1e5:.2f}bar). "
            f"Either no seed converged, or every converged root sat on "
            f"the non-physical (chi<=0) branch of the association "
            f"closure and could not be repaired."
        )
    l1, l2, l3 = _gas_lnphi_evaluate(v, y_gas, T, P)
    return float(np.real(l1)), float(np.real(l2)), float(np.real(l3))
