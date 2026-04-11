"""
ELV residual system and Jacobian for the eCPA EoS.

ELV(x0, T, P, ms, params=None) -> 10-element residual vector.
ELV_jac(x0, T, P, ms, params=None) -> 10×10 Jacobian (complex-step).

The 10 unknowns are:
  0  Zw          aqueous-phase compressibility
  1  x1w         H₂O mol-frac (aqueous)
  2  epsr        relative permittivity
  3  Zc          CO₂-rich-phase compressibility
  4  x1c         H₂O mol-frac (CO₂-rich)
  5  chi1w       H₂O association fraction (aqueous)
  6  chi1c       H₂O association fraction (CO₂-rich)
  7  Ndchi1WdNw  ∂(N·χ₁W)/∂N_w at constant T,V,N_others
  8  Ndchi1WdNc  ∂(N·χ₁W)/∂N_c
  9  Vdchi1WdV   V·∂χ₁W/∂V

Parameters override
-------------------
`params` is an optional dict {name: value} of scalar overrides.  When
provided, the named variables are temporarily injected into this module's
global namespace during the ELV call and restored on exit.  All constants
from ecpa.constants must therefore be importable into this namespace with
`from .constants import *`.
"""
import numpy as np
# Make all constants available as module-level globals so that the
# params-override mechanism (globals() mutation) works correctly.
from .constants import *  # noqa: F401,F403

# ── Configuration ──────────────────────────────────────────────────────────────
USE_COMPLEX_JAC = False   # True → exact complex-step Jacobian; False → FD (default)


# ── Residual ───────────────────────────────────────────────────────────────────
def ELV(x0, T, P, ms, params=None):
    """
    Complex-safe eCPA ELV residual.

    x0 may be real or complex (for Jacobian computation).
    T, P, ms are always real.
    """
    def _denom_sym(a, b, eps=1e-30):
        return max(abs(a.real), abs(b.real), eps)

    def _rel_err(a, b, scale=1.0, eps=1e-30):
        denom = max(abs(float(scale)), abs(a.real), eps)
        return (a - b) / denom

    x0 = np.asarray(x0)
    T  = np.float64(T)
    P  = np.float64(P)
    ms = np.float64(ms)

    # ── Temporarily inject params into this module's globals ──────────────────
    _saved = None
    if params is not None:
        _saved = {}
        g = globals()
        for k, v in params.items():
            if k in g:
                _saved[k] = g[k]
            g[k] = v

    try:
        Zw, x1w, epsr, Zc, x1c, chi1w, chi1c, Ndchi1WdNw, Ndchi1WdNc, Vdchi1WdV = x0

        # PARAMETERS (T-dependent)
        k14 = Akij*(T/Tc4)**2 + Bkij*(T/Tc4) + Ckij
        S14 = ASij*(T/Tc4)**2 + BSij*(T/Tc4) + CSij
        U4s = Uref4s + alfa4s*R*((1-T/Talfa4s)**2 - (1-Tref/Talfa4s)**2)

        # ── AQUEOUS PHASE ─────────────────────────────────────────────────────
        x2w = x1w*ms*Mw
        x3w = x2w
        x4w = 1 - x1w - x2w - x3w

        rho = P/Zw/R/T
        a1  = a01*(1 + c11*(1 - (T/Tc1)**0.5))**2
        a4  = a04*(1 + c14*(1 - (T/Tc4)**0.5))**2
        b   = b1*x1w + b2*x2w + b3*x3w + b4*x4w
        U1s = Uref1s + alfa1s*R*((1-T/Talfa1s)**2 - (1-Tref/Talfa1s)**2)
        U14 = np.log(2)*(a4/b4 - 2*(a1*a4)**0.5*(1-k14)/(b1+b4))
        U41 = np.log(2)*(a1/b1 - 2*(a4*a1)**0.5*(1-k14)/(b4+b1))
        gE  = (1/b)*(x1w*x2w*U1s*(b1+b2) + x1w*x3w*U1s*(b1+b3)
                     + x4w*x2w*U4s*(b4+b2) + x4w*x3w*U4s*(b4+b3)
                     + x1w*x4w*(b1*U14 + b4*U41))
        a   = b*(x1w*a1/b1 + x2w*a02/b2 + x3w*a03/b3 + x4w*a4/b4 - gE/np.log(2))

        A   = a*P/R**2/T**2;  B = b*P/R/T
        A1  = a1*P/R**2/T**2; B1 = b1*P/R/T
        B2  = b2*P/R/T;       B3 = b3*P/R/T
        A4  = a4*P/R**2/T**2; B4 = b4*P/R/T

        Zphys     = Zw/(Zw-B) - A/(Zw+B)
        lnPHI1phys = (-np.log(Zw-B)
                      + B1/B*(B/(Zw-B) - A/(Zw+B))
                      - np.log((Zw+B)/Zw)*(A1/B1
                        - 1/(B*np.log(2))*(x2w*U1s/R/T*(B1+B2)
                                           + x3w*U1s/R/T*(B1+B3)
                                           + x4w/R/T*(B1*U14+B4*U41)
                                           - B1*gE/R/T)))
        lnPHI4phys = (-np.log(Zw-B)
                      + B4/B*(B/(Zw-B) - A/(Zw+B))
                      - np.log((Zw+B)/Zw)*(A4/B4
                        - 1/(B*np.log(2))*(x1w/R/T*(B1*U14+B4*U41)
                                           + x2w*U4s/R/T*(B4+B2)
                                           + x3w*U4s/R/T*(B4+B3)
                                           - B4*gE/R/T)))

        # Association (aqueous)
        eta   = B/4/Zw
        g_eta = 1/(1-1.9*eta)
        dg_deta = 1.9/(1-1.9*eta)**2
        delta = (g_eta*kappaW)*(np.exp(epsW/T) - 1)
        DELTA = delta*P/R/T
        chi4w      = Zw/(Zw + 2*x1w*chi1w*S14*DELTA)
        chi1w_new  = Zw/(Zw + 2*x1w*chi1w*DELTA + 2*x4w*chi4w*S14*DELTA)
        Zassoc     = -2*(1 + eta/g_eta*dg_deta)*(x1w*(1-chi1w) + x4w*(1-chi4w))
        lnPHI1assoc = (4*np.log(chi1w)
                       + B1/(8*g_eta*Zw)*dg_deta*(x1w*4*(chi1w-1) + x4w*4*(chi4w-1)))
        lnPHI4assoc = (4*np.log(chi4w)
                       + B4/(8*g_eta*Zw)*dg_deta*(x1w*4*(chi1w-1) + x4w*4*(chi4w-1)))

        # Debye–Hückel (aqueous)
        xiZi  = x2w*Z2**2 + x3w*Z2**2
        debye = (e**2*Na*rho*xiZi/(kb*T*epsr*eps0))**0.5
        X2    = 1/Sg2**3*(np.log(1+debye*Sg2) - debye*Sg2 + 0.5*(debye*Sg2)**2)
        X3    = 1/Sg3**3*(np.log(1+debye*Sg3) - debye*Sg3 + 0.5*(debye*Sg3)**2)
        lnPHI1dh = 0
        lnPHI4dh = 0
        if ms > 0:
            ZDH = (1/(4*np.pi*Na*rho*xiZi)
                   * (x2w*Z2**2*(X2 - 0.5*debye**3/(1+debye*Sg2))
                      + x3w*Z3**2*(X3 - 0.5*debye**3/(1+debye*Sg3))))
        else:
            ZDH = 0

        # Permittivity
        rho1  = rho*x1w
        M     = Na*rho/(3*eps0)*(x1w*pol1 + x2w*pol2 + x3w*pol3 + x4w*pol4)
        eps_inf = (2*M+1)/(1-M)
        Pww   = 2*rho*x1w*delta*chi1w**2
        Pwc   = 2*rho*x4w*S14*delta*chi1w*chi4w
        Pw    = Pww + Pwc
        gw    = 1 + zww*Pww*np.cos(GAMMA1)/(Pw*np.cos(THETA1)+1)
        T1    = (2*epsr + eps_inf)*(epsr - eps_inf)/(epsr*(eps_inf + 2)**2)
        T2    = Na*rho/(9*eps0*kb*T)*(x1w*gw*dip01**2)

        dFdchiw  = 1 + 2*x1w*rho*delta*chi1w**2
        dFdchic  = 2*x2w*rho*S14*delta*chi4w**2
        dGdchiw  = 2*x1w*rho*S14*delta*chi1w**2
        dGdchic  = 1
        dFdV     = -(2*rho*x1w*delta*chi1w + 2*rho*x2w*S14*delta*chi4w)*chi1w**2
        dGdV     = -(2*rho*x1w*S14*delta*chi1w)*chi1w**2
        dFdNw    = 2*rho*delta*chi1w**3
        dGdNw    = 2*rho*S14*delta*chi1w**3
        dFdNc    = 2*rho*S14*delta*chi1w**2*chi4w
        dGdNc    = 0
        VddeltadV   = -delta*eta*dg_deta/g_eta
        NddeltadN1  = delta*eta*b1*dg_deta/(g_eta*b)
        NddeltadN4  = delta*eta*b4*dg_deta/(g_eta*b)
        dFddelta = -chi1w**2*(1 + (delta-1)*(2*rho*x1w*chi1w + 2*rho*x4w*chi4w*S14))
        dGddelta = -chi4w**2*(1 + (delta-1)*(2*rho*x1w*chi1w*S14))

        Vdchi4WdV   = -dGdchic**-1*(dGdchiw*Vdchi1WdV  + dGdV  + dGddelta*VddeltadV)
        Ndchi4WdNw  = -dGdchic**-1*(dGdchiw*Ndchi1WdNw + dGdNw + dGddelta*NddeltadN1)
        Ndchi4WdNc  = -dGdchic**-1*(dGdchiw*Ndchi1WdNc + dGdNc + dGddelta*NddeltadN4)
        Vdchi1WdV_new  = -dFdchiw**-1*(dFdchic*Vdchi4WdV  + dFdV  + dFddelta*VddeltadV)
        Ndchi1WdNw_new = -dFdchiw**-1*(dFdchic*Ndchi4WdNw + dFdNw + dFddelta*NddeltadN1)
        Ndchi1WdNc_new = -dFdchiw**-1*(dFdchic*Ndchi4WdNc + dFdNc + dFddelta*NddeltadN4)

        dgwdPw   = -(gw-1)*np.cos(THETA1)/(Pw*np.cos(THETA1)+1)
        dgwdPww  = (gw-1)/Pww
        VdPwwdV  = Pww*(-1 + 1/delta*VddeltadV  + 2/chi1w*Vdchi1WdV)
        VdPwcdV  = Pwc*(-1 + 1/delta*VddeltadV  + 1/chi1w*Vdchi1WdV + 1/chi4w*Vdchi4WdV)
        NdPwwdN1 = Pww*(1/x1w + 1/delta*NddeltadN1 + 2/chi1w*Ndchi1WdNw)
        NdPwcdN1 = Pwc*(1/chi1w*Ndchi1WdNw + 1/delta*NddeltadN1 + 1/chi4w*Ndchi4WdNw)
        NdPwwdN4 = Pww*(2/chi1w*Ndchi1WdNc + 1/delta*NddeltadN4)
        NdPwcdN4 = Pwc*(1/x4w + 1/chi1w*Ndchi1WdNc + 1/delta*NddeltadN4 + 1/chi4w*Ndchi4WdNc)
        VdPwdV   = VdPwwdV + VdPwcdV
        NdPwdN1  = NdPwwdN1 + NdPwcdN1
        NdPwdN4  = NdPwwdN4 + NdPwcdN4
        VdgwdV   = dgwdPw*VdPwdV  + dgwdPww*VdPwwdV
        NdgwdN1  = dgwdPw*NdPwdN1 + dgwdPww*NdPwwdN1
        NdgwdN4  = dgwdPw*NdPwdN4 + dgwdPww*NdPwwdN4

        dFder    = (2*epsr**2 + eps_inf**2)/(epsr**2*(eps_inf + 2)**2)
        dFdeinf  = (epsr*eps_inf - 4*eps_inf - 4*epsr**2 - 2*epsr)/(epsr*(eps_inf + 2)**3)
        VdeinfdV   = -3*M/(1-M)**2
        NdeinfdN1  = Na*pol1*rho/(eps0*(1-M)**2)
        NdeinfdN4  = Na*pol4*rho/(eps0*(1-M)**2)
        VdFdV    = Na*rho1*dip01**2/(9*eps0*kb*T)*(gw - VdgwdV)
        NdFdN1   = -Na*rho*dip01**2/(9*eps0*kb*T)*(gw + x1w*NdgwdN1)
        NdFdN4   = -Na*rho*dip01**2/(9*eps0*kb*T)*(x1w*NdgwdN4)
        VderdV   = -(dFder)**-1*(dFdeinf*VdeinfdV + VdFdV)
        NderdN1  = -(dFder)**-1*(dFdeinf*NdeinfdN1 + NdFdN1)
        NderdN4  = -(dFder)**-1*(dFdeinf*NdeinfdN4 + NdFdN4)

        if ms > 0:
            daDHBder = (debye**2/(8*np.pi*Na*rho*epsr*xiZi)
                        * (x2w*Z2**2*(debye/(1+debye*Sg2) - 1/Rb2)
                           + x3w*Z3**2*(debye/(1+debye*Sg3) - 1/Rb3)))
        else:
            daDHBder = 0

        ZPerm      = -daDHBder*VderdV
        lnPHI1perm =  daDHBder*NderdN1
        lnPHI4perm =  daDHBder*NderdN4

        Zw_new   = Zphys + Zassoc + ZDH + ZPerm          # Born = 0
        lnPHI1w  = lnPHI1phys + lnPHI1assoc + lnPHI1perm # DH/Born = 0 for H₂O fugacity
        lnPHI4w  = lnPHI4phys + lnPHI4assoc + lnPHI4perm
        f1w = np.exp(lnPHI1w)*P*x1w*1e-5
        f4w = np.exp(lnPHI4w)*P*x4w*1e-5

        # ── CO₂-RICH PHASE ────────────────────────────────────────────────────
        x4c = 1 - x1c
        rho = P/Zc/R/T
        b   = b1*x1c + b4*x4c
        a14 = (a1*a4)**0.5*(1-k14)
        a   = x1c**2*a1 + 2*x1c*x4c*a14 + x4c**2*a4
        A   = a*P/R**2/T**2; B = b*P/R/T
        A14 = a14*P/R**2/T**2

        Zphys     = Zc/(Zc-B) - A/(Zc+B)
        lnPHI1phys = (-np.log(Zc-B) + B1/B*(B/(Zc-B) - A/(Zc+B))
                      + A/B*(B1/B - 2*(x1c*A1 + x4c*A14)/A)*np.log(1+B/Zc))
        lnPHI4phys = (-np.log(Zc-B) + B4/B*(B/(Zc-B) - A/(Zc+B))
                      + A/B*(B4/B - 2*(x1c*A14 + x4c*A4)/A)*np.log(1+B/Zc))

        eta   = B/4/Zc
        g_eta = 1/(1-1.9*eta)
        dg_deta = 1.9/(1-1.9*eta)**2
        delta = (g_eta*kappaW)*(np.exp(epsW/T) - 1)
        DELTA = delta*P/R/T
        chi4c     = Zc/(Zc + 2*x1c*chi1c*S14*DELTA)
        chi1c_new = Zc/(Zc + 2*x1c*chi1c*DELTA + 2*x4c*chi4c*S14*DELTA)
        Zassoc    = -2*(1 + eta/g_eta*dg_deta)*(x1c*(1-chi1c) + x4c*(1-chi4c))
        lnPHI1assoc = (4*np.log(chi1c)
                       + B1/(8*g_eta*Zc)*dg_deta*(x1c*4*(chi1c-1) + x4c*4*(chi4c-1)))
        lnPHI4assoc = (4*np.log(chi4c)
                       + B4/(8*g_eta*Zc)*dg_deta*(x1c*4*(chi1c-1) + x4c*4*(chi4c-1)))

        Zc_new  = Zphys + Zassoc
        lnPHI1c = lnPHI1phys + lnPHI1assoc
        lnPHI4c = lnPHI4phys + lnPHI4assoc
        f1c = np.exp(lnPHI1c)*P*x1c*1e-5
        f4c = np.exp(lnPHI4c)*P*x4c*1e-5

        # ── Residuals ─────────────────────────────────────────────────────────
        f1_x  = _rel_err(Zw,         Zw_new)
        f2_x  = _rel_err(T1,         T2)
        f3_x  = _rel_err(Zc,         Zc_new)
        f4_x  = (f1w - f1c) / _denom_sym(f1w, f1c)
        f5_x  = (f4w - f4c) / _denom_sym(f4w, f4c)
        f6_x  = _rel_err(chi1w,      chi1w_new)
        f7_x  = _rel_err(chi1c,      chi1c_new)
        f8_x  = _rel_err(Ndchi1WdNw, Ndchi1WdNw_new)
        f9_x  = _rel_err(Ndchi1WdNc, Ndchi1WdNc_new)
        f10_x = _rel_err(Vdchi1WdV,  Vdchi1WdV_new)

        return np.array([f1_x, f2_x, f3_x, f4_x, f5_x,
                         f6_x, f7_x, f8_x, f9_x, f10_x])

    finally:
        if _saved is not None:
            g = globals()
            for k in list(params.keys()):
                if k in _saved:
                    g[k] = _saved[k]
                elif k in g:
                    del g[k]


# ── Jacobian ───────────────────────────────────────────────────────────────────
def ELV_jac(x0, T, P, ms, params=None):
    """
    Exact 10×10 Jacobian of ELV via complex-step differentiation.
    10× more accurate than finite differences at the same cost (10 ELV calls).
    """
    h  = 1e-20
    x0 = np.asarray(x0, dtype=complex)
    n  = len(x0)
    J  = np.zeros((n, n), dtype=float)
    for i in range(n):
        xp    = x0.copy()
        xp[i] += 1j * h
        J[:, i] = ELV(xp, T, P, ms, params).imag / h
    return J
