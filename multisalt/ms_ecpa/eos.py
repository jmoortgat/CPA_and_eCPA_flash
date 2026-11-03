"""eCPA equation of state: residuals, fugacities, and equilibrium driver.

The ``residual_elv`` and ``gas_fugacities`` function bodies are preserved
verbatim from the original ``eCPA_ELV.py`` (only renamed); all model
parameters are imported from :mod:`ms_ecpa.parameters`.
"""
from dataclasses import dataclass, asdict

import numpy as np
from scipy.optimize import fsolve

from .parameters import *  # noqa: F401,F403  (model constants by name)
from . import parameters as _prm  # live module ref (runtime-togglable flags)
from . import extensions as _ext
from .guess import build_initial_guess, T0, Tf
from .newton import newton_solve


def residual_elv(x0, T, P, Yc_fw, mi):
    if Yc_fw == 0:
        Zw, x2w,  epsr, Zh,  x1h, chi1w, chi1h, Ndchi1WdNw, Ndchi1WdNh,   Vdchi1WdV   = x0
        x3w = 0
    elif Yc_fw == 1:
        Zw, x3w, epsr, Zh,  x1h, chi1w, chi1h, Ndchi1WdNw,  Ndchi1WdNc, Vdchi1WdV   = x0
        x2w = 0
    else:
        Zw, x2w, x3w, epsr, Zh,  x1h, chi1w, chi1h, Ndchi1WdNw, Ndchi1WdNh,  Ndchi1WdNc, Vdchi1WdV   = x0

    #PARAMETERS
    a1 = a01*(1 + c11*(1 - (T/Tc1)**0.5))**2
    a2 = a02*(1 + c12*(1 - (T/Tc2)**0.5))**2
    a3 = a03*(1 + c13*(1 - (T/Tc3)**0.5))**2
    ai = a0i*(1 + c1i*(1 - (T/Tref)**0.5))**2
    k13 = (Ak13*(T/Tc3)**2 + Bk13*(T/Tc3) + Ck13)
    S13 = _prm.s13_of_T(T)          # one definition; see parameters.s13_of_T
    k12 = Ak12*(T/Tc1)**2 + Bk12*(T/Tc1) + Ck12
    a12 = (a1*a2)**0.5*(1-k12)
    a13 = (a1*a3)**0.5*(1-k13)
    a23 = (a2*a3)**0.5*(1-k23)
    A1 = a1*P/R**2/T**2
    B1 = b1*P/R/T
    A2 = a2*P/R**2/T**2
    B2 = b2*P/R/T
    A3 = a3*P/R**2/T**2
    B3 = b3*P/R/T
    Bi = bi*P/R/T
    A12 = a12*P/R**2/T**2
    A13 = a13*P/R**2/T**2
    A23 = a23*P/R**2/T**2
    U1i = Uref1i + alfa1i*R*((1-T/Talfa1i)**2 - (1-Tref/Talfa1i)**2)
    U2i = A_U2i*(T/Tc2) + B_U2i
    U3i = A_U3i*(T/Tc3) + B_U3i
    U12 = np.log(2)*(a2/b2-2*(a1*a2)**0.5*(1-k12)/(b1+b2))
    U21 = np.log(2)*(a1/b1-2*(a2*a1)**0.5*(1-k12)/(b2+b1))
    U13 = np.log(2)*(a3/b3-2*(a1*a3)**0.5*(1-k13)/(b1+b3))
    U31 = np.log(2)*(a1/b1-2*(a3*a1)**0.5*(1-k13)/(b3+b1))
    U23 = np.log(2)*(a3/b3-2*(a2*a3)**0.5*(1-k23)/(b2+b3))
    U32 = np.log(2)*(a2/b2-2*(a3*a2)**0.5*(1-k23)/(b3+b2))

    #AQUEOUS PHASE
    x1w = (1-x2w-x3w)/(1 + sum(mi)*Mw)
    xiw = x1w*Mw*mi

    #Physical
    rho = P/Zw/R/T     
    b = b1*x1w + b2*x2w + b3*x3w + sum(bi*xiw)
    gE = 1/b*(x1w*x2w*(b1*U12 + b2*U21)  + x1w*x3w*(b1*U13 + b3*U31) + x2w*x3w*(b2*U23 + b3*U32) + x1w*sum(xiw*U1i*(b1+bi)) + x2w*sum(xiw*U2i*(b2+bi)) + x3w*sum(xiw*U3i*(b3+bi)))
    # Ion-ion HV energies (production = all zeros).  Published
    # double sum extended to the ion block: symmetric DU_ij adds
    # x_i x_j (b_i + b_j) DU_ij / b per unordered ion pair.  No new explicit
    # lnPHI terms needed (DU_kj = 0 for the partitioning species k): the term
    # enters only through gE / a below.  All-zero matrix -> analytically zero,
    # branch skipped, bit-exact baseline.  Mirrors ms_ecpa.lnphi._aq_core_state.
    if np.any(_prm.Uref_ii):
        gE = gE + np.dot(xiw, _prm.Uref_ii @ (bi * xiw)) / b
    a = b*(x1w*a1/b1 + x2w*a2/b2 + x3w*a3/b3 + sum(xiw*ai/bi) - gE/np.log(2))
    A = a*P/R**2/T**2
    B = b*P/R/T
    Zphys = Zw/(Zw-B) - A/(Zw+B)
    lnPHI1phys = -np.log(Zw-B) + B1/B*(B/(Zw-B) - A/(Zw+B)) - np.log((Zw+B)/Zw)*(A1/B1 - 1/(B*np.log(2))*(x2w/R/T*(B1*U12+B2*U21) + x3w/R/T*(B1*U13+B3*U31) + sum(xiw*U1i/R/T*(B1+Bi)) - B1*gE/R/T))
    lnPHI2phys = -np.log(Zw-B) + B2/B*(B/(Zw-B) - A/(Zw+B)) - np.log((Zw+B)/Zw)*(A2/B2 - 1/(B*np.log(2))*(x1w/R/T*(B1*U12+B2*U21) + x3w/R/T*(B2*U23+B3*U32) + sum(xiw*U2i/R/T*(B2+Bi)) - B2*gE/R/T))
    lnPHI3phys = -np.log(Zw-B) + B3/B*(B/(Zw-B) - A/(Zw+B)) - np.log((Zw+B)/Zw)*(A3/B3 - 1/(B*np.log(2))*(x1w/R/T*(B1*U13+B3*U31) + x2w/R/T*(B3*U32+B2*U23) + sum(xiw*U3i/R/T*(B3+Bi)) - B3*gE/R/T))

    #Association
    eta = B/4/Zw 
    g_eta = 1/(1-1.9*eta)
    dg_deta =  1.9/(1-1.9*eta)**2
    delta = (g_eta*kappaW)*(np.exp(epsW/T) - 1)
    DELTA = delta*P/R/T
    chi3w = Zw/(Zw + 2*x1w*chi1w*S13*DELTA)
    chi1w_new = Zw/(Zw + 2*x1w*chi1w*DELTA + 2*x3w*chi3w*S13*DELTA)
    Zassoc = -2*(1 + eta/g_eta*dg_deta)*(x1w*(1-chi1w) + x3w*(1-chi3w))
    lnPHI1assoc = 4*np.log(chi1w) + B1/(2*g_eta*Zw)*dg_deta*(x1w*(chi1w - 1) + x3w*(chi3w - 1))
    lnPHI2assoc = B2/(2*g_eta*Zw)*dg_deta*(x1w*(chi1w - 1) + x3w*(chi3w - 1))
    lnPHI3assoc = 4*np.log(chi3w) + B3/(2*g_eta*Zw)*dg_deta*(x1w*(chi1w - 1) + x3w*(chi3w - 1))

    #DH
    xiZi = sum(xiw*Zi**2)
    debye = (e**2*Na*rho*xiZi/(kb*T*epsr*eps0))**0.5 
    Xi = 1/Sgi**3*(np.log(1+debye*Sgi) -debye*Sgi+ 0.5*(debye*Sgi)**2)
    lnPHI1dh = 0
    lnPHI2dh = 0
    lnPHI3dh = 0
    if sum(mi) > 0:
        ZDH = 1/(4*np.pi*Na*rho*xiZi) * sum(xiw*Zi**2*(Xi - 0.5*debye**3/(1+debye*Sgi)))
    else:
        ZDH = 0
        
    #Born
    ZBorn = 0
    lnPHI1born = 0
    lnPHI2born = 0
    lnPHI3born = 0
    
    #Perm
    #epsr
    rho1 = rho*x1w
    # Extension terms (ms_ecpa.extensions); neutral constants without one.
    # Mirrors ms_ecpa.lnphi._aq_core_state.
    ext = _ext.aqueous_terms(T, rho, x1w, x2w, x3w, xiw) if sum(mi) > 0 \
        else _ext.NEUTRAL
    r_d = ext.density_factor
    M = Na*rho*r_d/(3*eps0)*(x1w*pol1 + x2w*pol2 + x3w*pol3 + sum(xiw*poli))
    eps_inf = (2*M+1)/(1-M)
    Pww = 2*rho*x1w*delta*chi1w**2
    Pwc = 2*rho*x3w*S13*delta*chi1w*chi3w
    Pw = Pww + Pwc
    gw = 1 + zww*Pww*np.cos(GAMMA1)/(Pw*np.cos(THETA1)+1)
    w_d = ext.dipole_weight
    Z_ext = ext.Z
    l1x, l2x, l3x = ext.lnphi
    VdwdV, NdwdN1 = ext.V_dw_dV, ext.N_dw_dNw
    T1 = (2*epsr + eps_inf)*(epsr - eps_inf)/(epsr*(eps_inf + 2)**2)
    T2 = Na*rho*r_d/(9*eps0*kb*T)*(x1w*w_d*gw*dip01**2)
    
    #dCHIi/dPHI           PHI = V, Nw, Nc
    # Each row carries chi_new**2 of its OWN residual (F -> chi1w, G ->
    # chi3w), and d/ddelta is +chi**2*dQ/ddelta.  Corrected 2026-09; ground
    # truth in tests/test_chi_chain_derivatives.py.  Must stay in lockstep
    # with lnphi.py (tests/test_lnphi_consistency.py enforces it).
    dFdchiw = 1 + 2*x1w*rho*delta*chi1w**2
    dFdchic = 2*x3w*rho*S13*delta*chi1w**2
    dGdchiw = 2*x1w*rho*S13*delta*chi3w**2
    dGdchic = 1
    dFdV = -(2*rho*x1w*delta*chi1w + 2*rho*x3w*S13*delta*chi3w)*chi1w**2
    dGdV = -(2*rho*x1w*S13*delta*chi1w)*chi3w**2
    dFdNw = 2*rho*delta*chi1w**3
    dGdNw = 2*rho*S13*delta*chi1w*chi3w**2
    VddeltadV = -delta*eta*dg_deta/g_eta
    NddeltadN1 = delta*eta*b1*dg_deta/(g_eta*b)
    dFddelta = chi1w**2*(2*rho*x1w*chi1w + 2*rho*x3w*chi3w*S13)
    dGddelta = chi3w**2*(2*rho*x1w*chi1w*S13)
    Vdchi3WdV = -dGdchic**-1*(dGdchiw*Vdchi1WdV + dGdV + dGddelta*VddeltadV)
    Ndchi3WdNw = -dGdchic**-1*(dGdchiw*Ndchi1WdNw + dGdNw + dGddelta*NddeltadN1)
    Vdchi1WdV_new = -dFdchiw**-1*(dFdchic*Vdchi3WdV + dFdV + dFddelta*VddeltadV)
    Ndchi1WdNw_new = -dFdchiw**-1*(dFdchic*Ndchi3WdNw + dFdNw + dFddelta*NddeltadN1)        
    
    #dgw/dPHI           PHI = V, Nw, Nc
    dgwdPw = -(gw-1)*np.cos(THETA1)/(Pw*np.cos(THETA1)+1)
    dgwdPww = (gw-1)/Pww
    VdPwwdV = Pww*(-1 + 1/delta*VddeltadV + 2/chi1w*Vdchi1WdV)
    VdPwcdV = Pwc*(-1 + 1/delta*VddeltadV + 1/chi1w*Vdchi1WdV + 1/chi3w*Vdchi3WdV)
    NdPwwdN1 = Pww*(1/x1w + 1/(delta)*NddeltadN1 + 2/chi1w*Ndchi1WdNw)
    NdPwcdN1 = Pwc*(1/chi1w*Ndchi1WdNw + 1/(delta)*NddeltadN1 + 1/chi3w*Ndchi3WdNw)
    VdPwdV = VdPwwdV + VdPwcdV
    NdPwdN1 = NdPwwdN1 + NdPwcdN1
    VdgwdV = dgwdPw*VdPwdV + dgwdPww*VdPwwdV
    NdgwdN1 = dgwdPw*NdPwdN1 + dgwdPww*NdPwwdN1

    #depsr/dPHI           PHI = V, Nw, Nc
    dFder = (2*epsr**2 + eps_inf**2)/(epsr**2*(eps_inf + 2)**2)
    dFdeinf = (epsr*eps_inf - 4*eps_inf - 4*epsr**2 - 2*epsr)/(epsr*(eps_inf + 2)**3)
    # Density-factor and dipole-weight chains (see lnphi.py); r_d = 1.0,
    # w_d = 1.0, VdwdV = NdwdN1 = 0.0 without an extension.
    VdeinfdV = -3*M*r_d/(1-M)**2
    NdeinfdN1 = Na*pol1*rho*r_d/(eps0*(1-M)**2)
    VdFdV = Na*rho1*r_d*dip01**2/(9*eps0*kb*T)*(w_d*(gw*r_d -VdgwdV) - gw*VdwdV)
    NdFdN1 = -Na*rho*r_d*dip01**2/(9*eps0*kb*T)*(gw*w_d + x1w*(w_d*NdgwdN1 + gw*NdwdN1))
    VderdV = -(dFder)**-1*(dFdeinf*VdeinfdV + VdFdV)
    NderdN1 = -(dFder)**-1*(dFdeinf*NdeinfdN1 + NdFdN1)
    
    if Yc_fw > 0:
        dFdNc = 2*rho*S13*delta*chi1w**2*chi3w
        dGdNc = 0
        NddeltadN3 = delta*eta*b3*dg_deta/(g_eta*b)
        Ndchi3WdNc = -dGdchic**-1*(dGdchiw*Ndchi1WdNc + dGdNc + dGddelta*NddeltadN3)
        Ndchi1WdNc_new = -dFdchiw**-1*(dFdchic*Ndchi3WdNc + dFdNc + dFddelta*NddeltadN3)
        NdPwwdN3 = Pww*(2/chi1w*Ndchi1WdNc + 1/(delta)*NddeltadN3)
        NdPwcdN3 = Pwc*(1/x3w + 1/chi1w*Ndchi1WdNc + 1/(delta)*NddeltadN3 + 1/chi3w*Ndchi3WdNc)
        NdPwdN3 = NdPwwdN3 + NdPwcdN3
        NdgwdN3 = dgwdPw*NdPwdN3 + dgwdPww*NdPwwdN3
        NdFdN3 = -Na*rho*r_d*dip01**2/(9*eps0*kb*T)*(x1w*w_d*NdgwdN3)
        NdeinfdN3 = Na*pol3*rho*r_d/(eps0*(1-M)**2)
        NderdN3 = -(dFder)**-1*(dFdeinf*NdeinfdN3 + NdFdN3)
    else:
        NderdN3 = 0        
        
    if Yc_fw < 1:
        dFdNh = 0
        dGdNh = 0
        NddeltadN2 = delta*eta*b2*dg_deta/(g_eta*b)
        Ndchi3WdNh = -dGdchic**-1*(dGdchiw*Ndchi1WdNh + dGdNh + dGddelta*NddeltadN2)
        Ndchi1WdNh_new = -dFdchiw**-1*(dFdchic*Ndchi3WdNh + dFdNh + dFddelta*NddeltadN2)
        NdPwwdN2 = Pww*(2/chi1w*Ndchi1WdNh + 1/(delta)*NddeltadN2)
        NdPwcdN2 = Pwc*(1/chi1w*Ndchi1WdNh + 1/(delta)*NddeltadN2 + 1/chi3w*Ndchi3WdNh)
        NdPwdN2 = NdPwwdN2 + NdPwcdN2
        NdgwdN2 = dgwdPw*NdPwdN2 + dgwdPww*NdPwwdN2
        NdeinfdN2 = Na*pol2*rho*r_d/(eps0*(1-M)**2)
        NdFdN2 = -Na*rho*r_d*dip01**2/(9*eps0*kb*T)*(x1w*w_d*NdgwdN2)
        NderdN2 = -(dFder)**-1*(dFdeinf*NdeinfdN2 + NdFdN2)
    else:
        NderdN2 = 0
    
    if sum(mi) > 0:
        daDHBder = debye**2/(8*np.pi*Na*rho*epsr*xiZi)*sum(xiw*Zi**2*(debye/(1+debye*Sgi)-1/Rbi))
    else:
        daDHBder = 0
    ZPerm = -daDHBder*VderdV
    lnPHI1perm = daDHBder*NderdN1
    lnPHI2perm = daDHBder*NderdN2
    lnPHI3perm = daDHBder*NderdN3    
    
    #total - aqueous phase
    # (Z_ext / l*x: extension contributions, 0.0 without an extension)
    Zw_new = Zphys + Zassoc + ZDH + ZBorn + ZPerm + Z_ext
    lnPHI1w = lnPHI1phys + lnPHI1assoc + lnPHI1dh + lnPHI1born + lnPHI1perm + l1x
    lnPHI2w = lnPHI2phys + lnPHI2assoc + lnPHI2dh + lnPHI2born + lnPHI2perm + l2x
    lnPHI3w = lnPHI3phys + lnPHI3assoc + lnPHI3dh + lnPHI3born + lnPHI3perm + l3x
    f1w = np.exp(lnPHI1w)*P*x1w*1e-5                     #bar
    f2w = np.exp(lnPHI2w)*P*x2w*1e-5                     #bar
    f3w = np.exp(lnPHI3w)*P*x3w*1e-5                     #bar

    
    #GAS PHASE
    x3h = Yc_fw*(1-x1h)
    x2h = 1 - x1h - x3h
    
    #Physical
    b = b1*x1h + b2*x2h + b3*x3h
    a = a1*x1h**2 + 2*a12*x1h*x2h + a2*x2h**2 + 2*a13*x1h*x3h + a3*x3h**2 + 2*a23*x2h*x3h 
    A = a*P/R**2/T**2
    B = b*P/R/T
    Zphys = Zh/(Zh-B) - A/(Zh+B)
    lnPHI1phys = -np.log(Zh-B) + B1/B*(B/(Zh-B)-A/(Zh+B)) + A/B*(B1/B - 2*(x1h*A1 + x2h*A12 + x3h*A13)/A)*np.log(1+B/Zh) 
    lnPHI2phys = -np.log(Zh-B) + B2/B*(B/(Zh-B)-A/(Zh+B)) + A/B*(B2/B - 2*(x1h*A12 + x2h*A2 + x3h*A23)/A)*np.log(1+B/Zh) 
    lnPHI3phys = -np.log(Zh-B) + B3/B*(B/(Zh-B)-A/(Zh+B)) + A/B*(B3/B - 2*(x1h*A13 + x2h*A23 + x3h*A3)/A)*np.log(1+B/Zh) 

    #Association
    eta = B/4/Zh 
    g_eta = 1/(1-1.9*eta)
    dg_deta =  1.9/(1-1.9*eta)**2
    delta = (g_eta*kappaW)*(np.exp(epsW/T) - 1)
    DELTA = delta*P/R/T
    chi3h = Zh/(Zh + 2*x1h*chi1h*S13*DELTA)
    chi1h_new = Zh/(Zh + 2*x1h*chi1h*DELTA + 2*x3h*chi3h*S13*DELTA)
    Zassoc = -2*(1 + eta/g_eta*dg_deta)*(x1h*(1-chi1h) + x3h*(1-chi3h))
    lnPHI1assoc = 4*np.log(chi1h) + B1/(2*g_eta*Zh)*dg_deta*(x1h*(chi1h - 1) + x3h*(chi3h - 1))
    lnPHI2assoc = B2/(2*g_eta*Zh)*dg_deta*(x1h*(chi1h - 1) + x3h*(chi3h - 1))
    lnPHI3assoc = 4*np.log(chi3h) + B3/(2*g_eta*Zh)*dg_deta*(x1h*(chi1h - 1) + x3h*(chi3h - 1))
    
    #total - gas phase
    Zh_new = Zphys + Zassoc
    lnPHI1h = lnPHI1phys + lnPHI1assoc
    lnPHI2h = lnPHI2phys + lnPHI2assoc
    lnPHI3h = lnPHI3phys + lnPHI3assoc
    f1h = np.exp(lnPHI1h)*P*x1h*1e-5                     #bar
    f2h = np.exp(lnPHI2h)*P*x2h*1e-5                     #bar
    f3h = np.exp(lnPHI3h)*P*x3h*1e-5                     #bar

    
    #Objective functions
    f1_x = (Zw - Zw_new)/Zw
    f2_x = (chi1w-chi1w_new)/chi1w
    f3_x = (Zh - Zh_new)/Zh
    f4_x = (chi1h-chi1h_new)/chi1h
    f5_x = (f1w - f1h)/f1w
    f8_x = (Ndchi1WdNw-Ndchi1WdNw_new)/Ndchi1WdNw
    f11_x = (Vdchi1WdV-Vdchi1WdV_new)/Vdchi1WdV
    f12_x = (T1 - T2)/T2
    if Yc_fw > 0:
        f7_x = (f3w - f3h)/f3w
        f10_x = (Ndchi1WdNc-Ndchi1WdNc_new)/Ndchi1WdNc
    if Yc_fw < 1:
        f6_x = (f2w - f2h)
        f9_x = (Ndchi1WdNh-Ndchi1WdNh_new)/Ndchi1WdNh
    
    if Yc_fw == 0:
        return [f1_x, f2_x, f3_x, f4_x, f5_x, f6_x, f8_x, f9_x, f11_x, f12_x]
    if Yc_fw == 1:
        return [f1_x, f2_x, f3_x, f4_x, f5_x, f7_x, f8_x, f10_x, f11_x, f12_x]
    else:
        return [f1_x, f2_x, f3_x, f4_x, f5_x, f6_x, f7_x, f8_x, f9_x, f10_x, f11_x, f12_x]
    
    

def gas_fugacities(x0, T, P, Yc_fw):
    if Yc_fw == 0:
        Zw, x2w,  epsr, Zh,  x1h, chi1w, chi1h, Ndchi1WdNw, Ndchi1WdNh,   Vdchi1WdV   = x0
        x3w = 0
        #Ndchi1WdNc = 0
    elif Yc_fw == 1:
        Zw, x3w, epsr, Zh,  x1h, chi1w, chi1h, Ndchi1WdNw,  Ndchi1WdNc, Vdchi1WdV   = x0
        x2w = 0
        #Ndchi1WdNh = 0
    else:
        Zw, x2w, x3w, epsr, Zh,  x1h, chi1w, chi1h, Ndchi1WdNw, Ndchi1WdNh,  Ndchi1WdNc, Vdchi1WdV   = x0

    #PARAMETERS
    a1 = a01*(1 + c11*(1 - (T/Tc1)**0.5))**2
    a2 = a02*(1 + c12*(1 - (T/Tc2)**0.5))**2
    a3 = a03*(1 + c13*(1 - (T/Tc3)**0.5))**2
    k13 = (Ak13*(T/Tc3)**2 + Bk13*(T/Tc3) + Ck13)
    S13 = _prm.s13_of_T(T)          # one definition; see parameters.s13_of_T
    k12 = Ak12*(T/Tc1)**2 + Bk12*(T/Tc1) + Ck12
    a12 = (a1*a2)**0.5*(1-k12)
    a13 = (a1*a3)**0.5*(1-k13)
    a23 = (a2*a3)**0.5*(1-k23)
    A1 = a1*P/R**2/T**2
    B1 = b1*P/R/T
    A2 = a2*P/R**2/T**2
    B2 = b2*P/R/T
    A3 = a3*P/R**2/T**2
    B3 = b3*P/R/T
    A12 = a12*P/R**2/T**2
    A13 = a13*P/R**2/T**2
    A23 = a23*P/R**2/T**2

    #GAS PHASE
    x3h = Yc_fw*(1-x1h)
    x2h = 1 - x1h - x3h
    
    #Physical
    b = b1*x1h + b2*x2h + b3*x3h
    a = a1*x1h**2 + 2*a12*x1h*x2h + a2*x2h**2 + 2*a13*x1h*x3h + a3*x3h**2 + 2*a23*x2h*x3h 
    A = a*P/R**2/T**2
    B = b*P/R/T
    lnPHI1phys = -np.log(Zh-B) + B1/B*(B/(Zh-B)-A/(Zh+B)) + A/B*(B1/B - 2*(x1h*A1 + x2h*A12 + x3h*A13)/A)*np.log(1+B/Zh) 
    lnPHI2phys = -np.log(Zh-B) + B2/B*(B/(Zh-B)-A/(Zh+B)) + A/B*(B2/B - 2*(x1h*A12 + x2h*A2 + x3h*A23)/A)*np.log(1+B/Zh) 
    lnPHI3phys = -np.log(Zh-B) + B3/B*(B/(Zh-B)-A/(Zh+B)) + A/B*(B3/B - 2*(x1h*A13 + x2h*A23 + x3h*A3)/A)*np.log(1+B/Zh) 

    #Association
    eta = B/4/Zh 
    g_eta = 1/(1-1.9*eta)
    dg_deta =  1.9/(1-1.9*eta)**2
    delta = (g_eta*kappaW)*(np.exp(epsW/T) - 1)
    DELTA = delta*P/R/T
    chi3h = Zh/(Zh + 2*x1h*chi1h*S13*DELTA)
    lnPHI1assoc = 4*np.log(chi1h) + B1/(2*g_eta*Zh)*dg_deta*(x1h*(chi1h - 1) + x3h*(chi3h - 1))
    lnPHI2assoc = B2/(2*g_eta*Zh)*dg_deta*(x1h*(chi1h - 1) + x3h*(chi3h - 1))
    lnPHI3assoc = 4*np.log(chi3h) + B3/(2*g_eta*Zh)*dg_deta*(x1h*(chi1h - 1) + x3h*(chi3h - 1))
    
    #total - gas phase
    lnPHI1h = lnPHI1phys + lnPHI1assoc
    lnPHI2h = lnPHI2phys + lnPHI2assoc
    lnPHI3h = lnPHI3phys + lnPHI3assoc
    f1h = np.exp(lnPHI1h)*P*x1h*1e-5                     #bar
    f2h = np.exp(lnPHI2h)*P*x2h*1e-5                     #bar
    f3h = np.exp(lnPHI3h)*P*x3h*1e-5                     #bar
    
    return f1h, f2h, f3h



P0 = 1
Pf = 1500


@dataclass
class ELVResult:
    """Result of a converged (or attempted) eCPA equilibrium calculation."""
    T: float          # K
    P: float          # bar
    YC: float         # water-free CO2 mole fraction in the gas phase
    mi: tuple         # ion molalities (Na, Cl, K, Ca, Mg, SO4) [mol/kg water]
    converged: bool   # solver status flag -- see `residual` below
    ier: int
    # System / salinity
    salinity: float   # total ionic molality [mol/kg]
    ws: float         # ion mass fraction [%]
    xs: float         # ion mole fraction [%]
    # Aqueous phase
    Zw: float
    rho_w: float      # kg/m3
    eps_r: float      # dielectric constant
    x1w: float        # water mole fraction (aqueous)
    x2w: float        # CH4 mole fraction (aqueous)
    x3w: float        # CO2 mole fraction (aqueous)
    mh: float         # CH4 molality [mol/kg]
    wh: float         # CH4 mass fraction [%]
    mc: float         # CO2 molality [mol/kg]
    wc: float         # CO2 mass fraction [%]
    # Gas phase
    Zh: float
    rho_h: float      # kg/m3
    x1h: float        # water mole fraction (gas)
    x2h: float        # CH4 mole fraction (gas)
    x3h: float        # CO2 mole fraction (gas)
    # Fugacities [bar]
    f_h2o: float
    f_ch4: float
    f_co2: float
    # Raw fsolve solution vector
    solution: object = None
    # max|residual_elv| at `solution`. `converged` is only the solver's own
    # status flag: for solver='newton' that flag *is* max|F| < tol, but for
    # the default solver='fsolve' it is MINPACK hybrd's ier == 1, which tests
    # the step between the last two iterates and says nothing about how far
    # from zero the residual actually is. A caller that needs a residual
    # guarantee must test this field rather than `converged`.
    residual: float = float("nan")

    def as_dict(self):
        return asdict(self)


def _solve_residual(guess, T, P, YC, m, solver, tol, maxiter, xtol=None):
    """Dispatch a single equilibrium solve. Returns ``(solution, ier)`` where
    ``ier == 1`` denotes success (matching the original fsolve convention).

    For ``solver='newton'`` that status is ``max|F| < tol``. For
    ``solver='fsolve'`` it is MINPACK hybrd's step test between the last two
    iterates, which ``tol`` does not control; ``xtol``, when given, tightens
    that step test (MINPACK's default 1.49e-8 applies when it is None).
    Either way ``solve_elv`` reports the residual at the returned vector in
    ``ELVResult.residual``.
    """
    if solver == "newton":
        sol, ok, _ = newton_solve(
            residual_elv, guess, args=(T, P * 1e5, YC, m), tol=tol, maxiter=maxiter
        )
        return sol, (1 if ok else 0)
    if solver == "fsolve":
        kw = {} if xtol is None else {"xtol": float(xtol)}
        sol, _info, ier, _msg = fsolve(
            residual_elv, guess, args=(T, P * 1e5, YC, m), full_output=True, **kw
        )
        return sol, ier
    raise ValueError(f"unknown solver {solver!r} (use 'newton' or 'fsolve')")


# Loosest bounds that still separate the water-rich aqueous root from the
# trivial solute-rich one. Over a 630-state scan of the whole validity window
# (T 290-528 K, P 5-1400 bar, YC 0/0.5/1, six ion vectors; 14508 rung solves)
# every converged vector had eps_r in [16.3, 85.2] and x2w + x3w <= 0.16, so
# these sit a factor of ~3 away from anything the model actually produces.
_EPSR_MIN = 5.0
_XSOLUTE_MAX = 0.5


def _on_aqueous_branch(x, YC):
    """True when solution vector ``x`` is still on the water-rich root.

    The residual system also admits a trivial root in which the "aqueous"
    phase is the solute-rich one, with a gas-like permittivity (eps_r ~ 1)
    and x3w ~ 1. The solvers report success on that root exactly as they do
    on the physical one, so without this test a salinity rung that jumps
    branches is handed on as the next warm start and the endpoint inherits
    the wrong branch while still reporting converged=True.
    """
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)):
        return False
    if YC == 0:
        Zw, x2w, epsr = x[0], x[1], x[2]
        x3w = 0.0
    elif YC == 1:
        Zw, x3w, epsr = x[0], x[1], x[2]
        x2w = 0.0
    else:
        Zw, x2w, x3w, epsr = x[0], x[1], x[2], x[3]
    return bool(Zw > 0.0 and epsr > _EPSR_MIN
                and (x2w + x3w) < _XSOLUTE_MAX)


def _continuation_newton(guess, T, P, YC, mi, tol, maxiter, dm):
    """Adaptive salinity homotopy m(s) = s * mi for the Newton solver.

    Starts from the salt-free solution and advances the ionic strength with a
    step that doubles after easy solves and halves on failures. Follows the
    same proportional-scaling path as the original fixed ramp, so it converges
    to the same root, but typically in far fewer steps. Returns (solution, ier).
    """
    mion = float(np.sum(mi))
    sol, ok, _ = newton_solve(
        residual_elv, guess, args=(T, P * 1e5, YC, 0 * mi), tol=tol, maxiter=maxiter
    )
    if not ok:
        return sol, 0

    s = 0.0
    ds = min(1.0, dm / mion)  # start at the legacy step; grow only when easy
    ds_min = 1e-4
    while s < 1.0 - 1e-12:
        s_try = min(1.0, s + ds)
        cand, ok, nit = newton_solve(
            residual_elv, sol, args=(T, P * 1e5, YC, s_try * mi),
            tol=tol, maxiter=maxiter,
        )
        if ok and np.all(np.isfinite(cand)):
            s, sol = s_try, cand
            if nit <= 2:
                ds = min(1.6 * ds, 1.0)  # accelerate on easy steps
        else:
            ds *= 0.5
            if ds < ds_min:
                return sol, 0  # could not advance to target salinity
    return sol, 1


def solve_elv(T, P, YC, mi, guess_dir=None, dm=0.2,
              solver="fsolve", tol=1e-10, maxiter=100, xtol=None):
    """Compute the H2O-CH4-CO2-ions phase equilibrium with the eCPA EoS.

    Non-interactive replacement for the original script's ``__main__`` block.

    Parameters
    ----------
    T : float          absolute temperature [K], 283 < T < 533
    P : float          pressure [bar], 1 < P < 1500
    YC : float         water-free CO2 mole fraction in the gas phase, in [0, 1]
    mi : sequence      ion molalities (Na, Cl, K, Ca, Mg, SO4) [mol/kg water]
    guess_dir : path-like, optional   directory holding the binary guess tables
    dm : float         salinity-continuation step [mol/kg] (default 0.2)
    solver : str       'fsolve' (legacy MINPACK, default; fastest here) or
                       'newton' (damped Newton + complex-step Jacobian; more
                       accurate, provides an analytical-precision Jacobian, but
                       ~3x slower per solve in pure Python)
    tol : float        convergence tolerance on max|residual| (newton only)
    maxiter : int      max Newton iterations per solve (newton only)
    xtol : float, optional   step tolerance handed to fsolve (fsolve only).
                       MINPACK's default 1.49e-8 applies when omitted; pass a
                       smaller value to polish the root (1e-10 to 1e-12 buys
                       3 to 6 orders on the residual here; below ~1e-13
                       MINPACK gives up with ier = 3). It tightens the step
                       test, not the residual -- read ``ELVResult.residual``
                       to see what the residual at the answer actually is.

    Returns
    -------
    ELVResult
    """
    mi = np.asarray(mi, dtype=float)
    if mi.shape != (6,):
        raise ValueError("mi must have 6 entries: (Na, Cl, K, Ca, Mg, SO4)")
    if np.any(_prm.Uref_ii) and not np.array_equal(_prm.Uref_ii,
                                                   _prm.Uref_ii.T):
        raise ValueError(
            "parameters.Uref_ii must be symmetric (DU_ij = DU_ji); use "
            "parameters.set_ion_ion_du() to set entries"
        )
    if _prm.S12A != 0.0:
        raise ValueError(
            "parameters.S12A is nonzero, but this kernel does not implement "
            "slot-2 water cross-association. eos.py (residual_elv/solve_elv/"
            "gas_fugacities) carries no chi2w terms at all, while lnphi.py "
            "implements them fully and reads _prm.S12A live on every call. "
            "At the production S12A = 0 the two kernels are bit-identical, "
            "so this guard never fires in production -- but under a nonzero "
            "S12A this one would silently evaluate the S12A = 0 physics and "
            "report agreement. Route through lnphi_aq/lnphi_gas (which "
            "flash_kv and stability_flash already do), or port the chi2w "
            "terms into this kernel."
        )
    if not (T0 < T < Tf):
        raise ValueError(f"Temperature must satisfy {T0} < T < {Tf} K")
    if not (P0 < P < Pf):
        raise ValueError(f"Pressure must satisfy {P0} < P < {Pf} bar")
    if not (0 <= YC <= 1):
        raise ValueError("YC (water-free CO2 mole fraction) must be in [0, 1]")
    Zm = float(np.sum(mi * Zi))
    if abs(Zm) > 1e-8:
        raise ValueError(
            "System is not electrically neutral: sum(m_i * z_i) = "
            f"{Zm:.3e} (must be 0)"
        )

    # Initial guess from the binary CO2-H2O / CH4-H2O tables.
    GUESS = build_initial_guess(T, P, YC, guess_dir=guess_dir)

    # Solve, ramping ionic strength from 0 to the target for convergence.
    mion = float(np.sum(mi))
    if mion == 0:
        solution, ier = _solve_residual(GUESS, T, P, YC, 0 * mi, solver, tol,
                                        maxiter, xtol)
    elif solver == "newton":
        # Adaptive homotopy (few steps when easy, more when hard).
        solution, ier = _continuation_newton(GUESS, T, P, YC, mi, tol, maxiter, dm)
    else:
        # Legacy fixed-step salinity ramp (matches the original script exactly).
        Nit = int(mion / dm) + 1
        dm_ion = mi / Nit
        m = 0 * mi
        ier = 1
        ramp_ok = True
        for _ in range(Nit):
            solution, ier = _solve_residual(GUESS, T, P, YC, m, solver, tol,
                                            maxiter, xtol)
            # A failed rung must not be carried forward. Every step of this
            # ramp overwrote `ier`, so only the LAST solve's status survived:
            # a rung could fail, hand its bad vector on as GUESS, and the
            # final solve could then converge from that corrupted warm start
            # and report converged=True. The homotopy is a path -- if it
            # breaks anywhere, the endpoint is not the root we were tracking.
            # _continuation_newton (the other branch) already bails this way.
            if ier != 1:
                ramp_ok = False
                break
            # A rung can also break the path while reporting success, by
            # converging onto the trivial solute-rich root (see
            # _on_aqueous_branch). That vector would otherwise become the
            # next warm start and the endpoint would follow it there, so it
            # ends the ramp as squarely as a solver failure does.
            if not _on_aqueous_branch(solution, YC):
                ier = 0
                ramp_ok = False
                break
            GUESS = solution
            m = m + dm_ion
            if np.sum(m) > mion:
                break
        if ramp_ok:
            solution, ier = _solve_residual(GUESS, T, P, YC, m, solver,
                                            tol, maxiter, xtol)

    # The endpoint solve can land on the trivial root the same way a rung
    # can, and so can the Newton homotopy (which tests only its own status
    # flag and finiteness), so the vector that is about to be reported gets
    # the same check the rungs get.
    if ier == 1 and not _on_aqueous_branch(solution, YC):
        ier = 0

    # Residual at the vector we are about to report. The solvers' status
    # flags do not measure this (see ELVResult.residual), and it costs one
    # extra evaluation against the hundreds the solve itself spends.
    with np.errstate(all="ignore"):
        resid = float(np.max(np.abs(np.asarray(
            residual_elv(solution, T, P * 1e5, YC, mi), dtype=float))))

    # Unpack the solution vector (layout depends on YC).
    if YC == 0:
        Zw, x2w, epsr, Zh, x1h, chi1w, chi1h, Ndchi1WdNw, Ndchi1WdNh, Vdchi1WdV = solution
        x3w = 0.0
    elif YC == 1:
        Zw, x3w, epsr, Zh, x1h, chi1w, chi1h, Ndchi1WdNw, Ndchi1WdNc, Vdchi1WdV = solution
        x2w = 0.0
    else:
        (Zw, x2w, x3w, epsr, Zh, x1h, chi1w, chi1h,
         Ndchi1WdNw, Ndchi1WdNh, Ndchi1WdNc, Vdchi1WdV) = solution

    # Derived properties at the target condition.
    x1w = (1 - x2w - x3w) / (1 + np.sum(mi) * Mw)
    xiw = x1w * Mw * mi
    MMw = (Mw * x1w + Mh * x2w + Mc * x3w + np.sum(MMi * xiw))
    Pen_i = (A_Peni * T / Tc1 + B_Peni) * 1e-6
    v = Zw * R * T / (P * 1e5) + np.sum(xiw * Pen_i)
    rho_w = (1 / v * MMw)
    x3h = YC * (1 - x1h)
    x2h = 1 - x1h - x3h
    MMh = (Mw * x1h + Mh * x2h + Mc * x3h)
    rho_h = P * 1e5 / Zh / R / T * MMh
    # Total mass per mole of aqueous solution -- identical to MMw above.
    # This used to omit the dissolved-CO2 term x3w*Mc while keeping the
    # CH4 term, so wh/wc/ws were mass-per-mass-of-CO2-free-solution, i.e.
    # mass ratios, not the mass fractions ELVResult documents them to be.
    # For the case-A regression state the four aqueous mass fractions
    # (water + ions + CO2 + CH4) summed to 103.18 % on the old
    # denominator and sum to 100.00 % on this one.
    denom = MMw
    mh = x2w / (x1w * Mw)
    wh = 100 * x2w * Mh / denom
    mc = x3w / (x1w * Mw)
    wc = 100 * x3w * Mc / denom
    ms = float(np.sum(mi))
    ws = 100 * np.sum(xiw * MMi) / denom
    xs = 100 * float(np.sum(xiw))
    f_h2o, f_ch4, f_co2 = gas_fugacities(solution, T, P * 1e5, YC)

    return ELVResult(
        T=float(T), P=float(P), YC=float(YC), mi=tuple(float(x) for x in mi),
        converged=(ier == 1), ier=int(ier),
        salinity=ms, ws=float(ws), xs=xs,
        Zw=float(Zw), rho_w=float(rho_w), eps_r=float(epsr),
        x1w=float(x1w), x2w=float(x2w), x3w=float(x3w),
        mh=float(mh), wh=float(wh), mc=float(mc), wc=float(wc),
        Zh=float(Zh), rho_h=float(rho_h),
        x1h=float(x1h), x2h=float(x2h), x3h=float(x3h),
        f_h2o=float(f_h2o), f_ch4=float(f_ch4), f_co2=float(f_co2),
        solution=np.asarray(solution), residual=resid,
    )
