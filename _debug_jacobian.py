"""Diagnostic: compare each intermediate derivative analytically vs numerically."""
import numpy as np
import CPA2

comps = CPA2.make_components_co2_h2o()
Omega, Tc, Pc, Mw = comps["Omega"], comps["Tc"], comps["Pc"], comps["Mw"]
T, P = 323.0, 100.0
kij = CPA2.kij_ecpa(T)
swc = CPA2.s14_ecpa(T)

# Get converged K, x, y
kw = dict(Omega=Omega, Tc=Tc, Pc=Pc, Mw=Mw, kij12=kij, swc=swc,
          tol=1e-10, maxiter=1000)
tie = CPA2.tie_line_two_comp(T=T, P_bar=P, accelerated=True, **kw)
x = tie["x"]; y = tie["y"]
print(f"x = {x},  y = {y}")

# Compute at base composition
ep0 = CPA2._eos_aq(T, P, x, kij)
Z0, Chi0, Chi10 = CPA2.ZChi(ep0["A"], ep0["B"], x, ep0["Kapa"], ep0["Eps"], swc)
lnphi0 = CPA2._lnphi_aq(ep0, Z0, Chi0, Chi10)
print(f"Z={Z0:.8f}  Chi={Chi0:.8f}  Chi1={Chi10:.8f}")
print(f"A={ep0['A']:.8f}  B={ep0['B']:.8f}")

# Numerical perturbation
h = 1e-7
xp = np.array([x[0]+h, 1-(x[0]+h)])
xm = np.array([x[0]-h, 1-(x[0]-h)])

ep_p = CPA2._eos_aq(T, P, xp, kij)
ep_m = CPA2._eos_aq(T, P, xm, kij)
Zp, Chip, Chi1p = CPA2.ZChi(ep_p["A"], ep_p["B"], xp, ep_p["Kapa"], ep_p["Eps"], swc)
Zm, Chim, Chi1m = CPA2.ZChi(ep_m["A"], ep_m["B"], xm, ep_m["Kapa"], ep_m["Eps"], swc)
lnphi_p = CPA2._lnphi_aq(ep_p, Zp, Chip, Chi1p)
lnphi_m = CPA2._lnphi_aq(ep_m, Zm, Chim, Chi1m)

# Numerical derivatives
dA_dx_num = (ep_p["A"] - ep_m["A"]) / (2*h)
dB_dx_num = (ep_p["B"] - ep_m["B"]) / (2*h)
dZ_dx_num = (Zp - Zm) / (2*h)
dChi_dx_num = (Chip - Chim) / (2*h)
dChi1_dx_num = (Chi1p - Chi1m) / (2*h)
dlnphi_dx_num = (lnphi_p - lnphi_m) / (2*h)

print(f"\nNumerical: dA/dx={dA_dx_num:.6f}  dB/dx={dB_dx_num:.6f}")
print(f"Numerical: dZ/dx={dZ_dx_num:.8f}  dChi/dx={dChi_dx_num:.8f}  dChi1/dx={dChi1_dx_num:.8f}")
print(f"Numerical: dlnphi/dx = {dlnphi_dx_num}")

# Now compute analytical intermediates
# Replicate what _dlnphi_dx_phase does internally
A, B = ep0["A"], ep0["B"]
A1, B1, A4, B4 = ep0["A1"], ep0["B1"], ep0["A4"], ep0["B4"]
x4, x1 = ep0["x4"], ep0["x1"]

eta = B / (4*Z0)
g = CPA2._g(eta)
dg = CPA2._dgdeta(eta)
d2g = CPA2._d2gdeta2(eta)
dgog = dg / g
delta = g * ep0["Kapa"] * np.expm1(ep0["Eps"])
delta1 = swc * delta
Sigma = x1*(Chi0-1) + x4*(Chi10-1)

dB_dx_an = B4 - B1

# HV dA_dx
hv = ep0["hv"]
U14, U41, gE = hv["U14"], hv["U41"], hv["gE"]
a1d, b1d, a4d, b4d = ep0["a1"], ep0["b1"], ep0["a4"], ep0["b4"]
bd = ep0["b"]
RT = ep0["R"] * ep0["T"]
cross_BU = b1d*U14 + b4d*U41
dgE_dx = cross_BU * ((1-2*x4)*bd - x1*x4*(b4d-b1d)) / bd**2
S = x1*a1d/b1d + x4*a4d/b4d - gE/CPA2._LN2
dS_dx = -a1d/b1d + a4d/b4d - dgE_dx/CPA2._LN2
db_dx = b4d - b1d
da_dx = db_dx * S + bd * dS_dx
dA_dx_an = da_dx * ep0["P_bar"] / RT**2

print(f"\nAnalytical: dA/dx={dA_dx_an:.6f}  dB/dx={dB_dx_an:.6f}")
print(f"  dA error: {abs(dA_dx_an-dA_dx_num):.2e}")
print(f"  dB error: {abs(dB_dx_an-dB_dx_num):.2e}")

# Check dgE_dx
gE_p = (xp[1]*xp[0]/(b1d*xp[1]+b4d*xp[0])) * cross_BU
gE_m = (xm[1]*xm[0]/(b1d*xm[1]+b4d*xm[0])) * cross_BU
dgE_dx_num = (gE_p - gE_m) / (2*h)
print(f"\n  gE={gE:.6f}  dgE/dx analytical={dgE_dx:.6f}  numerical={dgE_dx_num:.6f}  err={abs(dgE_dx-dgE_dx_num):.2e}")

# Now solve implicit system analytically
dlogd_dZ = dgog * (-B/(4*Z0**2))
dlogd_dB = dgog / (4*Z0)
ZmB = Z0 - B; ZpB = Z0 + B
h_val = 1 + eta*dgog
dh = dgog + eta*(d2g/g - dgog**2)

dFZ_dZ = 1 + B/ZmB**2 - A/ZpB**2 + dh*B/(2*Z0**2)*Sigma
dFZ_dChi = -2*h_val*x1
dFZ_dChi1 = -2*h_val*x4
dFZ_dx = (dA_dx_an/ZpB + dB_dx_an*(-Z0/ZmB**2-A/ZpB**2-dh/(2*Z0)*Sigma)
          + 2*h_val*((Chi0-1)-(Chi10-1)))

c3 = 4*x1**2*delta*delta1
c2 = 2*x1*Z0*(delta+delta1)
c1 = 2*delta1*Z0*(x4-x1)+Z0**2

dFchi_dChi = 3*c3*Chi0**2+2*c2*Chi0+c1
dc3_dZ = c3*2*dlogd_dZ
dc2_dZ = 2*x1*(delta+delta1)+2*x1*Z0*(delta+delta1)*dlogd_dZ
dc1_dZ = 2*delta1*(x4-x1)+2*delta1*Z0*(x4-x1)*dlogd_dZ+2*Z0
dc0_dZ = -2*Z0
dFchi_dZ = dc3_dZ*Chi0**3+dc2_dZ*Chi0**2+dc1_dZ*Chi0+dc0_dZ

dc3_dB = c3*2*dlogd_dB
dc2_dB = 2*x1*Z0*(delta+delta1)*dlogd_dB
dc1_dB = 2*delta1*Z0*(x4-x1)*dlogd_dB
dc3_dn = -8*x1*delta*delta1
dc2_dn = -2*Z0*(delta+delta1)
dc1_dn = 4*delta1*Z0
dFchi_dx = (dc3_dn+dc3_dB*dB_dx_an)*Chi0**3+(dc2_dn+dc2_dB*dB_dx_an)*Chi0**2+(dc1_dn+dc1_dB*dB_dx_an)*Chi0

D = Z0 + 2*x1*Chi0*delta1
dD_dZ = 1 + 2*x1*Chi0*delta1*dlogd_dZ
dG_dZ = -(D - Z0*dD_dZ)/D**2
dD_dChi = 2*x1*delta1
dG_dChi = Z0*dD_dChi/D**2
dG_dChi1 = 1.0
dD_dx = -2*Chi0*delta1 + 2*x1*Chi0*delta1*dlogd_dB*dB_dx_an
dG_dx = Z0*dD_dx/D**2

J_impl = np.array([
    [dFZ_dZ, dFZ_dChi, dFZ_dChi1],
    [dFchi_dZ, dFchi_dChi, 0.0],
    [dG_dZ, dG_dChi, dG_dChi1],
])
rhs = -np.array([dFZ_dx, dFchi_dx, dG_dx])
d_impl = np.linalg.solve(J_impl, rhs)
dZ_dx_an, dChi_dx_an, dChi1_dx_an = d_impl

print(f"\nImplicit system results:")
print(f"  dZ/dx:   analytical={dZ_dx_an:.8f}   numerical={dZ_dx_num:.8f}   err={abs(dZ_dx_an-dZ_dx_num):.2e}")
print(f"  dChi/dx: analytical={dChi_dx_an:.8f}  numerical={dChi_dx_num:.8f}  err={abs(dChi_dx_an-dChi_dx_num):.2e}")
print(f"  dChi1/dx: analytical={dChi1_dx_an:.8f}  numerical={dChi1_dx_num:.8f}  err={abs(dChi1_dx_an-dChi1_dx_num):.2e}")

# Also check FunZ residual derivatives numerically
FZ_0 = CPA2.FunZ(Z0, A, B, x, Chi0, Chi10)
# Perturb Z
FZ_Zp = CPA2.FunZ(Z0+1e-8, A, B, x, Chi0, Chi10)
FZ_Zm = CPA2.FunZ(Z0-1e-8, A, B, x, Chi0, Chi10)
dFZ_dZ_num = (FZ_Zp - FZ_Zm) / (2e-8)
print(f"\n  dFZ/dZ:  analytical={dFZ_dZ:.8f}  numerical={dFZ_dZ_num:.8f}  err={abs(dFZ_dZ-dFZ_dZ_num):.2e}")

# Perturb Chi
FZ_Cp = CPA2.FunZ(Z0, A, B, x, Chi0+1e-8, Chi10)
FZ_Cm = CPA2.FunZ(Z0, A, B, x, Chi0-1e-8, Chi10)
dFZ_dChi_num = (FZ_Cp - FZ_Cm) / (2e-8)
print(f"  dFZ/dChi: analytical={dFZ_dChi:.8f}  numerical={dFZ_dChi_num:.8f}  err={abs(dFZ_dChi-dFZ_dChi_num):.2e}")

# Full x perturbation of FZ (at fixed Z, Chi, Chi1 — only A, B, nw, nc change)
FZ_xp = CPA2.FunZ(Z0, ep_p["A"], ep_p["B"], xp, Chi0, Chi10)
FZ_xm = CPA2.FunZ(Z0, ep_m["A"], ep_m["B"], xm, Chi0, Chi10)
dFZ_dx_num = (FZ_xp - FZ_xm) / (2*h)
print(f"  dFZ/dx:  analytical={dFZ_dx:.8f}  numerical={dFZ_dx_num:.8f}  err={abs(dFZ_dx-dFZ_dx_num):.2e}")

# Full dlnphi/dx from _dlnphi_dx_phase
dlnphi_an = CPA2._dlnphi_dx_phase(ep0, Z0, Chi0, Chi10, swc, True)
print(f"\nFinal dlnphi/dx:  analytical={dlnphi_an}  numerical={dlnphi_dx_num}")
print(f"  error: {np.abs(dlnphi_an - dlnphi_dx_num)}")
