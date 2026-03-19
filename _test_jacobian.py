"""
_test_jacobian.py — Verify analytical Newton Jacobian against numerical.

Tests at a range of (T, P) conditions that the analytical d(lng)/d(lnK)
matches the numerical finite-difference Jacobian.
"""
import numpy as np
import CPA2

comps = CPA2.make_components_co2_h2o()
Omega, Tc, Pc, Mw = comps["Omega"], comps["Tc"], comps["Pc"], comps["Mw"]

# Test conditions: a mix of easy and hard points
test_points = [
    (298, 50),    # moderate
    (323, 100),   # typical subsurface
    (373, 200),   # high T, high P
    (400, 500),   # near critical region
    (283, 10),    # low T, low P
    (500, 1000),  # extreme
    (310, 74),    # near CO2 critical point
]

print("=" * 90)
print("JACOBIAN VERIFICATION: analytical vs numerical (central differences)")
print("=" * 90)

n_pass = 0
n_fail = 0
max_err = 0.0

for T, P in test_points:
    kij = CPA2.kij_ecpa(T)
    swc = CPA2.s14_ecpa(T)

    # First run a converged flash to get reasonable K-values
    kw = dict(Omega=Omega, Tc=Tc, Pc=Pc, Mw=Mw, kij12=kij, swc=swc,
              tol=1e-10, maxiter=1000)
    tie = CPA2.tie_line_two_comp(T=T, P_bar=P, accelerated=True, **kw)
    if not tie["converged"]:
        print(f"  T={T:5.0f}K  P={P:5.0f}bar  SKIP (flash did not converge)")
        continue

    K = tie["K"]
    lnK = np.log(K)
    x = tie["x"]
    y = tie["y"]

    # Recompute EOS at converged compositions
    ep_aq  = CPA2._eos_aq(T, P, x, kij)
    ep_vap = CPA2._eos_vap(T, P, y, kij)
    Zx, Chix, Chi1x = CPA2.ZChi(ep_aq["A"], ep_aq["B"], x,
                                  ep_aq["Kapa"], ep_aq["Eps"], swc)
    Zy, Chiy, Chi1y = CPA2.ZChi(ep_vap["A"], ep_vap["B"], y,
                                  ep_vap["Kapa"], ep_vap["Eps"], swc)

    # ── Test 1: dlnphi_dx per phase ──
    dphi_x_an = CPA2._dlnphi_dx_phase(ep_aq, Zx, Chix, Chi1x, swc, True)
    dphi_x_nm = CPA2._dlnphi_dx_numerical(T, P, x, kij, swc, True)
    dphi_y_an = CPA2._dlnphi_dx_phase(ep_vap, Zy, Chiy, Chi1y, swc, False)
    dphi_y_nm = CPA2._dlnphi_dx_numerical(T, P, y, kij, swc, False)

    err_x = np.max(np.abs(dphi_x_an - dphi_x_nm))
    err_y = np.max(np.abs(dphi_y_an - dphi_y_nm))
    max_err_phase = max(err_x, err_y)

    # ── Test 2: full Newton Jacobian ──
    J_an = CPA2.newton_jacobian(lnK, T, P, kij, swc,
                                 ep_aq, ep_vap,
                                 Zx, Chix, Chi1x, Zy, Chiy, Chi1y,
                                 x, y)
    J_nm = CPA2.newton_jacobian_numerical(lnK, T, P, kij, swc, kw)

    err_J = np.max(np.abs(J_an - J_nm))
    rel_err_J = err_J / max(np.max(np.abs(J_nm)), 1e-10)

    passed = rel_err_J < 1e-3 and max_err_phase < 1e-3
    status = "PASS" if passed else "FAIL"
    if passed:
        n_pass += 1
    else:
        n_fail += 1
    max_err = max(max_err, rel_err_J)

    print(f"\n  T={T:5.0f}K  P={P:5.0f}bar  [{status}]")
    print(f"    dlnphi_dx aq:  analytical={dphi_x_an}  numerical={dphi_x_nm}  |err|={err_x:.2e}")
    print(f"    dlnphi_dx vap: analytical={dphi_y_an}  numerical={dphi_y_nm}  |err|={err_y:.2e}")
    print(f"    J_analytical:\n{J_an}")
    print(f"    J_numerical:\n{J_nm}")
    print(f"    |J_an - J_nm|_max = {err_J:.2e}   rel = {rel_err_J:.2e}")

print(f"\n{'='*90}")
print(f"SUMMARY: {n_pass} passed, {n_fail} failed, max relative error = {max_err:.2e}")
print(f"{'='*90}")
