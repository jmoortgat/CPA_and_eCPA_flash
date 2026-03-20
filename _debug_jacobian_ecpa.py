"""
Validate the analytical 3×3 Jacobian of the eCPA aqueous Newton system.

For each (Zw, epsr, chi1w) at several (T, P, ms) conditions, compare every
element J[i,j] against a centered finite-difference estimate.  Also exercises
_newton_aq on warm and perturbed starts.

Usage::
    python _debug_jacobian_ecpa.py
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from ecpa.stability import (
    _eval_aq_all, _eval_aq_all_with_jac, _newton_aq, _lnphi_aq_inner
)
import ecpa.constants  # ensure globals are set

# ── Test conditions ──────────────────────────────────────────────────────────

CONDITIONS = [
    # (T, P, x1w, ms,   label)
    (373.0, 100.0, 0.940, 1.0,   "T=373 P=100 ms=1"),
    (373.0, 200.0, 0.920, 2.0,   "T=373 P=200 ms=2"),
    (423.0,  50.0, 0.955, 0.5,   "T=423 P=50  ms=0.5"),
    (348.0,  50.0, 0.980, 0.0,   "T=348 P=50  ms=0 (no DH)"),
]

H_CFD = 1e-6   # centered FD step


def residual(Zw, epsr, chi1w, x1w, ms, T, P):
    chi1w_e = min(chi1w, 1.0 - 1e-12)
    Zw_new, T1, T2, chi1w_new, _, _ = _eval_aq_all(
        Zw, epsr, chi1w_e, x1w, ms, T, P)
    return np.array([Zw - Zw_new, T1 - T2, chi1w - chi1w_new])


def cfd_jacobian(Zw, epsr, chi1w, x1w, ms, T, P):
    """Centered finite-difference 3×3 Jacobian."""
    v0 = np.array([Zw, epsr, chi1w])
    J_num = np.zeros((3, 3))
    for j in range(3):
        h = max(abs(v0[j]) * H_CFD, 1e-8)
        vp = v0.copy(); vp[j] += h
        vm = v0.copy(); vm[j] -= h
        Fp = residual(*vp, x1w, ms, T, P)
        Fm = residual(*vm, x1w, ms, T, P)
        J_num[:, j] = (Fp - Fm) / (2*h)
    return J_num


# ── Main loop ────────────────────────────────────────────────────────────────

all_pass = True
for T, P, x1w, ms, label in CONDITIONS:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # Get a converged solution to use as the evaluation point
    try:
        lnphi1, lnphi4, sol = _lnphi_aq_inner(x1w, ms, T, P)
    except RuntimeError as exc:
        print(f"  SKIP — _lnphi_aq_inner failed: {exc}")
        continue

    Zw, epsr, chi1w = sol[0], sol[1], sol[2]
    print(f"  Converged solution: Zw={Zw:.6f}  epsr={epsr:.4f}  chi1w={chi1w:.6f}")

    # Evaluate analytical Jacobian
    chi1w_e = min(chi1w, 1.0 - 1e-12)
    _, _, _, _, _, _, J_an = _eval_aq_all_with_jac(
        Zw, epsr, chi1w_e, x1w, ms, T, P)

    # Centered FD Jacobian
    J_num = cfd_jacobian(Zw, epsr, chi1w_e, x1w, ms, T, P)

    # Compare element-by-element
    names = ["Zw", "epsr", "chi1w"]
    F_labels = ["F0=Zw-Zw_new", "F1=T1-T2", "F2=chi1w-chi1w_new"]
    print(f"\n  {'Element':12s}  {'Analytical':>14s}  {'Numerical':>14s}  "
          f"{'Abs err':>10s}  {'Rel err':>10s}")
    cond_pass = True
    for i in range(3):
        for j in range(3):
            an = J_an[i, j]
            nu = J_num[i, j]
            abs_err = abs(an - nu)
            rel_err = abs_err / (abs(nu) + 1e-30)
            tol_ok = abs_err < 1e-5 or rel_err < 1e-4
            flag = "" if tol_ok else "  *** FAIL"
            if not tol_ok:
                cond_pass = False; all_pass = False
            print(f"  J[{i},{j}]=∂{F_labels[i][:2]}/∂{names[j]:5s}  "
                  f"{an:14.6e}  {nu:14.6e}  {abs_err:10.2e}  {rel_err:10.2e}{flag}")

    if cond_pass:
        print("\n  All elements OK.")
    else:
        print("\n  *** Some elements FAILED.")

    # ── Newton warm-start test ────────────────────────────────────────────────
    print(f"\n  Newton solver tests:")
    # Exact warm start (should converge in 1 step at the residual check)
    v_ex = _newton_aq(sol, x1w, ms, T, P)
    print(f"    Exact warm start:      {'converged' if v_ex is not None else 'FAILED'}")

    # 1% perturbation
    v_perturbed = sol * np.array([1.01, 1.005, 0.99])
    v_pt = _newton_aq(v_perturbed, x1w, ms, T, P)
    if v_pt is not None:
        err = np.max(np.abs(v_pt - sol) / (np.abs(sol) + 1e-30))
        print(f"    1%-perturbed start:    converged  (max_rel_err={err:.2e})")
    else:
        print(f"    1%-perturbed start:    FAILED (fell back to fsolve expected)")

    # 5% perturbation
    v_perturbed5 = sol * np.array([1.05, 1.01, 0.95])
    v_pt5 = _newton_aq(v_perturbed5, x1w, ms, T, P)
    if v_pt5 is not None:
        err = np.max(np.abs(v_pt5 - sol) / (np.abs(sol) + 1e-30))
        print(f"    5%-perturbed start:    converged  (max_rel_err={err:.2e})")
    else:
        print(f"    5%-perturbed start:    FAILED (expected for large perturbation)")

print(f"\n{'='*60}")
print(f"  Overall: {'ALL PASS' if all_pass else 'SOME FAILURES — see above'}")
print(f"{'='*60}\n")
