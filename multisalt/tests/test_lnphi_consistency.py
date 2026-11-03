"""Consistency test: the standalone lnφ kernels in :mod:`ms_ecpa.lnphi` must
reproduce the fugacity values that the coupled ``residual_elv`` produces at
the converged equilibrium.

Strategy
--------
1. For each of the 5 baseline regression cases, run :func:`solve_elv` to get
   the converged aqueous + gas compositions and the reported fugacities.
2. Call :func:`lnphi_aq` and :func:`lnphi_gas` *with default initial guesses*
   (no warm-start from the solved state — the flash will not have them).
3. Recompute fugacities ``f_i = exp(lnφ_i) * x_i * P_bar`` from each kernel.
4. Assert that each kernel's fugacities match ``solve_elv``'s to high
   precision (relative tolerance 1e-6), and that the aqueous and gas
   fugacities agree with each other to the same precision (isofugacity).

This proves the kernels (a) implement the correct physics and (b) converge
robustly from a generic starting guess — both required for downstream use
by the K-value flash and Michelsen stability search.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from ms_ecpa import solve_elv  # noqa: E402
from ms_ecpa.lnphi import lnphi_aq, lnphi_gas  # noqa: E402


CASES = [
    dict(name="A: 323K 500bar YC=0.8 mixed brine",
         T=323, P=500, YC=0.8, mi=[1, 2, 0.6, 0.2, 0.2, 0.2]),
    dict(name="B: 350K 100bar YC=1 (pure CO2-H2O)",
         T=350, P=100, YC=1, mi=[0, 0, 0, 0, 0, 0]),
    dict(name="C: 350K 100bar YC=0 (pure CH4-H2O)",
         T=350, P=100, YC=0, mi=[0, 0, 0, 0, 0, 0]),
    dict(name="D: 400K 300bar YC=0.5 NaCl 2m",
         T=400, P=300, YC=0.5, mi=[2, 2, 0, 0, 0, 0]),
    dict(name="E: 300K 600bar YC=1 CaCl2 1m",
         T=300, P=600, YC=1, mi=[0, 2, 0, 1, 0, 0]),
]


# Relative tolerance: 1e-6 on each fugacity, both kernel-vs-solve_elv and
# aqueous-vs-gas (isofugacity).  fsolve's residual is converged to ~1e-8, so
# the converged compositions inherit ~1e-7 relative noise; 1e-6 is a small
# safety margin above that.
REL_TOL = 1e-6


def _check_case(case):
    r = solve_elv(case["T"], case["P"], case["YC"], case["mi"])
    assert r.converged, f"{case['name']}: solve_elv did not converge"

    P_Pa = case["P"] * 1e5
    P_bar = case["P"]
    x_aq = (r.x1w, r.x2w, r.x3w)
    y_gas = (r.x1h, r.x2h, r.x3h)
    mi = np.asarray(case["mi"], dtype=float)

    # Use DEFAULT initial guesses to demonstrate the kernels' standalone
    # robustness (not pulling Zw, epsr, chi1w from r.solution).
    la = lnphi_aq(x_aq, mi, case["T"], P_Pa)
    lg = lnphi_gas(y_gas, case["T"], P_Pa)

    # Compute fugacities and compare against solve_elv's reported triple.
    f_aq_h2o, f_aq_ch4, f_aq_co2 = (
        float(np.exp(la[i]) * x_aq[i] * P_bar) for i in range(3)
    )
    f_g_h2o, f_g_ch4, f_g_co2 = (
        float(np.exp(lg[i]) * y_gas[i] * P_bar) for i in range(3)
    )

    failures = []

    def _rel(a, b):
        if abs(a) < 1e-12 and abs(b) < 1e-12:
            return 0.0
        denom = max(abs(a), abs(b))
        return abs(a - b) / denom

    # Kernel-vs-solve_elv (gas-side; this is what solve_elv prints)
    for label, kernel_val, ref_val in [
        ("f_h2o", f_g_h2o, r.f_h2o),
        ("f_ch4", f_g_ch4, r.f_ch4),
        ("f_co2", f_g_co2, r.f_co2),
    ]:
        rel = _rel(kernel_val, ref_val)
        if rel > REL_TOL and ref_val > 1e-6:
            failures.append(
                f"  lnphi_gas[{label}]: got {kernel_val:.6e}, expected "
                f"{ref_val:.6e}, rel={rel:.2e} > {REL_TOL:.0e}"
            )

    # Isofugacity (aqueous vs gas) for species present in both phases
    for label, fa, fg in [
        ("H2O", f_aq_h2o, f_g_h2o),
        ("CH4", f_aq_ch4, f_g_ch4),
        ("CO2", f_aq_co2, f_g_co2),
    ]:
        if min(abs(fa), abs(fg)) < 1e-10:
            continue  # species not present in one of the phases
        rel = _rel(fa, fg)
        if rel > REL_TOL:
            failures.append(
                f"  isofugacity[{label}]: aq={fa:.6e}, gas={fg:.6e}, "
                f"rel={rel:.2e} > {REL_TOL:.0e}"
            )

    return failures


def test_lnphi_consistency():
    """pytest entry: all 5 baseline cases must satisfy isofugacity to 1e-6."""
    for case in CASES:
        failures = _check_case(case)
        assert not failures, (
            f"{case['name']} mismatches:\n" + "\n".join(failures)
        )


def main():
    n_fail = 0
    for case in CASES:
        failures = _check_case(case)
        if failures:
            n_fail += 1
            print(f"FAIL  {case['name']}")
            print("\n".join(failures))
        else:
            print(f"PASS  {case['name']}")
    if n_fail:
        print(f"\n{n_fail} case(s) FAILED")
        return 1
    print(f"\nAll {len(CASES)} cases reproduce solve_elv fugacities and "
          "satisfy isofugacity to 1e-6.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
