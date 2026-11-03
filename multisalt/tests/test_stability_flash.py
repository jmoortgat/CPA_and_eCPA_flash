"""Verify the combined stability_flash driver.

Phase G of Step 2.

Three checks:

1. For each of the 5 baseline equilibrium cases: ``stability_flash``
   returns a 2-phase result whose compositions match ``flash_kv`` (and
   thus, transitively, ``solve_elv``).
2. For the "obviously single-phase" case (350 K, 1500 bar, near-pure
   water): ``stability_flash`` returns a 1-phase result.
3. Iteration-efficiency check: total iterations (SSI + Newton)
   inside the ``stability_flash`` path should not exceed Wilson-init
   ``flash_kv`` by more than a small fixed overhead on at least 3/5
   baseline cases.  The hierarchical algorithm spends extra iterations
   inside the stability test, so this is a modest target compared to
   raw flash_kv; the value of the combined driver is in robustness +
   single-phase classification, not pure speed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from ms_ecpa import solve_elv  # noqa: E402
from ms_ecpa.parameters import Mw  # noqa: E402
from ms_ecpa.flash import flash_kv  # noqa: E402
from ms_ecpa.stability import stability_flash  # noqa: E402


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


def _feed(r, mi, beta=0.5):
    x_aq = np.array([r.x1w, r.x2w, r.x3w])
    y_gas = np.array([r.x1h, r.x2h, r.x3h])
    m_tot = float(np.sum(mi))
    scale = 1.0 - Mw * r.x1w * m_tot
    x_part = x_aq / scale
    z = (1.0 - beta) * x_part + beta * y_gas
    return z / z.sum()


def _check_two_phase(case):
    r = solve_elv(case["T"], case["P"], case["YC"], case["mi"])
    assert r.converged, f"{case['name']}: solve_elv did not converge"
    mi = np.asarray(case["mi"], dtype=float)
    z = _feed(r, mi)

    fr_wil = flash_kv(case["T"], case["P"], z, mi)
    assert fr_wil.converged, (
        f"{case['name']}: reference flash_kv did not converge"
    )

    sf = stability_flash(case["T"], case["P"], z, mi)
    failures = []
    if sf.phases != 2:
        failures.append(f"  expected 2 phases, got {sf.phases}: {sf.message}")
        return failures, sf, fr_wil
    if sf.flash_result is None:
        failures.append("  flash_result is None for 2-phase result")
        return failures, sf, fr_wil
    if not sf.flash_result.converged:
        failures.append(f"  flash_result not converged: {sf.flash_result.message}")
        return failures, sf, fr_wil

    # Compositions match flash_kv (and solve_elv via flash_kv).
    fr = sf.flash_result
    for label, a, b in [
        ("x_aq", fr.x_aq, fr_wil.x_aq),
        ("y_gas", fr.y_gas, fr_wil.y_gas),
    ]:
        for i in range(3):
            if abs(a[i] - b[i]) > 1e-4:
                failures.append(
                    f"  {label}[{i}] mismatch vs Wilson flash_kv "
                    f"({a[i]:.6f} vs {b[i]:.6f})"
                )
    # And against solve_elv.
    for label, got, ref in [
        ("x1w", fr.x_aq[0], r.x1w), ("x2w", fr.x_aq[1], r.x2w),
        ("x3w", fr.x_aq[2], r.x3w),
        ("y1", fr.y_gas[0], r.x1h), ("y2", fr.y_gas[1], r.x2h),
        ("y3", fr.y_gas[2], r.x3h),
    ]:
        if abs(got - ref) > 1e-4:
            failures.append(
                f"  {label} mismatch vs solve_elv ({got:.6f} vs {ref:.6f})"
            )
    return failures, sf, fr_wil


def _check_single_phase():
    T = 350.0
    P = 1500.0
    z = np.array([0.999, 0.0, 0.001])
    mi = np.zeros(6)
    sf = stability_flash(T, P, z, mi)
    failures = []
    if sf.phases != 1:
        failures.append(
            f"  expected 1 phase, got {sf.phases}: {sf.message}"
        )
    if not sf.stable:
        failures.append(f"  expected stable=True, got {sf.stable}")
    return failures, sf


def test_baseline_two_phase():
    for case in CASES:
        failures, sf, _ = _check_two_phase(case)
        assert not failures, (
            f"{case['name']} mismatches:\n" + "\n".join(failures)
        )


def test_single_phase():
    failures, sf = _check_single_phase()
    assert not failures, "single-phase test failed:\n" + "\n".join(failures)


def test_iteration_efficiency():
    """At least 3/5 cases should have total iters not much larger than
    Wilson-flash_kv (combined-driver tax for the stability test is OK)."""
    near_or_better = 0
    for case in CASES:
        failures, sf, fr_wil = _check_two_phase(case)
        if failures or sf.flash_result is None:
            continue
        wil_total = fr_wil.n_iter_ssi + fr_wil.n_iter_newton
        sf_total = sf.flash_result.n_iter_ssi + sf.flash_result.n_iter_newton
        # Inner flash inside stability_flash should be at most == Wilson's
        if sf_total <= wil_total:
            near_or_better += 1
    assert near_or_better >= 3, (
        f"only {near_or_better}/5 baseline cases had stability_flash inner-flash "
        f"iters <= Wilson flash_kv"
    )


def main():
    n_fail = 0
    print("Baseline 2-phase cases:")
    for case in CASES:
        failures, sf, fr_wil = _check_two_phase(case)
        wil_total = fr_wil.n_iter_ssi + fr_wil.n_iter_newton
        sf_total = (sf.flash_result.n_iter_ssi + sf.flash_result.n_iter_newton
                    if sf.flash_result else -1)
        if failures:
            n_fail += 1
            print(f"FAIL  {case['name']}")
            print("\n".join(failures))
        else:
            print(f"PASS  {case['name']}  phases={sf.phases}  "
                  f"tpd_min={sf.tpd_min:+.3e}  "
                  f"inner flash_kv: SSI+Nwt={sf_total} (Wilson: {wil_total})")

    print("\nSingle-phase case:")
    failures, sf = _check_single_phase()
    if failures:
        n_fail += 1
        print("FAIL  single-phase case")
        print("\n".join(failures))
    else:
        print(f"PASS  single-phase case  phases={sf.phases}  "
              f"tpd_min={sf.tpd_min:+.3e}")

    if n_fail:
        print(f"\n{n_fail} case(s) FAILED")
        return 1
    print(f"\nAll {len(CASES)} 2-phase + 1 single-phase case classified correctly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
