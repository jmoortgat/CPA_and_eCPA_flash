"""Verify the Michelsen TPD stability primitive for the multi-salt eCPA.

Phase E of Step 2.

For each of the 5 baseline regression cases, the equilibrium state produced
by ``solve_elv`` is by construction a 2-phase point.  Reconstruct the
partition-basis feed ``z`` from the converged aqueous + gas phase
compositions (β = 0.5 convention, same as ``tests/test_flash_kv.py``),
call :func:`michelsen_tpd`, and assert:

* the mixture is classified as unstable (``stable == False``)
* ``tpd_min < -0.02`` (well below the soft-guard threshold the combined
  driver uses)

Plus one "obviously single-phase" case: hot, very high pressure, ~pure water.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from ms_ecpa import solve_elv  # noqa: E402
from ms_ecpa.parameters import Mw  # noqa: E402
from ms_ecpa.stability import michelsen_tpd  # noqa: E402


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


def feed_from_equilibrium(r, mi, beta=0.5):
    """Reconstruct a partition-basis feed z from a converged solve_elv result.

    Replicates the same construction used in tests/test_flash_kv.py.
    """
    x_aq = np.array([r.x1w, r.x2w, r.x3w])
    y_gas = np.array([r.x1h, r.x2h, r.x3h])
    m_tot = float(np.sum(mi))
    scale = 1.0 - Mw * r.x1w * m_tot
    x_part = x_aq / scale
    z = (1.0 - beta) * x_part + beta * y_gas
    return z / z.sum()


def _check_baseline_case(case):
    r = solve_elv(case["T"], case["P"], case["YC"], case["mi"])
    assert r.converged, f"{case['name']}: solve_elv did not converge"
    mi = np.asarray(case["mi"], dtype=float)
    z = feed_from_equilibrium(r, mi)
    stab = michelsen_tpd(z, mi, case["T"], case["P"])

    failures = []
    if stab.stable:
        failures.append(
            f"  expected unstable, but michelsen_tpd reported stable; "
            f"tpd_min={stab.tpd_min:.3e}; msg={stab.message}"
        )
    if not (stab.tpd_min < -0.02):
        failures.append(
            f"  tpd_min={stab.tpd_min:.3e} not < -0.02; msg={stab.message}"
        )
    return failures, stab


def _check_single_phase_case():
    """Obvious single-phase: hot+high-P, ~pure water, no salt."""
    T = 350.0
    P = 1500.0
    z = np.array([0.999, 0.0, 0.001])
    mi = np.zeros(6)
    stab = michelsen_tpd(z, mi, T, P)
    failures = []
    if not stab.stable:
        failures.append(
            f"  expected single-phase, got unstable; "
            f"tpd_min={stab.tpd_min:.3e}; msg={stab.message}"
        )
    return failures, stab


def test_baseline_cases_are_unstable():
    """pytest entry: every baseline equilibrium is detected as unstable."""
    for case in CASES:
        failures, stab = _check_baseline_case(case)
        assert not failures, (
            f"{case['name']} mismatches:\n" + "\n".join(failures)
        )


def test_single_phase_case():
    failures, stab = _check_single_phase_case()
    assert not failures, "single-phase test failed:\n" + "\n".join(failures)


def main():
    n_fail = 0
    print("Baseline (should detect as unstable):")
    for case in CASES:
        failures, stab = _check_baseline_case(case)
        if failures:
            n_fail += 1
            print(f"FAIL  {case['name']}")
            print("\n".join(failures))
        else:
            best = stab.trials[stab.best_trial_idx]
            print(f"PASS  {case['name']}  tpd_min={stab.tpd_min:+.3e}  "
                  f"trial='{best.label}' ({best.side}), iters={best.n_iter}")
    print("\nObvious single-phase (should detect as stable):")
    failures, stab = _check_single_phase_case()
    if failures:
        n_fail += 1
        print("FAIL  single-phase case")
        print("\n".join(failures))
    else:
        print(f"PASS  single-phase case  tpd_min={stab.tpd_min:+.3e}")
    if n_fail:
        print(f"\n{n_fail} case(s) FAILED")
        return 1
    print(f"\nAll {len(CASES)} baseline + 1 single-phase case classified correctly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
