"""Verify make_k_init_from_trial improves flash_kv convergence.

Phase F of Step 2.

For each of the 5 baseline cases, run two flashes:

* "Wilson"  : flash_kv with the default Wilson K_init.
* "Trial K" : flash_kv with K_init = make_k_init_from_trial(best_trial, ...).

Assert that the trial-K initialisation converges in **fewer total SSI
iterations** than Wilson in at least 3 of the 5 baseline cases.  Both must
agree on the final composition to the same tolerance the rest of the suite
uses (``1e-4`` mole-fraction).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from ms_ecpa import solve_elv  # noqa: E402
from ms_ecpa.parameters import Mw  # noqa: E402
from ms_ecpa.flash import flash_kv  # noqa: E402
from ms_ecpa.stability import michelsen_tpd, make_k_init_from_trial  # noqa: E402


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


def _check_case(case):
    r = solve_elv(case["T"], case["P"], case["YC"], case["mi"])
    assert r.converged, f"{case['name']}: solve_elv did not converge"
    mi = np.asarray(case["mi"], dtype=float)
    z = _feed(r, mi)

    # Reference: Wilson-initialised flash
    fr_wil = flash_kv(case["T"], case["P"], z, mi)
    assert fr_wil.converged, f"{case['name']}: Wilson flash did not converge"

    # Trial-K flash
    stab = michelsen_tpd(z, mi, case["T"], case["P"])
    assert stab.best_trial_idx >= 0, (
        f"{case['name']}: stability test found no negative-tpd trial; "
        f"tpd_min={stab.tpd_min:.3e}"
    )
    K_trial = make_k_init_from_trial(
        stab.trials[stab.best_trial_idx], z, case["T"], case["P"])
    fr_trial = flash_kv(case["T"], case["P"], z, mi, K_init=K_trial)
    assert fr_trial.converged, (
        f"{case['name']}: trial-K flash did not converge: {fr_trial.message}"
    )

    # Compositions must match within the same tolerance the suite uses.
    for label, a, b in [
        ("x_aq", fr_wil.x_aq, fr_trial.x_aq),
        ("y_gas", fr_wil.y_gas, fr_trial.y_gas),
    ]:
        for i in range(3):
            assert abs(a[i] - b[i]) < 1e-4, (
                f"{case['name']}: {label}[{i}] mismatch "
                f"(Wilson={a[i]:.6f}, trial={b[i]:.6f})"
            )

    return fr_wil, fr_trial, K_trial


def test_trial_k_improves_iters():
    improved = 0
    for case in CASES:
        fr_wil, fr_trial, K_trial = _check_case(case)
        total_wil = fr_wil.n_iter_ssi + fr_wil.n_iter_newton
        total_trial = fr_trial.n_iter_ssi + fr_trial.n_iter_newton
        if total_trial < total_wil:
            improved += 1
    assert improved >= 3, (
        f"trial-K-init helped only {improved}/5 cases; expected >=3"
    )


def main():
    print(f"{'case':<35}{'Wilson SSI+Nwt':>16}{'Trial SSI+Nwt':>16}{'improved':>10}")
    print("-" * 77)
    improved = 0
    for case in CASES:
        fr_wil, fr_trial, K_trial = _check_case(case)
        tw = fr_wil.n_iter_ssi + fr_wil.n_iter_newton
        tt = fr_trial.n_iter_ssi + fr_trial.n_iter_newton
        flag = "Y" if tt < tw else ("=" if tt == tw else "N")
        if tt < tw:
            improved += 1
        print(f"{case['name']:<35}{tw:>16d}{tt:>16d}{flag:>10}")
    print(f"\nImproved: {improved}/{len(CASES)}")
    if improved < 3:
        print(f"FAIL: expected at least 3, got {improved}")
        return 1
    print(f"PASS: at least 3 of 5 cases improved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
