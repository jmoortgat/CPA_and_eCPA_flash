"""Verify that the multi-salt aqueous closure reduces to the prior project's
NaCl-1:1 form.

The prior project (``code/ecpa/flash.py``) solves

    x1w + 2·n_salt/N_aq(x1w) + (1 − K1·x1w)/K4 − 1 = 0          (legacy)

with a single CO2 gas (K4 = K_CO2) and binary salt NaCl (one ν=1 cation, one
ν=1 anion).  Here we restate this in the multi-salt / multi-gas notation
used by :func:`ms_ecpa.flash.aqueous_closure`.

The key reductions
------------------
* Single-gas CO2 ⇒  ``y_H2O + y_CO2 = 1`` ⇒  ``y_CO2 = 1 − K1·x1w``.
* NaCl-1:1 with `ms` (mol-salt / kg-water) ⇒ ``m_total = 2·ms`` and the ion
  mole fraction sum is ``M_w·x1w·m_total = 2·M_w·x1w·ms``.

Substituting into the new-form equation:

    R = x1w·(1 + M_w·m_total) + 0 + y_CO2/K_CO2 − 1
      = x1w + 2·M_w·x1w·ms + (1 − K1·x1w)/K4 − 1

For algebraic equivalence to the legacy ``2·n_salt / N_aq(x1w)``, note that
in the legacy NaCl-only world ``N_aq = N_aq(x1w)`` is the *total* number of
aqueous moles, and ``ms_aq = n_salt / (N_aq · x1w · M_w)``.  By definition
``ms_aq = m_total / 2`` (NaCl 1:1) ⇒ ``n_salt = N_aq · x1w · M_w · ms_aq =
N_aq · x1w · M_w · (m_total/2)``.  Hence

    2·n_salt/N_aq = M_w · x1w · m_total  (= our ion-fraction sum)  ✓

This file checks the reduction numerically.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from ms_ecpa.parameters import Mw  # noqa: E402
from ms_ecpa.flash import aqueous_closure, compute_aqueous_x1w_from_KP  # noqa: E402


def legacy_eq_x1w_nacl(x1w, K1, K4, n_h2o, n_co2, n_salt):
    """Verbatim copy of the prior project's _eq_x1w residual."""
    dK = K4 - K1
    u = 1.0 - K1 * x1w
    v = n_h2o - K1 * x1w
    Naq = K4 * v / (u * x1w * dK)
    return x1w + 2.0 * n_salt / Naq + u / K4 - 1.0


def test_reduction_to_nacl_1_1():
    """At a representative point, the new and legacy residuals agree."""
    # Choose a physical (T, P, K1, K4) combo not too extreme
    K1, K4 = 0.0015, 8.0
    # Legacy state: total feed moles
    n_h2o = 0.7
    n_co2 = 0.3
    # Pick an x1w in the legacy bracket and back out n_salt to make the legacy
    # equation satisfied at that x1w.
    x1w = 0.95
    dK = K4 - K1
    u = 1.0 - K1 * x1w
    v = n_h2o - K1 * x1w
    Naq = K4 * v / (u * x1w * dK)
    # Solve legacy residual = 0 for n_salt: 0 = x1w + 2 ns/Naq + u/K4 − 1
    n_salt = (1.0 - x1w - u / K4) * Naq / 2.0

    R_legacy = legacy_eq_x1w_nacl(x1w, K1, K4, n_h2o, n_co2, n_salt)
    assert abs(R_legacy) < 1e-12, f"legacy residual not zero: {R_legacy}"

    # Multi-salt mapping: ms_aq = ns / (Naq · x1w · M_w);
    # NaCl-1:1 ⇒ m_total = 2 ms_aq.
    ms_aq = n_salt / (Naq * x1w * Mw)
    m_tot = 2.0 * ms_aq
    mi = np.array([ms_aq, ms_aq, 0.0, 0.0, 0.0, 0.0])  # Na+, Cl-
    assert abs(float(np.sum(mi)) - m_tot) < 1e-12

    # Single-gas (CO2) limit: y_CO2 = 1 − y_H2O = 1 − K1·x1w; y_CH4 = 0
    y_co2 = 1.0 - K1 * x1w
    y_h2o = K1 * x1w
    y = np.array([y_h2o, 0.0, y_co2])
    K = np.array([K1, np.nan, K4])  # K_CH4 = NaN since CH4 absent

    R_new = aqueous_closure(x1w, K, y, mi)
    assert abs(R_new) < 1e-12, (
        f"multi-salt closure should match legacy NaCl-1:1 form at this x1w: "
        f"R_new={R_new}")


def test_closed_form_x1w_matches_residual():
    """``compute_aqueous_x1w_from_KP`` should drive the residual to zero."""
    K = np.array([0.001, 5.0, 8.0])
    y = np.array([0.0005, 0.6, 0.3995])  # arbitrary, sums to 1
    mi = np.array([1.5, 1.5, 0.5, 0.2, 0.0, 0.2])  # mixed multi-salt
    # Charge balance: 1.5 + 0.5 + 2*0.2 = 2.4 (cations);
    # 1.5 + 2*0.2 = 1.9 (anions); not balanced, but this routine doesn't need it
    x1w, x_ch4, x_co2 = compute_aqueous_x1w_from_KP(K, y, mi)
    R = aqueous_closure(x1w, K, y, mi)
    assert abs(R) < 1e-12, f"closed-form x1w should satisfy closure exactly: R={R}"
    # Sanity: x_aq + x_ion sums to ≤1
    m_tot = float(np.sum(mi))
    x_ion = Mw * x1w * m_tot
    assert abs(x1w + x_ch4 + x_co2 + x_ion - 1.0) < 1e-12


def main():
    test_reduction_to_nacl_1_1()
    print("PASS  reduction to NaCl-1:1 legacy form")
    test_closed_form_x1w_matches_residual()
    print("PASS  closed-form x1w drives residual to zero")
    print("\nAll aqueous-closure tests pass.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
