"""Verify the slot-2 water cross-association machinery (S12A).

Production S12A = 0.0 is an analytic no-op (chi2 = 1 identically);
the bit-exactness of the baseline is enforced by the golden-case
regression tests elsewhere in this suite. Here:

1. Slot-swap symmetry: clone the CO2 (slot-3) parameters into slot 2
   and set S12A = S13(T). The kernels must then be exactly symmetric
   under swapping the slot-2/slot-3 compositions — one test that
   exercises every mirrored term (association, the analytic
   chi-derivative system, and the permittivity chains) in both the
   aqueous and gas kernels.
2. Direction sanity: S12A > 0 must lower lnphi of the slot-2 gas in
   the water-rich phase (solvation attracts the gas into water) and
   lower lnphi of water in the gas-rich phase.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import ms_ecpa.parameters as prm  # noqa: E402
import ms_ecpa.lnphi as ln        # noqa: E402
from ms_ecpa.lnphi import lnphi_aq, lnphi_gas  # noqa: E402


def _clone_slot3_into_slot2(T):
    orig = {n: getattr(prm, n) for n in
            ("Tc2", "Pc2", "b2", "a02", "c12", "pol2",
             "Ak12", "Bk12", "Ck12", "S12A")}
    orig_AU = prm.A_U2i.copy()
    orig_BU = prm.B_U2i.copy()
    vals = dict(
        Tc2=prm.Tc3, Pc2=prm.Pc3, b2=prm.b3, a02=prm.a03,
        c12=prm.c13, pol2=prm.pol3,
        # k12(T) on T/Tc1 must reproduce k13(T) on T/Tc3:
        Ak12=prm.Ak13 * (prm.Tc1 / prm.Tc3) ** 2,
        Bk12=prm.Bk13 * (prm.Tc1 / prm.Tc3),
        Ck12=prm.Ck13,
    )
    for ns in (prm, ln):
        for name, v in vals.items():
            if hasattr(ns, name):
                setattr(ns, name, v)
    prm.A_U2i[:] = prm.A_U3i
    prm.B_U2i[:] = prm.B_U3i
    prm.S12A = (prm.AS13 * (T / prm.Tc3) ** 2
                + prm.BS13 * (T / prm.Tc3) + prm.CS13)
    return orig, orig_AU, orig_BU


def _restore(orig, orig_AU, orig_BU):
    for name, v in orig.items():
        for ns in (prm, ln):
            if hasattr(ns, name):
                setattr(ns, name, v)
    prm.A_U2i[:] = orig_AU
    prm.B_U2i[:] = orig_BU


def test_slot_swap_symmetry():
    T, P_Pa = 323.15, 80e5
    mi = np.array([1.0, 1.0, 0, 0, 0, 0], float)
    la_ref = lnphi_aq((0.96, 0.0, 0.03), mi, T, P_Pa)
    lg_ref = lnphi_gas((0.02, 0.0, 0.98), T, P_Pa)
    orig = _clone_slot3_into_slot2(T)
    try:
        la_sw = lnphi_aq((0.96, 0.03, 0.0), mi, T, P_Pa)
        lg_sw = lnphi_gas((0.02, 0.98, 0.0), T, P_Pa)
    finally:
        _restore(*orig)
    assert abs(la_sw[0] - la_ref[0]) < 1e-12
    assert abs(la_sw[1] - la_ref[2]) < 1e-12
    assert abs(lg_sw[0] - lg_ref[0]) < 1e-12
    assert abs(lg_sw[1] - lg_ref[2]) < 1e-12


def test_s12a_direction():
    T, P_Pa = 310.0, 20e5
    mi = np.zeros(6)
    la0 = lnphi_aq((0.9995, 0.0005, 0.0), mi, T, P_Pa)
    lg0 = lnphi_gas((0.005, 0.995, 0.0), T, P_Pa)
    old = prm.S12A
    try:
        prm.S12A = 0.05
        la1 = lnphi_aq((0.9995, 0.0005, 0.0), mi, T, P_Pa)
        lg1 = lnphi_gas((0.005, 0.995, 0.0), T, P_Pa)
    finally:
        prm.S12A = old
    # Solvation lowers the slot-2 gas fugacity coefficient in water ...
    assert la1[1] < la0[1]
    # ... and water's in the gas-rich phase.
    assert lg1[0] < lg0[0]


if __name__ == "__main__":
    test_slot_swap_symmetry()
    test_s12a_direction()
    print("S12A verification passed")
