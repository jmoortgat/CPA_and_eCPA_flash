"""The gas-side association fraction must stay on the physical branch.

The closure ``chi (Z + 2 y1 chi Delta) = Z`` is a quadratic with one
positive and one negative root.  At liquid-like ``Zh`` -- the regime the
water-rich TPD trial compositions probe -- the unconstrained 2x2 Newton in
``lnphi_gas`` could settle on the NEGATIVE root: it satisfies the residual
to machine precision, so the solver reported success, and then
``4 * log(chi)`` produced a silent NaN for the water fugacity coefficient.
``lnphi_aq`` has carried the mirror-image guard (``chi1w > 0``) on the
aqueous side for some time; these tests pin the gas-side one.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ms_ecpa.lnphi import (  # noqa: E402
    _gas_assoc_state,
    _gas_chi_positive_branch,
    _gas_chi_triple,
    _gas_lnphi_evaluate,
    _gas_residual,
    _gas_root_is_physical,
    lnphi_gas,
)
from ms_ecpa.newton import newton_solve  # noqa: E402

#: A state with a liquid-like Zh where the water-rich trial composition
#: drives the gas Newton onto the negative chi branch.  (T, P_bar, y_gas)
WATER_RICH = [
    (324.65, 101.325, (0.70, 0.30, 0.0)),
    (324.65, 202.650, (0.70, 0.30, 0.0)),
    (323.15, 405.300, (0.90, 0.10, 0.0)),
    (325.00, 405.300, (0.95, 0.05, 0.0)),
    (310.00, 152.000, (0.70, 0.30, 0.0)),
]


@pytest.mark.parametrize("T,P_bar,y", WATER_RICH)
def test_lnphi_gas_never_returns_nan(T, P_bar, y):
    """Whatever happens, lnphi_gas returns finite numbers or raises --
    never a NaN that a caller can absorb without noticing."""
    try:
        out = lnphi_gas(y, T, P_bar * 1e5)
    except RuntimeError:
        return                       # an explicit failure is acceptable
    assert np.all(np.isfinite(out)), f"NaN lnphi at {y}, {T} K, {P_bar} bar"


@pytest.mark.parametrize("T,P_bar,y", WATER_RICH)
def test_accepted_root_is_on_the_physical_branch(T, P_bar, y):
    """Reach into the kernel: whatever root lnphi_gas settles on, every
    one of the three monomer fractions must be strictly positive."""
    P = P_bar * 1e5
    try:
        lnphi_gas(y, T, P)
    except RuntimeError:
        return
    # re-derive the accepted root the way lnphi_gas does
    v, ok, _ = newton_solve(_gas_residual, np.array([1.0, 0.99]),
                            args=(np.asarray(y, float), T, P),
                            tol=1e-10, maxiter=80)
    if ok and _gas_root_is_physical(v, np.asarray(y, float), T, P):
        tri = _gas_chi_triple(v[0], v[1], np.asarray(y, float), T, P)
        assert all(c > 0.0 for c in tri)


def test_negative_branch_is_rejected_by_the_predicate():
    """The negative root of the water-water quadratic is a genuine root
    of the residual; the predicate must still reject it."""
    T, P = 324.65, 101.325e5
    y = np.array([0.70, 0.30, 0.0])
    st = _gas_assoc_state(0.092332, y, T, P)
    assert st is not None
    _B, DELTA, _S13, _S12A = st
    Z = 0.092332
    u = 2.0 * y[0] * DELTA
    disc = np.sqrt(Z * Z + 4.0 * u * Z)
    pos = 2.0 * Z / (Z + disc)
    neg = -(Z + disc) / (2.0 * u)
    assert neg < 0.0 < pos <= 1.0
    # both satisfy the water-water closure; only one is admissible
    for c in (pos, neg):
        assert abs(c * (Z + u * c) - Z) < 1e-12 * max(1.0, abs(Z))
    assert _gas_root_is_physical(np.array([Z, pos]), y, T, P)
    assert not _gas_root_is_physical(np.array([Z, neg]), y, T, P)


def test_positive_branch_seed_is_in_range_and_solves_the_closure():
    T, P = 324.65, 101.325e5
    y = np.array([0.70, 0.30, 0.0])
    for Z in (0.05, 0.092332, 0.3, 0.8, 1.0):
        c = _gas_chi_positive_branch(Z, y, T, P)
        if c is None:
            continue
        assert 0.0 < c <= 1.0 + 1e-12
        tri = _gas_chi_triple(Z, c, y, T, P)
        assert all(q > 0.0 for q in tri)


def test_evaluate_raises_on_a_non_physical_chi():
    """The last line of defence: a non-physical chi must never reach the
    logarithm, even if a future caller forgets to screen its root."""
    T, P = 324.65, 101.325e5
    y = np.array([0.70, 0.30, 0.0])
    with pytest.raises(ValueError, match="non-physical"):
        _gas_lnphi_evaluate(np.array([0.092332, -0.166501]), y, T, P)


def test_outside_the_closure_domain_is_rejected():
    """Zh inside the mixture co-volume makes log(Zh - B) undefined; the
    predicate must refuse such a pair rather than let it through."""
    T, P = 324.65, 400e5
    y = np.array([0.70, 0.30, 0.0])
    assert _gas_assoc_state(1e-6, y, T, P) is None
    assert not _gas_root_is_physical(np.array([1e-6, 0.5]), y, T, P)


@pytest.mark.parametrize("T,P_bar,y", [
    (298.15, 1.0, (0.02, 0.98, 0.0)),
    (373.15, 100.0, (0.10, 0.45, 0.45)),
    (473.15, 300.0, (0.30, 0.20, 0.50)),
    (323.15, 600.0, (0.01, 0.00, 0.99)),
])
def test_ordinary_gas_states_are_untouched(T, P_bar, y):
    """Normal vapour states must still converge and stay physical."""
    out = lnphi_gas(y, T, P_bar * 1e5)
    assert np.all(np.isfinite(out))
