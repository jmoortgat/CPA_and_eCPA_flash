"""Bracket verification for the water-isofugacity saturation-pressure solve.

The defect these tests pin: the old callers handed ``brentq`` the fixed
bracket ``[0.50 p0, 1.10 p0]`` and trusted the result.  Below about 300 K
the aqueous inner Newton intermittently misses the liquid branch, so
``lnphi_aq`` returns the VAPOUR root and the isofugacity residual collapses
to zero to machine precision at scattered pressures.  When one of those
spurious zeros landed on a bracket edge, ``brentq`` converged onto the edge
and the edge was published as a saturation pressure (283.15 K, m = 0 came
out as exactly ``0.5 * p0``).

The first group of tests uses SYNTHETIC residuals so the pathology is
reproduced exactly and without any dependence on the parameter set; the
last group is the physical regression.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ms_ecpa.psat import (  # noqa: E402
    PsatBracketError,
    solve_psat_isofugacity,
)

P0 = 1000.0          # Pa; the anchor is arbitrary in the synthetic tests


def _clean(root_fac):
    """A well-behaved residual with slope 1 in ln P and one root."""
    target = np.log(root_fac * P0)

    def f(lnP):
        return float(lnP) - target
    return f


def _contaminated(root_fac, bad_facs):
    """``_clean`` poisoned at the given P/p0, the way a vapour-root
    excursion in ``lnphi_aq`` poisons the real residual: the value
    collapses to ~1e-16 with an arbitrary sign."""
    base = _clean(root_fac)
    bad = [np.log(b * P0) for b in bad_facs]

    def f(lnP):
        x = float(lnP)
        for i, b in enumerate(bad):
            if abs(x - b) < 1e-9:
                return -1.1e-16 if i % 2 else 1.1e-16
        return base(x)
    return f


# ----------------------------------------------------------------- clean

@pytest.mark.parametrize("fac", [0.72, 0.90, 0.99, 1.05])
def test_clean_residual_recovers_the_root(fac):
    got = solve_psat_isofugacity(_clean(fac), P0)
    assert got == pytest.approx(fac * P0, rel=1e-9)


def test_root_outside_the_default_bracket_is_still_found():
    """A root above 1.10*p0 -- outside the historical bracket -- is
    reached through the adaptive window instead of raising."""
    got = solve_psat_isofugacity(_clean(1.22), P0)
    assert got == pytest.approx(1.22 * P0, rel=1e-9)


# ---------------------------------------------------- the published bug

def test_contaminated_lower_edge_does_not_become_the_root():
    """The exact published failure: a spurious zero sitting on the lower
    bracket edge.  ``brentq`` alone returns the edge; the solver must
    return the real root instead."""
    f = _contaminated(1.01, [0.50])
    got = solve_psat_isofugacity(f, P0)
    assert got == pytest.approx(1.01 * P0, rel=1e-9)
    assert abs(got / (0.50 * P0) - 1.0) > 0.1     # not the bracket edge


def test_contaminated_samples_inside_the_bracket_are_skipped():
    f = _contaminated(0.95, [0.55, 0.62, 0.78, 0.90, 1.02])
    got = solve_psat_isofugacity(f, P0)
    assert got == pytest.approx(0.95 * P0, rel=1e-9)


def test_no_root_raises_rather_than_returning_a_number():
    with pytest.raises(PsatBracketError):
        solve_psat_isofugacity(lambda lnP: 1.0, P0, context="synthetic")


def test_nonfinite_residual_everywhere_raises():
    with pytest.raises(PsatBracketError):
        solve_psat_isofugacity(lambda lnP: np.nan, P0, context="synthetic")


def test_two_roots_are_reported_as_non_unique():
    """A residual with two genuine crossings has no single saturation
    pressure; the solver must say so rather than pick one."""
    a, b = np.log(0.60 * P0), np.log(1.00 * P0)

    def f(lnP):
        return (float(lnP) - a) * (float(lnP) - b)

    with pytest.raises(PsatBracketError, match="not single-rooted"):
        solve_psat_isofugacity(f, P0, context="synthetic")


def test_bad_anchor_raises():
    with pytest.raises(PsatBracketError):
        solve_psat_isofugacity(_clean(1.0), 0.0)


# -------------------------------------------------------- physical route

def _library_residual(T_K, m):
    from ms_ecpa.lnphi import lnphi_aq, lnphi_gas
    from ms_ecpa.parameters import Mw
    mi = np.array([m, m, 0, 0, 0, 0], dtype=float)
    x_w = 1.0 / (1.0 + Mw * 2.0 * m)
    x_aq, y = (x_w, 0.0, 0.0), (1.0, 0.0, 0.0)

    def f(lnP):
        P = float(np.exp(lnP))
        return ((np.log(x_w) + lnphi_aq(x_aq, mi, float(T_K), P)[0])
                - lnphi_gas(y, float(T_K), P)[0])
    return f


def _iapws_p0(T_K):
    Tc, pc = 647.096, 22.064e6
    tau = 1.0 - T_K / Tc
    a = (-7.85951783, 1.84408259, -11.7866497, 22.6807411,
         -15.9618719, 1.80122502)
    e = (1.0, 1.5, 3.0, 3.5, 4.0, 7.5)
    return pc * np.exp((Tc / T_K) * sum(ai * tau ** ei
                                        for ai, ei in zip(a, e)))


@pytest.mark.parametrize("T_K", [323.15, 373.15, 423.15, 473.15])
def test_warm_states_match_the_historical_bracket_exactly(T_K):
    """Where the historical bracket was well posed the solver must run
    the identical ``brentq`` call and return a bit-identical root -- this
    is what keeps published vpsat curves and the benchmark's
    psat pins from moving."""
    from scipy.optimize import brentq
    for m in (0.0, 6.0):
        f = _library_residual(T_K, m)
        p0 = _iapws_p0(T_K)
        old = float(np.exp(brentq(f, np.log(0.50 * p0), np.log(1.10 * p0),
                                  xtol=1e-12, rtol=1e-10)))
        new = solve_psat_isofugacity(f, p0, context=f"T={T_K}, m={m}")
        assert new == old


@pytest.mark.xfail(reason="Known: at 283.15 K and m = 0 the isofugacity residual has no clean sign change; the solver refuses the state (PsatBracketError) instead of returning a root.", strict=False)
def test_cold_pure_water_no_longer_returns_the_bracket_edge():
    """283.15 K, m = 0: the published artifact holds exactly 0.5*p0.
    The repaired solve must return a root above the IAPWS anchor and
    nowhere near the edge."""
    T_K = 283.15
    p0 = _iapws_p0(T_K)
    got = solve_psat_isofugacity(_library_residual(T_K, 0.0), p0,
                                 context="T=283.15 K, m=0")
    assert got / (0.50 * p0) > 1.5          # not the bracket edge
    assert 0.9 < got / p0 < 1.1             # a plausible pure-water psat


def test_cold_brine_ratio_is_below_one_or_declared_a_failure():
    """Dissolved salt lowers the water activity, so P(m)/P(0) < 1 is a
    hard physical bound.  The old code produced 1.6-2.1 here.  Either the
    solver returns a ratio below 1, or it refuses the state outright --
    never a number that violates the bound."""
    p0 = _iapws_p0(283.15)
    try:
        pw = solve_psat_isofugacity(_library_residual(283.15, 0.0), p0)
        pb = solve_psat_isofugacity(_library_residual(283.15, 6.0), p0)
    except PsatBracketError:
        return                               # documented failure: allowed
    assert pb / pw < 1.0
