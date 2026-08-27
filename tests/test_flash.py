"""Regression tests for the eCPA K-value flash.

Reference values were computed with the released code at v1.0.0. They pin
the two-phase split of the CO2 + H2O + NaCl ternary at three representative
conditions spanning the validated window (storage-aquifer to deep
geothermal). Tolerances are loose enough for cross-platform floating-point
variation and tight enough to catch any change in the physics.
"""

import pytest

from ecpa.flash import flash_co2_h2o_salt_kv

# (T [K], P [bar], z_CO2, m_NaCl [mol/kg]) -> (x_CO2_aq, beta, y_H2O)
REFERENCE = {
    (323.15, 100.0, 0.3, 1.0): (1.670443e-02, 0.281389, 2.727190e-03),
    (323.15, 100.0, 0.3, 3.0): (1.108594e-02, 0.271518, 2.502037e-03),
    (423.15, 500.0, 0.5, 6.0): (9.465707e-03, 0.471360, 5.325096e-02),
}


@pytest.mark.parametrize("cond,expected", REFERENCE.items(),
                         ids=[f"T{c[0]:.0f}_P{c[1]:.0f}_m{c[3]:.0f}"
                              for c in REFERENCE])
def test_flash_kv_regression(cond, expected):
    T, P, z, m = cond
    x_ref, beta_ref, y_ref = expected
    r = flash_co2_h2o_salt_kv(T, P, z, m)
    assert r["x_aq"]["x4w"] == pytest.approx(x_ref, rel=1e-4)
    assert r["beta"] == pytest.approx(beta_ref, rel=1e-4)
    assert r["x_c"]["x1c"] == pytest.approx(y_ref, rel=1e-4)


def test_salting_out_monotonic():
    """CO2 solubility must decrease with NaCl molality at fixed (T, P)."""
    sols = []
    for m in (0.5, 2.0, 4.0):
        r = flash_co2_h2o_salt_kv(323.15, 100.0, 0.3, m)
        sols.append(r["x_aq"]["x4w"])
    assert sols[0] > sols[1] > sols[2]


def test_mole_fractions_normalized():
    r = flash_co2_h2o_salt_kv(373.15, 200.0, 0.4, 2.0)
    assert sum(r["x_aq"].values()) == pytest.approx(1.0, abs=1e-10)
    assert sum(r["x_c"].values()) == pytest.approx(1.0, abs=1e-10)
    assert 0.0 < r["beta"] < 1.0


def test_salt_free_rejected():
    """The eCPA entry point requires m > 0 (use the CPA module salt-free)."""
    with pytest.raises(ValueError):
        flash_co2_h2o_salt_kv(323.15, 100.0, 0.3, 0.0)
