"""Stability of water-rich brine feeds near the CO2 saturation of the brine.

The gas-side reference of ``michelsen_tpd`` has no ions, so without the
second test against the brine reference (``BRINE_REFERENCE_FOR_WATER_RICH``)
a brine with more CO2 than its saturation value but less than the salt-free
saturation value is reported single-phase.  Each case below was reported
single-phase before the brine reference was added.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

import ms_ecpa.stability as stability  # noqa: E402
from ms_ecpa.flash import flash_kv  # noqa: E402
from ms_ecpa.stability import stability_flash  # noqa: E402

# (T [K], P [bar], NaCl molality, feed z_CO2 as a multiple of the saturation value)
CASES = [
    (323.15, 60.0, 1.0, 1.05),
    (373.15, 60.0, 4.0, 1.50),
    (423.15, 60.0, 6.0, 1.75),
]


def _saturation(T, P, mi):
    """CO2 mole fraction of the saturated brine, partitioning basis."""
    r = flash_kv(T, P, np.array([0.5, 0.0, 0.5]), mi)
    assert r.converged and 0.0 < r.beta < 1.0
    x = np.asarray(r.x_aq, dtype=float)
    return x[2] / x.sum()


def _feed(zc):
    return np.array([1.0 - zc, 0.0, zc])


@pytest.mark.parametrize("T,P,m,factor", CASES)
def test_supersaturated_brine_splits(T, P, m, factor):
    mi = np.array([m, m, 0.0, 0.0, 0.0, 0.0])
    zsat = _saturation(T, P, mi)
    r = stability_flash(T, P, _feed(factor * zsat), mi)
    assert r.phases == 2
    assert r.stability.reference == "brine"
    assert r.flash_result is not None and 0.0 < r.flash_result.beta < 1.0
    # the aqueous phase is the saturated brine
    x = np.asarray(r.flash_result.x_aq, dtype=float)
    assert x[2] / x.sum() == pytest.approx(zsat, rel=1e-6)


@pytest.mark.parametrize("T,P,m,factor", CASES)
def test_undersaturated_brine_single_phase(T, P, m, factor):
    mi = np.array([m, m, 0.0, 0.0, 0.0, 0.0])
    zsat = _saturation(T, P, mi)
    r = stability_flash(T, P, _feed(0.95 * zsat), mi)
    assert r.phases == 1


def test_original_reference_misses_split(monkeypatch):
    T, P, m, factor = CASES[0]
    mi = np.array([m, m, 0.0, 0.0, 0.0, 0.0])
    zsat = _saturation(T, P, mi)
    monkeypatch.setattr(stability, "BRINE_REFERENCE_FOR_WATER_RICH", False)
    r = stability_flash(T, P, _feed(factor * zsat), mi)
    assert r.phases == 1
