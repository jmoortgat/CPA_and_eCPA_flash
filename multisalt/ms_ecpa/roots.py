"""Is this root the one we asked for?

The three-variable aqueous closure has more than one solution, and a Newton
that converges has only proved it found *a* root, not the liquid one. Four of
the 799 brine-density rows in the Papers II/III figure converge cleanly onto
Zw ~ 350 and report densities of 0.8 to 5.5 kg/m3 where the measurement is
1100 to 1215 kg/m3 -- at 200 to 500 bar, where nothing is that light. They are
not inaccurate, they are meaningless, and they were carried into the figure
and into the reported AAD, which they alone inflate from 1.25 % to 1.74 %.

This is the same shape as the stale ``_accept_v_gas`` predicate: a check that
decides which root to keep, sitting far from the code that produces it, and
silently wrong. The lesson applied here is that a solver must be asked to
prove the root is the phase that was requested, and a seed ladder must run on
"converged to the wrong branch" and not only on "failed to converge".

The criterion is parameter-free
-------------------------------
A gas-free aqueous brine is a compressed liquid. Salt raises its density and
compression raises it further, so it cannot be much lighter than the
saturated liquid density of pure water at the same temperature -- which
``water.rho_w_sat`` already gives us (IAPWS-95 eq 2.6). Bracketing
against that, rather than against a hard-coded density window, keeps the test
valid across the whole 273-647 K range without a magic number per salt.

The brackets are deliberately loose. They exist to reject roots that are wrong
by two or three orders of magnitude, not to adjudicate a few per cent -- that
is the model's job, not the guard's.
"""
from __future__ import annotations

import numpy as np

from .water import rho_w_sat
from . import parameters as _P

#: A liquid brine may not fall below this fraction of pure water's saturated
#: liquid density at the same T. Saturated MgCl2 sits near 1.4x that density
#: and the lightest brine in GasBrineBench near 0.92x, so 0.5 is far outside
#: anything real while still catching a root that is off by 100x.
LIQUID_LO = 0.5

#: ... nor above this multiple. The densest brine measured is ~1.4x.
LIQUID_HI = 3.0


def liquid_density_bounds(T_K: float) -> tuple[float, float]:
    """(lo, hi) mass-density bounds [kg/m3] for an aqueous liquid at ``T_K``."""
    rho_w = float(rho_w_sat(T_K)) * _P.Mw          # mol/m3 -> kg/m3
    return LIQUID_LO * rho_w, LIQUID_HI * rho_w


def is_liquid_root(rho_kg_m3, T_K: float) -> bool:
    """True when ``rho`` could be a liquid aqueous phase at ``T_K``."""
    r = float(np.real(rho_kg_m3))
    if not np.isfinite(r) or r <= 0.0:
        return False
    lo, hi = liquid_density_bounds(T_K)
    return lo <= r <= hi
