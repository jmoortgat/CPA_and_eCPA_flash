"""Pure-water auxiliary correlations (not part of the EoS itself)."""
from __future__ import annotations

import numpy as np


def rho_w_sat(T):
    """Saturated-liquid molar density of pure water [mol/m3].

    Wagner & Pruss auxiliary saturation equation (IAPWS-95, eq 2.6),
    rho'/rho_c = 1 + sum b_k tau^(k/3 exponents); valid 273.16-647 K.
    Used by the liquid-branch root guard (roots.py) and by the
    molality -> molarity conversions of the colligative scripts.
    T is a plain real float here (never complex-step perturbed).
    """
    Tc, rhoc = 647.096, 322.0  # K, kg/m3
    tau = 1.0 - min(float(np.real(T)), Tc - 1e-6) / Tc
    r = (1.0 + 1.99274064 * tau ** (1 / 3) + 1.09965342 * tau ** (2 / 3)
         - 0.510839303 * tau ** (5 / 3) - 1.75493479 * tau ** (16 / 3)
         - 45.5170352 * tau ** (43 / 3) - 6.74694450e5 * tau ** (110 / 3))
    return r * rhoc / 0.018015268  # kg/m3 -> mol/m3
