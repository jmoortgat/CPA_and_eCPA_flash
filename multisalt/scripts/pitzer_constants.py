"""Pitzer ion-interaction parameters and osmotic-coefficient model at 25 C.

PROVENANCE
----------
Promoted 2026-07-08 from
``step3_physical_refit_2026_07/pitzer_constants.py``
(verbatim physics; this header only).  The tabulated beta0/beta1/C_phi
values were transcribed from the printed Pitzer & Mayorga (1973) Tables
I/VI and verified during the 2026-07 ion-parameter audit (see
``literature_comparison_2026_07``); the self-test below
anchors NaCl phi(1 m) to the Robinson & Stokes value.

Source
------
K. S. Pitzer and G. Mayorga, "Thermodynamics of Electrolytes. II.
Activity and Osmotic Coefficients for Strong Electrolytes with One or
Both Ions Univalent", J. Phys. Chem. 77 (19), 2300-2308 (1973).

Working equation (their eq 2, p. 2300):

    phi - 1 = |z+ z-| * f_phi
              + m * (2 nu+ nu- / nu) * B_phi
              + m^2 * (2 (nu+ nu-)^{3/2} / nu) * C_phi

with (eqs 5, 8, p. 2300; constants p. 2301):

    f_phi  = -A_phi * sqrt(I) / (1 + b sqrt(I)),   b = 1.2
    B_phi  = beta0 + beta1 * exp(-alpha sqrt(I)),  alpha = 2.0
    A_phi  = 0.392  (25 C, water)
    I      = (1/2) sum_i m_i z_i^2 = (m/2) (nu+ z+^2 + nu- z-^2)
    nu     = nu+ + nu-,  m = stoichiometric molality [mol/kg]

For a symmetric 1-1 salt all the valence factors are unity, so Table I
lists beta0, beta1, C_phi directly.  For 2-1 and 1-2 salts the paper
tabulates the *products* (4/3)beta0, (4/3)beta1 and (2^{5/2}/3)C_phi
(stated p. 2301, col. 1: "we shall tabulate 4/3 b(0), 4/3 b(1), and
(2^{5/2}/3) C_phi ... these numerical factors must be removed before
the virial coefficients are used in work with mixed electrolytes").
The dict below stores the BARE beta0, beta1, C_phi with the table
factors already removed.

Tabulated (raw) values transcribed from the paper:
  Table I (1-1 inorganic, p. 2301):
    NaCl    0.0765   0.2664   0.00127   (max m = 6)
    KCl     0.04835  0.2122  -0.00084   (max m = 4.8)
  Table VI (2-1 inorganic, p. 2302), columns are
  (4/3)b0, (4/3)b1, (2^{5/2}/3)C_phi:
    MgCl2   0.4698   2.242    0.00979   (max m = 4.5)
    CaCl2   0.4212   2.152   -0.00064   (max m = 2.5)
    Na2SO4  0.0261   1.484    0.00938   (max m = 4)   [1-2, same factors]

Bare parameters after removing the factors (4/3 and 2^{5/2}/3 = 1.88562):
    MgCl2   beta0 = 0.35235, beta1 = 1.6815,  C_phi =  0.005192
    CaCl2   beta0 = 0.3159,  beta1 = 1.614,   C_phi = -0.000339
    Na2SO4  beta0 = 0.019575, beta1 = 1.113,  C_phi =  0.004974

Plain numpy, no other dependencies.
"""

import numpy as np

# --- universal constants of the Pitzer-Mayorga 1973 formulation (25 C) ---
A_PHI = 0.392   # Debye-Hueckel coefficient for the osmotic function, p. 2301
B_PITZER = 1.2  # kg^{1/2} mol^{-1/2}, p. 2301
ALPHA = 2.0     # kg^{1/2} mol^{-1/2}, p. 2301

_F43 = 4.0 / 3.0
_FC21 = 2.0 ** 2.5 / 3.0  # = 1.8856181, factor on C_phi for 2-1 / 1-2 salts

SALTS = {
    # 1-1: Table I values are the bare parameters.
    "NaCl": {
        "nu_plus": 1, "nu_minus": 1, "z_plus": 1, "z_minus": -1,
        "beta0": 0.0765, "beta1": 0.2664, "Cphi": 0.00127,
    },
    "KCl": {
        "nu_plus": 1, "nu_minus": 1, "z_plus": 1, "z_minus": -1,
        "beta0": 0.04835, "beta1": 0.2122, "Cphi": -0.00084,
    },
    # 2-1: Table VI values divided by 4/3 (betas) and 2^{5/2}/3 (Cphi).
    "MgCl2": {
        "nu_plus": 1, "nu_minus": 2, "z_plus": 2, "z_minus": -1,
        "beta0": 0.4698 / _F43, "beta1": 2.242 / _F43,
        "Cphi": 0.00979 / _FC21,
    },
    "CaCl2": {
        "nu_plus": 1, "nu_minus": 2, "z_plus": 2, "z_minus": -1,
        "beta0": 0.4212 / _F43, "beta1": 2.152 / _F43,
        "Cphi": -0.00064 / _FC21,
    },
    # 1-2: same valence factors as 2-1 (nu+ nu- = 2, nu = 3, |z+z-| = 2).
    "Na2SO4": {
        "nu_plus": 2, "nu_minus": 1, "z_plus": 1, "z_minus": -2,
        "beta0": 0.0261 / _F43, "beta1": 1.484 / _F43,
        "Cphi": 0.00938 / _FC21,
    },
}


def phi_osm(m, salt):
    """Osmotic coefficient phi(m) at 25 C, Pitzer-Mayorga 1973 eq 2.

    Parameters
    ----------
    m : float or array
        Stoichiometric salt molality [mol/kg water].
    salt : str
        Key into SALTS.

    Returns
    -------
    phi : float or array
    """
    p = SALTS[salt]
    m = np.asarray(m, dtype=float)
    nup, num = p["nu_plus"], p["nu_minus"]
    zp, zm = p["z_plus"], p["z_minus"]
    nu = nup + num

    I = 0.5 * m * (nup * zp**2 + num * zm**2)
    sqI = np.sqrt(I)

    f_phi = -A_PHI * sqI / (1.0 + B_PITZER * sqI)
    B_phi = p["beta0"] + p["beta1"] * np.exp(-ALPHA * sqI)

    phi = (1.0
           + abs(zp * zm) * f_phi
           + m * (2.0 * nup * num / nu) * B_phi
           + m**2 * (2.0 * (nup * num) ** 1.5 / nu) * p["Cphi"])
    return phi[()] if phi.ndim == 0 else phi


if __name__ == "__main__":
    print("Pitzer-Mayorga 1973 osmotic coefficients at 25 C")
    print(f"{'salt':<8}{'beta0':>10}{'beta1':>10}{'Cphi':>11}"
          f"{'phi(0.001m)':>13}{'phi(1m)':>10}")
    ok = True
    for salt in SALTS:
        p = SALTS[salt]
        phi_dil = phi_osm(1e-3, salt)
        phi_1 = phi_osm(1.0, salt)
        print(f"{salt:<8}{p['beta0']:>10.5f}{p['beta1']:>10.4f}"
              f"{p['Cphi']:>11.6f}{phi_dil:>13.4f}{phi_1:>10.4f}")
        # dilute limit: phi -> 1 as m -> 0
        if not abs(phi_dil - 1.0) < 0.05:
            ok = False
            print(f"  FAIL: {salt} dilute limit phi(0.001) = {phi_dil}")
    # literature anchor: NaCl phi(1m) ~ 0.936 (Robinson & Stokes)
    phi_nacl = phi_osm(1.0, "NaCl")
    if abs(phi_nacl - 0.936) > 0.003:
        ok = False
        print(f"FAIL: NaCl phi(1m) = {phi_nacl:.4f}, expected ~0.936")
    print("self-test:", "PASS" if ok else "FAIL")
