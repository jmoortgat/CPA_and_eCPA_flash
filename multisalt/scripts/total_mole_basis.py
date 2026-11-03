#!/usr/bin/env python3
"""Total-mole-basis water activity and osmotic coefficient (production).

PROVENANCE
----------
Promoted 2026-07-08 from
``colligative_note_2026_07/total_mole_basis.py`` (rev 2).
The physics core (``dU_wi``, ``a_w_pure``, ``hv_a_mix``, ``solve_chi``,
``solve_epsr``, ``A_r``, ``pressure``, ``solve_V``, ``ln_f_w``) is adapted
verbatim from the independent reference implementation in
``cacl2_audit_2026_07/audit_terms.py`` (PART 2, basis
"total"), which the 2026-07-08 forensic audit verified term-by-term
against the production code and against the published equations
(Schlaikjer 2018 eqs 1-7; Maribo-Mogensen 2015 eqs 3-15).  On the
total-mole basis it reproduces Schlaikjer 2018's own Fig. 3 Na2SO4 curve
to <= 0.01 in phi at 298 and 373 K, the Pitzer NaCl targets at the ~1 %
level, and the Debye-Hueckel limiting law essentially exactly.

Basis: ions are ordinary mole-counted components -- they enter the
translational term ``-n_T ln(1 - B/V)`` and the P(V) closure (the
published formulation).  ``ln a_w = ln f_w(brine) - ln f_w(pure water)``
at equal (T, P); no hand-added ``ln x_w``.

Parameter binding (production behaviour, differs from the pilot)
----------------------------------------------------------------
Ion parameters are snapshotted from ``ms_ecpa.parameters`` into the
module-level namespace ``IP`` by :func:`reload_parameters` (called once
at import).  Nothing is hardcoded, so a parameter adoption in
``ms_ecpa/parameters.py`` propagates here automatically on the next
process start (or after an explicit :func:`reload_parameters` call).

For provenance figures only, :func:`printed_ca_row` is a context manager
that temporarily rebinds the Ca2+ slot to the *printed* Schlaikjer 2018
Table 4 row (sigma = 1.17 A, R_B = 1.73 A, dU_ref = 595.83 R, Gamma =
6261.18 K, alpha_0 = 5.138e-40 C^2 m^2/J, plus the pre-correction
Peneloux coefficients), regardless of what the production parameter set
currently carries.  See Paper II SI Section S8.3 for the
physically-bracketed repair that supersedes that row.

Component vector n = [n_w, n_Na, n_Cl, n_K, n_Ca, n_Mg, n_SO4]  [mol],
built for 1 kg of water (n_w = 55.5084).
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import brentq

_CODE = Path(__file__).resolve().parents[1]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

import ms_ecpa.parameters as P                    # noqa: E402
from ms_ecpa.water import rho_w_sat              # noqa: E402,F401
from ms_ecpa import extensions as _ext            # noqa: E402

R, Na, kb, e, eps0 = P.R, P.Na, P.kb, P.e, P.eps0
N_W = 55.5084          # mol water / kg water
T_REF = 298.15

# salt -> (cation slot, n cations, anion slot, n anions, nu)
_SALT_STOICH = {
    "NaCl":   (1, 1.0, 2, 1.0, 2.0),
    "KCl":    (3, 1.0, 2, 1.0, 2.0),
    "CaCl2":  (4, 1.0, 2, 2.0, 3.0),
    "MgCl2":  (5, 1.0, 2, 2.0, 3.0),
    "Na2SO4": (1, 2.0, 6, 1.0, 3.0),
}
NU = {salt: s[4] for salt, s in _SALT_STOICH.items()}

_CA = 4          # Ca2+ slot in the 7-component vector (index 3 among ions)

# Printed Schlaikjer 2018 Table 4 Ca2+ row (pre-correction), for the
# provenance overlay only.  Gamma is the a0/(R b) energy coefficient.
_PRINTED_CA_ROW = dict(
    sigma_m=1.17e-10, Rb_m=1.73e-10, UrefR_K=595.83, Gamma_K=6261.18,
    Talfa_K=492.90, alfa_K=376.77, pol=5.138e-40,
    A_Pen=30.4741, B_Pen=-39.4861,
)

# Module-level ion-parameter snapshot; populated by reload_parameters().
IP = SimpleNamespace()


def reload_parameters() -> None:
    """(Re)snapshot the ion parameters from ``ms_ecpa.parameters``."""
    # ``copy=True`` on every row.  ``np.asarray`` of an already-float64
    # array returns that same object, which would leave IP aliasing the
    # production arrays instead of holding a snapshot: printed_ca_row's
    # slot assignments would then write through into ms_ecpa.parameters,
    # and any ms_ecpa call made inside that block would see the printed
    # Ca2+ row in place of the production one.
    IP.B_VEC = np.concatenate([[P.b1], P.bi]).astype(float)   # m3/mol
    IP.ZI = np.concatenate([[0.0], P.Zi]).astype(float)
    IP.SG = np.array(P.Sgi, dtype=float, copy=True)           # ion diameters, m
    IP.RB = np.array(P.Rbi, dtype=float, copy=True)           # Born radii, m
    IP.POL = np.concatenate([[P.pol1], P.poli]).astype(float)
    IP.A0I = np.array(P.a0i, dtype=float, copy=True)          # ion a0, J m3/mol2
    IP.UREF = np.array(P.Uref1i, dtype=float, copy=True)      # J/mol
    IP.TALFA = np.array(P.Talfa1i, dtype=float, copy=True)    # K
    IP.ALFA = np.array(P.alfa1i, dtype=float, copy=True)      # K
    IP.APEN = np.array(P.A_Peni, dtype=float, copy=True)
    IP.BPEN = np.array(P.B_Peni, dtype=float, copy=True)
    # Ion-ion HV reference energies (production all zeros).
    IP.UREF_II = np.array(P.Uref_ii, dtype=float, copy=True)
    if np.any(IP.UREF_II) and not np.array_equal(IP.UREF_II, IP.UREF_II.T):
        raise ValueError(
            "parameters.Uref_ii must be symmetric (DU_ij = DU_ji); use "
            "parameters.set_ion_ion_du() to set entries"
        )
    _PURE_CACHE.clear()


@contextmanager
def printed_ca_row():
    """Temporarily rebind the Ca2+ slot to the printed Schlaikjer 2018 row.

    Only the Ca2+ entries are touched; all other ions keep the current
    production values.  The rebinding is local to ``IP`` (a true snapshot
    -- see :func:`reload_parameters`), so ``ms_ecpa.parameters`` is left
    untouched and a call into ms_ecpa from inside the block still sees the
    production Ca2+ row.  The pure-water reference is unaffected by
    construction (no ions), but the cache is cleared on entry/exit for
    hygiene.
    """
    row = _PRINTED_CA_ROW
    i = _CA - 1                       # ion index (6-vector)
    saved = (IP.B_VEC[_CA], IP.SG[i], IP.RB[i], IP.POL[_CA], IP.A0I[i],
             IP.UREF[i], IP.TALFA[i], IP.ALFA[i], IP.APEN[i], IP.BPEN[i])
    try:
        b_ca = (2.0 / 3.0) * np.pi * row["sigma_m"] ** 3 * Na
        IP.B_VEC[_CA] = b_ca
        IP.SG[i] = row["sigma_m"]
        IP.RB[i] = row["Rb_m"]
        IP.POL[_CA] = row["pol"]
        IP.A0I[i] = row["Gamma_K"] * R * b_ca
        IP.UREF[i] = row["UrefR_K"] * R
        IP.TALFA[i] = row["Talfa_K"]
        IP.ALFA[i] = row["alfa_K"]
        IP.APEN[i] = row["A_Pen"]
        IP.BPEN[i] = row["B_Pen"]
        _PURE_CACHE.clear()
        yield
    finally:
        (IP.B_VEC[_CA], IP.SG[i], IP.RB[i], IP.POL[_CA], IP.A0I[i],
         IP.UREF[i], IP.TALFA[i], IP.ALFA[i], IP.APEN[i], IP.BPEN[i]) = saved
        _PURE_CACHE.clear()


def n_vec(salt: str, m: float) -> np.ndarray:
    """1 kg water + m molal salt -> total-mole vector."""
    n = np.zeros(7)
    n[0] = N_W
    if m > 0:
        ic, nc, ia, na, _ = _SALT_STOICH[salt]
        n[ic] = nc * m
        n[ia] = na * m
    return n


# --------------------------------------------------------------------------
# published-equation assembly (adapted verbatim from audit_terms.py PART 2)
# --------------------------------------------------------------------------

def dU_wi(T):
    """Water-ion HV interaction energy, J/mol (Schlaikjer eq 6)."""
    return (IP.UREF + IP.ALFA * R *
            ((1 - T / IP.TALFA) ** 2 - (1 - T_REF / IP.TALFA) ** 2))


def a_w_pure(T):
    return P.a01 * (1 + P.c11 * (1 - np.sqrt(T / P.Tc1))) ** 2


def hv_a_mix(n, T):
    """a(T,n) intensive on total-mole basis [J m3 mol-2], eqs 4-5/15."""
    nT = n.sum()
    x = n / nT
    b = np.dot(x, IP.B_VEC)
    ai = np.concatenate([[a_w_pure(T)], IP.A0I])   # c1i = 0 for ions
    dU = dU_wi(T)
    xw = x[0]; xi = x[1:]
    gE = (1.0 / b) * np.sum(xw * xi * (IP.B_VEC[0] + IP.B_VEC[1:]) * dU)
    # Ion-ion HV energies (production = all zeros).  Published double sum
    # extended to the ion block: symmetric DU_ij adds x_i x_j (b_i + b_j)
    # DU_ij / b per unordered ion pair -- identical to the production
    # kernels (ms_ecpa/lnphi.py, ms_ecpa/eos.py).  All-zero matrix ->
    # analytically zero, branch skipped, bit-exact baseline.
    if np.any(IP.UREF_II):
        gE += np.dot(xi, IP.UREF_II @ (IP.B_VEC[1:] * xi)) / b
    a = b * (np.dot(x, ai / IP.B_VEC) - gE / np.log(2))
    return a, b, gE


def solve_chi(cw, Bext, V, T):
    """4C water monomer fraction; cw = n_w/V [mol/m3]."""
    eta = Bext / (4 * V)
    g = 1.0 / (1.0 - 1.9 * eta)
    delta = g * P.kappaW * (np.exp(P.epsW / T) - 1.0)   # m3/mol
    q = 2 * cw * delta
    X = (-1.0 + np.sqrt(1.0 + 4 * q)) / (2 * q) if q > 0 else 1.0
    return X, delta, eta, g


def _colligative_dielectric(n, V, T):
    """(density factor, dipole weight) from a registered extension."""
    ext = _ext.get()
    f = getattr(ext, "colligative_dielectric", None) if ext else None
    return (1.0, 1.0) if f is None else f(n, V, T)


def _colligative_helmholtz(n, V, T):
    ext = _ext.get()
    f = getattr(ext, "colligative_helmholtz", None) if ext else None
    return None if f is None else f(n, V, T)


def _ext_cache_key():
    ext = _ext.get()
    f = getattr(ext, "colligative_cache_key", None) if ext else None
    return None if f is None else f()


def solve_epsr(n, V, T):
    """Static permittivity, Maribo eqs 13-14.

    A registered extension (ms_ecpa.extensions) may supply a density
    factor r_d for the Clausius-Mossotti and dipole terms and a weight on
    the water dipole term; both are 1.0 otherwise (the published
    production formulation).
    """
    c = n / V                       # mol/m3 each
    r_d, w_d = _colligative_dielectric(n, V, T)
    M = (Na / (3 * eps0)) * np.dot(c, IP.POL) * r_d
    eps_inf = (2 * M + 1) / (1 - M)
    Bext = np.dot(n, IP.B_VEC)
    X, delta, eta, g = solve_chi(c[0], Bext, V, T)
    Pww = 2 * c[0] * delta * X ** 2
    gw = 1 + P.zww * Pww * np.cos(P.GAMMA1) / (Pww * np.cos(P.THETA1) + 1)
    rhs = (Na / (9 * eps0 * kb * T)) * c[0] * r_d * w_d * gw * P.dip01 ** 2

    def f(er):
        return (2 * er + eps_inf) * (er - eps_inf) / (er * (eps_inf + 2) ** 2) - rhs

    er = brentq(f, eps_inf + 1e-9, 500.0, xtol=1e-12)
    return er, eps_inf, X, gw


def A_r(n, V, T, basis="total", parts=False):
    """Residual Helmholtz energy / RT (extensive), independent assembly."""
    nT_all = n.sum()
    nT_trans = nT_all if basis == "total" else n[0]
    Bext = np.dot(n, IP.B_VEC)
    a, b, gE = hv_a_mix(n, T)
    D = a * nT_all ** 2                       # a n_T^2, basis-independent
    A_srk = (-nT_trans * np.log(1 - Bext / V)
             - D / (Bext * R * T) * np.log(1 + Bext / V))

    cw = n[0] / V
    X, delta, eta, g = solve_chi(cw, Bext, V, T)
    A_assoc = n[0] * 4 * (np.log(X) - X / 2 + 0.5)

    ni = n[1:]
    zi = IP.ZI[1:]
    if ni.sum() > 0:
        er, eps_inf, _, _ = solve_epsr(n, V, T)
        # Debye-Hueckel (eqs 10-11, limiting-law normalisation)
        SnZ2 = np.dot(ni, zi ** 2)
        kap = np.sqrt(e ** 2 * Na ** 2 * SnZ2 /
                      (V * eps0 * er * kb * T * Na))
        chi_i = (1.0 / IP.SG ** 3) * (np.log(1 + kap * IP.SG) - kap * IP.SG
                                      + 0.5 * (kap * IP.SG) ** 2)
        A_dh_J = -(V * kb * T / (4 * np.pi)) * np.dot(ni * zi ** 2, chi_i) / SnZ2
        A_dh = A_dh_J / (R * T)
        # Born (eq 12)
        A_born_J = (Na * e ** 2 / (8 * np.pi * eps0) *
                    (1.0 / er - 1.0) * np.dot(ni * zi ** 2, 1.0 / IP.RB))
        A_born = A_born_J / (R * T)
    else:
        A_dh = 0.0
        A_born = 0.0

    A_x = _colligative_helmholtz(n, V, T)   # extension term, or None
    if parts:
        d = dict(srk=A_srk, assoc=A_assoc, dh=A_dh, born=A_born,
                 total=A_srk + A_assoc + A_dh + A_born)
        if A_x is not None:
            d["ext"] = A_x
            d["total"] = d["total"] + A_x
        return d
    if A_x is not None:
        return A_srk + A_assoc + A_dh + A_born + A_x
    return A_srk + A_assoc + A_dh + A_born


def pressure(n, V, T, basis="total"):
    """P = -RT dA^r/dV + nT_trans RT/V (numerical dV)."""
    h = V * 1e-7
    dA = (A_r(n, V + h, T, basis)
          - A_r(n, V - h, T, basis)) / (2 * h)
    nT_trans = n.sum() if basis == "total" else n[0]
    return -R * T * dA + nT_trans * R * T / V


def solve_V(n, T, Pp, basis="total"):
    """Liquid-branch volume root: march out from the co-volume."""
    Bext = np.dot(n, IP.B_VEC)
    Vlo = Bext * 1.02
    f_lo = pressure(n, Vlo, T, basis) - Pp
    Vhi = Vlo
    # widened outer bracket (2.5 -> 4.0) as a continuation-free fallback for
    # the corrected-dielectric variants at high m; flag-OFF never needs it.
    for fac in np.concatenate([np.linspace(1.05, 2.5, 60),
                               np.linspace(2.55, 4.0, 30)]):
        Vhi = Bext * fac
        if (pressure(n, Vhi, T, basis) - Pp) * f_lo < 0:
            break
    return brentq(lambda V: pressure(n, V, T, basis) - Pp,
                  Vlo, Vhi, xtol=1e-16, rtol=8.9e-16)


def ln_f_w(n, T, Pp, basis="total", parts=False):
    """ln f_w [ln Pa] = ln(n_w R T / V) + d(A^r/RT)/dn_w|_{T,V}."""
    V = solve_V(n, T, Pp, basis)
    h = max(n[0] * 1e-7, 1e-9)
    np1 = n.copy(); np1[0] += h
    nm1 = n.copy(); nm1[0] -= h
    if parts:
        Ap = A_r(np1, V, T, basis, parts=True)
        Am = A_r(nm1, V, T, basis, parts=True)
        d = {k: (Ap[k] - Am[k]) / (2 * h) for k in Ap}
        d["ideal"] = np.log(n[0] * R * T / V)
        d["lnf"] = d["ideal"] + d["total"]
        d["V"] = V
        return d
    dA = (A_r(np1, V, T, basis)
          - A_r(nm1, V, T, basis)) / (2 * h)
    return np.log(n[0] * R * T / V) + dA


# --------------------------------------------------------------------------
# osmotic coefficient on the total-mole basis
# --------------------------------------------------------------------------

_PURE_CACHE: dict = {}


def _lnf_pure(T, Pp):
    key = (round(T, 6), round(Pp, 3), _ext_cache_key())
    if key not in _PURE_CACHE:
        _PURE_CACHE[key] = ln_f_w(n_vec("NaCl", 0.0), T, Pp,
                                  basis="total")
    return _PURE_CACHE[key]


def phi_osm_total(salt, m, T=298.15, Pp=1.0e5):
    """Osmotic coefficient, published total-mole ln a_w, at (T, P).

    phi = -55.5084 * [ln f_w(brine) - ln f_w(pure)] / (nu * m).
    Returns nan if the volume root cannot be bracketed (reported honestly
    by the callers rather than silently patched).
    """
    if m <= 0:
        return np.nan
    try:
        lnf_br = ln_f_w(n_vec(salt, m), T, Pp, basis="total")
        lnf_pu = _lnf_pure(T, Pp)
    except ValueError:          # brentq bracket failure
        return np.nan
    return float(-N_W * (lnf_br - lnf_pu) / (NU[salt] * m))


# populate IP at import
reload_parameters()


if __name__ == "__main__":
    # smoke test: DH limit (parameter-stable: Na+/Cl- rows are literature
    # values not touched by the Ca2+ correction) + printed-row rebinding
    # round trip.
    phi_dh = phi_osm_total("NaCl", 1e-3)
    print(f"NaCl  phi(0.001 m) = {phi_dh:.4f}  (DH/Pitzer 0.9884)  "
          f"{'OK' if abs(phi_dh - 0.9884) < 2e-3 else 'FAIL'}")
    phi_now = phi_osm_total("CaCl2", 1.0)
    with printed_ca_row():
        phi_printed = phi_osm_total("CaCl2", 1.0)
    phi_back = phi_osm_total("CaCl2", 1.0)
    print(f"CaCl2 phi(1 m): current row {phi_now:.4f} | printed row "
          f"{phi_printed:.4f} (audit 0.6592) | restored {phi_back:.4f}  "
          f"{'OK' if abs(phi_back - phi_now) < 1e-12 else 'FAIL'}")
