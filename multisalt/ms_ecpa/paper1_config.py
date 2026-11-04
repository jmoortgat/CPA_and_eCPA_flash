"""Paper-I (I&ECR 2026) CO2-H2O-NaCl parameter configuration.

Binds the ms_ecpa library to the PUBLISHED ternary parameterization of

    Moortgat (2026), "..." I&ECR, DOI 10.1021/acs.iecr.6c02626
    (referred to throughout the project as "Paper I"),

whose eCPA parameters are those of Coelho (2025) for the ternary
CO2-H2O-NaCl system.  The ms_ecpa library was built as a generalization
of that earlier ternary code; its DEFAULT parameters are the Papers
II/III multi-salt values, which for Na+/Cl- deliberately differ from the
Paper-I ternary fit.  This module makes Paper I a loadable parameter
configuration of the same kernel, applied and restored in place.

Parameter provenance (authority order, values cross-checked 2026-09-12)
-----------------------------------------------------------------------
(a) Original Paper-I package on disk (the maintained copy):
      code/ecpa/ (Paper I package, this repository)
        constants.py  -- every numeric bound below
        elv.py        -- functional forms (shared water-salt and CO2-salt
                         Huron-Vidal energies; quadratic-in-T DU_cs)
(b) The Paper I article (matches (a) at published precision):
        Supporting Information Tables S2 (pure-component), S3 (binary/ion-
        neutral), S4 (Peneloux H2O polynomial); main text, section "Density
        and Peneloux volume correction".

What differs from the library defaults (Papers II/III values in
parentheses):

  Na+ :  sigma 2.356 A (2.63), R_B 1.665 A (1.54), b 16.49 cm3/mol
         (from sigma), a0 = 0 (959.54*R*b)
  Cl- :  sigma 3.187 A (3.571), R_B 1.828 A (1.99), b 40.83 cm3/mol
         (from sigma), a0 = 0 (1878.78*R*b)
  DU_ws (water-NaCl, shared by BOTH ions): Uref = -223.5*R J/mol,
         alfa = 1573 K, Talfa = 340 K  (per-ion values by default)
  DU_cs (CO2-NaCl, shared by BOTH ions):  Uref = 6056.13852 J/mol,
         alfa = 691.85326 K, Talfa = 243.79352 K, in the QUADRATIC form
             DU(T) = Uref + alfa*R*[(1-T/Talfa)^2 - (1-Tref/Talfa)^2]
         (the library default is the linear A*(T/Tc3) + B form; see the
         _QuadraticGasIonDU shim below)
  k_CO2,H2O / S_CO2,H2O coefficient rounding: Paper I carries
         (-0.49206, 2.10136, -1.57135) / (0.19173, -0.17299, -0.00909);
         the library defaults are (-0.4921, 2.1014, -1.5714) /
         (0.191728, -0.172991, -0.009094).
  Peneloux NaCl: constant c_s = -53.5 cm3/mol per mole of NaCl
         (bound as -26.75 cm3/mol on each ion slot, A_Pen = 0), replacing
         the per-ion T-linear multi-salt shifts.
  Peneloux H2O: Paper I's degree-4 polynomial c_H2O(T_R) fitted to
         IAPWS-95 (Table S4).  The library's density routes carry no
         water shift, so this term CANNOT be bound into the parameter
         arrays; it is exposed here as ``peneloux_h2o`` and applied
         by ``rho_water_paper1`` (the exact published pure-water
         density route of code/scripts/optimize_peneloux_h2o.py /
         benchmark_pure_water_density.py, including their M_H2O =
         18.015 g/mol -- note the EoS mole-fraction algebra in BOTH
         projects uses the rounded Mw = 0.018 kg/mol).

Unchanged (already identical between Paper I and the defaults): all
pure-component H2O/CO2 CPA parameters, the 4C association scheme,
dipole/polarizability closure (pol_Na = 2.221e-40, pol_Cl = 3.557e-40,
pol_CO2 = 2.6946e-40 match), Debye-Hueckel structure, Tc/Tref, ion
charges and molar masses (MM_Na + MM_Cl = 0.0585 = Paper I's Ms).

Scope: the configuration defines ONLY the Na+/Cl- slots and the
CO2-H2O pair.  CH4 and the K+/Ca2+/Mg2+/SO4^2- slots retain the
multi-salt defaults and are OUT OF SCOPE under this configuration
(Paper I parameterizes neither); call it with z_CH4 = 0 and
mi = (m, m, 0, 0, 0, 0) only.

Mechanics
---------
``lnphi.py`` and ``eos.py`` hold import-time copies of every parameter
name (``from .parameters import *``).  Six-vector ion parameters are
therefore mutated IN PLACE (the consumer modules alias the same ndarray
objects), while the six scalar k13/S13
coefficients and the A_U3i/B_U3i bindings are set explicitly on each
consumer module and restored afterwards.  ``restore_paper1()`` rebinds
the ORIGINAL array objects and values, so the restored library is
bit-exact (verified by tests/test_paper1_config.py).

None of that reaches the compiled kernel, which holds its own parameter
file, so ``apply_paper1()`` routes dispatched calls back to the Python
kernel for the duration and ``restore_paper1()`` puts the previous
backend mode back.

Usage::

    from ms_ecpa import paper1_config
    paper1_config.apply_paper1()
    try:
        ...  # any ms_ecpa call now runs the Paper-I model
    finally:
        paper1_config.restore_paper1()

or equivalently ``with paper1_config.paper1_parameters(): ...``.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np

from . import parameters as P
from . import lnphi as _lnphi
from . import eos as _eos
from . import flash as _flash
from . import flash3 as _flash3
from . import stability as _stability

# ---------------------------------------------------------------------------
# Paper-I published values
# (code/ecpa/constants.py; cross-checked against supplemental Tables S2-S4)
# ---------------------------------------------------------------------------
SG_NA = 2.356e-10        # m    (Debye-Hueckel distance sigma, Na+)
SG_CL = 3.187e-10        # m    (sigma, Cl-)
B_NA = 16.49e-6          # m3/mol  (co-volume, Na+; = 2/3*pi*sigma^3*Na)
B_CL = 40.83e-6          # m3/mol  (co-volume, Cl-)
RB_NA = 1.665e-10        # m    (Born radius, Na+)
RB_CL = 1.828e-10        # m    (Born radius, Cl-)
A0_ION = 0.0             # J.m3/mol2 (ion attractive a0; Paper I Table S2)

UREF_WS = -223.5 * P.R   # J/mol  (water-NaCl HV reference energy, both ions)
TALFA_WS = 340.0         # K
ALFA_WS = 1573.0         # K

UREF_CS = 6056.13852     # J/mol  (CO2-NaCl HV reference energy, both ions)
TALFA_CS = 243.79352     # K
ALFA_CS = 691.85326      # K

AK13 = -0.49206          # k_CO2,H2O = A*(T/Tc3)^2 + B*(T/Tc3) + C
BK13 = 2.10136
CK13 = -1.57135
AS13 = 0.19173           # S_CO2,H2O (cross-association strength), same form
BS13 = -0.17299
CS13 = -0.00909

PEN_NACL = -53.5e-6      # m3 per mole of NaCl (Coelho 2025); -26.75 per ion

# Degree-4 Peneloux H2O polynomial c(T_R) [cm3/mol], T_R = T/Tc1
# (fitted to 455 IAPWS-95 liquid-water conditions; Paper I Table S4).
PENELOUX_H2O_COEFFS = np.array([32.86215137869321, -96.0523604195117,
                                110.95367889169857, -58.11052913632152,
                                11.233443826522741])   # highest power first

#: published pure-water density route molar mass [kg/mol] -- the Paper-I
#: density scripts used 18.015 g/mol, NOT the rounded 0.018 kg/mol that
#: both projects use in the EoS mole-fraction algebra.
M_H2O_DENSITY = 18.015e-3


def peneloux_h2o(T_K: float) -> float:
    """Paper-I temperature-dependent Peneloux volume shift for H2O [m3/mol]."""
    c_cm3_mol = float(np.polyval(PENELOUX_H2O_COEFFS, float(T_K) / P.Tc1))
    return c_cm3_mol * 1e-6


# ---------------------------------------------------------------------------
# Quadratic gas-ion DU shim
# ---------------------------------------------------------------------------
class _QuadraticGasIonDU:
    """Stand-in for the ``A_U3i`` six-vector so that the kernels'
    unmodified expression ``U3i = A_U3i * (T / Tc3) + B_U3i`` evaluates to
    Paper I's quadratic DU_cs(T) in the Na+/Cl- slots.

    The kernels multiply ``A_U3i`` by the scalar ``s = T / Tc3`` exactly
    once per call site (lnphi.py and eos.py); ``__mul__`` receives that
    scalar, recovers T, and returns the full ndarray

        out[i] = lin_A[i] * s                          (non-NaCl slots)
        out[i] = Uref + alfa*R*[(1 - T/Talfa)^2
                                - (1 - Tref/Talfa)^2]  (Na+/Cl- slots)

    The paired ``B_U3i`` binding carries the pristine linear B in the
    non-NaCl slots and ZERO in the Na+/Cl- slots, so the sum is exact in
    every slot.  Arithmetic only (works for real or complex ``s``); any
    other ndarray-style use is unsupported by design and will raise.
    """

    def __init__(self, lin_A, mask, uref, talfa, alfa, Tc3, Tref, R):
        self._lin_A = np.asarray(lin_A, dtype=float)  # NaCl slots zeroed
        self._mask = np.asarray(mask, dtype=float)    # 1.0 in NaCl slots
        self._uref = float(uref)
        self._talfa = float(talfa)
        self._alfa = float(alfa)
        self._Tc3 = float(Tc3)
        self._Tref = float(Tref)
        self._R = float(R)

    def __mul__(self, s):
        T = s * self._Tc3
        quad = self._uref + self._alfa * self._R * (
            (1.0 - T / self._talfa) ** 2
            - (1.0 - self._Tref / self._talfa) ** 2)
        return self._lin_A * s + self._mask * quad

    __rmul__ = __mul__


# ---------------------------------------------------------------------------
# apply / restore
# ---------------------------------------------------------------------------
#: consumer modules holding import-time copies of the parameter names
_CONSUMERS = (P, _lnphi, _eos)

#: Modules that star-import or name-import the scalars in _SCALARS.
#: flash.py is here but NOT in _CONSUMERS: it imports Mw by name and
#: uses it in the aqueous closures, but carries none of the Ak13/AS13
#: block or the A_U3i/B_U3i shim, so it must not be snapshotted for
#: those. Missing names are skipped per module.
_SCALAR_CONSUMERS = (P, _lnphi, _eos, _flash, _flash3, _stability)

#: six-vector ion parameters: {name: (Na+ value, Cl- value)} -- mutated
#: in place on parameters.<name> (aliased by every star-importer).
_ARRAY_SLOTS = {
    "Sgi": (SG_NA, SG_CL),
    "bi": (B_NA, B_CL),
    "Rbi": (RB_NA, RB_CL),
    "a0i": (A0_ION, A0_ION),
    "Uref1i": (UREF_WS, UREF_WS),
    "Talfa1i": (TALFA_WS, TALFA_WS),
    "alfa1i": (ALFA_WS, ALFA_WS),
    "A_Peni": (0.0, 0.0),
    "B_Peni": (PEN_NACL / 2 * 1e6, PEN_NACL / 2 * 1e6),  # cm3/mol per ion
}

#: scalar coefficients rebound on every consumer module.
_SCALARS = {
    "Ak13": AK13, "Bk13": BK13, "Ck13": CK13,
    "AS13": AS13, "BS13": BS13, "CS13": CS13,
    # Paper I's EoS mole-fraction algebra uses the rounded water molar
    # mass, as the prior project's code/ecpa/constants.py does. The
    # library default moved to 0.01801528 on 2026-09-18 to agree with the
    # molality conversions applied to the data; pin the published value
    # here so this configuration keeps reproducing Paper I exactly.
    "Mw": 0.018,
}

_SAVED = None   # pristine snapshot while the configuration is applied


def _route_to_python_kernel() -> str:
    """Send dispatched calls to the Python kernel; return the prior mode.

    Everything this module binds lives on the PYTHON side: the six-vector
    slots are mutated in place on ``parameters`` and the scalars/shim are
    rebound on the consumer modules.  The compiled kernel keeps its own
    copy, loaded once from ``fortran/params/paperII_III.par`` (backend.py
    ``_engine``), and no rebinding here reaches it.  So under
    ``MS_ECPA_BACKEND=fortran`` -- the mode README.md and the slurm
    scripts recommend -- the dispatched ``stability_flash`` would go on
    returning Papers II/III numbers while ``is_applied`` reads True,
    converged and plausible (0.79 % on x_CO2 at 323.15 K / 200 bar) and
    with nothing on the result naming the parameter set that produced it.

    Falling back to the reference kernel is what ``backend.dispatch``
    already does for a request the compiled path cannot honour, on the
    stated grounds that a slow answer beats a silently different one; a
    parameter set the compiled path does not hold is the same case.  The
    cost is the 190x speedup, for the duration of the configuration only.

    A ``backend.use("fortran")`` issued INSIDE the bracket is not covered
    -- the guard would have to live in ``dispatch`` for that.
    """
    from . import backend
    prev = backend.mode()
    if prev != "python":
        backend.use("python")
    return prev


def apply_paper1() -> None:
    """Bind the library parameters to the Paper-I ternary values.

    Routes dispatched calls to the Python kernel for the duration (see
    ``_route_to_python_kernel``) -- the compiled kernel carries its
    own parameter file and these bindings do not reach it.

    Scope (repeated from the module docstring because it is not
    enforced): Na+/Cl- and the CO2-H2O pair only.  Call the library with
    ``z_CH4 = 0`` and ``mi = (m, m, 0, 0, 0, 0)`` while this is applied;
    CH4 and the K+/Ca2+/Mg2+/SO4^2- slots keep their multi-salt values
    and a result mixing them is neither paper's model.

    Raises ``RuntimeError`` if already applied (no nesting -- mirror the
    benchmark's try/finally bracketing).
    """
    global _SAVED
    if _SAVED is not None:
        raise RuntimeError("Paper-I configuration is already applied")

    saved = {
        # pristine copies of the mutated array slots
        "arrays": {k: np.array(getattr(P, k)[:2]) for k in _ARRAY_SLOTS},
        # pristine scalar values per consumer module
        "scalars": {m.__name__: {k: getattr(m, k)
                                 for k in _SCALARS if hasattr(m, k)}
                    for m in _SCALAR_CONSUMERS},
        # ORIGINAL A_U3i / B_U3i objects per consumer module
        "gas_du": {m.__name__: (getattr(m, "A_U3i"), getattr(m, "B_U3i"))
                   for m in _CONSUMERS},
    }
    # done before any mutation, so a failure here leaves the library clean
    saved["backend_mode"] = _route_to_python_kernel()

    # 1. six-vector slots: in-place (propagates through the star imports)
    for name, (v_na, v_cl) in _ARRAY_SLOTS.items():
        arr = getattr(P, name)
        arr[0] = v_na
        arr[1] = v_cl

    # 2. scalars: explicit rebinding on each consumer module
    for mod in _SCALAR_CONSUMERS:
        for name, val in _SCALARS.items():
            if hasattr(mod, name):
                setattr(mod, name, val)

    # 3. CO2-ion DU: quadratic Paper-I form via the A_U3i/B_U3i shim
    lin_A = np.array(saved["gas_du"][P.__name__][0], dtype=float)
    lin_B = np.array(saved["gas_du"][P.__name__][1], dtype=float)
    lin_A[:2] = 0.0
    lin_B[:2] = 0.0
    shim = _QuadraticGasIonDU(lin_A, np.array([1.0, 1.0, 0, 0, 0, 0]),
                              UREF_CS, TALFA_CS, ALFA_CS,
                              P.Tc3, P.Tref, P.R)
    for mod in _CONSUMERS:
        setattr(mod, "A_U3i", shim)
        setattr(mod, "B_U3i", lin_B)

    _SAVED = saved


def restore_paper1() -> None:
    """Restore the pristine Papers II/III library state (bit-exact).

    Also puts back the backend mode ``apply_paper1`` stepped down.
    """
    global _SAVED
    if _SAVED is None:
        raise RuntimeError("Paper-I configuration is not applied")

    for name in _ARRAY_SLOTS:
        getattr(P, name)[:2] = _SAVED["arrays"][name]
    for mod in _SCALAR_CONSUMERS:
        for name in _SCALARS:
            if name in _SAVED["scalars"][mod.__name__]:
                setattr(mod, name, _SAVED["scalars"][mod.__name__][name])
    for mod in _CONSUMERS:
        a_orig, b_orig = _SAVED["gas_du"][mod.__name__]
        setattr(mod, "A_U3i", a_orig)
        setattr(mod, "B_U3i", b_orig)
    # the compiled kernel is safe to use again once the defaults are back
    from . import backend
    backend.use(_SAVED["backend_mode"])

    _SAVED = None


def is_applied() -> bool:
    """True while the Paper-I configuration is bound."""
    return _SAVED is not None


@contextmanager
def paper1_parameters():
    """Context manager: apply on entry, restore (always) on exit."""
    apply_paper1()
    try:
        yield
    finally:
        restore_paper1()


# ---------------------------------------------------------------------------
# Paper-I density route (pure water)
# ---------------------------------------------------------------------------
def rho_water_paper1(T_K: float, P_bar: float) -> float:
    """Published Paper-I pure-water density [kg/m3].

    Exactly the route of the Paper-I density validation scripts
    (code/scripts/optimize_peneloux_h2o.py, benchmark_pure_water_density
    .py, paper Sec. "Density validation", AARE 0.30 % vs IAPWS-95):

        rho = M_H2O / (Zw*R*T/P + c_H2O(T)),   M_H2O = 18.015 g/mol,

    with Zw from the aqueous inner Newton at pure-water composition.
    Works with or without the configuration applied (the pure-water CPA
    parameters are identical in Paper I and the multi-salt defaults).
    """
    from .lnphi import _aq_residual_3var
    from .newton import newton_solve

    T = float(T_K)
    P_Pa = float(P_bar) * 1e5
    x_aq = np.array([1.0, 0.0, 0.0])
    mi = np.zeros(6)
    # Seed ladder mirroring lnphi_aq's pure-liquid-water cold start.
    Zw_liq = P_Pa * 18.015e-6 / (8.314 * T)
    ok = False
    v = None
    for v0 in (np.array([Zw_liq, 50.0, 0.10]),
               np.array([Zw_liq, 70.0, 0.06]),
               np.array([0.05, 50.0, 0.10])):
        v, ok, _ = newton_solve(_aq_residual_3var, v0,
                                args=(x_aq, mi, T, P_Pa),
                                tol=1e-10, maxiter=80)
        # Demote the converged flag instead of only skipping the `break`,
        # so that the post-loop check below governs what is ACCEPTED and
        # not merely whether to keep trying.  At pure water the
        # association closure reduces to 2*x1w*DELTA*chi1w^2 + Zw*chi1w
        # - Zw = 0, whose second root is negative at every Zw > 0 and
        # satisfies the 3x3 residual to machine precision; a relative
        # permittivity <= 0 is likewise the Onsager-Kirkwood closure
        # solved outside its domain.  Neither is a state of water, and
        # neither shows up as a non-finite residual, so a LAST rung
        # landing on one would otherwise be turned into a density.
        # Same predicate and same placement as lnphi.py's aqueous ladder.
        if ok and (np.real(v[2]) <= 0 or np.real(v[1]) <= 0):
            ok = False
        if ok:
            break
    if not ok:
        raise RuntimeError(
            f"rho_water_paper1: inner Newton failed at T={T} K, "
            f"P={P_bar} bar")
    Zw = float(np.real(v[0]))
    v_m = Zw * P.R * T / P_Pa + peneloux_h2o(T)
    return M_H2O_DENSITY / v_m
