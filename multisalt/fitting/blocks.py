"""The ms-eCPA free parameters, their brackets, and how to bind them.

Scope
-----
This is the **Papers II/III** parameter set -- the three blocks F. M. Coelho
Mourao Coelho fitted independently, recorded in ``ms_ecpa/parameters.py``:

    A_U3i, B_U3i   CO2-ion interaction energies      (his MIXED_SALTS/OPT/OPT7)
    A_U2i, B_U2i   CH4-ion interaction energies      (his CH4-Brine/OPT/OPT6)
    A_Peni, B_Peni Peneloux ion volume shifts        (his DENSITY/OPT/OPT_PEN6)

plus, optionally, the binary coefficients that pin each gas to water:

    Ak13, Bk13, Ck13   CO2-H2O kij(T)
    AS13, BS13, CS13   CO2-H2O solvation S13(T)
    Ak12, Bk12, Ck12   CH4-H2O kij(T)

It does NOT include the water-ion structural parameters (sigma, R_Born,
alpha0); those are fixed at their published values in this refit.

Why these names
---------------
The Python library and the Fortran port use the SAME six-vector names, so
one driver binds either backend:

    ms_ecpa.parameters.A_U3i   <->   .par file "A_U3i"   <->  Engine.set_param

Brackets
--------
Interaction energies are bracketed at the published value +/- 80 kJ/mol.
Peneloux shifts use [-150, 100] cm3/mol. Binary coefficients get a relative bracket, because their
magnitudes differ by orders of magnitude.

The brackets exist to keep the optimiser in a physical region and to give
``to_z`` a uniform step; they are deliberately generous. A parameter that
converges ONTO a bracket is a finding, not a success -- check it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Ion order, frozen across both projects.
IONS = ("Na", "Cl", "K", "Ca", "Mg", "SO4")

#: +/- bracket half-width for the gas-ion interaction energies [J/mol].
ENERGY_HALFWIDTH = 80_000.0

#: Absolute bracket for the Peneloux ion volume shifts [cm3/mol].
PENELOUX_LO, PENELOUX_HI = -150.0, 100.0


@dataclass(frozen=True)
class Block:
    """One named group of free parameters that can be fitted or held."""
    name: str
    vectors: tuple[str, ...]      # parameter-vector names, 6 entries each
    kind: str                     # "energy" | "peneloux" | "binary"
    constrains: tuple[str, ...]   # which data families determine it
    note: str = ""


#: The three blocks F. M. Coelho fitted, in the order the refit stages them.
#:
#: Ordering matters and is not arbitrary. Brine-density data is GAS-FREE, so
#: the Peneloux ion shifts are determined without any gas-ion parameter
#: entering -- but they DO affect solubility through the aqueous volume. The
#: dependency is one-way, so fitting density first and holding it fixed is
#: correct rather than a compromise, and it insulates the brine-density
#: result from the solubility/y_H2O trade-off.
BLOCKS = (
    Block("peneloux", ("A_Peni", "B_Peni"), "peneloux", ("rho",),
          "gas-free data; one-way coupling, so fit first and hold"),
    Block("co2_ion", ("A_U3i", "B_U3i"), "energy",
          ("co2_solubility", "co2_yh2o"),
          "CO2-ion energies; competes between solubility and y_H2O"),
    Block("ch4_ion", ("A_U2i", "B_U2i"), "energy",
          ("ch4_solubility", "ch4_yh2o"),
          "CH4-ion energies; separate block from CO2, they do not compete"),
)

#: Binary coefficients, per gas. Fitted from the SALT-FREE water binaries
#: before the ion energies are asked to explain salting-out.
BINARY_BLOCKS = (
    Block("co2_binary", ("Ak13", "Bk13", "Ck13", "AS13", "BS13", "CS13"),
          "binary", ("co2_binary",), "CO2-H2O kij(T) and solvation S13(T)"),
    Block("ch4_binary", ("Ak12", "Bk12", "Ck12"), "binary",
          ("ch4_binary",), "CH4-H2O kij(T)"),
)


#: The CO2-H2O solvation strength, fitted in a space where non-negativity is
#: STRUCTURAL rather than penalised.
#:
#:     S13(tau) = (SA13*tau + SB13)^2 + SC13,   SC13 >= 0,   tau = T/Tc3
#:
#: which is exactly the set of quadratics that are non-negative for all tau.
#: The model form is unchanged -- it is still the same quadratic in reduced
#: temperature -- only the coordinates the optimiser works in change, and in
#: these it cannot leave the physical domain.
#:
#: Why not a penalty: a penalty has to start from the published values, which
#: are INFEASIBLE (S13 < 0 below 289.8 K). With a weight small enough to be
#: smooth the optimiser trades the violation against accuracy and never quite
#: satisfies it; with a weight large enough to bind, the early steps are
#: dominated by escaping a steep barrier, which throws the parameters far from
#: any good fit -- measured, it halted after 17 evaluations having moved
#: y_H2O_sf by 0.039. Here there is no barrier and no infeasible region.
SOLVATION_VIRTUAL = ("SA13", "SB13", "SC13")

#: SC13 is a squared-distance offset and may not go negative; the others are
#: unconstrained. Bracket widths span the observed range of S13 (0 to ~0.5).
_VIRTUAL_BRACKETS = {"SC13": (0.0, 0.5)}

_SOLVATION_START = None


def expand(values: dict) -> dict:
    """Map virtual fitting parameters onto the library's own parameters."""
    if not all(k in values for k in SOLVATION_VIRTUAL):
        return values
    out = {k: v for k, v in values.items() if k not in SOLVATION_VIRTUAL}
    a = float(np.ravel(values["SA13"])[0])
    b = float(np.ravel(values["SB13"])[0])
    c = float(np.ravel(values["SC13"])[0])
    out["AS13"] = np.array([a * a])
    out["BS13"] = np.array([2.0 * a * b])
    out["CS13"] = np.array([b * b + c])
    return out


def _solvation_start():
    """(SA13, SB13, SC13) closest to the published S13(T) with SC13 >= 0.

    The published coefficients are not representable here -- they would need
    SC13 = -0.048 -- so the start is the nearest feasible curve rather than an
    exact re-expression. It tracks the published S13 to within 0.005 over most
    of the range and lifts the sub-290 K region from -0.010 to +0.009.
    """
    global _SOLVATION_START
    if _SOLVATION_START is not None:
        return _SOLVATION_START
    from scipy.optimize import least_squares
    import ms_ecpa.parameters as P
    T = np.linspace(273.0, 633.0, 73)
    tau = T / float(P.Tc3)
    pub = (float(P.AS13) * tau ** 2 + float(P.BS13) * tau + float(P.CS13))
    if getattr(P, "MCR1_CROSS_ENERGY", False):
        # Under m-CR1 the fitted quadratic multiplies exp(epsW/2T)-1 rather
        # than exp(epsW/T)-1, so the curve equivalent to the published cross
        # strength is rescaled by their ratio. It is still negative at low T
        # (it is the same physical cross strength), so the projection below
        # lifts it, exactly as in the unflagged case.
        eps = float(P.epsW)
        pub = pub * ((np.exp(eps / T) - 1.0) / (np.exp(0.5 * eps / T) - 1.0))
    a0 = np.sqrt(max(float(P.AS13), 1e-12))
    b0 = float(P.BS13) / (2.0 * a0)
    r = least_squares(lambda q: (q[0] * tau + q[1]) ** 2 + q[2] - pub,
                      [a0, b0, 0.0],
                      bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, np.inf]))
    # Nudge SC13 strictly inside its bracket. The projection lands it exactly
    # on the lower bound, and a start ON a bound is an awkward place to begin
    # -- it also trips the bracket-edge warning, which should mean "the
    # optimiser wanted to go further", not "this is where it started". 1e-4 is
    # four orders below the S13 values it offsets.
    _SOLVATION_START = {"SA13": np.array([r.x[0]]),
                        "SB13": np.array([r.x[1]]),
                        "SC13": np.array([max(float(r.x[2]), 1.0e-4)])}
    return _SOLVATION_START


def published_values(vectors):
    """Current library values for the named parameter vectors."""
    import ms_ecpa.parameters as P
    out = {}
    start = None
    for v in vectors:
        if v in SOLVATION_VIRTUAL:
            start = start or _solvation_start()
            out[v] = start[v].copy()
            continue
        val = getattr(P, v)
        out[v] = np.atleast_1d(np.asarray(val, dtype=float)).copy()
    return out


def brackets(vectors):
    """(lo, hi) arrays for the named vectors, aligned with published_values."""
    pub = published_values(vectors)
    lo, hi = {}, {}
    for v, x in pub.items():
        if v in _VIRTUAL_BRACKETS:
            a, b = _VIRTUAL_BRACKETS[v]
            lo[v] = np.full_like(x, a)
            hi[v] = np.full_like(x, b)
        elif v.startswith(("A_U", "B_U")):
            lo[v] = x - ENERGY_HALFWIDTH
            hi[v] = x + ENERGY_HALFWIDTH
        elif v.startswith(("A_Pen", "B_Pen")):
            lo[v] = np.full_like(x, PENELOUX_LO)
            hi[v] = np.full_like(x, PENELOUX_HI)
        else:
            # Binary coefficients span orders of magnitude, so bracket them
            # relatively. The floor exists so a published ZERO is not frozen
            # at zero -- and it must apply only in that case. Applied
            # unconditionally (the old `max(|x|*2, 1.0)`) it gave CS13, whose
            # value is -0.0091, a bracket 110x wider than itself, and since
            # `to_z` makes diff_step a fraction of the BRACKET, the optimiser
            # would have perturbed it by 22 % of its own value to build a
            # finite difference. That column of the Jacobian would have been
            # nonlinearity, not derivative.
            span = np.abs(x) * 2.0
            span = np.where(span > 0, span, 1.0)
            lo[v] = x - span
            hi[v] = x + span
    return lo, hi


def flatten(values: dict) -> np.ndarray:
    """Dict of vectors -> one flat vector, in the dict's insertion order."""
    return np.concatenate([np.atleast_1d(values[k]) for k in values])


def unflatten(flat: np.ndarray, template: dict) -> dict:
    """Inverse of :func:`flatten`, using ``template`` for the shapes."""
    out, i = {}, 0
    for k, v in template.items():
        n = np.atleast_1d(v).size
        out[k] = np.asarray(flat[i:i + n], dtype=float).reshape(np.shape(v))
        i += n
    if i != len(flat):
        raise ValueError(f"length mismatch: consumed {i} of {len(flat)}")
    return out


def to_z(x, lo, hi):
    """Physical -> z in [1, 2].

    scipy's ``diff_step`` is RELATIVE to the parameter value, so a parameter
    whose published value sits near zero gets an absurdly small finite
    difference and a parameter in the tens of thousands gets an absurdly
    large one. Mapping every parameter onto the same interval makes the step
    a uniform fraction of its bracket instead.
    """
    return 1.0 + (np.asarray(x, float) - lo) / (hi - lo)


def from_z(z, lo, hi):
    return lo + (np.asarray(z, float) - 1.0) * (hi - lo)


def bind_python(values: dict) -> None:
    """Bind parameter vectors into the live Python library.

    Writes to ``ms_ecpa.parameters`` AND to every module that star-imported
    the vectors, because ``from .parameters import *`` binds a module-level
    name that a later write to parameters does not reach. This is the same
    trap that made paper1_config leave flash.py half-configured.
    """
    values = expand(values)
    import ms_ecpa.parameters as P
    import ms_ecpa.eos as _eos
    import ms_ecpa.lnphi as _lnphi
    import ms_ecpa.flash as _flash
    import ms_ecpa.flash3 as _flash3
    consumers = (P, _eos, _lnphi, _flash, _flash3)
    for name, val in values.items():
        arr = np.asarray(val, dtype=float)
        for mod in consumers:
            if hasattr(mod, name):
                cur = getattr(mod, name)
                if isinstance(cur, np.ndarray) and cur.shape == arr.shape:
                    cur[...] = arr          # in place: star-imports alias it
                elif not isinstance(cur, np.ndarray) and arr.size == 1:
                    # The library holds this one as a SCALAR. published_values
                    # wraps everything in atleast_1d, so writing arr back
                    # would silently replace the float with a shape-(1,)
                    # array -- which does not raise here, it raises three
                    # layers down as "could not broadcast (2,1) into (2,)"
                    # inside the Newton residual. The six binary coefficients
                    # are all scalars, so this is the path they take.
                    setattr(mod, name, float(np.ravel(arr)[0]))
                else:
                    setattr(mod, name, arr.copy() if arr.ndim else float(arr))


def bind_fortran(engine, values: dict) -> None:
    """Bind the same vectors into an open Fortran engine."""
    for name, val in expand(values).items():
        engine.set_param(name, np.atleast_1d(np.asarray(val, dtype=float)))
