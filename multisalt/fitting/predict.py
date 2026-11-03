"""Model predictions for each fitted family, on a chosen backend.

These MUST agree with what the figure scripts compute, or the fit and the
figures describe different models. The retry ladders and the
degenerate-two-phase guard below are transcribed from
``scripts/fig_co2_singlesalt.py`` and its siblings for exactly that reason.

    The figure scripts keep their own copies of these ladders, so a change
    here must be made there too.

Backends
--------
``backend="fortran"`` routes through the compiled port, which is ~190x
faster and agrees with Python to 1e-13. It is valid ONLY under the
as-aqueous convention (see ecpa-fortran-backend). Everything fitted here is
as-aqueous, so the constraint is satisfied, but it is asserted rather than
assumed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_CODE = Path(__file__).resolve().parents[1]

#: Feed CO2/CH4 mole fractions to try. A flash needs a feed, but a
#: solubility measurement only fixes (T, P, brine): the gas-to-brine ratio
#: is the experimenter's, not the datum's. So we try a ladder and take the
#: first feed that yields a genuine two-phase split. Transcribed from
#: scripts/fig_co2_singlesalt.py:50.
Z_GAS_RETRIES = (0.05, 0.10, 0.02, 0.20, 0.30)

_ENGINE = None

#: Pressure grid the density warm start is walked along, matching
#: scripts/fig_brine_density.py::P_GRID_BAR.
_RHO_WARM_GRID = np.linspace(5.0, 700.0, 200)


def _engine():
    """Per-process Fortran engine; ctypes handles are not shared."""
    global _ENGINE
    if _ENGINE is None:
        tools = _CODE / "fortran" / "tools"
        if str(tools) not in sys.path:
            sys.path.insert(0, str(tools))
        from ecpa_ctypes import Engine
        par = os.environ.get(
            "ECPA_FIT_PARFILE",
            str(_CODE / "fortran" / "params" / "paperII_III.par"))
        _ENGINE = Engine(parfile=par)
    return _ENGINE


def _flash(T, P, z, mi, backend):
    """One stability+flash call, returning a uniform little dict or None."""
    if backend == "fortran":
        r = _engine().stability_flash(float(T), float(P),
                                      np.asarray(z, float),
                                      np.asarray(mi, float),
                                      salinity_convention="as-aqueous")
        if int(r.get("phases", 0)) != 2 or not r.get("has_flash"):
            return None
        f = r["flash"]
        if not f.get("converged"):
            return None
        return {"x_aq": f["x_aq"], "y_gas": f["y_gas"], "mi_aq": f["mi_aq"]}
    from ms_ecpa import stability_flash
    r = stability_flash(float(T), float(P), np.asarray(z, float),
                        np.asarray(mi, float),
                        salinity_convention="as-aqueous")
    if r.phases != 2 or r.flash_result is None or not r.flash_result.converged:
        return None
    fr = r.flash_result
    return {"x_aq": fr.x_aq, "y_gas": fr.y_gas, "mi_aq": fr.mi_aq}


def _slot(gas):
    """Which entry of the 3-vector feed the gas occupies."""
    return 2 if gas == "co2" else 1


def solubility(T, P, mi, gas, backend="python", as_molality=False):
    """Salt-free aqueous mole fraction, or molality [mol/kg water].

    Returns NaN when no feed on the ladder gives a genuine two-phase split;
    the caller turns that into a penalty residual rather than dropping the
    row, so a parameter set that stops converging is punished rather than
    silently scored on fewer points.
    """
    k = _slot(gas)
    for zg in Z_GAS_RETRIES:
        z = np.zeros(3)
        z[0] = 1.0 - zg
        z[k] = zg
        out = _flash(T, P, z, mi, backend)
        if out is None:
            continue
        xH = float(out["x_aq"][0])
        xg = float(out["x_aq"][k])
        if xH + xg <= 0:
            continue
        xsf = xg / (xH + xg)
        # Degenerate-two-phase guard (fig_co2_singlesalt.py:105): a salt-free
        # gas fraction numerically equal to the FEED fraction means the flash
        # preserved the feed ratio in the aqueous phase, i.e. no selective
        # partitioning happened. That is not a solubility.
        if abs(xsf - zg) < 1e-3 * max(zg, 1e-6):
            continue
        if as_molality:
            from ms_ecpa.parameters import Mw
            return xsf / ((1.0 - xsf) * Mw)
        return xsf
    return float("nan")


def y_h2o(T, P, mi, gas, backend="python"):
    """Water mole fraction in the gas phase."""
    k = _slot(gas)
    for zg in Z_GAS_RETRIES:
        z = np.zeros(3)
        z[0] = 1.0 - zg
        z[k] = zg
        out = _flash(T, P, z, mi, backend)
        if out is None:
            continue
        # Same degenerate-two-phase guard solubility() applies. A salt-free
        # aqueous gas fraction numerically equal to the FEED fraction means no
        # selective partitioning happened, so the "gas phase" is a copy of the
        # feed and its water content is an artifact of the assumed feed, not a
        # prediction. Measured on the current data: 0 of 593 rows are affected,
        # so this changes nothing today -- it stops the two functions drifting.
        xH, xg = float(out["x_aq"][0]), float(out["x_aq"][k])
        if xH + xg <= 0:
            continue
        if abs(xg / (xH + xg) - zg) < 1e-3 * max(zg, 1e-6):
            continue
        yw = float(out["y_gas"][0])
        if np.isfinite(yw) and 0.0 < yw < 1.0:
            return yw
    return float("nan")


def rho_brine(T, P, mi, backend="python", v_init=None):
    """Gas-free brine mass density [kg/m3].

    Delegates to the figure script's own routine so the fitted quantity and
    the plotted quantity cannot diverge. Gas-free, which is why the Peneloux
    ion shifts are determined here without any gas-ion parameter entering --
    the one-way coupling that lets this block be fitted first and held.
    """
    from_fortran = False
    if backend == "fortran":
        # The port has its own colligative route. Without it, rho was the one
        # family the Fortran backend did NOT accelerate -- it delegated to a
        # pure-Python routine and ran at the same speed, which would have made
        # brine density the bottleneck of every refit once everything else
        # went 190x faster.
        try:
            # Returns (rho, v_state), mirroring predict_rho_brine.
            rho, st = _engine().rho_brine(float(T), float(P),
                                          np.asarray(mi, float),
                                          v_init=v_init)
            rho = float(rho)
            engine_ran = True
        except Exception:
            rho, st = float("nan"), None
            engine_ran = False
        from ms_ecpa.roots import is_liquid_root
        if is_liquid_root(rho, float(T)):
            return rho, st
        if engine_ran and not np.isfinite(rho):
            # The compiled routine now carries the SAME seed ladder and the
            # same liquid-branch test as the Python one, so a NaN from it
            # means all seven seeds were tried and none landed on the liquid
            # branch. Re-running the identical ladder in Python reaches the
            # identical answer ~190x slower: it cost 252 s of the 286 s
            # regression pass. Fall through only when the compiled path
            # produced a root that is off-branch, or errored.
            return float("nan"), None
        # The compiled route landed off the liquid branch (or failed). Fall
        # through to the Python routine, which carries the seed ladder that
        # can walk back onto it -- the same "defer to the reference rather
        # than return a different ANSWER" rule the backend switch already
        # follows. It costs Python speed on the handful of states that need
        # it, and nothing on the rest.
        from_fortran = True
    scripts = _CODE / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from fig_brine_density import predict_rho_brine
    # A warm start that led the compiled route off the liquid branch is not a
    # good seed for the retry either, and the two routes' state triples are
    # not guaranteed to mean the same thing. Start the fallback cold and let
    # its own ladder find the branch.
    seed = None if from_fortran else v_init
    # Non-convergence does NOT arrive here as an exception: predict_rho_brine
    # returns (nan, None) by ordinary return once its seed ladder is exhausted
    # (fig_brine_density.py:_solve_Zw), and ms_ecpa.newton already turns a
    # singular Jacobian and a non-finite residual into converged=False rather
    # than letting either propagate. So a catch-all here would only ever catch
    # a code-level fault -- a bad bind, a shape mismatch, a ctypes failure --
    # and file it under the SAME NaN whose documented meaning is "the model
    # does not converge at this state", where it would be charged ~2200 sigma
    # per row (refit.py), steer the Peneloux stage, and be counted into the
    # reported non-convergence fraction. Catch only what a linear solve can
    # still raise; anything else stops the run at the line that caused it.
    try:
        rho, st = predict_rho_brine(float(T), float(P),
                                    np.asarray(mi, float), v_init=seed)
        return float(rho), st
    except (np.linalg.LinAlgError, FloatingPointError):
        return float("nan"), None


#: family name -> (what to predict, which gas). EXPLICIT, and consulted with
#: a plain lookup, so an unrecognised family raises instead of being predicted
#: as something else.
#:
#: It used to dispatch on `name.endswith("yh2o")` with solubility as the
#: fall-through. That silently mispredicted the moment the y_H2O families were
#: split: "co2_yh2o_sf" does not end in "yh2o", so 537 water-content rows were
#: predicted as SOLUBILITY and scored against water-content measurements --
#: 228 % AAD where the same rows score 24.7 % when named "co2_yh2o". A
#: fall-through branch that accepts any name cannot report that it guessed.
FAMILY_KIND = {
    "rho":              ("rho", None),
    "co2_solubility":   ("solubility", "co2"),
    "co2_binary":       ("solubility", "co2"),
    "ch4_solubility":   ("solubility", "ch4"),
    "ch4_binary":       ("solubility", "ch4"),
    "co2_yh2o":         ("yh2o", "co2"),
    "co2_yh2o_sf":      ("yh2o", "co2"),
    "co2_yh2o_brine":   ("yh2o", "co2"),
    "ch4_yh2o":         ("yh2o", "ch4"),
    "ch4_yh2o_sf":      ("yh2o", "ch4"),
    "ch4_yh2o_brine":   ("yh2o", "ch4"),
}


def predict_family(name: str, row, backend="python") -> float:
    try:
        kind, gas = FAMILY_KIND[name]
    except KeyError:
        raise KeyError(
            f"no prediction registered for family {name!r}; add it to "
            f"fitting.predict.FAMILY_KIND rather than letting it be "
            f"predicted as something else") from None
    ions = np.array([row["m_Na"], row["m_Cl"], row["m_K"],
                     row["m_Ca"], row["m_Mg"], row["m_SO4"]], dtype=float)
    ions = np.nan_to_num(ions)
    T, P = float(row["T_K"]), float(row["P_bar"])
    if kind == "rho":
        return rho_brine(T, P, ions, backend)[0]
    if kind == "yh2o":
        return y_h2o(T, P, ions, gas, backend)
    # CH4 rows are carried as molality, CO2 rows as salt-free mole fraction;
    # see fitting/dataset.py for why each gas is taken on its densest basis.
    return solubility(T, P, ions, gas, backend,
                      as_molality=(gas == "ch4"))


def predict_rho_family(df, backend="python"):
    """Density for a whole family, warm-started along each isotherm.

    The inner 3-variable Newton has more than one root, and the cold seed
    [0.05, 50, 0.10] lands on the wrong one for a large minority of states.
    scripts/fig_brine_density.py threads the converged state from the
    previous pressure into the next call (`v_init=state`) for exactly this
    reason, and scoring row-by-row from cold reproduces neither its numbers
    nor its convergence: 4.78 % against its 1.74 %, on the same points.

    So group by (salt system, T, molality), walk each group in pressure
    order, and carry the state -- the same traversal the figure script uses.
    """
    ion_cols = ["m_Na", "m_Cl", "m_K", "m_Ca", "m_Mg", "m_SO4"]
    d = df.reset_index(drop=True)
    key = d[ion_cols].fillna(0.0).round(6).astype(str).agg("|".join, axis=1)
    out = np.full(len(d), np.nan)
    for _, idx in d.groupby([key, d["T_K"]]).groups.items():
        idx = list(idx)
        r0 = d.loc[idx[0]]
        mi = np.nan_to_num(np.array([r0[c] for c in ion_cols], dtype=float))
        T = float(r0["T_K"])
        # Walk the same dense pressure grid the figure script walks
        # (linspace(5, 700, 200)), carrying the inner state, and evaluate the
        # experimental points from the state at the END of that walk. Warm
        # starting from the previous EXPERIMENTAL point instead is not the
        # same thing and does not reproduce the published numbers -- it gave
        # 2.72 % where the figure script gives 1.74 % on the same rows.
        state = None
        for Pg in _RHO_WARM_GRID:
            _, state = rho_brine(T, float(Pg), mi, backend, v_init=state)
        for i in idx:
            out[i] = rho_brine(T, float(d.loc[i, "P_bar"]), mi,
                               backend, v_init=state)[0]
    return out
