"""Single-point worker for the multi-salt eCPA sweep.

The worker takes one :class:`SweepPoint`, runs the stability+flash, and
returns a flat dict with EVERY input, output, warm-start state, and
performance metric we may want for later analysis. It never raises:
errors are captured as ``converged=False`` rows with ``error_type`` and
``error_msg`` set, so the per-row coverage of the sweep stays complete
and we can iterate later on the conditions where the model currently
fails.

The flat-dict design is deliberate: parquet writes column-by-column and
the schema stays stable as we extend things, so a 1M-row dataset stays
queryable from pandas/polars/duckdb without bespoke loaders.
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

# Sweep-level solver-strategy knobs read from the environment so a single
# code base can drive both benchmark passes without a rebuild:
#   SWEEP_USE_NEWTON  (default "1")   → set to "0" for pure-SSI comparison,
#                                        which effectively equals newton_tol=tol
#                                        (Newton switch never fires); forwarded
#                                        as use_newton=False to flash_kv.
#   SWEEP_SSI_MAXITER (default "80")  → outer SSI iteration cap; raise for the
#                                        SSI-only run so hard points don't
#                                        unfairly hit the ceiling and skew
#                                        iteration-count statistics.
# Both reach flash_kv on the Python backend only; the Fortran backend has no
# path for either, so run_one refuses that combination (see below).
_SSI_MAXITER_DEFAULT = 80
_USE_NEWTON = os.environ.get("SWEEP_USE_NEWTON", "1").lower() not in ("0", "false", "no")
_SSI_MAXITER = int(os.environ.get("SWEEP_SSI_MAXITER", str(_SSI_MAXITER_DEFAULT)))

# The sweep can be invoked from anywhere; make sure code/ is on the path.
_HERE = Path(__file__).resolve()
_CODE = _HERE.parents[1]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from ms_ecpa import stability_flash    # noqa: E402
from ms_ecpa.parameters import Mw      # noqa: E402
from ms_ecpa.roots import is_liquid_root  # noqa: E402

from .sweep_config import SweepPoint  # noqa: E402


# Component molar masses (kg/mol) for density back-out.
_M_GAS = (Mw, 0.01604, 0.04401)              # H2O, CH4, CO2
_M_ION = (0.02299, 0.03545, 0.03910,         # Na, Cl, K
          0.04008, 0.02431, 0.09606)         # Ca, Mg, SO4
_R_J_PER_MOL_K = 8.31446261815324


# ---------------------------------------------------------------------------
# Output schema (flat dict columns)
# ---------------------------------------------------------------------------

_OUTPUT_KEYS = (
    # --- INPUT (echoed verbatim for joinability) ---
    "point_id", "config_name", "feed_family", "salt_name",
    "T_K", "P_bar",
    "z_H2O", "z_CH4", "z_CO2",
    "m_Na", "m_Cl", "m_K", "m_Ca", "m_Mg", "m_SO4",
    "salinity_convention",
    # --- HIGH-LEVEL OUTCOME ---
    "converged", "phases", "tpd_min",
    "error_type", "error_msg",
    # Which rung of the stability_flash ladder actually produced the answer,
    # taken from StabilityFlashResult.message. The ladder is, in order:
    # flash_kv seeded from the TPD trial, then Wilson K, then water-biased,
    # then pure SSI, then dual-trial K, and finally dew_point_flash -- the
    # "dew-point rescue" the manuscript describes, which fixes x_aq at the
    # TPD composition and solves a scalar water-fugacity balance for beta
    # because the K-value SSI collapses to the trivial K = 1 attractor near
    # the dew line. Recording it is what makes "how often does the rescue
    # fire, and where" answerable: until now the sweep kept the message only
    # on FAILURE, so a state rescued successfully was indistinguishable from
    # one that converged on the first attempt.
    "solve_path",
    # --- PHASE EQUILIBRIUM ---
    "beta",
    "x_H2O_aq", "x_CH4_aq", "x_CO2_aq",
    "y_H2O_gas", "y_CH4_gas", "y_CO2_gas",
    "K_H2O", "K_CH4", "K_CO2",
    # --- EQUILIBRIUM AQUEOUS MOLALITY (chain output; equals m_prep otherwise) ---
    "m_Na_aq", "m_Cl_aq", "m_K_aq", "m_Ca_aq", "m_Mg_aq", "m_SO4_aq",
    # --- FUGACITIES (bar) — useful for downstream consistency checks ---
    "f_H2O_aq", "f_CH4_aq", "f_CO2_aq",
    "f_H2O_gas", "f_CH4_gas", "f_CO2_gas",
    # --- WARM-START STATE (everything needed to seed a future call) ---
    "Z_aq", "Z_gas", "eps_r",
    "chi_H2O_aq", "chi_H2O_gas",
    # --- DENSITIES (kg/m^3) ---
    "rho_aq", "rho_gas",
    # --- ITERATION COUNTS ---
    "n_iter_ssi", "n_iter_newton_polish",
    "n_iter_aq_inner_diag", "n_iter_gas_inner_diag",
    "max_residual",
    # --- TIMING ---
    # t_wall_ms: end-to-end stability_flash call as timed by sweep_core
    #            (kept for backwards compatibility with earlier parquets).
    # t_stab_ms: Michelsen TPD test alone (skippable in a warm-started
    #            reservoir simulator that trusts the previous timestep's
    #            phase state).
    # t_flash_ms: sum of every flash_kv attempt actually run, including
    #            attempts rejected as β-degenerate or phase-swapped and
    #            excluding the post-conv Newton-diagnostic reconstructor.
    # t_total_ms: stability_flash-internal end-to-end wall time
    #            (t_wall_ms - t_total_ms measures sweep_core overhead).
    # The last three are reported by the Python kernel only. On
    # ECPA_SWEEP_BACKEND=fortran the engine returns no timings and all three
    # stay NaN, so a merged parquet mixing backends carries timing means over
    # the Python subset alone -- aggregate them alongside a row count.
    "t_wall_ms", "t_stab_ms", "t_flash_ms", "t_total_ms",
)


def _empty_row() -> dict[str, Any]:
    """Schema-stable row template; all numeric outputs default to NaN, all
    iteration counts to 0, all strings to empty."""
    row: dict[str, Any] = {}
    for k in _OUTPUT_KEYS:
        if k in ("point_id",
                 "phases", "n_iter_ssi", "n_iter_newton_polish",
                 "n_iter_aq_inner_diag", "n_iter_gas_inner_diag"):
            row[k] = 0
        elif k in ("config_name", "feed_family", "salt_name",
                   "salinity_convention", "error_type", "error_msg"):
            row[k] = ""
        elif k == "converged":
            row[k] = False
        else:
            row[k] = float("nan")
    return row


def _phase_density_kg_per_m3(x_or_y, Z, T_K, P_bar, mi_aq=None):
    """Mass density from EoS Z, composition, and (T, P). For the aqueous
    phase the ion mole fractions enter via the closure
    ``x_ion = M_w · x_w · m_i,aq``; for the gas phase ions are absent.

    NOTE: this back-out uses ``v = Z·R·T/P`` only. The library's canonical
    aqueous molar volume (ms_ecpa/eos.py) adds the fitted Peneloux ion
    shifts, ``v = Z·R·T/P + Σ x_iw·Pen_i``, so ``rho_aq`` below is the
    untranslated value and reads low in brine (-6 % at NaCl 1 m, -17 % at
    3 m, 323 K / 100 bar). The published figures use this untranslated
    value, so it is kept here.
    """
    if not np.isfinite(Z) or Z <= 0:
        return float("nan")
    P_Pa = float(P_bar) * 1e5
    V_m3_per_mol = float(Z) * _R_J_PER_MOL_K * float(T_K) / P_Pa
    # Per-mole-of-phase mass: Σ x_i · M_i, where for aq we include ions.
    M_mix = sum(x_or_y[i] * _M_GAS[i] for i in range(3))
    if mi_aq is not None:
        x_w = x_or_y[0]
        for k, m_i in enumerate(mi_aq):
            x_ion = Mw * x_w * float(m_i)
            M_mix += x_ion * _M_ION[k]
    rho = float(M_mix / V_m3_per_mol)
    if mi_aq is not None and not is_liquid_root(rho, float(T_K)):
        # The three-variable aqueous closure has more than one root, and a
        # converged Newton has only proved it found *a* root: on some brine
        # states it settles on Zw ~ 350, which reports a density near
        # 1 kg/m3 at several hundred bar. isfinite(Z) and Z > 0 both pass
        # there, so without this bracket the vapour-branch number would be
        # written to a converged=True row. Same predicate the fitted
        # density route uses (fitting/predict.py); NaN is the honest answer.
        return float("nan")
    return rho


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Optional Fortran backend
# ---------------------------------------------------------------------------
# The sweep has always run the pure-Python ms_ecpa. The Fortran port exists --
# it was built for the benchmark paper -- and on this machine it is about
# 159x faster on a whole stability_flash call (387.7 ms -> 2.4 ms;
# fortran/tools/bench.py). At half a million grid points that is the
# difference between a scheduled campaign and a coffee break, and it matters
# far more for the refit, where the flash is driven 1e5-1e6 times per
# optimisation and again for every fold of a leave-one-source-out protocol.
#
# Set ECPA_SWEEP_BACKEND=fortran to use it. ECPA_SWEEP_PARFILE overrides the
# parameter file (default: the Papers II/III production set).
#
# CONSTRAINT: only under salinity_convention="as-aqueous". Python and Fortran
# disagree under "chain" (3.8 % on x1w, 9.5 % on eps_r at NaCl 6 m) for
# reasons not yet identified. run_one refuses the combination rather than
# producing numbers that differ from the Python reference.
#
# The Fortran carries its own route code for which rung of the ladder
# produced the answer, which is the same thing solve_path records on the
# Python side, so the two backends stay comparable.
_ROUTE_LABEL = {
    0:  "single-phase (hard guard)",
    1:  "two-phase (trial K-init)",
    2:  "two-phase (Wilson K-init)",
    3:  "two-phase (water-biased K-init)",
    4:  "two-phase (pure-SSI)",
    5:  "two-phase (dual-trial K-init)",
    6:  "two-phase (dew-point rescue)",
    7:  "single-phase (beta-degenerate flash)",
    8:  "soft fallback",
    9:  "hard failure",
    10: "two-phase (simulator warm-start)",
    -1: "stability error",
}

_ENGINE = None          # one per worker PROCESS; ctypes handles are not shared


def _fortran_engine():
    """Lazily open this process's engine. Never shared across processes."""
    global _ENGINE
    if _ENGINE is None:
        import os
        here = Path(__file__).resolve().parents[1]
        tools = here / "fortran" / "tools"
        if str(tools) not in sys.path:
            sys.path.insert(0, str(tools))
        from ecpa_ctypes import Engine
        par = os.environ.get("ECPA_SWEEP_PARFILE",
                             str(here / "fortran" / "params" / "paperII_III.par"))
        _ENGINE = Engine(parfile=par)
    return _ENGINE


def _use_fortran() -> bool:
    import os
    return os.environ.get("ECPA_SWEEP_BACKEND", "").lower() == "fortran"


def _row_from_fortran(point: SweepPoint, row: dict[str, Any], res: dict) -> dict:
    """Fill a sweep row from the Fortran engine's result dict."""
    row["phases"] = int(res.get("phases", 0))
    tpd = float(res.get("tpd_min", float("nan")))
    row["tpd_min"] = tpd if np.isfinite(tpd) else float("nan")
    route = int(res.get("route", -99))
    row["solve_path"] = _ROUTE_LABEL.get(route, f"route={route}")[:80]

    fr = res.get("flash") if res.get("has_flash") else None
    if row["phases"] == 1 or not fr:
        row["converged"] = (row["phases"] == 1)
        if row["phases"] != 1:
            row["error_type"] = "FlashUnconverged"
            row["error_msg"] = row["solve_path"]
        return row

    row["converged"] = bool(fr["converged"])
    row["beta"] = float(fr["beta"])
    row["x_H2O_aq"], row["x_CH4_aq"], row["x_CO2_aq"] = (float(v) for v in fr["x_aq"])
    row["y_H2O_gas"], row["y_CH4_gas"], row["y_CO2_gas"] = (float(v) for v in fr["y_gas"])
    row["K_H2O"], row["K_CH4"], row["K_CO2"] = (float(v) for v in fr["K"])
    (row["m_Na_aq"], row["m_Cl_aq"], row["m_K_aq"],
     row["m_Ca_aq"], row["m_Mg_aq"], row["m_SO4_aq"]) = (float(v) for v in fr["mi_aq"])
    row["f_H2O_aq"], row["f_CH4_aq"], row["f_CO2_aq"] = (float(v) for v in fr["fugacities_aq"])
    row["f_H2O_gas"], row["f_CH4_gas"], row["f_CO2_gas"] = (float(v) for v in fr["fugacities_gas"])
    row["Z_aq"] = float(fr["Z_aq"])
    row["Z_gas"] = float(fr["Z_gas"])
    row["eps_r"] = float(fr["eps_r"])
    row["chi_H2O_aq"] = float(fr["chi_H2O_aq"])
    row["chi_H2O_gas"] = float(fr["chi_H2O_gas"])
    row["n_iter_ssi"] = int(fr["n_iter_ssi"])
    row["n_iter_newton_polish"] = int(fr["n_iter_newton"])
    row["n_iter_aq_inner_diag"] = int(fr["n_iter_aq_inner"])
    row["n_iter_gas_inner_diag"] = int(fr["n_iter_gas_inner"])
    row["max_residual"] = float(fr["max_residual"])
    row["x1w"] = float(fr.get("x1w", float("nan")))
    # Densities are not returned by the C boundary; they are a pure function
    # of composition, Z, T and P, so they are computed here exactly as the
    # Python backend does.
    row["rho_aq"] = _phase_density_kg_per_m3(
        fr["x_aq"], fr["Z_aq"], point.T_K, point.P_bar, mi_aq=fr["mi_aq"])
    row["rho_gas"] = _phase_density_kg_per_m3(
        fr["y_gas"], fr["Z_gas"], point.T_K, point.P_bar, mi_aq=None)
    if not row["converged"]:
        row["error_type"] = "FlashUnconverged"
        row["error_msg"] = row["solve_path"]
    return row


def run_one(point: SweepPoint, *, timeout_s: float | None = None) -> dict[str, Any]:
    """Evaluate the stability+flash at ``point`` and return a flat dict row.

    Never raises. Failures are encoded as ``converged=False`` rows with
    ``error_type`` (Python exception name) and ``error_msg`` populated so
    downstream code can group conditions by failure mode and iterate.

    The ``timeout_s`` argument is reserved for future use (signal-based
    timeouts are not portable inside child workers); for now the inner
    Newton/SSI iteration limits already bound the per-call wall-clock.
    """
    row = _empty_row()
    # Echo inputs verbatim.
    row.update(dict(
        point_id=point.point_id, config_name=point.config_name,
        feed_family=point.feed_family, salt_name=point.salt_name,
        T_K=point.T_K, P_bar=point.P_bar,
        z_H2O=point.z_H2O, z_CH4=point.z_CH4, z_CO2=point.z_CO2,
        m_Na=point.m_Na, m_Cl=point.m_Cl, m_K=point.m_K,
        m_Ca=point.m_Ca, m_Mg=point.m_Mg, m_SO4=point.m_SO4,
        salinity_convention=point.salinity_convention,
    ))
    z = np.array([point.z_H2O, point.z_CH4, point.z_CO2], dtype=float)
    mi = np.array([point.m_Na, point.m_Cl, point.m_K,
                   point.m_Ca, point.m_Mg, point.m_SO4], dtype=float)

    if _use_fortran():
        if point.salinity_convention != "as-aqueous":
            row["error_type"] = "BackendRefused"
            row["error_msg"] = (
                "the Fortran backend is only validated under "
                "salinity_convention='as-aqueous'; it disagrees with the "
                "Python reference under 'chain' (3.8 % on x1w, 9.5 % on "
                "eps_r at NaCl 6 m)."
            )
            return row
        if (not _USE_NEWTON) or _SSI_MAXITER != _SSI_MAXITER_DEFAULT:
            # Neither knob has a path through the C boundary:
            # Engine.stability_flash takes only (T, P, z, mi, warm,
            # want_warm_out, salinity_convention), and ecpa_flash.f90 fixes
            # flash_opt_t at maxiter=80, use_newton=.true. with no setter.
            # Forwarding them here would be a no-op, so the pure-SSI pass
            # would run the default SSI+Newton path and be filed on disk as
            # the SSI-only benchmark -- a comparison of a run against itself.
            # The solver-strategy passes belong on the Python backend.
            row["error_type"] = "BackendRefused"
            row["error_msg"] = (
                "SWEEP_USE_NEWTON / SWEEP_SSI_MAXITER cannot be honoured by "
                "the Fortran backend (no C-boundary argument for either); "
                f"got SWEEP_USE_NEWTON={int(_USE_NEWTON)}, "
                f"SWEEP_SSI_MAXITER={_SSI_MAXITER}. Run the solver-strategy "
                "comparison on the Python backend."
            )
            return row
        t0 = time.perf_counter()
        try:
            res = _fortran_engine().stability_flash(
                float(point.T_K), float(point.P_bar), z, mi,
                salinity_convention=point.salinity_convention,
            )
        except Exception as exc:
            row["t_wall_ms"] = (time.perf_counter() - t0) * 1000.0
            row["error_type"] = type(exc).__name__
            row["error_msg"] = (str(exc) or repr(exc))[:400]
            return row
        row["t_wall_ms"] = (time.perf_counter() - t0) * 1000.0
        # t_stab_ms / t_flash_ms / t_total_ms stay NaN: the C boundary reports
        # no internal timings. t_total_ms used to be assigned the same
        # perf_counter delta as t_wall_ms, which made the overhead the schema
        # comment defines (t_wall_ms - t_total_ms) identically 0.0 on every
        # Fortran row -- an answer fixed by construction, not measured.
        return _row_from_fortran(point, row, res)

    t0 = time.perf_counter()
    try:
        r = stability_flash(
            float(point.T_K), float(point.P_bar), z, mi,
            salinity_convention=point.salinity_convention,
            use_newton=_USE_NEWTON,
            maxiter=_SSI_MAXITER,
        )
    except Exception as exc:
        row["t_wall_ms"] = (time.perf_counter() - t0) * 1000.0
        row["error_type"] = type(exc).__name__
        row["error_msg"] = (str(exc) or repr(exc))[:400]
        return row
    row["t_wall_ms"] = (time.perf_counter() - t0) * 1000.0
    # NaN, not 0.0: a result object that carries no split timing has not
    # measured 0 ms, and the consumers of these columns take a plain .mean()
    # with no finiteness test, so a fabricated 0 would round to a clean "0 ms"
    # in the performance table while NaN shows up as missing.
    _nan = float("nan")
    row["t_stab_ms"] = float(getattr(r, "t_stab_ms", _nan))
    row["t_flash_ms"] = float(getattr(r, "t_flash_ms", _nan))
    row["t_total_ms"] = float(getattr(r, "t_total_ms", _nan))

    row["phases"] = int(r.phases)
    row["tpd_min"] = float(r.tpd_min) if np.isfinite(r.tpd_min) else float("nan")
    row["solve_path"] = (r.message or "")[:80]

    # Single-phase outcome: no flash to harvest but still a converged row.
    if r.phases == 1 or r.flash_result is None:
        row["converged"] = (r.phases == 1)
        if r.phases != 1:
            row["error_type"] = "FlashUnconverged"
            row["error_msg"] = (r.message or "")[:400]
        return row

    fr = r.flash_result
    row["converged"] = bool(fr.converged)
    row["beta"] = float(fr.beta)
    row["x_H2O_aq"], row["x_CH4_aq"], row["x_CO2_aq"] = (float(x) for x in fr.x_aq)
    row["y_H2O_gas"], row["y_CH4_gas"], row["y_CO2_gas"] = (float(y) for y in fr.y_gas)
    row["K_H2O"], row["K_CH4"], row["K_CO2"] = (float(k) for k in fr.K)
    (row["m_Na_aq"], row["m_Cl_aq"], row["m_K_aq"],
     row["m_Ca_aq"], row["m_Mg_aq"], row["m_SO4_aq"]) = (float(m) for m in fr.mi_aq)
    row["f_H2O_aq"], row["f_CH4_aq"], row["f_CO2_aq"] = (float(f) for f in fr.fugacities_aq)
    row["f_H2O_gas"], row["f_CH4_gas"], row["f_CO2_gas"] = (float(f) for f in fr.fugacities_gas)
    row["Z_aq"] = float(fr.Z_aq)
    row["Z_gas"] = float(fr.Z_gas)
    row["eps_r"] = float(fr.eps_r)
    row["chi_H2O_aq"] = float(fr.chi_H2O_aq)
    row["chi_H2O_gas"] = float(fr.chi_H2O_gas)
    row["n_iter_ssi"] = int(fr.n_iter_ssi)
    row["n_iter_newton_polish"] = int(fr.n_iter_newton)
    row["n_iter_aq_inner_diag"] = int(fr.n_iter_aq_inner)
    row["n_iter_gas_inner_diag"] = int(fr.n_iter_gas_inner)
    row["max_residual"] = float(fr.max_residual)
    row["rho_aq"] = _phase_density_kg_per_m3(
        fr.x_aq, fr.Z_aq, point.T_K, point.P_bar, mi_aq=fr.mi_aq
    )
    row["rho_gas"] = _phase_density_kg_per_m3(
        fr.y_gas, fr.Z_gas, point.T_K, point.P_bar, mi_aq=None
    )

    if not fr.converged:
        row["error_type"] = "FlashUnconverged"
        row["error_msg"] = (fr.message or "")[:400]
    return row


def output_schema() -> tuple[str, ...]:
    """Return the flat output column ordering for documentation/tests."""
    return _OUTPUT_KEYS
