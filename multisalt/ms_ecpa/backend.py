"""Route ``stability_flash`` to the compiled Fortran kernel, for every caller at once.

The Fortran kernel is about 190 times faster than the Python kernel on a whole
``stability_flash`` call and agrees with it to 1e-13
(``reproduce/verify_reference.py`` compares 1,920 states point by point).
Switching it on here speeds up the sweep, the figure scripts and the fitting
code without any change to the callers.

Enabling it
-----------
    MS_ECPA_BACKEND=fortran        environment variable, or
    ms_ecpa.backend.use("fortran") in the running process

``MS_ECPA_PARFILE`` overrides the parameter file (default: the Papers II and
III parameter set).

Salinity convention
-------------------
The compiled kernel is valid only with ``salinity_convention="as-aqueous"``,
the library default and the convention used for every number in Papers II and
III. With ``"chain"`` the two kernels differ (3.8 % on x1w and 9.5 % on eps_r
at 6 mol/kg NaCl). A call with ``"chain"`` therefore falls back to the Python
kernel.

Parameters
----------
The compiled kernel holds its own copy of every parameter, read from the
parameter file when the ``Engine`` is constructed. Changes made at run time to
``ms_ecpa.parameters`` (the feature flags, ``set_ion_ion_du``,
``paper1_config.apply_paper1``) do not reach it. ``dispatch`` therefore also
falls back to the Python kernel whenever the Python-side parameters differ
from those the engine was built from. ``fitting`` binds both parameter stores
(``blocks.bind_fortran``) and calls its own engine directly.
"""
from __future__ import annotations

import os
import time
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import parameters as _prm

_ROOT = Path(__file__).resolve().parents[1]
_ENGINE = None
_MODE = None

#: Fortran route code -> the message the Python kernel would have produced.
_ROUTE = {
    0: "single-phase (hard guard)",
    1: "two-phase (trial K-init)",
    2: "two-phase (Wilson K-init)",
    3: "two-phase (water-biased K-init)",
    4: "two-phase (pure-SSI)",
    5: "two-phase (dual-trial K-init)",
    6: "two-phase (dew-point rescue)",
    7: "single-phase (beta-degenerate flash)",
    8: "soft fallback",
    9: "hard failure",
    10: "two-phase (simulator warm-start)",
    -1: "stability error",
}


class _DeferToPython(RuntimeError):
    """Raised so ``dispatch`` falls back to the reference kernel."""


#: Fortran route code for the dew-point rescue. Kept for diagnostics -- it is
#: how a caller identifies states that took the specialised near-saturation
#: path, which are worth watching.
#:
#: These were briefly deferred to the Python kernel, while only Python carried
#: the dissolved-gas refinement. ecpa_flash.f90 now carries it too, and the
#: two agree to 1e-15 relative on those states, so the deferral is gone and
#: with it a ~15x slowdown on the salt-free binary family.
DEW_RESCUE_ROUTE = 6


def mode() -> str:
    """Return the name of the kernel that ``stability_flash`` calls are routed to.

    On the first call the value is read from the environment variable
    ``MS_ECPA_BACKEND`` (default ``"python"``); after that it changes only
    through ``use``.

    Returns
    -------
    str
        ``"python"`` or ``"fortran"`` (lower case).  The value is the one
        requested; it does not check that the compiled library can be loaded
        (see ``available``).
    """
    global _MODE
    if _MODE is None:
        _MODE = os.environ.get("MS_ECPA_BACKEND", "python").lower()
    return _MODE


def use(which: str) -> None:
    """Select the Python or Fortran kernel for ``stability_flash`` in this process.

    With ``"fortran"``, calls made through ``dispatch`` go to
    ``stability_flash_fortran``.  A call falls back to the Python kernel,
    with a warning the first time for each reason (see ``fallbacks``), when
    ``salinity_convention`` is not ``"as-aqueous"``, when a keyword argument
    cannot be passed to the compiled kernel, when ``ms_ecpa.parameters`` was
    changed after import, when an extension is active, or when the compiled
    call fails.

    Parameters
    ----------
    which : {"python", "fortran"}
        Kernel to use.

    Raises
    ------
    ValueError
        If ``which`` is not ``"python"`` or ``"fortran"``.

    Notes
    -----
    The result of the compiled kernel has the fields of
    ``StabilityFlashResult`` that callers usually read (``phases``,
    ``tpd_min``, ``stable``, ``flash_result``, ``message``, timings), but no
    ``route`` attribute; ``stability`` is None and ``t_stab_ms`` and
    ``t_flash_ms`` are NaN.
    """
    global _MODE
    if which not in ("python", "fortran"):
        raise ValueError(f"unknown backend {which!r}")
    _MODE = which


def available() -> bool:
    """Return True if the compiled Fortran library can be loaded in this process.

    The check builds the engine (``fortran/tools/ecpa_ctypes.Engine``) with
    the parameter file named by ``MS_ECPA_PARFILE``, or the Papers II/III
    parameter file by default.  The engine is kept for later calls.

    Returns
    -------
    bool
        True if the engine was built; False if any error occurred (for
        example, the library is not compiled or the parameter file is
        missing).
    """
    try:
        _engine()
        return True
    except Exception:
        return False


def _engine():
    """Per-PROCESS engine. ctypes handles are not shared across processes."""
    global _ENGINE
    if _ENGINE is None:
        tools = _ROOT / "fortran" / "tools"
        if str(tools) not in sys.path:
            sys.path.insert(0, str(tools))
        from ecpa_ctypes import Engine
        par = os.environ.get(
            "MS_ECPA_PARFILE",
            str(_ROOT / "fortran" / "params" / "paperII_III.par"))
        _ENGINE = Engine(parfile=par)
    return _ENGINE


def _param_snapshot() -> dict:
    """Every number in ``ms_ecpa.parameters``, as it stands right now.

    Taken at import, which is before any caller can have touched the
    module, and compared per dispatched call. Arrays are copied because
    the runtime mutation paths (``set_ion_ion_du``, ``apply_paper1``)
    write into them IN PLACE -- the consumer modules alias the same
    ndarray objects, so a live reference would track the mutation and see
    nothing change.
    """
    snap = {}
    for name, val in vars(_prm).items():
        if name.startswith("_"):
            continue
        if isinstance(val, (bool, int, float)):
            snap[name] = val
        elif isinstance(val, np.ndarray):
            snap[name] = val.copy()
    return snap


#: Parameter state the parfile behind ``_engine`` was generated from
#: (fortran/tools/gen_params.py writes the .par out of this same module).
_PAR_REF = _param_snapshot()


def _drifted_parameter():
    """Name of the first parameter the compiled engine no longer shares.

    ``None`` when the Python store is still the one the parfile was
    written from, which is the case for every figure, sweep and
    verification script in the repo -- they all run pristine parameters,
    so this costs a comparison and never fires.
    """
    for name, ref in _PAR_REF.items():
        cur = getattr(_prm, name, None)
        if isinstance(ref, np.ndarray):
            if not (isinstance(cur, np.ndarray) and cur.shape == ref.shape
                    and np.array_equal(cur, ref)):
                return name
        elif cur != ref:
            return name
    return None


#: Reason -> how many dispatched calls fell back to the Python kernel for it.
_FALLBACKS: dict = {}


def fallbacks() -> dict:
    """Why dispatched calls fell back to the Python kernel, and how often.

    Empty after a run that really did go through the compiled kernel
    throughout -- which is otherwise not observable from the results,
    since the fallback returns the same numbers to 1e-13. Worth printing
    next to ``mode`` in anything that reports which kernel it used.
    """
    return dict(_FALLBACKS)


def _note_fallback(exc: Exception) -> None:
    """Record a fallback, and say so once per distinct reason."""
    reason = f"{type(exc).__name__}: {exc}"
    _FALLBACKS[reason] = n = _FALLBACKS.get(reason, 0) + 1
    if n == 1:
        warnings.warn(
            f"ms_ecpa.backend: compiled kernel not used ({reason}); "
            "answering with the Python kernel instead. Same numbers, "
            "~190x slower. backend.fallbacks() lists every reason.",
            RuntimeWarning, stacklevel=3)


@dataclass
class _Flash:
    """The FlashKVResult fields callers actually read."""
    converged: bool
    message: str
    K: tuple
    x_aq: tuple
    y_gas: tuple
    beta: float
    x1w: float
    Z_aq: float
    Z_gas: float
    eps_r: float
    n_iter_ssi: int
    n_iter_newton: int
    max_residual: float
    fugacities_aq: tuple
    fugacities_gas: tuple
    mi_aq: tuple
    chi_H2O_aq: float
    chi_H2O_gas: float
    n_iter_aq_inner: int = 0
    n_iter_gas_inner: int = 0
    salinity_convention: str = "as-aqueous"


@dataclass
class _Stab:
    phases: int
    tpd_min: float
    stable: bool
    flash_result: object
    message: str
    attempts: tuple = ()
    stability: object = None
    t_stab_ms: float = float("nan")
    t_flash_ms: float = float("nan")
    t_total_ms: float = float("nan")


def stability_flash_fortran(T, P, z, mi, *, salinity_convention="as-aqueous",
                            **kwargs):
    """Compiled stability + flash, shaped like the Python result.

    Any keyword the compiled path cannot honour (tolerances, iteration caps,
    solver switches) raises, which makes ``dispatch`` fall back to the
    Python kernel -- and the Python kernel DOES honour them. Previously these
    were swallowed by a ``**_ignored`` catch-all, so a caller asking for
    ``maxiter=500`` silently got the compiled default and a result that looked
    like the request had been met.
    """
    if kwargs:
        raise _DeferToPython(
            "compiled kernel cannot honour " + ", ".join(sorted(kwargs)))
    drifted = _drifted_parameter()
    if drifted is not None:
        raise _DeferToPython(
            f"ms_ecpa.parameters.{drifted} was changed after import; the "
            "compiled engine still holds the parfile value")
    from . import extensions as _ext
    if _ext.active():
        # The compiled engine evaluates the unextended model; an active
        # Python-side extension would be silently ignored there.
        raise _DeferToPython(
            "an ms_ecpa.extensions extension is active; the compiled "
            "engine does not carry it")
    t0 = time.perf_counter()
    r = _engine().stability_flash(float(T), float(P),
                                  np.asarray(z, dtype=float),
                                  np.asarray(mi, dtype=float),
                                  salinity_convention=salinity_convention)
    t_total_ms = 1e3 * (time.perf_counter() - t0)
    # The compiled entry point reports failure through rc, and `phases` is
    # not meaningful when rc != 0 -- a failed call that happens to leave
    # phases = 1 would otherwise be returned as a confident single-phase
    # verdict, which is indistinguishable from a real one. Defer to the
    # Python kernel instead, the same way an unhonourable keyword does.
    # Measured 2026-09-19: rc == 0 on every one of the 5,857 calls made by
    # the full experimental set, so this is a guard, not a live path.
    rc = int(r.get("rc", 0))
    if rc != 0:
        raise _DeferToPython(f"compiled stability_flash returned rc={rc}")
    phases = int(r.get("phases", 0))
    route = int(r.get("route", -99))
    msg = _ROUTE.get(route, f"route={r.get('route')}")
    fr = None
    if r.get("has_flash") and r.get("flash"):
        f = r["flash"]
        fr = _Flash(
            converged=bool(f["converged"]), message=msg,
            K=tuple(float(v) for v in f["K"]),
            x_aq=tuple(float(v) for v in f["x_aq"]),
            y_gas=tuple(float(v) for v in f["y_gas"]),
            beta=float(f["beta"]), x1w=float(f.get("x1w", float("nan"))),
            Z_aq=float(f["Z_aq"]), Z_gas=float(f["Z_gas"]),
            eps_r=float(f["eps_r"]),
            n_iter_ssi=int(f.get("n_iter_ssi", 0)),
            n_iter_newton=int(f.get("n_iter_newton", 0)),
            max_residual=float(f.get("max_residual", float("nan"))),
            fugacities_aq=tuple(float(v) for v in f["fugacities_aq"]),
            fugacities_gas=tuple(float(v) for v in f["fugacities_gas"]),
            mi_aq=tuple(float(v) for v in f["mi_aq"]),
            chi_H2O_aq=float(f.get("chi_H2O_aq", float("nan"))),
            chi_H2O_gas=float(f.get("chi_H2O_gas", float("nan"))),
            n_iter_aq_inner=int(f.get("n_iter_aq_inner", 0)),
            n_iter_gas_inner=int(f.get("n_iter_gas_inner", 0)),
            salinity_convention=salinity_convention)
    return _Stab(phases=phases, tpd_min=float(r.get("tpd_min", float("nan"))),
                 stable=(phases == 1), flash_result=fr, message=msg,
                 # Measured, not the 0.0 these used to default to. The split
                 # between stability and flash is not observable from out here,
                 # so those two stay NaN rather than claiming a number.
                 t_stab_ms=float("nan"), t_flash_ms=float("nan"),
                 t_total_ms=t_total_ms)


def dispatch(py_impl):
    """Wrap the Python kernel so it defers to Fortran when selected."""
    def _call(T, P, z, mi, *args, **kw):
        if (mode() == "fortran"
                and kw.get("salinity_convention", "as-aqueous") == "as-aqueous"):
            try:
                return stability_flash_fortran(T, P, z, mi, **kw)
            except Exception as exc:
                # A compiled-kernel failure must never silently become a
                # different ANSWER. Fall back to the reference and let it
                # decide, at Python speed -- but leave a record, because a
                # stale build or a bad MS_ECPA_PARFILE makes every call
                # fall back and the results are then indistinguishable
                # from a compiled run except by the wall clock.
                _note_fallback(exc)
        return py_impl(T, P, z, mi, *args, **kw)
    _call.__name__ = getattr(py_impl, "__name__", "stability_flash")
    _call.__doc__ = py_impl.__doc__
    _call._python_impl = py_impl
    return _call
