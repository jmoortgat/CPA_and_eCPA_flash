#!/usr/bin/env python3
"""Minimal ctypes wrapper over the Fortran libecpa C API.

Used by the golden-vector acceptance tests (test_fortran_port.py) and
as the reference example for C/C++/Python callers.

Engine contexts are opaque handles; one Engine per worker process is
the intended multiprocessing pattern.  Warm-start state is caller-owned
(the `Warm` struct) and passed per call.
"""
import ctypes as ct
import os
import platform
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_SOEXT = "dylib" if platform.system() == "Darwin" else "so"
# ECPA_LIB env var overrides the default library location so the test
# suite / bench can be pointed at an alternative build variant
# (e.g. build-ifx/, build-ifx-sapphire/) without editing code.
DEFAULT_LIB = Path(os.environ.get(
    "ECPA_LIB", HERE.parent / "build" / f"libecpa.{_SOEXT}"))


class Warm(ct.Structure):
    """Caller-owned warm-start state (mirrors FlashKVResult's
    warm-start tuple: K, Z_aq, Z_gas, eps_r, chi_H2O_aq, chi_H2O_gas,
    beta).  valid=0 -> cold start."""
    _fields_ = [
        ("K", ct.c_double * 3),
        ("Z_aq", ct.c_double), ("Z_gas", ct.c_double),
        ("eps_r", ct.c_double),
        ("chi_aq", ct.c_double), ("chi_gas", ct.c_double),
        ("beta", ct.c_double),
        ("valid", ct.c_int),
    ]


class FlashOut(ct.Structure):
    _fields_ = [
        ("converged", ct.c_int), ("errcode", ct.c_int),
        ("n_iter_ssi", ct.c_int), ("n_iter_newton", ct.c_int),
        ("n_iter_aq_inner", ct.c_int), ("n_iter_gas_inner", ct.c_int),
        ("K", ct.c_double * 3), ("x_aq", ct.c_double * 3),
        ("y_gas", ct.c_double * 3),
        ("beta", ct.c_double), ("x1w", ct.c_double),
        ("Z_aq", ct.c_double), ("Z_gas", ct.c_double),
        ("eps_r", ct.c_double),
        ("chi_aq", ct.c_double), ("chi_gas", ct.c_double),
        ("max_residual", ct.c_double),
        ("fug_aq", ct.c_double * 3), ("fug_gas", ct.c_double * 3),
        ("mi_aq", ct.c_double * 6),
    ]

    def as_dict(self):
        return dict(
            converged=bool(self.converged), errcode=int(self.errcode),
            n_iter_ssi=int(self.n_iter_ssi),
            n_iter_newton=int(self.n_iter_newton),
            n_iter_aq_inner=int(self.n_iter_aq_inner),
            n_iter_gas_inner=int(self.n_iter_gas_inner),
            K=list(self.K), x_aq=list(self.x_aq), y_gas=list(self.y_gas),
            beta=float(self.beta), x1w=float(self.x1w),
            Z_aq=float(self.Z_aq), Z_gas=float(self.Z_gas),
            eps_r=float(self.eps_r),
            chi_H2O_aq=float(self.chi_aq), chi_H2O_gas=float(self.chi_gas),
            max_residual=float(self.max_residual),
            fugacities_aq=list(self.fug_aq), fugacities_gas=list(self.fug_gas),
            mi_aq=list(self.mi_aq),
        )


def _dv(x, n):
    arr = (ct.c_double * n)()
    for i, v in enumerate(np.asarray(x, dtype=float)):
        arr[i] = v
    return arr


class EcpaError(RuntimeError):
    pass


# Salinity-convention codes accepted by the C API.  "as-aqueous" is the
# default here exactly as it is in ms_ecpa.flash.flash_kv.
SALINITY_CODES = {"as-aqueous": 0, "chain": 1}

# Salt codes for the colligative routes (ecpa_colligative.f90); same
# names/stoichiometry as total_mole_basis._SALT_STOICH.
SALT_CODES = {"NaCl": 1, "KCl": 2, "CaCl2": 3, "MgCl2": 4, "Na2SO4": 5}


def _salt_code(salt):
    if isinstance(salt, int):
        return salt
    try:
        return SALT_CODES[salt]
    except KeyError:
        raise ValueError(
            f"unknown salt {salt!r}; expected one of "
            f"{sorted(SALT_CODES)}") from None


def _salinity_code(conv):
    if isinstance(conv, int):
        return conv
    try:
        return SALINITY_CODES[conv]
    except KeyError:
        raise ValueError(
            f"unknown salinity_convention {conv!r}; "
            f"expected one of {sorted(SALINITY_CODES)}") from None


class Engine:
    def __init__(self, libpath=None, parfile=None):
        self.lib = ct.CDLL(str(libpath or DEFAULT_LIB))
        L = self.lib
        L.ecpa_create.restype = ct.c_void_p
        L.ecpa_destroy.argtypes = [ct.c_void_p]
        L.ecpa_load_params.restype = ct.c_int
        L.ecpa_load_params.argtypes = [ct.c_void_p, ct.c_char_p]
        L.ecpa_set_param.restype = ct.c_int
        L.ecpa_set_param.argtypes = [ct.c_void_p, ct.c_char_p,
                                     ct.POINTER(ct.c_double), ct.c_int]
        L.ecpa_get_param.restype = ct.c_int
        L.ecpa_get_param.argtypes = [ct.c_void_p, ct.c_char_p,
                                     ct.POINTER(ct.c_double), ct.c_int]
        L.ecpa_lnphi_aq.restype = ct.c_int
        L.ecpa_lnphi_aq.argtypes = [
            ct.c_void_p, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double),
            ct.c_double, ct.c_double, ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double), ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_int)]
        L.ecpa_lnphi_gas.restype = ct.c_int
        L.ecpa_lnphi_gas.argtypes = [
            ct.c_void_p, ct.POINTER(ct.c_double), ct.c_double, ct.c_double,
            ct.c_double, ct.c_double, ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double)]
        L.ecpa_flash_kv.restype = ct.c_int
        L.ecpa_flash_kv.argtypes = [
            ct.c_void_p, ct.c_double, ct.c_double,
            ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.c_int,
            ct.c_void_p, ct.POINTER(FlashOut), ct.c_void_p]
        L.ecpa_stability_flash.restype = ct.c_int
        L.ecpa_stability_flash.argtypes = [
            ct.c_void_p, ct.c_double, ct.c_double,
            ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.c_int,
            ct.c_void_p,
            ct.POINTER(ct.c_int), ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_int), ct.POINTER(ct.c_int),
            ct.POINTER(FlashOut), ct.c_void_p]

        L.ecpa_a_res.restype = ct.c_int
        L.ecpa_a_res.argtypes = [
            ct.c_void_p, ct.POINTER(ct.c_double), ct.c_double, ct.c_double,
            ct.POINTER(ct.c_double)]
        L.ecpa_ln_f_w.restype = ct.c_int
        L.ecpa_ln_f_w.argtypes = [
            ct.c_void_p, ct.POINTER(ct.c_double), ct.c_double, ct.c_double,
            ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)]
        L.ecpa_phi_osm.restype = ct.c_int
        L.ecpa_phi_osm.argtypes = [
            ct.c_void_p, ct.c_int, ct.c_double, ct.c_double, ct.c_double,
            ct.POINTER(ct.c_double)]
        L.ecpa_rho_brine.restype = ct.c_int
        L.ecpa_rho_brine.argtypes = [
            ct.c_void_p, ct.c_double, ct.c_double,
            ct.POINTER(ct.c_double), ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)]
        L.ecpa_psat_brine.restype = ct.c_int
        L.ecpa_psat_brine.argtypes = [
            ct.c_void_p, ct.c_double, ct.c_double, ct.c_double,
            ct.c_double, ct.c_double, ct.POINTER(ct.c_double)]

        self.h = L.ecpa_create()
        if not self.h:
            raise EcpaError("ecpa_create failed")
        if parfile is not None:
            self.load(parfile)

    def close(self):
        if getattr(self, "h", None):
            self.lib.ecpa_destroy(self.h)
            self.h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # -- parameters ----------------------------------------------------
    def load(self, path):
        rc = self.lib.ecpa_load_params(self.h, str(path).encode())
        if rc != 0:
            raise EcpaError(f"ecpa_load_params({path}) rc={rc} "
                            f"(ierr={rc // 100000}, line={rc % 100000})")

    def set_param(self, name, values):
        vals = np.atleast_1d(np.asarray(values, dtype=float))
        rc = self.lib.ecpa_set_param(self.h, name.encode(),
                                     _dv(vals, len(vals)), len(vals))
        if rc != 0:
            raise EcpaError(f"ecpa_set_param({name}) rc={rc}")

    def get_param(self, name):
        buf = (ct.c_double * 36)()
        rc = self.lib.ecpa_get_param(self.h, name.encode(), buf, 36)
        if rc < 0:
            raise EcpaError(f"ecpa_get_param({name}) rc={rc}")
        return np.array(buf[:rc])

    # -- kernels -------------------------------------------------------
    def lnphi_aq(self, x_aq, mi, T, P_Pa, v_init=None):
        lnphi = (ct.c_double * 3)()
        v_out = (ct.c_double * 3)()
        niter = ct.c_int(0)
        vi = _dv(v_init if v_init is not None else [-1.0, 0.0, 0.0], 3)
        rc = self.lib.ecpa_lnphi_aq(self.h, _dv(x_aq, 3), _dv(mi, 6),
                                    T, P_Pa, vi, lnphi, v_out, ct.byref(niter))
        if rc != 0:
            raise EcpaError(f"lnphi_aq failed rc={rc}")
        return list(lnphi), list(v_out), niter.value

    def lnphi_gas(self, y_gas, T, P_Pa, Z_init=None, chi_init=0.99):
        lnphi = (ct.c_double * 3)()
        v_out = (ct.c_double * 2)()
        rc = self.lib.ecpa_lnphi_gas(self.h, _dv(y_gas, 3), T, P_Pa,
                                     -1.0 if Z_init is None else float(Z_init),
                                     float(chi_init), lnphi, v_out)
        if rc != 0:
            raise EcpaError(f"lnphi_gas failed rc={rc}")
        return list(lnphi), list(v_out)

    def flash_kv(self, T, P_bar, z, mi, warm=None, want_warm_out=False,
                 salinity_convention="as-aqueous"):
        out = FlashOut()
        warm_out = Warm()
        rc = self.lib.ecpa_flash_kv(
            self.h, T, P_bar, _dv(z, 3), _dv(mi, 6),
            _salinity_code(salinity_convention),
            ct.byref(warm) if warm is not None else None,
            ct.byref(out),
            ct.byref(warm_out) if want_warm_out else None)
        d = out.as_dict()
        d["rc"] = rc
        if want_warm_out:
            d["warm_out"] = warm_out
        return d

    # -- colligative routes (phase 2) ----------------------------------
    def a_res(self, n, V, T):
        """Total-mole-basis A^r/RT (extensive) at a given volume.
        Mirrors scripts/total_mole_basis.py::A_r(basis="total")."""
        A = ct.c_double(0.0)
        rc = self.lib.ecpa_a_res(self.h, _dv(n, 7), float(V), float(T),
                                 ct.byref(A))
        if rc != 0:
            raise EcpaError(f"ecpa_a_res rc={rc}")
        return A.value

    def ln_f_w(self, n, T, P_Pa):
        """(ln f_w [ln Pa], V [m3]).  Mirrors total_mole_basis::ln_f_w
        and ::solve_V (basis="total")."""
        lnf = ct.c_double(0.0)
        V = ct.c_double(0.0)
        rc = self.lib.ecpa_ln_f_w(self.h, _dv(n, 7), float(T), float(P_Pa),
                                  ct.byref(lnf), ct.byref(V))
        if rc != 0:
            return float("nan"), float("nan")
        return lnf.value, V.value

    def phi_osm(self, salt, m, T=298.15, P_Pa=1.0e5):
        """Osmotic coefficient, total-mole basis.  Mirrors
        scripts/total_mole_basis.py::phi_osm_total."""
        phi = ct.c_double(0.0)
        rc = self.lib.ecpa_phi_osm(self.h, _salt_code(salt), float(m),
                                   float(T), float(P_Pa),
                                   ct.byref(phi))
        if rc != 0:
            raise EcpaError(f"ecpa_phi_osm({salt}, {m}) rc={rc}")
        return phi.value

    def rho_brine(self, T_K, P_bar, mi, v_init=None):
        """CO2-free brine mass density [kg/m3].  Mirrors
        scripts/fig_brine_density.py::predict_rho_brine.  Returns
        (rho, v_state); rho is NaN and v_state None when the inner
        Newton ladder fails (the reference returns (nan, None))."""
        rho = ct.c_double(0.0)
        v_out = (ct.c_double * 3)()
        vi = _dv(v_init if v_init is not None else [-1.0, 0.0, 0.0], 3)
        rc = self.lib.ecpa_rho_brine(self.h, float(T_K), float(P_bar),
                                     _dv(mi, 6), vi, ct.byref(rho), v_out)
        if rc != 0:
            return float("nan"), None
        return rho.value, list(v_out)

    def psat_brine(self, T_K, m, p0_Pa, lo_fac=None, hi_fac=None):
        """NaCl-brine bubble pressure [bar].  Mirrors
        the water-isofugacity psat_model of the vapor-pressure scripts
        (pass lo_fac=0.55 for the vpsat_salt_2026_08 variant).  p0_Pa is
        the IAPWS pure-water saturation pressure the reference reads
        from CoolProp and uses only to anchor the Brent bracket."""
        out = ct.c_double(0.0)
        rc = self.lib.ecpa_psat_brine(
            self.h, float(T_K), float(m), float(p0_Pa),
            -1.0 if lo_fac is None else float(lo_fac),
            -1.0 if hi_fac is None else float(hi_fac), ct.byref(out))
        if rc != 0:
            return float("nan")
        return out.value

    def stability_flash(self, T, P_bar, z, mi, warm=None, want_warm_out=False,
                        salinity_convention="as-aqueous"):
        out = FlashOut()
        warm_out = Warm()
        phases = ct.c_int(0)
        route = ct.c_int(0)
        has_flash = ct.c_int(0)
        tpd_min = ct.c_double(0.0)
        rc = self.lib.ecpa_stability_flash(
            self.h, T, P_bar, _dv(z, 3), _dv(mi, 6),
            _salinity_code(salinity_convention),
            ct.byref(warm) if warm is not None else None,
            ct.byref(phases), ct.byref(tpd_min), ct.byref(route),
            ct.byref(has_flash), ct.byref(out),
            ct.byref(warm_out) if want_warm_out else None)
        d = dict(rc=rc, phases=phases.value, tpd_min=tpd_min.value,
                 route=route.value, has_flash=bool(has_flash.value),
                 flash=out.as_dict() if has_flash.value else None)
        if want_warm_out:
            d["warm_out"] = warm_out
        return d
