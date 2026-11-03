#!/usr/bin/env python3
"""Acceptance gate for the Fortran port: drive the ctypes wrapper
against the Python-generated golden vectors for all three parameter
files.

Run (from anywhere):
    python3 -m pytest fortran/tools/test_fortran_port.py -v

Prerequisites:
    make -C fortran                      (builds build/libecpa.*)
    python3 fortran/tools/gen_params.py  (writes params/*.par)
    python3 fortran/tools/gen_golden.py  (writes golden/*.json)

Tolerances (per the port acceptance spec):
    lnphi / inner state : 1e-12 relative
    flash converged qty : 1e-9  relative
    stability tpd_min   : 1e-6  absolute; phase decision exact
    iteration counts    : +/- 2 (documented fp-trajectory slack;
                          measured agreement is reported in the summary)
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ecpa_ctypes import Engine, EcpaError  # noqa: E402

FORTRAN = HERE.parent
CONFIGS = ["paperII_III", "paper1"]

TOL_LNPHI = 1e-12
TOL_FLASH = 1e-9
TOL_TPD = 1e-6
ITER_SLACK = 2

STATS = {}


#: NaN-vs-NaN comparisons seen this run, per (tag, field). Both sides being
#: NaN is a genuine port match, but it is also how a field that should be
#: finite hides, so it is counted and reported rather than passed in silence.
NAN_MATCHES = {}


def _dev(a, b):
    """Relative deviation with a unit floor (lnphi values are O(1)).

    One-sided NaN returns ``inf``, not ``nan``. The callers accumulate with
    ``worst = max(worst, _dev(...))``, and ``max(0.0, nan)`` is ``0.0`` in
    Python -- so a NaN deviation used to be dropped by the accumulator and
    then sail through ``assert worst < TOL`` because ``nan > TOL`` is also
    False. Every NaN, on either side, was invisible to this suite. ``inf``
    propagates through both steps and fails loudly.
    """
    a_nan, b_nan = math.isnan(a), math.isnan(b)
    if a_nan and b_nan:
        return 0.0
    if a_nan or b_nan:
        return float("inf")
    return abs(a - b) / max(1.0, abs(b))


@pytest.fixture(scope="module", params=CONFIGS)
def config(request):
    tag = request.param
    par = FORTRAN / "params" / f"{tag}.par"
    gold = FORTRAN / "golden" / f"golden_{tag}.json"
    if not par.exists():
        pytest.skip(f"missing {par}; run gen_params.py")
    if not gold.exists():
        pytest.skip(f"missing {gold}; run gen_golden.py")
    eng = Engine(parfile=par)
    data = json.loads(gold.read_text())
    yield tag, eng, data
    eng.close()


def _note(tag, key, val):
    STATS.setdefault(tag, {}).setdefault(key, 0.0)
    STATS[tag][key] = max(STATS[tag][key], val)


def _count(tag, key, inc=1):
    STATS.setdefault(tag, {}).setdefault(key, 0)
    STATS[tag][key] += inc


def test_lnphi_aq(config):
    tag, eng, data = config
    worst = 0.0
    for c in data["lnphi_aq"]:
        T, Pbar = c["T"], c["P_bar"]
        if not c["ok"]:
            with pytest.raises(EcpaError):
                eng.lnphi_aq(c["x_aq"], c["mi"], T, Pbar * 1e5)
            _count(tag, "aq_fail_matched")
            continue
        lnphi, v, _ = eng.lnphi_aq(c["x_aq"], c["mi"], T, Pbar * 1e5)
        for a, b in zip(lnphi, c["lnphi"]):
            worst = max(worst, _dev(a, b))
        if c.get("v") is not None:
            for a, b in zip(v, c["v"]):
                worst = max(worst, _dev(a, b))
        _count(tag, "aq_ok")
    _note(tag, "max_dev_lnphi_aq", worst)
    assert worst < TOL_LNPHI, f"[{tag}] lnphi_aq max relative dev {worst:.3e}"


def test_lnphi_gas(config):
    tag, eng, data = config
    worst = 0.0
    for c in data["lnphi_gas"]:
        T, Pbar = c["T"], c["P_bar"]
        if not c["ok"]:
            with pytest.raises(EcpaError):
                eng.lnphi_gas(c["y_gas"], T, Pbar * 1e5)
            _count(tag, "gas_fail_matched")
            continue
        lnphi, v = eng.lnphi_gas(c["y_gas"], T, Pbar * 1e5)
        for a, b in zip(lnphi, c["lnphi"]):
            worst = max(worst, _dev(a, b))
        if c.get("v") is not None:
            for a, b in zip(v, c["v"]):
                worst = max(worst, _dev(a, b))
        _count(tag, "gas_ok")
    _note(tag, "max_dev_lnphi_gas", worst)
    assert worst < TOL_LNPHI, f"[{tag}] lnphi_gas max relative dev {worst:.3e}"


def _compare_flash(tag, got, ref, worst):
    scalars = ["beta", "x1w", "Z_aq", "Z_gas", "eps_r",
               "chi_H2O_aq", "chi_H2O_gas"]
    vectors = [("K", "K"), ("x_aq", "x_aq"), ("y_gas", "y_gas"),
               ("fugacities_aq", "fug_aq"), ("fugacities_gas", "fug_gas")]
    for k in scalars:
        worst = max(worst, _dev(got[k], ref[k]))
    for gk, rk in vectors:
        for a, b in zip(got[gk], ref[rk]):
            worst = max(worst, _dev(a, b))
    return worst


def test_flash_kv(config):
    tag, eng, data = config
    worst = 0.0
    iter_mismatch = 0
    for c in data["flash"]:
        T, Pbar = c["T"], c["P_bar"]
        got = eng.flash_kv(T, Pbar, c["z"], c["mi"])
        if c.get("raised"):
            # Python raised out of flash_kv; Fortran must report a
            # non-clean errcode (not a quietly converged result).
            assert got["errcode"] != 0 or not got["converged"], \
                f"[{tag}] flash at T={T} P={Pbar}: Python raised, " \
                f"Fortran returned clean result"
            _count(tag, "flash_raise_matched")
            continue
        assert got["converged"] == c["converged"], \
            f"[{tag}] flash T={T} P={Pbar} z={c['z']} salt={c['salt']}: " \
            f"converged {got['converged']} vs {c['converged']}"
        if c["converged"]:
            worst = _compare_flash(tag, got, c, worst)
            if abs(got["n_iter_ssi"] - c["n_iter_ssi"]) > ITER_SLACK or \
               abs(got["n_iter_newton"] - c["n_iter_newton"]) > ITER_SLACK:
                iter_mismatch += 1
            if got["n_iter_ssi"] != c["n_iter_ssi"]:
                _count(tag, "flash_ssi_count_diff")
            _count(tag, "flash_converged")
        else:
            _count(tag, "flash_unconverged_matched")
    _note(tag, "max_dev_flash", worst)
    assert worst < TOL_FLASH, f"[{tag}] flash max relative dev {worst:.3e}"
    assert iter_mismatch == 0, \
        f"[{tag}] {iter_mismatch} cases with iteration counts off by " \
        f"more than {ITER_SLACK}"


def test_stability_flash(config):
    tag, eng, data = config
    worst = 0.0
    worst_tpd = 0.0
    for c in data["stability"]:
        T, Pbar = c["T"], c["P_bar"]
        got = eng.stability_flash(T, Pbar, c["z"], c["mi"])
        if c.get("raised"):
            assert got["rc"] != 0, \
                f"[{tag}] stability T={T} P={Pbar}: Python raised, " \
                f"Fortran rc=0"
            _count(tag, "stab_raise_matched")
            continue
        assert got["phases"] == c["phases"], \
            f"[{tag}] stability T={T} P={Pbar} z={c['z']} " \
            f"salt={c['salt']}: phases {got['phases']} vs {c['phases']} " \
            f"(route={got['route']}, py msg: {c.get('message')})"
        if math.isnan(c["tpd_min"]) or math.isnan(got["tpd_min"]):
            assert math.isnan(c["tpd_min"]) == math.isnan(got["tpd_min"])
        else:
            worst_tpd = max(worst_tpd, abs(got["tpd_min"] - c["tpd_min"]))
        if c.get("flash") is not None and got["flash"] is not None \
           and c["flash"]["converged"] and got["flash"]["converged"]:
            worst = _compare_flash(tag, got["flash"], c["flash"], worst)
        _count(tag, "stab_ok")
    _note(tag, "max_dev_stab_flash", worst)
    _note(tag, "max_dev_tpd_min", worst_tpd)
    assert worst_tpd < TOL_TPD, f"[{tag}] tpd_min max abs dev {worst_tpd:.3e}"
    assert worst < TOL_FLASH, \
        f"[{tag}] stability flash max relative dev {worst:.3e}"


def test_zz_report():
    """Prints the per-config deviation summary (run pytest with -s)."""
    print("\n=== Fortran-port verification summary ===")
    for tag, st in STATS.items():
        print(f"[{tag}]")
        for k in sorted(st):
            v = st[k]
            if isinstance(v, float):
                print(f"    {k:24s} {v:.3e}")
            else:
                print(f"    {k:24s} {v}")
