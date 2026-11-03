#!/usr/bin/env python3
"""Acceptance gate for the OPT-IN ``salinity_convention="chain"`` route
(phase 2).  Companion to test_fortran_port.py, which covers the phase-1
"as-aqueous" default.

Run:
    python3 -m pytest fortran/tools/test_fortran_chain.py -v -s

Prerequisites:
    make -C fortran
    python3 fortran/tools/gen_params.py
    python3 fortran/tools/gen_golden_chain.py

Tolerances match the phase-1 gate:
    flash converged qty : 1e-9  relative (mi_aq included)
    stability tpd_min   : 1e-6  absolute; phase decision exact
    iteration counts    : +/- 2 (measured agreement reported in the
                          summary; the slack has never been used)

Failure cases are part of the contract: where the chain closure is
invalid (sigma_x >= 1) the Python reference raises inside the SSI step
and flash_kv returns an unconverged result.  The port must agree.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ecpa_ctypes import Engine  # noqa: E402

FORTRAN = HERE.parent
CONFIGS = ["paperII_III", "paper1"]

TOL_FLASH = 1e-9
TOL_TPD = 1e-6
ITER_SLACK = 2

# Knife-edge states where the Python reference and the port take
# DIFFERENT SSI trajectories and so reach different fates.  These are
# NOT chain defects: `test_known_divergences_are_pre_existing` proves
# each one diverges identically under the default "as-aqueous"
# convention, using the as-aqueous flags recorded alongside every chain
# record in the golden file.  Entries are (tag, T, P_bar, salt).
#
# paper1 / 473.15 K / 400 bar / salt-free: the Python SSI drives
# K_H2O -> NaN and dies on the degenerate-aqueous guard at it=11; the
# port's trajectory stays finite and converges at SSI=7 + 1 Newton.
# The same bifurcation occurs under paperII_III defaults, so it is a
# property of the shared aqueous kernel, not of this parameter set --
# see PORTING_NOTES.md 12.
KNOWN_PREEXISTING_DIVERGENCES = {
    # Emptied 2026-09-19. The single entry here -- ("paper1", 473.15, 400.0,
    # "none") -- was excused as a pre-existing Python/port divergence. It is
    # gone: the aqueous cold start in _ssi_step now uses the same
    # physics-scaled Zw seed lnphi_aq uses, so both kernels start in the
    # liquid basin and agree under 'as-aqueous'. The guard test below is what
    # caught that the entry had gone stale; leave it in place.
}

STATS = {}


def _dev(a, b):
    if math.isnan(a) and math.isnan(b):
        return 0.0
    return abs(a - b) / max(1.0, abs(b))


@pytest.fixture(scope="module", params=CONFIGS)
def config(request):
    tag = request.param
    par = FORTRAN / "params" / f"{tag}.par"
    gold = FORTRAN / "golden" / f"golden_chain_{tag}.json"
    if not par.exists():
        pytest.skip(f"missing {par}; run gen_params.py")
    if not gold.exists():
        pytest.skip(f"missing {gold}; run gen_golden_chain.py")
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


def _compare_flash(got, ref, worst):
    scalars = ["beta", "x1w", "Z_aq", "Z_gas", "eps_r",
               "chi_H2O_aq", "chi_H2O_gas"]
    vectors = [("K", "K"), ("x_aq", "x_aq"), ("y_gas", "y_gas"),
               ("fugacities_aq", "fug_aq"), ("fugacities_gas", "fug_gas"),
               ("mi_aq", "mi_aq")]
    for k in scalars:
        worst = max(worst, _dev(got[k], ref[k]))
    for gk, rk in vectors:
        for a, b in zip(got[gk], ref[rk]):
            worst = max(worst, _dev(a, b))
    return worst


def test_chain_is_not_the_default(config):
    """The port must keep "as-aqueous" as the default, exactly as the
    Python reference does.  A default flip would silently invalidate
    every phase-1 golden vector, so it is pinned here."""
    tag, eng, data = config
    T, Pbar = 373.15, 200.0
    z = np.array([0.90, 0.0, 0.10])
    mi = np.array([2.0, 2.0, 0.0, 0.0, 0.0, 0.0])
    default = eng.flash_kv(T, Pbar, z, mi)
    as_aq = eng.flash_kv(T, Pbar, z, mi, salinity_convention="as-aqueous")
    chain = eng.flash_kv(T, Pbar, z, mi, salinity_convention="chain")
    assert default["mi_aq"] == as_aq["mi_aq"]
    assert default["beta"] == as_aq["beta"]
    # mi_aq under "as-aqueous" is the input mi, untouched.
    assert np.allclose(default["mi_aq"], mi, rtol=0, atol=0)
    # ...and the chain genuinely amplifies it, i.e. the flag is live.
    assert chain["converged"]
    assert chain["mi_aq"][0] > mi[0]
    _count(tag, "default_is_as_aqueous")


def test_chain_flash_kv(config):
    tag, eng, data = config
    worst = 0.0
    iter_mismatch = 0
    for c in data["flash"]:
        T, Pbar = c["T"], c["P_bar"]
        got = eng.flash_kv(T, Pbar, c["z"], c["mi"],
                           salinity_convention="chain")
        if c.get("raised"):
            assert got["errcode"] != 0 or not got["converged"], \
                f"[{tag}] chain flash T={T} P={Pbar}: Python raised, " \
                f"Fortran returned a clean result"
            _count(tag, "flash_raise_matched")
            continue
        if got["converged"] != c["converged"] and \
           (tag, T, Pbar, c["salt"]) in KNOWN_PREEXISTING_DIVERGENCES:
            _count(tag, "flash_known_divergence")
            continue
        assert got["converged"] == c["converged"], \
            f"[{tag}] chain flash T={T} P={Pbar} z={c['z']} " \
            f"salt={c['salt']}: converged {got['converged']} vs " \
            f"{c['converged']} (errcode={got['errcode']})"
        if c["converged"]:
            worst = _compare_flash(got, c, worst)
            if abs(got["n_iter_ssi"] - c["n_iter_ssi"]) > ITER_SLACK or \
               abs(got["n_iter_newton"] - c["n_iter_newton"]) > ITER_SLACK:
                iter_mismatch += 1
            if got["n_iter_ssi"] != c["n_iter_ssi"]:
                _count(tag, "flash_ssi_count_diff")
            _count(tag, "flash_converged")
        else:
            # Includes the invalid-chain-closure states; the reference
            # reports them unconverged and so must the port.
            _count(tag, "flash_unconverged_matched")
    _note(tag, "max_dev_chain_flash", worst)
    assert worst < TOL_FLASH, \
        f"[{tag}] chain flash max relative dev {worst:.3e}"
    assert iter_mismatch == 0, \
        f"[{tag}] {iter_mismatch} chain cases with iteration counts off " \
        f"by more than {ITER_SLACK}"


def test_known_divergences_are_pre_existing(config):
    """Every state on the KNOWN_PREEXISTING_DIVERGENCES allowlist must
    diverge the SAME way under the default "as-aqueous" convention.

    This is what keeps the allowlist honest: it can only ever excuse a
    trajectory bifurcation that already exists in the phase-1 default
    route, never a defect introduced by the chain closure.  A listed
    state that agrees under "as-aqueous" fails here.
    """
    tag, eng, data = config
    listed = {(t, T, P, s) for (t, T, P, s)
              in KNOWN_PREEXISTING_DIVERGENCES if t == tag}
    if not listed:
        pytest.skip(f"no known divergences listed for {tag}")
    seen = set()
    for c in data["flash"]:
        key = (tag, c["T"], c["P_bar"], c["salt"])
        if key not in listed:
            continue
        seen.add(key)
        got = eng.flash_kv(c["T"], c["P_bar"], c["z"], c["mi"])
        assert "as_aqueous_converged" in c, \
            "golden file predates the as-aqueous cross-check; " \
            "re-run gen_golden_chain.py"
        assert bool(got["converged"]) != bool(c["as_aqueous_converged"]), (
            f"[{tag}] T={c['T']} P={c['P_bar']} salt={c['salt']} is on the "
            f"KNOWN_PREEXISTING_DIVERGENCES allowlist, but Python and the "
            f"port AGREE under 'as-aqueous' "
            f"(both converged={got['converged']}).  The chain route is "
            f"therefore responsible -- investigate instead of excusing it.")
        _count(tag, "known_divergence_confirmed_pre_existing")
    assert seen == listed, \
        f"[{tag}] allowlist entries not present in the golden grid: " \
        f"{listed - seen}"


def test_chain_stability_flash(config):
    tag, eng, data = config
    worst = 0.0
    worst_tpd = 0.0
    for c in data["stability"]:
        T, Pbar = c["T"], c["P_bar"]
        got = eng.stability_flash(T, Pbar, c["z"], c["mi"],
                                  salinity_convention="chain")
        if c.get("raised"):
            assert got["rc"] != 0, \
                f"[{tag}] chain stability T={T} P={Pbar}: Python raised, " \
                f"Fortran rc=0"
            _count(tag, "stab_raise_matched")
            continue
        assert got["phases"] == c["phases"], \
            f"[{tag}] chain stability T={T} P={Pbar} z={c['z']} " \
            f"salt={c['salt']}: phases {got['phases']} vs {c['phases']} " \
            f"(route={got['route']}, py msg: {c.get('message')})"
        if math.isnan(c["tpd_min"]) or math.isnan(got["tpd_min"]):
            assert math.isnan(c["tpd_min"]) == math.isnan(got["tpd_min"])
        else:
            worst_tpd = max(worst_tpd, abs(got["tpd_min"] - c["tpd_min"]))
        if c.get("flash") is not None and got["flash"] is not None \
           and c["flash"]["converged"] and got["flash"]["converged"]:
            worst = _compare_flash(got["flash"], c["flash"], worst)
        _count(tag, "stab_ok")
    _note(tag, "max_dev_chain_stab_flash", worst)
    _note(tag, "max_dev_chain_tpd_min", worst_tpd)
    assert worst_tpd < TOL_TPD, \
        f"[{tag}] chain tpd_min max abs dev {worst_tpd:.3e}"
    assert worst < TOL_FLASH, \
        f"[{tag}] chain stability flash max relative dev {worst:.3e}"


def test_zz_report():
    print("\n=== Fortran-port CHAIN verification summary ===")
    for tag, st in STATS.items():
        print(f"[{tag}]")
        for k in sorted(st):
            v = st[k]
            if isinstance(v, float):
                print(f"    {k:26s} {v:.3e}")
            else:
                print(f"    {k:26s} {v}")
