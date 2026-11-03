#!/usr/bin/env python3
"""Golden vectors for the OPT-IN ``salinity_convention="chain"`` route
(phase 2 of the Fortran port).

This is a SEPARATE generator so the phase-1 golden files
(golden_<tag>.json) are never regenerated.  It writes
``fortran/golden/golden_chain_<tag>.json``.

What it certifies
-----------------
``ms_ecpa.flash.flash_kv`` and ``ms_ecpa.stability.stability_flash``
called with ``salinity_convention="chain"``, i.e. the salt-conservation
closure ``m_aq* = m_prep * z_w / [(1-beta) * x_w_aq]`` implemented in
``ms_ecpa.flash.compute_aqueous_x1w_chain``.  The record carries
``mi_aq`` (the chain's distinguishing output) in addition to every
quantity the phase-1 flash block compares.

The state grid deliberately includes states where the chain closure is
INVALID (``sigma_x >= 1``: the aqueous phase cannot hold the salt mass
the feed/phase-split combination implies).  Python raises a RuntimeError
inside the SSI step there and ``flash_kv`` returns an unconverged
result; the Fortran port must agree, so those states are part of the
gate rather than being filtered out.

Usage: python3 gen_golden_chain.py [outdir]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(HERE))

from ms_ecpa import flash_kv, stability_flash        # noqa: E402
from ms_ecpa import paper1_config                    # noqa: E402
import gen_params                                     # noqa: E402
from gen_golden import SALTS_FULL                     # noqa: E402


# ---------------------------------------------------------------------
# State grid.  Chosen so that sigma_x = Mw*z_w*m_tot/(1-beta) spans the
# comfortable regime (<0.2), the tight regime (0.5-0.9) and the invalid
# regime (>=1), and so that both single and mixed brines appear.
# ---------------------------------------------------------------------

T_CHAIN = [285.15, 313.15, 373.15, 423.15, 473.15, 523.15]
P_CHAIN = [10.0, 50.0, 150.0, 400.0, 800.0]

Z_CHAIN_FULL = [
    [0.95, 0.0, 0.05],
    [0.90, 0.0, 0.10],
    [0.85, 0.15, 0.0],
    [0.80, 0.10, 0.10],
    [0.60, 0.20, 0.20],
    [0.40, 0.30, 0.30],    # gas-rich: large beta -> chain amplification
]
Z_CHAIN_NACL = [
    [0.95, 0.0, 0.05],
    [0.90, 0.0, 0.10],
    [0.75, 0.0, 0.25],
]

SALTS_CHAIN_FULL = ["none", "NaCl1", "NaCl3", "NaCl6", "KCl2", "CaCl2_2",
                    "MgCl2_1", "Na2SO4_1", "mix_low", "seawater"]
SALTS_CHAIN_NACL = ["none", "NaCl1", "NaCl3", "NaCl6"]

STAB_CASES_FULL = [
    (298.15, 50.0,  [0.98, 0.0, 0.02], "NaCl1"),
    (298.15, 50.0,  [0.96, 0.0, 0.04], "NaCl3"),
    (313.15, 100.0, [0.97, 0.0, 0.03], "seawater"),
    (313.15, 100.0, [0.95, 0.05, 0.0], "NaCl1"),
    (353.15, 200.0, [0.98, 0.01, 0.01], "mix_low"),
    (373.15, 20.0,  [0.99, 0.0, 0.01], "NaCl6"),
    (373.15, 20.0,  [0.90, 0.05, 0.05], "seawater"),
    (423.15, 150.0, [0.97, 0.015, 0.015], "NaCl1"),
    (473.15, 300.0, [0.95, 0.025, 0.025], "NaCl3"),
    (523.15, 500.0, [0.90, 0.05, 0.05], "NaCl1"),
    (285.15, 30.0,  [0.95, 0.05, 0.0], "KCl2"),
    (285.15, 30.0,  [0.95, 0.0, 0.05], "CaCl2_2"),
    (313.15, 600.0, [0.995, 0.0, 0.005], "MgCl2_1"),
    (313.15, 600.0, [0.99, 0.01, 0.0], "Na2SO4_1"),
    (448.15, 120.0, [0.60, 0.0, 0.40], "NaCl1"),
    (373.15, 60.0,  [0.50, 0.0, 0.50], "NaCl3"),
]
STAB_CASES_NACL = [
    (298.15, 50.0,  [0.98, 0.0, 0.02], "NaCl1"),
    (298.15, 50.0,  [0.96, 0.0, 0.04], "NaCl3"),
    (313.15, 100.0, [0.97, 0.0, 0.03], "NaCl1"),
    (373.15, 20.0,  [0.99, 0.0, 0.01], "NaCl6"),
    (423.15, 150.0, [0.97, 0.0, 0.03], "NaCl1"),
    (523.15, 100.0, [0.97, 0.0, 0.03], "NaCl1"),
]


def flash_to_rec(fr):
    """Same fields as gen_golden.flash_to_rec PLUS mi_aq (the chain's
    distinguishing output) and the reported convention."""
    return dict(
        converged=bool(fr.converged),
        K=[float(v) for v in fr.K],
        x_aq=[float(v) for v in fr.x_aq],
        y_gas=[float(v) for v in fr.y_gas],
        beta=float(fr.beta), x1w=float(fr.x1w),
        Z_aq=float(fr.Z_aq), Z_gas=float(fr.Z_gas), eps_r=float(fr.eps_r),
        chi_H2O_aq=float(fr.chi_H2O_aq), chi_H2O_gas=float(fr.chi_H2O_gas),
        n_iter_ssi=int(fr.n_iter_ssi), n_iter_newton=int(fr.n_iter_newton),
        max_residual=float(fr.max_residual),
        fug_aq=[float(v) for v in fr.fugacities_aq],
        fug_gas=[float(v) for v in fr.fugacities_gas],
        mi_aq=[float(v) for v in fr.mi_aq],
        salinity_convention=str(fr.salinity_convention),
    )


def gen_flash_block(salt_names, z_list):
    cases = []
    i = 0
    for T in T_CHAIN:
        for Pbar in P_CHAIN:
            for k in range(3 if len(z_list) > 2 else 2):
                z = z_list[(i + k) % len(z_list)]
                sname = salt_names[(i + k) % len(salt_names)]
                mi = SALTS_FULL[sname]
                rec = dict(T=T, P_bar=Pbar, z=z, mi=mi, salt=sname)
                try:
                    fr = flash_kv(T, Pbar, np.asarray(z), np.asarray(mi),
                                  salinity_convention="chain")
                    rec["raised"] = False
                    rec.update(flash_to_rec(fr))
                except Exception as exc:
                    rec["raised"] = True
                    rec["error"] = str(exc)[:160]
                # Same state under the DEFAULT convention.  Recorded so
                # the gate can prove that any convergence-fate mismatch
                # it tolerates is pre-existing (present under
                # "as-aqueous" too) rather than a chain defect.
                try:
                    fr0 = flash_kv(T, Pbar, np.asarray(z), np.asarray(mi))
                    rec["as_aqueous_raised"] = False
                    rec["as_aqueous_converged"] = bool(fr0.converged)
                except Exception:
                    rec["as_aqueous_raised"] = True
                    rec["as_aqueous_converged"] = False
                cases.append(rec)
            i += 1
    return cases


def gen_stab_block(stab_cases):
    cases = []
    expanded = list(stab_cases) + [(T, Pbar * 2.5, z, s)
                                   for (T, Pbar, z, s) in stab_cases]
    for (T, Pbar, z, sname) in expanded:
        mi = SALTS_FULL[sname]
        rec = dict(T=T, P_bar=Pbar, z=z, mi=mi, salt=sname)
        try:
            r = stability_flash(T, Pbar, np.asarray(z), np.asarray(mi),
                                salinity_convention="chain")
            rec["raised"] = False
            rec["phases"] = int(r.phases)
            rec["tpd_min"] = float(r.tpd_min)
            rec["stable"] = bool(r.stable)
            rec["message"] = r.message[:80]
            if r.flash_result is not None:
                rec["flash"] = flash_to_rec(r.flash_result)
        except Exception as exc:
            rec["raised"] = True
            rec["error"] = str(exc)[:160]
        cases.append(rec)
    return cases


def gen_config(tag, salt_names, z_list, stab_cases):
    t0 = time.time()
    fl = gen_flash_block(salt_names, z_list)
    t1 = time.time()
    st = gen_stab_block(stab_cases)
    t2 = time.time()
    n_conv = sum(1 for c in fl if not c.get("raised") and c.get("converged"))
    n_inval = sum(1 for c in fl if not c.get("raised")
                  and not c.get("converged"))
    print(f"[{tag}] chain flash {n_conv}/{len(fl)} converged "
          f"({n_inval} unconverged/invalid-closure) ({t1-t0:.1f}s), "
          f"stability {len(st)} cases ({t2-t1:.1f}s)")
    return dict(tag=tag, salinity_convention="chain", flash=fl, stability=st)


def main():
    outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "golden"
    outdir.mkdir(parents=True, exist_ok=True)

    blk = gen_config("paperII_III", SALTS_CHAIN_FULL, Z_CHAIN_FULL,
                     STAB_CASES_FULL)
    (outdir / "golden_chain_paperII_III.json").write_text(json.dumps(blk))

    paper1_config.apply_paper1()
    try:
        blk = gen_config("paper1", SALTS_CHAIN_NACL, Z_CHAIN_NACL,
                         STAB_CASES_NACL)
        (outdir / "golden_chain_paper1.json").write_text(json.dumps(blk))
    finally:
        paper1_config.restore_paper1()




if __name__ == "__main__":
    main()
