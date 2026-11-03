#!/usr/bin/env python3
"""Generate golden verification vectors from the Python reference
implementation (ms_ecpa) for the Fortran port acceptance gate.

For EACH of the parameter files (paperII_III, paper1):

  (a) lnphi block: lnphi_aq / lnphi_gas component vectors plus the
      converged inner state (Z, eps_r, chi) at ~200 diverse states
      (T 278-573 K, P 1-1000 bar, all five single salts + mixed
      brines to ~6 mol/kg total ion, aqueous feeds spanning CO2/CH4;
      gas compositions across the H2O/CH4/CO2 triangle).
  (b) flash block: full flash_kv results at ~100 states.
  (c) stability block: stability_flash phase decisions at ~50 states
      bracketing phase boundaries.

Paper-1 scope note: paper1_config parameterizes only CO2-H2O-NaCl, so
its state grids drop CH4 and non-NaCl salts (fewer states by design).

Output: fortran/golden/golden_<tag>.json

Usage: python3 gen_golden.py [outdir]   (runs the Python library; the
Fortran side is exercised by tools/test_fortran_port.py)
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

import ms_ecpa.parameters as P                      # noqa: E402
from ms_ecpa import flash_kv, stability_flash       # noqa: E402
from ms_ecpa.lnphi import (                         # noqa: E402
    lnphi_aq, lnphi_gas, _aq_residual_3var, _gas_residual,
    _gas_accept_or_repair)
from ms_ecpa.newton import newton_solve             # noqa: E402
from ms_ecpa import paper1_config                   # noqa: E402
import gen_params                                    # noqa: E402


# ---------------------------------------------------------------------
# State grids (deterministic; no RNG)
# ---------------------------------------------------------------------

SALTS_FULL = {
    "none":    [0, 0, 0, 0, 0, 0],
    "NaCl1":   [1, 1, 0, 0, 0, 0],
    "NaCl3":   [3, 3, 0, 0, 0, 0],
    "NaCl6":   [6, 6, 0, 0, 0, 0],
    "KCl2":    [0, 2, 2, 0, 0, 0],
    "CaCl2_2": [0, 4, 0, 2, 0, 0],
    "MgCl2_1": [0, 2, 0, 0, 1, 0],
    "Na2SO4_1": [2, 0, 0, 0, 0, 1],
    "mix_low":  [1.0, 1.5, 0.2, 0.1, 0.05, 0.05],
    "seawater": [0.485, 0.566, 0.011, 0.011, 0.055, 0.029],
}
SALTS_NACL = {k: v for k, v in SALTS_FULL.items()
              if k in ("none", "NaCl1", "NaCl3", "NaCl6")}

T_LNPHI = [278.15, 313.15, 373.15, 433.15, 503.15, 573.15]
P_LNPHI = [1.0, 10.0, 50.0, 100.0, 300.0, 600.0, 1000.0]

X_AQ_FULL = [
    [0.995, 0.0, 0.005],
    [0.97, 0.0, 0.03],
    [0.985, 0.015, 0.0],
    [0.96, 0.02, 0.02],
    [0.90, 0.04, 0.06],
]
X_AQ_NACL = [
    [0.995, 0.0, 0.005],
    [0.98, 0.0, 0.02],
    [0.95, 0.0, 0.05],
]
Y_GAS_FULL = [
    [0.01, 0.0, 0.99],
    [0.02, 0.98, 0.0],
    [0.05, 0.475, 0.475],
    [0.30, 0.35, 0.35],
    [0.001, 0.2, 0.799],
]
Y_GAS_NACL = [
    [0.01, 0.0, 0.99],
    [0.05, 0.0, 0.95],
    [0.30, 0.0, 0.70],
]

T_FLASH = [285.15, 313.15, 353.15, 423.15, 473.15, 523.15]
P_FLASH = [10.0, 50.0, 150.0, 400.0, 800.0]
Z_FLASH_FULL = [
    [0.90, 0.0, 0.10],
    [0.85, 0.15, 0.0],
    [0.80, 0.10, 0.10],
    [0.60, 0.20, 0.20],
]
Z_FLASH_NACL = [
    [0.90, 0.0, 0.10],
    [0.75, 0.0, 0.25],
]

# stability states: mix of clearly 1-phase, clearly 2-phase and
# near-boundary (small gas feed at high P dissolves fully; the same
# feed at low P splits).
STAB_CASES_FULL = [
    # (T, P, z, salt)
    (298.15, 1.0,  [0.999, 0.0, 0.001], "none"),
    (298.15, 1.0,  [0.99, 0.0, 0.01], "NaCl1"),
    (298.15, 50.0, [0.995, 0.0, 0.005], "none"),
    (298.15, 50.0, [0.98, 0.0, 0.02], "none"),
    (298.15, 50.0, [0.96, 0.0, 0.04], "NaCl3"),
    (313.15, 100.0, [0.99, 0.0, 0.01], "none"),
    (313.15, 100.0, [0.97, 0.0, 0.03], "NaCl1"),
    (313.15, 100.0, [0.95, 0.05, 0.0], "none"),
    (313.15, 100.0, [0.999, 0.001, 0.0], "none"),
    (353.15, 200.0, [0.98, 0.01, 0.01], "mix_low"),
    (353.15, 200.0, [0.9995, 0.0, 0.0005], "none"),
    (373.15, 20.0, [0.99, 0.0, 0.01], "none"),
    (373.15, 20.0, [0.9, 0.05, 0.05], "seawater"),
    (423.15, 150.0, [0.97, 0.015, 0.015], "NaCl1"),
    (423.15, 150.0, [0.998, 0.001, 0.001], "none"),
    (473.15, 300.0, [0.95, 0.025, 0.025], "NaCl3"),
    (473.15, 300.0, [0.99, 0.005, 0.005], "none"),
    (523.15, 500.0, [0.9, 0.05, 0.05], "NaCl1"),
    (523.15, 100.0, [0.97, 0.0, 0.03], "none"),
    (573.15, 800.0, [0.9, 0.05, 0.05], "none"),
    (285.15, 30.0, [0.95, 0.05, 0.0], "KCl2"),
    (285.15, 30.0, [0.95, 0.0, 0.05], "CaCl2_2"),
    (313.15, 600.0, [0.995, 0.0, 0.005], "MgCl2_1"),
    (313.15, 600.0, [0.99, 0.01, 0.0], "Na2SO4_1"),
    (448.15, 120.0, [0.60, 0.0, 0.40], "none"),
]
STAB_CASES_NACL = [
    (298.15, 1.0,  [0.999, 0.0, 0.001], "none"),
    (298.15, 50.0, [0.98, 0.0, 0.02], "none"),
    (298.15, 50.0, [0.96, 0.0, 0.04], "NaCl3"),
    (313.15, 100.0, [0.97, 0.0, 0.03], "NaCl1"),
    (313.15, 100.0, [0.9995, 0.0, 0.0005], "none"),
    (373.15, 20.0, [0.99, 0.0, 0.01], "NaCl6"),
    (423.15, 150.0, [0.97, 0.0, 0.03], "NaCl1"),
    (473.15, 300.0, [0.99, 0.0, 0.01], "none"),
    (523.15, 100.0, [0.97, 0.0, 0.03], "NaCl1"),
    (285.15, 60.0, [0.95, 0.0, 0.05], "NaCl3"),
]


def _aq_inner_state(x_aq, mi, T, P_Pa):
    """Converged (Zw, eps_r, chi1w): the exact seed discipline of
    lnphi_aq (default physics seed, then its fallback ladder)."""
    Zw_liq = float(P_Pa) * 18.015e-6 / (8.314 * float(T))
    seeds = [np.array([Zw_liq, 50.0, 0.10]),
             np.array([Zw_liq, 60.0, 0.07]),
             np.array([Zw_liq, 50.0, 0.80]),
             np.array([Zw_liq, 50.0, 0.90]),
             np.array([0.07, 60.0, 0.13]),
             np.array([0.10, 30.0, 0.05]),
             np.array([0.30, 70.0, 0.20]),
             np.array([0.20, 40.0, 0.10]),
             np.array([0.50, 50.0, 0.10]),
             np.array([0.50, 80.0, 0.50]),
             np.array([Zw_liq, 78.0, 0.85]),
             np.array([Zw_liq, 78.0, 0.95]),
             np.array([Zw_liq, 90.0, 0.07]),
             np.array([Zw_liq, 100.0, 0.07]),
             np.array([Zw_liq, 110.0, 0.07]),
             np.array([Zw_liq, 90.0, 0.85]),
             np.array([Zw_liq, 100.0, 0.90]),
             np.array([Zw_liq, 80.0, 0.97]),
             np.array([Zw_liq, 85.0, 0.95]),
             np.array([Zw_liq, 30.0, 0.10]),
             np.array([Zw_liq, 40.0, 0.20]),
             np.array([Zw_liq, 50.0, 0.30]),
             np.array([Zw_liq, 75.0, 0.98]),
             np.array([Zw_liq, 70.0, 0.99]),
             np.array([Zw_liq, 70.0, 0.93]),
             np.array([Zw_liq, 65.0, 0.95]),
             np.array([Zw_liq, 5.0, 0.05]),
             np.array([Zw_liq, 10.0, 0.05]),
             np.array([Zw_liq, 15.0, 0.10]),
             np.array([Zw_liq, 25.0, 0.12]),
             np.array([1.4 * Zw_liq, 42.0, 0.09])]
    from ms_ecpa import extensions as _ext
    seeds += [np.asarray(v, float) for v in _ext.extra_aqueous_seeds(Zw_liq)]
    for s in seeds:
        v, ok, nit = newton_solve(_aq_residual_3var, s,
                                  args=(np.asarray(x_aq), np.asarray(mi),
                                        T, P_Pa), tol=1e-10, maxiter=80)
        if ok and v[2] > 0:
            return [float(x) for x in v]
    return None


def _gas_inner_state(y_gas, T, P_Pa):
    from ms_ecpa.parameters import b1, b2, b3, R
    y1, y2, y3 = [float(v) for v in y_gas]
    b_mix = b1 * y1 + b2 * y2 + b3 * y3
    Zh_B = b_mix * float(P_Pa) / (R * float(T)) * 1.2 + 0.3
    Zh_liq = b_mix * float(P_Pa) / (R * float(T)) * 2.0
    for z0 in (1.0, max(Zh_B, 1.5), max(Zh_B * 1.3, 2.0), 3.0, 0.7, 0.5,
               0.3, Zh_liq, b_mix * float(P_Pa) / (R * float(T)) * 1.5):
        v, ok, nit = newton_solve(_gas_residual, np.array([z0, 0.99]),
                                  args=(np.asarray(y_gas), T, P_Pa),
                                  tol=1e-10, maxiter=80)
        # Same physical guard lnphi_gas applies (and the twin of the
        # `v[2] > 0` test in _aq_inner_state above): without it this
        # recorded the NON-PHYSICAL chi<=0 root while the `lnphi` field
        # of the same record came from the guarded lnphi_gas, so the two
        # halves of the golden case described different states.
        v, ok = _gas_accept_or_repair(v, ok, np.asarray(y_gas), T, P_Pa,
                                      1e-10, 80)
        if ok:
            return [float(x) for x in v]
    return None


def gen_lnphi_block(salts, x_list, y_list):
    cases_aq = []
    cases_gas = []
    salt_names = list(salts.keys())
    i = 0
    for T in T_LNPHI:
        for Pbar in P_LNPHI:
            P_Pa = Pbar * 1e5
            # cycle salt and composition to keep the count ~200
            sname = salt_names[i % len(salt_names)]
            x_aq = x_list[i % len(x_list)]
            mi = salts[sname]
            rec = dict(T=T, P_bar=Pbar, x_aq=x_aq, mi=mi, salt=sname)
            try:
                la = lnphi_aq(np.asarray(x_aq), np.asarray(mi), T, P_Pa)
                rec["ok"] = True
                rec["lnphi"] = [float(v) for v in la]
                rec["v"] = _aq_inner_state(x_aq, mi, T, P_Pa)
            except Exception as exc:
                rec["ok"] = False
                rec["error"] = str(exc)[:120]
            cases_aq.append(rec)

            for jx in (1, 2):
                sname2 = salt_names[(i + 3 * jx) % len(salt_names)]
                x_aq2 = x_list[(i + jx) % len(x_list)]
                rec = dict(T=T, P_bar=Pbar, x_aq=x_aq2, mi=salts[sname2],
                           salt=sname2)
                try:
                    la = lnphi_aq(np.asarray(x_aq2),
                                  np.asarray(salts[sname2]), T, P_Pa)
                    rec["ok"] = True
                    rec["lnphi"] = [float(v) for v in la]
                    rec["v"] = _aq_inner_state(x_aq2, salts[sname2], T, P_Pa)
                except Exception as exc:
                    rec["ok"] = False
                    rec["error"] = str(exc)[:120]
                cases_aq.append(rec)

            for jy in range(2):
                y = y_list[(i + jy) % len(y_list)]
                rec = dict(T=T, P_bar=Pbar, y_gas=y)
                try:
                    lg = lnphi_gas(np.asarray(y), T, P_Pa)
                    rec["ok"] = True
                    rec["lnphi"] = [float(v) for v in lg]
                    rec["v"] = _gas_inner_state(y, T, P_Pa)
                except Exception as exc:
                    rec["ok"] = False
                    rec["error"] = str(exc)[:120]
                cases_gas.append(rec)
            i += 1
    return cases_aq, cases_gas


def flash_to_rec(fr):
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
    )


def gen_flash_block(salts, z_list):
    cases = []
    salt_names = list(salts.keys())
    i = 0
    for T in T_FLASH:
        for Pbar in P_FLASH:
            for k in range(3 if len(z_list) > 2 else 2):
                z = z_list[(i + k) % len(z_list)]
                sname = salt_names[(i + k) % len(salt_names)]
                mi = salts[sname]
                rec = dict(T=T, P_bar=Pbar, z=z, mi=mi, salt=sname)
                try:
                    fr = flash_kv(T, Pbar, np.asarray(z), np.asarray(mi))
                    rec["raised"] = False
                    rec.update(flash_to_rec(fr))
                except Exception as exc:
                    rec["raised"] = True
                    rec["error"] = str(exc)[:120]
                cases.append(rec)
            i += 1
    return cases


def gen_stab_block(salts, stab_cases):
    cases = []
    # each listed case runs at its own P and at 2.5x P (extra bracketing
    # of the phase boundary at deterministic states)
    expanded = list(stab_cases) + [(T, Pbar * 2.5, z, s)
                                   for (T, Pbar, z, s) in stab_cases]
    for (T, Pbar, z, sname) in expanded:
        mi = salts.get(sname, SALTS_FULL[sname])
        rec = dict(T=T, P_bar=Pbar, z=z, mi=mi, salt=sname)
        try:
            r = stability_flash(T, Pbar, np.asarray(z), np.asarray(mi))
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


def gen_config(tag, salts, x_list, y_list, z_list, stab_cases):
    t0 = time.time()
    aq, gas = gen_lnphi_block(salts, x_list, y_list)
    t1 = time.time()
    fl = gen_flash_block(salts, z_list)
    t2 = time.time()
    st = gen_stab_block(salts, stab_cases)
    t3 = time.time()
    n_ok_aq = sum(1 for c in aq if c.get("ok"))
    n_ok_gas = sum(1 for c in gas if c.get("ok"))
    n_conv_fl = sum(1 for c in fl if not c.get("raised") and c.get("converged"))
    print(f"[{tag}] lnphi_aq {n_ok_aq}/{len(aq)} ok ({t1-t0:.1f}s), "
          f"lnphi_gas {n_ok_gas}/{len(gas)} ok, "
          f"flash {n_conv_fl}/{len(fl)} converged ({t2-t1:.1f}s), "
          f"stability {len(st)} cases ({t3-t2:.1f}s)")
    return dict(tag=tag, lnphi_aq=aq, lnphi_gas=gas, flash=fl, stability=st)


def main():
    outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "golden"
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) pristine defaults
    blk = gen_config("paperII_III", SALTS_FULL, X_AQ_FULL, Y_GAS_FULL,
                     Z_FLASH_FULL, STAB_CASES_FULL)
    (outdir / "golden_paperII_III.json").write_text(json.dumps(blk))

    # 2) paper 1 (CO2 + NaCl scope)
    paper1_config.apply_paper1()
    try:
        blk = gen_config("paper1", SALTS_NACL, X_AQ_NACL, Y_GAS_NACL,
                         Z_FLASH_NACL, STAB_CASES_NACL)
        (outdir / "golden_paper1.json").write_text(json.dumps(blk))
    finally:
        paper1_config.restore_paper1()




if __name__ == "__main__":
    main()
