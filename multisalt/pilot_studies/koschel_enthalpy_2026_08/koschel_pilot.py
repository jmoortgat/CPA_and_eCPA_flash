"""Koschel enthalpy-of-solution comparison: thermal-property check of the
PRODUCTION multi-salt eCPA parameter set (no rebinding, no fitting).

Experimental target: Koschel et al. 2006 (FPE 247, 107-120) flow-calorimetric
enthalpies of solution of CO2 in water and 1 m / 3 m NaCl at 323.1 and
373.1 K, 2-20 MPa (koschel_data.csv; ~5% stated uncertainty). The measured
dH_sol is per mole CO2, for dissolving PURE compressed CO2 at (T, p) into
the solvent at (T, p), in the low-loading (fully dissolved) regime.

Model quantity (ideal-gas parts cancel at equal T):

    dH_sol(T, p, m) = Hres_CO2^aq,dilute - Hres_CO2^pure
                    = -R T^2 [ d ln(phi_CO2^aq)/dT - d ln(phi_CO2^pure)/dT ]_P

with the aqueous fugacity coefficient at near-infinite dilution
(x_CO2 = 1e-4; dilution-insensitivity verified: 1e-4 vs 1e-3 shifts
dH_sol by < 0.05 kJ/mol) and the pure-gas one at y = (0, 0, 1).
Temperature derivatives by centered finite difference, dT = 0.2 K
(step-insensitivity verified: 0.1 vs 0.4 K shifts < 1e-4 kJ/mol).
Everything evaluates through the public lnphi_aq / lnphi_gas kernels, so
the per-phase inner Newton (complex-step Jacobian) closes the triple at
every temperature evaluation.

This probes the TEMPERATURE DERIVATIVE of the CO2-water and CO2-ion
interactions -- information the isothermal solubility fits constrain only
indirectly -- using zero adjusted parameters.

Stages:
    python koschel_pilot.py --stage points   # model dH at the 22 states
    python koschel_pilot.py --stage curves   # smooth P-curves per (T, m)
    python koschel_pilot.py --stage figure   # 2-panel house-style figure
    python koschel_pilot.py --stage all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Resolve against THIS repository. This was an absolute path into the
# pre-restructure tree, and since it was inserted at the FRONT of
# sys.path it would import that tree's package rather than this one --
# producing a figure from the old code while appearing to run here.
CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))

from ecpa_ms import lnphi_aq, lnphi_gas     # noqa: E402

HERE = Path(__file__).resolve().parent
CSV = HERE / "koschel_data.csv"
RESULTS_JSON = HERE / "koschel_results.json"
CURVES_CSV = HERE / "curves_cache.csv"
FIG_PNG = HERE / "fig_koschel_enthalpy.png"

R_J = 8.31446261815324
XC_DILUTE = 1e-4
DT_FD = 0.2


def dh_sol(T, P_MPa, m_NaCl, xc=XC_DILUTE, dT=DT_FD):
    """Model enthalpy of solution [kJ/mol CO2] at (T [K], P [MPa], m)."""
    P = P_MPa * 1e6
    mi = np.array([m_NaCl, m_NaCl, 0.0, 0.0, 0.0, 0.0])
    x = np.array([1.0 - xc, 0.0, xc])
    y = np.array([0.0, 0.0, 1.0])
    try:
        d_aq = (lnphi_aq(x, mi, T + dT, P)[2]
                - lnphi_aq(x, mi, T - dT, P)[2]) / (2 * dT)
        d_g = (lnphi_gas(y, T + dT, P)[2]
               - lnphi_gas(y, T - dT, P)[2]) / (2 * dT)
    except Exception:
        return float("nan")
    return float(-R_J * T * T * (d_aq - d_g) / 1000.0)


def load_data():
    import pandas as pd
    df = pd.read_csv(CSV, comment="#")
    assert len(df) == 22, f"expected 22 rows, got {len(df)}"
    return df


def stage_points(st):
    df = load_data()
    rows = []
    for r in df.itertuples():
        v = dh_sol(r.T_K, r.P_MPa, r.m_NaCl)
        rows.append({"m": r.m_NaCl, "T_K": r.T_K, "P_MPa": r.P_MPa,
                     "dH_exp": r.dH_kJ_mol, "delta_exp": r.delta_kJ_mol,
                     "dH_model": v, "dev_kJ": v - r.dH_kJ_mol})
        print(f"  m={r.m_NaCl} T={r.T_K} P={r.P_MPa:6.2f}: model "
              f"{v:7.2f}  exp {r.dH_kJ_mol:7.2f}  dev {v-r.dH_kJ_mol:+5.2f}",
              flush=True)
    devs = np.array([x["dev_kJ"] for x in rows])
    st["points"] = rows
    st["summary"] = {
        "mad_kJ_mol": float(np.mean(np.abs(devs))),
        "max_abs_dev_kJ_mol": float(np.max(np.abs(devs))),
        "bias_kJ_mol": float(np.mean(devs)),
        "mad_323K": float(np.mean(np.abs(devs[np.array(
            [x["T_K"] < 350 for x in rows])]))),
        "mad_373K": float(np.mean(np.abs(devs[np.array(
            [x["T_K"] > 350 for x in rows])]))),
    }
    print(json.dumps(st["summary"], indent=2))


def stage_curves(st):
    import pandas as pd
    rows = []
    P_grid = np.linspace(1.5, 21.0, 27)
    for T in (323.1, 373.1):
        for m in (0.0, 1.0, 3.0):
            for P in P_grid:
                rows.append(dict(T_K=T, m=m, P_MPa=float(P),
                                 dH=dh_sol(T, float(P), m)))
            print(f"  curve T={T} m={m} done", flush=True)
    pd.DataFrame(rows).to_csv(CURVES_CSV, index=False)
    st["curves_csv"] = str(CURVES_CSV)


def stage_figure(st):
    import pandas as pd
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "font.size": 14, "axes.labelsize": 16,
        "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12,
        "font.weight": "bold", "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.linewidth": 1.6, "axes.edgecolor": "black",
        "xtick.major.width": 1.6, "ytick.major.width": 1.6,
        "xtick.major.size": 6, "ytick.major.size": 6,
        "xtick.color": "black", "ytick.color": "black",
        "xtick.direction": "in", "ytick.direction": "in",
        "lines.linewidth": 2.2, "lines.markersize": 7,
        "legend.frameon": True, "legend.edgecolor": "black",
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })
    WONG = ["#000000", "#E69F00", "#56B4E9", "#009E73",
            "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
    LS = {0.0: "-", 1.0: "--", 3.0: "-."}
    COL = {0.0: WONG[0], 1.0: WONG[5], 3.0: WONG[6]}
    MARK = {0.0: "o", 1.0: "s", 3.0: "^"}
    LBL = {0.0: "H$_2$O", 1.0: "1 m NaCl", 3.0: "3 m NaCl"}

    df = load_data()
    cur = pd.read_csv(CURVES_CSV)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharey=True)
    for ax, T, tag in zip(axes, (323.1, 373.1), ("(a)", "(b)")):
        for m in (0.0, 1.0, 3.0):
            c = cur[(np.isclose(cur.T_K, T)) & (np.isclose(cur.m, m))]
            ax.plot(c.P_MPa, c.dH, LS[m], color=COL[m], label=LBL[m])
            sub = df[(np.isclose(df.T_K, T)) & (np.isclose(df.m_NaCl, m))]
            ax.errorbar(sub.P_MPa, sub.dH_kJ_mol, yerr=sub.delta_kJ_mol,
                        fmt=MARK[m], color=COL[m], mfc="white", mew=1.8,
                        capsize=4, elinewidth=1.6, linestyle="none")
        ax.set_xlabel("P [MPa]")
        ax.text(0.04, 0.05, f"{tag} {T:.1f} K", transform=ax.transAxes,
                ha="left", fontsize=14, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.85,
                          edgecolor="none", pad=2))
    axes[0].set_ylabel(r"$\Delta H_{\rm sol}$ [kJ mol$^{-1}$ CO$_2$]")
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_PNG)
    print("figure ->", FIG_PNG)
    st["figure"] = str(FIG_PNG.resolve().relative_to(Path(__file__).resolve().parents[3]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["points", "curves", "figure", "all"])
    args = ap.parse_args()
    stages = (["points", "curves", "figure"] if args.stage == "all"
              else [args.stage])
    st = {}
    if RESULTS_JSON.exists():
        st = json.loads(RESULTS_JSON.read_text())
    for name in stages:
        t0 = time.perf_counter()
        print(f"=== stage {name} ===", flush=True)
        {"points": stage_points, "curves": stage_curves,
         "figure": stage_figure}[name](st)
        st["method"] = {
            "quantity": "dH_sol = Hres_CO2(aq, dilute) - Hres_CO2(pure gas)",
            "xc_dilute": XC_DILUTE, "dT_FD_K": DT_FD,
            "kernels": "public lnphi_aq / lnphi_gas, production parameters, "
                       "zero adjusted or rebound parameters",
        }
        RESULTS_JSON.write_text(json.dumps(st, indent=2))
        print(f"=== stage {name} done ({time.perf_counter()-t0:.1f}s) ===",
              flush=True)


if __name__ == "__main__":
    main()
