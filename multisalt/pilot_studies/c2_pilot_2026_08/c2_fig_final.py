"""Final C2 paper figure: (a) binary isotherms with the clean quadratic
kij(T); (b) Setschenow constants -- transfer vs two-scale calibration.

Writes c2_final_curves.csv (cache) and c2_ethane_main.png (paper figure).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c2_pilot as cp                     # noqa: E402
import c2_ions as ci                      # noqa: E402
import ms_ecpa.parameters as prm          # noqa: E402

CSV = HERE / "c2_final_curves.csv"
PNG = HERE / "c2_ethane_main.png"


def compute():
    st = json.load(open(HERE / "refit_state.json"))
    kij = st["fitQ"]["best"]["x"]
    ions = json.load(open(HERE / "c2_ions_results.json"))
    s_na, s_k = ions["s_nacl"], ions["s_k"]
    rows = []
    P_bar = np.geomspace(4.0, 690.0, 16)
    Tc1 = prm.Tc1
    for T in (310.93, 344.26, 377.59, 410.93, 444.26):
        for P in P_bar:
            kv = float(np.polyval(kij, T / Tc1))
            x = cp._predict_x2((0.0, 0.0, kv, 1.0, cp.POL_C2, T, float(P),
                                cp.MI_WATER))
            rows.append(dict(kind="binary", T_K=T, P_bar=float(P), val=x,
                             variant="fit"))
        print(f"binary T={T} done", flush=True)
    wc = {}
    Ts = np.arange(283.15, 348.16, 5.0)
    for name, sn, sk, salts in (("transfer", 1.0, 1.0, ("NaCl", "KCl")),
                                ("fitted", s_na, s_k, ("NaCl", "KCl"))):
        for salt in salts:
            for T in Ts:
                v = ci.ks(kij, sn, sk, float(T), salt, wcache=wc)
                rows.append(dict(kind="ks", T_K=float(T), P_bar=np.nan,
                                 val=v, variant=f"{salt}_{name}"))
            print(f"ks {salt} {name} done", flush=True)
    pd.DataFrame(rows).to_csv(CSV, index=False)
    print("->", CSV)


def figure():
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    mpl.rcParams.update({
        "font.size": 14, "axes.labelsize": 16,
        "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 10,
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
    LS = ["-", "--", "-.", ":", (0, (3, 1, 1, 1, 1, 1))]

    df = cp.load_water()
    cur = pd.read_csv(CSV)
    ksdf = cp.load_ks()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.4))

    Ts = [310.93, 344.26, 377.59, 410.93, 444.26]
    for i, T in enumerate(Ts):
        c = cur[(cur.kind == "binary") & np.isclose(cur.T_K, T)].copy()
        # mask the dew-line-rescue degenerate branch (returns x2 ~ 1e-6)
        c.loc[c.val < 2e-5, "val"] = np.nan
        ax1.plot(c.P_bar, c.val * 1e4, linestyle=LS[i % 5],
                 color=WONG[i % 8], label=f"{T:.0f} K")
        sub = df[np.isclose(df.T_K, T) & df.source.str.startswith("CM")]
        ax1.plot(sub.P_MPa * 10.0, sub.x_molefrac * 1e4, "o",
                 color=WONG[i % 8], mfc="white", mew=1.8, linestyle="none")
    ax1.set_xscale("log")
    ax1.set_xlabel("P [bar]")
    ax1.set_ylabel(r"$10^4\,x_{\mathrm{C_2H_6}}$")
    ax1.legend(loc="upper left")
    ax1.text(0.97, 0.03, "(a)", transform=ax1.transAxes, ha="right",
             fontsize=15, fontweight="bold")

    LABMARK = {"EuckenHertzberg": ("s", "Eucken--Hertzberg"),
               "MorrisonBillett": ("D", "Morrison--Billett"),
               "Mishnina": ("o", "Mishnina et al."),
               "BenNaimYaacobi": ("^", "Ben-Naim--Yaacobi")}
    for lab, (mark, lbl) in LABMARK.items():
        sub = ksdf[(ksdf.salt == "NaCl") & (ksdf.lab == lab)]
        if len(sub):
            ax2.plot(sub.T_K, sub.ks_dm3_mol, mark, color="black",
                     mfc="white", mew=1.6, linestyle="none", label=lbl)
    subk = ksdf[(ksdf.salt == "KCl")]
    ax2.plot(subk.T_K, subk.ks_dm3_mol, "v", color=WONG[3], mfc="white",
             mew=1.6, linestyle="none", label="KCl (exp)")
    for name, col, ls, lbl in (
            ("NaCl_transfer", WONG[1], ":", "NaCl transfer"),
            ("NaCl_fitted", WONG[5], "-", "NaCl calibrated"),
            ("KCl_transfer", WONG[6], ":", "KCl transfer"),
            ("KCl_fitted", WONG[3], "-.", "KCl calibrated")):
        c = cur[(cur.kind == "ks") & (cur.variant == name)]
        ax2.plot(c.T_K, c.val, linestyle=ls, color=col, label=lbl)
    ax2.set_xlabel("T [K]")
    ax2.set_ylabel(r"$k_S$ [dm$^3$ mol$^{-1}$]")
    ax2.legend(loc="upper right", ncol=2, fontsize=9)
    ax2.text(0.03, 0.03, "(b)", transform=ax2.transAxes, ha="left",
             fontsize=15, fontweight="bold")

    fig.tight_layout()
    fig.savefig(PNG)
    print("->", PNG)


if __name__ == "__main__":
    compute()
    figure()
