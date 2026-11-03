"""Split the two-panel C2 figure into a main-text panel (binary
isotherms -> c2_ethane_main.png, OVERWRITES the previous two-panel
file of the same name) and an SI panel (Setschenow calibration ->
c2_setschenow_SI.png). Renders purely from the cached curves in
c2_final_curves.csv --- no model evaluation.

Run from multisalt/: PYTHONPATH=. python pilot_studies/c2_pilot_2026_08/c2_fig_split.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c2_pilot as cp                     # noqa: E402

FIG_DIR = Path(os.environ.get("ECPA_PAPER_FIGURES",
               Path(__file__).resolve().parents[2] / "figures"))
FIG_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib as mpl                  # noqa: E402
mpl.use("Agg")
import matplotlib.pyplot as plt           # noqa: E402
mpl.rcParams.update({
    "font.size": 14, "axes.labelsize": 16,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 10,
    "font.weight": "bold", "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.linewidth": 1.6, "axes.edgecolor": "black",
    "xtick.major.width": 1.6, "ytick.major.width": 1.6,
    "xtick.major.size": 6, "ytick.major.size": 6,
    "xtick.direction": "in", "ytick.direction": "in",
    "lines.linewidth": 2.2, "lines.markersize": 7,
    "legend.frameon": True, "legend.edgecolor": "black",
    "savefig.dpi": 300, "savefig.bbox": "tight",
})
WONG = ["#000000", "#E69F00", "#56B4E9", "#009E73",
        "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
LS = ["-", "--", "-.", ":", (0, (3, 1, 1, 1, 1, 1))]


def main():
    df = cp.load_water()
    cur = pd.read_csv(HERE / "c2_final_curves.csv")
    ksdf = cp.load_ks()

    # ── main-text panel: binary isotherms ────────────────────────────
    fig, ax1 = plt.subplots(figsize=(7.0, 5.2), constrained_layout=True)
    Ts = [310.93, 344.26, 377.59, 410.93, 444.26]
    for i, T in enumerate(Ts):
        c = cur[(cur.kind == "binary") & np.isclose(cur.T_K, T)].copy()
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
    fig.savefig(FIG_DIR / "c2_binary_main.png")
    plt.close(fig)
    print("->", FIG_DIR / "c2_binary_main.png")

    # ── SI panel: Setschenow calibration ─────────────────────────────
    fig, ax2 = plt.subplots(figsize=(7.0, 5.2), constrained_layout=True)
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
    fig.savefig(FIG_DIR / "c2_setschenow_SI.png")
    plt.close(fig)
    print("->", FIG_DIR / "c2_setschenow_SI.png")


if __name__ == "__main__":
    main()
