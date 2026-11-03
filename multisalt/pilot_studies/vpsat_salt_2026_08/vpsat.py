"""Vapor pressure of water over NaCl brine vs T and molality.

P_sat(T, m) from the isofugacity condition for water between the
brine (aqueous kernel with ion molalities) and pure steam (gas
kernel):  x_w phi_w^aq(T,P) = phi_w^gas(T,P),  solved by Brent on P.
Reference: IAPWS (CoolProp) pure-water saturation line and the
Pitzer water activities at 298.15 K (a_w = exp(-2 m Mw phi_osm)).

Run from multisalt/: PYTHONPATH=. python pilot_studies/vpsat_salt_2026_08/vpsat.py [--recompute]
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy.optimize import brentq

from ecpa_ms.lnphi import lnphi_aq, lnphi_gas
from ecpa_ms.parameters import Mw

HERE = Path(__file__).resolve().parent
FIG = Path(os.environ.get("ECPA_PAPER_FIGURES",
           Path(__file__).resolve().parents[2] / "figures")) / "vpsat_salt_SI.png"
FIG.parent.mkdir(parents=True, exist_ok=True)

MOLAL = (0.0, 1.0, 3.0, 6.0)
# 298.15 K is inserted explicitly: the 10 K grid from 283.15 steps
# 283.15, 293.15, 303.15 and skips it, so the one temperature where an
# independent Pitzer activity-coefficient value exists for every molality
# was never a plotted point -- the printed comparison against it was
# interpolated. It is the anchor of this figure, so it is on the grid.
T_GRID = np.unique(np.concatenate([np.arange(283.15, 574.0, 10.0),
                                   [298.15]]))



def psat_model(T, m):
    """Brine bubble pressure from water isofugacity, bracketed around
    the IAPWS pure-water saturation pressure (the vapor-pressure
    lowering never exceeds ~30% at 6 mol/kg, and the gas kernel needs
    a vapor root, which exists only near/below saturation)."""
    from CoolProp.CoolProp import PropsSI
    mi = np.array([m, m, 0, 0, 0, 0], dtype=float)
    m_tot = 2.0 * m
    x_w = 1.0 / (1.0 + Mw * m_tot)   # true water fraction; the kernel
    x_aq = (x_w, 0.0, 0.0)           # reconstructs ions from mi
    y = (1.0, 0.0, 0.0)

    def f(lnP):
        P = float(np.exp(lnP))
        la = lnphi_aq(x_aq, mi, float(T), P)
        lg = lnphi_gas(y, float(T), P)
        return (np.log(x_w) + la[0]) - lg[0]

    p0 = PropsSI("P", "T", float(T), "Q", 0, "Water")   # Pa
    lo, hi = np.log(0.55 * p0), np.log(1.10 * p0)
    return float(np.exp(brentq(f, lo, hi, xtol=1e-12, rtol=1e-10))) / 1e5


def main():
    import argparse
    import pandas as pd
    from CoolProp.CoolProp import PropsSI
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true",
                    help="recompute the vapor pressures instead of reading vpsat.csv")
    if ap.parse_args().recompute or not (HERE / "vpsat.csv").exists():
        rows = []
        for m in MOLAL:
            for T in T_GRID:
                try:
                    p = psat_model(float(T), m)
                except Exception:
                    p = np.nan
                rows.append((m, float(T), p))
        df = pd.DataFrame(rows, columns=["m", "T_K", "P_bar"])
        df.to_csv(HERE / "vpsat.csv", index=False)
    else:
        df = pd.read_csv(HERE / "vpsat.csv")

    # Pitzer water activities at 298.15 K (phi = 0.9355, 1.0453, 1.2713)
    AW_298 = {1.0: np.exp(-2 * 1.0 * Mw * 0.9355),
              3.0: np.exp(-2 * 3.0 * Mw * 1.0453),
              6.0: np.exp(-2 * 6.0 * Mw * 1.2713)}

    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    mpl.rcParams.update({
        "font.size": 14, "axes.labelsize": 16,
        "xtick.labelsize": 13, "ytick.labelsize": 13,
        "legend.fontsize": 11,
        "font.weight": "bold", "axes.labelweight": "bold",
        "axes.linewidth": 1.6, "axes.edgecolor": "black",
        "xtick.major.width": 1.6, "ytick.major.width": 1.6,
        "xtick.major.size": 6, "ytick.major.size": 6,
        "xtick.direction": "in", "ytick.direction": "in",
        "lines.linewidth": 2.2, "lines.markersize": 8,
        "legend.frameon": True, "legend.edgecolor": "black",
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })
    WONG = ["#000000", "#E69F00", "#0072B2", "#D55E00"]
    LS = ["-", "--", "-.", ":"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8),
                                   constrained_layout=True)
    for i, m in enumerate(MOLAL):
        sub = df[df.m == m]
        ax1.semilogy(sub.T_K, sub.P_bar, LS[i], color=WONG[i],
                     label=("salt-free" if m == 0
                            else f"{m:.0f} mol kg$^{{-1}}$"))
    Tref = np.linspace(283.15, 573.15, 100)
    pref = [PropsSI("P", "T", t, "Q", 0, "Water") / 1e5 for t in Tref]
    ax1.semilogy(Tref[::7], np.array(pref)[::7], "o", mfc="white",
                 color="black", ms=6, lw=0, label="IAPWS (pure water)")
    ax1.set_xlabel(r"$T$ [K]")
    ax1.set_ylabel(r"$P_{\mathrm{sat}}$ [bar]")
    ax1.legend(loc="lower right", fontsize=10)
    ax1.text(0.05, 0.93, "(a)", transform=ax1.transAxes, fontsize=15,
             fontweight="bold")

    p0 = df[df.m == 0].set_index("T_K").P_bar
    for i, m in enumerate(MOLAL[1:], start=1):
        sub = df[df.m == m].set_index("T_K")
        ax2.plot(sub.index, sub.P_bar / p0, LS[i], color=WONG[i],
                 label=f"{m:.0f} mol kg$^{{-1}}$")
        ax2.plot([298.15], [AW_298[m]], "D", color=WONG[i], mec="k",
                 ms=9, lw=0)
    ax2.plot([], [], "D", color="0.5", mec="k", lw=0,
             label=r"Pitzer $a_w$ (298 K)")
    ax2.set_xlabel(r"$T$ [K]")
    ax2.set_ylabel(r"$P_{\mathrm{sat}}(m)\,/\,P_{\mathrm{sat}}(0)$")
    ax2.set_ylim(0.70, 1.02)
    ax2.legend(loc="center left", fontsize=10)
    ax2.text(0.9, 0.93, "(b)", transform=ax2.transAxes, fontsize=15,
             fontweight="bold")
    fig.savefig(FIG)
    print("->", FIG)
    # console summary
    for m in MOLAL[1:]:
        sub = df[df.m == m].set_index("T_K")
        r = (sub.P_bar / p0).dropna()
        def at(T):
            return float(r.iloc[int(np.argmin(np.abs(r.index.to_numpy() - T)))])
        # 298.15 K, not 293 K: AW_298 is a Pitzer activity at 298.15 K, and
        # that temperature is on the grid precisely so this is like-for-like.
        print(f"m={m}: P/P0 at 298K={at(298.15):.4f} (Pitzer298 {AW_298[m]:.4f}), "
              f"at 373K={at(373):.4f}, at 573K={at(573):.4f}")


if __name__ == "__main__":
    main()
