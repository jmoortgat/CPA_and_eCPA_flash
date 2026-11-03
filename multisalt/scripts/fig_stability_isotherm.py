#!/usr/bin/env python3
"""Phase-stability TPD-vs-pressure isotherm figure.

Sweeps pressure at fixed (T, m_NaCl) and stores the min-TPD value across
the 8 Michelsen trial compositions in
``results_cache/fig_stability_isotherm_data.parquet``. Default conditions:
T = 323 K, m_NaCl = 1 mol/kg, P in [1, 500] bar (36 log-spaced points).

Usage:
    python3 scripts/fig_stability_isotherm.py            # use cache
    python3 scripts/fig_stability_isotherm.py --force    # recompute
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE))
sys.path.insert(0, str(_CODE / "scripts"))

from ms_ecpa import michelsen_tpd                      # noqa: E402
from paper_style import (                              # noqa: E402
    apply_style, bold_ticks, PAPER_FIGURES, FIGSIZE_15COL, WONG, LINESTYLES,
)
from data_loader import load_or_compute, add_cache_flag  # noqa: E402

# Family of isotherms spanning the calibration window.  All four use the
# same salt (NaCl 1 m) and the same CO2-rich feed z_CO2 = 0.60, so the
# only variable is temperature.  See long note below on why z_CO2 = 0.60
# is the right choice for this diagnostic.
T_LIST = (323.0, 373.0, 423.0, 473.0)
M_NACL = 1.0
P_GRID = np.geomspace(1.0, 500.0, 36)        # 36 log-spaced points
# Feed composition for the TPD-vs-P diagnostic. We pick a CO2-rich
# feed (z_CO2 = 0.60) for one practical reason: the single-phase EoS
# reference that michelsen_tpd builds at the feed composition needs a
# consistent gas-side OR aqueous-side density root to anchor d_i.
# Water-rich feeds (e.g. z_CO2 = 0.05) sit inside the two-phase
# envelope for an interior P window where neither root exists, so the
# reference fails and the figure shows a non-physical gap. With
# z_CO2 = 0.60 the reference converges across the full grid and the
# resulting TPD curve traces the single->two-phase transition we want to
# display: deeply negative once the gas split sets in around 15--20 bar,
# recovering toward zero at high P.  Below the split the trial panel's
# minimum is strictly positive, and ``michelsen_tpd`` does not report its
# value (see ``compute_tpd_sweep``), so those pressures are left out of
# the curve rather than drawn at zero.
Z_CO2 = 0.60

# Match the production thresholds used inside ``stability_flash``:
# any TPD below TPD_HARD triggers the flash; the soft fallback triggers
# a tight-tolerance confirmation.  See ``ms_ecpa/stability.py``:
#   stability_flash(..., tpd_threshold=-0.005, tpd_soft_threshold=-0.1)
TPD_HARD = -0.005
TPD_SOFT = -0.1


def compute_tpd_sweep() -> pd.DataFrame:
    mi = np.array([M_NACL, M_NACL, 0, 0, 0, 0])
    z = np.array([1.0 - Z_CO2, 0.0, Z_CO2])
    rows = []
    for T_K in T_LIST:
        print(f"  TPD sweep at T = {T_K:g} K, m_NaCl = {M_NACL} mol/kg, "
              f"z_CO2 = {Z_CO2} ...")
        t0 = time.perf_counter()
        n_ok = 0
        n_undet = 0
        for P in P_GRID:
            try:
                res = michelsen_tpd(z, mi, float(T_K), float(P))
            except Exception:
                rows.append({"T_K": float(T_K), "P_bar": float(P),
                             "tpd_min": float("nan")})
                continue
            # ``michelsen_tpd`` initialises tpd_min to 0.0 and lowers it only
            # for a trial that converges non-trivially below zero, leaving
            # best_trial_idx = -1 when none does.  The 0.0 it returns in that
            # case is the initialiser, not a minimum the panel measured: the
            # true minimum is either strictly positive (all trials gave
            # tpd >= 0) or undefined (every trial trivial or failed).  Record
            # it as missing, so neither outcome reaches the figure as a marker
            # sitting on the zero line inside the "stable" band.
            if res.best_trial_idx < 0:
                rows.append({"T_K": float(T_K), "P_bar": float(P),
                             "tpd_min": float("nan")})
                n_undet += 1
                continue
            rows.append({"T_K": float(T_K), "P_bar": float(P),
                         "tpd_min": float(res.tpd_min)})
            n_ok += 1
        msg = f"{n_ok}/{len(P_GRID)} with a determined min TPD"
        if n_undet:
            msg += f" ({n_undet} undetermined, no trial below zero)"
        print(f"  ...{time.perf_counter()-t0:.1f}s; " + msg)
    return pd.DataFrame(rows)


def make_figure(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE_15COL)
    # Colour + linestyle per isotherm, from cool (low T) to warm (high T)
    # in the Wong palette.  Also cycle line styles so grayscale prints
    # remain distinguishable.
    T_COLOR = (WONG[5], WONG[3], WONG[1], WONG[6])   # blue, bluish-green, orange, vermillion
    for k, T_K in enumerate(sorted(df["T_K"].unique())):
        sub = df[(df["T_K"] == T_K) & np.isfinite(df["tpd_min"])]
        if not len(sub):
            continue
        color = T_COLOR[k % len(T_COLOR)]
        ls = LINESTYLES[k % len(LINESTYLES)]
        ax.plot(sub["P_bar"], sub["tpd_min"], marker="o",
                color=color, linestyle=ls, markersize=5,
                linewidth=2.2, label=f"{int(T_K)} K")

    # Zero line + the two production thresholds.
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.axhline(TPD_HARD, color="#666666", linewidth=1.2, linestyle="--",
               label=f"hard threshold (${TPD_HARD:g}$)")
    ax.axhline(TPD_SOFT, color="#333333", linewidth=1.2, linestyle=":",
               label=f"soft fallback (${TPD_SOFT:g}$)")

    # Y-axis: extend well above 0 so the "stable / single-phase" label
    # has clear headroom above the shallowest plotted TPD.  ymax = +5
    # matches the deep-negative reach on the low-T isotherm.
    ymin_auto, _ = ax.get_ylim()
    ymax = 5.0
    ax.set_ylim(ymin_auto, ymax)
    ymin = ymin_auto
    ax.axhspan(TPD_HARD, ymax, facecolor="green", alpha=0.05)
    ax.axhspan(ymin, TPD_HARD, facecolor="red", alpha=0.05)

    # Region annotations on the right (high-P) side so they don't
    # overlap the low-P near-zero curves or the deep negatives.
    ax.text(0.97, 0.97, "stable\n(single phase)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=11, fontweight="bold", color="darkgreen")
    ax.text(0.97, 0.07, "unstable\n(two phase)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=11, fontweight="bold", color="darkred")

    ax.set_xlabel(r"$P$ [bar]")
    ax.set_ylabel(r"min$_j$ TPD$_j$")
    ax.set_xscale("log")
    ax.grid(alpha=0.25, which="both")
    bold_ticks(ax)
    # Legend above the panel; three columns fit the four isotherms +
    # two threshold lines in two neat rows.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=3, fontsize=10, frameon=True, borderaxespad=0.0,
              columnspacing=1.2, handletextpad=0.4)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_cache_flag(p)
    args = p.parse_args()
    apply_style()
    df = load_or_compute("fig_stability_isotherm", compute_tpd_sweep,
                         force=args.force)
    make_figure(df, PAPER_FIGURES / "stability_tpd_isotherm_NaCl.png")
    print(f"  stability_tpd_isotherm_NaCl.png")


if __name__ == "__main__":
    raise SystemExit(main())
