"""SI figure: water content of the CO2-rich phase down to 1 bar.

Model isotherms (salt-free solid, 2m NaCl dashed) against the pure-water
Raoult limit Psat_w(T)/P (thin dotted). House style.
Run from multisalt/: PYTHONPATH=. python pilot_studies/ywater_lowP_2026_08/fig_ywater.py [--recompute]
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

HERE = Path(__file__).resolve().parent
CSV = HERE / "ywater_lowP.csv"
FIG = Path(os.environ.get("ECPA_PAPER_FIGURES",
           Path(__file__).resolve().parents[2] / "figures")) / "ywater_lowP_SI.png"
FIG.parent.mkdir(parents=True, exist_ok=True)

T_LIST = (298.15, 323.15, 348.15, 373.15)
P_LIST = np.geomspace(1.0, 100.0, 18)
SALTS = {"water": (0,) * 6, "NaCl2m": (2, 2, 0, 0, 0, 0)}


def run_one(task):
    T, P, mi = task
    from ecpa_ms import stability_flash
    for zg in (0.05, 0.02, 0.10):
        z = np.array([1.0 - zg, 0.0, zg])
        try:
            r = stability_flash(float(T), float(P),
                                z, np.asarray(mi, float))
        except Exception:
            continue
        if r.phases == 2 and r.flash_result is not None \
                and r.flash_result.converged:
            return float(r.flash_result.y_gas[0])
    return np.nan


def main():
    import argparse
    from multiprocessing import Pool
    import pandas as pd
    from CoolProp.CoolProp import PropsSI
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true",
                    help="recompute the water contents instead of reading ywater_lowP.csv")
    tasks, keys = [], []
    for salt, mi in SALTS.items():
        for T in T_LIST:
            for P in P_LIST:
                tasks.append((T, float(P), mi))
                keys.append((salt, T, float(P)))
    if ap.parse_args().recompute or not CSV.exists():
        with Pool(8) as pool:
            out = pool.map(run_one, tasks)
        pd.DataFrame([dict(salt=s, T_K=T, P_bar=P, y_H2O_gas=v)
                      for (s, T, P), v in zip(keys, out)]).to_csv(CSV, index=False)
    df = pd.read_csv(CSV)
    res = {(r.salt, r.T_K, r.P_bar): r.y_H2O_gas for r in df.itertuples()}
    res = {k: res.get(k, np.nan) for k in keys}

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
        "lines.linewidth": 2.2, "lines.markersize": 7,
        "legend.frameon": True, "legend.edgecolor": "black",
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })
    WONG = ["#000000", "#E69F00", "#56B4E9", "#009E73",
            "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

    fig, ax = plt.subplots(figsize=(7.0, 5.4), constrained_layout=True)
    for i, T in enumerate(T_LIST):
        col = WONG[[0, 1, 5, 6][i]]
        y_w = np.array([res[("water", T, float(P))] for P in P_LIST])
        y_b = np.array([res[("NaCl2m", T, float(P))] for P in P_LIST])
        ok = np.isfinite(y_w)
        ax.loglog(P_LIST[ok], y_w[ok], "-", color=col,
                  label=f"{T:.0f} K")
        ok = np.isfinite(y_b)
        ax.loglog(P_LIST[ok], y_b[ok], "--", color=col)
        psat = PropsSI("P", "T", T, "Q", 0, "Water") / 1e5
        Pr = P_LIST[P_LIST > psat * 1.05]
        ax.loglog(Pr, psat / Pr, ":", color=col, lw=1.3)
    ax.set_xlabel(r"$P$ [bar]")
    ax.set_ylabel(r"$y_{\mathrm{H_2O}}^{\mathrm{gas}}$")
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([], [], color="0.3", ls="-"),
                Line2D([], [], color="0.3", ls="--"),
                Line2D([], [], color="0.3", ls=":", lw=1.3)]
    labels += ["salt-free", r"2 mol kg$^{-1}$ NaCl",
               r"Raoult limit $P_{\rm sat}/P$"]
    ax.legend(handles, labels, loc="lower left", ncol=2, fontsize=9.5)
    fig.savefig(FIG)
    print("->", FIG)


if __name__ == "__main__":
    main()
