"""Paper figures for the C3H8 addition (part_II_threephase variant).

Figure 1  c3_propane_main.png   (a) binary C3H8-H2O isotherms: model
                                curves vs Kobayashi smoothed + Chapoy;
                                (b) Setschenow k_smm(T): NaCl transfer
                                vs calibrated + KCl.
Figure 2  c3_threephase_main.png (a) three-phase locus P3(T) model vs
                                Kobayashi measured; (b) compositions
                                along the locus (y_H2O, x_C3, x_H2O in
                                liquid C3) model vs measured.

Writes into figures/ (or $ECPA_PAPER_FIGURES).
Run from multisalt/:  PYTHONPATH=. python pilot_studies/c3_pilot_2026_08/c3_fig_final.py
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(HERE))

import c3_pilot as cp                                   # noqa: E402

FIG_DIR = Path(os.environ.get("ECPA_PAPER_FIGURES",
               Path(__file__).resolve().parents[2] / "figures"))
FIG_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib as mpl                                 # noqa: E402
mpl.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402

mpl.rcParams.update({
    "font.size": 14, "axes.labelsize": 16,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 11,
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
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1, 1, 1))]

ISO_T = (310.93, 344.26, 377.59, 410.93)   # Kobayashi grid isotherms
CHAP_T = 288.13
P_CURVE = np.concatenate([np.arange(4.0, 30.0, 2.0),
                          np.arange(30.0, 210.0, 10.0)])


def _predict_x2_3p(task):
    """Curve predictor via the THREE-phase driver: replaces metastable
    vapor--water splits above the three-phase locus (the plain
    two-phase predictor rides the metastable vapor branch past P3,
    producing a spurious solubility spike near the locus)."""
    Ak, Bk, Ck, T, P_bar, zg = task
    from ms_ecpa import stability_flash3
    try:
        cp.rebind_c3(Ak, Bk, Ck)
        mi = np.zeros(6)
        z = np.array([1.0 - zg, zg, 0.0])
        try:
            r = stability_flash3(float(T), float(P_bar), z, mi)
        except Exception:
            return float("nan")
        if r.phases == 3 and r.flash3 is not None:
            return float(r.flash3.x_aq[1])
        if r.phases == 2 and r.two_phase.flash_result is not None \
                and r.two_phase.flash_result.converged:
            return float(r.two_phase.flash_result.x_aq[1])
        return float("nan")
    finally:
        cp.restore_ch4()


def compute_curves(kij, pool):
    """Model isotherms with dew-line-degenerate points masked: each
    point is computed at two feeds (z_gas 0.05 and 0.30) and kept only
    if the two agree within 2% — the same feed-independence criterion
    the fit's probe stage uses."""
    tasks, keys = [], []
    for T in ISO_T + (CHAP_T,):
        for P in P_CURVE:
            for zg in (0.05, 0.30):
                tasks.append((kij[0], kij[1], kij[2], float(T), float(P),
                              zg))
                keys.append((T, P, zg))
    out = pool.map(_predict_x2_3p, tasks)
    raw = {}
    for (T, P, zg), x in zip(keys, out):
        raw.setdefault((T, P), {})[zg] = x
    curves = {}
    for (T, P), d in raw.items():
        a, b = d.get(0.05, np.nan), d.get(0.30, np.nan)
        if np.isfinite(a) and np.isfinite(b) and abs(a / b - 1.0) < 0.02:
            x = a
        else:
            x = np.nan
        curves.setdefault(T, []).append((P, x))
    return {T: np.array(sorted(v)) for T, v in curves.items()}


def compute_ks_curves(kij, s_nacl, s_k):
    Ts = np.arange(283.15, 353.16, 5.0)
    wc = {}
    nacl_fit, nacl_tr = [], []
    for T in Ts:
        nacl_fit.append(cp.ks_model(kij, s_nacl, float(T), "NaCl",
                                    water_cache=wc))
        nacl_tr.append(cp.ks_model(kij, 1.0, float(T), "NaCl",
                                   water_cache=wc))
    # KCl: transfer curve and the two-scale ((Na,Cl) at s_nacl, K+ at
    # s_k) curve.  The water reference MUST be computed before the
    # custom rebind: cp._mgas_f rebinds and restores internally, which
    # would wipe an already-applied custom rebind.
    Ts_k = np.arange(283.15, 313.16, 5.0)
    kcl_fit, kcl_tr = [], []
    import ms_ecpa.parameters as prm
    from ms_ecpa import stability_flash
    for T in Ts_k:
        sw = cp._mgas_f((*kij, 1.0, cp.POL_C3, float(T),
                         cp.KS_P_TOTAL_BAR, cp.MI_WATER,
                         cp.Z_GAS_RETRIES))
        sb = None
        try:
            cp.rebind_c3(*kij, s_all=s_nacl)
            prm.A_U2i[2] = cp._ORIG_A_U2i[2] * cp.TC_RATIO * s_k
            prm.B_U2i[2] = cp._ORIG_B_U2i[2] * s_k
            for zg in cp.Z_GAS_RETRIES:
                z = np.array([1.0 - zg, zg, 0.0])
                try:
                    r = stability_flash(float(T), cp.KS_P_TOTAL_BAR, z,
                                        np.array(cp.mi_for("KCl", 1.0)))
                except Exception:
                    continue
                if r.phases == 2 and r.flash_result is not None:
                    fr = r.flash_result
                    sb = (fr.x_aq[1] / (prm.Mw * fr.x_aq[0]),
                          fr.fugacities_gas[1])
                    break
        finally:
            cp.restore_ch4()
        if sb is None or not np.isfinite(sw[0]):
            kcl_fit.append(np.nan)
        else:
            kcl_fit.append(float(np.log10((sw[0] / sw[1])
                                          / (sb[0] / sb[1]))))
        kcl_tr.append(cp.ks_model(kij, 1.0, float(T), "KCl",
                                  water_cache=wc))
    return (Ts, np.array(nacl_tr), np.array(nacl_fit),
            Ts_k, np.array(kcl_tr), np.array(kcl_fit))


def fig1a(kij, curves):
    """Main-text panel: binary isotherms -> c3_propane_main.png."""
    import pandas as pd
    kob = pd.read_csv(HERE / "raw" /
                      "kobayashi1951_2phase_xC3_water_smoothed.csv",
                      comment="#")
    chap = pd.read_csv(HERE / "raw" / "chapoy2004_solubility.csv",
                       comment="#")
    fig, ax = plt.subplots(figsize=(7.0, 5.2), constrained_layout=True)
    for i, T in enumerate(ISO_T):
        c = WONG[(i % 7) + 1]
        cv = curves[T]
        ok = np.isfinite(cv[:, 1])
        ax.plot(cv[ok, 0], 1e4 * cv[ok, 1], ls=LINESTYLES[i % 5],
                color=c, label=f"{T:.0f} K")
        sub = kob[np.isclose(kob.T_K, T, atol=0.05)]
        ax.plot(sub.P_bar, 1e4 * sub.xC3, "o", color=c, mec="k", ms=6,
                lw=0)
    cv = curves[CHAP_T]
    ok = np.isfinite(cv[:, 1]) & (cv[:, 0] <= 40)
    ax.plot(cv[ok, 0], 1e4 * cv[ok, 1], ls=LINESTYLES[4],
            color=WONG[6], label=f"{CHAP_T:.0f} K")
    sub = chap[np.isclose(chap.T_K, CHAP_T, atol=0.05)]
    ax.plot(sub.P_bar, 1e4 * sub.x_C3H8, "s", color=WONG[6], mec="k",
            ms=6, lw=0)
    ax.set_xlabel(r"$P$ [bar]")
    ax.set_ylabel(r"$x_{\mathrm{C_3H_8}}^{\mathrm{aq}} \times 10^4$")
    ax.set_xlim(0, 210)
    ax.legend(loc="lower right", ncol=2, title="isotherms",
              title_fontproperties={"weight": "bold"})
    fig.savefig(FIG_DIR / "c3_binary_main.png")
    plt.close(fig)
    # Report the file that was actually written. This printed
    # c3_propane_main.png while line 194 saves c3_binary_main.png -- the name
    # the manuscript references -- so a regeneration looked like it had
    # produced the wrong file, and the stale c3_propane_main.png sitting
    # beside it (nothing writes or reads it any more) made that plausible.
    print(f"-> {FIG_DIR / 'c3_binary_main.png'}")


def fig1b(ks):
    """SI panel: Setschenow calibration -> c3_setschenow_SI.png."""
    fig, ax = plt.subplots(figsize=(7.0, 5.2), constrained_layout=True)
    Ts, na_tr, na_fit, Tk, k_tr, k_fit = ks
    ax.plot(Ts, na_tr, ls=":", color=WONG[1],
            label=r"NaCl, CH$_4$ transfer")
    ax.plot(Ts, na_fit, ls="-", color=WONG[1],
            label=r"NaCl, $s_{\mathrm{NaCl}}$ = 0.842")
    mb_T = [285.75, 303.15, 322.55, 344.85]
    mb_k = [0.216, 0.194, 0.178, 0.165]
    ax.plot(mb_T, mb_k, "o", color=WONG[1], mec="k", ms=8, lw=0,
            label="Morrison & Billett")
    ax.plot(Tk, k_tr, ls=":", color=WONG[5],
            label=r"KCl, CH$_4$ transfer")
    ax.plot(Tk, k_fit, ls="--", color=WONG[5],
            label=r"KCl, $s_{\mathrm{K}}$ = 1.676")
    ax.plot([298.2], [0.191], "s", color=WONG[5], mec="k", ms=8, lw=0,
            label="Yano (KCl)")
    ax.set_xlabel(r"$T$ [K]")
    ax.set_ylabel(r"$k_{\mathrm{smm}}$ [kg mol$^{-1}$]")
    ax.legend(loc="upper right", fontsize=9.5)
    fig.savefig(FIG_DIR / "c3_setschenow_SI.png")
    plt.close(fig)
    print(f"-> {FIG_DIR / 'c3_setschenow_SI.png'}")


def fig2():
    import pandas as pd
    val = pd.read_csv(HERE / "c3_locus_validation.csv")
    kob = pd.read_csv(HERE / "raw" / "kobayashi1951_3phase_smoothed.csv",
                      comment="#")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6),
                             constrained_layout=True)
    ax = axes[0]
    ax.plot(kob.T_K, kob.P_bar, "-", color=WONG[0],
            label="measured locus (Kobayashi)")
    ok = np.isfinite(val.P3_model_bar)
    ax.plot(val.T_K[ok], val.P3_model_bar[ok], "o", color=WONG[5],
            mec="k", label=r"eCPA $P_3(T)$")
    ax.axvspan(357.0, 370.5, color="0.85", zorder=0)
    ax.text(363.5, 12.0, "near-critical\nendpoint", ha="center",
            fontsize=10, fontweight="bold")
    ax.set_xlabel(r"$T$ [K]")
    ax.set_ylabel(r"$P$ [bar]")
    ax.legend(loc="upper left")
    ax.text(0.03, 0.83, "(a)", transform=ax.transAxes, fontsize=15,
            fontweight="bold", va="top")

    ax = axes[1]
    m = val[val.T_K <= 356.0]  # inside the validated window
    ax.semilogy(kob.T_K, kob.yH2O_vapor, "-", color=WONG[2])
    ax.semilogy(m.T_K, m.yH2O_model, "o", color=WONG[2], mec="k", lw=0,
                label=r"$y_{\mathrm{H_2O}}$ vapor")
    ax.semilogy(kob.T_K, kob.xC3_water, "--", color=WONG[3])
    ax.semilogy(m.T_K, m.xC3_water_model, "s", color=WONG[3], mec="k",
                lw=0, label=r"$x_{\mathrm{C_3H_8}}$ aqueous")
    ax.semilogy(kob.T_K, kob.xH2O_liqC3, "-.", color=WONG[7])
    ax.semilogy(m.T_K, m.xH2O_liq_model, "^", color=WONG[7], mec="k",
                lw=0, label=r"$x_{\mathrm{H_2O}}$ liq. C$_3$H$_8$")
    ax.set_xlabel(r"$T$ [K]")
    ax.set_ylabel("mole fraction")
    ax.set_xlim(280, 372)
    ax.legend(loc="upper left", fontsize=10)
    ax.text(0.9, 0.1, "(b)", transform=ax.transAxes, fontsize=15,
            fontweight="bold", va="top")
    fig.savefig(FIG_DIR / "c3_threephase_main.png")
    plt.close(fig)
    print(f"-> {FIG_DIR / 'c3_threephase_main.png'}")


def main():
    st = json.loads((HERE / "work_state.json").read_text())
    kij = st["fitB"]["best"]["x"]
    s_nacl = st["fitS"]["best"]["x"][0]
    s_k = st["fitK"]["best"]["x"][0]
    print(f"kij={kij}  s_NaCl={s_nacl:.4f}  s_K={s_k:.4f}")
    import pandas as pd
    cache = HERE / "c3_curves_cache.csv"
    if cache.exists():
        df = pd.read_csv(cache)
        curves = {T: sub[["P", "x"]].to_numpy()
                  for T, sub in df.groupby("T_K")}
        print(f"curves from cache: {cache}")
    else:
        with mp.Pool(cp.N_WORKERS) as pool:
            curves = compute_curves(kij, pool)
        rows = [dict(T_K=T, P=p, x=x)
                for T, arr in curves.items() for p, x in arr]
        pd.DataFrame(rows).to_csv(cache, index=False)
    ks_cache = HERE / "c3_ks_cache.csv"
    if ks_cache.exists():
        df = pd.read_csv(ks_cache)
        ks = (df.Ts.dropna().to_numpy(), df.na_tr.dropna().to_numpy(),
              df.na_fit.dropna().to_numpy(), df.Tk.dropna().to_numpy(),
              df.k_tr.dropna().to_numpy(), df.k_fit.dropna().to_numpy())
        print(f"ks from cache: {ks_cache}")
    else:
        ks = compute_ks_curves(kij, s_nacl, s_k)
        n = max(len(v) for v in ks)
        def pad(v):
            v = list(v) + [np.nan] * (n - len(v))
            return v
        pd.DataFrame(dict(Ts=pad(ks[0]), na_tr=pad(ks[1]),
                          na_fit=pad(ks[2]), Tk=pad(ks[3]),
                          k_tr=pad(ks[4]), k_fit=pad(ks[5]))
                     ).to_csv(ks_cache, index=False)
    fig1a(kij, curves)
    fig1b(ks)
    fig2()


if __name__ == "__main__":
    main()
