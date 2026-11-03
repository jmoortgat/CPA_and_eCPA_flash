"""Three-phase (V + Laq + L_CO2) locus of the CO2--H2O binary.

For each temperature below the CO2 critical point, the model
three-phase pressure P3(T) is the root of the stage-2 dense-trial
tangent-plane distance as a function of pressure: TPD > 0 below P3
(the CO2-rich liquid is unstable), TPD < 0 above it.  The locus is
compared against the pure-CO2 Span--Wagner saturation curve (via
CoolProp), which the experimental VLLE locus of Wendland et al.
(1999, J. Chem. Eng. Data 44:901) follows to within ~0.3 bar --- the
aqueous phase barely perturbs the CO2-rich phases.

Outputs: vlle_locus.csv + fig_vlle_locus.png (house style), and a
console table.

Run from ``code/``:  PYTHONPATH=. python pilot_studies/three_phase_2026_08/vlle_locus.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy.optimize import brentq

from ecpa_ms import stability_flash3

HERE = Path(__file__).resolve().parent
Z_FEED = np.array([0.5, 0.0, 0.5])
MI0 = np.zeros(6)


def locus_indicator(T, P):
    """Signed distance-to-locus indicator, positive below P3.

    The incumbent non-aqueous root can be either the vapor or the
    dense (liquid-CO2) root depending on where the two-phase SSI
    lands, so the raw stage-2 TPD is not sign-consistent across the
    locus.  Convention here: with a VAPOR incumbent, the dense trial's
    TPD is positive below P3 and negative above (use it directly);
    with a DENSE incumbent, the vapor trial's TPD carries the opposite
    sign (negate it).  A missing trial root pins the sign: no dense
    root exists far below P3 (+1), no vapor root far above it (-1).
    """
    r = stability_flash3(T, float(P), Z_FEED, MI0)
    if r.phases != 2:
        # Isolated stage-1 misclassification (observed: the production
        # two-phase stability declares 274.15 K / 43.783 bar
        # single-phase on a knife edge).  Report "no information".
        return np.nan
    tpd = r.tpd3_min
    if r.incumbent_gas_root == "dense":
        return -tpd if np.isfinite(tpd) else -1.0
    return tpd if np.isfinite(tpd) else 1.0


def locate_p3(T, p_lo, p_hi, n_scan=9):
    """Robustly locate the indicator's sign change in [p_lo, p_hi] (bar).

    Scans a small grid first (isolated NaN points from stage-1
    misclassifications are simply skipped), then Brents the innermost
    finite sign-changing pair.
    """
    Ps = np.linspace(p_lo, p_hi, n_scan)
    fs = np.array([locus_indicator(T, p) for p in Ps])
    ok = np.isfinite(fs)
    Ps, fs = Ps[ok], fs[ok]
    for i in range(len(Ps) - 1):
        if fs[i] > 0 >= fs[i + 1]:
            try:
                return brentq(lambda P: locus_indicator(T, P),
                              Ps[i], Ps[i + 1],
                              xtol=1e-3, rtol=1e-6, maxiter=60)
            except ValueError:
                return 0.5 * (Ps[i] + Ps[i + 1])
    return np.nan


def main():
    import argparse
    from CoolProp.CoolProp import PropsSI

    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true",
                    help="recompute the locus instead of reading vlle_locus.csv")
    recompute = ap.parse_args().recompute or not (HERE / "vlle_locus.csv").exists()

    rows = []
    print(f"{'T [K]':>8} {'P3 model [bar]':>15} {'Psat CO2 (SW) [bar]':>20} "
          f"{'dP [bar]':>9} {'dP [%]':>7}")
    if not recompute:
        import csv
        with open(HERE / "vlle_locus.csv") as fh:
            rows = [tuple(float(v) for v in r) for r in list(csv.reader(fh))[1:]]
        for T, p3, psat, dp in rows:
            print(f"{T:8.2f} {p3:15.3f} {psat:20.3f} {dp:9.3f} "
                  f"{100 * dp / psat:7.2f}")
    # Restricted to the parameterization's calibration range: T >=
    # 283.15 K (floor) up to just below the CO2 critical point.
    for T in ((283.15, 286.15, 289.15, 292.15, 295.15, 298.15,
               300.15, 302.15) if recompute else ()):
        psat = PropsSI("P", "T", T, "Q", 0, "CO2") / 1e5
        p3 = locate_p3(T, psat - 8.0, psat + 8.0)
        dp = p3 - psat
        rows.append((T, p3, psat, dp))
        print(f"{T:8.2f} {p3:15.3f} {psat:20.3f} {dp:9.3f} "
              f"{100 * dp / psat:7.2f}")

    if recompute:
        import csv
        with open(HERE / "vlle_locus.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["T_K", "P3_model_bar", "Psat_CO2_SpanWagner_bar",
                        "dP_bar"])
            w.writerows(rows)

    # ── Figure (house style) ────────────────────────────────────────────
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    mpl.rcParams.update({
        "font.size": 14, "axes.labelsize": 16,
        "xtick.labelsize": 13, "ytick.labelsize": 13,
        "legend.fontsize": 12,
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

    arr = np.asarray(rows, dtype=float)
    Ts = np.linspace(282.0, 304.0, 200)
    psat_curve = np.array(
        [PropsSI("P", "T", t, "Q", 0, "CO2") / 1e5 for t in Ts])

    fig, ax = plt.subplots(figsize=(6.0, 4.4), constrained_layout=True)
    ax.plot(Ts, psat_curve, "-", color=WONG[0],
            label=r"pure-CO$_2$ $P_{\rm sat}$ (Span--Wagner)")
    ax.plot(arr[:, 0], arr[:, 1], "o", color=WONG[5],
            markeredgecolor="k",
            label=r"eCPA three-phase locus $P_3(T)$")
    ax.set_xlabel(r"$T$ [K]")
    ax.set_ylabel(r"$P$ [bar]")
    ax.legend(loc="upper left")
    fig.savefig(HERE / "fig_vlle_locus.png")
    print(f"\nwrote {HERE / 'vlle_locus.csv'} and fig_vlle_locus.png")


if __name__ == "__main__":
    main()
