"""Propane--water three-phase (V + Laq + L_C3H8) locus validation.

With the fitted kij(T) from c3_pilot.py (fitB), locate the model
three-phase pressure P3(T) via the stage-2 TPD sign change of
stability_flash3 (root-aware, as in the CO2 vlle_locus study) and
compare against Kobayashi's MEASURED smoothed three-phase locus and
phase compositions (dissertation Table IV) --- data the CO2 system
does not offer (its locus references are pressure-only).

Comparisons at each Kobayashi locus temperature >= 283.15 K:
  P3 [bar];  and at the model's own P3: the three-phase compositions
  y_H2O (vapor), x_H2O (liquid propane), x_C3 (water-rich liquid).

Run from multisalt/:
  PYTHONPATH=. python pilot_studies/c3_pilot_2026_08/c3_locus.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
sys.path.insert(0, str(CODE))

from c3_pilot import rebind_c3, restore_ch4  # noqa: E402  (same dir)
from ms_ecpa import stability_flash3          # noqa: E402

sys.path.insert(0, str(HERE))

Z_FEED = np.array([0.5, 0.5, 0.0])   # water + propane (slot 2)
MI0 = np.zeros(6)


def locus_indicator(kij, T, P):
    """Signed stage-2 TPD at (T, P), or NaN where it is not defined.

    A NaN tpd3_min means the stage-2 search returned nothing usable at this
    pressure -- not that the tangent plane sits on one side of zero. An
    earlier version substituted -1.0 or +1.0 there, which manufactured a
    sign change across any NaN region: the bisection then converged on the
    EDGE OF THAT REGION and reported it as a three-phase pressure. Near the
    critical endpoint, where the stage-2 search fails over a widening band,
    that is precisely where the spurious points appeared -- 33.01 bar at
    360.93 K below the 33.56 bar at 355.37 K, a locus falling with rising
    temperature, and three deviations near -11 %. Returning NaN instead
    lets locate_p3 refuse to bracket, so an undefined locus reads as
    undefined rather than as a number.
    """
    rebind_c3(*kij)
    try:
        r = stability_flash3(T, float(P), Z_FEED, MI0)
    finally:
        restore_ch4()
    if r.phases != 2:
        return np.nan
    tpd = r.tpd3_min
    if not np.isfinite(tpd):
        return np.nan
    return -tpd if r.incumbent_gas_root == "dense" else tpd


def locate_p3(kij, T, p_lo, p_hi, n_scan=25):
    """Bracket and bisect a genuine sign change of a FINITE indicator.

    Both ends of the bracket must be finite AND ADJACENT in the scan. The
    earlier version dropped the NaN entries first and then bracketed
    neighbours of the compacted array, which silently joins two pressures
    that straddle an undefined band and bisects across it. Adjacency is
    checked on the original grid for that reason.

    The fallback when brentq cannot bracket is also removed: returning the
    midpoint of the bracket turns a failed root-find into a plausible-looking
    pressure. An unlocatable locus returns NaN.
    """
    Ps = np.linspace(p_lo, p_hi, n_scan)
    fs = np.array([locus_indicator(kij, T, p) for p in Ps])
    for i in range(len(Ps) - 1):
        a, b = fs[i], fs[i + 1]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if a > 0 >= b:
            try:
                return brentq(lambda P: locus_indicator(kij, T, P),
                              Ps[i], Ps[i + 1], xtol=2e-3, maxiter=50)
            except (ValueError, RuntimeError):
                return np.nan
    return np.nan


def compositions_at(kij, T, P):
    """Three-phase compositions at (T, P); falls back to the two
    branches straddling P when the driver keeps two phases."""
    rebind_c3(*kij)
    try:
        r = stability_flash3(T, float(P), Z_FEED, MI0)
        if r.phases == 3:
            f3 = r.flash3
            return dict(yH2O=f3.y_gas[0], xH2O_liq=f3.x_lco2[0],
                        xC3_water=f3.x_aq[1], how="3-phase")
        # Straddle: vapor branch just below, LL branch just above.
        lo = stability_flash3(T, float(P) - 0.15, Z_FEED, MI0)
        hi = stability_flash3(T, float(P) + 0.15, Z_FEED, MI0)
        out = dict(yH2O=np.nan, xH2O_liq=np.nan, xC3_water=np.nan,
                   how="straddle")
        fl = lo.two_phase.flash_result
        fh = hi.two_phase.flash_result
        if fl is not None and fl.converged:
            out["yH2O"] = fl.y_gas[0]
            out["xC3_water"] = fl.x_aq[1]
        if fh is not None and fh.converged:
            out["xH2O_liq"] = fh.y_gas[0]   # dense branch: "gas" slot
            if not np.isfinite(out["xC3_water"]):
                out["xC3_water"] = fh.x_aq[1]
        return out
    finally:
        restore_ch4()


def main():
    import pandas as pd
    st = json.loads((HERE / "work_state.json").read_text())
    kij = st["fitB"]["best"]["x"]
    print(f"fitted kij: {kij}")

    kob = pd.read_csv(HERE / "raw" / "kobayashi1951_3phase_smoothed.csv",
                      comment="#")
    kob = kob[kob.T_K >= 283.15]
    rows = []
    print(f"{'T [K]':>8} {'P3 mod':>8} {'P3 Kob':>8} {'dP%':>6} "
          f"{'yH2O mod/exp':>17} {'xH2O_l mod/exp':>17} "
          f"{'xC3_w mod/exp':>17}")
    for r in kob.itertuples():
        T = float(r.T_K)
        P_exp = float(r.P_bar)
        p3 = locate_p3(kij, T, 0.93 * P_exp, 1.07 * P_exp)
        comp = (compositions_at(kij, T, p3)
                if np.isfinite(p3) else dict(yH2O=np.nan, xH2O_liq=np.nan,
                                             xC3_water=np.nan, how="none"))
        rows.append(dict(T_K=T, P3_model_bar=p3, P3_exp_bar=P_exp,
                         dP_pct=100 * (p3 - P_exp) / P_exp,
                         yH2O_model=comp["yH2O"], yH2O_exp=r.yH2O_vapor,
                         xH2O_liq_model=comp["xH2O_liq"],
                         xH2O_liq_exp=r.xH2O_liqC3,
                         xC3_water_model=comp["xC3_water"],
                         xC3_water_exp=r.xC3_water, how=comp["how"]))
        d = rows[-1]
        print(f"{T:8.2f} {p3:8.3f} {P_exp:8.3f} {d['dP_pct']:6.2f} "
              f"{d['yH2O_model']:8.5f}/{d['yH2O_exp']:8.5f} "
              f"{d['xH2O_liq_model']:8.5f}/{d['xH2O_liq_exp']:8.5f} "
              f"{d['xC3_water_model']:.3e}/{d['xC3_water_exp']:.3e}",
              flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "c3_locus_validation.csv", index=False)
    ok = df.dP_pct.abs()
    print(f"\nP3: mean |dP| {ok.mean():.2f}%  max {ok.max():.2f}%")
    for col, lab in (("yH2O", "y_H2O vapor"), ("xH2O_liq", "x_H2O liq C3"),
                     ("xC3_water", "x_C3 water")):
        m = df[f"{col}_model"] / df[f"{col}_exp"] - 1.0
        m = m[np.isfinite(m)]
        print(f"{lab}: mean dev {100*m.mean():+.1f}%  AAD "
              f"{100*m.abs().mean():.1f}%  (n={len(m)})")
    print(f"-> {HERE / 'c3_locus_validation.csv'}")


if __name__ == "__main__":
    main()
