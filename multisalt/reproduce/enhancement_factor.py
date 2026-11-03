"""Where does the y_H2O error actually live? The enhancement factor says.

The water content of a CO2-rich phase is conventionally written as

    y_H2O = E(T,P) * Psat_w(T) / P          so       E = y_H2O * P / Psat_w(T)

E is the ENHANCEMENT FACTOR: the factor by which a real CO2 phase carries more
water than an ideal gas in equilibrium with pure water would. It tends to 1 in
the ideal-gas limit, and its departure from 1 is exactly the quantity under
suspicion -- the gas-phase non-ideality (phi_w in the CO2-rich phase), the
Poynting correction, and the water activity of the liquid.

Why this and not a density check. A correct pure-CO2 density does not settle
the question. y_w = x_w phi_w^aq / phi_w^gas depends on water's PARTIAL property in the
CO2-rich phase, and a correct pure-component density implies nothing about a
trace component's fugacity coefficient. E isolates that, and both E_exp and
E_model are observable -- no fitted parameter stands between them.

Reading the result:
  E_model / E_exp < 1  ->  our phi_w^gas is too LARGE: the CO2 phase is not
                           holding enough water, i.e. the CO2-H2O cross
                           attraction is too weak at those conditions.
  E_model / E_exp > 1  ->  the opposite.
If that ratio drifts with PRESSURE at fixed T, the deficiency is in how the
cross term responds to density -- which a T-only S13(T) cannot repair, and
which is the shape the y_H2O bias already showed.

SALT-FREE ROWS ONLY. 84.8 % of the CO2 y_H2O data carries no salt, and
excluding the rest keeps this a clean CO2-H2O binary: no ion energies, no
Born or Debye-Huckel term, no water-activity depression from salt.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import CoolProp.CoolProp as CP


def psat_water_bar(T_K: float) -> float:
    """Saturation pressure of pure water [bar], IAPWS-95 through CoolProp."""
    try:
        return float(CP.PropsSI("P", "T", float(T_K), "Q", 0, "Water")) / 1e5
    except Exception:
        return float("nan")


def main() -> int:
    from fitting import dataset as ds, refit as rf, blocks as B, parallel as par

    fams = ds.build(split_yh2o=True)
    fam = fams["co2_yh2o_sf"]            # salt-free half
    pool = par.start(fams)
    PUB = B.published_values(("A_Peni", "B_Peni", "A_U3i", "B_U3i",
                              "A_U2i", "B_U2i"))
    pred = rf.predicted(fam, "fortran", pool=pool, vals=PUB)
    pool.shutdown()
    obs = rf.observed(fam)

    d = fam.df.copy()
    d["y_model"], d["y_exp"] = pred, obs
    d["psat"] = [psat_water_bar(T) for T in d.T_K]
    d = d[np.isfinite(d.y_model) & np.isfinite(d.psat) & (d.y_exp > 0)]
    # Below water's saturation pressure there is no liquid water phase and E
    # is not defined; a small margin keeps the near-saturation states out too,
    # where E diverges and says nothing about the cross term.
    d = d[d.P_bar > 1.5 * d.psat]

    d["E_exp"] = d.y_exp * d.P_bar / d.psat
    d["E_model"] = d.y_model * d.P_bar / d.psat
    d["ratio"] = d.E_model / d.E_exp

    print("ENHANCEMENT FACTOR  E = y_H2O * P / Psat_w(T)   (CO2-H2O, salt-free)")
    print(f"{len(d)} rows; water Psat from IAPWS-95 via CoolProp\n")
    print("  E_model / E_exp   ( <1 = our CO2 phase carries too little water )")
    print("  %-14s %14s %14s %14s %14s" % ("T band", "P<50", "50-100",
                                           "100-300", ">300"))
    for tlo, thi in ((273, 304), (304, 320), (320, 350), (350, 400),
                     (400, 450), (450, 520)):
        sel = d[(d.T_K >= tlo) & (d.T_K < thi)]
        if len(sel) < 3:
            continue
        line = "  %-14s" % f"{tlo}-{thi} K"
        for plo, phi in ((0, 50), (50, 100), (100, 300), (300, 1e9)):
            g = sel[(sel.P_bar >= plo) & (sel.P_bar < phi)]
            line += "%14s" % (f"{g.ratio.mean():.3f}({len(g)})"
                              if len(g) >= 2 else "  -")
        print(line)

    print("\n  absolute scale -- is E itself the right size?")
    print("  %-14s %10s %10s %10s %8s" % ("P band", "E_exp", "E_model",
                                          "ratio", "N"))
    for plo, phi in ((0, 50), (50, 100), (100, 200), (200, 400), (400, 1e9)):
        g = d[(d.P_bar >= plo) & (d.P_bar < phi)]
        if len(g) < 3:
            continue
        print("  %-14s %10.3f %10.3f %10.3f %8d" %
              (f"{plo}-{phi if phi < 1e8 else 'inf'} bar",
               g.E_exp.mean(), g.E_model.mean(), g.ratio.mean(), len(g)))

    out = _ROOT / "reproduce" / "enhancement_factor.csv"
    d[["T_K", "P_bar", "psat", "y_exp", "y_model", "E_exp", "E_model",
       "ratio"]].to_csv(out, index=False)
    print(f"\n  [saved] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
