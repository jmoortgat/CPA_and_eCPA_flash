"""Predicted CH4 and CO2 behavior in the nineteen mixed-brine recipes.

Generates the two tables of Supporting-Information Section S9.6 of
Part I.  For every mixed-brine recipe of data/co2_mixed_brines.parquet
the ion-specific ms-eCPA prediction is compared with a NaCl brine of the
SAME ionic strength -- the surrogate a salt-lumped description uses.

Pass A: one reference condition, T = 323 K, P = 100 bar.
Pass B: every distinct (T, P) at which that recipe's own CO2 data were
        measured.
Pass C: sensitivity to the assumed feed, over total gas fraction and over
        water-free CO2 fraction, at the reference condition.

Feed is z = (H2O, CH4, CO2) = (1-zg, zg(1-yc), zg*yc), zg = 0.30,
yc = 0.50 unless scanned.  Salinity convention: library default
(as-aqueous, i.e. the prepared molalities are the aqueous molalities).

Run:  MS_ECPA_BACKEND=fortran python reproduce/mixed_brine_ch4_prediction.py
"""
import json

import numpy as np
import pandas as pd

from ms_ecpa import stability_flash
from ms_ecpa.parameters import Mw

IONS = ["m_Na", "m_Cl", "m_K", "m_Ca", "m_Mg", "m_SO4"]
ZI = np.array([1.0, -1.0, 1.0, 2.0, 2.0, -2.0])
DATA = "data/co2_mixed_brines.parquet"
TREF, PREF, ZG, YC = 323.0, 100.0, 0.30, 0.50


def point(T, P, mi, zg=ZG, yc=YC):
    """One stability test + flash -> aqueous molalities of CH4 and CO2."""
    z = [1.0 - zg, zg * (1.0 - yc), zg * yc]
    r = stability_flash(T, P, z, list(mi))
    fr = r.flash_result
    if r.phases != 2 or fr is None or not fr.converged:
        raise RuntimeError(f"no converged two-phase flash at {T} K, {P} bar")
    xw, xh, xc = fr.x_aq
    return dict(m_ch4=xh / (Mw * xw), m_co2=xc / (Mw * xw),
                y_co2_dry=fr.y_gas[2] / (fr.y_gas[1] + fr.y_gas[2]))


def compare(T, P, mi, sur, **kw):
    a, b = point(T, P, mi, **kw), point(T, P, sur, **kw)
    return dict(T=T, P=P, m_ch4=a["m_ch4"], m_ch4_nacl=b["m_ch4"],
                m_co2=a["m_co2"], m_co2_nacl=b["m_co2"],
                y_co2_dry=a["y_co2_dry"],
                d_ch4=100.0 * (a["m_ch4"] / b["m_ch4"] - 1.0),
                d_co2=100.0 * (a["m_co2"] / b["m_co2"] - 1.0),
                d_S=100.0 * ((a["m_co2"] / a["m_ch4"])
                             / (b["m_co2"] / b["m_ch4"]) - 1.0))


def main(out="results_cache/mixed_brine_ch4_prediction.json"):
    df = pd.read_parquet(DATA)
    res = {}
    for mix_id, sub in df.groupby("mix_id"):
        mi = sub[IONS].drop_duplicates()
        assert len(mi) == 1, f"{mix_id}: ion molalities vary within the recipe"
        mi = mi.iloc[0].to_numpy(float)
        Ic = 0.5 * float(np.sum(mi * ZI ** 2))
        sur = [Ic, Ic, 0.0, 0.0, 0.0, 0.0]
        rec = dict(Ic=Ic, mi=mi.tolist(), n_rows=int(len(sub)))
        rec["ref"] = compare(TREF, PREF, mi, sur)                       # A
        rec["window"] = [compare(float(T), float(P), mi, sur)           # B
                         for T, P in sorted(set(zip(sub.T_K, sub.P_bar)))]
        rec["scan_zg"] = {f"{v:.2f}": compare(TREF, PREF, mi, sur, zg=v)
                          for v in (0.10, 0.20, 0.30, 0.40, 0.50)}      # C
        rec["scan_yc"] = {f"{v:.2f}": compare(TREF, PREF, mi, sur, yc=v)
                          for v in (0.10, 0.25, 0.50, 0.75, 0.90)}
        res[mix_id] = rec

    win = [w for r in res.values() for w in r["window"]]
    summary = dict(
        n_recipes=len(res), n_window_states=len(win),
        n_flashes=2 * (len(win) + len(res) * 11),
        ref_d_ch4=[min(r["ref"]["d_ch4"] for r in res.values()),
                   max(r["ref"]["d_ch4"] for r in res.values())],
        ref_d_co2=[min(r["ref"]["d_co2"] for r in res.values()),
                   max(r["ref"]["d_co2"] for r in res.values())],
        ref_d_S=[min(r["ref"]["d_S"] for r in res.values()),
                 max(r["ref"]["d_S"] for r in res.values())],
        ref_ratio=sorted(r["ref"]["d_ch4"] / r["ref"]["d_co2"]
                         for r in res.values()),
        win_d_ch4=[min(w["d_ch4"] for w in win), max(w["d_ch4"] for w in win)],
        win_d_co2=[min(w["d_co2"] for w in win), max(w["d_co2"] for w in win)],
        win_d_S=[min(w["d_S"] for w in win), max(w["d_S"] for w in win)],
        win_all_ch4_positive=all(w["d_ch4"] > 0 for w in win),
        win_all_S_negative=all(w["d_S"] < 0 for w in win),
        win_ch4_dominates=all(w["d_ch4"] > abs(w["d_co2"]) for w in win),
        max_spread_zg_pp=max(max(s[k] for s in r["scan_zg"].values())
                             - min(s[k] for s in r["scan_zg"].values())
                             for r in res.values()
                             for k in ("d_ch4", "d_co2", "d_S")),
        max_spread_yc_pp=max(max(s[k] for s in r["scan_yc"].values())
                             - min(s[k] for s in r["scan_yc"].values())
                             for r in res.values()
                             for k in ("d_ch4", "d_co2", "d_S")),
    )
    with open(out, "w") as fh:
        json.dump(dict(summary=summary, recipes=res), fh, indent=1)
    for k, v in summary.items():
        print(f"{k:22s} {v}")


if __name__ == "__main__":
    main()
