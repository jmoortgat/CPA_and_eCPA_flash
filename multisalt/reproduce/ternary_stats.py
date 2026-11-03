"""Errors of the CH4-CO2-H2O ternary predictions (Paper III, Tables S9 and S10).

For each of the 34 measured ternary states (data/ccw_ternary.parquet):

* stability-and-flash route: the flash of scripts/fig_ccw_ternary.py, retried
  over several total gas feed fractions (the route of Paper III, Figure 5);
* direct route: the two-phase equilibrium solved at the measured water-free
  CO2 fraction of the gas, Y_CO2 (ms_ecpa.eos.solve_elv), so that no feed
  assumption enters.

The direct route is repeated with the CH4-CO2 binary parameter k_hc set to 0,
0.5, 1 and 2 times its tabulated value (Table S10).

Usage (from multisalt/):
    PYTHONPATH=.:scripts python3 reproduce/ternary_stats.py
Writes results_cache/ternary_stats.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import ms_ecpa.eos as eos                       # noqa: E402
import ms_ecpa.lnphi as lnphi                   # noqa: E402
import ms_ecpa.parameters as prm                # noqa: E402
from data_loader import load_experimental       # noqa: E402
from fig_ccw_ternary import _flash_one          # noqa: E402

OUT = ROOT / "results_cache" / "ternary_stats.json"
VIEWS = {"x_CH4_aq": ("x_CH4_aq_exp", "x2w"), "x_CO2_aq": ("x_CO2_aq_exp", "x3w"),
         "y_H2O_gas": ("y_H2O_gas_exp", "x1h")}
PLOTTED = (344, 376, 423)
K23 = prm.k23


def set_k23(value):
    """k_hc enters as a module-level name in each module that uses it."""
    for mod in (prm, eos, lnphi):
        mod.k23 = value


def stats(exp, pred):
    """AARE (%) and N per view, by isotherm, for the plotted isotherms and for all."""
    out = {}
    groups = [(str(int(T)), exp.T_K == T) for T in sorted(exp.T_K.unique())]
    groups += [("plotted", exp.T_K.isin(PLOTTED)), ("all", exp.T_K > 0)]
    for name, sel in groups:
        row = {}
        for view, (col, _) in VIEWS.items():
            e, p = exp.loc[sel, col].to_numpy(float), pred.loc[sel, view].to_numpy(float)
            ok = np.isfinite(e) & np.isfinite(p)
            row[view] = dict(N=int(ok.sum()),
                             AARE=float(100 * np.mean(np.abs(p[ok] - e[ok]) / e[ok])) if ok.any() else None)
        out[name] = row
    return out


def direct(exp):
    rows = []
    for r in exp.itertuples():
        s = eos.solve_elv(float(r.T_K), float(r.P_bar), float(r.Yc_gas), [0.0] * 6)
        rows.append({v: (getattr(s, a) if s.converged else np.nan) for v, (_, a) in VIEWS.items()})
    return rows


def main():
    import pandas as pd
    exp = load_experimental("ccw_ternary").reset_index(drop=True)
    route_a = pd.DataFrame([{v: _flash_one(r.T_K, r.P_bar, r.Yc_gas)[v + "_pred"] for v in VIEWS}
                            for r in exp.itertuples()])
    res = {"n_states": len(exp), "sources": exp.groupby("T_K").author.first().to_dict(),
           "stability_flash": stats(exp, route_a)}
    scan = {}
    for f in (0.0, 0.5, 1.0, 2.0):
        set_k23(f * K23)
        scan[f"{f:g}"] = stats(exp, pd.DataFrame(direct(exp)))
    set_k23(K23)
    res["direct"] = scan["1"]
    res["k_hc_scan"] = {k: v["all"] for k, v in scan.items()}
    res["k_hc"] = K23
    OUT.write_text(json.dumps(res, indent=2, default=str) + "\n")
    for route in ("stability_flash", "direct"):
        print(route)
        for g, row in res[route].items():
            print(f"  {g:8s}", "  ".join(f"{v}: N={x['N']:2d} AARE={x['AARE'] if x['AARE'] is None else round(x['AARE'], 1)}"
                                         for v, x in row.items()))
    print("k_hc scan (all states):")
    for k, row in res["k_hc_scan"].items():
        print(f"  {k:4s}", "  ".join(f"{v}: {round(x['AARE'], 1)}" for v, x in row.items()))


if __name__ == "__main__":
    main()
