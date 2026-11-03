"""Every state the papers validate against: does it converge, and if not, why?

Papers II and III claim the flash converges everywhere. That claim is worth
making true rather than true by construction: a guard that drops failures from
an average, or a figure pass that paints them over with a neighbour's value,
makes the number look right while hiding the states that need work.

So this is a census, not a filter. For every row of every family it records
whether a result was produced, and for the failures which stage gave out and
what it said -- so the failures can be grouped into regimes and attacked with
a fallback scheme, a wider seed ladder, or more iterations, rather than
excluded.

Two distinct solvers can fail, and they are reported separately because they
need different remedies:

  * the stability + flash path (solubility, y_H2O families)
  * the gas-free density solver (rho), whose three-variable Newton has several
    roots and its own seed ladder
"""
from __future__ import annotations

import collections, json, sys, time
import warnings; warnings.filterwarnings("ignore")
import numpy as np

from fitting import blocks as B, dataset as ds, parallel as par
from fitting import predict as pr, refit as rf
from fitting.predict import Z_GAS_RETRIES, _slot
from ms_ecpa.backend import _ROUTE

OUT = sys.argv[1] if len(sys.argv) > 1 else "results_census"
import pathlib; pathlib.Path(OUT).mkdir(parents=True, exist_ok=True)

GAS_FAMS = ("co2_solubility", "co2_binary", "co2_yh2o",
            "ch4_solubility", "ch4_binary", "ch4_yh2o")


def diagnose_gas(row, fam_name):
    """Why did this row produce nothing? Walk the same ladder predict() walks."""
    gas = "co2" if fam_name.startswith("co2") else "ch4"
    k = _slot(gas)
    mi = np.nan_to_num(np.array([row[c] for c in ds.IONS], dtype=float))
    T, P = float(row["T_K"]), float(row["P_bar"])
    eng = pr._engine()
    seen = []
    for zg in Z_GAS_RETRIES:
        z = np.zeros(3); z[0] = 1.0 - zg; z[k] = zg
        try:
            r = eng.stability_flash(T, P, z, mi, salinity_convention="as-aqueous")
        except Exception as exc:
            seen.append(f"exception:{type(exc).__name__}")
            continue
        ph = int(r.get("phases", 0))
        route = int(r.get("route", -99))
        if ph != 2:
            seen.append(f"single-phase(route {route})")
            continue
        if not r.get("has_flash") or not r["flash"].get("converged"):
            seen.append(f"flash-not-converged(route {route})")
            continue
        f = r["flash"]
        xH, xg = float(f["x_aq"][0]), float(f["x_aq"][k])
        if xH + xg <= 0:
            seen.append("non-positive aqueous")
            continue
        xsf = xg / (xH + xg)
        if abs(xsf - zg) < 1e-3 * max(zg, 1e-6):
            seen.append(f"degenerate: x_sf==feed(route {route})")
            continue
        return None, seen                       # it actually worked
    # every rung exhausted; report the dominant reason
    tally = collections.Counter(s.split("(")[0] for s in seen)
    return tally.most_common(1)[0][0] if tally else "no attempt", seen


def main() -> int:
    t0 = time.perf_counter()
    fams = ds.build(split_yh2o=True, keep_unsplit=True)
    pool = par.start(fams)
    PUB = B.published_values(("A_Peni", "B_Peni", "A_U3i", "B_U3i",
                              "A_U2i", "B_U2i"))
    report = {}
    print("%-18s %6s %8s %8s   %s" % ("family", "N", "solved", "failed", "dominant reasons"))
    for name in GAS_FAMS + ("rho",):
        fam = fams[name]
        pred = rf.predicted(fam, "fortran", pool=pool, vals=PUB)
        bad = ~np.isfinite(pred)
        entry = dict(n=int(fam.n), failed=int(bad.sum()),
                     solved=int(fam.n - bad.sum()))
        reasons = collections.Counter()
        rows = []
        if bad.any() and name in GAS_FAMS:
            for i in np.where(bad)[0]:
                r = fam.df.iloc[i]
                why, trace = diagnose_gas(r, name)
                reasons[why or "unknown"] += 1
                rows.append(dict(T_K=float(r["T_K"]), P_bar=float(r["P_bar"]),
                                 source=str(r.get("source", "")), why=why))
        elif bad.any():
            for i in np.where(bad)[0]:
                r = fam.df.iloc[i]
                reasons["density solver: no liquid root"] += 1
                rows.append(dict(T_K=float(r["T_K"]), P_bar=float(r["P_bar"]),
                                 source=str(r.get("source", "")),
                                 why="density solver"))
        entry["reasons"] = dict(reasons)
        entry["rows"] = rows
        report[name] = entry
        top = ", ".join(f"{k} x{v}" for k, v in reasons.most_common(3)) or "-"
        print("%-18s %6d %8d %8d   %s" % (name, fam.n, entry["solved"],
                                          entry["failed"], top))
    pool.shutdown()

    tot_n = sum(v["n"] for k, v in report.items() if k in GAS_FAMS)
    tot_f = sum(v["failed"] for k, v in report.items() if k in GAS_FAMS)
    print(f"\n  flash path : {tot_n - tot_f} / {tot_n} solved "
          f"({100*(tot_n-tot_f)/tot_n:.3f} %)")
    print(f"  density    : {report['rho']['solved']} / {report['rho']['n']} solved "
          f"({100*report['rho']['solved']/report['rho']['n']:.3f} %)")

    # where do the failures sit?
    for name, v in report.items():
        if not v["rows"]:
            continue
        T = np.array([r["T_K"] for r in v["rows"]])
        P = np.array([r["P_bar"] for r in v["rows"]])
        print(f"\n  {name}: {len(T)} failures   "
              f"T {T.min():.0f}-{T.max():.0f} K   P {P.min():.1f}-{P.max():.1f} bar")
        srcs = collections.Counter(r["source"].split("/")[0].strip()
                                   for r in v["rows"])
        print("     by source: " + ", ".join(f"{k} x{n}" for k, n in srcs.most_common(4)))
    with open(f"{OUT}/census.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n  [saved] {OUT}/census.json      ({time.perf_counter()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
