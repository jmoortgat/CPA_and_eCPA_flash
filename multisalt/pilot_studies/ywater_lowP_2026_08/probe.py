"""Water content of the CO2-rich phase at LOW pressure.

At P -> 1 bar the water content of the gas approaches the Raoult
limit y_H2O ~ a_w * Psat_w(T) / P, which becomes large at warm
temperatures.  Probe: y_H2O(P) along isotherms down to 1 bar,
salt-free and 2m NaCl, with a two-feed consistency check to flag the
dew-line-degeneracy region.

Run from multisalt/: PYTHONPATH=. python pilot_studies/ywater_lowP_2026_08/probe.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

T_LIST = (298.15, 323.15, 348.15, 373.15)
P_LIST = (1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0)
SALTS = {"water": (0,) * 6, "NaCl2m": (2, 2, 0, 0, 0, 0)}


def run_one(task):
    T, P, mi, zg = task
    from ecpa_ms import stability_flash
    z = np.array([1.0 - zg, zg * 0.0, zg])  # water + CO2 only
    z = np.array([1.0 - zg, 0.0, zg])
    try:
        r = stability_flash(float(T), float(P), z, np.asarray(mi, float))
        if r.phases == 2 and r.flash_result is not None \
                and r.flash_result.converged:
            return float(r.flash_result.y_gas[0]), int(r.phases)
        return (np.nan, int(r.phases))
    except Exception:
        return (np.nan, -1)


def main():
    from multiprocessing import Pool
    from CoolProp.CoolProp import PropsSI
    tasks, keys = [], []
    for salt, mi in SALTS.items():
        for T in T_LIST:
            for P in P_LIST:
                for zg in (0.05, 0.30):
                    tasks.append((T, P, mi, zg))
                    keys.append((salt, T, P, zg))
    with Pool(8) as pool:
        out = pool.map(run_one, tasks)
    res = {}
    for k, v in zip(keys, out):
        res[k] = v
    print(f"{'salt':7s} {'T':>7s} {'P':>6s} {'y_H2O(5%)':>10s} "
          f"{'y_H2O(30%)':>10s} {'consist':>8s} {'Raoult':>8s}")
    import csv
    rows = []
    for salt in SALTS:
        for T in T_LIST:
            psat = PropsSI("P", "T", T, "Q", 0, "Water") / 1e5
            for P in P_LIST:
                a = res[(salt, T, P, 0.05)]
                b = res[(salt, T, P, 0.30)]
                ya, yb = a[0], b[0]
                cons = (np.isfinite(ya) and np.isfinite(yb)
                        and abs(ya / yb - 1) < 0.02)
                raoult = psat / P
                print(f"{salt:7s} {T:7.2f} {P:6.1f} {ya:10.4f} "
                      f"{yb:10.4f} {str(cons):>8s} {raoult:8.4f}")
                rows.append(dict(salt=salt, T_K=T, P_bar=P, y_lo=ya,
                                 y_hi=yb, consistent=cons,
                                 raoult=raoult))
    with open(Path(__file__).parent / "probe.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
