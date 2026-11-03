"""Stage-split timing probe for stability_flash3 over the VLLE window.

Captures what sweep3.py did not: the per-stage wall times --- the
two-phase driver's t_stab/t_flash (stage 1), the third-phase TPD
trials (stage 2), and the three-phase flash (stage 3) --- on a
representative subsample of the CO2 sweep grid.

Run from multisalt/: PYTHONPATH=. python pilot_studies/three_phase_2026_08/timing_probe.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

T_GRID = [283.15, 288.15, 293.15, 298.15, 303.15]
P_GRID = [float(p) for p in np.arange(40.0, 80.0 + 1e-9, 4.0)]
FEEDS = [(0.50, 0.00, 0.50), (0.50, 0.03, 0.47), (0.50, 0.06, 0.44)]
SALTS = [(0, 0, 0, 0, 0, 0), (1, 1, 0, 0, 0, 0), (0, 2, 0, 1, 0, 0)]


def run_one(task):
    z, mi, T, P = task
    from ecpa_ms import stability_flash3
    row = dict(T=T, P=P, phases=-1,
               t_stab=np.nan, t_flash=np.nan,
               t_stab3=np.nan, t_flash3=np.nan, t_total=np.nan,
               n_ssi3=0)
    try:
        r = stability_flash3(T, P, np.asarray(z, float),
                             np.asarray(mi, float))
        row["phases"] = r.phases
        row["t_stab"] = r.two_phase.t_stab_ms
        row["t_flash"] = r.two_phase.t_flash_ms
        row["t_stab3"] = r.t_stab3_ms
        row["t_flash3"] = r.t_flash3_ms
        row["t_total"] = r.t_total_ms
        if r.phases == 3:
            row["n_ssi3"] = r.flash3.n_iter_ssi
    except Exception:
        pass
    return row


def main():
    import pandas as pd
    from multiprocessing import Pool
    tasks = [(z, mi, T, P) for z in FEEDS for mi in SALTS
             for T in T_GRID for P in P_GRID]
    print(f"{len(tasks)} probe points ...")
    with Pool(8) as pool:
        rows = pool.map(run_one, tasks, chunksize=4)
    df = pd.DataFrame(rows)
    df.to_csv(Path(__file__).parent / "timing_probe.csv", index=False)
    ok = df[df.phases > 0]
    print(f"converged {len(ok)}/{len(df)}")
    for col in ("t_stab", "t_flash", "t_stab3", "t_flash3", "t_total"):
        v = ok[col].dropna()
        print(f"{col:9s}: p50 {v.median():7.0f}  p95 {v.quantile(.95):7.0f} ms")
    three = ok[ok.phases == 3]
    two = ok[ok.phases == 2]
    print(f"\n3-phase states: {len(three)}; their t_flash3 p50 "
          f"{three.t_flash3.median():.0f} ms (n_ssi3 p50 "
          f"{three.n_ssi3.median():.0f})")
    print(f"2-phase states: stage-2 overhead t_stab3 p50 "
          f"{two.t_stab3.median():.0f} ms vs stage-1 t_stab p50 "
          f"{two.t_stab.median():.0f} ms, t_flash p50 "
          f"{two.t_flash.median():.0f} ms")


if __name__ == "__main__":
    main()
