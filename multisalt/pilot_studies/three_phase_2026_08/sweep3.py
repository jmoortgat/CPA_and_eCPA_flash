"""Modest three-phase sweep: stability_flash3 across the VLLE window.

Grid: T in [283.15, 303.15] K (calibration floor up to just below the
CO2 critical point), P in [40, 80] bar — brackets the three-phase
locus (44.8 bar at 283 K to ~70 bar at 302 K) — five feeds (binary
CO2 + four CH4-bearing ternaries) and five salt configurations.
Every point runs the full hierarchical driver; the worker never
raises.  Records phase count, convergence, stage-2 TPD, phase
fractions, cross-phase isofugacity residuals, and split timings.

Outputs: sweep3.parquet + a console summary.

Run from ``code/``:
    PYTHONPATH=. python pilot_studies/three_phase_2026_08/sweep3.py
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

HERE = Path(__file__).resolve().parent

T_GRID = [283.15, 288.15, 293.15, 298.15, 303.15]
P_GRID = [float(p) for p in np.arange(40.0, 80.0 + 1e-9, 2.0)]
FEEDS = [
    ("binary_CO2", (0.50, 0.00, 0.50)),
    ("tern_c1_01", (0.50, 0.01, 0.49)),
    ("tern_c1_03", (0.50, 0.03, 0.47)),
    ("tern_c1_06", (0.50, 0.06, 0.44)),
    ("tern_c1_10", (0.50, 0.10, 0.40)),
]
SALTS = [
    ("no_salt", (0, 0, 0, 0, 0, 0)),
    ("NaCl_1m", (1, 1, 0, 0, 0, 0)),
    ("NaCl_3m", (3, 3, 0, 0, 0, 0)),
    ("KCl_1m", (0, 1, 1, 0, 0, 0)),
    ("CaCl2_1m", (0, 2, 0, 1, 0, 0)),
]


def _isofug(f_ref, f_other, z):
    pres = np.asarray(z) > 0
    fa = np.asarray(f_ref)[pres]
    fo = np.asarray(f_other)[pres]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.abs(fo / fa - 1.0)
    return float(np.max(r)) if np.all(np.isfinite(r)) else float("nan")


def run_one(task):
    feed_name, z, salt_name, mi, T, P = task
    from ecpa_ms import stability_flash3
    row = dict(feed=feed_name, salt=salt_name, T_K=T, P_bar=P,
               z_H2O=z[0], z_CH4=z[1], z_CO2=z[2],
               phases=-1, ok=False, tpd3=np.nan, root="",
               beta_g=np.nan, beta_l=np.nan,
               isofug_g=np.nan, isofug_l=np.nan,
               n_ssi3=0, n_newton3=0,
               t_total_ms=np.nan, msg="", err="")
    t0 = time.perf_counter()
    try:
        r = stability_flash3(T, P, np.asarray(z, float),
                             np.asarray(mi, float))
        row["phases"] = int(r.phases)
        row["tpd3"] = float(r.tpd3_min)
        row["root"] = r.incumbent_gas_root
        row["msg"] = r.message[:80]
        if r.phases == 3:
            f3 = r.flash3
            row["ok"] = bool(f3.converged)
            row["beta_g"] = f3.beta_g
            row["beta_l"] = f3.beta_l
            row["isofug_g"] = _isofug(f3.fugacities_aq,
                                      f3.fugacities_gas, z)
            row["isofug_l"] = _isofug(f3.fugacities_aq,
                                      f3.fugacities_lco2, z)
            row["n_ssi3"] = int(f3.n_iter_ssi)
            row["n_newton3"] = int(f3.n_iter_newton)
        elif r.phases == 2:
            fr = r.two_phase.flash_result
            row["ok"] = bool(fr is not None and fr.converged)
            if fr is not None:
                row["beta_g"] = fr.beta
                row["isofug_g"] = _isofug(fr.fugacities_aq,
                                          fr.fugacities_gas, z)
        else:
            row["ok"] = True  # classified single-phase
    except Exception as exc:  # noqa: BLE001 — sweep must not die
        row["err"] = f"{type(exc).__name__}: {exc}"[:160]
    row["t_total_ms"] = (time.perf_counter() - t0) * 1e3
    return row


def main():
    import pandas as pd
    from multiprocessing import Pool

    tasks = [(fn, z, sn, mi, T, P)
             for fn, z in FEEDS for sn, mi in SALTS
             for T in T_GRID for P in P_GRID]
    print(f"{len(tasks)} points on {os.cpu_count()} cores ...")
    t0 = time.perf_counter()
    with Pool() as pool:
        rows = pool.map(run_one, tasks, chunksize=8)
    dt = time.perf_counter() - t0
    df = pd.DataFrame(rows)
    df.to_parquet(HERE / "sweep3.parquet")
    print(f"done in {dt:.1f} s -> {HERE / 'sweep3.parquet'}")

    # ── Summary ──────────────────────────────────────────────────────────
    n = len(df)
    print(f"\npoints: {n}")
    print("phase counts:", df.phases.value_counts().sort_index().to_dict())
    print(f"errors: {(df.err != '').sum()}")
    print(f"not-ok rows: {(~df.ok).sum()}")
    three = df[df.phases == 3]
    if len(three):
        print(f"\nthree-phase states: {len(three)}")
        print(f"  isofug residual: max {three[['isofug_g', 'isofug_l']].max().max():.2e}")
        print(f"  SSI iters: p50 {three.n_ssi3.median():.0f}, "
              f"max {three.n_ssi3.max()}")
        print(f"  per-salt counts: "
              f"{three.groupby('salt').size().to_dict()}")
        print(f"  per-feed counts: "
              f"{three.groupby('feed').size().to_dict()}")
    print(f"\nt_total_ms: p50 {df.t_total_ms.median():.0f}, "
          f"p95 {df.t_total_ms.quantile(0.95):.0f}, "
          f"max {df.t_total_ms.max():.0f}")
    bad = df[(df.err != "") | (~df.ok)]
    if len(bad):
        print("\nWORST 15 problem rows:")
        cols = ["feed", "salt", "T_K", "P_bar", "phases", "msg", "err"]
        print(bad[cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
