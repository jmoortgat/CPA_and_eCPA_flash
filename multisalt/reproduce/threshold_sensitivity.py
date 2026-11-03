"""Sensitivity of the phase classification to the hard TPD threshold (Paper III, Table S15).

1. Reclassifies the recorded tm* (tpd_min) of every state of the flash grid
   (results_cache/sweep/merged.parquet) under hard thresholds scaled about the
   production value -0.005, and counts the states whose classification changes.
2. With --rerun: re-runs every state with -0.005 < tm* < 0 through
   stability_flash with the hard threshold set to 0, so that each is passed to
   the flash, and records the outcomes (results_cache/sweep/threshold_rerun.csv).

Usage (from multisalt/):
    PYTHONPATH=. python3 reproduce/threshold_sensitivity.py [--rerun]
Writes results_cache/sweep/threshold_sensitivity.json.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SWEEP = ROOT / "results_cache" / "sweep" / "merged.parquet"
OUT = ROOT / "results_cache" / "sweep" / "threshold_sensitivity.json"
RERUN = ROOT / "results_cache" / "sweep" / "threshold_rerun.csv"
PROD = -0.005
SCALED = [("zero", 0.0), ("div10", -0.0005), ("div5", -0.001), ("div2", -0.0025),
          ("production", PROD), ("x2", -0.01), ("x4", -0.02), ("x5", -0.025),
          ("x10", -0.05)]


def crossing(d, t):
    """States whose classification by the hard threshold changes.

    Hard rule: single-phase (no flash) if tm* > threshold.
    Tightening (t > PROD): states with PROD < tm* <= t (t < 0) or
    PROD < tm* < 0 (t = 0, strict: exact zeros are the trivial trial).
    Loosening (t < PROD): states with t < tm* <= PROD.
    """
    tm = d.tpd_min
    if t == PROD:
        return pd.Series(False, index=d.index)
    if t > PROD:
        return (tm > PROD) & ((tm < 0) if t == 0 else (tm <= t))
    return (tm > t) & (tm <= PROD)


def table(d):
    out = {}
    for name, t in SCALED:
        m = crossing(d, t)
        out[name] = dict(threshold=t, n=int(m.sum()), pct=100 * m.sum() / len(d),
                         n_two_phase=int((d.phases[m] == 2).sum()),
                         n_single_phase=int((d.phases[m] == 1).sum()))
    return out


def run_one(args):
    from ms_ecpa import stability_flash
    idx, T, P, z, mi, conv = args
    t0 = time.perf_counter()
    try:
        r = stability_flash(T, P, np.array(z), np.array(mi),
                            tpd_threshold=0.0, salinity_convention=conv)
        fr = r.flash_result
        return dict(idx=idx, phases=int(r.phases), route=int(r.route),
                    message=r.message, tpd_min=float(r.tpd_min),
                    flash_converged=(bool(fr.converged) if fr is not None else None),
                    beta=(float(fr.beta) if fr is not None else None),
                    max_residual=(float(fr.max_residual) if fr is not None else None),
                    t_ms=1e3 * (time.perf_counter() - t0), error=None)
    except Exception as exc:  # recorded as a failure, never dropped
        return dict(idx=idx, phases=None, route=None, message=None,
                    tpd_min=None, flash_converged=None, beta=None,
                    max_residual=None, t_ms=1e3 * (time.perf_counter() - t0),
                    error=repr(exc)[:300])


def summarize_rerun(rr, sub):
    two = rr.phases.eq(2) & rr.flash_converged.eq(True)
    return dict(
        n=len(rr), two_phase_converged=int(two.sum()),
        single_phase=int(rr.phases.eq(1).sum()),
        failed=int(len(rr) - two.sum() - rr.phases.eq(1).sum()),
        exceptions=int(rr.error.notna().sum()),
        beta_min_two_phase=(float(rr.beta[two].min()) if two.any() else None),
        beta_max_two_phase=(float(rr.beta[two].max()) if two.any() else None),
        max_residual_two_phase=(float(rr.max_residual[two].max()) if two.any() else None),
        T_range_K=[float(sub.T_K.min()), float(sub.T_K.max())])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true",
                    help="re-run the -0.005 < tm* < 0 states with threshold 0")
    args = ap.parse_args()
    d = pd.read_parquet(SWEEP)
    res = {"n_grid": len(d), "n_two_phase": int((d.phases == 2).sum()),
           "n_single_phase": int((d.phases == 1).sum()), "table": table(d)}
    sub = d[(d.tpd_min > PROD) & (d.tpd_min < 0)]
    if args.rerun:
        jobs = [(int(i), float(r.T_K), float(r.P_bar), [r.z_H2O, r.z_CH4, r.z_CO2],
                 [r.m_Na, r.m_Cl, r.m_K, r.m_Ca, r.m_Mg, r.m_SO4], r.salinity_convention)
                for i, r in sub.iterrows()]
        with Pool(int(os.environ.get("NWORK", 8))) as pool:
            rr = pd.DataFrame(pool.map(run_one, jobs, chunksize=1))
        rr.to_csv(RERUN, index=False)
    if RERUN.exists():
        res["rerun_threshold0"] = summarize_rerun(pd.read_csv(RERUN), sub)
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    for name, row in res["table"].items():
        print(f"{name:11s} {row['threshold']:8.4f} {row['n']:5d} {row['pct']:6.3f} %")
    if "rerun_threshold0" in res:
        print("rerun:", res["rerun_threshold0"])


if __name__ == "__main__":
    main()
