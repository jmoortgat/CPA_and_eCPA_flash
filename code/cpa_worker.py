"""
cpa_worker.py — top-level worker for parallel CPA (salt-free CO₂ + H₂O)
phase-envelope computation.

Must live at the top level so ProcessPoolExecutor can pickle tasks.

Each task is a tuple:
    (T, z_co2, P_scan)

where P_scan is a list of pressures [bar] to scan for two-phase detection.

Usage
-----
from concurrent.futures import ProcessPoolExecutor, as_completed
import cpa2_worker

tasks = [(T, z_co2, P_scan) for T, z_co2 in grid]
with ProcessPoolExecutor(max_workers=N) as executor:
    futures = {executor.submit(cpa2_worker.compute_one_point, task): task
               for task in tasks}
    for future in as_completed(futures):
        result = future.result()
        # result keys: T, z_co2, P_lo, P_hi
"""

import numpy as np
import sys
import os

# CPA.py lives in the same directory as this worker
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import CPA  # noqa: E402


def compute_one_point(task: tuple) -> dict:
    """
    Scan pressures at (T, z_co2) and return P_lo (bubble) and P_hi (dew).

    Parameters
    ----------
    task : (T, z_co2, P_scan)
        T       : float — temperature [K]
        z_co2   : float — feed CO₂ mole fraction
        P_scan  : list[float] — pressures [bar] to probe, in ascending order

    Returns
    -------
    dict with keys: T, z_co2, P_lo, P_hi
        P_lo : lower phase-boundary pressure [bar] (bubble line), or nan
        P_hi : upper phase-boundary pressure [bar] (dew/miscibility), or nan
    """
    T, z_co2, P_scan = task

    two_phase = []
    for P in sorted(P_scan):
        try:
            res = CPA.flash_co2_h2o_tpz(T=float(T), P_bar=float(P),
                                          z_co2=float(z_co2))
            converged = res["tie"]["converged"]
        except Exception:
            converged = False
        two_phase.append((P, converged))

    P_lo = np.nan
    P_hi = np.nan

    # Find lowest two-phase P and highest two-phase P
    tp_pressures = [P for P, ok in two_phase if ok]
    if tp_pressures:
        P_tp_min = min(tp_pressures)
        P_tp_max = max(tp_pressures)

        # P_lo: boundary below the two-phase region
        below_single = [P for P, ok in two_phase if not ok and P < P_tp_min]
        if below_single:
            P_lo = 0.5 * (max(below_single) + P_tp_min)
        else:
            P_lo = P_tp_min  # scan starts already in two-phase region

        # P_hi: boundary above the two-phase region
        above_single = [P for P, ok in two_phase if not ok and P > P_tp_max]
        if above_single:
            P_hi = 0.5 * (P_tp_max + min(above_single))
        # else: two-phase extends to top of scan range → P_hi = nan (> scan range)

    return dict(T=float(T), z_co2=float(z_co2), P_lo=P_lo, P_hi=P_hi)
