"""Evaluate the fitting objective across a compute node's cores.

Why this exists
---------------
One evaluation of the objective costs ~41 s on a single core: every row of
every family is an independent flash, and there are 6,016 of them. The
optimiser needs 13 evaluations per iteration (one objective + a 12-parameter
finite-difference Jacobian), so a staged refit runs ~7 h serially and the
six-point Pareto scan on top of it runs ~40 h. On one 48-core node the same
work takes roughly 15 min and 1.5 h.

What is and is not independent
------------------------------
Solubility and y_H2O rows ARE independent -- each is a cold flash at its own
(T, P, brine) -- so they can be cut into arbitrary chunks.

Brine density rows are NOT. ``predict.predict_rho_family`` walks a dense
pressure grid carrying the inner Newton state, because the 3-variable solve
has several roots and a cold start lands on the wrong one for a large
minority of states (4.78 % AAD cold against 1.74 % warm, on the same rows).
Cutting that walk across workers would silently change the answer. So the
density family is cut ALONG ITS OWN GROUP BOUNDARIES -- one task per
(salt composition, isotherm) -- which are independent of each other by
construction, and each worker runs the identical traversal on its group.

Parameters are sent with every task and rebound in the worker before it
predicts. Binding is twelve ``set_param`` calls and costs nothing measurable;
caching it against a "have the parameters changed?" check would buy nothing
and would introduce exactly the kind of stale-state bug this project has
already hit six times.

Workers are forked, so they inherit the assembled dataframes rather than
pickling 6,016 rows per task. That makes fork REQUIRED, not a preference:
under spawn each worker would re-read GasBrineBench on startup.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import blocks as B
from . import predict as pr

#: Set in the parent before the pool is forked; inherited by every worker.
_FAMS: dict | None = None

_ION_COLS = ["m_Na", "m_Cl", "m_K", "m_Ca", "m_Mg", "m_SO4"]


def n_workers(requested=None) -> int:
    if requested:
        return max(1, int(requested))
    env = os.environ.get("MS_ECPA_WORKERS")
    if env:
        return max(1, int(env))
    try:
        # Respects the cpuset SLURM hands the job; os.cpu_count() reports the
        # whole node and would oversubscribe by 48/32 on a shared allocation.
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def start(fams: dict, workers=None):
    """Fork a worker pool that has ``fams`` already in memory."""
    global _FAMS
    _FAMS = fams
    n = n_workers(workers)
    if n <= 1:
        return None
    ctx = mp.get_context("fork")
    return ProcessPoolExecutor(max_workers=n, mp_context=ctx)


def rho_group_indices(df) -> list[np.ndarray]:
    """Row indices of each (salt composition, isotherm) warm-start group.

    Must reproduce the grouping inside ``predict.predict_rho_family`` exactly,
    or a task would hold a partial walk and get different numbers.
    """
    d = df.reset_index(drop=True)
    key = d[_ION_COLS].fillna(0.0).round(6).astype(str).agg("|".join, axis=1)
    return [np.asarray(sorted(idx), dtype=int)
            for _, idx in d.groupby([key, d["T_K"]]).groups.items()]


def _chunks(n: int, k: int) -> list[np.ndarray]:
    if n == 0:
        return []
    return [c for c in np.array_split(np.arange(n), max(1, min(k, n)))
            if len(c)]


def tasks_for(fam, n_workers_: int) -> list[np.ndarray]:
    """Cut one family into units of work that are independent of each other."""
    if fam.name == "rho":
        return rho_group_indices(fam.df)
    # Several chunks per worker, so a slow chunk cannot leave a core idle for
    # the rest of the evaluation.
    return _chunks(fam.n, 4 * n_workers_)


def _predict_chunk(name, idx, vals, backend):
    """Runs IN THE WORKER: bind the trial parameters, predict these rows."""
    B.bind_python(vals)
    if backend == "fortran":
        B.bind_fortran(pr._engine(), vals)
    sub = _FAMS[name].df.iloc[idx]
    if name == "rho":
        return idx, pr.predict_rho_family(sub, backend)
    return idx, np.array([pr.predict_family(name, r, backend)
                          for _, r in sub.iterrows()], dtype=float)


def predict_family_parallel(fam, vals, backend, pool) -> np.ndarray:
    """Same array as ``refit.predicted``, assembled from the pool."""
    nw = getattr(pool, "_max_workers", 1)
    units = tasks_for(fam, nw)
    out = np.full(fam.n, np.nan)
    futs = [pool.submit(_predict_chunk, fam.name, idx, vals, backend)
            for idx in units]
    for f in futs:
        idx, vals_ = f.result()
        out[idx] = vals_
    return out
