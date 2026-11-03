"""Parallel chunk runner for the multi-salt eCPA sweep.

Drives :func:`sweep_core.run_one` over a slice of the grid via
``multiprocessing.Pool`` and writes the per-chunk results to parquet.

Invocation
----------
::

    python -m sweep.sweep_runner \\
        --config smoke \\
        --chunk-id 0 --n-chunks 1 \\
        --workers 8 \\
        --out-dir results_cache/sweep_smoke/

In the SLURM array setup (see ``slurm/run_sweep.slurm``) each array task
calls this entry point with its own ``--chunk-id`` between 0 and
``--n-chunks`` − 1. Chunks are independent and writes go to separate
parquet files, so worker crashes affect at most one chunk and the rest
of the sweep keeps making progress.

The runner is intentionally robust against any single-point failure:
the worker captures exceptions as flat-dict rows with ``converged=False``
and a populated ``error_type`` / ``error_msg``, so the per-row coverage
of the grid is always 100% and downstream analysis can target failure
modes directly without rerunning the sweep.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
_CODE = _HERE.parents[1]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from sweep.sweep_config import iter_grid, estimate_size, SweepPoint  # noqa
from sweep.sweep_core import run_one, output_schema                  # noqa


def _slice_grid(config: str, chunk_id: int, n_chunks: int):
    """Yield only the SweepPoints that belong to ``chunk_id`` of
    ``n_chunks``. The grid generator iterates a deterministic order so
    static slicing is reproducible.
    """
    for i, pt in enumerate(iter_grid(config)):
        if (i % n_chunks) == chunk_id:
            yield pt


def _override_convention(pt: SweepPoint, convention: str) -> SweepPoint:
    """Return a copy of ``pt`` with ``salinity_convention`` replaced."""
    return SweepPoint(
        point_id=pt.point_id, config_name=pt.config_name,
        feed_family=pt.feed_family, salt_name=pt.salt_name,
        T_K=pt.T_K, P_bar=pt.P_bar,
        z_H2O=pt.z_H2O, z_CH4=pt.z_CH4, z_CO2=pt.z_CO2,
        m_Na=pt.m_Na, m_Cl=pt.m_Cl, m_K=pt.m_K,
        m_Ca=pt.m_Ca, m_Mg=pt.m_Mg, m_SO4=pt.m_SO4,
        salinity_convention=convention,
    )


def _gather_points(config: str, chunk_id: int, n_chunks: int,
                   convention: str) -> list[SweepPoint]:
    pts = []
    for pt in _slice_grid(config, chunk_id, n_chunks):
        if pt.salinity_convention != convention:
            pt = _override_convention(pt, convention)
        pts.append(pt)
    return pts


def _flush(rows: list[dict], out_path: Path, schema: tuple[str, ...]):
    """Write the accumulated rows to ``out_path`` (parquet, single file)."""
    df = pd.DataFrame(rows, columns=list(schema))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True,
                   choices=("smoke", "standard", "full", "valid", "valid_dense"))
    p.add_argument("--chunk-id", type=int, default=0,
                   help="Zero-based chunk index for this task.")
    p.add_argument("--n-chunks", type=int, default=1,
                   help="Total number of chunks the grid is split into.")
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1),
                   help="Worker processes per chunk (default: all cores).")
    p.add_argument("--convention", default="as-aqueous",
                   choices=("as-aqueous", "chain"),
                   help="salinity_convention forwarded to flash_kv.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--chunk-progress-every", type=int, default=2000,
                   help="Print progress every N points.")
    args = p.parse_args(argv)

    if args.chunk_id < 0 or args.chunk_id >= args.n_chunks:
        raise SystemExit(f"chunk-id {args.chunk_id} out of range [0, {args.n_chunks})")

    schema = output_schema()
    out_path = args.out_dir / f"chunk_{args.chunk_id:05d}_of_{args.n_chunks:05d}.parquet"
    meta_path = args.out_dir / f"chunk_{args.chunk_id:05d}.meta.json"

    points = _gather_points(args.config, args.chunk_id, args.n_chunks,
                            args.convention)
    if not points:
        print(f"[chunk {args.chunk_id}] empty slice; writing zero-row parquet")
        _flush([], out_path, schema)
        return 0

    print(f"[chunk {args.chunk_id}/{args.n_chunks}] config={args.config} "
          f"points={len(points)} workers={args.workers} "
          f"convention={args.convention}")
    grid_total = estimate_size(args.config)["total"]
    print(f"[chunk {args.chunk_id}] grid total = {grid_total}; this chunk "
          f"covers {100*len(points)/grid_total:.2f}%")

    t0 = time.perf_counter()
    # perf_counter has an undefined origin (time since boot here), so it is
    # usable only as a difference; the wall-clock start is captured separately
    # to keep started_utc on the same clock as finished_utc.
    t_start_utc = datetime.now(tz=timezone.utc)
    rows: list[dict] = []
    n_done = 0
    n_failed = 0
    chunksize = max(1, len(points) // (args.workers * 64))

    # Spawn fresh interpreters (the default 'fork' on Linux works too, but
    # 'spawn' is more portable and avoids shared-state surprises with
    # numpy/scipy threads).
    ctx = mp.get_context("spawn" if sys.platform != "linux" else "fork")
    with ctx.Pool(processes=args.workers, maxtasksperchild=512) as pool:
        for row in pool.imap_unordered(run_one, points, chunksize=chunksize):
            rows.append(row)
            n_done += 1
            if not row["converged"]:
                n_failed += 1
            if (n_done % args.chunk_progress_every) == 0:
                rate = n_done / (time.perf_counter() - t0)
                eta_s = (len(points) - n_done) / max(rate, 1e-6)
                print(f"[chunk {args.chunk_id}] {n_done}/{len(points)} "
                      f"({100*n_done/len(points):.1f}%)  rate={rate:.0f} pts/s "
                      f"  failures={n_failed} ({100*n_failed/n_done:.1f}%)  "
                      f"eta={eta_s/60:.1f} min")
    elapsed = time.perf_counter() - t0

    _flush(rows, out_path, schema)

    meta = dict(
        config=args.config, chunk_id=args.chunk_id, n_chunks=args.n_chunks,
        workers=args.workers, convention=args.convention,
        n_points=len(points), n_failed=n_failed,
        elapsed_s=elapsed,
        rate_pts_per_s=len(points) / max(elapsed, 1e-9),
        host=socket.gethostname(), pid=os.getpid(),
        started_utc=t_start_utc.isoformat(),
        finished_utc=datetime.now(tz=timezone.utc).isoformat(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID", ""),
        slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID", ""),
    )
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"[chunk {args.chunk_id}] DONE in {elapsed:.1f}s  "
          f"converged={len(points)-n_failed}/{len(points)}  "
          f"out={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
