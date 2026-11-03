"""Data-caching helpers for paper-figure scripts.

Implements the pattern the user codified: every figure script splits into
**compute -> save -> load -> plot**. Default behaviour is "if the cached
data file exists, skip recomputation"; a ``--force`` flag (or
``force=True`` argument) bypasses the cache.

Two distinct caches:

1. ``code/data/`` -- **experimental input data** consolidated from
   ``$ECPA_SOURCE_MATERIALS/EoS/.../EXP_*.txt``. Built once by
   ``scripts/build_experimental_parquets.py``. Tracked in git.

2. ``code/results_cache/`` -- **per-figure model outputs** (flash predictions,
   stability sweeps, etc.). Tracked in git so a reader cloning the repo can
   regenerate every figure in seconds without re-running the slow flash.

``data/`` is tracked in git; ``results_cache/`` holds model output.

In addition, each figure script that computes error metrics writes a small
``results_cache/<name>_stats.json`` sidecar via :func:`save_stats`. A separate
aggregator (``scripts/build_paper_stats_tex.py``) consumes these JSON files
and emits a single ``paper_tables/paper_stats.tex`` file of ``\\newcommand``
definitions for the numbers cited in the papers, so that no number is typed
by hand.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

_CODE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = _CODE_DIR / "data"
RESULTS_CACHE = _CODE_DIR / "results_cache"


def load_experimental(kind: str) -> pd.DataFrame:
    """Load consolidated experimental data, e.g. ``load_experimental("co2_brines")``.

    Files live under ``code/data/<kind>.parquet`` and are built by
    ``scripts/build_experimental_parquets.py``. The function gives a clear
    error if the parquet is missing, telling the user to run the builder.
    """
    if _use_gasbrinebench(kind):
        import gasbrinebench_source as _gbb
        return _gbb.ADAPTERS[kind]()

    path = DATA_DIR / f"{kind}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Experimental parquet '{path}' is missing. "
            f"Run `python3 scripts/build_experimental_parquets.py` once to build it."
        )
    return pd.read_parquet(path)


def _use_gasbrinebench(kind: str) -> bool:
    """Whether this kind is served from GasBrineBench rather than a parquet.

    The released database is the source of record as of 2026-09-18. Two kinds
    stay local by design:

    * ``ch4_henry`` -- Henry constants are derived from Henry's law, whose
      limited validity is why we build an equation of state at all. They are
      a figure input and must never be a fit or validation target.
    * ``ccb_ternary`` -- no ternary measurement in brine exists anywhere.

    ``ECPA_EXP_SOURCE=legacy`` forces the old parquets, for A/B comparison.
    """
    import os
    if os.environ.get("ECPA_EXP_SOURCE", "").lower() == "legacy":
        return False
    import sys
    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import gasbrinebench_source as _gbb
    except ImportError:
        return False
    return kind in _gbb.ADAPTERS and _gbb.available()


def load_or_compute(
    name: str,
    compute_fn: Callable[[], pd.DataFrame],
    *,
    force: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Return ``results_cache/<name>_data.parquet`` if present; else compute,
    save, return.

    Parameters
    ----------
    name : str
        Cache slug, e.g. ``"fig_flash_parity"``.
    compute_fn : callable () -> pd.DataFrame
        Slow compute step. Will only be invoked if the cache is missing
        or ``force=True``.
    force : bool, default False
        If True, recompute even when the cache file exists. (CLI: ``--force``.)
    verbose : bool, default True
        Print a brief message indicating whether the cache hit.
    """
    RESULTS_CACHE.mkdir(parents=True, exist_ok=True)
    path = RESULTS_CACHE / f"{name}_data.parquet"
    if path.exists() and not force:
        if verbose:
            print(f"  [cache] {path.name}  ({path.stat().st_size/1024:.1f} kB)")
        return pd.read_parquet(path)
    if verbose:
        action = "force-recompute" if force else "compute (no cache)"
        print(f"  [{action}] {path.name}")
    df = compute_fn()
    df.to_parquet(path)
    if verbose:
        print(f"  [saved] {path.name}  ({path.stat().st_size/1024:.1f} kB)")
    return df


def save_stats(name: str, stats: dict[str, Any], *, verbose: bool = True) -> Path:
    """Write ``results_cache/<name>_stats.json`` with a JSON-serializable dict.

    The keys of ``stats`` are consumed by ``scripts/build_paper_stats_tex.py``
    and become ``\\newcommand`` definitions in ``paper_stats.tex``. Values
    should be plain scalars (floats, ints, strings) or short lists; anything
    complex (per-salt breakdown, etc.) should be nested under a subkey.

    Convention: floats are rounded to 2 decimals on emit; if a specific
    field needs more precision it should be emitted as a pre-formatted
    string here.
    """
    RESULTS_CACHE.mkdir(parents=True, exist_ok=True)
    path = RESULTS_CACHE / f"{name}_stats.json"
    with path.open("w") as f:
        json.dump(stats, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    if verbose:
        print(f"  [stats] {path.name}  ({len(stats)} keys)")
    return path


def load_stats(name: str) -> dict[str, Any]:
    """Read ``results_cache/<name>_stats.json`` back into a dict.

    Used by ``scripts/build_paper_stats_tex.py`` to consolidate all per-script
    stats into a single LaTeX macro file. Returns an empty dict if the file
    is missing (so the aggregator can degrade gracefully when a script hasn't
    been run yet).
    """
    path = RESULTS_CACHE / f"{name}_stats.json"
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def add_cache_flag(parser) -> None:
    """Add a standard ``--force`` flag to an argparse parser.

    Any figure script that uses ``load_or_compute`` should add this flag::

        from data_loader import add_cache_flag, load_or_compute
        p = argparse.ArgumentParser(...)
        add_cache_flag(p)
        args = p.parse_args()
        df = load_or_compute("fig_x", compute, force=args.force)
    """
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute even if the cache file exists.",
    )
