"""Post-process the multi-salt sweep output.

Merges per-chunk parquets into a single dataset, computes summary
statistics, and emits a focused 'failure inventory' so the next pass
can target the conditions where the model currently does not converge.

Usage
-----
::

    python -m sweep.sweep_postprocess \\
        --in-dir results_cache/sweep_smoke/ \\
        --out-dir results_cache/sweep_smoke_summary/ \\
        [--config valid_dense] [--allow-incomplete]

Outputs
-------
* ``merged.parquet``      — all chunks concatenated
* ``summary.json``        — overall convergence + timing stats
* ``failures.parquet``    — rows where ``converged == False``, sorted by
                            (feed_family, salt_name, T_K, P_bar)
* ``failure_modes.csv``   — counts grouped by (error_type, feed_family,
                            salt_name); the natural starting point for
                            the next iteration on robustness

Completeness
------------
Every chunk filename written by ``sweep_runner`` carries both the chunk
index and the array size (``chunk_00007_of_00064.parquet``), so a complete
sweep is distinguishable from a truncated one. The merge refuses an
incomplete or mixed-generation chunk set unless ``--allow-incomplete`` is
given, and with ``--config`` it additionally checks the merged row count
against the grid size. What it found either way is recorded in
``summary.json``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve()
_CODE = _HERE.parents[1]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))


_CHUNK_RE = re.compile(r"chunk_(\d+)_of_(\d+)\.parquet$")


def _fail_or_warn(msg: str, allow: bool) -> None:
    if allow:
        print(f"WARNING: {msg}")
    else:
        raise SystemExit(f"{msg}\n(pass --allow-incomplete to merge anyway)")


def _chunk_set_status(chunks: list[Path]) -> tuple[int, list[int]]:
    """Return ``(n_expected, missing_chunk_ids)`` for a globbed chunk set.

    ``sweep_runner`` flushes its rows only once the worker pool has drained,
    so an array task killed by the wall clock leaves no parquet at all: the
    points it held are an absence, not failed rows, and a plain glob would
    quietly take every count below over the survivors. ``run_postprocess``
    is documented to launch with ``--dependency=afterany``, i.e. explicitly
    after failed array tasks, so the absence is the expected case rather
    than a corner one. Parsing the ``_of_NNNNN`` suffix also separates two
    sweep generations run into the same out-dir with different
    ``--n-chunks``: both match the glob and would otherwise be concatenated.
    """
    by_total: dict[int, list[int]] = {}
    for c in chunks:
        m = _CHUNK_RE.search(c.name)
        if m is None:
            raise SystemExit(
                f"cannot parse a chunk index from {c.name}; expected "
                "chunk_<id>_of_<n_chunks>.parquet as written by sweep_runner"
            )
        by_total.setdefault(int(m.group(2)), []).append(int(m.group(1)))
    if len(by_total) > 1:
        raise SystemExit(
            "chunk files from more than one sweep generation share "
            f"this directory (--n-chunks values {sorted(by_total)}); "
            "concatenating them would double-count grid points — point "
            "--in-dir at a single run"
        )
    n_expected, ids = next(iter(by_total.items()))
    missing = sorted(set(range(n_expected)) - set(ids))
    return n_expected, missing


def _q(s: pd.Series, q: float) -> float | None:
    """Quantile of ``s``, or ``None`` when the subset is empty."""
    return float(s.quantile(q)) if len(s) else None


def _imax(s: pd.Series) -> int | None:
    """Max of an integer counter, or ``None`` when the subset is empty."""
    return int(s.max()) if len(s) else None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--config", default=None,
                   help="sweep config name the chunks came from (e.g. "
                        "valid_dense); when given, the merged row count is "
                        "checked against the grid size")
    p.add_argument("--allow-incomplete", action="store_true",
                   help="downgrade the completeness checks from a hard "
                        "failure to a warning (for inspecting a sweep that "
                        "is still running)")
    args = p.parse_args(argv)

    chunks = sorted(args.in_dir.glob("chunk_*.parquet"))
    if not chunks:
        raise SystemExit(f"no chunks found under {args.in_dir}")

    n_expected, missing = _chunk_set_status(chunks)
    print(f"merging {len(chunks)} of {n_expected} chunks from {args.in_dir}")
    if missing:
        shown = ", ".join(str(i) for i in missing[:10])
        if len(missing) > 10:
            shown += ", ..."
        _fail_or_warn(
            f"incomplete chunk set: {len(missing)} of {n_expected} chunks "
            f"never wrote a parquet (ids {shown}); their grid points would "
            "be missing from the row count, from pct_converged and from the "
            "failure inventory without appearing anywhere as failures",
            args.allow_incomplete)

    df = pd.concat([pd.read_parquet(c) for c in chunks], ignore_index=True)
    print(f"total rows: {len(df)}; converged: {int(df['converged'].sum())}")

    expected_rows = None
    if args.config is not None:
        # Imported here so the merge still runs when only pandas is
        # available and no --config cross-check was asked for.
        from sweep.sweep_config import estimate_size
        expected_rows = int(estimate_size(args.config)["total"])
        if len(df) != expected_rows:
            _fail_or_warn(
                f"merged row count {len(df)} does not match the "
                f"{args.config!r} grid size {expected_rows}",
                args.allow_incomplete)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_dir / "merged.parquet", index=False)

    # ── Overall summary ───────────────────────────────────────────────
    converged = df[df["converged"]]
    failed = df[~df["converged"]]
    # Iteration counters are initialised to 0 by sweep_core._empty_row and
    # stay there on every path that returns that template untouched: a
    # backend refusal, either exception handler, and the single-phase
    # return (converged, but no flash was run). Over the full frame the
    # quantiles below would therefore count those rows as zero-iteration
    # solves. Scope them to converged two-phase states — the same
    # denominator make_sweep_figures uses when it quotes this metric.
    twophase = converged[converged["phases"] == 2]
    summary = dict(
        n_chunks_found=int(len(chunks)),
        n_chunks_expected=int(n_expected),
        chunks_missing=missing,
        n_rows=int(len(df)),
        n_rows_expected=expected_rows,
        n_converged=int(len(converged)),
        n_failed=int(len(failed)),
        pct_converged=100.0 * len(converged) / max(len(df), 1),
        # Timing is over every row: a point that failed still spent the
        # wall time, so it belongs in the sweep's cost.
        wall_ms_p50=float(df["t_wall_ms"].median()),
        wall_ms_p95=float(df["t_wall_ms"].quantile(0.95)),
        wall_ms_p99=float(df["t_wall_ms"].quantile(0.99)),
        wall_ms_max=float(df["t_wall_ms"].max()),
        n_iter_basis="converged & phases == 2",
        n_iter_n_rows=int(len(twophase)),
        n_iter_ssi_p50=_q(twophase["n_iter_ssi"], 0.50),
        n_iter_ssi_p95=_q(twophase["n_iter_ssi"], 0.95),
        n_iter_ssi_max=_imax(twophase["n_iter_ssi"]),
        n_iter_inner_aq_p95=_q(twophase["n_iter_aq_inner_diag"], 0.95),
        n_iter_inner_gas_p95=_q(twophase["n_iter_gas_inner_diag"], 0.95),
    )
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # ── Failure inventory ─────────────────────────────────────────────
    if len(failed):
        failed_sorted = failed.sort_values(
            ["feed_family", "salt_name", "T_K", "P_bar"]
        )
        failed_sorted.to_parquet(args.out_dir / "failures.parquet", index=False)
        modes = (failed_sorted
                 .groupby(["error_type", "feed_family", "salt_name"])
                 .size()
                 .reset_index(name="n_failures")
                 .sort_values("n_failures", ascending=False))
        modes.to_csv(args.out_dir / "failure_modes.csv", index=False)
        print("\nTop failure modes:")
        print(modes.head(20).to_string(index=False))
    else:
        print("\n100% convergence on this sweep — no failure inventory written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
