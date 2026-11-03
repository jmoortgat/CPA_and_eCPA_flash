#!/usr/bin/env python3
"""Regenerate the paper's production figure set.

Each individual ``fig_*.py`` script knows its own cache; this orchestrator
just calls them in dependency order. Total runtime when caches are hot:
~5 seconds. Cold (first run, or after ``--force``): ~5 minutes for the
flash predictions, plus ~30 seconds for the TPD isotherm.

"Production set" means the figures of the published papers (see
``scripts/FIGURES_MAP.md``). One ``scripts/fig_*.py`` is deliberately
outside it -- see the note above ``FIG_SCRIPTS``.

After all figures are regenerated, the orchestrator also runs
``build_paper_stats_tex.py`` to consolidate every per-figure
``results_cache/*_stats.json`` into
``paper_tables/paper_stats.tex`` (macro definitions) and
``paper_tables/tables/aare_per_salt.tex``.

Cache-freshness safeguard: on entry, the orchestrator compares the mtime
of every ``results_cache/<script>_data.parquet`` against every file that
decides what that cache holds -- the model source (``ms_ecpa/*.py``, plus
the built Fortran library when ``MS_ECPA_BACKEND=fortran``) and the
experimental input (``data/*.parquet``) -- and against a stamp of the
environment knobs that select the experimental database. If any cache is
older than one of those sources, or was built against a different
database, AND ``--force`` was not passed, it aborts with a clear message
pointing at the stale caches. That covers "I refit but forgot --force",
and equally "I fixed flash.py" and "I switched to GasBrineBench", which
move every number without touching ``parameters.py``.

What the safeguard still cannot see: GasBrineBench revising itself under
a fixed ``ECPA_GBB`` (the env stamp is unchanged and no local mtime
moves). Catching that needs the database revision recorded inside the
cache at write time, i.e. in ``scripts/data_loader.py``.

Usage:
    python3 scripts/make_all_paper_figures.py             # use caches
    python3 scripts/make_all_paper_figures.py --force     # recompute everything
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
_CODE = _SCRIPTS.parent
_RESULTS_CACHE = _CODE / "results_cache"
_SOURCE_STAMP = _RESULTS_CACHE / "exp_source.json"

# The production set, in dependency order. scripts/fig_convention_comparison.py
# is intentionally absent: convention_regime.pdf and convention_gasrich.pdf are
# not figures of the published papers, and its
# conv_*_stats.json sidecars are read by neither build_paper_stats_tex.py (which
# loads six named files) nor build_paper_tables.py. It is an exploratory script,
# not paper output, and its four conv_* caches are outside the freshness check
# below for the same reason. If one of those PDFs is ever included in the paper,
# add the script here and its slugs to CACHE_SLUGS in the same edit.
FIG_SCRIPTS = [
    # Phase 1 figures (production set)
    "fig_flash_predictions.py",     # → 4 figures from one cache
    "fig_stability_isotherm.py",    # → 1 figure
    # Phase 2 figures (one family per script)
    "fig_brine_density.py",         # → main + SI
    "fig_colligative.py",           # → main + SI (osmotic coefficients)
    "fig_binaries.py",              # → main + SI (CO2-H2O + CH4-H2O)
    "fig_co2_singlesalt.py",        # → main + SI
    "fig_co2_mixed_brines.py",      # → main + SI
    "fig_ch4_singlesalt.py",        # → main + SI
    "fig_ch4_henry.py",             # → SI
    "fig_ccw_ternary.py",           # → main + SI
    "fig_ccb_ternary.py",           # → SI
    "fig_additional_gases.py",      # → main (N2 + H2 via pilot rebinds)
    # Phase 3: parameter-fit figures (analytic; no flash)
    "fig_param_fits.py",            # → 4 figures (kij, dU_CH4, dU_CO2, Peneloux)
]

# The cache slug per script -- must match load_or_compute(name, ...).
CACHE_SLUGS = {
    "fig_flash_predictions.py":  "fig_flash_predictions",
    "fig_stability_isotherm.py": "fig_stability_isotherm",
    "fig_brine_density.py":      "fig_brine_density",
    "fig_colligative.py":        "fig_colligative",
    "fig_binaries.py":           "fig_binaries",
    "fig_co2_singlesalt.py":     "fig_co2_singlesalt",
    "fig_co2_mixed_brines.py":   "fig_co2_mixed_brines",
    "fig_ch4_singlesalt.py":     "fig_ch4_singlesalt",
    "fig_ch4_henry.py":          "fig_ch4_henry",
    "fig_ccw_ternary.py":        "fig_ccw_ternary",
    "fig_ccb_ternary.py":        "fig_ccb_ternary",
    "fig_additional_gases.py":   "fig_additional_gases",
    "fig_param_fits.py":         "fig_param_fits",
}

# build_paper_tables.py maintains its own two caches (not tied to a single
# fig_*.py). List them so the freshness check catches them under a stale
# source the same way.
EXTRA_CACHES = (
    "ch4_brine_per_author",
    "binaries_ch4_water_per_author",
)


def _model_sources() -> list[Path]:
    """Every file whose contents decide the numbers inside a cache.

    Not just ``parameters.py``: a cached prediction is the output of the
    whole pipeline, so an edit to ``flash.py``, ``stability.py``,
    ``eos.py``, ``lnphi.py`` or ``roots.py`` invalidates it exactly as a
    refit does. The experimental parquets belong here too -- they set the
    (T, P, composition) grid the predictions are computed on, and the
    table builders join live experiment against cached prediction, so a
    rebuilt parquet against an old cache mixes two datasets in one table.
    """
    srcs = sorted((_CODE / "ms_ecpa").glob("*.py"))
    srcs += sorted((_CODE / "data").glob("*.parquet"))
    if os.environ.get("MS_ECPA_BACKEND", "python").lower() == "fortran":
        # Only when the Fortran kernel is the one doing the arithmetic; a
        # rebuild says nothing about a cache the Python kernel computed.
        lib = os.environ.get("ECPA_LIB")
        cands = ([Path(lib)] if lib else
                 [_CODE / "fortran" / "build" / f"libecpa.{ext}"
                  for ext in ("dylib", "so")])
        srcs += cands
    return [p for p in srcs if p.exists()]


def _exp_source_selector() -> dict[str, str]:
    """The environment knobs that choose which experimental database is read.

    ``ECPA_EXP_SOURCE=legacy`` swaps GasBrineBench back for the parquets in
    ``data/``; ``ECPA_GBB`` points at a different checkout. Flipping either
    changes every AARE and leaves no file mtime behind, so the caches carry
    a stamp of these two values instead (see ``_source_stamp_mismatch``).
    """
    return {
        "ECPA_EXP_SOURCE": os.environ.get("ECPA_EXP_SOURCE", ""),
        "ECPA_GBB": os.environ.get("ECPA_GBB", ""),
    }


def _source_stamp_mismatch() -> str | None:
    """Why the caches belong to a different experimental database, or None.

    No stamp means no evidence either way -- caches predating the stamp are
    treated as fine, exactly as a missing cache is.
    """
    if not _SOURCE_STAMP.exists():
        return None
    try:
        recorded = json.loads(_SOURCE_STAMP.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(recorded, dict) or recorded == _exp_source_selector():
        return None

    def _fmt(d: dict) -> str:
        return ", ".join(f"{k}={v or '<unset>'}" for k, v in sorted(d.items()))

    return f"built under {_fmt(recorded)}; this run has {_fmt(_exp_source_selector())}"


def _write_source_stamp() -> None:
    """Record which experimental database this run's caches belong to.

    Written only after a full successful run, so it always describes the
    state of ``results_cache/`` as of the last time the orchestrator saw it
    through. Under ``--allow-stale`` that is the user's assertion that the
    caches are valid for this source, recorded as such.
    """
    if not _RESULTS_CACHE.exists():
        return
    try:
        _SOURCE_STAMP.write_text(
            json.dumps(_exp_source_selector(), indent=2, sort_keys=True) + "\n")
    except OSError as exc:
        # A stamp we cannot write costs a future guard, not this run.
        print(f"  (note: could not write {_SOURCE_STAMP.name}: {exc})",
              file=sys.stderr)


def _check_cache_freshness() -> list[tuple[str, str]]:
    """Return ``(cache file, why it is stale)`` for every stale cache.

    A cache that doesn't exist yet is NOT considered stale -- it'll be
    computed cleanly on first run. Only an existing cache that is older
    than some model/data source, or that was built against a different
    experimental database, is reported.
    """
    names = [f"{slug}_data.parquet" for slug in CACHE_SLUGS.values()]
    names += [f"{slug}.parquet" for slug in EXTRA_CACHES]
    present = [(n, _RESULTS_CACHE / n) for n in names]
    present = [(n, p) for n, p in present if p.exists()]
    if not present:
        return []

    switched = _source_stamp_mismatch()
    if switched is not None:
        # Every cache is suspect, whatever its mtime says.
        return [(n, switched) for n, _ in present]

    newest_mtime, newest_src = 0.0, None
    for src in _model_sources():
        m = src.stat().st_mtime
        if m > newest_mtime:
            newest_mtime, newest_src = m, src
    if newest_src is None:
        return []
    try:
        rel = str(newest_src.relative_to(_CODE))
    except ValueError:      # ECPA_LIB may point outside the tree
        rel = str(newest_src)
    return [(n, f"older than {rel}")
            for n, p in present if p.stat().st_mtime < newest_mtime]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--force", action="store_true",
                   help="Pass --force to each fig_*.py (recompute caches).")
    p.add_argument("--allow-stale", action="store_true",
                   help="Skip the cache-freshness check (source mtimes + database stamp).")
    args = p.parse_args()

    if not args.force and not args.allow_stale:
        stale = _check_cache_freshness()
        if stale:
            print("ERROR: the following caches are stale:", file=sys.stderr)
            for name, why in stale:
                print(f"  - results_cache/{name}  ({why})", file=sys.stderr)
            print(
                "\nIf you meant to use the current code and data, re-run with --force.\n"
                "If you know these caches are still valid, re-run with --allow-stale.",
                file=sys.stderr,
            )
            return 1

    for s in FIG_SCRIPTS:
        print(f"\n=== {s} ===")
        cmd = [sys.executable, str(_SCRIPTS / s)]
        if args.force:
            cmd.append("--force")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"  FAILED with exit code {r.returncode}", file=sys.stderr)
            return r.returncode

    # Build the three per-author inventory tables (Tables 3, 4, 5 in the
    # SI) via stability_flash. Runs BEFORE build_paper_stats_tex.py so the
    # Susak full-T AAD is available as a stats macro.
    print("\n=== build_paper_tables.py ===")
    bt_cmd = [sys.executable, str(_SCRIPTS / "build_paper_tables.py")]
    if args.force:
        bt_cmd.append("--force")
    r = subprocess.run(bt_cmd)
    if r.returncode != 0:
        print(f"  FAILED with exit code {r.returncode}", file=sys.stderr)
        return r.returncode

    # Consolidate all *_stats.json into paper_stats.tex + tables/aare_per_salt.tex.
    print("\n=== build_paper_stats_tex.py ===")
    r = subprocess.run([sys.executable, str(_SCRIPTS / "build_paper_stats_tex.py")])
    if r.returncode != 0:
        print(f"  FAILED with exit code {r.returncode}", file=sys.stderr)
        return r.returncode

    _write_source_stamp()
    print("\nAll figures + tables regenerated; paper_stats.tex updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
