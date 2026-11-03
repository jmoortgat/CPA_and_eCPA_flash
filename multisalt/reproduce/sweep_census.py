"""Convergence over the swept grid, counted by what actually happened.

``sweep_core`` records ``converged = (phases == 1)`` for every single-phase
outcome. That is right for a state the stability test declares stable, and
wrong for one where every flash attempt failed and the soft fallback then
called it single-phase: the column cannot tell the two apart, so a total
taken from it can only ever round upward.

So this script classifies each state from the message the sweep recorded,
rather than trusting that column. A soft fallback, a hard failure or an
inconclusive stability panel is a failure whatever ``converged`` says, and
any of them fails the run. A beta-degenerate state is a converged flash whose
Rachford-Rice found no interior split -- a genuine single-phase verdict, but
only if beta really sat at a bound, so that is checked rather than assumed.

The number Paper III can then state is not "converged" but the thing that is
actually true: every state either resolves into two phases or is classified
single-phase by a computation that converged.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

#: How a state is classified, from the free-text message the sweep records.
#:
#: The two backends write DIFFERENT things into ``solve_path``. The compiled
#: one writes a fixed route label ("soft fallback"); the Python one writes the
#: result message, which carries its numbers inline -- "single-phase
#: (tpd_min=-3.757e-03)" -- so every state is its own category and an exact
#: match against a table of route labels recognises nothing at all. An earlier
#: version of this script did exactly that: it found no failure labels, and
#: would have reported a clean grid because it could not read the column.
#:
#: So classification is by substring, and anything unrecognised is a FAILURE
#: rather than a pass. A census that cannot read its input must not be the
#: reason a convergence claim looks good.
FAILURE_MARKS = ("soft fallback", "hard failure", "inconclusive",
                 "stability error", "not converged", "exception")
DEGENERATE_MARKS = ("degenerate",)


def classify(msg: str) -> str:
    m = (msg or "").strip().lower()
    if not m:
        return "unreadable"
    for k in FAILURE_MARKS:
        if k in m:
            return "failure"
    for k in DEGENERATE_MARKS:
        if k in m:
            return "single_degenerate"
    if m.startswith("two-phase"):
        return "two_phase"
    if m.startswith("single-phase"):
        return "single_phase"
    return "unreadable"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = sorted(Path(args.in_dir).glob("*.parquet"))
    if not files:
        print(f"FATAL: no parquet chunks in {args.in_dir}")
        return 1
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    n = len(df)
    print(f"states: {n}   from {len(files)} chunk file(s)")

    kind = df["solve_path"].astype(str).map(classify)
    counts = Counter(kind)

    # Show the messages themselves, with their numeric tails collapsed so the
    # per-state values do not explode into thousands of singleton categories.
    shapes = Counter(df["solve_path"].astype(str)
                     .str.replace(r"[-+0-9.eE]+", "N", regex=True))
    print("\nmessage shapes:")
    for lab, c in sorted(shapes.items(), key=lambda kv: -kv[1])[:15]:
        print(f"   {lab:<46s} {c:8d}  ({100.0 * c / n:7.4f} %)")

    print("\nclassification:")
    for k in ("two_phase", "single_phase", "single_degenerate",
              "failure", "unreadable"):
        c = counts.get(k, 0)
        mark = "   <-- MUST BE ZERO" if k in ("failure", "unreadable") and c else ""
        print(f"   {k:<20s} {c:8d}  ({100.0 * c / n:7.4f} %){mark}")

    # A beta-degenerate state is a genuine single-phase verdict only when beta
    # really sits at a bound. An interior beta there is a resolved two-phase
    # split being reported as one phase.
    interior = 0
    deg = df[kind == "single_degenerate"]
    if len(deg) and "beta" in deg.columns:
        b = pd.to_numeric(deg["beta"], errors="coerce")
        interior = int(((b > 1e-6) & (b < 1 - 1e-6)).sum())

    # The flash's own flag, wherever a flash actually ran.
    ran = kind == "two_phase"
    unconverged = 0
    if "converged" in df.columns:
        unconverged = int((ran & (~df["converged"].astype(bool))).sum())

    print(f"\nbeta-degenerate with INTERIOR beta: {interior}")
    print(f"flash ran but did NOT converge     : {unconverged}")

    res = {
        "states": n, "chunk_files": len(files),
        "classification": {k: int(v) for k, v in counts.items()},
        "message_shapes": {k: int(v) for k, v in shapes.items()},
        "beta_degenerate_interior": interior,
        "unconverged_flash": unconverged,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(f"\nwrote {out}")

    bad = (counts.get("failure", 0) + counts.get("unreadable", 0)
           + interior + unconverged)
    if bad:
        print("\nFATAL: the grid does not fully resolve.")
        print(f"   {counts.get('failure', 0)} state(s) on a failure path")
        print(f"   {counts.get('unreadable', 0)} state(s) whose message this "
              "script could not classify")
        print(f"   {interior} beta-degenerate state(s) with an interior beta")
        print(f"   {unconverged} flash(es) that ran and did not converge")
        print("Paper III's convergence claim cannot be made as it stands.")
        return 1
    tp, sp = counts.get("two_phase", 0), counts.get("single_phase", 0)
    print(f"\nEvery one of the {n} states resolves: {tp} into two phases, "
          f"{sp} classified single-phase by a converged computation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
