#!/usr/bin/env python3
"""Regenerate ONLY the "psat" block of the colligative golden files.

Why this exists as a separate tool: the a_res / phi_osm blocks in
gen_golden_colligative.py take ~10 minutes per parameter set (the
oracle runs two Brent solves and two nested central differences per
evaluation), while psat is 24 cheap states.  When the psat grid changes
-- as it did when the 283.15 K states were found to be ill-posed in the
reference (PORTING_NOTES.md 17) -- redoing the slow blocks buys
nothing.

    python3 fortran/tools/regen_psat_block.py

Rewrites golden/golden_colligative_<tag>.json in place, touching only
the "psat" key.
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
for p in (str(CODE), str(CODE / "scripts"), str(HERE)):
    sys.path.insert(0, p)

from ms_ecpa import paper1_config                  # noqa: E402
import gen_params                                   # noqa: E402
import gen_golden_colligative as ggc                # noqa: E402

OUT = HERE.parent / "golden"

if not ggc.HAVE_COOLPROP:
    sys.exit("CoolProp is unavailable; the psat block cannot be built.")


def patch(tag):
    path = OUT / f"golden_colligative_{tag}.json"
    if not path.exists():
        print(f"[{tag}] no golden file; skipped")
        return
    blk = json.loads(path.read_text())
    old = len(blk.get("psat", []))
    blk["psat"] = ggc.gen_psat_block()
    n_ok = sum(1 for c in blk["psat"] if c.get("ok"))
    path.write_text(json.dumps(blk))
    print(f"[{tag}] psat block {old} -> {len(blk['psat'])} states "
          f"({n_ok} ok); other blocks untouched")


patch("paperII_III")

paper1_config.apply_paper1()
try:
    patch("paper1")
finally:
    paper1_config.restore_paper1()

