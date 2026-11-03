#!/usr/bin/env python3
"""Prove a change was behaviour-preserving, point by point.

Compares the live kernel against reproduce/reference_states.json. The stored
states were recorded with the released kernel (Python backend) by
--write; the input states (T, P, z, ions) are those of the original
capture.

Point by point, deliberately. An AAD is an average, and compensating errors
can leave one unchanged while individual predictions move a long way -- so
"the AAD is still 1.74 %" is much weaker evidence than it looks. This
compares every state's phases, route, beta, both phase compositions, both
compressibilities and eps_r.

    python3 reproduce/verify_reference.py [--backend python|fortran] [--limit N]
    python3 reproduce/verify_reference.py --write    # re-record the reference
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

TOL = 1e-10
#: Fortran backend only: four single-phase states at 573.15 K and 2000 bar (above
#: the 1500-bar range of the papers) where the two kernels agree to ~1e-8.
FORTRAN_LOOSE = {1897, 1898, 1899, 1901}
TOL_LOOSE = 1e-7
SCALARS = ("beta", "Z_aq", "Z_gas", "eps_r")
VECTORS = ("x_aq", "y_gas")


def _close(a, b, tol=TOL):
    if a is None or b is None:
        return a is b
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isnan(a) != math.isnan(b):
        return False
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="python",
                    choices=("python", "fortran"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--write", action="store_true",
                    help="re-record every state with the Python kernel")
    a = ap.parse_args(argv)
    if a.write:
        return write_reference()

    ref = json.loads((HERE / "reference_states.json").read_text())
    states = ref["flash"]
    if a.limit:
        states = states[:a.limit]

    if a.backend == "fortran":
        sys.path.insert(0, str(ROOT / "fortran" / "tools"))
        from ecpa_ctypes import Engine
        eng = Engine(parfile=str(ROOT / "fortran" / "params" / "paperII_III.par"))

        def run(T, P, z, mi):
            r = eng.stability_flash(T, P, np.array(z), np.array(mi),
                                    salinity_convention="as-aqueous")
            f = r.get("flash") if r.get("has_flash") else None
            return int(r["phases"]), (f or {})
    else:
        from ms_ecpa import stability_flash

        def run(T, P, z, mi):
            r = stability_flash(T, P, np.array(z), np.array(mi),
                                salinity_convention="as-aqueous")
            fr = r.flash_result
            if fr is None:
                return int(r.phases), {}
            return int(r.phases), {
                "beta": fr.beta, "x_aq": list(fr.x_aq), "y_gas": list(fr.y_gas),
                "Z_aq": fr.Z_aq, "Z_gas": fr.Z_gas, "eps_r": fr.eps_r,
                "converged": bool(fr.converged)}

    bad, checked = [], 0
    for i, s in enumerate(states):
        if "err" in s:
            continue
        try:
            phases, f = run(s["T"], s["P"], s["z"], s["mi"])
        except Exception as exc:
            bad.append((i, "raised", type(exc).__name__, ""))
            continue
        checked += 1
        if phases != s.get("phases"):
            bad.append((i, "phases", s.get("phases"), phases))
            continue
        if "beta" not in s:
            continue
        if bool(f.get("converged")) != bool(s.get("conv")):
            bad.append((i, "converged", s.get("conv"), f.get("converged")))
        tol = TOL_LOOSE if (a.backend == "fortran" and i in FORTRAN_LOOSE) else TOL
        for k in SCALARS:
            if not _close(f.get(k), s.get(k), tol):
                bad.append((i, k, s.get(k), f.get(k)))
        for k in VECTORS:
            for j, (p, q) in enumerate(zip(f.get(k, []), s.get(k, []))):
                if not _close(p, q, tol):
                    bad.append((i, f"{k}[{j}]", q, p))

    print(f"backend={a.backend}  states checked: {checked}")
    if not bad:
        loose = (f" ({len(FORTRAN_LOOSE)} states at {TOL_LOOSE:g}, see FORTRAN_LOOSE)"
                 if a.backend == "fortran" else "")
        print("IDENTICAL: every state matches the reference to "
              f"{TOL:g} relative{loose}.")
        return 0
    print(f"MISMATCHES: {len(bad)}")
    for row in bad[:20]:
        print(f"  state {row[0]:5d}  {row[1]:12s} ref={row[2]!r}  now={row[3]!r}")
    return 1


def write_reference() -> int:
    """Re-record the reference outputs for the stored input states."""
    from ms_ecpa import stability_flash
    path = HERE / "reference_states.json"
    ref = json.loads(path.read_text())
    out = []
    for s in ref["flash"]:
        new = {k: s[k] for k in ("T", "P", "z", "mi")}
        try:
            r = stability_flash(s["T"], s["P"], np.array(s["z"]), np.array(s["mi"]),
                                salinity_convention="as-aqueous")
        except Exception as exc:
            new["err"] = type(exc).__name__
            out.append(new)
            continue
        new["phases"] = int(r.phases)
        new["route"] = int(r.route)
        new["tpd"] = float(r.tpd_min)
        fr = r.flash_result
        if fr is not None:
            new.update({"beta": fr.beta, "x_aq": list(map(float, fr.x_aq)),
                        "y_gas": list(map(float, fr.y_gas)), "Z_aq": fr.Z_aq,
                        "Z_gas": fr.Z_gas, "eps_r": fr.eps_r,
                        "conv": bool(fr.converged)})
        out.append(new)
    ref["flash"] = out
    path.write_text(json.dumps(ref, indent=None) + "\n")
    print(f"wrote {len(out)} states to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
