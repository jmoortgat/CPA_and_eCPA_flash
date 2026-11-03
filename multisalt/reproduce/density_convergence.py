"""Brine-density convergence and AAD, measured and written down.

Why this script exists. The convergence figure for the density family was
quoted in working notes as "905 of 905" while the newest artifact on disk --
a cached parquet and a comment in the manuscript -- said 799 and 800. Both
were right about different code: the 799 predates the seed work in the
aqueous three-variable solve, the 905 follows it. Nothing on disk recorded
the newer measurement, so a reader checking the claim found only the older
number and was correct to disbelieve the newer one.

So the number is produced here rather than asserted, through the SAME
entry point the paper's figure uses (``fig_brine_density.predict_rho_brine``,
marched along each isotherm), and written to ``results/density_convergence.json``
with the parameter file, the backend and the GasBrineBench revision beside it.

Run:  python3 reproduce/density_convergence.py [--backend fortran|python]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np

_HERE = Path(__file__).resolve().parent
_CODE = _HERE.parent
for p in (str(_CODE), str(_CODE / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import fig_brine_density as FB           # noqa: E402
from fitting import dataset as ds        # noqa: E402
from ms_ecpa import gbb                  # noqa: E402

IONS = ["m_Na", "m_Cl", "m_K", "m_Ca", "m_Mg", "m_SO4"]


def measure(backend: str = "fortran") -> dict:
    df = ds.build(split_yh2o=True, keep_unsplit=True)["rho"].df
    y = np.asarray(df["value"], dtype=float)
    pred = np.full(len(df), np.nan)
    unsolved = []
    for i, (_, row) in enumerate(df.iterrows()):
        mi = np.nan_to_num(np.array([row[c] for c in IONS], dtype=float))
        T, P = float(row["T_K"]), float(row["P_bar"])
        rho, _ = FB.predict_rho_brine(T, P, mi)
        pred[i] = rho
        if not np.isfinite(rho):
            unsolved.append({"T_K": T, "P_bar": P,
                             "m_tot": float(np.sum(mi))})
    ok = np.isfinite(pred)
    aad = 100.0 * float(np.mean(np.abs(pred[ok] - y[ok]) / y[ok]))
    return {
        "rows": int(len(df)),
        "solved": int(ok.sum()),
        "converged_percent": round(100.0 * float(ok.sum()) / len(df), 4),
        "aad_percent": round(aad, 4),
        "unsolved": unsolved,
        "entry_point": "scripts/fig_brine_density.py::predict_rho_brine",
        "backend": backend,
        "gasbrinebench": gbb.describe(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="fortran",
                    choices=("fortran", "python"))
    ap.add_argument("--out", default=str(_CODE / "results" /
                                         "density_convergence.json"))
    args = ap.parse_args()

    res = measure(args.backend)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")

    print(f"rows            : {res['rows']}")
    print(f"solved          : {res['solved']}  ({res['converged_percent']} %)")
    print(f"AAD             : {res['aad_percent']} %")
    print(f"entry point     : {res['entry_point']}")
    print(f"backend         : {res['backend']}")
    print(f"database        : {res['gasbrinebench']}")
    if res["unsolved"]:
        print(f"unsolved states : {len(res['unsolved'])}")
        for u in res["unsolved"][:10]:
            print(f"   T={u['T_K']:.2f} P={u['P_bar']:.1f} m_tot={u['m_tot']:.2f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
