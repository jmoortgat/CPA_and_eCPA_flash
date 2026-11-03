"""Assert that the reported numbers have not moved.

The first stage of the SLURM chain in sweep/slurm/. Every later stage runs
under ``afterok``, so a kernel that has moved stops the chain before the long
computations.

It re-scores the six gas families and the brine-density family and compares
each against the value verified on 2026-09-19, refusing anything that has
drifted by more than ``TOL``. A tolerance rather than equality because the
two kernels differ in the last digits by construction (independent Newton
implementations, different compilers); a tolerance of 0.002 percentage points
is far below anything reportable and far above floating-point trajectory
noise.

Exits non-zero, loudly, naming the family and both numbers, so the failure is
readable in a SLURM log without opening anything else.
"""
from __future__ import annotations

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

from fitting import dataset as ds       # noqa: E402
from fitting import predict as pr       # noqa: E402
from ms_ecpa import gbb                 # noqa: E402

#: Verified on GasBrineBench v1.1.1 (Fortran backend). v1.1.1 corrects
#: three mixed-brine recipes of v1.1.0; on v1.1.0 co2_solubility is 6.5441 %
#: and every other value below is the same.
#: family -> (rows, solved, AAD %)
BASELINE = {
    "co2_solubility": (2245, 2245, 6.5722),
    "co2_binary":     (559,   514, 9.0074),
    "co2_yh2o":       (633,   593, 24.6533),
    "ch4_solubility": (581,   581, 9.3413),
    "ch4_binary":     (833,   833, 7.6786),
    "ch4_yh2o":       (260,   260, 15.4882),
}
RHO_BASELINE = (905, 905, 1.2119)

#: Percentage points. A reportable change is 0.01 pp; trajectory noise is
#: nearer 1e-6 pp.
TOL = 0.002


def main() -> int:
    fams = ds.build(split_yh2o=True, keep_unsplit=True)
    failures, report = [], {}

    for name, (n_exp, solved_exp, aad_exp) in BASELINE.items():
        d = fams[name].df
        p = np.array([pr.predict_family(name, row, backend="fortran")
                      for _, row in d.iterrows()], dtype=float)
        y = np.asarray(d["value"], dtype=float)
        m = np.isfinite(p)
        aad = 100.0 * float(np.mean(np.abs(p[m] - y[m]) / y[m]))
        report[name] = {"rows": int(len(p)), "solved": int(m.sum()),
                        "aad_percent": round(aad, 4)}
        flag = ""
        if len(p) != n_exp or int(m.sum()) != solved_exp:
            failures.append(f"{name}: {int(m.sum())}/{len(p)} solved, "
                            f"expected {solved_exp}/{n_exp}")
            flag = "   <-- COUNT CHANGED"
        if abs(aad - aad_exp) > TOL:
            failures.append(f"{name}: AAD {aad:.4f} %, expected "
                            f"{aad_exp:.4f} % (drift {aad - aad_exp:+.4f} pp)")
            flag = "   <-- AAD MOVED"
        print("  %-16s %4d/%4d  AAD %8.4f %%  (expected %8.4f)%s"
              % (name, int(m.sum()), len(p), aad, aad_exp, flag), flush=True)

    import fig_brine_density as FB
    ions = ["m_Na", "m_Cl", "m_K", "m_Ca", "m_Mg", "m_SO4"]
    d = fams["rho"].df
    y = np.asarray(d["value"], dtype=float)
    p = np.full(len(d), np.nan)
    for i, (_, row) in enumerate(d.iterrows()):
        mi = np.nan_to_num(np.array([row[c] for c in ions], dtype=float))
        p[i] = FB.predict_rho_brine(float(row["T_K"]), float(row["P_bar"]), mi)[0]
    m = np.isfinite(p)
    aad = 100.0 * float(np.mean(np.abs(p[m] - y[m]) / y[m]))
    n_exp, solved_exp, aad_exp = RHO_BASELINE
    report["rho"] = {"rows": int(len(p)), "solved": int(m.sum()),
                     "aad_percent": round(aad, 4)}
    flag = ""
    if len(p) != n_exp or int(m.sum()) != solved_exp:
        failures.append(f"rho: {int(m.sum())}/{len(p)} solved, "
                        f"expected {solved_exp}/{n_exp}")
        flag = "   <-- COUNT CHANGED"
    if abs(aad - aad_exp) > TOL:
        failures.append(f"rho: AAD {aad:.4f} %, expected {aad_exp:.4f} %")
        flag = "   <-- AAD MOVED"
    print("  %-16s %4d/%4d  AAD %8.4f %%  (expected %8.4f)%s"
          % ("rho", int(m.sum()), len(p), aad, aad_exp, flag), flush=True)

    report["gasbrinebench"] = gbb.describe()
    out = _CODE / "results" / "verify_baseline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n  database: {report['gasbrinebench']}")
    print(f"  wrote {out}")

    if failures:
        print("\nFATAL: the reported numbers have moved:")
        for f in failures:
            print("   " + f)
        print("\nStopping the chain. Nothing downstream should be computed "
              "from a kernel whose published numbers no longer reproduce.")
        return 1
    print("\n  all families and density reproduce the verified values")
    return 0


if __name__ == "__main__":
    sys.exit(main())
