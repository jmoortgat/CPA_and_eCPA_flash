"""Does the aqueous-trial composition basis change reported classifications?

III-1.3 reports a 1440-state sweep in which the stability verdict differs at 36
states, of which 25 reach a genuine two-phase flash. That sweep was run ad hoc
and never committed, and the ad-hoc c3_locus search turned out not to match the
shipped code, so this reproduces it from the kernel as it now stands.

Grid, exactly as the response item describes it: 1 and 4 mol/kg NaCl, twelve
feeds (binary CO2 and binary CH4 from 2 % to 90 % gas, plus two ternaries), six
isotherms 283-523 K, ten pressures log-spaced 1-2000 bar.  2*12*6*10 = 1440.

Each state is classified TWICE -- once with the trial on the flash's
composition basis, once on the original partition basis -- by toggling
ms_ecpa.stability.AQ_TRIAL_ON_FLASH_BASIS.  Python backend only: the toggle is
a Python module attribute and the compiled kernel would ignore it.
"""
import os, sys, warnings, itertools, json
warnings.filterwarnings("ignore")
os.environ.pop("MS_ECPA_BACKEND", None)          # the toggle is Python-side
os.environ.setdefault("ECPA_GBB_REFRESH", "never")
from pathlib import Path
_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))
import numpy as np
from multiprocessing import Pool

M_NACL = (1.0, 4.0)
GASFRAC = (0.02, 0.10, 0.30, 0.60, 0.90)
T_GRID = np.array([283.15, 323.15, 373.15, 423.15, 473.15, 523.15])
P_GRID = np.geomspace(1.0, 2000.0, 10)

def feeds():
    """The twelve feeds, each with a label that IDENTIFIES it uniquely.

    The label carries the gas fraction. An earlier version labelled all five
    CO2 feeds "CO2", so a row of the output did not say which composition it
    came from and the state could not be rebuilt from the saved results -- a
    later check that tried to re-flash a differing state silently used a
    different feed at the same (T, P, m) and measured nothing.
    """
    out = []
    for g in GASFRAC:
        out.append((f"CO2_{g:.2f}", np.array([1 - g, 0.0, g])))
        out.append((f"CH4_{g:.2f}", np.array([1 - g, g, 0.0])))
    out.append(("tern_lo", np.array([0.80, 0.10, 0.10])))
    out.append(("tern_hi", np.array([0.40, 0.30, 0.30])))
    return out                                    # 12

FEEDS = feeds()
STATES = [(m, lab, z, float(T), float(P))
          for m in M_NACL for (lab, z) in FEEDS
          for T in T_GRID for P in P_GRID]

def classify(basis, T, P, z, mi):
    import ms_ecpa
    import ms_ecpa.stability as st
    st.AQ_TRIAL_ON_FLASH_BASIS = basis
    try:
        r = ms_ecpa.stability_flash(T, P * 1e5, z, mi)
    except Exception as exc:
        return dict(err=type(exc).__name__)
    beta = float("nan"); conv = False
    if r.flash_result is not None:
        beta = float(r.flash_result.beta); conv = bool(r.flash_result.converged)
    return dict(phases=int(r.phases), tpd=float(getattr(r, "tpd_min", float("nan"))),
                beta=beta, conv=conv)

def one(s):
    m, lab, z, T, P = s
    mi = np.zeros(6); mi[0] = mi[1] = m
    new = classify(True,  T, P, z, mi)
    old = classify(False, T, P, z, mi)
    return dict(m=m, feed=lab, z=[float(v) for v in z],
                T=T, P=P, new=new, old=old)

if __name__ == "__main__":
    print(f"{len(STATES)} states x 2 conventions", flush=True)
    with Pool(int(os.environ.get("ECPA_WORKERS", "8"))) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(one, STATES, chunksize=8)):
            rows.append(r)
            if (i + 1) % 120 == 0: print(f"  {i+1}/{len(STATES)}", flush=True)

    err = [r for r in rows if "err" in r["new"] or "err" in r["old"]]
    ok  = [r for r in rows if r not in err]
    diff = [r for r in ok if r["new"]["phases"] != r["old"]["phases"]]
    new2 = [r for r in diff if r["new"]["phases"] == 2 and r["old"]["phases"] == 1]
    other = [r for r in diff if r not in new2]
    genuine = [r for r in new2
               if r["new"]["conv"] and 1e-6 < r["new"]["beta"] < 1 - 1e-6]
    degen = [r for r in new2 if r not in genuine]

    print(f"\nstates classified under both : {len(ok)}   solver errors: {len(err)}")
    print(f"stability/flash verdict differs: {len(diff)}  ({100*len(diff)/max(len(ok),1):.2f} %)")
    print(f"   flash-basis 2-phase where original said 1-phase: {len(new2)}")
    print(f"   the other direction                            : {len(other)}")
    print(f"   of those {len(new2)}: genuine interior-beta two-phase: {len(genuine)}"
          f"   degenerate/unconverged: {len(degen)}")
    # A stable state's trials collapse to the trivial fixed point W = z, where
    # sum W = 1 and tpd = 1 - 1 is exactly zero. A tpd_old of +0.00000 is
    # therefore the correct value for a state the original convention calls
    # stable, not a missing field.
    #
    # PROVENANCE. III-1.3 as first drafted reported 36 differing / 25 genuine
    # on a sweep that was run ad hoc and never committed. This script does not
    # reproduce that: the kernel has since gained the LIQUID_EPSR_MIN aqueous
    # guard, and the counts are about three times larger. The letter must quote
    # what THIS script produces, which is why it now lives in reproduce/.
    if genuine:
        print("\ngenuine two-phase states the original convention called stable:")
        for r in sorted(genuine, key=lambda r: (r["T"], r["P"]))[:30]:
            print(f"   m={r['m']:.0f} {r['feed']:8s} T={r['T']:7.2f} P={r['P']:9.3f}  "
                  f"beta={r['new']['beta']:.4f}  tpd_new={r['new']['tpd']:+.5f} "
                  f"tpd_old={r['old']['tpd']:+.5f}")
    out = str(_CODE / "results" / "trial_basis_sweep.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(n=len(ok), n_err=len(err), n_diff=len(diff), n_new2=len(new2),
                   n_genuine=len(genuine), n_degen=len(degen), rows=rows),
              open(out, "w"), indent=1, default=float)
    print(f"\nwrote {out}")
