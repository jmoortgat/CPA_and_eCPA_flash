#!/usr/bin/env python3
"""Golden vectors for the phase-2 colligative routes.

Separate generator (the phase-1 golden_<tag>.json files are never
regenerated).  Writes fortran/golden/golden_colligative_<tag>.json.

WHICH PYTHON FUNCTION EACH BLOCK CERTIFIES
------------------------------------------
  "a_res"     scripts/total_mole_basis.py :: A_r(basis="total")
              -- the UN-AMPLIFIED primitive of the oracle, evaluated at
              a volume supplied in the record so both sides see exactly
              the same V.  This is the block that actually certifies
              the assembly; phi_osm below cannot, because the reference
              reaches it through two nested central differences whose
              noise floor is ~1e-7 relative (PORTING_NOTES.md 15).
  "ln_f_w"    scripts/total_mole_basis.py :: ln_f_w + solve_V
              -- one central difference and one Brent solve on a
              numerically-differentiated pressure.
  "phi_osm"   scripts/total_mole_basis.py :: phi_osm_total
              -- the phase-4 'oracle' (INDEPENDENT total-mole-basis
              Helmholtz assembly; does not call the lnphi kernels).
  "rho_brine" scripts/fig_brine_density.py :: predict_rho_brine
              -- production aqueous kernel + Peneloux ion shifts.
  "psat"      the vapor-pressure scripts' water-isofugacity
              psat_model -- water isofugacity, Brent on ln P.
              CoolProp supplies only the bracket anchor p0; it is
              recorded per state so the port can be handed the same
              bracket instead of reimplementing IAPWS.  If CoolProp is
              unavailable the psat block is emitted empty and the
              corresponding test skips.

Usage: python3 gen_golden_colligative.py [outdir]
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "scripts"))
sys.path.insert(0, str(HERE))

from ms_ecpa import paper1_config                  # noqa: E402
import ms_ecpa.parameters as _P                    # noqa: E402
# Read _P.Mw LIVE at each use. A module-level `from ... import Mw`
# binds once, so it kept the production value while
# paper1_config.apply_paper1() rebound parameters.Mw to the published
# Paper-I value -- generating the paper1 goldens with one water molar
# mass while the Fortran .par file carried the other.
import gen_params                                   # noqa: E402

# The oracle snapshots ion parameters into its own IP namespace at
# import, so it must be (re)loaded AFTER any configuration change --
# see total_mole_basis.reload_parameters.
import total_mole_basis as tmb                      # noqa: E402

sys.path.insert(0, str(CODE / "scripts"))
from fig_brine_density import predict_rho_brine     # noqa: E402

try:
    from CoolProp.CoolProp import PropsSI           # noqa: E402
    HAVE_COOLPROP = True
except Exception:
    HAVE_COOLPROP = False


# --- phi_osm grid -----------------------------------------------------
# Molalities start at 0.1: phi is a DIFFERENCE of two ln f_w values, so
# at m -> 0 the cancellation makes it arbitrarily ill-conditioned with
# respect to last-ulp differences (see PORTING_NOTES.md 15).  The
# low-m limit is still probed (0.1) but not pushed to 1e-3.
PHI_SALTS_FULL = ["NaCl", "KCl", "CaCl2", "MgCl2", "Na2SO4"]
PHI_SALTS_NACL = ["NaCl"]
PHI_M = [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]
PHI_T = [298.15, 348.15, 423.15]
PHI_P_PA = 1.0e5
# a_res / ln_f_w share the phi grid plus the pure-water state (m = 0).
AR_M = [0.0] + PHI_M

# --- rho_brine grid ---------------------------------------------------
RHO_SALTS_FULL = {
    "NaCl":   lambda m: [m, m, 0, 0, 0, 0],
    "KCl":    lambda m: [0, m, m, 0, 0, 0],
    "CaCl2":  lambda m: [0, 2 * m, 0, m, 0, 0],
    "MgCl2":  lambda m: [0, 2 * m, 0, 0, m, 0],
    "Na2SO4": lambda m: [2 * m, 0, 0, 0, 0, m],
}
RHO_SALTS_NACL = {"NaCl": RHO_SALTS_FULL["NaCl"]}
RHO_M = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0]
RHO_T = [298.15, 348.15, 398.15, 448.15]
RHO_P = [5.0, 50.0, 200.0, 400.0, 700.0]

# --- psat grid --------------------------------------------------------
# T starts at 323.15 K, NOT at the 283.15 K the reference sweeps.
# At 283.15 K the pure-water saturation pressure is 0.0123 bar, and
# down there the REFERENCE's own lnphi_aq is multivalued: sweeping the
# Brent bracket with ms_ecpa alone, the isofugacity residual changes
# sign 3-20 times (vs exactly once at every T >= 323.15 K), because the
# aqueous 3-var inner Newton lands on different roots at neighbouring
# pressures.  scipy's brentq then returns whichever branch its sample
# points happened to straddle.  That is not a reproducible target for
# ANY reimplementation, so those states are excluded from the gate and
# reported instead -- see PORTING_NOTES.md 17.
PSAT_M = [0.0, 1.0, 3.0, 6.0]
PSAT_T = [323.15, 373.15, 423.15, 473.15, 523.15, 573.15]


def gen_ar_block(salts):
    """A_r at a recorded volume (isolates the assembly from the root
    find) PLUS the ln_f_w / solve_V pair at the same state."""
    cases = []
    for salt in salts:
        for T in PHI_T:
            for m in AR_M:
                n = tmb.n_vec(salt, m)
                rec = dict(salt=salt, m=m, T=T, P_Pa=PHI_P_PA,
                           n=[float(v) for v in n])
                try:
                    V = float(tmb.solve_V(n, T, PHI_P_PA, basis="total"))
                    rec["V"] = V
                    rec["A_r"] = float(tmb.A_r(n, V, T, basis="total"))
                    # A second, off-root volume: the assembly must be
                    # right everywhere, not only at the P(V) root.
                    V2 = V * 1.37
                    rec["V2"] = V2
                    rec["A_r2"] = float(tmb.A_r(n, V2, T, basis="total"))
                    rec["ln_f_w"] = float(
                        tmb.ln_f_w(n, T, PHI_P_PA, basis="total"))
                    rec["ok"] = True
                except Exception as exc:
                    rec["ok"] = False
                    rec["error"] = str(exc)[:120]
                cases.append(rec)
    return cases


def gen_phi_block(salts):
    cases = []
    for salt in salts:
        for T in PHI_T:
            for m in PHI_M:
                rec = dict(salt=salt, m=m, T=T, P_Pa=PHI_P_PA)
                try:
                    rec["phi"] = float(tmb.phi_osm_total(
                        salt, m, T=T, Pp=PHI_P_PA))
                except Exception as exc:
                    rec["phi"] = float("nan")
                    rec["error"] = str(exc)[:120]
                cases.append(rec)
    return cases


def gen_rho_block(salts):
    cases = []
    for salt, mk in salts.items():
        for T in RHO_T:
            for m in RHO_M:
                mi = np.asarray(mk(m), dtype=float)
                for Pbar in RHO_P:
                    rec = dict(salt=salt, m=m, T=T, P_bar=Pbar,
                               mi=[float(v) for v in mi])
                    try:
                        rho, st = predict_rho_brine(T, Pbar, mi)
                        rec["rho"] = float(rho)
                        rec["ok"] = bool(np.isfinite(rho))
                        rec["v"] = ([float(np.real(v)) for v in st]
                                    if st is not None else None)
                    except Exception as exc:
                        rec["rho"] = float("nan")
                        rec["ok"] = False
                        rec["error"] = str(exc)[:120]
                    cases.append(rec)
    return cases


def gen_psat_block():
    if not HAVE_COOLPROP:
        return []
    from scipy.optimize import brentq
    from ms_ecpa.lnphi import lnphi_aq, lnphi_gas

    def psat_model(T, m):
        """Verbatim vpsat_newphys.py::psat_model, with p0 returned so
        the port can reuse the identical bracket."""
        mi = np.array([m, m, 0, 0, 0, 0], dtype=float)
        x_w = 1.0 / (1.0 + _P.Mw * 2.0 * m)
        x_aq = (x_w, 0.0, 0.0)
        y = (1.0, 0.0, 0.0)

        def f(lnP):
            P = float(np.exp(lnP))
            la = lnphi_aq(x_aq, mi, float(T), P)
            lg = lnphi_gas(y, float(T), P)
            return (np.log(x_w) + la[0]) - lg[0]

        p0 = PropsSI("P", "T", float(T), "Q", 0, "Water")
        P = float(np.exp(brentq(f, np.log(0.50 * p0), np.log(1.10 * p0),
                                xtol=1e-12, rtol=1e-10))) / 1e5
        return P, float(p0)

    cases = []
    for m in PSAT_M:
        for T in PSAT_T:
            rec = dict(m=m, T=T, lo_fac=0.50, hi_fac=1.10)
            try:
                rec["p0_Pa"] = float(PropsSI("P", "T", float(T), "Q", 0,
                                             "Water"))
            except Exception as exc:
                rec["p0_Pa"] = float("nan")
                rec["P_bar"] = float("nan")
                rec["ok"] = False
                rec["error"] = str(exc)[:120]
                cases.append(rec)
                continue
            try:
                P, _ = psat_model(float(T), m)
                rec["P_bar"] = float(P)
                rec["ok"] = True
            except Exception as exc:
                rec["P_bar"] = float("nan")
                rec["ok"] = False
                rec["error"] = str(exc)[:120]
            cases.append(rec)
    return cases


def gen_config(tag, phi_salts, rho_salts):
    # The oracle caches ion parameters at import; re-snapshot after any
    # configuration change so the block reflects THIS parameter set.
    tmb.reload_parameters()
    ta = time.time()
    ar = gen_ar_block(phi_salts)
    t0 = time.time()
    phi = gen_phi_block(phi_salts)
    t1 = time.time()
    rho = gen_rho_block(rho_salts)
    t2 = time.time()
    psat = gen_psat_block()
    t3 = time.time()
    n_ar = sum(1 for c in ar if c.get("ok"))
    n_phi = sum(1 for c in phi if np.isfinite(c["phi"]))
    n_rho = sum(1 for c in rho if c.get("ok"))
    n_ps = sum(1 for c in psat if c.get("ok"))
    print(f"[{tag}] a_res/ln_f_w {n_ar}/{len(ar)} ok ({t0-ta:.1f}s), "
          f"phi_osm {n_phi}/{len(phi)} finite ({t1-t0:.1f}s), "
          f"rho_brine {n_rho}/{len(rho)} ok ({t2-t1:.1f}s), "
          f"psat {n_ps}/{len(psat)} ok ({t3-t2:.1f}s)"
          f"{'' if HAVE_COOLPROP else '  [CoolProp missing: psat skipped]'}")
    return dict(tag=tag, a_res=ar, phi_osm=phi, rho_brine=rho, psat=psat,
                have_coolprop=HAVE_COOLPROP)


def main():
    outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "golden"
    outdir.mkdir(parents=True, exist_ok=True)

    blk = gen_config("paperII_III", PHI_SALTS_FULL, RHO_SALTS_FULL)
    (outdir / "golden_colligative_paperII_III.json").write_text(
        json.dumps(blk))

    paper1_config.apply_paper1()
    try:
        blk = gen_config("paper1", PHI_SALTS_NACL, RHO_SALTS_NACL)
        (outdir / "golden_colligative_paper1.json").write_text(json.dumps(blk))
    finally:
        paper1_config.restore_paper1()
        tmb.reload_parameters()




if __name__ == "__main__":
    main()
