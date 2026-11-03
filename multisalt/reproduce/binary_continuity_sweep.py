"""Does the CO2-H2O binary switch roots as pressure varies?

A physical isotherm is smooth: x_CO2(P) and y_H2O(P) have no jumps, and the
compressibility factors move continuously. A solver that accepts whatever root
its Newton lands on does NOT guarantee that -- it can hop between branches of
the cubic from one pressure to the next, and the result is a plausible curve
with a step in it.

This matters beyond tidiness. The binary currently cannot fit solubility and
water content at the same time (reproduce/enhancement_factor.py and the
constrained refits), and a root-switching defect would produce exactly that
symptom while looking like a physical trade-off. Several audit findings point
at this class on this path:

    lnphi.py:903   _gas_root_is_physical may accept the mechanically unstable
                   MIDDLE root of the Z equation (dP/dV > 0 -- not a phase)
    flash.py:547   the SSI aqueous inner Newton cold-starts at Zw = 0.05 and
                   accepts whatever it converges to; its seed ladder runs only
                   on failure to converge
    lnphi.py:696   lnphi_aq screens chi1w > 0 only, with no liquid-branch test

So: sweep P finely along fixed isotherms, record the ROOTS as well as the
compositions, and flag every step that is large relative to the local trend.
The feed fraction actually accepted by the retry ladder and the solver route
are recorded too, because a change in either is a legitimate alternative
explanation for a step and must not be mistaken for a branch hop.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

from fitting.predict import Z_GAS_RETRIES, _engine

#: Isotherms spanning the binary data: below CO2's Tc, through it, and above.
T_LIST = (283.0, 298.0, 310.0, 323.0, 348.0, 373.0, 423.0, 473.0)

#: Dense in P so a branch hop cannot hide between samples.
P_GRID = np.unique(np.concatenate([
    np.linspace(5.0, 100.0, 120),
    np.linspace(100.0, 400.0, 120),
    np.linspace(400.0, 700.0, 60),
]))

#: A branch hop is not a large STEP -- it is a break in the local TREND.
#:
#: Comparing each step to the isotherm's median step does not work: at low
#: pressure x_CO2 follows Henry's law, x ~ P, so d(ln x) is large there and
#: tiny once solubility saturates. A median-step test duly flags the whole
#: normal low-pressure regime (313 false "hops" on the first attempt). What
#: identifies a hop is the SECOND difference: the local log-log slope
#: d(ln v)/d(ln P) drifts smoothly along a physical isotherm, and jumps when
#: the solver changes branch.
JUMP_FACTOR = 12.0


def state_at(T, P):
    """One flash, returning composition AND the roots it sat on."""
    eng = _engine()
    mi = np.zeros(6)
    for zg in Z_GAS_RETRIES:
        z = np.array([1.0 - zg, 0.0, zg])
        r = eng.stability_flash(float(T), float(P), z, mi,
                                salinity_convention="as-aqueous")
        if int(r.get("phases", 0)) != 2 or not r.get("has_flash"):
            continue
        f = r["flash"]
        if not f.get("converged"):
            continue
        xH, _, xC = f["x_aq"]
        xsf = xC / (xH + xC)
        if abs(xsf - zg) < 1e-3 * max(zg, 1e-6):   # degenerate: feed preserved
            continue
        return dict(x_sf=float(xsf), y_h2o=float(f["y_gas"][0]),
                    Z_aq=float(f["Z_aq"]), Z_gas=float(f["Z_gas"]),
                    chi_aq=float(f.get("chi_H2O_aq", np.nan)),
                    route=int(r.get("route", -99)), zg=float(zg))
    return None


def scan_isotherm(T):
    rows = [(P, state_at(T, P)) for P in P_GRID]
    got = [(P, s) for P, s in rows if s is not None]
    if len(got) < 20:
        return dict(T=T, n=len(got), jumps=[], note="too few converged states")
    P = np.array([p for p, _ in got])
    out = {k: np.array([s[k] for _, s in got]) for k in got[0][1]}
    jumps = []
    for field in ("x_sf", "y_h2o", "Z_aq", "Z_gas"):
        v = out[field]
        ok = np.isfinite(v) & (np.abs(v) > 0)
        if ok.sum() < 20:
            continue
        Pk, vk = P[ok], np.abs(v[ok])
        # local log-log slope, then how much that slope changes step to step
        slope = np.diff(np.log(vk)) / np.diff(np.log(Pk))
        dslope = np.abs(np.diff(slope))
        med = np.median(dslope[dslope > 0]) if np.any(dslope > 0) else 0.0
        if med <= 0:
            continue
        for i in np.where(dslope > JUMP_FACTOR * med)[0]:
            j = i + 1                      # the step whose slope broke trend
            same_feed = out["zg"][ok][j] == out["zg"][ok][j + 1]
            same_route = out["route"][ok][j] == out["route"][ok][j + 1]
            jumps.append(dict(field=field, P_lo=float(Pk[j]),
                              P_hi=float(Pk[j + 1]),
                              v_lo=float(vk[j]), v_hi=float(vk[j + 1]),
                              ratio=float(dslope[i] / med),
                              slope_lo=float(slope[i]), slope_hi=float(slope[j]),
                              same_feed=bool(same_feed),
                              same_route=bool(same_route)))
    return dict(T=T, n=len(got), jumps=jumps, note="")


def main() -> int:
    print(f"CO2-H2O binary continuity: {len(T_LIST)} isotherms x "
          f"{len(P_GRID)} pressures\n")
    total, unexplained = 0, 0
    for T in T_LIST:
        r = scan_isotherm(T)
        js = r["jumps"]
        total += len(js)
        bad = [j for j in js if j["same_feed"] and j["same_route"]]
        unexplained += len(bad)
        flag = "  <-- BRANCH HOP" if bad else ""
        print(f"  T = {T:5.0f} K   {r['n']:3d} states   "
              f"{len(js):2d} steps flagged, {len(bad):2d} with no feed/route "
              f"change{flag} {r['note']}")
        for j in bad[:4]:
            print(f"        {j['field']:6s} {j['P_lo']:7.1f} -> {j['P_hi']:7.1f} bar : "
                  f"{j['v_lo']:.6g} -> {j['v_hi']:.6g}   slope "
                  f"{j['slope_lo']:+.3f} -> {j['slope_hi']:+.3f}  "
                  f"({j['ratio']:.0f}x median slope change)")
    print(f"\n  {total} steps flagged; {unexplained} not explained by a change "
          f"of feed or solver route")
    print("  (a step with the same feed AND the same route is a branch hop)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
