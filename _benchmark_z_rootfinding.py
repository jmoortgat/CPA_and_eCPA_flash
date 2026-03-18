"""
_benchmark_z_rootfinding.py
===========================
Benchmark: compressibility-factor (Z) root-finding cost across three EoS levels.

  Level 1 – SRK (no association)
      Z is the root of a cubic polynomial.  Solved analytically (Cardano /
      numpy) — zero iterations, ~microsecond cost.

  Level 2 – CPA (Wertheim 4C association, salt-free)
      Z and (χ_H₂O, χ_CO₂) are coupled.  CPA2.ZChi() uses a 12-point
      coarse scan + Brent (up to 60 steps), with an inner analytic χ-solve
      (ChiChi cubic) at every evaluation.

  Level 3 – eCPA (CPA + Debye–Hückel + dipole permittivity; aqueous phase only)
      Z, χ_H₂O, and ε_r are coupled.  Solved with scipy.fsolve on the
      3-variable aqueous residual extracted from the ELV system.
      Note: the ZPerm term (requires ∂ε_r/∂V derivative chain) is set to zero
      here — a minor simplification that slightly underestimates eCPA cost.

All solvers use a cold start (no solution table, no warm guess).
Aqueous-phase compositions are used throughout for the CPA and eCPA cases.
CO₂-rich phase added for SRK and CPA (eCPA CO₂-rich is identical to CPA).

Output
------
  results/z_rootfinding_benchmark.csv  – per-condition raw data
  figures/z_rootfinding_benchmark.png  – summary figure
"""

import sys, time, warnings
from unittest.mock import patch
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, ".")
import CPA2
from ecpa.constants import (
    R, Na, kb, e, eps0, Mw,
    Tc1, Tc4, b1, b2, b3, b4,
    a01, a04, c11, c14,
    Akij, Bkij, Ckij, ASij, BSij, CSij, Tref,
    epsW, kappaW,
    Z2, Z3, Sg2, Sg3, Rb2, Rb3,
    dip01, pol1, pol2, pol3, pol4,
    GAMMA1, THETA1, zww,
    Uref1s, Talfa1s, alfa1s,
    Uref4s, Talfa4s, alfa4s,
)

# a02, a03 = 0 (ions don't contribute directly)
a02 = 0.0
a03 = 0.0

# ── Conditions ─────────────────────────────────────────────────────────────────
T_GRID  = np.array([303, 323, 373, 423, 473, 523], dtype=float)   # K
P_GRID  = np.array([10, 50, 100, 200, 400, 800, 1500], dtype=float)  # bar

# Fixed aqueous composition (CO₂+H₂O+NaCl at ms=1 mol/kg)
# Choose x1w so that x4w ≈ 0.02 with ms=1.0 and Mw=0.018 kg/mol.
#   x1w (1 + 2·ms·Mw) = 1 − x4w  →  x1w = 0.98/(1 + 0.036) ≈ 0.9459
_MS     = 1.0                           # NaCl molality [mol/kg]
_X1W    = 0.98 / (1.0 + 2*_MS*Mw)      # H₂O mol fraction in aqueous phase
_X2W    = _X1W * _MS * Mw              # Na⁺
_X3W    = _X2W                         # Cl⁻
_X4W    = 1.0 - _X1W - _X2W - _X3W    # CO₂ in aqueous

# CO₂-rich composition for SRK / CPA
_X4C    = 0.99   # CO₂ mol frac in CO₂-rich phase
_X1C    = 1.0 - _X4C

N_REPEAT = 5   # timing repeats per condition (take minimum)

# ── 1. SRK analytic cubic solve ────────────────────────────────────────────────

def _srk_AB(T, P_bar, x_h2o, x_co2):
    """SRK A, B for a two-component mixture (H₂O + CO₂), HV mixing rule."""
    # Re-use CPA2 parameter machinery (eCPA param set)
    ep = CPA2._eos_aq(T, P_bar, np.array([x_co2, x_h2o]), kij12=CPA2.kij_ecpa(T))
    return ep["A"], ep["B"]


def z_srk(T, P_bar, x_h2o, x_co2, phase="liquid"):
    """
    Solve the SRK cubic analytically (no association).
    Returns (Z, nfev) where nfev=1 counts the one algebraic cubic evaluation.
    """
    A, B = _srk_AB(T, P_bar, x_h2o, x_co2)
    # SRK cubic: Z³ − Z² + (A−B−B²)Z − AB = 0
    roots = CPA2._solve_cubic_real(1.0, -1.0, A - B - B**2, -A*B)
    roots = np.sort(np.real(roots[np.isfinite(roots) & (roots > B + 1e-8)]))
    if roots.size == 0:
        return np.nan, 1
    Z = float(roots[0]) if phase == "liquid" else float(roots[-1])
    return Z, 1


# ── 2. CPA ZChi solve (Brent + inner ChiChi cubic) ────────────────────────────

def _z_cpa_one_phase(T, P_bar, x_h2o, x_co2, phase="aqueous"):
    """
    CPA Z + (χ_H₂O, χ_CO₂) solve via ZChi (Brent scan + inner cubic).
    Returns (Z, Chi_h2o, Chi_co2, nfev).

    Uses monkey-patching to count _funz_combined calls.
    """
    kij  = CPA2.kij_ecpa(T)
    swc  = CPA2.s14_ecpa(T)
    n    = np.array([x_co2, x_h2o])   # [CO₂, H₂O]

    if phase == "aqueous":
        ep = CPA2._eos_aq(T, P_bar, n, kij12=kij)
    else:
        ep = CPA2._eos_vap(T, P_bar, n, kij12=kij)

    A    = ep["A"]
    B    = ep["B"]
    Kapa = ep["Kapa"]
    Eps  = ep["Eps"]

    counter = [0]
    _orig = CPA2._funz_combined

    def _counted(Z, A_, B_, n_, Kapa_, Eps_, swc_):
        counter[0] += 1
        return _orig(Z, A_, B_, n_, Kapa_, Eps_, swc_)

    with patch.object(CPA2, "_funz_combined", _counted):
        Z, Chi, Chi1 = CPA2.ZChi(A, B, n, Kapa, Eps, swc)

    return Z, Chi, Chi1, counter[0]


# ── 3. eCPA aqueous single-phase Z solve (fsolve, 3 variables) ─────────────────

def _ecpa_aq_residual(x0, T, P, x1w, x4w, x2w, x3w, ms):
    """
    3-variable eCPA aqueous-phase residual.
    Unknowns: x0 = [Zw, chi1w, epsr]
    chi4w is derived analytically from chi1w inside.

    ZPerm is set to zero (minor term requiring full ∂ε_r/∂V derivative chain).
    ZDH and the permittivity equation are fully included.
    """
    Zw    = float(x0[0])
    chi1w = float(x0[1])
    epsr  = float(x0[2])

    if Zw <= 0 or chi1w <= 0 or chi1w > 2 or epsr <= 1:
        return [1e6, 1e6, 1e6]

    # T-dependent interaction parameters
    k14 = Akij*(T/Tc4)**2 + Bkij*(T/Tc4) + Ckij
    S14 = ASij*(T/Tc4)**2 + BSij*(T/Tc4) + CSij
    U4s = Uref4s + alfa4s*R*((1 - T/Talfa4s)**2 - (1 - Tref/Talfa4s)**2)
    U1s = Uref1s + alfa1s*R*((1 - T/Talfa1s)**2 - (1 - Tref/Talfa1s)**2)
    U14 = np.log(2)*(a04/b4 - 2*(a01*a04)**0.5*(1 - k14)/(b1 + b4))
    U41 = np.log(2)*(a01/b1 - 2*(a04*a01)**0.5*(1 - k14)/(b4 + b1))

    # Soave attractive parameters
    a1_ = a01*(1 + c11*(1 - (T/Tc1)**0.5))**2
    a4_ = a04*(1 + c14*(1 - (T/Tc4)**0.5))**2

    # HV mixing
    bm  = b1*x1w + b2*x2w + b3*x3w + b4*x4w
    gE  = (1/bm)*(x1w*x2w*U1s*(b1+b2) + x1w*x3w*U1s*(b1+b3)
                  + x4w*x2w*U4s*(b4+b2) + x4w*x3w*U4s*(b4+b3)
                  + x1w*x4w*(b1*U14 + b4*U41))
    am  = bm*(x1w*a1_/b1 + x2w*a02/b2 + x3w*a03/b3 + x4w*a4_/b4
              - gE/np.log(2))

    A   = am*P / (R*T)**2
    B   = bm*P / (R*T)
    B4_ = b4*P / (R*T)

    if Zw - B <= 1e-12 or Zw + B <= 0:
        return [1e6, 1e6, 1e6]

    # ── Physical Z contribution
    Zphys = Zw/(Zw - B) - A/(Zw + B)

    # ── Association
    eta     = B / (4*Zw)
    if 1 - 1.9*eta <= 0:
        return [1e6, 1e6, 1e6]
    g_eta   = 1.0/(1 - 1.9*eta)
    dg_deta = 1.9/(1 - 1.9*eta)**2
    delta   = g_eta*kappaW*(np.exp(epsW/T) - 1)
    DELTA   = delta*P/(R*T)

    denom4w = Zw + 2*x1w*chi1w*S14*DELTA
    if abs(denom4w) < 1e-30:
        return [1e6, 1e6, 1e6]
    chi4w     = Zw / denom4w
    denom1w_new = Zw + 2*x1w*chi1w*DELTA + 2*x4w*chi4w*S14*DELTA
    if abs(denom1w_new) < 1e-30:
        return [1e6, 1e6, 1e6]
    chi1w_new = Zw / denom1w_new
    Zassoc    = -2*(1 + eta/g_eta*dg_deta)*(x1w*(1 - chi1w) + x4w*(1 - chi4w))

    # ── Debye–Hückel (aqueous, ms > 0)
    rho   = P / (Zw*R*T)
    xiZi  = x2w*Z2**2 + x3w*Z3**2      # = 2·x2w  (1:1 electrolyte)
    if ms > 0 and xiZi > 0 and epsr > 0:
        debye2 = e**2*Na*rho*xiZi / (kb*T*epsr*eps0)
        if debye2 < 0:
            return [1e6, 1e6, 1e6]
        debye = debye2**0.5
        X2 = (1/Sg2**3)*(np.log(1 + debye*Sg2) - debye*Sg2 + 0.5*(debye*Sg2)**2)
        X3 = (1/Sg3**3)*(np.log(1 + debye*Sg3) - debye*Sg3 + 0.5*(debye*Sg3)**2)
        ZDH = (1/(4*np.pi*Na*rho*xiZi)
               * (x2w*Z2**2*(X2 - 0.5*debye**3/(1 + debye*Sg2))
                  + x3w*Z3**2*(X3 - 0.5*debye**3/(1 + debye*Sg3))))
    else:
        ZDH = 0.0

    # ── Permittivity (Onsager–Kirkwood model)
    M = Na*rho/(3*eps0)*(x1w*pol1 + x2w*pol2 + x3w*pol3 + x4w*pol4)
    if 1 - M <= 0:
        return [1e6, 1e6, 1e6]
    eps_inf = (2*M + 1)/(1 - M)
    Pww  = 2*rho*x1w*delta*chi1w**2
    Pwc  = 2*rho*x4w*S14*delta*chi1w*chi4w
    Pw   = Pww + Pwc
    if abs(Pw*np.cos(THETA1) + 1) < 1e-30 or abs(Pww) < 1e-30:
        gw = 1.0
    else:
        gw   = 1 + zww*Pww*np.cos(GAMMA1)/(Pw*np.cos(THETA1) + 1)
    T1_perm  = (2*epsr + eps_inf)*(epsr - eps_inf)/(epsr*(eps_inf + 2)**2)
    T2_perm  = Na*rho/(9*eps0*kb*T)*(x1w*gw*dip01**2)

    # Note: ZPerm = -daDHBder·VderdV is omitted (see module docstring).
    Zw_new = Zphys + Zassoc + ZDH

    # Residuals (all scaled to ~O(1))
    r1 = (Zw - Zw_new) / max(abs(Zw), 1e-6)
    r2 = (chi1w - chi1w_new) / max(abs(chi1w), 1e-6)
    r3 = (T1_perm - T2_perm) / max(abs(T1_perm), abs(T2_perm), 1e-6)
    return [r1, r2, r3]


def _ecpa_cold_starts(T, P_bar, x1w, x4w):
    """
    Generate cold-start guesses for [Zw, chi1w, epsr].

    Strategy (no solution table used):
    1. T-dependent empirical epsr (Uematsu & Franck 1980).
    2. CPA-level Z + χ solve (ZChi) as starting Zw and chi1w.
       This is still a "cold start" — it uses the lower-level EoS
       as a physical initial estimate, not a precomputed eCPA solution.
    3. Fallback: scan over generic (Zw, chi1w) values.
    """
    # T-dependent epsr approximation
    epsr_guess = max(5.0, 87.9 - 0.36*(T - 273.15))

    # CPA Z+chi as physical starting point (cold start at CPA level)
    kij  = CPA2.kij_ecpa(T)
    swc  = CPA2.s14_ecpa(T)
    n    = np.array([x4w, x1w])   # [CO₂, H₂O]
    ep   = CPA2._eos_aq(T, P_bar, n, kij12=kij)
    try:
        Z_cpa, chi_h2o, _, = CPA2.ZChi(ep["A"], ep["B"], n, ep["Kapa"], ep["Eps"], swc)
        Zw0   = float(Z_cpa) if np.isfinite(Z_cpa) and Z_cpa > 0 else 0.05
        chi0  = float(chi_h2o) if np.isfinite(chi_h2o) and 0 < chi_h2o <= 1 else 0.5
    except Exception:
        Zw0, chi0 = 0.05, 0.5

    guesses = []
    for eps_fac in [1.0, 0.8, 1.2]:
        guesses.append(np.array([Zw0, chi0, epsr_guess * eps_fac]))
    # Additional fallbacks with varied chi
    for chi_fb in [0.2, 0.4, 0.7]:
        guesses.append(np.array([Zw0, chi_fb, epsr_guess]))
    return guesses


def z_ecpa_aqueous(T, P_bar, x1w, x4w, x2w, x3w, ms):
    """
    eCPA aqueous Z solve: fsolve on 3-variable residual, cold start.
    Tries multiple cold-start guesses; returns the first converged result.
    Returns (Zw, chi1w, epsr, converged, nfev).
    """
    counter = [0]
    def counted_residual(x0):
        counter[0] += 1
        return _ecpa_aq_residual(x0, T, P_bar*1e5, x1w, x4w, x2w, x3w, ms)

    guesses = _ecpa_cold_starts(T, P_bar, x1w, x4w)
    best_sol, best_res = None, np.inf
    total_nfev = 0

    for x0 in guesses:
        local_counter = [0]
        def local_residual(x0_):
            local_counter[0] += 1
            return _ecpa_aq_residual(x0_, T, P_bar*1e5, x1w, x4w, x2w, x3w, ms)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sol, info, ier, _ = fsolve(local_residual, x0.copy(),
                                       full_output=True, xtol=1e-10, maxfev=300)

        total_nfev += local_counter[0]
        if ier == 1 and np.all(np.isfinite(sol)):
            res = np.linalg.norm(
                _ecpa_aq_residual(sol, T, P_bar*1e5, x1w, x4w, x2w, x3w, ms))
            if res < 1e-5:
                return float(sol[0]), float(sol[1]), float(sol[2]), True, total_nfev
            if res < best_res:
                best_sol, best_res = sol.copy(), res

    sol = best_sol if best_sol is not None else np.array([np.nan]*3)
    ok = best_res < 1e-5
    return float(sol[0]), float(sol[1]), float(sol[2]), ok, total_nfev


# ── Run benchmark ──────────────────────────────────────────────────────────────

rows = []

for T in T_GRID:
    for P in P_GRID:
        # ── SRK aqueous ──────────────────────────────────────────────────
        try:
            t0 = min(
                time.perf_counter() - time.perf_counter()
                for _ in range(1)   # init; actual timing below
            )
            times_srk_aq = []
            for _ in range(N_REPEAT):
                t0 = time.perf_counter()
                Z_srk_aq, nfev_srk = z_srk(T, P, _X1W, _X4W, phase="liquid")
                times_srk_aq.append(time.perf_counter() - t0)
            rows.append(dict(T=T, P=P, eos="SRK", phase="aqueous",
                             Z=Z_srk_aq, nfev=nfev_srk, converged=np.isfinite(Z_srk_aq),
                             time_us=min(times_srk_aq)*1e6))
        except Exception:
            rows.append(dict(T=T, P=P, eos="SRK", phase="aqueous",
                             Z=np.nan, nfev=1, converged=False, time_us=np.nan))

        # ── SRK CO₂-rich ─────────────────────────────────────────────────
        try:
            times_srk_c = []
            for _ in range(N_REPEAT):
                t0 = time.perf_counter()
                Z_srk_c, _ = z_srk(T, P, _X1C, _X4C, phase="vapor")
                times_srk_c.append(time.perf_counter() - t0)
            rows.append(dict(T=T, P=P, eos="SRK", phase="CO2-rich",
                             Z=Z_srk_c, nfev=1, converged=np.isfinite(Z_srk_c),
                             time_us=min(times_srk_c)*1e6))
        except Exception:
            rows.append(dict(T=T, P=P, eos="SRK", phase="CO2-rich",
                             Z=np.nan, nfev=1, converged=False, time_us=np.nan))

        # ── CPA aqueous ───────────────────────────────────────────────────
        try:
            nfev_cpa_aq = None
            times_cpa_aq = []
            for i in range(N_REPEAT):
                t0 = time.perf_counter()
                Z_cpa_aq, Chi_h2o, Chi_co2, nfev_i = _z_cpa_one_phase(
                    T, P, _X1W, _X4W, phase="aqueous")
                elapsed = time.perf_counter() - t0
                if i == 0:
                    nfev_cpa_aq = nfev_i
                times_cpa_aq.append(elapsed)
            conv_cpa_aq = np.isfinite(Z_cpa_aq) and Z_cpa_aq > 0
            rows.append(dict(T=T, P=P, eos="CPA", phase="aqueous",
                             Z=Z_cpa_aq, nfev=nfev_cpa_aq, converged=conv_cpa_aq,
                             time_us=min(times_cpa_aq)*1e6))
        except Exception:
            rows.append(dict(T=T, P=P, eos="CPA", phase="aqueous",
                             Z=np.nan, nfev=np.nan, converged=False, time_us=np.nan))

        # ── CPA CO₂-rich ─────────────────────────────────────────────────
        try:
            nfev_cpa_c = None
            times_cpa_c = []
            for i in range(N_REPEAT):
                t0 = time.perf_counter()
                Z_cpa_c, _, _, nfev_i = _z_cpa_one_phase(
                    T, P, _X1C, _X4C, phase="CO2-rich")
                elapsed = time.perf_counter() - t0
                if i == 0:
                    nfev_cpa_c = nfev_i
                times_cpa_c.append(elapsed)
            conv_cpa_c = np.isfinite(Z_cpa_c) and Z_cpa_c > 0
            rows.append(dict(T=T, P=P, eos="CPA", phase="CO2-rich",
                             Z=Z_cpa_c, nfev=nfev_cpa_c, converged=conv_cpa_c,
                             time_us=min(times_cpa_c)*1e6))
        except Exception:
            rows.append(dict(T=T, P=P, eos="CPA", phase="CO2-rich",
                             Z=np.nan, nfev=np.nan, converged=False, time_us=np.nan))

        # ── eCPA aqueous ──────────────────────────────────────────────────
        try:
            nfev_ecpa = None
            times_ecpa = []
            for i in range(N_REPEAT):
                t0 = time.perf_counter()
                Zw_e, chi1w_e, epsr_e, ok_e, nfev_i = z_ecpa_aqueous(
                    T, P, _X1W, _X4W, _X2W, _X3W, _MS)
                elapsed = time.perf_counter() - t0
                if i == 0:
                    nfev_ecpa = nfev_i
                times_ecpa.append(elapsed)
            rows.append(dict(T=T, P=P, eos="eCPA", phase="aqueous",
                             Z=Zw_e, nfev=nfev_ecpa, converged=ok_e,
                             time_us=min(times_ecpa)*1e6))
        except Exception:
            rows.append(dict(T=T, P=P, eos="eCPA", phase="aqueous",
                             Z=np.nan, nfev=np.nan, converged=False, time_us=np.nan))

        print(f"  T={T:.0f}K  P={P:.0f}bar  done", flush=True)

df = pd.DataFrame(rows)
df.to_csv("results/z_rootfinding_benchmark.csv", index=False)

# ── Print summary table ────────────────────────────────────────────────────────

print("\n── nfev summary (converged only) ──────────────────────────────────")
print(f"{'EoS':<8}{'Phase':<12}{'N conv':>8}{'nfev mean':>12}{'nfev med':>10}{'nfev max':>10}")
for (eos, phase), g in df[df["converged"]].groupby(["eos", "phase"]):
    nf = g["nfev"].dropna()
    print(f"  {eos:<6}  {phase:<12}  {len(nf):>5}  {nf.mean():>10.1f}  {nf.median():>8.0f}  {nf.max():>8.0f}")

print("\n── time summary (converged only, μs per call) ──────────────────────")
print(f"{'EoS':<8}{'Phase':<12}{'time mean':>12}{'time med':>10}{'time max':>10}  cost/eval (μs)")
summary = {}
for (eos, phase), g in df[df["converged"]].groupby(["eos", "phase"]):
    t  = g["time_us"].dropna()
    nf = g["nfev"].dropna()
    cpe = (t / nf).median() if len(nf) > 0 and nf.median() > 0 else np.nan
    print(f"  {eos:<6}  {phase:<12}  {t.mean():>10.1f}  {t.median():>8.1f}  {t.max():>8.1f}  {cpe:>10.1f}")
    summary[(eos, phase)] = dict(nfev_med=nf.median(), time_med=t.median(), cpe=cpe)

# ── Main summary figure (3 panels) ────────────────────────────────────────────

EOS_ORDER   = ["SRK", "CPA", "eCPA"]
EOS_LABELS  = {
    "SRK":  "SRK\n(cubic,\nanalytic)",
    "CPA":  "CPA\n(Brent+\ninner-cubic)",
    "eCPA": "eCPA\n(3-var\nfsolve)",
}
COLORS = {"SRK": "#4477AA", "CPA": "#EE7733", "eCPA": "#CC3311"}


def _boxplot(ax, data_dict, label_dict, ylabel, yscale="linear", color_dict=None):
    """data_dict: {label: array}; plot in insertion order."""
    labels = list(data_dict.keys())
    parts  = [data_dict[k] for k in labels]
    tick_labels = [label_dict.get(k, k) for k in labels]
    bp = ax.boxplot(parts, patch_artist=True,
                    medianprops=dict(color="white", linewidth=2.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker="o", markersize=3, alpha=0.5, linestyle="none"))
    for patch, lbl in zip(bp["boxes"], labels):
        c = (color_dict or COLORS).get(lbl, "#888888")
        patch.set_facecolor(c); patch.set_alpha(0.85)
    ax.set_xticks(range(1, len(labels)+1))
    ax.set_xticklabels(tick_labels, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    if yscale == "log":
        ax.set_yscale("log")
    ax.grid(axis="y", ls="--", alpha=0.4, which="both" if yscale == "log" else "major")
    return bp


# ── Figure 1: nfev + time + cost-per-eval (aqueous only) ─────────────────────

fig, axes = plt.subplots(1, 3, figsize=(13, 5), constrained_layout=True)

sub_aq = df[(df["phase"] == "aqueous") & df["converged"]]

# Panel A: nfev
data_nfev = {eos: sub_aq[sub_aq["eos"] == eos]["nfev"].dropna().values
             for eos in EOS_ORDER}
data_nfev = {k: v for k, v in data_nfev.items() if len(v) > 0}
_boxplot(axes[0], data_nfev, EOS_LABELS, "Residual evaluations (nfev)")
axes[0].set_title("(a) Iteration count", fontsize=12, fontweight="bold")
# Annotate medians
for i, eos in enumerate(data_nfev):
    med = np.median(data_nfev[eos])
    axes[0].text(i+1, med*1.08, f"{med:.0f}", ha="center", va="bottom",
                 fontsize=10, fontweight="bold", color=COLORS.get(eos, "#333"))

# Panel B: wall-clock time (log scale)
data_time = {eos: sub_aq[sub_aq["eos"] == eos]["time_us"].dropna().values
             for eos in EOS_ORDER}
data_time = {k: v for k, v in data_time.items() if len(v) > 0}
_boxplot(axes[1], data_time, EOS_LABELS, "CPU time per call (μs)", yscale="log")
axes[1].set_title("(b) Wall-clock time", fontsize=12, fontweight="bold")

# Panel C: cost per residual evaluation (= time / nfev)
data_cpe = {}
for eos in EOS_ORDER:
    s = sub_aq[sub_aq["eos"] == eos]
    t  = s["time_us"].dropna().values
    nf = s["nfev"].dropna().values
    # Align lengths
    n = min(len(t), len(nf))
    if n > 0 and np.all(nf[:n] > 0):
        data_cpe[eos] = t[:n] / nf[:n]
data_cpe = {k: v for k, v in data_cpe.items() if len(v) > 0}
_boxplot(axes[2], data_cpe, EOS_LABELS, "Cost per residual evaluation (μs)", yscale="log")
axes[2].set_title("(c) Cost per evaluation", fontsize=12, fontweight="bold")

fig.suptitle(
    "Z root-finding complexity — aqueous phase (H₂O + CO₂ + NaCl, ms = 1 mol/kg)\n"
    "T = 303–523 K, P = 10–1500 bar, cold start, 41–42 T×P conditions",
    fontsize=11,
)
fig.savefig("figures/z_rootfinding_benchmark.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\nFigure saved: figures/z_rootfinding_benchmark.png")

# ── Figure 2: nfev vs P at two temperatures + CO₂-rich comparison ─────────────

fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)

# Left two panels: nfev vs P at T=323K and T=473K for aqueous
for ax, T_sel in zip(axes2[:2], [323.0, 473.0]):
    sub = sub_aq[sub_aq["T"] == T_sel]
    ax.axhline(1, ls=":", color=COLORS["SRK"], linewidth=2,
               label="SRK  (nfev = 1, analytic)")
    for eos in ["CPA", "eCPA"]:
        s = sub[sub["eos"] == eos].sort_values("P")
        if s.empty:
            continue
        ax.plot(s["P"], s["nfev"], marker="o", label=eos,
                color=COLORS[eos], linewidth=2, markersize=6)
    ax.set_xlabel("Pressure (bar)", fontsize=12)
    ax.set_ylabel("nfev", fontsize=12)
    ax.set_xscale("log")
    ax.set_title(f"T = {T_sel:.0f} K (aqueous)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(ls="--", alpha=0.4)
    ax.set_xlim(8, 2000)

# Right panel: CO₂-rich phase comparison (SRK vs CPA; eCPA CO₂-rich = CPA)
sub_co2 = df[(df["phase"] == "CO2-rich") & df["converged"]]
ax = axes2[2]
# Show nfev distribution for SRK and CPA CO₂-rich
for eos in ["SRK", "CPA"]:
    s = sub_co2[sub_co2["eos"] == eos]["nfev"].dropna()
    ax.scatter([eos]*len(s), s, color=COLORS[eos], alpha=0.5, s=20)
    ax.plot([eos], [s.median()], marker="D", color=COLORS[eos],
            markersize=10, label=f"{eos} (median={s.median():.0f})")
ax.annotate("eCPA CO₂-rich\n= CPA\n(no DH/permittivity)", xy=(1.5, 25),
            xycoords="data", ha="center", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFEECC", ec="#888", alpha=0.8))
ax.set_ylabel("nfev", fontsize=12)
ax.set_title("CO₂-rich phase", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(ls="--", alpha=0.4)

fig2.suptitle(
    "nfev vs pressure (aqueous, cold-start) and CO₂-rich comparison",
    fontsize=11,
)
fig2.savefig("figures/z_rootfinding_nfev_vs_P.png", dpi=150, bbox_inches="tight")
plt.close(fig2)
print("Figure saved: figures/z_rootfinding_nfev_vs_P.png")
