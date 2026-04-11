"""
_bench_cpa_vs_srk.py
====================
Per-SSI-iteration cost benchmark: CPA vs. SRK-equivalent, across every 10th
two-phase point of the 9825-point scan grid.

For each sampled (T, P) condition the script:

  1. Retrieves converged (x, y) from the extended ELV table.
  2. Computes EOS parameters for both phases.
  3. Times each cost component for ONE SSI iteration:
       CPA:  _eos_*  +  ZChi Newton (warm-start)  +  _lnphi_*
       SRK:  _eos_*  +  _solve_cubic_real          +  _lnphi_* (Chi=Chi1=1)
  4. Counts the actual number of 3×3 Newton iterations used by ZChi.

Outputs
-------
  bench_cpa_vs_srk.npz   — raw arrays for all sampled points
  (summary printed to stdout)
"""

import numpy as np
import pandas as pd
import time
import CPA

# ── Load scan grid and ELV table ─────────────────────────────────────────────
print("Loading scan grid and ELV table …")
d          = np.load("scan_results_extended.npz", allow_pickle=True)
T_grid     = d["T_grid"]
P_grid     = d["P_grid"]
z_grid     = d["z_grid"]
phase_id   = d["phase_id"]          # (nT, nP, nz)

df = pd.read_parquet("CPA_ELV_all_extended.parquet")

# Index the table by (T_K, P_bar) for fast lookup
df_idx = df.set_index(["T_K", "P_bar"])

comps = CPA.make_components_co2_h2o()
Omega, Tc, Pc, Mw = comps["Omega"], comps["Tc"], comps["Pc"], comps["Mw"]

# ── ZChi Newton with iteration counter ───────────────────────────────────────
def _ZChi_newton_counted(A, B, n, Kapa, Eps, swc, Z0, Chi0, Chi10):
    """
    Identical to CPA._ZChi_newton but returns (Z, Chi, Chi1, n_newton_iters).
    n_newton_iters = number of Newton steps applied (0 if already converged).
    """
    n_arr = np.asarray(n, dtype=float).reshape(2)
    nc, nw = n_arr[0], n_arr[1]
    Z    = float(Z0);  Chi  = float(Chi0);  Chi1 = float(Chi10)
    _no_cross = abs(swc) < 1e-30
    max_resid = np.inf
    nit = 0

    for nit in range(CPA._ZCHI_NEWTON_MAXITER):
        if Z <= B:
            raise RuntimeError("Z <= B")
        eta   = B / (4.0 * Z)
        g     = CPA._g(eta);  dg = CPA._dgdeta(eta);  d2g = CPA._d2gdeta2(eta)
        delta  = g * Kapa * np.expm1(Eps)
        delta1 = swc * delta
        dgog      = dg / g
        dlogd_dZ  = dgog * (-B / (4.0 * Z * Z))
        ZmB = Z - B;  ZpB = Z + B
        h_val = 1.0 + eta * dgog
        dh    = dgog + eta * (d2g / g - dgog * dgog)
        Sigma = nw * (Chi - 1.0) + nc * (Chi1 - 1.0)
        F1    = Z - Z / ZmB + A / ZpB - 2.0 * h_val * Sigma

        if _no_cross:
            c2b = 2.0 * nw * Z * delta
            F2  = c2b * Chi * Chi + Z * Z * Chi - Z * Z
            max_resid = max(abs(F1), abs(F2))
            if max_resid < CPA._ZCHI_NEWTON_TOL:
                break
            dc2b_dZ = 2.0 * nw * delta * (1.0 + Z * dlogd_dZ)
            J11 = 1.0 + B/ZmB**2 - A/ZpB**2 + dh*B/(2.0*Z*Z)*Sigma
            J12 = -2.0 * h_val * nw
            J21 = dc2b_dZ * Chi * Chi + 2.0 * Z * Chi - 2.0 * Z
            J22 = 2.0 * c2b * Chi + Z * Z
            dv  = np.linalg.solve(np.array([[J11,J12],[J21,J22]]),
                                  -np.array([F1, F2]))
            Z    = max(Z + float(dv[0]), B + 1e-10)
            Chi  = float(np.clip(Chi + float(dv[1]), 0.0, 1.0))
            Chi1 = 1.0
        else:
            c3 = 4.0*nw*nw*delta*delta1
            c2 = 2.0*nw*Z*(delta+delta1)
            c1 = 2.0*delta1*Z*(nc-nw) + Z*Z
            c0 = -(Z*Z)
            F2 = c3*Chi**3 + c2*Chi**2 + c1*Chi + c0
            D  = Z + 2.0*nw*Chi*delta1
            if D <= 0.0:
                raise RuntimeError("D <= 0")
            F3 = Chi1*D - Z
            max_resid = max(abs(F1), abs(F2), abs(F3))
            if max_resid < CPA._ZCHI_NEWTON_TOL:
                break
            J11 = 1.0+B/ZmB**2-A/ZpB**2+dh*B/(2.0*Z*Z)*Sigma
            J12 = -2.0*h_val*nw;  J13 = -2.0*h_val*nc
            dc3 = c3*2.0*dlogd_dZ
            dc2 = 2.0*nw*(delta+delta1) + c2*dlogd_dZ
            dc1 = 2.0*delta1*(nc-nw) + 2.0*delta1*Z*(nc-nw)*dlogd_dZ + 2.0*Z
            dc0 = -2.0*Z
            J21 = dc3*Chi**3+dc2*Chi**2+dc1*Chi+dc0
            J22 = 3.0*c3*Chi**2+2.0*c2*Chi+c1
            dD  = 1.0+2.0*nw*Chi*delta1*dlogd_dZ
            J31 = Chi1*dD-1.0;  J32 = Chi1*2.0*nw*delta1;  J33 = D
            dv  = np.linalg.solve(
                np.array([[J11,J12,J13],[J21,J22,0.0],[J31,J32,J33]]),
                -np.array([F1, F2, F3]))
            Z    = max(Z+float(dv[0]), B+1e-10)
            Chi  = float(np.clip(Chi+float(dv[1]), 0.0, 1.0))
            _hi  = 10.0 if delta1 < 0.0 else 1.0
            Chi1 = float(np.clip(Chi1+float(dv[2]), 0.0, _hi))
    else:
        raise RuntimeError("ZChi Newton did not converge")

    return Z, Chi, Chi1, nit   # nit = number of steps applied

# ── SRK cubic root selection ──────────────────────────────────────────────────
def _srk_Z(A, B, prefer_liquid):
    roots = CPA._solve_cubic_real(1.0, -1.0, A - B - B**2, -A * B)
    phys  = [r for r in roots if r > B and np.isfinite(r)]
    if not phys:
        return float("nan")
    return min(phys) if prefer_liquid else max(phys)

# ── Main benchmark loop ───────────────────────────────────────────────────────
STRIDE = 10          # every 10th two-phase point
N_REPS = 2000        # timing repetitions per cost component

two_ph_pts = []      # (iT, iP, iz) indices of two-phase points
for iT in range(len(T_grid)):
    for iP in range(len(P_grid)):
        for iz in range(len(z_grid)):
            if phase_id[iT, iP, iz] == 4:
                two_ph_pts.append((iT, iP, iz))

sample = two_ph_pts[::STRIDE]
n_sample = len(sample)
print(f"  {len(two_ph_pts)} two-phase points, sampling every {STRIDE}th → {n_sample} points\n")

# Result arrays
t_eos_cpa  = np.full(n_sample, np.nan)   # µs: _eos_aq + _eos_vap (CPA params)
t_zchi_cpa = np.full(n_sample, np.nan)   # µs: ZChi Newton × 2 phases (warm-start)
t_lnphi_cpa= np.full(n_sample, np.nan)   # µs: _lnphi_aq + _lnphi_vap (full assoc)
t_eos_srk  = np.full(n_sample, np.nan)   # µs: _eos_aq + _eos_vap (same, Kapa→0)
t_zchi_srk = np.full(n_sample, np.nan)   # µs: _solve_cubic_real × 2 phases
t_lnphi_srk= np.full(n_sample, np.nan)   # µs: _lnphi_aq/_vap with Chi=Chi1=1

nit_aq_warm  = np.full(n_sample, np.nan)  # ZChi Newton iters, aq,  warm start
nit_vap_warm = np.full(n_sample, np.nan)  # ZChi Newton iters, vap, warm start
nit_aq_cold  = np.full(n_sample, np.nan)  # ZChi Newton iters, aq,  perturbed start
nit_vap_cold = np.full(n_sample, np.nan)  # ZChi Newton iters, vap, perturbed start
zchi_fallback= np.zeros(n_sample, dtype=bool)  # True if any scan+Brent fallback

T_arr = np.full(n_sample, np.nan)
P_arr = np.full(n_sample, np.nan)

n_done = 0
n_skip = 0

for k, (iT, iP, iz) in enumerate(sample):
    T = float(T_grid[iT])
    P = float(P_grid[iP])
    T_arr[k] = T;  P_arr[k] = P

    kij = CPA.kij_ecpa(T)
    swc = CPA.s14_ecpa(T)

    # Get equilibrium compositions from table
    try:
        row = df_idx.loc[(T, P)]
        if isinstance(row, pd.DataFrame):   # multiple rows at same (T,P)
            row = row.iloc[0]
        xw_W = float(row["xw_W"])      # x_H2O in aqueous phase
        xw_C = float(row["xw_C"])      # x_H2O in CO2-rich phase
        Z_W  = float(row["Z_W"])
        Z_C  = float(row["Z_C"])
        chi_W= float(row["chiw_W"])
        chi_C= float(row["chiw_C"])
    except (KeyError, IndexError):
        n_skip += 1
        continue

    x = np.array([1.0 - xw_W, xw_W])   # [CO2, H2O] aqueous
    y = np.array([1.0 - xw_C, xw_C])   # [CO2, H2O] CO2-rich

    # ── CPA EOS parameters ────────────────────────────────────────────────
    ep_aq  = CPA._eos_aq (T, P, x, kij)
    ep_vap = CPA._eos_vap(T, P, y, kij)

    # Converged ZChi values from table (warm start)
    chi1_W = CPA._chi1_from_Z_chi(Z_W, chi_W, ep_aq["B"], x[1],
                                    ep_aq["Kapa"], ep_aq["Eps"], swc)
    chi1_C = CPA._chi1_from_Z_chi(Z_C, chi_C, ep_vap["B"], y[1],
                                    ep_vap["Kapa"], ep_vap["Eps"], swc)

    # ── Iteration counts ──────────────────────────────────────────────────
    # (a) From exact table warm start (best case — simulates 2nd+ SSI iter)
    try:
        _, _, _, nit_w_aq  = _ZChi_newton_counted(
            ep_aq["A"], ep_aq["B"], x,
            ep_aq["Kapa"], ep_aq["Eps"], swc, Z_W, chi_W, chi1_W)
        _, _, _, nit_w_vap = _ZChi_newton_counted(
            ep_vap["A"], ep_vap["B"], y,
            ep_vap["Kapa"], ep_vap["Eps"], swc, Z_C, chi_C, chi1_C)
        nit_aq_warm[k]  = nit_w_aq
        nit_vap_warm[k] = nit_w_vap
    except Exception:
        pass

    # (b) From 0.5%-perturbed warm start (simulates typical SSI inter-step gap)
    try:
        Z_Wp  = Z_W  * 1.005
        Z_Cp  = Z_C  * 1.005
        chi_Wp = min(chi_W  * 1.005, 1.0)
        chi_Cp = min(chi_C  * 1.005, 1.0)
        chi1_Wp = CPA._chi1_from_Z_chi(Z_Wp, chi_Wp, ep_aq["B"], x[1],
                                         ep_aq["Kapa"], ep_aq["Eps"], swc)
        chi1_Cp = CPA._chi1_from_Z_chi(Z_Cp, chi_Cp, ep_vap["B"], y[1],
                                         ep_vap["Kapa"], ep_vap["Eps"], swc)
        _, _, _, nit_c_aq  = _ZChi_newton_counted(
            ep_aq["A"], ep_aq["B"], x,
            ep_aq["Kapa"], ep_aq["Eps"], swc, Z_Wp, chi_Wp, chi1_Wp)
        _, _, _, nit_c_vap = _ZChi_newton_counted(
            ep_vap["A"], ep_vap["B"], y,
            ep_vap["Kapa"], ep_vap["Eps"], swc, Z_Cp, chi_Cp, chi1_Cp)
        nit_aq_cold[k]  = nit_c_aq
        nit_vap_cold[k] = nit_c_vap
    except Exception:
        pass

    # ── Timing: CPA components ────────────────────────────────────────────
    def _time(fn):
        t0 = time.perf_counter()
        for _ in range(N_REPS): fn()
        return (time.perf_counter() - t0) / N_REPS * 1e6   # µs

    try:
        t_eos_cpa[k] = _time(lambda: (CPA._eos_aq(T, P, x, kij),
                                       CPA._eos_vap(T, P, y, kij)))

        t_zchi_cpa[k] = _time(lambda: (
            CPA._ZChi_newton(ep_aq["A"], ep_aq["B"], x,
                              ep_aq["Kapa"], ep_aq["Eps"], swc, Z_W, chi_W, chi1_W),
            CPA._ZChi_newton(ep_vap["A"], ep_vap["B"], y,
                              ep_vap["Kapa"], ep_vap["Eps"], swc, Z_C, chi_C, chi1_C)))

        t_lnphi_cpa[k] = _time(lambda: (
            CPA._lnphi_aq (ep_aq,  Z_W, chi_W,  chi1_W),
            CPA._lnphi_vap(ep_vap, Z_C, chi_C,  chi1_C)))
    except Exception:
        n_skip += 1
        continue

    # ── Timing: SRK-equivalent components ────────────────────────────────
    # EOS params are the same infrastructure; only Kapa is zeroed for Z solve
    # (EOS param cost is the same — we don't patch _eos_aq for this timing)
    t_eos_srk[k] = t_eos_cpa[k]   # same mixing rule overhead

    # SRK cubic solve (both phases; prefer_liquid based on xH2O)
    liq_aq  = x[1] > 0.5
    liq_vap = y[1] > 0.5
    t_zchi_srk[k] = _time(lambda: (
        _srk_Z(ep_aq["A"],  ep_aq["B"],  liq_aq),
        _srk_Z(ep_vap["A"], ep_vap["B"], liq_vap)))

    # SRK lnphi = CPA lnphi with Chi=Chi1=1 (no Wertheim terms)
    t_lnphi_srk[k] = _time(lambda: (
        CPA._lnphi_aq (ep_aq,  Z_W, 1.0, 1.0),
        CPA._lnphi_vap(ep_vap, Z_C, 1.0, 1.0)))

    n_done += 1
    if n_done % 50 == 0:
        print(f"  {n_done}/{n_sample} done …", flush=True)

print(f"\n  Done: {n_done} points timed, {n_skip} skipped (T/P not in table)\n")

# ── Summary statistics ────────────────────────────────────────────────────────
ok = np.isfinite(t_zchi_cpa)

t_iter_cpa = t_eos_cpa + t_zchi_cpa + t_lnphi_cpa
t_iter_srk = t_eos_srk + t_zchi_srk + t_lnphi_srk
ratio_iter  = t_iter_cpa / t_iter_srk

ratio_zchi  = t_zchi_cpa / t_zchi_srk

all_nit_warm = np.concatenate([nit_aq_warm[ok], nit_vap_warm[ok]])
all_nit_warm = all_nit_warm[np.isfinite(all_nit_warm)]
all_nit_cold = np.concatenate([nit_aq_cold[ok], nit_vap_cold[ok]])
all_nit_cold = all_nit_cold[np.isfinite(all_nit_cold)]

print("=" * 65)
print("PER-ITERATION COST: CPA vs. SRK-equivalent")
print(f"  Sampled points: {ok.sum()}")
print()
print("  Component timing (both phases combined, µs):")
print(f"    EOS parameters   CPA: {np.median(t_eos_cpa[ok]):.2f}  SRK: {np.median(t_eos_srk[ok]):.2f}")
print(f"    ZChi / cubic-Z   CPA: {np.median(t_zchi_cpa[ok]):.2f}  SRK: {np.median(t_zchi_srk[ok]):.2f}  ratio: {np.median(ratio_zchi[ok]):.2f}x")
print(f"    ln(phi)          CPA: {np.median(t_lnphi_cpa[ok]):.2f}  SRK: {np.median(t_lnphi_srk[ok]):.2f}")
print()
print(f"  Per-iteration total (µs):")
print(f"    CPA:  median={np.median(t_iter_cpa[ok]):.2f}  mean={np.mean(t_iter_cpa[ok]):.2f}  p90={np.percentile(t_iter_cpa[ok],90):.2f}")
print(f"    SRK:  median={np.median(t_iter_srk[ok]):.2f}  mean={np.mean(t_iter_srk[ok]):.2f}  p90={np.percentile(t_iter_srk[ok],90):.2f}")
print(f"    CPA/SRK ratio:  median={np.median(ratio_iter[ok]):.2f}x  mean={np.mean(ratio_iter[ok]):.2f}x  p90={np.percentile(ratio_iter[ok],90):.2f}x")
print()
print("  ZChi Newton iterations (from exact warm start):")
print(f"    mean={all_nit_warm.mean():.2f}  median={np.median(all_nit_warm):.0f}  max={all_nit_warm.max():.0f}")
print("  ZChi Newton iterations (from 0.5%-perturbed warm start):")
print(f"    mean={all_nit_cold.mean():.2f}  median={np.median(all_nit_cold):.0f}  max={all_nit_cold.max():.0f}")
print("=" * 65)

# ── Save ──────────────────────────────────────────────────────────────────────
np.savez("bench_cpa_vs_srk.npz",
         T=T_arr, P=P_arr,
         t_eos_cpa=t_eos_cpa, t_zchi_cpa=t_zchi_cpa, t_lnphi_cpa=t_lnphi_cpa,
         t_eos_srk=t_eos_srk, t_zchi_srk=t_zchi_srk, t_lnphi_srk=t_lnphi_srk,
         nit_aq_warm=nit_aq_warm, nit_vap_warm=nit_vap_warm,
         nit_aq_cold=nit_aq_cold, nit_vap_cold=nit_vap_cold)
print("  Saved bench_cpa_vs_srk.npz")
