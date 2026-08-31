"""
_scan_table_warmstart.py — Add 5th strategy (table warm-start) to extended scan.

Loads the eCPA solution table (CPA_ELV_all.parquet at ms=0) to extract K-values,
uses them as initial guesses for CPA.tie_line_two_comp, and compares to the
4 cold-start strategies already in scan_results_extended.npz.
"""
import time
import sys
import numpy as np
import pandas as pd

import CPA

# ═══════════════════════════════════════════════════════════════════════════════
# Load existing scan results
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading existing scan results...")
d = np.load("scan_results_extended.npz")
T_grid = d["T_grid"]
P_grid = d["P_grid"]
z_grid = d["z_grid"]
phase_id = d["phase_id"]
flash_conv_old = d["flash_conv"]      # (86, 18, 19, 4)
flash_iter_old = d["flash_iter"]
flash_resid_old = d["flash_resid"]
flash_m_old = d["flash_m"]
STRAT_NAMES_OLD = list(d["flash_strategy_names"])

nT, nP, nz = len(T_grid), len(P_grid), len(z_grid)
mask_2ph = (phase_id == 4)
n_2ph = int(mask_2ph.sum())
print(f"Grid: {nT}T x {nP}P x {nz}z = {nT*nP*nz:,} points, {n_2ph} two-phase")

# ═══════════════════════════════════════════════════════════════════════════════
# Load solution table and build K-value lookup
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading solution table (CPA_ELV_all.parquet at ms=0)...")
df = pd.read_parquet("results/CPA_ELV_all.parquet")

# Compute K-values: K_CO2 = (1 - xw_C) / (1 - xw_W), K_H2O = xw_C / xw_W
df["K_CO2"] = (1.0 - df["xw_C"]) / (1.0 - df["xw_W"])
df["K_H2O"] = df["xw_C"] / df["xw_W"]

# Group by T for fast lookup
table_temps = np.sort(df["T_K"].unique())
table_by_T = {}
for T_val, grp in df.groupby("T_K"):
    grp_sorted = grp.sort_values("P_bar")
    table_by_T[T_val] = (grp_sorted["P_bar"].values,
                         grp_sorted["K_CO2"].values,
                         grp_sorted["K_H2O"].values)

print(f"Solution table: {len(df)} rows, T={table_temps[0]}-{table_temps[-1]}K")


def lookup_K(T, P):
    """Interpolate K-values from solution table at (T, P).

    Uses bilinear interpolation: bracket T between two table temperatures,
    interpolate in P at each, then interpolate between them in T.
    Returns (K_CO2, K_H2O) or None if outside table range.
    """
    if T < table_temps[0] or T > table_temps[-1]:
        return None

    # Find bracketing temperatures
    idx = np.searchsorted(table_temps, T)
    if idx == 0:
        T_lo = T_hi = table_temps[0]
    elif idx >= len(table_temps):
        T_lo = T_hi = table_temps[-1]
    else:
        T_lo = table_temps[idx - 1]
        T_hi = table_temps[idx]

    def interp_at_T(T_key, P_target):
        if T_key not in table_by_T:
            return None
        P_arr, K1_arr, K2_arr = table_by_T[T_key]
        if P_target < P_arr[0] or P_target > P_arr[-1]:
            return None
        K1 = np.interp(P_target, P_arr, K1_arr)
        K2 = np.interp(P_target, P_arr, K2_arr)
        return K1, K2

    if T_lo == T_hi:
        return interp_at_T(T_lo, P)

    r_lo = interp_at_T(T_lo, P)
    r_hi = interp_at_T(T_hi, P)

    if r_lo is None and r_hi is None:
        return None
    if r_lo is None:
        return r_hi
    if r_hi is None:
        return r_lo

    # Linear interpolation in T
    w = (T - T_lo) / (T_hi - T_lo)
    K1 = r_lo[0] * (1 - w) + r_hi[0] * w
    K2 = r_lo[1] * (1 - w) + r_hi[1] * w
    return K1, K2


# ═══════════════════════════════════════════════════════════════════════════════
# Run 5th strategy: table warm-start + accelerated SSI
# ═══════════════════════════════════════════════════════════════════════════════
comps = CPA.make_components_co2_h2o()
Omega, Tc, Pc, Mw = comps["Omega"], comps["Tc"], comps["Pc"], comps["Mw"]

shape = (nT, nP, nz)
flash_conv_tbl = np.full(shape, False)
flash_iter_tbl = np.zeros(shape, dtype=np.int16)
flash_resid_tbl = np.full(shape, np.nan)
flash_m_tbl = np.full(shape, np.nan)
table_avail = np.full(shape, False)  # track where table K was available

print("\nRunning table warm-start strategy on all grid points...")
t_start = time.time()
n_done = 0

for iT, T in enumerate(T_grid):
    kij = CPA.kij_ecpa(T)
    swc = CPA.s14_ecpa(T)

    for iP, P in enumerate(P_grid):
        K_from_table = lookup_K(T, P)

        for iz, zc in enumerate(z_grid):
            kw = dict(Omega=Omega, Tc=Tc, Pc=Pc, Mw=Mw,
                      kij12=kij, swc=swc, tol=1e-10, maxiter=1000)

            if K_from_table is not None:
                K_init = np.array([K_from_table[0], K_from_table[1]])
                table_avail[iT, iP, iz] = True
                tie = CPA.tie_line_two_comp(
                    T=T, P_bar=P, K_init=K_init, accelerated=True, **kw)
            else:
                # Fall back to accelerated Wilson K
                tie = CPA.tie_line_two_comp(
                    T=T, P_bar=P, accelerated=True, **kw)

            flash_conv_tbl[iT, iP, iz] = tie["converged"]
            flash_iter_tbl[iT, iP, iz] = tie["iterations"]
            flash_resid_tbl[iT, iP, iz] = tie["residual_norm"]
            flash_m_tbl[iT, iP, iz] = tie["final_m"]

            n_done += 1

    elapsed = time.time() - t_start
    rate = n_done / elapsed if elapsed > 0 else 0
    eta = (nT * nP * nz - n_done) / rate if rate > 0 else 0
    sys.stdout.write(
        f"\r  T={T:6.0f}K  [{n_done:>6d}/{nT*nP*nz}  "
        f"{100*n_done/(nT*nP*nz):5.1f}%]  "
        f"{rate:.0f} pts/s  ETA {eta/60:.1f} min   ")
    sys.stdout.flush()

print(f"\n\nTotal time: {time.time() - t_start:.1f}s")

# ═══════════════════════════════════════════════════════════════════════════════
# Summary: compare all 5 strategies on two-phase points
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("COMPARISON OF 5 FLASH STRATEGIES ON TWO-PHASE POINTS")
print("=" * 90)

all_names = STRAT_NAMES_OLD + ["acc_tableK"]
print(f"\n{'Strategy':>15s}  {'Conv':>6s}  {'Conv%':>6s}  "
      f"{'Mean It':>8s}  {'Med It':>7s}  {'Max It':>7s}")

for s in range(5):
    if s < 4:
        c = flash_conv_old[..., s][mask_2ph]
        it = flash_iter_old[..., s][mask_2ph]
    else:
        c = flash_conv_tbl[mask_2ph]
        it = flash_iter_tbl[mask_2ph]

    conv_mask = c.astype(bool)
    nc = int(conv_mask.sum())
    it_c = it[conv_mask]
    print(f"{all_names[s]:>15s}  {nc:>6d}  {100*nc/n_2ph:>5.1f}%  "
          f"{it_c.mean():>8.1f}  {np.median(it_c):>7.0f}  {it_c.max():>7d}")

# Table coverage stats
n_2ph_with_table = int(table_avail[mask_2ph].sum())
n_2ph_no_table = n_2ph - n_2ph_with_table
print(f"\nTable coverage: {n_2ph_with_table}/{n_2ph} two-phase points "
      f"({100*n_2ph_with_table/n_2ph:.1f}%) have table K-values")
print(f"  Points outside table T-range (T<283K or T>533K): {n_2ph_no_table}")

# Breakdown: table-covered vs not
if n_2ph_with_table > 0:
    mask_tbl_2ph = mask_2ph & table_avail
    c_tbl = flash_conv_tbl[mask_tbl_2ph]
    it_tbl = flash_iter_tbl[mask_tbl_2ph]
    nc_tbl = int(c_tbl.sum())
    it_tbl_c = it_tbl[c_tbl.astype(bool)]
    print(f"\n  Table-covered subset ({n_2ph_with_table} pts):")
    print(f"    Conv: {nc_tbl}/{n_2ph_with_table} ({100*nc_tbl/n_2ph_with_table:.1f}%)  "
          f"Mean it: {it_tbl_c.mean():.1f}  Median: {np.median(it_tbl_c):.0f}")

    # Compare to acc_wilson on same subset
    c_aw = flash_conv_old[..., 1][mask_tbl_2ph]
    it_aw = flash_iter_old[..., 1][mask_tbl_2ph]
    nc_aw = int(c_aw.sum())
    it_aw_c = it_aw[c_aw.astype(bool)]
    print(f"    acc_wilson on same subset: Conv: {nc_aw} "
          f"Mean it: {it_aw_c.mean():.1f}  Median: {np.median(it_aw_c):.0f}")

    # Compare to acc_stabK on same subset
    c_sk = flash_conv_old[..., 2][mask_tbl_2ph]
    it_sk = flash_iter_old[..., 2][mask_tbl_2ph]
    nc_sk = int(c_sk.sum())
    it_sk_c = it_sk[c_sk.astype(bool)]
    print(f"    acc_stabK on same subset:  Conv: {nc_sk} "
          f"Mean it: {it_sk_c.mean():.1f}  Median: {np.median(it_sk_c):.0f}")

# ── Iteration comparison: table vs acc_stabK where both converge ──────────
print("\n" + "-" * 90)
print("ITERATION COMPARISON: table warm-start vs acc_stabK (where both converge)")
print("-" * 90)

both_conv = mask_2ph & flash_conv_tbl.astype(bool) & flash_conv_old[..., 2].astype(bool)
if both_conv.sum() > 0:
    it_t = flash_iter_tbl[both_conv]
    it_s = flash_iter_old[..., 2][both_conv]
    n_bc = int(both_conv.sum())

    faster = int((it_t < it_s).sum())
    same = int((it_t == it_s).sum())
    slower = int((it_t > it_s).sum())

    print(f"  N both converge: {n_bc}")
    print(f"  Table faster:  {faster} ({100*faster/n_bc:.1f}%)")
    print(f"  Same:          {same} ({100*same/n_bc:.1f}%)")
    print(f"  Table slower:  {slower} ({100*slower/n_bc:.1f}%)")
    print(f"  Mean iter: table={it_t.mean():.1f}  stabK={it_s.mean():.1f}  "
          f"ratio={it_t.mean()/it_s.mean():.2f}")

# ── Convergence regressions ──────────────────────────────────────────────
print("\n" + "-" * 90)
print("CONVERGENCE REGRESSIONS (table fails where other methods succeed)")
print("-" * 90)

for s, name in enumerate(STRAT_NAMES_OLD):
    other_conv = flash_conv_old[..., s].astype(bool)
    tbl_fail = mask_2ph & other_conv & ~flash_conv_tbl.astype(bool)
    n_reg = int(tbl_fail.sum())
    print(f"  {name} converges but table fails: {n_reg}")

# Table converges but others fail
for s, name in enumerate(STRAT_NAMES_OLD):
    other_fail = ~flash_conv_old[..., s].astype(bool)
    tbl_win = mask_2ph & other_fail & flash_conv_tbl.astype(bool)
    n_win = int(tbl_win.sum())
    if n_win > 0:
        print(f"  Table converges but {name} fails: {n_win}")

print("\nDone.")
