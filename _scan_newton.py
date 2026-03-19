"""
_scan_newton.py  —  SSI+Newton scan on the full 9825-point parameter-space grid.

Steps
-----
1. Extend CPA_ELV_all.parquet to cover T = 273–282 K and T = 538–700 K
   (the 35 scan-grid temperatures currently missing from the table).
   Columns computed by CPA2.py: Z_W, xw_W, Z_C, xw_C, chiw_W, chiw_C.
   eps_r and isochoric-derivative columns (Ndchi1w_dNw, Ndchi1w_dNc,
   Vdchi1w_dV) are NOT computed here; they require the original ELV solver
   and are set to NaN.  The warm-start K-value lookup only needs xw_W, xw_C.

2. Run SSI-only (Jex-accelerated) and SSI+Newton on all 9825 two-phase
   points from scan_results_extended.npz with table K warm-start.
   Both approaches use the extended table.

3. Save scan_newton_results.npz.

4. Produce SI figures:
   (a) Three-panel heatmap: SSI iters, Newton iters, wall-time ratio.
   (b) Bar chart: convergence %, mean iterations, mean wall-time for all
       strategies (std-SSI → acc-SSI → acc-StabK → robust → table+SSI →
       table+SSI+Newton).
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import CPA2

# ─────────────────────────────────────────────────────────────────────────────
# 0. Paths and configuration
# ─────────────────────────────────────────────────────────────────────────────
ELV_SRC   = "CPA_ELV_all.parquet"
ELV_EXT   = "CPA_ELV_all_extended.parquet"   # extended table written here
SCAN_FILE = "scan_results_extended.npz"
OUT_FILE  = "scan_newton_results.npz"
FIGDIR    = "figures/scan"
os.makedirs(FIGDIR, exist_ok=True)

# P grid used to generate new ELV rows: 0.1-bar steps from 1–10 bar,
# 1-bar steps from 10–1500 bar (≈1490 P values per T).
_P_fine   = np.arange(1.0,  10.0, 0.1)
_P_coarse = np.arange(10.0, 1501, 1.0)
P_ELV = np.unique(np.concatenate([_P_fine, _P_coarse]))

# Composition used when generating ELV rows (binary → tie-line is
# composition-independent; any two-phase z works).
Z_ELV = np.array([0.4, 0.6])   # z_CO2 = 0.4

comps = CPA2.make_components_co2_h2o()
Omega, Tc, Pc, Mw = comps["Omega"], comps["Tc"], comps["Pc"], comps["Mw"]

# ─────────────────────────────────────────────────────────────────────────────
# 1. Identify missing scan-grid temperatures
# ─────────────────────────────────────────────────────────────────────────────
print("Step 1 — Extending ELV table for missing T values")

d = np.load(SCAN_FILE, allow_pickle=True)
T_grid = d["T_grid"]   # 86 values: 273–698 K at 5 K steps

existing_df = pd.read_parquet(ELV_SRC)
table_T_min = existing_df["T_K"].min()   # 283
table_T_max = existing_df["T_K"].max()   # 533

missing_T = sorted([T for T in T_grid
                    if T < table_T_min or T > table_T_max])
print(f"  Existing table: T={table_T_min:.0f}–{table_T_max:.0f} K")
print(f"  Missing scan T values ({len(missing_T)}): "
      f"{missing_T[0]:.0f}–{missing_T[-1]:.0f} K")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Generate ELV rows for missing T values
# ─────────────────────────────────────────────────────────────────────────────
new_rows = []
kw_elv = dict(Omega=Omega, Tc=Tc, Pc=Pc, Mw=Mw, tol=1e-10, maxiter=1000,
              use_newton=True)

t0_ext = time.time()
for iT_miss, T in enumerate(missing_T):
    kij = CPA2.kij_ecpa(T)
    swc = CPA2.s14_ecpa(T)
    n_conv = 0
    for P in P_ELV:
        tie = CPA2.tie_line_two_comp(T=T, P_bar=P, kij12=kij, swc=swc, **kw_elv)
        if not tie["converged"]:
            continue
        x, y = tie["x"], tie["y"]
        # Skip trivial / near-critical solutions
        if abs(x[0] - y[0]) < 1e-4:
            continue
        # Must have valid Z and chi
        if "Z" not in tie or "chi" not in tie:
            continue
        Z_W_val = float(tie["Z"][0])
        Z_C_val = float(tie["Z"][1])
        Chi_W, Chi1_W = tie["chi"]["liq"]   # (Chix, Chi1x) aqueous
        Chi_V, Chi1_V = tie["chi"]["vap"]   # (Chiy, Chi1y) CO2-rich
        row = {
            "T_K":          float(T),
            "P_bar":        float(P),
            "Z_W":          Z_W_val,
            "xw_W":         float(x[1]),   # x_H2O in aqueous phase
            "eps_r":        np.nan,         # not computable without ELV solver
            "Z_C":          Z_C_val,
            "xw_C":         float(y[1]),   # x_H2O in CO2-rich phase
            "chiw_W":       float(Chi_W),
            "chiw_C":       float(Chi_V),
            "Ndchi1w_dNw":  np.nan,         # requires isochoric derivative
            "Ndchi1w_dNc":  np.nan,
            "Vdchi1w_dV":   np.nan,
        }
        new_rows.append(row)
        n_conv += 1
    sys.stdout.write(
        f"\r  T={T:5.0f}K  ({iT_miss+1}/{len(missing_T)})  "
        f"two-phase rows: {n_conv}   ")
    sys.stdout.flush()

print(f"\n  Generated {len(new_rows)} new ELV rows in {time.time()-t0_ext:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Merge with existing table and save
# ─────────────────────────────────────────────────────────────────────────────
if new_rows:
    new_df = pd.DataFrame(new_rows)
    extended_df = pd.concat([existing_df, new_df], ignore_index=True)
else:
    extended_df = existing_df.copy()

extended_df.to_parquet(ELV_EXT, index=False)
print(f"  Extended table saved to {ELV_EXT}  "
      f"({len(existing_df)} + {len(new_rows)} = {len(extended_df)} rows, "
      f"T={extended_df['T_K'].min():.0f}–{extended_df['T_K'].max():.0f} K)")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Build K-value lookup from extended table
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 2 — Building K-value lookup from extended table")

table_temps = np.sort(extended_df["T_K"].unique())
table_by_T = {}
for T_val, grp in extended_df.groupby("T_K"):
    grp_s = grp.sort_values("P_bar")
    table_by_T[float(T_val)] = (
        grp_s["P_bar"].values,
        ((1.0 - grp_s["xw_C"].values) /
         np.maximum(1.0 - grp_s["xw_W"].values, 1e-12)),   # K_CO2
        (grp_s["xw_C"].values /
         np.maximum(grp_s["xw_W"].values, 1e-12)),          # K_H2O
        grp_s["Z_W"].values,                                # Z aqueous
        grp_s["chiw_W"].values,                             # χ_H2O aqueous
        grp_s["Z_C"].values,                                # Z CO2-rich
        grp_s["chiw_C"].values,                             # χ_H2O CO2-rich
    )

def lookup_K(T, P):
    """
    Bilinear interpolation from extended ELV table.

    Returns dict with keys:
      'K'            : (K_CO2, K_H2O)
      'ZChi_aq_init' : (Z_W, chiw_W)   — for ZChi warm start, aqueous phase
      'ZChi_vap_init': (Z_C, chiw_C)   — for ZChi warm start, CO2-rich phase
    Returns None if T/P outside table range.
    """
    if T < table_temps[0] or T > table_temps[-1]:
        return None
    idx = np.searchsorted(table_temps, T)
    if idx == 0:
        T_lo = T_hi = table_temps[0]
    elif idx >= len(table_temps):
        T_lo = T_hi = table_temps[-1]
    else:
        T_lo = table_temps[idx - 1]
        T_hi = table_temps[idx]

    def _interp(T_key, P_target):
        if T_key not in table_by_T:
            return None
        P_arr, K1, K2, ZW, chiW, ZC, chiC = table_by_T[T_key]
        if P_target < P_arr[0] or P_target > P_arr[-1]:
            return None
        return (np.interp(P_target, P_arr, K1),
                np.interp(P_target, P_arr, K2),
                np.interp(P_target, P_arr, ZW),
                np.interp(P_target, P_arr, chiW),
                np.interp(P_target, P_arr, ZC),
                np.interp(P_target, P_arr, chiC))

    if T_lo == T_hi:
        r = _interp(T_lo, P)
    else:
        r_lo = _interp(T_lo, P)
        r_hi = _interp(T_hi, P)
        if r_lo is None and r_hi is None:
            return None
        if r_lo is None:
            r = r_hi
        elif r_hi is None:
            r = r_lo
        else:
            w = (T - T_lo) / (T_hi - T_lo)
            r = tuple(r_lo[i]*(1-w) + r_hi[i]*w for i in range(6))

    if r is None:
        return None
    K_CO2, K_H2O, Z_W, chiw_W, Z_C, chiw_C = r
    return {
        "K":             (K_CO2, K_H2O),
        "ZChi_aq_init":  (Z_W,   chiw_W),
        "ZChi_vap_init": (Z_C,   chiw_C),
    }

print(f"  Lookup covers T={table_temps[0]:.0f}–{table_temps[-1]:.0f} K")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Load existing scan results
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 3 — Running SSI+Newton scan on 9825 two-phase points")

P_grid = d["P_grid"]
z_grid = d["z_grid"]
phase_id    = d["phase_id"]          # [nT, nP, nz]
flash_iter_old  = d["flash_iter"]    # [nT, nP, nz, 4]
flash_conv_old  = d["flash_conv"]    # [nT, nP, nz, 4]
wall_time_old   = d["wall_time"]     # [nT, nP, nz]  (total per grid point)

nT, nP, nz = len(T_grid), len(P_grid), len(z_grid)
two_ph = phase_id == 4
n_2ph  = int(two_ph.sum())
print(f"  Grid: {nT}T × {nP}P × {nz}z = {nT*nP*nz:,} points, {n_2ph} two-phase")

# Pre-allocate result arrays  (two strategies: table+SSI and table+SSI+Newton)
shape = (nT, nP, nz)
STRATS = ["tbl_ssi", "tbl_newton"]
n_strat = len(STRATS)

conv_new  = np.zeros((*shape, n_strat), dtype=bool)
iter_tot  = np.zeros((*shape, n_strat), dtype=np.int32)
iter_ssi  = np.zeros((*shape, n_strat), dtype=np.int32)
iter_nwt  = np.zeros((*shape, n_strat), dtype=np.int32)
resid_new = np.full((*shape, n_strat), np.nan)
t_ms_new  = np.full((*shape, n_strat), np.nan)
tbl_avail = np.zeros(shape, dtype=bool)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Main scan loop
# ─────────────────────────────────────────────────────────────────────────────
kw_base = dict(Omega=Omega, Tc=Tc, Pc=Pc, Mw=Mw, tol=1e-10, maxiter=1000)

t_start = time.time()
n_done  = 0

for iT, T in enumerate(T_grid):
    kij = CPA2.kij_ecpa(float(T))
    swc = CPA2.s14_ecpa(float(T))

    for iP, P in enumerate(P_grid):
        tbl = lookup_K(float(T), float(P))

        for iz, zc in enumerate(z_grid):
            if not two_ph[iT, iP, iz]:
                continue   # only process two-phase points

            kw = dict(kw_base, kij12=kij, swc=swc)
            K_init        = np.array(tbl["K"])             if tbl is not None else None
            ZChi_aq_init  = tbl["ZChi_aq_init"]           if tbl is not None else None
            ZChi_vap_init = tbl["ZChi_vap_init"]          if tbl is not None else None
            if K_init is not None:
                tbl_avail[iT, iP, iz] = True

            # Strategy 0: table warm-start + accelerated SSI only
            t0 = time.perf_counter()
            r0 = CPA2.tie_line_two_comp(
                T=T, P_bar=P, K_init=K_init, accelerated=True,
                use_newton=False,
                ZChi_aq_init=ZChi_aq_init, ZChi_vap_init=ZChi_vap_init, **kw)
            t_ms_new[iT, iP, iz, 0] = (time.perf_counter() - t0) * 1e3
            conv_new[iT, iP, iz, 0]  = r0["converged"]
            iter_tot[iT, iP, iz, 0]  = r0["iterations"]
            iter_ssi[iT, iP, iz, 0]  = r0.get("ssi_iterations", r0["iterations"])
            iter_nwt[iT, iP, iz, 0]  = r0.get("newton_iterations", 0)
            resid_new[iT, iP, iz, 0] = r0["residual_norm"]

            # Strategy 1: table warm-start + accelerated SSI + Newton polish
            t0 = time.perf_counter()
            r1 = CPA2.tie_line_two_comp(
                T=T, P_bar=P, K_init=K_init, accelerated=True,
                use_newton=True,
                ZChi_aq_init=ZChi_aq_init, ZChi_vap_init=ZChi_vap_init, **kw)
            t_ms_new[iT, iP, iz, 1] = (time.perf_counter() - t0) * 1e3
            conv_new[iT, iP, iz, 1]  = r1["converged"]
            iter_tot[iT, iP, iz, 1]  = r1["iterations"]
            iter_ssi[iT, iP, iz, 1]  = r1.get("ssi_iterations", r1["iterations"])
            iter_nwt[iT, iP, iz, 1]  = r1.get("newton_iterations", 0)
            resid_new[iT, iP, iz, 1] = r1["residual_norm"]

            n_done += 1

    elapsed = time.time() - t_start
    rate    = n_done / elapsed if elapsed > 0 else 1
    eta     = (n_2ph - n_done) / rate if rate > 0 else 0
    sys.stdout.write(
        f"\r  T={T:6.0f}K  [{n_done:>5d}/{n_2ph}]  "
        f"{rate:.0f} pts/s  ETA {eta/60:.1f} min   ")
    sys.stdout.flush()

elapsed_total = time.time() - t_start
print(f"\n  Done in {elapsed_total:.1f}s ({n_2ph/elapsed_total:.0f} pts/s)")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Save results
# ─────────────────────────────────────────────────────────────────────────────
np.savez_compressed(
    OUT_FILE,
    T_grid=T_grid, P_grid=P_grid, z_grid=z_grid,
    phase_id=phase_id,
    strat_names=np.array(STRATS),
    conv=conv_new, iter_tot=iter_tot, iter_ssi=iter_ssi, iter_nwt=iter_nwt,
    resid=resid_new, t_ms=t_ms_new, tbl_avail=tbl_avail,
)
print(f"  Results saved to {OUT_FILE}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Summary statistics
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SUMMARY: Table warm-start + SSI  vs  Table warm-start + SSI + Newton")
print("=" * 80)

for s, name in enumerate(STRATS):
    c  = conv_new[..., s][two_ph]
    it = iter_tot[..., s][two_ph]
    ts = iter_ssi[..., s][two_ph]
    tn = iter_nwt[..., s][two_ph]
    tm = t_ms_new[..., s][two_ph]
    nc = c.sum()
    it_c = it[c]; ts_c = ts[c]; tn_c = tn[c]; tm_c = tm[c]
    print(f"\n  {name}:")
    print(f"    Converged:      {nc}/{n_2ph} ({100*nc/n_2ph:.2f}%)")
    print(f"    Mean iter:      {it_c.mean():.1f}  (SSI={ts_c.mean():.1f}, Newton={tn_c.mean():.1f})")
    print(f"    Median iter:    {np.median(it_c):.0f}  "
          f"(SSI={np.median(ts_c):.0f}, Newton={np.median(tn_c):.0f})")
    print(f"    Mean wall time: {tm_c.mean():.3f} ms/call")

n_nwt_used = (iter_nwt[..., 1][two_ph] > 0).sum()
print(f"\n  Newton polish triggered at {n_nwt_used}/{n_2ph} "
      f"({100*n_nwt_used/n_2ph:.1f}%) two-phase points")

n_tbl = tbl_avail[two_ph].sum()
print(f"  Table K available at {n_tbl}/{n_2ph} "
      f"({100*n_tbl/n_2ph:.1f}%) two-phase points")

# Speedup in wall time
both = conv_new[..., 0][two_ph] & conv_new[..., 1][two_ph]
t0_ = t_ms_new[..., 0][two_ph][both]
t1_ = t_ms_new[..., 1][two_ph][both]
print(f"\n  Wall time (where both converge, n={both.sum()}):")
print(f"    tbl+SSI:        {t0_.mean():.3f} ms  (median {np.median(t0_):.3f})")
print(f"    tbl+SSI+Newton: {t1_.mean():.3f} ms  (median {np.median(t1_):.3f})")
print(f"    Speedup:        {t0_.mean()/t1_.mean():.2f}x  "
      f"(negative = Newton is slower)")

# ─────────────────────────────────────────────────────────────────────────────
# 9. Plots
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating figures...")

def tp_avg(arr3d, mask3d):
    """Average arr3d[iT,iP,iz] over iz, only where mask3d is True."""
    out = np.full((nT, nP), np.nan)
    for iT_ in range(nT):
        for iP_ in range(nP):
            m = mask3d[iT_, iP_, :]
            if m.any():
                v = arr3d[iT_, iP_, :][m].astype(float)
                v = v[np.isfinite(v)]
                if len(v):
                    out[iT_, iP_] = v.mean()
    return out


# ── Figure A: Three-panel heatmap (SSI iters | Newton iters | wall-time ratio)
# for the SSI+Newton strategy (strategy index 1)
conv_mask = conv_new[..., 1] & two_ph

ssi_map  = iter_ssi[..., 1].astype(float); ssi_map[~conv_mask] = np.nan
nwt_map  = iter_nwt[..., 1].astype(float); nwt_map[~conv_mask] = np.nan
tot_map  = iter_tot[..., 1].astype(float); tot_map[~conv_mask] = np.nan

ssi_avg = tp_avg(ssi_map, two_ph)
nwt_avg = tp_avg(nwt_map, two_ph)
tot_avg = tp_avg(tot_map, two_ph)

# Wall-time ratio: (tbl+SSI) / (tbl+SSI+Newton)  — shows where Newton helps
t_ratio = np.full(shape, np.nan)
both3d = conv_new[..., 0] & conv_new[..., 1]
t_ratio[both3d] = (t_ms_new[..., 0][both3d] /
                   np.maximum(t_ms_new[..., 1][both3d], 1e-9))
ratio_avg = tp_avg(t_ratio, two_ph)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True,
                          gridspec_kw={"right": 0.88, "wspace": 0.08})

panels = [
    (ssi_avg, "SSI iterations\n(table + SSI + Newton)", "viridis", 0, 20, "Mean SSI iterations"),
    (nwt_avg, "Newton iterations\n(table + SSI + Newton)", "plasma", 0,  5, "Mean Newton iterations"),
    (ratio_avg, "Wall-time ratio\n(SSI-only / SSI+Newton)", "RdYlGn", 0.5, 2.0, "Speed-up ratio"),
]

for ax, (data, title, cmap, vmin, vmax, cblabel) in zip(axes, panels):
    im = ax.pcolormesh(T_grid, P_grid, data.T, cmap=cmap,
                       vmin=vmin, vmax=vmax, shading="nearest")
    ax.set_yscale("log")
    ax.set_xlabel("Temperature (K)", fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.set_ylim(P_grid[0], P_grid[-1])
    cax = fig.add_axes([ax.get_position().x1 + 0.003,
                        ax.get_position().y0,
                        0.008,
                        ax.get_position().height])
    fig.colorbar(im, cax=cax, label=cblabel)

axes[0].set_ylabel("Pressure (bar)", fontsize=9)
fig.suptitle("SSI + Newton polish — iteration breakdown and speed-up "
             "(averaged over $z_{\\rm CO_2}$, two-phase points)",
             fontsize=10, y=1.01)
fig.savefig(f"{FIGDIR}/newton_heatmap.pdf", bbox_inches="tight", dpi=150)
fig.savefig(f"{FIGDIR}/newton_heatmap.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"  Saved {FIGDIR}/newton_heatmap.pdf")


# ── Figure B: Bar chart — all 6 strategies (+ 2 new)
# Collect stats for all strategies
#   Old strategies 0–3 from scan_results_extended.npz
#   New strategies: tbl+SSI (s=0), tbl+SSI+Newton (s=1) from this scan

old_names   = list(d["flash_strategy_names"])
all_names   = old_names + ["tbl+SSI", "tbl+SSI+Newton"]
n_all_strat = len(all_names)

conv_rates_all = []
mean_iters_all = []
mean_ms_all    = []

for s in range(4):
    c  = flash_conv_old[..., s][two_ph]
    it = flash_iter_old[..., s][two_ph]
    nc = c.sum()
    conv_rates_all.append(100 * nc / n_2ph)
    mean_iters_all.append(it[c].mean() if nc else np.nan)
    # old scan has per-cell wall_time (whole cell = stab+4 flashes), not per-strategy
    mean_ms_all.append(np.nan)

for s in range(2):
    c  = conv_new[..., s][two_ph]
    it = iter_tot[..., s][two_ph]
    tm = t_ms_new[..., s][two_ph]
    nc = c.sum()
    conv_rates_all.append(100 * nc / n_2ph)
    mean_iters_all.append(it[c].mean() if nc else np.nan)
    mean_ms_all.append(tm[c].mean() if nc else np.nan)

labels_short = [
    "Std SSI\n+Wilson K",
    "Acc SSI\n+Wilson K",
    "Acc SSI\n+Stab K",
    "Robust\n(best)",
    "Tbl+Acc SSI\n(no Newton)",
    "Tbl+Acc SSI\n+Newton",
]
colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c"]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
x = np.arange(n_all_strat)

# Panel (a): convergence rate
ax = axes[0]
bars = ax.bar(x, conv_rates_all, color=colors, edgecolor="white", linewidth=0.5)
for bar, cr in zip(bars, conv_rates_all):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.03,
            f"{cr:.2f}%", ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(labels_short, fontsize=8)
ax.set_ylabel("Convergence rate (%)")
ax.set_title("(a) Convergence rate (9825 two-phase points)", fontsize=9)
ax.set_ylim(min(conv_rates_all) - 1, 101)

# Panel (b): mean iterations
ax = axes[1]
bars = ax.bar(x, mean_iters_all, color=colors, edgecolor="white", linewidth=0.5)
for bar, mi in zip(bars, mean_iters_all):
    if np.isfinite(mi):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.2,
                f"{mi:.1f}", ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(labels_short, fontsize=8)
ax.set_ylabel("Mean iterations (converged points)")
ax.set_title("(b) Mean total iterations", fontsize=9)

fig.suptitle(f"Flash strategy comparison — {n_2ph} two-phase points "
             f"($T=273$–698 K, $P=1$–1500 bar, $z_{{\\rm CO_2}}=0.001$–0.999)",
             fontsize=9, y=1.02)
fig.tight_layout()
fig.savefig(f"{FIGDIR}/strategy_comparison_bar.pdf", bbox_inches="tight", dpi=150)
fig.savefig(f"{FIGDIR}/strategy_comparison_bar.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"  Saved {FIGDIR}/strategy_comparison_bar.pdf")


# ── Figure C: Wall-time heatmap for SSI+Newton (absolute, ms per call)
wt_avg = tp_avg(t_ms_new[..., 1], two_ph)

fig, ax = plt.subplots(figsize=(7, 4.5))
im = ax.pcolormesh(T_grid, P_grid, wt_avg.T, cmap="inferno",
                   vmin=0, vmax=5, shading="nearest")
ax.set_yscale("log")
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Pressure (bar)")
ax.set_title("Wall time per flash call — table + SSI + Newton (ms)")
ax.set_ylim(P_grid[0], P_grid[-1])
fig.colorbar(im, ax=ax, label="Wall time (ms)")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/newton_walltime_heatmap.pdf", bbox_inches="tight", dpi=150)
fig.savefig(f"{FIGDIR}/newton_walltime_heatmap.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"  Saved {FIGDIR}/newton_walltime_heatmap.pdf")

print("\nDone.")
