"""
Generate publication-quality figures from the eCPA v3 solution table (scan_v3_table.npz).

Produces (all saved to figures/scan_v3/):
  1. ecpa_phase_map.png/pdf     — T-P two-phase fraction for 6 NaCl molalities
  2. ecpa_composition_aq.png/pdf — T-P map of CO2 mole-% in aqueous phase
  3. ecpa_composition_c.png/pdf  — T-P map of H2O mole-% in CO2-rich phase
  4. ecpa_ssi_heatmap.png/pdf   — T-P median SSI iteration count (ternary extension)
  5. ecpa_newton_stats.png/pdf  — Newton iteration counts (aq and CO2-rich solvers)
  6. ecpa_phase_envelope.png/pdf — Bubble/dew point lines in T-P for selected ms

Usage:
    python _plot_scan_v3_figures.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import os

# ── output directory ──────────────────────────────────────────────────────────
OUT_DIR = "figures/scan_v3"
os.makedirs(OUT_DIR, exist_ok=True)

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading scan_v3_table.npz …")
npz = np.load("results/scan_v3_table.npz")
T_grid  = npz["T_grid"]          # (70,)
P_grid  = npz["P_grid"]          # (50,)
z_grid  = npz["z_grid"]          # (25,)
ms_grid = npz["ms_grid"]         # (14,)
is_two  = npz["is_two_phase"]    # (70, 50, 25, 14)  bool
x4w     = npz["x4w"]            # CO2 mole fraction in aqueous phase
x1c     = npz["x1c"]            # H2O mole fraction in CO2-rich phase
beta    = npz["beta"]            # vapour/CO2-rich mole fraction

print("Loading scan_v3_metrics.parquet …")
df = pd.read_parquet("results/scan_v3_metrics.parquet")
# eCPA-only rows (CPA2 rows have eos_type != 'eCPA')
df_ecpa = df[df["eos_type"] == "eCPA"].copy()
df_2ph  = df_ecpa[df_ecpa["is_two_phase"]].copy()

# map ms_grid → ims indices used throughout
ms_labels = {m: f"{m:.1f}" for m in ms_grid}
# select representative ms indices for multi-panel figures
MS_PANEL_IDX = [0, 2, 4, 6, 10, 13]   # ms ≈ 0, 0.1, 0.5, 1, 3, 6  (NPZ-based figs)
MS_PANEL_VAL = ms_grid[MS_PANEL_IDX]
# ms=0 uses CPA (not eCPA); for parquet/eCPA-only figures use 1e-5 instead
MS_PANEL_IDX_ECPA = [1, 2, 4, 6, 10, 13]   # ms ≈ 1e-5, 0.1, 0.5, 1, 3, 6
MS_PANEL_VAL_ECPA = ms_grid[MS_PANEL_IDX_ECPA]

# ── colour style ──────────────────────────────────────────────────────────────
GREY      = "#d0d0d0"
DARK_GREY = "#888888"
CMAP_FRAC  = "Blues"
CMAP_CO2   = "YlOrRd"
CMAP_H2O   = "BuPu"
CMAP_SSI   = "inferno_r"

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def _axes_style(ax, xlabel=True, ylabel=True):
    ax.set_yscale("log")
    ax.set_xlim(T_grid[0], T_grid[-1])
    ax.set_ylim(P_grid[0], P_grid[-1])
    if xlabel:
        ax.set_xlabel("Temperature  (K)")
    if ylabel:
        ax.set_ylabel("Pressure  (bar)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatter(labelOnlyBase=False))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def _add_cbar(fig, im, ax, label, orientation="vertical"):
    """Attach a colourbar next to ax without disturbing layout."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    if orientation == "vertical":
        cax = divider.append_axes("right", size="5%", pad=0.05)
    else:
        cax = divider.append_axes("bottom", size="8%", pad=0.35)
    cb = fig.colorbar(im, cax=cax, orientation=orientation)
    cb.set_label(label, fontsize=8)
    return cb


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Phase regime maps (two-phase fraction across z, per T-P-ms)
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 1: phase regime maps …")
fig, axes = plt.subplots(2, 3, figsize=(11, 6.5),
                          constrained_layout=True)

# For each (iT, iP, ims): fraction of z values that are two-phase
frac_2ph = is_two.mean(axis=2)   # (70, 50, 14) — mean over z

for k, (ims, ms_val) in enumerate(zip(MS_PANEL_IDX, MS_PANEL_VAL)):
    ax = axes.flat[k]
    data = frac_2ph[:, :, ims].T   # (50, 70): P rows, T cols — imshow expects [row, col]

    # background (single-phase) grey
    ax.set_facecolor(GREY)

    im = ax.imshow(
        data,
        origin="lower",
        extent=[T_grid[0], T_grid[-1], 0, len(P_grid) - 1],
        aspect="auto",
        cmap=CMAP_FRAC,
        vmin=0, vmax=1,
    )
    # imshow with log-y is tricky — use pcolormesh instead
    ax.cla()
    ax.set_facecolor(GREY)

    # build masked arrays
    Tm, Pm = np.meshgrid(T_grid, P_grid)
    frac = frac_2ph[:, :, ims].T   # (nP, nT)

    pcm = ax.pcolormesh(Tm, Pm, frac, cmap=CMAP_FRAC, vmin=0, vmax=1,
                         shading="nearest")
    ax.set_yscale("log")
    ax.set_xlim(T_grid[0], T_grid[-1])
    ax.set_ylim(P_grid[0], P_grid[-1])

    ms_str = f"0" if ms_val < 1e-4 else f"{ms_val:.1f}".rstrip("0").rstrip(".")
    ax.set_title(fr"$m_s = {ms_str}\ \mathrm{{mol\,kg^{{-1}}}}$")

    if k >= 3:
        ax.set_xlabel("Temperature  (K)")
    else:
        ax.set_xticklabels([])
    if k % 3 == 0:
        ax.set_ylabel("Pressure  (bar)")

    _add_cbar(fig, pcm, ax, "Two-phase fraction")

fig.suptitle(
    r"Fraction of CO$_2$ feed fractions ($z = 0.05$–$0.90$) in two-phase region",
    fontsize=10, y=1.01,
)

for ext in ("png", "pdf"):
    fig.savefig(f"{OUT_DIR}/ecpa_phase_map.{ext}")
plt.close(fig)
print("  → saved ecpa_phase_map")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — CO2 content in aqueous phase  (x4w in mol-%)
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 2: aqueous CO2 composition maps …")

# Choose representative (iz, ims) panels — 2 rows × 4 cols
# Row 1: z = 0.25 (lean), z = 0.50 (intermediate)
# Row 2: z = 0.75 (rich),  z = 0.90 (very rich)
# Columns: ms = 0, 1, 3, 6
IZ_PANELS  = [5, 12, 18, 23]   # z ≈ 0.20, 0.40, 0.65, 0.87
IMS_PANELS = [0,  6, 10, 13]   # ms = 0, 1, 3, 6

fig, axes = plt.subplots(4, 4, figsize=(12, 10),
                          constrained_layout=True)

vmax_co2 = 0.10   # x4w rarely exceeds ~10 mol%

for row, iz in enumerate(IZ_PANELS):
    for col, ims in enumerate(IMS_PANELS):
        ax = axes[row, col]
        ax.set_facecolor(GREY)

        # x4w is NaN where single-phase
        data_co2 = x4w[:, :, iz, ims].T   # (nP, nT)
        mask_2ph  = is_two[:, :, iz, ims].T

        Tm, Pm = np.meshgrid(T_grid, P_grid)

        # plot single-phase background first
        sp_mask = ~mask_2ph
        ax.pcolormesh(Tm, Pm,
                       np.where(sp_mask, 0.0, np.nan),
                       cmap=matplotlib.colors.ListedColormap([GREY]),
                       shading="nearest")

        val = np.where(mask_2ph, data_co2 * 100, np.nan)   # in mol-%
        pcm = ax.pcolormesh(Tm, Pm, val, cmap=CMAP_CO2,
                             vmin=0, vmax=vmax_co2 * 100,
                             shading="nearest")
        ax.set_yscale("log")
        ax.set_xlim(T_grid[0], T_grid[-1])
        ax.set_ylim(P_grid[0], P_grid[-1])

        z_val  = z_grid[iz]
        ms_val = ms_grid[ims]
        ms_str = "0" if ms_val < 1e-4 else f"{ms_val:.0f}"
        ax.set_title(f"$z={z_val:.2f},\\ m_s={ms_str}$", fontsize=8)

        if row == 3:
            ax.set_xlabel("T (K)", fontsize=8)
        else:
            ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel("P (bar)", fontsize=8)

        _add_cbar(fig, pcm, ax,
                   r"$x_{\mathrm{CO_2}}^{\mathrm{aq}}$ (mol-%)")

fig.suptitle(
    r"CO$_2$ mole-% in aqueous phase — eCPA ternary scan",
    fontsize=11, y=1.01,
)

for ext in ("png", "pdf"):
    fig.savefig(f"{OUT_DIR}/ecpa_composition_aq.{ext}")
plt.close(fig)
print("  → saved ecpa_composition_aq")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — H2O content in CO2-rich phase  (x1c in mol-%)
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 3: CO2-rich H2O composition maps …")

fig, axes = plt.subplots(4, 4, figsize=(12, 10),
                          constrained_layout=True)

vmax_h2o = 15.0   # x1c up to ~15 mol% at high T

for row, iz in enumerate(IZ_PANELS):
    for col, ims in enumerate(IMS_PANELS):
        ax = axes[row, col]
        ax.set_facecolor(GREY)

        data_h2o = x1c[:, :, iz, ims].T   # (nP, nT) H2O in CO2-rich
        mask_2ph  = is_two[:, :, iz, ims].T

        Tm, Pm = np.meshgrid(T_grid, P_grid)
        val = np.where(mask_2ph, data_h2o * 100, np.nan)
        pcm = ax.pcolormesh(Tm, Pm, val, cmap=CMAP_H2O,
                             vmin=0, vmax=vmax_h2o,
                             shading="nearest")
        ax.set_yscale("log")
        ax.set_xlim(T_grid[0], T_grid[-1])
        ax.set_ylim(P_grid[0], P_grid[-1])

        z_val  = z_grid[iz]
        ms_val = ms_grid[ims]
        ms_str = "0" if ms_val < 1e-4 else f"{ms_val:.0f}"
        ax.set_title(f"$z={z_val:.2f},\\ m_s={ms_str}$", fontsize=8)

        if row == 3:
            ax.set_xlabel("T (K)", fontsize=8)
        else:
            ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel("P (bar)", fontsize=8)

        _add_cbar(fig, pcm, ax,
                   r"$x_{\mathrm{H_2O}}^{\mathrm{CO_2}}$ (mol-%)")

fig.suptitle(
    r"H$_2$O mole-% in CO$_2$-rich phase — eCPA ternary scan",
    fontsize=11, y=1.01,
)

for ext in ("png", "pdf"):
    fig.savefig(f"{OUT_DIR}/ecpa_composition_c.{ext}")
plt.close(fig)
print("  → saved ecpa_composition_c")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — SSI iteration heatmap (eCPA ternary extension of fig:ssi_heatmap)
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 4: SSI iteration heatmap …")

# Build 2D median-iteration map per (T, P, ms) from parquet
# Only two-phase eCPA rows
pivot_cache = {}
for ims_p, ms_val in zip(MS_PANEL_IDX_ECPA, MS_PANEL_VAL_ECPA):
    sub = df_2ph[np.abs(df_2ph["ms_feed"] - ms_val) < 1e-6]
    # pivot: T on x, P on y
    piv = sub.groupby(["T", "P"])["n_ssi_iters"].median().unstack("T")
    pivot_cache[ims_p] = piv

fig, axes = plt.subplots(2, 3, figsize=(11, 6.5),
                          constrained_layout=True)

vmin_ssi, vmax_ssi = 1, 20

for k, (ims_p, ms_val) in enumerate(zip(MS_PANEL_IDX_ECPA, MS_PANEL_VAL_ECPA)):
    ax = axes.flat[k]
    ax.set_facecolor(GREY)

    # two-phase fraction mask (any z)
    frac = frac_2ph[:, :, ims_p].T   # (nP, nT)
    Tm, Pm = np.meshgrid(T_grid, P_grid)

    piv = pivot_cache[ims_p]
    # align to full T,P grid
    piv_full = piv.reindex(columns=T_grid, index=P_grid)
    data = piv_full.values   # (nP, nT), NaN where single-phase

    # grey background for single-phase area
    ax.pcolormesh(Tm, Pm, np.where(frac > 0, np.nan, 1.0),
                   cmap=matplotlib.colors.ListedColormap([GREY]),
                   shading="nearest")

    pcm = ax.pcolormesh(Tm, Pm, data, cmap=CMAP_SSI,
                         vmin=vmin_ssi, vmax=vmax_ssi,
                         shading="nearest")
    ax.set_yscale("log")
    ax.set_xlim(T_grid[0], T_grid[-1])
    ax.set_ylim(P_grid[0], P_grid[-1])

    ms_str = "≈0" if ms_val < 1e-4 else f"{ms_val:.1f}".rstrip("0").rstrip(".")
    ax.set_title(fr"$m_s = {ms_str}\ \mathrm{{mol\,kg^{{-1}}}}$")

    if k >= 3:
        ax.set_xlabel("Temperature  (K)")
    else:
        ax.set_xticklabels([])
    if k % 3 == 0:
        ax.set_ylabel("Pressure  (bar)")

    cb = _add_cbar(fig, pcm, ax, "Median SSI iterations")
    cb.set_ticks([1, 5, 10, 15, 20])

    # add legend for single-phase region (first panel only)
    if k == 0:
        from matplotlib.patches import Rectangle
        handles = [Rectangle((0, 0), 1, 1, facecolor=GREY, edgecolor=DARK_GREY,
                               label="Single-phase")]
        ax.legend(handles=handles, loc="upper right", fontsize=7)

fig.suptitle(
    r"Median SSI iterations — eCPA ternary flash (CO$_2$+H$_2$O+NaCl)",
    fontsize=10, y=1.01,
)

for ext in ("png", "pdf"):
    fig.savefig(f"{OUT_DIR}/ecpa_ssi_heatmap.{ext}")
plt.close(fig)
print("  → saved ecpa_ssi_heatmap")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Newton iteration statistics (aqueous + CO2-rich solvers)
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 5: Newton iteration statistics …")

# Panel A: histogram of n_newton_aq_iters / call (warm-start aq Newton)
# Panel B: histogram of n_newton_c_iters / call (warm-start CO2-rich Newton)
# Panel C: T-P heatmap of n_newton_aq_ok / n_newton_aq (success rate) for ms=1
# Panel D: T-P heatmap of n_newton_c_ok / n_newton_c  (success rate) for ms=1

ims_ref = 6   # ms = 1.0 mol/kg
ms_ref  = ms_grid[ims_ref]

sub_ref = df_2ph[np.abs(df_2ph["ms_feed"] - ms_ref) < 1e-6]

fig = plt.figure(figsize=(12, 9), constrained_layout=True)
gs  = GridSpec(2, 2, figure=fig)

# --- A: aqueous Newton iterations per call (avg iters per SSI step)
ax_a = fig.add_subplot(gs[0, 0])
aq_iters = df_2ph["n_newton_aq_iters"] / df_2ph["n_newton_aq"].clip(lower=1)
aq_iters_clean = aq_iters[df_2ph["n_newton_aq"] > 0]
ax_a.hist(aq_iters_clean, bins=30, color="#3a86ff", edgecolor="white", linewidth=0.4)
ax_a.set_xlabel("Mean Newton iterations/call (aqueous solver)")
ax_a.set_ylabel("Count (flash calls)")
ax_a.set_title("Aqueous Newton convergence")
med = aq_iters_clean.median()
ax_a.axvline(med, color="k", linestyle="--", linewidth=1)
ax_a.text(med + 0.05, ax_a.get_ylim()[1] * 0.85, f"median={med:.1f}", fontsize=8)

# --- B: CO2-rich Newton iterations per call
ax_b = fig.add_subplot(gs[0, 1])
c_iters = df_2ph["n_newton_c_iters"] / df_2ph["n_newton_c"].clip(lower=1)
c_iters_clean = c_iters[df_2ph["n_newton_c"] > 0]
ax_b.hist(c_iters_clean, bins=30, color="#ff6b6b", edgecolor="white", linewidth=0.4)
ax_b.set_xlabel(r"Mean Newton iterations/call (CO$_2$-rich solver)")
ax_b.set_ylabel("Count (flash calls)")
ax_b.set_title(r"CO$_2$-rich Newton convergence")
med_c = c_iters_clean.median()
ax_b.axvline(med_c, color="k", linestyle="--", linewidth=1)
ax_b.text(med_c + 0.05, ax_b.get_ylim()[1] * 0.85, f"median={med_c:.1f}", fontsize=8)

# --- C: aqueous Newton success rate over T-P (ms=1)
ax_c = fig.add_subplot(gs[1, 0])
ax_c.set_facecolor(GREY)
piv_aq_ok  = sub_ref.groupby(["T", "P"])["n_newton_aq_ok"].sum().unstack("T")
piv_aq_tot = sub_ref.groupby(["T", "P"])["n_newton_aq"].sum().unstack("T")
piv_aq_rate = (piv_aq_ok / piv_aq_tot.clip(lower=1)).reindex(
    columns=T_grid, index=P_grid)
Tm, Pm = np.meshgrid(T_grid, P_grid)
pcm_c = ax_c.pcolormesh(Tm, Pm, piv_aq_rate.values * 100,
                          cmap="RdYlGn", vmin=0, vmax=100, shading="nearest")
ax_c.set_yscale("log")
ax_c.set_xlim(T_grid[0], T_grid[-1])
ax_c.set_ylim(P_grid[0], P_grid[-1])
ax_c.set_xlabel("Temperature  (K)")
ax_c.set_ylabel("Pressure  (bar)")
ax_c.set_title(fr"Aqueous Newton success rate (%)  [$m_s={ms_ref:.0f}$]")
_add_cbar(fig, pcm_c, ax_c, "Success rate (%)")

# --- D: CO2-rich Newton success rate over T-P (ms=1)
ax_d = fig.add_subplot(gs[1, 1])
ax_d.set_facecolor(GREY)
piv_c_ok  = sub_ref.groupby(["T", "P"])["n_newton_c_ok"].sum().unstack("T")
piv_c_tot = sub_ref.groupby(["T", "P"])["n_newton_c"].sum().unstack("T")
piv_c_rate = (piv_c_ok / piv_c_tot.clip(lower=1)).reindex(
    columns=T_grid, index=P_grid)
pcm_d = ax_d.pcolormesh(Tm, Pm, piv_c_rate.values * 100,
                          cmap="RdYlGn", vmin=0, vmax=100, shading="nearest")
ax_d.set_yscale("log")
ax_d.set_xlim(T_grid[0], T_grid[-1])
ax_d.set_ylim(P_grid[0], P_grid[-1])
ax_d.set_xlabel("Temperature  (K)")
ax_d.set_ylabel("Pressure  (bar)")
ax_d.set_title(fr"CO$_2$-rich Newton success rate (%)  [$m_s={ms_ref:.0f}$]")
_add_cbar(fig, pcm_d, ax_d, "Success rate (%)")

fig.suptitle("Inner Newton solver performance — eCPA ternary scan", fontsize=11)

for ext in ("png", "pdf"):
    fig.savefig(f"{OUT_DIR}/ecpa_newton_stats.{ext}")
plt.close(fig)
print("  → saved ecpa_newton_stats")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6 — Phase envelope: bubble/dew lines in T-P for several ms and z
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 6: phase envelope plots …")

# Strategy: for each (z, ms), find the high-P boundary (bubble point) and
# low-P boundary (dew point) by scanning P at fixed T and locating where
# is_two_phase transitions from False to True.

def _find_envelope(is_2ph_Tp):
    """
    is_2ph_Tp: (nT, nP) bool array — True where two-phase.
    Returns bubble_P[nT], dew_P[nT] arrays (NaN where boundary not found).
    """
    nT, nP = is_2ph_Tp.shape
    bubble = np.full(nT, np.nan)   # high-P boundary (entering 2ph from high P)
    dew    = np.full(nT, np.nan)   # low-P boundary (entering 2ph from low P)
    for iT in range(nT):
        row = is_2ph_Tp[iT, :]
        if not row.any():
            continue
        # lowest P that is two-phase
        idx_lo = np.argmax(row)   # first True
        if idx_lo > 0:
            dew[iT] = P_grid[idx_lo]
        # highest P that is two-phase
        idx_hi = len(row) - 1 - np.argmax(row[::-1])
        if idx_hi < nP - 1:
            bubble[iT] = P_grid[idx_hi]
    return dew, bubble


# Two panels: left=fixed z sweep over ms, right=fixed ms sweep over z
Z_ENVELOPE_IDX  = [8, 14, 20]     # z ≈ 0.30, 0.50, 0.70
MS_ENVELOPE_IDX = [0, 4, 6, 10]   # ms = 0, 0.5, 1, 3

colours_z  = ["#2196F3", "#FF9800", "#4CAF50"]
colours_ms = ["#1a1aff", "#ff5500", "#228B22", "#9B59B6"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

# Left: fixed ms=0, sweep z
ims_env = 0  # ms = 0
for colour, iz in zip(colours_z, Z_ENVELOPE_IDX):
    is_2ph = is_two[:, :, iz, ims_env]   # (nT, nP)
    dew, bubble = _find_envelope(is_2ph)
    z_val = z_grid[iz]
    ax1.semilogy(T_grid, dew,    color=colour, lw=1.5, ls="-",
                  label=f"$z={z_val:.2f}$")
    ax1.semilogy(T_grid, bubble, color=colour, lw=1.5, ls="--")

ax1.set_xlim(T_grid[0], T_grid[-1])
ax1.set_ylim(P_grid[0], P_grid[-1])
ax1.set_xlabel("Temperature  (K)")
ax1.set_ylabel("Pressure  (bar)")
ax1.set_title(r"Dew (—) and bubble (- -) lines, $m_s = 0$")
ax1.legend(title="CO₂ feed fraction", fontsize=8)
ax1.grid(True, which="both", ls=":", alpha=0.4)

# Right: fixed z=0.50, sweep ms
iz_env = 14   # z ≈ 0.50
for colour, ims in zip(colours_ms, MS_ENVELOPE_IDX):
    is_2ph = is_two[:, :, iz_env, ims]
    dew, bubble = _find_envelope(is_2ph)
    ms_val = ms_grid[ims]
    ms_str = "0" if ms_val < 1e-4 else f"{ms_val:.1f}".rstrip("0").rstrip(".")
    ax2.semilogy(T_grid, dew,    color=colour, lw=1.5, ls="-",
                  label=fr"$m_s={ms_str}$")
    ax2.semilogy(T_grid, bubble, color=colour, lw=1.5, ls="--")

ax2.set_xlim(T_grid[0], T_grid[-1])
ax2.set_ylim(P_grid[0], P_grid[-1])
ax2.set_xlabel("Temperature  (K)")
ax2.set_ylabel("Pressure  (bar)")
ax2.set_title(r"Dew (—) and bubble (- -) lines, $z = 0.50$")
ax2.legend(title="NaCl molality", fontsize=8)
ax2.grid(True, which="both", ls=":", alpha=0.4)

fig.suptitle(
    r"Phase envelope — CO$_2$+H$_2$O+NaCl (eCPA EoS, $T=288$–$633$ K)",
    fontsize=11,
)

for ext in ("png", "pdf"):
    fig.savefig(f"{OUT_DIR}/ecpa_phase_envelope.{ext}")
plt.close(fig)
print("  → saved ecpa_phase_envelope")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 7 — Summary: Salting-out effect (CO2 solubility vs ms at fixed T,P,z)
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 7: salting-out summary …")

# x4w (CO2 aq mole fraction) as a function of ms at fixed (T, P, z) for several T
Mw = 0.018015  # kg/mol H2O
# mc [mol/kg] = x4w / (x1w * Mw)

IZ_SALT = 14    # z ≈ 0.546
IP_SALT = 30    # P ≈ 88 bar
# skip ms=0 (CPA, not eCPA — phase labels can swap near critical region)
IMS_SALT = slice(1, None)   # ms_grid[1:] = 1e-5 … 6 mol/kg

T_SALT_IDX = [8, 16, 24, 32, 44]   # 5 isotherms spread over 288–633 K
T_SALT_VAL = T_grid[T_SALT_IDX]

cmap_T = plt.cm.plasma
norm_T = mcolors.Normalize(vmin=T_SALT_VAL.min(), vmax=T_SALT_VAL.max())

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)

ms_plot = ms_grid[IMS_SALT]
x1w_arr = npz["x1w"]

for iT, T_val in zip(T_SALT_IDX, T_SALT_VAL):
    clr = cmap_T(norm_T(T_val))
    x4w_vs_ms  = x4w[iT, IP_SALT, IZ_SALT, IMS_SALT]
    x1w_vs_ms  = x1w_arr[iT, IP_SALT, IZ_SALT, IMS_SALT]
    mask       = is_two[iT, IP_SALT, IZ_SALT, IMS_SALT]

    mc   = np.where(mask, x4w_vs_ms / (x1w_vs_ms * Mw), np.nan)
    xco2 = np.where(mask, x4w_vs_ms * 100, np.nan)

    ax_l.plot(ms_plot, mc,   color=clr, lw=1.5, marker="o", ms=4,
               label=f"{T_val:.0f} K")
    ax_r.plot(ms_plot, xco2, color=clr, lw=1.5, marker="o", ms=4)

ax_l.set_xlabel(r"NaCl molality $m_s$ (mol kg$^{-1}$)")
ax_l.set_ylabel(r"CO$_2$ solubility $m_c$ (mol kg$^{-1}$)")
ax_l.set_title(f"CO$_2$ solubility vs. salinity\n"
               f"($z={z_grid[IZ_SALT]:.2f}$, $P={P_grid[IP_SALT]:.0f}$ bar)")
ax_l.legend(title="Temperature", fontsize=8)
ax_l.grid(True, ls=":", alpha=0.5)

ax_r.set_xlabel(r"NaCl molality $m_s$ (mol kg$^{-1}$)")
ax_r.set_ylabel(r"$x_{\mathrm{CO_2}}^{\mathrm{aq}}$ (mol-%)")
ax_r.set_title(f"CO$_2$ mole-% in aqueous phase\n"
               f"($z={z_grid[IZ_SALT]:.2f}$, $P={P_grid[IP_SALT]:.0f}$ bar)")
ax_r.grid(True, ls=":", alpha=0.5)

sm = plt.cm.ScalarMappable(cmap=cmap_T, norm=norm_T)
sm.set_array([])
fig.colorbar(sm, ax=[ax_l, ax_r], label="Temperature  (K)", shrink=0.8)

fig.suptitle(r"Salting-out effect — CO$_2$+H$_2$O+NaCl (eCPA)", fontsize=11)

for ext in ("png", "pdf"):
    fig.savefig(f"{OUT_DIR}/ecpa_salting_out.{ext}")
plt.close(fig)
print("  → saved ecpa_salting_out")


# ═══════════════════════════════════════════════════════════════════════════════
# Done
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\nAll figures saved to {OUT_DIR}/")
