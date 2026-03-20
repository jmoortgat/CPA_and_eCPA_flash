"""
Generate publication-quality figures from the eCPA v3 solution table (scan_v3_table.npz).

Style: scienceplots 'science' + cmcrameri colormaps + LaTeX rendering.

Produces (all saved to figures/scan_v3/):
  1. ecpa_phase_map.png/pdf      — T-P two-phase fraction (6 NaCl molalities)
  2. ecpa_composition_aq.png/pdf — T-P aqueous CO2 mol-% (2 rows × 3 ms panels)
  3. ecpa_composition_c.png/pdf  — T-P CO2-rich H2O mol-% (2 rows × 3 ms panels)
  4. ecpa_ssi_heatmap.png/pdf    — T-P median SSI iterations (ternary extension)
  5. ecpa_newton_stats.png/pdf   — Newton convergence histograms + success maps
  6. ecpa_phase_envelope.png/pdf — Bubble/dew lines in T-P (z and ms sweeps)
  7. ecpa_salting_out.png/pdf    — CO2 solubility vs NaCl molality (5 isotherms)

Usage:
    python _plot_scan_v3_figures.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import cmcrameri.cm as cmc
import scienceplots  # noqa: F401 — registers styles
import os

# ── output directory ──────────────────────────────────────────────────────────
OUT_DIR = "figures/scan_v3"
os.makedirs(OUT_DIR, exist_ok=True)

# ── global style ──────────────────────────────────────────────────────────────
plt.style.use(["science"])   # LaTeX, clean ticks, minimal frame
# Override a handful of defaults for larger readability
plt.rcParams.update({
    "figure.dpi":      150,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
    "axes.linewidth":  0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
})

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading scan_v3_table.npz ...")
npz    = np.load("results/scan_v3_table.npz")
T_grid = npz["T_grid"]          # (70,)
P_grid = npz["P_grid"]          # (50,)
z_grid = npz["z_grid"]          # (25,)
ms_grid= npz["ms_grid"]         # (14,)
is_two = npz["is_two_phase"]    # (70, 50, 25, 14)  bool
x4w    = npz["x4w"]             # CO2 mole fraction in aqueous phase  (NaN if single-ph)
x1c    = npz["x1c"]             # H2O mole fraction in CO2-rich phase
x1w    = npz["x1w"]             # H2O mole fraction in aqueous phase
Mw     = 0.018015               # kg/mol H2O

print("Loading scan_v3_metrics.parquet ...")
df      = pd.read_parquet("results/scan_v3_metrics.parquet")
df_ecpa = df[df["eos_type"] == "eCPA"].copy()
df_2ph  = df_ecpa[df_ecpa["is_two_phase"]].copy()

# ── panel indices ─────────────────────────────────────────────────────────────
# For NPZ-based panels (includes ms=0 CPA rows in phase/composition maps)
MS_NPZ_IDX = [0, 2, 4, 6, 10, 13]        # ms = 0, 0.1, 0.5, 1, 3, 6
MS_NPZ_VAL = ms_grid[MS_NPZ_IDX]

# For parquet/eCPA-only panels (skip ms=0 which uses CPA)
MS_ECPA_IDX = [1, 2, 4, 6, 10, 13]       # ms ≈ 1e-5, 0.1, 0.5, 1, 3, 6
MS_ECPA_VAL = ms_grid[MS_ECPA_IDX]

# Fixed z ≈ 0.55 for composition panels
IZ_MID = 14   # z_grid[14] = 0.546

# ── colormaps ─────────────────────────────────────────────────────────────────
CMAP_FRAC  = cmc.batlow      # two-phase fraction (0–1)
CMAP_AQ    = cmc.lapaz       # CO2 in aq: deep blue → light → cream
CMAP_CO2   = cmc.lajolla     # H2O in CO2-rich: cream → deep red
CMAP_SSI   = cmc.lajolla     # iterations: few=light, many=dark red
CMAP_SUCC  = cmc.cork        # success rate: diverging green/pink
CMAP_TEMP  = cmc.lipari      # temperature colour scale for line plots

GREY       = "#c8c8c8"       # single-phase background
DARK_GREY  = "#666666"


# ── helpers ───────────────────────────────────────────────────────────────────
def _ms_label(ms_val):
    """Nice LaTeX string for a molality value."""
    if ms_val < 1e-4:
        return r"$m_s = 0$"
    elif ms_val < 0.01:
        return r"$m_s \approx 0$"
    else:
        s = f"{ms_val:g}"
        return rf"$m_s = {s}\ \mathrm{{mol\,kg^{{-1}}}}$"


def _log_yaxis(ax, show_labels=True):
    ax.set_yscale("log")
    ax.set_ylim(P_grid[0], P_grid[-1])
    if show_labels:
        ax.yaxis.set_major_formatter(
            mticker.LogFormatter(labelOnlyBase=False, minor_thresholds=(2, 0.5)))
    else:
        ax.yaxis.set_major_formatter(mticker.NullFormatter())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(axis="y", which="minor", length=2)


def _pcolor(ax, data, cmap, vmin, vmax, Tm, Pm):
    """Masked pcolormesh with grey background for NaN (single-phase)."""
    grey_data = np.where(np.isnan(data), 0.0, np.nan)
    ax.pcolormesh(Tm, Pm, grey_data,
                  cmap=mcolors.ListedColormap([GREY]),
                  shading="nearest", zorder=1)
    pcm = ax.pcolormesh(Tm, Pm, data, cmap=cmap,
                        vmin=vmin, vmax=vmax,
                        shading="nearest", zorder=2)
    return pcm


def _shared_cbar(fig, pcm, axes_row, label, pad=0.04):
    """Single colourbar spanning a list of axes."""
    cb = fig.colorbar(pcm, ax=axes_row, pad=pad, fraction=0.025, aspect=25)
    cb.set_label(label)
    cb.ax.tick_params(labelsize=7)
    return cb


Tm, Pm = np.meshgrid(T_grid, P_grid)   # both (nP, nT)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Phase regime maps
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 1: phase regime maps ...")

frac_2ph = is_two.mean(axis=2)   # (70, 50, 14) — mean over z

fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.8),
                          constrained_layout=False)
fig.subplots_adjust(left=0.08, right=0.88, top=0.91, bottom=0.10,
                    wspace=0.08, hspace=0.35)

for k, (ims, ms_val) in enumerate(zip(MS_NPX := MS_NPZ_IDX, MS_NPZ_VAL)):
    ax = axes.flat[k]
    ax.set_facecolor(GREY)
    data = np.where(frac_2ph[:, :, ims].T > 0,
                    frac_2ph[:, :, ims].T, np.nan)
    pcm = _pcolor(ax, frac_2ph[:, :, ims].T, CMAP_FRAC, 0, 1, Tm, Pm)
    _log_yaxis(ax, show_labels=(k % 3 == 0))
    ax.set_xlim(T_grid[0], T_grid[-1])
    ax.set_title(_ms_label(ms_val), pad=3)
    if k >= 3:
        ax.set_xlabel(r"$T$ (K)")
    else:
        ax.set_xticklabels([])
    if k % 3 == 0:
        ax.set_ylabel(r"$P$ (bar)")

# shared colourbar on the right
cbar_ax = fig.add_axes([0.90, 0.10, 0.022, 0.81])
cb = fig.colorbar(pcm, cax=cbar_ax)
cb.set_label("Two-phase fraction", labelpad=4)
cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cb.ax.tick_params(labelsize=7)

fig.suptitle(
    r"Fraction of $z_{\mathrm{CO_2}} \in [0.05, 0.90]$ in two-phase region",
    fontsize=9, y=0.97,
)

for ext in ("png", "pdf"):
    fig.savefig(f"{OUT_DIR}/ecpa_phase_map.{ext}")
plt.close(fig)
print("  -> saved ecpa_phase_map")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Aqueous CO2 composition  (2 rows × 3 ms panels, fixed z≈0.55)
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 2: aqueous CO2 composition maps ...")

# choose 6 ms panels (skip ms=0 CPA): ims = 1,2,4,6,10,13
IMS_COMP = MS_ECPA_IDX       # 6 values

vmax_aq = 10.0   # mol-%

fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.8),
                          constrained_layout=False)
fig.subplots_adjust(left=0.08, right=0.88, top=0.91, bottom=0.10,
                    wspace=0.08, hspace=0.35)

pcm_ref = None
for k, ims in enumerate(IMS_COMP):
    ax  = axes.flat[k]
    ms_val = ms_grid[ims]
    ax.set_facecolor(GREY)

    raw  = x4w[:, :, IZ_MID, ims].T   # (nP, nT)
    mask = is_two[:, :, IZ_MID, ims].T
    val  = np.where(mask, raw * 100, np.nan)

    pcm = _pcolor(ax, val, CMAP_AQ, 0, vmax_aq, Tm, Pm)
    pcm_ref = pcm
    _log_yaxis(ax, show_labels=(k % 3 == 0))
    ax.set_xlim(T_grid[0], T_grid[-1])
    ax.set_title(_ms_label(ms_val), pad=3)
    if k >= 3:
        ax.set_xlabel(r"$T$ (K)")
    else:
        ax.set_xticklabels([])
    if k % 3 == 0:
        ax.set_ylabel(r"$P$ (bar)")

cbar_ax = fig.add_axes([0.90, 0.10, 0.022, 0.81])
cb = fig.colorbar(pcm_ref, cax=cbar_ax)
cb.set_label(r"$x_{\mathrm{CO_2}}^{\mathrm{aq}}$ (mol-\%)", labelpad=4)
cb.ax.tick_params(labelsize=7)

fig.suptitle(
    rf"CO$_2$ mole-\% in aqueous phase ($z = {z_grid[IZ_MID]:.2f}$)",
    fontsize=9, y=0.97,
)

for ext in ("png", "pdf"):
    fig.savefig(f"{OUT_DIR}/ecpa_composition_aq.{ext}")
plt.close(fig)
print("  -> saved ecpa_composition_aq")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — H2O in CO2-rich phase (same layout as Fig 2)
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 3: CO2-rich H2O composition maps ...")

vmax_c = 15.0   # mol-%

fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.8),
                          constrained_layout=False)
fig.subplots_adjust(left=0.08, right=0.88, top=0.91, bottom=0.10,
                    wspace=0.08, hspace=0.35)

pcm_ref = None
for k, ims in enumerate(IMS_COMP):
    ax     = axes.flat[k]
    ms_val = ms_grid[ims]
    ax.set_facecolor(GREY)

    raw  = x1c[:, :, IZ_MID, ims].T
    mask = is_two[:, :, IZ_MID, ims].T
    val  = np.where(mask, raw * 100, np.nan)

    pcm = _pcolor(ax, val, CMAP_CO2, 0, vmax_c, Tm, Pm)
    pcm_ref = pcm
    _log_yaxis(ax, show_labels=(k % 3 == 0))
    ax.set_xlim(T_grid[0], T_grid[-1])
    ax.set_title(_ms_label(ms_val), pad=3)
    if k >= 3:
        ax.set_xlabel(r"$T$ (K)")
    else:
        ax.set_xticklabels([])
    if k % 3 == 0:
        ax.set_ylabel(r"$P$ (bar)")

cbar_ax = fig.add_axes([0.90, 0.10, 0.022, 0.81])
cb = fig.colorbar(pcm_ref, cax=cbar_ax)
cb.set_label(r"$x_{\mathrm{H_2O}}^{\mathrm{CO_2}}$ (mol-\%)", labelpad=4)
cb.ax.tick_params(labelsize=7)

fig.suptitle(
    rf"H$_2$O mole-\% in CO$_2$-rich phase ($z = {z_grid[IZ_MID]:.2f}$)",
    fontsize=9, y=0.97,
)

for ext in ("png", "pdf"):
    fig.savefig(f"{OUT_DIR}/ecpa_composition_c.{ext}")
plt.close(fig)
print("  -> saved ecpa_composition_c")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — SSI iteration heatmap
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 4: SSI iteration heatmap ...")

pivot_cache = {}
for ims, ms_val in zip(MS_ECPA_IDX, MS_ECPA_VAL):
    sub = df_2ph[np.abs(df_2ph["ms_feed"] - ms_val) < 1e-6]
    piv = sub.groupby(["T", "P"])["n_ssi_iters"].median().unstack("T")
    piv_full = piv.reindex(columns=T_grid, index=P_grid)
    pivot_cache[ims] = piv_full.values   # (nP, nT)

vmin_ssi, vmax_ssi = 1, 20

fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.8),
                          constrained_layout=False)
fig.subplots_adjust(left=0.08, right=0.88, top=0.91, bottom=0.10,
                    wspace=0.08, hspace=0.35)

pcm_ref = None
for k, (ims, ms_val) in enumerate(zip(MS_ECPA_IDX, MS_ECPA_VAL)):
    ax = axes.flat[k]
    ax.set_facecolor(GREY)

    # grey for single-phase (where all z are single-phase)
    frac = frac_2ph[:, :, ims].T
    data = np.where(frac > 0, pivot_cache[ims], np.nan)

    pcm = _pcolor(ax, data, CMAP_SSI, vmin_ssi, vmax_ssi, Tm, Pm)
    pcm_ref = pcm
    _log_yaxis(ax, show_labels=(k % 3 == 0))
    ax.set_xlim(T_grid[0], T_grid[-1])
    ax.set_title(_ms_label(ms_val), pad=3)
    if k >= 3:
        ax.set_xlabel(r"$T$ (K)")
    else:
        ax.set_xticklabels([])
    if k % 3 == 0:
        ax.set_ylabel(r"$P$ (bar)")

cbar_ax = fig.add_axes([0.90, 0.10, 0.022, 0.81])
cb = fig.colorbar(pcm_ref, cax=cbar_ax, extend="max")
cb.set_label("Median SSI iterations", labelpad=4)
cb.set_ticks([1, 5, 10, 15, 20])
cb.ax.tick_params(labelsize=7)

# add legend patch for single-phase background
from matplotlib.patches import Rectangle
handles = [Rectangle((0, 0), 1, 1, fc=GREY, ec=DARK_GREY, lw=0.5,
                      label="Single-phase")]
axes.flat[0].legend(handles=handles, loc="upper right", fontsize=6,
                     handlelength=1.2, handletextpad=0.4, borderpad=0.4)

fig.suptitle(
    r"Median SSI iterations --- eCPA ternary flash (CO$_2$+H$_2$O+NaCl)",
    fontsize=9, y=0.97,
)

for ext in ("png", "pdf"):
    fig.savefig(f"{OUT_DIR}/ecpa_ssi_heatmap.{ext}")
plt.close(fig)
print("  -> saved ecpa_ssi_heatmap")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Newton solver statistics
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 5: Newton iteration statistics ...")

ims_ref = 6   # ms = 1.0 mol/kg
ms_ref  = ms_grid[ims_ref]
sub_ref = df_2ph[np.abs(df_2ph["ms_feed"] - ms_ref) < 1e-6]

fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.2))
fig.subplots_adjust(left=0.10, right=0.95, top=0.90, bottom=0.10,
                    wspace=0.40, hspace=0.44)

# --- A: aqueous Newton iters per warm-start call
ax_a = axes[0, 0]
aq_iters = (df_2ph["n_newton_aq_iters"] /
            df_2ph["n_newton_aq"].clip(lower=1)).where(df_2ph["n_newton_aq"] > 0)
ax_a.hist(aq_iters.dropna(), bins=np.arange(0.5, 12.5, 1.0),
          color=cmc.batlow(0.3), rwidth=0.8)
med_aq = aq_iters.dropna().median()
ax_a.axvline(med_aq, color="0.2", lw=0.8, ls="--")
ax_a.text(med_aq + 0.15, ax_a.get_ylim()[1] * 0.90,
          rf"median$={med_aq:.1f}$", fontsize=7)
ax_a.set_xlabel(r"Mean Newton iter./call (aqueous)")
ax_a.set_ylabel("Count")
ax_a.set_title("Aqueous inner solver")

# --- B: CO2-rich Newton iters per call
ax_b = axes[0, 1]
c_iters = (df_2ph["n_newton_c_iters"] /
           df_2ph["n_newton_c"].clip(lower=1)).where(df_2ph["n_newton_c"] > 0)
ax_b.hist(c_iters.dropna(), bins=np.arange(0.5, 8.5, 1.0),
          color=cmc.lajolla(0.55), rwidth=0.8)
med_c = c_iters.dropna().median()
ax_b.axvline(med_c, color="0.2", lw=0.8, ls="--")
ax_b.text(med_c + 0.1, ax_b.get_ylim()[1] * 0.90,
          rf"median$={med_c:.1f}$", fontsize=7)
ax_b.set_xlabel(r"Mean Newton iter./call (CO$_2$-rich)")
ax_b.set_ylabel("Count")
ax_b.set_title(r"CO$_2$-rich inner solver")

# --- C: aqueous Newton success rate T-P map
ax_c = axes[1, 0]
ax_c.set_facecolor(GREY)
piv_aq_ok  = sub_ref.groupby(["T", "P"])["n_newton_aq_ok"].sum().unstack("T")
piv_aq_tot = sub_ref.groupby(["T", "P"])["n_newton_aq"].sum().unstack("T")
rate_aq = (piv_aq_ok / piv_aq_tot.clip(lower=1)).reindex(
    columns=T_grid, index=P_grid).values * 100
pcm_aq = _pcolor(ax_c, rate_aq, CMAP_SUCC, 60, 100, Tm, Pm)
_log_yaxis(ax_c)
ax_c.set_xlim(T_grid[0], T_grid[-1])
ax_c.set_xlabel(r"$T$ (K)")
ax_c.set_ylabel(r"$P$ (bar)")
ax_c.set_title(rf"Aqueous success rate [\%], $m_s={ms_ref:.0f}$")
cb_c = fig.colorbar(pcm_aq, ax=ax_c, pad=0.03, fraction=0.046)
cb_c.set_label(r"Success (\%)", labelpad=3)
cb_c.ax.tick_params(labelsize=7)

# --- D: CO2-rich Newton success rate T-P map
ax_d = axes[1, 1]
ax_d.set_facecolor(GREY)
piv_c_ok  = sub_ref.groupby(["T", "P"])["n_newton_c_ok"].sum().unstack("T")
piv_c_tot = sub_ref.groupby(["T", "P"])["n_newton_c"].sum().unstack("T")
rate_c = (piv_c_ok / piv_c_tot.clip(lower=1)).reindex(
    columns=T_grid, index=P_grid).values * 100
pcm_c = _pcolor(ax_d, rate_c, CMAP_SUCC, 60, 100, Tm, Pm)
_log_yaxis(ax_d)
ax_d.set_xlim(T_grid[0], T_grid[-1])
ax_d.set_xlabel(r"$T$ (K)")
ax_d.set_ylabel(r"$P$ (bar)")
ax_d.set_title(rf"CO$_2$-rich success rate [\%], $m_s={ms_ref:.0f}$")
cb_d = fig.colorbar(pcm_c, ax=ax_d, pad=0.03, fraction=0.046)
cb_d.set_label(r"Success (\%)", labelpad=3)
cb_d.ax.tick_params(labelsize=7)

fig.suptitle("Inner Newton solver performance --- eCPA ternary scan",
             fontsize=9, y=0.97)

for ext in ("png", "pdf"):
    fig.savefig(f"{OUT_DIR}/ecpa_newton_stats.{ext}")
plt.close(fig)
print("  -> saved ecpa_newton_stats")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6 — Phase envelope: bubble/dew lines in T-P
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 6: phase envelope ...")

def _find_envelope(is_2ph_iT_iP):
    """
    is_2ph_iT_iP: (nT, nP) bool — True where two-phase.
    Returns dew_P[nT], bubble_P[nT] (NaN where boundary not found).
    """
    nT, nP = is_2ph_iT_iP.shape
    dew    = np.full(nT, np.nan)
    bubble = np.full(nT, np.nan)
    for iT in range(nT):
        row = is_2ph_iT_iP[iT, :]
        if not row.any():
            continue
        idx_lo = np.argmax(row)
        if idx_lo > 0:
            dew[iT] = P_grid[idx_lo]
        idx_hi = len(row) - 1 - np.argmax(row[::-1])
        if idx_hi < nP - 1:
            bubble[iT] = P_grid[idx_hi]
    return dew, bubble


# left: sweep z at ms=0 (CPA)
Z_ENV_IDX  = [5, 11, 17, 22]    # z ≈ 0.20, 0.40, 0.60, 0.80
MS_ENV_IDX = [1, 4, 6, 10]      # ms ≈ 1e-5, 0.5, 1, 3

# 4 colours from lipari for z, 4 from batlow for ms
c_z  = [cmc.lipari(v) for v in [0.15, 0.38, 0.62, 0.85]]
c_ms = [cmc.batlow(v) for v in [0.15, 0.40, 0.60, 0.85]]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.2))
fig.subplots_adjust(left=0.09, right=0.97, top=0.89, bottom=0.14,
                    wspace=0.38)

ims_left = 0   # ms=0 for left panel
for colour, iz in zip(c_z, Z_ENV_IDX):
    is_2ph = is_two[:, :, iz, ims_left]   # (nT, nP)
    dew, bubble = _find_envelope(is_2ph)
    label = rf"$z = {z_grid[iz]:.2f}$"
    ax1.semilogy(T_grid, dew,    color=colour, lw=1.2, ls="-",  label=label)
    ax1.semilogy(T_grid, bubble, color=colour, lw=1.2, ls="--")

ax1.set_xlim(T_grid[0], T_grid[-1])
ax1.set_ylim(P_grid[0], P_grid[-1])
ax1.set_xlabel(r"$T$ (K)")
ax1.set_ylabel(r"$P$ (bar)")
ax1.set_title(r"Sweep $z_{\mathrm{CO_2}}$, $m_s = 0$")
from matplotlib.lines import Line2D
h_auto, l_auto = ax1.get_legend_handles_labels()
h_extra = [Line2D([0], [0], color="0.35", lw=1.2, ls="-",  label="Dew"),
           Line2D([0], [0], color="0.35", lw=1.2, ls="--", label="Bubble")]
ax1.legend(handles=h_auto + h_extra, labels=l_auto + ["Dew", "Bubble"],
           fontsize=6.5, title=r"CO$_2$ feed", title_fontsize=7,
           handlelength=2.0, framealpha=0.9)

iz_right = IZ_MID  # z ≈ 0.55
for colour, ims in zip(c_ms, MS_ENV_IDX):
    is_2ph = is_two[:, :, iz_right, ims]
    dew, bubble = _find_envelope(is_2ph)
    ms_val = ms_grid[ims]
    ms_str = r"\approx 0" if ms_val < 1e-3 else f"{ms_val:g}"
    label  = rf"$m_s = {ms_str}$"
    ax2.semilogy(T_grid, dew,    color=colour, lw=1.2, ls="-",  label=label)
    ax2.semilogy(T_grid, bubble, color=colour, lw=1.2, ls="--")

ax2.set_xlim(T_grid[0], T_grid[-1])
ax2.set_ylim(P_grid[0], P_grid[-1])
ax2.set_xlabel(r"$T$ (K)")
ax2.set_ylabel(r"$P$ (bar)")
ax2.set_title(rf"Sweep $m_s$, $z = {z_grid[iz_right]:.2f}$")
ax2.legend(fontsize=6.5, title=r"NaCl molality", title_fontsize=7,
           handlelength=2.0, framealpha=0.9)

fig.suptitle(
    r"Phase envelope --- CO$_2$+H$_2$O+NaCl (eCPA, $T = 288$--633\,K)",
    fontsize=9, y=0.98,
)

for ext in ("png", "pdf"):
    fig.savefig(f"{OUT_DIR}/ecpa_phase_envelope.{ext}")
plt.close(fig)
print("  -> saved ecpa_phase_envelope")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 7 — Salting-out: CO2 solubility vs NaCl molality
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 7: salting-out summary ...")

IZ_SALT  = 14    # z ≈ 0.546
IP_SALT  = 30    # P ≈ 88 bar
IMS_SALT = slice(1, None)   # skip ms=0 (CPA, phase labels swap at high T)

# Choose 5 isotherms, well-spread, all in two-phase region at these conditions
T_SALT_IDX = [4, 12, 22, 34, 48]
T_SALT_VAL = T_grid[T_SALT_IDX]

norm_T = mcolors.Normalize(vmin=T_SALT_VAL.min(), vmax=T_SALT_VAL.max())
ms_plot = ms_grid[IMS_SALT]

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(6.5, 3.0))
fig.subplots_adjust(left=0.10, right=0.86, top=0.88, bottom=0.16,
                    wspace=0.40)

for iT, T_val in zip(T_SALT_IDX, T_SALT_VAL):
    clr = CMAP_TEMP(norm_T(T_val))
    x4w_ms  = x4w[iT, IP_SALT, IZ_SALT, IMS_SALT]
    x1w_ms  = x1w[iT, IP_SALT, IZ_SALT, IMS_SALT]
    mask    = is_two[iT, IP_SALT, IZ_SALT, IMS_SALT]
    mc   = np.where(mask, x4w_ms / (x1w_ms * Mw), np.nan)
    xco2 = np.where(mask, x4w_ms * 100, np.nan)
    ax_l.plot(ms_plot, mc,   color=clr, lw=1.2, marker="o", ms=3,
               label=rf"${T_val:.0f}$ K")
    ax_r.plot(ms_plot, xco2, color=clr, lw=1.2, marker="o", ms=3)

ax_l.set_xlabel(r"$m_s$ (mol\,kg$^{-1}$)")
ax_l.set_ylabel(r"$m_c^{\mathrm{aq}}$ (mol\,kg$^{-1}$)")
ax_l.set_title(rf"CO$_2$ solubility, $z={z_grid[IZ_SALT]:.2f}$, $P={P_grid[IP_SALT]:.0f}$ bar")
ax_l.legend(fontsize=7, title="$T$", title_fontsize=7,
             handlelength=1.5, framealpha=0.9)

ax_r.set_xlabel(r"$m_s$ (mol\,kg$^{-1}$)")
ax_r.set_ylabel(r"$x_{\mathrm{CO_2}}^{\mathrm{aq}}$ (mol-\%)")
ax_r.set_title(rf"CO$_2$ mole-\%, $z={z_grid[IZ_SALT]:.2f}$, $P={P_grid[IP_SALT]:.0f}$ bar")

sm = plt.cm.ScalarMappable(cmap=CMAP_TEMP, norm=norm_T)
sm.set_array([])
cbar_ax = fig.add_axes([0.88, 0.16, 0.025, 0.72])
cb = fig.colorbar(sm, cax=cbar_ax)
cb.set_label(r"$T$ (K)", labelpad=3)
cb.ax.tick_params(labelsize=7)

fig.suptitle(
    r"Salting-out effect --- CO$_2$+H$_2$O+NaCl (eCPA EoS)",
    fontsize=9, y=0.99,
)

for ext in ("png", "pdf"):
    fig.savefig(f"{OUT_DIR}/ecpa_salting_out.{ext}")
plt.close(fig)
print("  -> saved ecpa_salting_out")

print(f"\nAll figures saved to {OUT_DIR}/")
