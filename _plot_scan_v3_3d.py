"""
3D surface plot: CO2 mole-% in aqueous phase vs T-P for 6 NaCl molalities.

T on x-axis, log(P) on y-axis, x_CO2^aq (mol-%) on z-axis.
Each of 6 panels shows a different NaCl molality — surfaces sink with salinity.
A grey "footprint" at z=0 shows the two-phase region boundary.

Usage:
    python _plot_scan_v3_3d.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import cmcrameri.cm as cmc
import scienceplots   # noqa: F401
import os

OUT_DIR = "figures/scan_v3"
os.makedirs(OUT_DIR, exist_ok=True)

# ── style ─────────────────────────────────────────────────────────────────────
# scienceplots mostly works for 3D; override pane/grid colours manually
plt.style.use(["science"])
plt.rcParams.update({
    "figure.dpi":   150,
    "savefig.dpi":  300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.5,
})

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading data ...")
npz    = np.load("results/scan_v3_table.npz")
T_grid = npz["T_grid"]       # (70,)
P_grid = npz["P_grid"]       # (50,)
z_grid = npz["z_grid"]       # (25,)
ms_grid= npz["ms_grid"]      # (14,)
is_two = npz["is_two_phase"] # (70, 50, 25, 14)
x4w    = npz["x4w"]          # CO2 in aq,  NaN where single-phase

IZ_MID      = 14    # z ≈ 0.55
IMS_PANELS  = [1, 2, 4, 6, 10, 13]   # ms ≈ 1e-5, 0.1, 0.5, 1, 3, 6  (eCPA)
MS_VALS     = ms_grid[IMS_PANELS]

# ── grids ─────────────────────────────────────────────────────────────────────
logP = np.log10(P_grid)       # use log10(P) as y-coordinate
T_mesh, logP_mesh = np.meshgrid(T_grid, logP)   # (nP, nT)

# ── colour setup ──────────────────────────────────────────────────────────────
CMAP      = cmc.lapaz
VMIN, VMAX = 0.0, 10.0       # mol-% range for colour + z-axis

GREY_PANE   = "#f0f0f0"
GREY_SHADOW = "#b0b0b0"

def _ms_label(ms_val):
    if ms_val < 1e-4:
        return r"$m_s \approx 0$"
    s = f"{ms_val:g}"
    return rf"$m_s = {s}\ \mathrm{{mol\,kg^{{-1}}}}$"

# ── P-axis tick helper ────────────────────────────────────────────────────────
P_TICKS     = [1, 10, 100, 1000]
LOGP_TICKS  = np.log10(P_TICKS)
P_TICKLABELS= ["1", "10", "100", "1000"]

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(11, 7.5))
fig.subplots_adjust(left=0.00, right=0.92, top=0.95, bottom=0.02,
                    wspace=0.02, hspace=0.05)

# shared normaliser / scalar mappable for the colourbar
norm = mcolors.Normalize(vmin=VMIN, vmax=VMAX)
sm   = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
sm.set_array([])

ELEV, AZIM = 28, -55    # viewing angle — same for all panels

for k, (ims, ms_val) in enumerate(zip(IMS_PANELS, MS_VALS)):
    ax = fig.add_subplot(2, 3, k + 1, projection="3d")

    # ── surface data ──────────────────────────────────────────────────────────
    val  = x4w[:, :, IZ_MID, ims].T * 100    # (nP, nT) in mol-%
    mask = is_two[:, :, IZ_MID, ims].T        # True where two-phase

    # mask single-phase to NaN (leaves transparent gaps in surface)
    val_masked = np.where(mask, val, np.nan)

    # face colours: map CO2 content through CMAP
    fcolors = CMAP(norm(np.nan_to_num(val_masked, nan=0.0)))
    # make single-phase faces fully transparent
    fcolors[~mask, 3] = 0.0

    surf = ax.plot_surface(
        T_mesh, logP_mesh, val_masked,
        facecolors=fcolors,
        linewidth=0, antialiased=True,
        shade=True,
        rstride=1, cstride=1,
    )

    # ── grey shadow (footprint) on the z=0 floor ──────────────────────────────
    # fill two-phase footprint as a flat surface at z=0
    shadow = np.where(mask, 0.0, np.nan)
    shadow_colors = np.full((*mask.shape, 4), [0.72, 0.72, 0.72, 0.55])
    shadow_colors[~mask, 3] = 0.0

    ax.plot_surface(
        T_mesh, logP_mesh, shadow,
        facecolors=shadow_colors,
        linewidth=0, antialiased=False,
        rstride=1, cstride=1,
        zorder=0,
    )

    # ── axes styling ──────────────────────────────────────────────────────────
    ax.set_xlim(T_grid[0], T_grid[-1])
    ax.set_ylim(logP[0], logP[-1])
    ax.set_zlim(0, VMAX)

    ax.set_xticks([300, 400, 500, 600])
    ax.set_xticklabels(["300", "400", "500", "600"], fontsize=6.5)
    ax.set_yticks(LOGP_TICKS)
    ax.set_yticklabels(P_TICKLABELS, fontsize=6.5)
    ax.set_zticks([0, 2, 4, 6, 8, 10])
    ax.set_zticklabels(["0", "2", "4", "6", "8", "10"], fontsize=6.5)

    if k % 3 == 0:
        ax.set_ylabel(r"$P$ (bar)", labelpad=4, fontsize=8)
    else:
        ax.set_yticklabels([])
    if k >= 3:
        ax.set_xlabel(r"$T$ (K)", labelpad=4, fontsize=8)
    else:
        ax.set_xticklabels([])
    if k % 3 == 2:
        ax.set_zlabel(r"$x_{\mathrm{CO_2}}^{\mathrm{aq}}$ (mol-\%)",
                      labelpad=6, fontsize=8)
    else:
        ax.set_zticklabels([])

    ax.set_title(_ms_label(ms_val), pad=2, fontsize=8.5)
    ax.view_init(elev=ELEV, azim=AZIM)

    # clean pane colours
    ax.xaxis.pane.fill = True;  ax.xaxis.pane.set_facecolor(GREY_PANE)
    ax.yaxis.pane.fill = True;  ax.yaxis.pane.set_facecolor(GREY_PANE)
    ax.zaxis.pane.fill = True;  ax.zaxis.pane.set_facecolor("#e8e8e8")
    ax.xaxis.pane.set_edgecolor("0.6")
    ax.yaxis.pane.set_edgecolor("0.6")
    ax.zaxis.pane.set_edgecolor("0.6")
    ax.grid(True, lw=0.35, color="0.75")
    ax.tick_params(axis="both", pad=1)

# ── shared colourbar ──────────────────────────────────────────────────────────
cbar_ax = fig.add_axes([0.93, 0.12, 0.018, 0.74])
cb = fig.colorbar(sm, cax=cbar_ax)
cb.set_label(r"$x_{\mathrm{CO_2}}^{\mathrm{aq}}$ (mol-\%)", labelpad=4)
cb.ax.tick_params(labelsize=8)

fig.suptitle(
    rf"CO$_2$ solubility in aqueous phase ($z = {z_grid[IZ_MID]:.2f}$) --- "
    r"eCPA EoS, CO$_2$+H$_2$O+NaCl",
    fontsize=9.5, y=0.99,
)

for ext in ("png", "pdf"):
    fig.savefig(f"{OUT_DIR}/ecpa_composition_aq_3d.{ext}")
plt.close(fig)
print(f"Saved to {OUT_DIR}/ecpa_composition_aq_3d.png/.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Bonus: single 3D axes with all 6 surfaces stacked, coloured by ms
# ═══════════════════════════════════════════════════════════════════════════════
print("Rendering stacked single-axis version ...")

# 6 colours along batlow for the 6 molalities
ms_colours = [cmc.batlow(v) for v in np.linspace(0.1, 0.9, len(IMS_PANELS))]

fig2 = plt.figure(figsize=(7.5, 5.5))
fig2.subplots_adjust(left=0.0, right=0.84, top=0.95, bottom=0.02)
ax3 = fig2.add_subplot(1, 1, 1, projection="3d")

from scipy.ndimage import binary_closing

for (ims, ms_val), colour in zip(zip(IMS_PANELS, MS_VALS), ms_colours):
    val  = x4w[:, :, IZ_MID, ims].T * 100
    mask = is_two[:, :, IZ_MID, ims].T

    # morphological closing: fill isolated single-cell holes that cause
    # rendering artefacts at NaN boundaries in matplotlib 3D
    mask_clean = binary_closing(mask, structure=np.ones((3, 3)))
    val_masked = np.where(mask_clean, np.where(mask, val, np.nanmean(val[mask])), np.nan)

    ax3.plot_surface(
        T_mesh, logP_mesh, val_masked,
        color=colour[:3],
        alpha=0.80,
        linewidth=0, antialiased=True, shade=True,
        rstride=1, cstride=1,
    )

ax3.set_xlim(T_grid[0], T_grid[-1])
ax3.set_ylim(logP[0], logP[-1])
ax3.set_zlim(0, VMAX)
ax3.set_xticks([300, 400, 500, 600])
ax3.set_xticklabels(["300", "400", "500", "600"], fontsize=8)
ax3.set_yticks(LOGP_TICKS)
ax3.set_yticklabels(P_TICKLABELS, fontsize=8)
ax3.set_zticks([0, 2, 4, 6, 8, 10])
ax3.set_xlabel(r"$T$ (K)", labelpad=6, fontsize=9)
ax3.set_ylabel(r"$P$ (bar)", labelpad=6, fontsize=9)
ax3.set_zlabel(r"$x_{\mathrm{CO_2}}^{\mathrm{aq}}$ (mol-\%)", labelpad=8, fontsize=9)
ax3.view_init(elev=26, azim=-42)   # tilt a little higher to see low-T peak better

ax3.xaxis.pane.fill = True;  ax3.xaxis.pane.set_facecolor(GREY_PANE)
ax3.yaxis.pane.fill = True;  ax3.yaxis.pane.set_facecolor(GREY_PANE)
ax3.zaxis.pane.fill = True;  ax3.zaxis.pane.set_facecolor("#e8e8e8")
ax3.xaxis.pane.set_edgecolor("0.6")
ax3.yaxis.pane.set_edgecolor("0.6")
ax3.zaxis.pane.set_edgecolor("0.6")
ax3.grid(True, lw=0.35, color="0.75")

# manual legend from batlow colour patches
from matplotlib.patches import Patch
handles = [Patch(facecolor=c, edgecolor="none",
                 label=(r"$m_s \approx 0$" if mv < 1e-3 else rf"$m_s = {mv:g}$"))
           for c, mv in zip(ms_colours, MS_VALS)]
ax3.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.90),
           fontsize=8, title=r"NaCl (mol\,kg$^{-1}$)", title_fontsize=8,
           framealpha=0.9, handlelength=1.2)

fig2.suptitle(
    rf"CO$_2$ solubility --- salting-out effect ($z = {z_grid[IZ_MID]:.2f}$, eCPA)",
    fontsize=10, y=0.99,
)

for ext in ("png", "pdf"):
    fig2.savefig(f"{OUT_DIR}/ecpa_composition_aq_3d_stacked.{ext}")
plt.close(fig2)
print(f"Saved to {OUT_DIR}/ecpa_composition_aq_3d_stacked.png/.pdf")
