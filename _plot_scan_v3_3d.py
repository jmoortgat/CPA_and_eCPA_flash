"""
3D stacked surface plots from the eCPA v3 solution table.

Generates 6 figures (figures/scan_v3/):

CO2 in aqueous phase (x4w, mol-%), 3 feed fractions:
  ecpa_aq_co2_z025_3d.png/pdf  (z ≈ 0.26)
  ecpa_aq_co2_z050_3d.png/pdf  (z ≈ 0.51)
  ecpa_aq_co2_z075_3d.png/pdf  (z ≈ 0.76)

H2O in CO2-rich phase (x1c, mol-%), 3 feed fractions:
  ecpa_co2_h2o_z025_3d.png/pdf
  ecpa_co2_h2o_z050_3d.png/pdf
  ecpa_co2_h2o_z075_3d.png/pdf

Each figure: 6 NaCl molalities stacked on one 3D axis, coloured by ms (batlow).

Usage:
    python _plot_scan_v3_3d.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cmcrameri.cm as cmc
import scienceplots  # noqa: F401
from scipy.ndimage import binary_closing
import os

OUT_DIR = "figures/scan_v3"
os.makedirs(OUT_DIR, exist_ok=True)

plt.style.use(["science"])
plt.rcParams.update({
    "figure.dpi":   150,
    "savefig.dpi":  300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.5,
})

# ── data ──────────────────────────────────────────────────────────────────────
print("Loading data ...")
npz    = np.load("results/scan_v3_table.npz")
T_grid = npz["T_grid"]
P_grid = npz["P_grid"]
z_grid = npz["z_grid"]
ms_grid= npz["ms_grid"]
is_two = npz["is_two_phase"]   # (70,50,25,14)
x4w    = npz["x4w"]            # CO2 in aqueous, NaN where single-phase
x1c    = npz["x1c"]            # H2O in CO2-rich, NaN where single-phase

# ms panels: skip ms=0 (CPA); use eCPA ms values
IMS_PANELS = [1, 2, 4, 6, 10, 13]   # ms ≈ 1e-5, 0.1, 0.5, 1, 3, 6
MS_VALS    = ms_grid[IMS_PANELS]

# z panels: ≈ 25%, 50%, 75% CO2 feed
IZ_PANELS = [6, 13, 20]   # z ≈ 0.26, 0.51, 0.76

# T-P meshgrid (log P as y-coordinate)
logP = np.log10(P_grid)
T_mesh, logP_mesh = np.meshgrid(T_grid, logP)   # (nP, nT)
P_TICKS      = [1, 10, 100, 1000]
LOGP_TICKS   = np.log10(P_TICKS)
P_TICKLABELS = ["1", "10", "100", "1000"]

# ms colours: batlow (dark teal → pink), 6 levels
MS_COLOURS = [cmc.batlow(v) for v in np.linspace(0.10, 0.88, len(IMS_PANELS))]

GREY_PANE  = "#f0f0f0"
# viewing angles: aq (broad surface) vs h2o (nearly-T-only surface)
ELEV_AQ,  AZIM_AQ  = 26, -55
ELEV_H2O, AZIM_H2O = 30, -38   # rotate more toward T-face for H2O


def _ms_label(ms_val):
    if ms_val < 1e-3:
        return r"$m_s \approx 0$"
    return rf"$m_s = {ms_val:g}$"


def _make_stacked(iz, data_arr, zlabel, vmax, outname,
                  close_size=3, elev=ELEV_AQ, azim=AZIM_AQ):
    """
    iz        : index into z_grid
    data_arr  : (70,50,25,14) array in mol-%
    zlabel    : string for the z-axis label (LaTeX)
    vmax      : max for z-axis (data is clipped, not clamped)
    outname   : stem, e.g. 'ecpa_aq_co2_z050_3d'
    """
    fig = plt.figure(figsize=(7.2, 5.4))
    # leave room on left for legend, right for z-axis ticks
    fig.subplots_adjust(left=0.0, right=0.96, top=0.97, bottom=0.02)
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    # draw highest surface last so it sits in front (painter's algorithm)
    for (ims, ms_val), colour in zip(
            zip(reversed(IMS_PANELS), reversed(MS_VALS)), reversed(MS_COLOURS)):
        val  = data_arr[:, :, iz, ims].T * 100    # (nP, nT)
        mask = is_two[:, :, iz, ims].T

        # close isolated holes to eliminate NaN-boundary artefacts
        mask_cl = binary_closing(mask, structure=np.ones((close_size, close_size)))
        # fill closed holes with local mean; no upper clipping — show true data range
        fill_val = float(np.nanmean(val[mask])) if mask.any() else 0.0
        val_surf = np.where(mask_cl, np.where(mask, val, fill_val), np.nan)
        val_surf = np.clip(val_surf, 0, None)   # only clip negatives (numerical noise)

        ax.plot_surface(
            T_mesh, logP_mesh, val_surf,
            color=colour[:3],
            alpha=0.82,
            linewidth=0, antialiased=True, shade=True,
            rstride=1, cstride=1,
        )

    # axes limits and ticks
    ax.set_xlim(T_grid[0], T_grid[-1])
    ax.set_ylim(logP[0], logP[-1])
    ax.set_zlim(0, vmax)

    ax.set_xticks([300, 400, 500, 600])
    ax.set_xticklabels(["300", "400", "500", "600"], fontsize=7.5)
    ax.set_yticks(LOGP_TICKS)
    ax.set_yticklabels(P_TICKLABELS, fontsize=7.5)
    # clean round z-ticks: 5 intervals, snapped to nearest 5 or 1
    step = max(1, round(vmax / 5 / 5) * 5) if vmax >= 10 else 1
    zticks = np.arange(0, vmax + step * 0.01, step)
    ax.set_zticks(zticks)
    ax.set_zticklabels([f"{v:.0f}" for v in zticks], fontsize=7.5)

    ax.set_xlabel(r"$T$ (K)",   labelpad=5, fontsize=9)
    ax.set_ylabel(r"$P$ (bar)", labelpad=5, fontsize=9)
    # matplotlib 3D z-label is frequently clipped; use fig.text() instead
    ax.set_zlabel("")
    fig.text(0.97, 0.52, zlabel, rotation=90, ha="center", va="center",
             fontsize=9, transform=fig.transFigure)

    ax.view_init(elev=elev, azim=azim)

    # pane styling
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = True
        pane.set_edgecolor("0.60")
    ax.xaxis.pane.set_facecolor(GREY_PANE)
    ax.yaxis.pane.set_facecolor(GREY_PANE)
    ax.zaxis.pane.set_facecolor("#e8e8e8")
    ax.grid(True, lw=0.35, color="0.75")
    ax.tick_params(axis="both", pad=1)

    # legend: figure-level, anchored to top-left of figure in figure coordinates
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c[:3], edgecolor="0.4", lw=0.4,
                     label=_ms_label(mv))
               for c, mv in zip(MS_COLOURS, MS_VALS)]
    fig.legend(
        handles=handles,
        title=r"NaCl (mol\,kg$^{-1}$)",
        title_fontsize=8,
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.97),   # figure coords: top-left
        framealpha=0.92,
        handlelength=1.2,
        borderpad=0.5,
    )

    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT_DIR}/{outname}.{ext}")
    plt.close(fig)
    print(f"  -> {OUT_DIR}/{outname}.png/.pdf")


# ── CO2 in aqueous phase  (mol-%) ─────────────────────────────────────────────
print("\nCO2 in aqueous phase ...")
zlabel_aq = r"$x_{\mathrm{CO_2}}^{\mathrm{aq}}$ (mol-\%)"

for iz in IZ_PANELS:
    z_val  = z_grid[iz]
    z_pct  = round(z_val * 100)
    outname = f"ecpa_aq_co2_z{z_pct:03d}_3d"
    print(f"  z = {z_val:.2f} ({z_pct}%) ...")
    _make_stacked(iz, x4w, zlabel_aq, vmax=25.0, outname=outname)

# ── H2O in CO2-rich phase  (mol-%) ───────────────────────────────────────────
print("\nH2O in CO2-rich phase ...")
zlabel_c = r"$x_{\mathrm{H_2O}}^{\mathrm{CO_2}}$ (mol-\%)"

for iz in IZ_PANELS:
    z_val  = z_grid[iz]
    z_pct  = round(z_val * 100)
    outname = f"ecpa_co2_h2o_z{z_pct:03d}_3d"
    print(f"  z = {z_val:.2f} ({z_pct}%) ...")
    _make_stacked(iz, x1c, zlabel_c, vmax=75.0, outname=outname,
                  close_size=5, elev=ELEV_H2O, azim=AZIM_H2O)

print("\nDone.")
