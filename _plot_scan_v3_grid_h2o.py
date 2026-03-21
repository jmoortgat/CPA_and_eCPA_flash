"""
Tight 2D heatmap grid: H2O in CO2-rich phase (x1c, mol-%).

Layout: 3 rows (z≈0.085, 0.510, 0.900) × 6 cols (ms≈0, 0.1, 0.5, 1, 3, 6).
Log-normalised colormap; no panel titles; row/column header labels only.

Output: figures/scan_v3/ecpa_composition_c_grid.png/.pdf
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cmcrameri.cm as cmc
import scienceplots  # noqa: F401
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
T_grid = npz["T_grid"]          # (70,)
P_grid = npz["P_grid"]          # (50,)
z_grid = npz["z_grid"]          # (25,)
ms_grid= npz["ms_grid"]         # (14,)
is_two = npz["is_two_phase"]    # (70,50,25,14)
x1c    = npz["x1c"]             # (70,50,25,14)  H2O in CO2-rich phase

# ── panel indices ──────────────────────────────────────────────────────────────
# rows: z ≈ 0.085, 0.510, 0.900
IZ_ROWS = [1, 13, 24]
# cols: eCPA ms panels (skip ms=0 CPA)
IMS_COLS = [1, 2, 4, 6, 10, 13]   # ms ≈ 1e-5, 0.1, 0.5, 1, 3, 6

NR = len(IZ_ROWS)
NC = len(IMS_COLS)

# ── colormap / norm ────────────────────────────────────────────────────────────
CMAP = cmc.lajolla    # warm palette: distinguishes from CO2-aq figure (lapaz)
VMIN = 0.05           # mol-%
VMAX = 75.0           # mol-% (true max ≈ 72 mol-%)
NORM = mcolors.LogNorm(vmin=VMIN, vmax=VMAX)

GREY_SINGLE = "#d0d0d0"

# ── T–P meshgrid for pcolormesh ────────────────────────────────────────────────
def _edges(v):
    mid = 0.5 * (v[:-1] + v[1:])
    lo  = v[0]  - (mid[0]  - v[0])
    hi  = v[-1] + (v[-1]  - mid[-1])
    return np.concatenate([[lo], mid, [hi]])

def _find_envelope(is_2ph):
    """
    is_2ph: (nT, nP) bool — True where two-phase.
    Returns dew_logP[nT], bubble_logP[nT] (NaN where boundary absent).
    'dew'    = lowest-P two-phase boundary.
    'bubble' = highest-P two-phase boundary.
    """
    nT, nP = is_2ph.shape
    dew_lP    = np.full(nT, np.nan)
    bubble_lP = np.full(nT, np.nan)
    for iT in range(nT):
        row = is_2ph[iT, :]
        if not row.any():
            continue
        idx_lo = np.argmax(row)
        if idx_lo > 0:
            dew_lP[iT] = logP[idx_lo]
        idx_hi = len(row) - 1 - np.argmax(row[::-1])
        if idx_hi < nP - 1:
            bubble_lP[iT] = logP[idx_hi]
    return dew_lP, bubble_lP

T_C        = T_grid - 273.15
T_edges    = _edges(T_C)
logP       = np.log10(P_grid)
logP_edges = _edges(logP)

T_TICKS    = [50, 150, 250, 350]
P_TICKS    = [1, 10, 100, 1000]
LOGP_TICKS = np.log10(P_TICKS)
P_LABELS   = ["1", "10", "100", "1000"]

# ── column / row labels ────────────────────────────────────────────────────────
def _ms_col_label(ms_val):
    if ms_val < 1e-3:
        return r"$m_s \approx 0$"
    return rf"$m_s = {ms_val:g}$"

COL_LABELS = [_ms_col_label(ms_grid[i]) for i in IMS_COLS]
ROW_LABELS = [rf"$z = {z_grid[iz]:.2f}$" for iz in IZ_ROWS]

# ── figure ─────────────────────────────────────────────────────────────────────
FIG_W = 7.2
FIG_H = 4.0

fig, axes = plt.subplots(
    NR, NC,
    figsize=(FIG_W, FIG_H),
    gridspec_kw=dict(wspace=0.04, hspace=0.06,
                     left=0.13, right=0.88,
                     top=0.92, bottom=0.10),
)

for ri, iz in enumerate(IZ_ROWS):
    for ci, ims in enumerate(IMS_COLS):
        ax = axes[ri, ci]

        val  = x1c[:, :, iz, ims].T * 100          # (50, 70) mol-%
        mask = is_two[:, :, iz, ims].T              # (50, 70) bool

        ax.set_facecolor(GREY_SINGLE)

        val_plot = np.where(mask, np.clip(val, VMIN, None), np.nan)
        pc = ax.pcolormesh(
            T_edges, logP_edges, val_plot,
            cmap=CMAP, norm=NORM, shading="flat", rasterized=True,
        )

        # phase envelope overlay
        dew_lP, bubble_lP = _find_envelope(is_two[:, :, iz, ims])
        ax.plot(T_C, dew_lP,    color="r", lw=0.8, ls="-",  alpha=0.90, zorder=3)
        ax.plot(T_C, bubble_lP, color="r", lw=0.8, ls="--", alpha=0.90, zorder=3)

        ax.set_xlim(T_C[0], T_C[-1])
        ax.set_ylim(logP[0], logP[-1])

        if ri == NR - 1:
            ax.set_xticks(T_TICKS)
            labels = [str(t) for t in T_TICKS]
            if ci < NC - 1:
                labels[-1] = ""   # drop "350" to avoid overlap with next panel
            ax.set_xticklabels(labels, fontsize=6.5)
        else:
            ax.set_xticks([])

        if ci == 0:
            ax.set_yticks(LOGP_TICKS)
            ax.set_yticklabels(P_LABELS, fontsize=6.5)
        else:
            ax.set_yticks([])

        ax.tick_params(axis="both", length=2, pad=1.5)

# ── envelope legend — bottom-right white strip, outside panels ─────────────────
from matplotlib.lines import Line2D
_handles = [
    Line2D([0], [0], color="r", lw=0.9, ls="-",  label="Dew point"),
    Line2D([0], [0], color="r", lw=0.9, ls="--", label="Bubble point"),
]
fig.legend(handles=_handles, fontsize=6, ncol=2,
           loc="lower right",
           bbox_to_anchor=(0.885, 0.01),
           bbox_transform=fig.transFigure,
           framealpha=0.0, handlelength=1.8, borderpad=0.3,
           handletextpad=0.5, columnspacing=1.0)

# ── axis titles ────────────────────────────────────────────────────────────────
fig.text(0.505, 0.01, r"$T$ ($^\circ$C)", ha="center", va="bottom", fontsize=8)
axes[1, 0].set_ylabel(r"$P$ (bar)", fontsize=8, labelpad=2)

# ── column headers ─────────────────────────────────────────────────────────────
for ci, lbl in enumerate(COL_LABELS):
    ax = axes[0, ci]
    x_fig = ax.get_position().x0 + ax.get_position().width / 2
    fig.text(x_fig, 0.945, lbl, ha="center", va="bottom", fontsize=7.5,
             transform=fig.transFigure)

# ── row labels ─────────────────────────────────────────────────────────────────
for ri, lbl in enumerate(ROW_LABELS):
    ax = axes[ri, 0]
    y_fig = ax.get_position().y0 + ax.get_position().height / 2
    fig.text(0.048, y_fig, lbl, ha="left", va="center", fontsize=7.5,
             rotation=90, transform=fig.transFigure)

# ── colorbar ──────────────────────────────────────────────────────────────────
cax = fig.add_axes([0.895, 0.10, 0.018, 0.82])
cb  = fig.colorbar(pc, cax=cax, norm=NORM, cmap=CMAP)
cb.set_label(r"$y_{\mathrm{H_2O}}$ (mol-\%)", fontsize=8, labelpad=4)
cb_ticks = [0.05, 0.1, 0.5, 1, 5, 10, 25, 75]
cb.set_ticks(cb_ticks)
cb.set_ticklabels([f"{v:g}" for v in cb_ticks], fontsize=6.5)

# ── save ───────────────────────────────────────────────────────────────────────
outname = "ecpa_composition_c_grid"
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT_DIR}/{outname}.{ext}")
plt.close(fig)
print(f"  -> {OUT_DIR}/{outname}.png/.pdf")
print("Done.")
