"""
Single-row SSI iteration heatmap: median K-value SSI iterations per (T, P) cell,
mediated over all feed fractions z, for 6 NaCl molalities.

Matches the layout style of ecpa_composition_aq_grid / ecpa_composition_c_grid.

Output: figures/scan_v3/ecpa_ssi_heatmap.png/.pdf
"""

import numpy as np
import pandas as pd
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
npz     = np.load("results/scan_v3_table.npz")
T_grid  = npz["T_grid"]          # (70,) K
P_grid  = npz["P_grid"]          # (50,)
ms_grid = npz["ms_grid"]         # (14,)
is_two  = npz["is_two_phase"]    # (70, 50, 25, 14)

df      = pd.read_parquet("results/scan_v3_metrics.parquet")
df_2ph  = df[(df["eos_type"] == "eCPA") & df["is_two_phase"]].copy()

# fraction of z-values that are two-phase at each (T, P, ms) — used for mask
frac_2ph = is_two.mean(axis=2)   # (70, 50, 14)

# ── panel indices ──────────────────────────────────────────────────────────────
IMS_COLS = [1, 2, 4, 6, 10, 13]   # ms ≈ 1e-5, 0.1, 0.5, 1, 3, 6
NC = len(IMS_COLS)

# ── colormap / norm ────────────────────────────────────────────────────────────
CMAP       = cmc.lajolla          # light (few iters) → dark red (many)
VMIN, VMAX = 1, 20
NORM       = mcolors.Normalize(vmin=VMIN, vmax=VMAX)

GREY_SINGLE = "#d0d0d0"

# ── T-P coordinate arrays (Celsius, log P) ────────────────────────────────────
def _edges(v):
    mid = 0.5 * (v[:-1] + v[1:])
    lo  = v[0]  - (mid[0]  - v[0])
    hi  = v[-1] + (v[-1]  - mid[-1])
    return np.concatenate([[lo], mid, [hi]])

T_C        = T_grid - 273.15
T_edges    = _edges(T_C)
logP       = np.log10(P_grid)
logP_edges = _edges(logP)

T_TICKS    = [50, 150, 250, 350]
P_TICKS    = [1, 10, 100, 1000]
LOGP_TICKS = np.log10(P_TICKS)
P_LABELS   = ["1", "10", "100", "1000"]

# ── build median SSI pivot for each ms panel ──────────────────────────────────
# group parquet rows by (T_K, P) → median n_ssi_iters across all z
print("Building SSI pivot tables ...")
pivot_cache = {}
for ims in IMS_COLS:
    ms_val = ms_grid[ims]
    sub    = df_2ph[np.abs(df_2ph["ms_feed"] - ms_val) < 1e-6]
    piv    = sub.groupby(["T", "P"])["n_ssi_iters"].median().unstack("T")
    # reindex to full grid (NaN where no two-phase rows)
    piv_full = piv.reindex(index=P_grid, columns=T_grid)
    # (nP, nT) — matches pcolormesh(T_edges, logP_edges, data)
    pivot_cache[ims] = piv_full.values

# ── column header labels ───────────────────────────────────────────────────────
def _ms_col_label(ms_val):
    if ms_val < 1e-3:
        return r"$m_s \approx 0$"
    return rf"$m_s = {ms_val:g}$"

COL_LABELS = [_ms_col_label(ms_grid[i]) for i in IMS_COLS]

# ── figure ─────────────────────────────────────────────────────────────────────
FIG_W = 7.2
FIG_H = 2.2    # single row → shorter figure

fig, axes = plt.subplots(
    1, NC,
    figsize=(FIG_W, FIG_H),
    gridspec_kw=dict(wspace=0.04,
                     left=0.13, right=0.88,
                     top=0.88, bottom=0.16),
)

for ci, ims in enumerate(IMS_COLS):
    ax = axes[ci]

    # mask: grey where entirely single-phase
    ax.set_facecolor(GREY_SINGLE)

    # SSI data: NaN where no two-phase rows exist at this (T,P,ms)
    data     = pivot_cache[ims]                      # (nP, nT)
    frac_row = frac_2ph[:, :, ims].T                # (nP, nT)
    data_plot = np.where(frac_row > 0, data, np.nan)

    pc = ax.pcolormesh(
        T_edges, logP_edges, data_plot,
        cmap=CMAP, norm=NORM, shading="flat", rasterized=True,
    )

    ax.set_xlim(T_C[0], T_C[-1])
    ax.set_ylim(logP[0], logP[-1])

    # T-axis ticks; suppress "350" on all but last panel
    ax.set_xticks(T_TICKS)
    labels = [str(t) for t in T_TICKS]
    if ci < NC - 1:
        labels[-1] = ""
    ax.set_xticklabels(labels, fontsize=6.5)

    # P-axis: leftmost column only
    if ci == 0:
        ax.set_yticks(LOGP_TICKS)
        ax.set_yticklabels(P_LABELS, fontsize=6.5)
    else:
        ax.set_yticks([])

    ax.tick_params(axis="both", length=2, pad=1.5)

# ── axis titles ────────────────────────────────────────────────────────────────
fig.text(0.505, 0.01, r"$T$ ($^\circ$C)", ha="center", va="bottom", fontsize=8)
axes[0].set_ylabel(r"$P$ (bar)", fontsize=8, labelpad=2)

# ── column headers ─────────────────────────────────────────────────────────────
for ci, lbl in enumerate(COL_LABELS):
    ax = axes[ci]
    x_fig = ax.get_position().x0 + ax.get_position().width / 2
    fig.text(x_fig, 0.92, lbl, ha="center", va="bottom", fontsize=7.5,
             transform=fig.transFigure)

# ── colorbar ──────────────────────────────────────────────────────────────────
cax = fig.add_axes([0.895, 0.16, 0.018, 0.72])
cb  = fig.colorbar(pc, cax=cax, norm=NORM, cmap=CMAP, extend="max")
cb.set_label("Median SSI iterations", fontsize=8, labelpad=4)
cb.set_ticks([1, 5, 10, 15, 20])
cb.ax.tick_params(labelsize=6.5)

# ── save ───────────────────────────────────────────────────────────────────────
outname = "ecpa_ssi_heatmap"
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT_DIR}/{outname}.{ext}")
plt.close(fig)
print(f"  -> {OUT_DIR}/{outname}.png/.pdf")
print("Done.")
