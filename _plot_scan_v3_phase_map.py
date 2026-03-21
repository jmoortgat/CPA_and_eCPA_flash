"""
Single-row two-phase fraction map: fraction of CO2 feed compositions z in [0.05, 0.90]
that are two-phase at each (T, P), for 6 NaCl molalities.

Matches the layout style of the other scan_v3 grid figures.

Output: figures/scan_v3/ecpa_phase_map.png/.pdf
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
npz     = np.load("results/scan_v3_table.npz")
T_grid  = npz["T_grid"]          # (70,) K
P_grid  = npz["P_grid"]          # (50,)
ms_grid = npz["ms_grid"]         # (14,)
is_two  = npz["is_two_phase"]    # (70, 50, 25, 14)

# fraction of z values that are two-phase at each (T, P, ms)
frac_2ph = is_two.mean(axis=2)   # (70, 50, 14)

# ── panel indices ──────────────────────────────────────────────────────────────
# Include ms=0 (CPA, salt-free baseline) since we're showing phase fraction,
# not composition (no phase-label swap issue for fraction data)
IMS_COLS = [0, 2, 4, 6, 10, 13]   # ms = 0, 0.1, 0.5, 1, 3, 6
NC = len(IMS_COLS)

# ── colormap / norm ────────────────────────────────────────────────────────────
CMAP       = cmc.batlow
VMIN, VMAX = 0.0, 1.0
NORM       = mcolors.Normalize(vmin=VMIN, vmax=VMAX)

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

# ── column header labels ───────────────────────────────────────────────────────
def _ms_col_label(ms_val):
    if ms_val < 1e-6:
        return r"$m_s = 0$"
    if ms_val < 1e-3:
        return r"$m_s \approx 0$"
    return rf"$m_s = {ms_val:g}$"

COL_LABELS = [_ms_col_label(ms_grid[i]) for i in IMS_COLS]

# ── figure ─────────────────────────────────────────────────────────────────────
FIG_W = 7.2
FIG_H = 2.2

fig, axes = plt.subplots(
    1, NC,
    figsize=(FIG_W, FIG_H),
    gridspec_kw=dict(wspace=0.04,
                     left=0.13, right=0.88,
                     top=0.88, bottom=0.16),
)

for ci, ims in enumerate(IMS_COLS):
    ax = axes[ci]

    # fraction: 0 = always single-phase, 1 = always two-phase
    frac = frac_2ph[:, :, ims].T    # (nP, nT)

    pc = ax.pcolormesh(
        T_edges, logP_edges, frac,
        cmap=CMAP, norm=NORM, shading="flat", rasterized=True,
    )

    ax.set_xlim(T_C[0], T_C[-1])
    ax.set_ylim(logP[0], logP[-1])

    ax.set_xticks(T_TICKS)
    labels = [str(t) for t in T_TICKS]
    if ci < NC - 1:
        labels[-1] = ""
    ax.set_xticklabels(labels, fontsize=6.5)

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
cb  = fig.colorbar(pc, cax=cax, norm=NORM, cmap=CMAP)
cb.set_label("Two-phase fraction", fontsize=8, labelpad=4)
cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cb.set_ticklabels(["0", "0.25", "0.5", "0.75", "1"], fontsize=6.5)

# ── save ───────────────────────────────────────────────────────────────────────
outname = "ecpa_phase_map"
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT_DIR}/{outname}.{ext}")
plt.close(fig)
print(f"  -> {OUT_DIR}/{outname}.png/.pdf")
print("Done.")
