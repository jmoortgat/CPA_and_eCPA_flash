"""Regenerate figures/scan/ from saved scan_newton_results.npz without recomputing."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    'font.weight':       'bold',
    'axes.labelweight':  'bold',
    'axes.titleweight':  'bold',
    'axes.labelsize':    11,
    'axes.titlesize':    11,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
})

def _bold_ticks(ax):
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight('bold')

SCAN_FILE   = "results/scan_results_extended.npz"
NEWTON_FILE = "results/scan_newton_results.npz"
FIGDIR      = "figures/scan"
os.makedirs(FIGDIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
d  = np.load(SCAN_FILE,   allow_pickle=True)
dn = np.load(NEWTON_FILE, allow_pickle=True)

T_grid = d["T_grid"]
P_grid = d["P_grid"]
nT, nP = len(T_grid), len(P_grid)
T_C  = T_grid - 273.15          # Celsius for axes
logP = np.log10(P_grid)         # log10(P) for axes — matches scan_v4 style

_Ptick_bar = [1, 3, 10, 30, 100, 300, 1000, 1500]
_Ptick_pos = [np.log10(p) for p in _Ptick_bar]
_Ptick_lbl = [str(p) for p in _Ptick_bar]
shape  = (nT, nP, len(d["z_grid"]))

two_ph        = (d["phase_id"] == 4)                   # genuine two-phase points (9825)
any_two_ph    = two_ph.any(axis=-1)                    # (nT, nP) — envelope over all z
flash_conv_old = d["flash_conv"]
flash_iter_old = d["flash_iter"]

conv_new = dn["conv"]
iter_tot = dn["iter_tot"]
iter_ssi = dn["iter_ssi"]
iter_nwt = dn["iter_nwt"]
t_ms_new = dn["t_ms"]
tbl_avail = dn["tbl_avail"]
STRATS   = list(dn["strat_names"])

n_2ph = two_ph.sum()


def tp_avg(arr3d, mask3d):
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


# ── Figure A: newton_heatmap — 3 panels each with their own colorbar ───────────
conv_mask = conv_new[..., 1] & two_ph
ssi_map  = iter_ssi[..., 1].astype(float); ssi_map[~conv_mask]  = np.nan
nwt_map  = iter_nwt[..., 1].astype(float); nwt_map[~conv_mask]  = np.nan

ssi_avg   = tp_avg(ssi_map, two_ph)
nwt_avg   = tp_avg(nwt_map, two_ph)

t_ratio   = np.full(shape, np.nan)
both3d    = conv_new[..., 0] & conv_new[..., 1]
t_ratio[both3d] = (t_ms_new[..., 0][both3d] /
                   np.maximum(t_ms_new[..., 1][both3d], 1e-9))
ratio_avg = tp_avg(t_ratio, two_ph)

panels = [
    (ssi_avg,   "SSI iterations\n(table + SSI + Newton)",   "viridis",  0,   20,  "Mean SSI iterations"),
    (nwt_avg,   "Newton iterations\n(table + SSI + Newton)", "plasma",   0,    5,  "Mean Newton iterations"),
    (ratio_avg, "Wall-time ratio\n(SSI-only / SSI+Newton)",  "RdYlGn",  0.5,  2.0, "Speed-up ratio"),
]

# 6-column gridspec: data | cbar | gap | data | cbar | gap | data | cbar
# width_ratios: [data, cbar, gap, data, cbar, gap, data, cbar]
# Layout: data | cbar | spacer | data | cbar | spacer | data | cbar
# Small wspace keeps panel tight to its colorbar; spacer columns provide
# the larger visual gap between each colorbar and the next data panel.
fig = plt.figure(figsize=(15, 4.5))
gs  = gridspec.GridSpec(
    1, 8,
    width_ratios=[1, 0.045, 0.30, 1, 0.045, 0.30, 1, 0.045],
    wspace=0.05,
    left=0.06, right=0.98, bottom=0.12, top=0.90,
)

data_cols = [0, 3, 6]
cbar_cols = [1, 4, 7]

for spacer_col in [2, 5]:
    fig.add_subplot(gs[0, spacer_col]).set_visible(False)

for col_d, col_c, (data, title, cmap, vmin, vmax, cblabel) in zip(data_cols, cbar_cols, panels):
    ax  = fig.add_subplot(gs[0, col_d])
    cax = fig.add_subplot(gs[0, col_c])

    im = ax.pcolormesh(T_C, logP, data.T, cmap=cmap,
                       vmin=vmin, vmax=vmax, shading="nearest")
    # Single-phase shading and phase boundary
    ax.contourf(T_C, logP, any_two_ph.T,
                levels=[-0.5, 0.5], colors=['0.82'], alpha=0.75, zorder=2)
    ax.contour(T_C, logP, any_two_ph.T,
               levels=[0.5], colors=['k'], linewidths=1.2, zorder=3)
    ax.set_xlabel(r"$T$ (°C)", fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlim(T_C[0], T_C[-1])
    ax.set_ylim(logP[0], logP[-1])
    ax.set_yticks(_Ptick_pos)
    if col_d == 0:
        ax.set_ylabel("Pressure (bar)", fontsize=11, fontweight='bold')
        ax.set_yticklabels(_Ptick_lbl, fontsize=9)
    else:
        ax.tick_params(labelleft=False)
    _bold_ticks(ax)

    cb = fig.colorbar(im, cax=cax, label=cblabel)
    cb.set_label(cblabel, fontsize=11, fontweight='bold')
    for t in cax.get_yticklabels():
        t.set_fontweight('bold')
    cax.tick_params(labelsize=9)

fig.savefig(f"{FIGDIR}/newton_heatmap.pdf", bbox_inches="tight", dpi=150)
fig.savefig(f"{FIGDIR}/newton_heatmap.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"Saved {FIGDIR}/newton_heatmap.png")


# ── Figure B: strategy_comparison_bar ─────────────────────────────────────────
old_names    = list(d["flash_strategy_names"])
all_names    = old_names + ["tbl+SSI", "tbl+SSI+Newton"]
n_all_strat  = len(all_names)

conv_rates_all = []
mean_iters_all = []

for s in range(4):
    c  = flash_conv_old[..., s][two_ph]
    it = flash_iter_old[..., s][two_ph]
    nc = c.sum()
    conv_rates_all.append(100 * nc / n_2ph)
    mean_iters_all.append(it[c].mean() if nc else np.nan)

for s in range(2):
    c  = conv_new[..., s][two_ph]
    it = iter_tot[..., s][two_ph]
    nc = c.sum()
    conv_rates_all.append(100 * nc / n_2ph)
    mean_iters_all.append(it[c].mean() if nc else np.nan)

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

ax = axes[0]
bars = ax.bar(x, conv_rates_all, color=colors, edgecolor="white", linewidth=0.5)
for bar, cr in zip(bars, conv_rates_all):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
            f"{cr:.2f}%", ha="center", va="bottom", fontsize=9, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(labels_short, fontsize=9, fontweight='bold')
ax.set_ylabel("Convergence rate (%)", fontsize=11, fontweight='bold')
ax.set_title("(a) Convergence rate (9825 two-phase points)", fontsize=11, fontweight='bold')
ax.set_ylim(min(conv_rates_all) - 1, 101)
_bold_ticks(axes[0])

ax = axes[1]
bars = ax.bar(x, mean_iters_all, color=colors, edgecolor="white", linewidth=0.5)
for bar, mi in zip(bars, mean_iters_all):
    if np.isfinite(mi):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{mi:.1f}", ha="center", va="bottom", fontsize=9, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(labels_short, fontsize=9, fontweight='bold')
ax.set_ylabel("Mean iterations (converged points)", fontsize=11, fontweight='bold')
ax.set_title("(b) Mean total iterations", fontsize=11, fontweight='bold')
_bold_ticks(axes[1])

fig.tight_layout()
fig.savefig(f"{FIGDIR}/strategy_comparison_bar.pdf", bbox_inches="tight", dpi=150)
fig.savefig(f"{FIGDIR}/strategy_comparison_bar.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"Saved {FIGDIR}/strategy_comparison_bar.png")

print("Done.")
