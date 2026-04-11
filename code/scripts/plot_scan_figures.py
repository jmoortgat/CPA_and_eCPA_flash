"""Generate scan_v4 figures to replace scan_v3 figures in the paper.

Produces (in figures/scan_v4/):
  ecpa_phase_map.pdf           — binary two-phase/single-phase map
  ecpa_composition_aq_grid.pdf — CO2 mol frac in aq phase
  ecpa_composition_c_grid.pdf  — H2O mol frac in CO2-rich
  ecpa_timing_heatmap.pdf      — mean wall time per call as proxy for solver effort

Columns shown: ms = 1e-5, 1.0, 2.0, 6.0  (ms=0, 0.5, 4.0 removed per reviewer)
All fonts are bold throughout.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
import pandas as pd
from pathlib import Path
import os

os.makedirs('figures/scan_v4', exist_ok=True)

# ── Global bold font settings ─────────────────────────────────────────────────
plt.rcParams.update({
    'font.weight':        'bold',
    'axes.labelweight':   'bold',
    'axes.titleweight':   'bold',
    'axes.labelsize':     11,
    'axes.titlesize':     11,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
})

T_MAX = 300.0   # °C — clip horizontal axis here

# ── Load NPZ and metrics ──────────────────────────────────────────────────────
data    = np.load('results/scan_v4_table.npz')
T_grid  = data['T_grid']           # (361,) K
P_grid  = data['P_grid']           # (100,) bar
ms_grid = data['ms_grid']          # (14,)
x4w     = data['x4w'].copy()       # CO2 mol frac in aq,      shape (361,100,14)
x1c     = data['x1c'].copy()       # H2O mol frac in CO2-rich, shape (361,100,14)
x4c     = data['x4c']              # CO2 mol frac in CO2-rich  (for swap fix)
x1w     = data['x1w']              # H2O mol frac in aq        (for swap fix)
is_2ph  = data['is_two_phase']     # bool, shape (361,100,14)

metrics = pd.read_parquet('results/scan_v4_metrics.parquet')

T_C = T_grid - 273.15   # Celsius

# ── Fix ms=0 (CPA) phase-labeling bug ────────────────────────────────────────
_swap = (x4w[:, :, 0] > 0.5) & is_2ph[:, :, 0]
x4w[:, :, 0][_swap] = x4c[:, :, 0][_swap]
x1c[:, :, 0][_swap] = x1w[:, :, 0][_swap]
print(f'ms=0 phase-swap fix: corrected {_swap.sum()} cells '
      f'({100*_swap.sum()/is_2ph[:,:,0].sum():.1f}% of two-phase)')

# ── Column selection ──────────────────────────────────────────────────────────
MS_COLS   = [1e-5, 1.0, 2.0, 6.0]
ms_idx    = [np.argmin(np.abs(ms_grid - m)) for m in MS_COLS]
ms_labels = [
    r'$m_s\!=\!10^{-5}$',
    r'$m_s\!=\!1.0$',
    r'$m_s\!=\!2.0$',
    r'$m_s\!=\!6.0$',
]
N_COLS = len(MS_COLS)

# ── Helper: make all tick labels bold on an axis ──────────────────────────────
def bold_ticks(ax):
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight('bold')

# ── Phase-envelope helper ─────────────────────────────────────────────────────
def phase_envelope(iph_col):
    dew_T, dew_P, bub_T, bub_P = [], [], [], []
    for iT in range(len(T_C)):
        idx = np.where(iph_col[iT, :])[0]
        if len(idx):
            dew_T.append(T_C[iT]); dew_P.append(P_grid[idx[0]])
            bub_T.append(T_C[iT]); bub_P.append(P_grid[idx[-1]])
    return (np.array(dew_T), np.array(dew_P),
            np.array(bub_T), np.array(bub_P))

YTICKS     = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.301]
YTICKLABS  = ['1', '3', '10', '30', '100', '300', '1000', '2000']


# ── 1. PHASE MAP ──────────────────────────────────────────────────────────────
print('Generating phase map...', flush=True)
fig, axes = plt.subplots(1, N_COLS, figsize=(10, 4), sharey=True)
cmap_ph = mcolors.ListedColormap(['#2a6eba', '#f4a300'])

for col, (ms_i, label) in enumerate(zip(ms_idx, ms_labels)):
    ax = axes[col]
    iph = is_2ph[:, :, ms_i].astype(float)
    ax.pcolormesh(T_C, np.log10(P_grid), iph.T,
                  cmap=cmap_ph, vmin=0, vmax=1, shading='nearest', rasterized=True)
    ax.set_xlabel(r'$T$ (°C)')
    ax.set_title(label)
    ax.set_xlim(T_C[0], T_MAX)
    if col == 0:
        ax.set_ylabel(r'$\log_{10}P$ (bar)')
        ax.set_yticks(YTICKS)
        ax.set_yticklabels(YTICKLABS)
    bold_ticks(ax)

from matplotlib.patches import Patch
handles = [Patch(facecolor='#f4a300', label='two-phase'),
           Patch(facecolor='#2a6eba', label='single-phase')]
leg = fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=10,
                 bbox_to_anchor=(0.5, -0.04))
for text in leg.get_texts():
    text.set_fontweight('bold')
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig('figures/scan_v4/ecpa_phase_map.pdf', bbox_inches='tight', dpi=150)
plt.close()
print('  → ecpa_phase_map.pdf', flush=True)


# ── 2. AQUEOUS COMPOSITION GRID ───────────────────────────────────────────────
print('Generating aq composition grid...', flush=True)

x_co2_aq_pct = np.where(is_2ph, np.clip(x4w, 1e-7, None), np.nan) * 100.0

fig, axes = plt.subplots(1, N_COLS, figsize=(10, 4), sharey=True)

vmin, vmax = 0.05, 25.0
norm_aq = LogNorm(vmin=vmin, vmax=vmax)
cmap_aq = plt.cm.plasma_r.copy()
cmap_aq.set_bad('lightgray')

for col, (ms_i, label) in enumerate(zip(ms_idx, ms_labels)):
    ax = axes[col]
    arr = x_co2_aq_pct[:, :, ms_i]
    ax.pcolormesh(T_C, np.log10(P_grid), arr.T,
                  norm=norm_aq, cmap=cmap_aq, shading='nearest', rasterized=True)

    dT, dP, bT, bP = phase_envelope(is_2ph[:, :, ms_i])

    if len(dT):
        m = dT <= T_MAX
        if m.any():
            ax.plot(dT[m], np.log10(dP[m]), 'r-', lw=1.2)

    if len(bT):
        m = (bT <= T_MAX) & (bP < P_grid[-1] * 0.99)
        if m.any():
            ax.plot(bT[m], np.log10(bP[m]), 'r--', lw=1.2)

    ax.text(0.76, 0.08, 'single-\nphase', transform=ax.transAxes,
            fontsize=9, fontweight='bold', ha='center', va='bottom', color='#555555',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.6))
    ax.set_xlabel(r'$T$ (°C)')
    ax.set_title(label)
    ax.set_xlim(T_C[0], T_MAX)
    if col == 0:
        ax.set_ylabel(r'$\log_{10}P$ (bar)')
        ax.set_yticks(YTICKS)
        ax.set_yticklabels(YTICKLABS)
    bold_ticks(ax)

cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm_aq, cmap=cmap_aq), cax=cbar_ax)
cb.set_label(r'$x_{\mathrm{CO_2}}$ (mol%)', fontsize=11, fontweight='bold')
for tick in cb.ax.get_yticklabels():
    tick.set_fontweight('bold')
fig.tight_layout(rect=[0, 0, 0.91, 1])
fig.savefig('figures/scan_v4/ecpa_composition_aq_grid.pdf', bbox_inches='tight', dpi=150)
plt.close()
print('  → ecpa_composition_aq_grid.pdf', flush=True)


# ── 3. CO2-RICH COMPOSITION GRID ──────────────────────────────────────────────
print('Generating CO2-rich composition grid...', flush=True)

y_h2o_c_pct = np.where(is_2ph, x1c, np.nan) * 100.0

fig, axes = plt.subplots(1, N_COLS, figsize=(10, 4), sharey=True)

vmin2, vmax2 = 0.01, 70.0
norm_c = LogNorm(vmin=vmin2, vmax=vmax2)
cmap_c = plt.cm.inferno_r.copy()
cmap_c.set_bad('lightgray')

for col, (ms_i, label) in enumerate(zip(ms_idx, ms_labels)):
    ax = axes[col]
    arr = y_h2o_c_pct[:, :, ms_i]
    ax.pcolormesh(T_C, np.log10(P_grid), arr.T,
                  norm=norm_c, cmap=cmap_c, shading='nearest', rasterized=True)

    dT, dP, bT, bP = phase_envelope(is_2ph[:, :, ms_i])

    if len(dT):
        m = dT <= T_MAX
        if m.any():
            ax.plot(dT[m], np.log10(dP[m]), 'r-', lw=1.2)

    if len(bT):
        m = (bT <= T_MAX) & (bP < P_grid[-1] * 0.99)
        if m.any():
            ax.plot(bT[m], np.log10(bP[m]), 'r--', lw=1.2)

    ax.text(0.76, 0.08, 'single-\nphase', transform=ax.transAxes,
            fontsize=9, fontweight='bold', ha='center', va='bottom', color='#555555',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.6))
    ax.set_xlabel(r'$T$ (°C)')
    ax.set_title(label)
    ax.set_xlim(T_C[0], T_MAX)
    if col == 0:
        ax.set_ylabel(r'$\log_{10}P$ (bar)')
        ax.set_yticks(YTICKS)
        ax.set_yticklabels(YTICKLABS)
    bold_ticks(ax)

cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm_c, cmap=cmap_c), cax=cbar_ax)
cb.set_label(r'$y_{\mathrm{H_2O}}$ (mol%)', fontsize=11, fontweight='bold')
for tick in cb.ax.get_yticklabels():
    tick.set_fontweight('bold')
fig.tight_layout(rect=[0, 0, 0.91, 1])
fig.savefig('figures/scan_v4/ecpa_composition_c_grid.pdf', bbox_inches='tight', dpi=150)
plt.close()
print('  → ecpa_composition_c_grid.pdf', flush=True)


# ── 4. TIMING HEATMAP ────────────────────────────────────────────────────────
print('Generating timing heatmap...', flush=True)

ms_vals_sorted = sorted(metrics['ms_feed'].unique())
ms_sel_T = [min(ms_vals_sorted, key=lambda x: abs(x - m)) for m in MS_COLS]

fig, axes = plt.subplots(1, N_COLS, figsize=(10, 4), sharey=True)

norm_t = LogNorm(vmin=1, vmax=200)
cmap_t = plt.cm.YlOrRd.copy()
cmap_t.set_bad('lightgray')

for col, (ms_v, label, ms_i) in enumerate(zip(ms_sel_T, ms_labels, ms_idx)):
    ax = axes[col]

    sub = metrics[metrics['ms_feed'] == ms_v].copy()
    piv = sub.pivot_table(index='T', columns='P', values='wall_time_ms', aggfunc='mean')
    T_keys = np.array(sorted(piv.index))
    P_keys = np.array(sorted(piv.columns))

    iT_map = np.array([np.argmin(np.abs(T_grid - t)) for t in T_keys])
    iP_map = np.array([np.argmin(np.abs(P_grid - p)) for p in P_keys])

    W = np.full((len(T_grid), len(P_grid)), np.nan)
    for ii, iT_w in enumerate(iT_map):
        for jj, iP_w in enumerate(iP_map):
            val = piv.iloc[ii, jj]
            if np.isfinite(val):
                W[iT_w, iP_w] = val

    W[~is_2ph[:, :, ms_i]] = np.nan

    ax.pcolormesh(T_C, np.log10(P_grid), W.T,
                  norm=norm_t, cmap=cmap_t, shading='nearest', rasterized=True)

    ax.text(0.76, 0.08, 'single-\nphase', transform=ax.transAxes,
            fontsize=9, fontweight='bold', ha='center', va='bottom', color='#555555',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.6))

    ax.set_xlabel(r'$T$ (°C)')
    ax.set_title(label)
    ax.set_xlim(T_C[0], T_MAX)
    if col == 0:
        ax.set_ylabel(r'$\log_{10}P$ (bar)')
        ax.set_yticks(YTICKS)
        ax.set_yticklabels(YTICKLABS)
    bold_ticks(ax)

cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm_t, cmap=cmap_t), cax=cbar_ax)
cb.set_label('Wall time (ms)', fontsize=11, fontweight='bold')
for tick in cb.ax.get_yticklabels():
    tick.set_fontweight('bold')
fig.tight_layout(rect=[0, 0, 0.91, 1])
fig.savefig('figures/scan_v4/ecpa_timing_heatmap.pdf', bbox_inches='tight', dpi=150)
plt.close()
print('  → ecpa_timing_heatmap.pdf', flush=True)

print('All scan_v4 figures done.')
