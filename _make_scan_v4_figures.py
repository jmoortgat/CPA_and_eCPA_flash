"""Generate scan_v4 figures to replace scan_v3 figures in the paper.

Produces (in figures/scan_v4/):
  ecpa_phase_map.pdf       — binary two-phase/single-phase map, 6 ms columns
  ecpa_composition_aq_grid.pdf — CO2 mol frac in aq phase, 1 z × 6 ms
  ecpa_composition_c_grid.pdf  — H2O mol frac in CO2-rich, 1 z × 6 ms
  ecpa_timing_heatmap.pdf  — mean wall time per call as proxy for solver effort
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm, BoundaryNorm
import pandas as pd
from pathlib import Path
import os

os.makedirs('figures/scan_v4', exist_ok=True)

# ── Load NPZ and metrics ──────────────────────────────────────────────────────
data    = np.load('results/scan_v4_table.npz')
T_grid  = data['T_grid']           # (361,) K
P_grid  = data['P_grid']           # (100,) bar
ms_grid = data['ms_grid']          # (14,)
x4w     = data['x4w']              # CO2 mol frac in aq, shape (361,100,14)
x1c     = data['x1c']              # H2O mol frac in CO2-rich, shape (361,100,14)
is_2ph  = data['is_two_phase']     # bool, shape (361,100,14)

metrics = pd.read_parquet('results/scan_v4_metrics.parquet')

T_C = T_grid - 273.15   # Celsius

# 6 ms columns for all figures (similar to scan_v3)
MS_COLS = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0]
ms_idx  = [np.argmin(np.abs(ms_grid - m)) for m in MS_COLS]
ms_labels = [f'$m_s = {MS_COLS[i]:.1f}$' for i in range(len(MS_COLS))]

# Helper: detect phase envelope for one (T_C, P_grid) grid and ms index
def phase_envelope(iph_col):
    """Return dew-point (lower P boundary) and bubble-point (upper P boundary) curves."""
    dew_T, dew_P, bub_T, bub_P = [], [], [], []
    nT, nP = iph_col.shape
    for iT in range(nT):
        row = iph_col[iT, :]
        # dew: first P from bottom where is_2ph = True
        idx = np.where(row)[0]
        if len(idx) > 0:
            dew_T.append(T_C[iT]); dew_P.append(P_grid[idx[0]])
            bub_T.append(T_C[iT]); bub_P.append(P_grid[idx[-1]])
    return (np.array(dew_T), np.array(dew_P),
            np.array(bub_T), np.array(bub_P))


# ── 1. PHASE MAP ──────────────────────────────────────────────────────────────
print('Generating phase map...', flush=True)
fig, axes = plt.subplots(1, 6, figsize=(14, 4), sharey=True)
fig.suptitle(r'Phase state at representative feed $z = 0.5$', fontsize=11)

cmap_ph = mcolors.ListedColormap(['#2a6eba', '#f4a300'])

for col, (ms_i, label) in enumerate(zip(ms_idx, ms_labels)):
    ax = axes[col]
    iph = is_2ph[:, :, ms_i].astype(float)   # (nT, nP)
    ax.pcolormesh(T_C, np.log10(P_grid), iph.T,
                  cmap=cmap_ph, vmin=0, vmax=1, shading='nearest', rasterized=True)
    ax.set_xlabel(r'$T$ (°C)', fontsize=9)
    ax.set_title(label, fontsize=9)
    if col == 0:
        ax.set_ylabel(r'$\log_{10}P$ (bar)', fontsize=9)
        ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.301])
        ax.set_yticklabels(['1', '3', '10', '30', '100', '300', '1000', '2000'], fontsize=7)
    ax.set_xlim(T_C[0], T_C[-1])

# legend patches
from matplotlib.patches import Patch
handles = [Patch(facecolor='#f4a300', label='two-phase'),
           Patch(facecolor='#2a6eba', label='single-phase')]
fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=9,
           bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig('figures/scan_v4/ecpa_phase_map.pdf', bbox_inches='tight', dpi=150)
plt.close()
print('  → ecpa_phase_map.pdf', flush=True)


# ── 2. AQUEOUS COMPOSITION GRID ───────────────────────────────────────────────
print('Generating aq composition grid...', flush=True)

# x_CO2 in aq = x4w; mask single-phase as NaN
x_co2_aq = np.where(is_2ph, x4w, np.nan)          # mol fraction
x_co2_aq_pct = x_co2_aq * 100.0                   # mol%

fig, axes = plt.subplots(1, 6, figsize=(14, 4), sharey=True)
fig.suptitle(r'Equilibrium CO$_2$ mol\% in aqueous phase ($z = 0.5$)', fontsize=11)

vmin, vmax = 0.05, 25.0   # mol%
norm_aq = LogNorm(vmin=vmin, vmax=vmax)
cmap_aq = plt.cm.plasma_r

for col, (ms_i, label) in enumerate(zip(ms_idx, ms_labels)):
    ax = axes[col]
    arr = x_co2_aq_pct[:, :, ms_i]   # (nT, nP)
    im = ax.pcolormesh(T_C, np.log10(P_grid), arr.T,
                       norm=norm_aq, cmap=cmap_aq, shading='nearest', rasterized=True)
    # Phase envelope
    dT, dP, bT, bP = phase_envelope(is_2ph[:, :, ms_i])
    if len(dT):
        ax.plot(dT, np.log10(dP), 'r-',  lw=1.0, label='dew')
        ax.plot(bT, np.log10(bP), 'r--', lw=1.0, label='bubble')
    ax.set_xlabel(r'$T$ (°C)', fontsize=9)
    ax.set_title(label, fontsize=9)
    if col == 0:
        ax.set_ylabel(r'$\log_{10}P$ (bar)', fontsize=9)
        ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.301])
        ax.set_yticklabels(['1', '3', '10', '30', '100', '300', '1000', '2000'], fontsize=7)
    ax.set_xlim(T_C[0], T_C[-1])

cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm_aq, cmap=cmap_aq), cax=cbar_ax)
cb.set_label(r'$x_{\mathrm{CO_2}}$ (mol\%)', fontsize=9)
fig.tight_layout(rect=[0, 0, 0.91, 1])
fig.savefig('figures/scan_v4/ecpa_composition_aq_grid.pdf', bbox_inches='tight', dpi=150)
plt.close()
print('  → ecpa_composition_aq_grid.pdf', flush=True)


# ── 3. CO2-RICH COMPOSITION GRID ──────────────────────────────────────────────
print('Generating CO2-rich composition grid...', flush=True)

y_h2o_c = np.where(is_2ph, x1c, np.nan)       # mol fraction
y_h2o_pct = y_h2o_c * 100.0                   # mol%

fig, axes = plt.subplots(1, 6, figsize=(14, 4), sharey=True)
fig.suptitle(r'Equilibrium H$_2$O mol\% in CO$_2$-rich phase ($z = 0.5$)', fontsize=11)

vmin2, vmax2 = 0.01, 70.0
norm_c = LogNorm(vmin=vmin2, vmax=vmax2)
cmap_c = plt.cm.inferno_r

for col, (ms_i, label) in enumerate(zip(ms_idx, ms_labels)):
    ax = axes[col]
    arr = y_h2o_pct[:, :, ms_i]
    im = ax.pcolormesh(T_C, np.log10(P_grid), arr.T,
                       norm=norm_c, cmap=cmap_c, shading='nearest', rasterized=True)
    dT, dP, bT, bP = phase_envelope(is_2ph[:, :, ms_i])
    if len(dT):
        ax.plot(dT, np.log10(dP), 'r-',  lw=1.0)
        ax.plot(bT, np.log10(bP), 'r--', lw=1.0)
    ax.set_xlabel(r'$T$ (°C)', fontsize=9)
    ax.set_title(label, fontsize=9)
    if col == 0:
        ax.set_ylabel(r'$\log_{10}P$ (bar)', fontsize=9)
        ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.301])
        ax.set_yticklabels(['1', '3', '10', '30', '100', '300', '1000', '2000'], fontsize=7)
    ax.set_xlim(T_C[0], T_C[-1])

cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm_c, cmap=cmap_c), cax=cbar_ax)
cb.set_label(r'$y_{\mathrm{H_2O}}$ (mol\%)', fontsize=9)
fig.tight_layout(rect=[0, 0, 0.91, 1])
fig.savefig('figures/scan_v4/ecpa_composition_c_grid.pdf', bbox_inches='tight', dpi=150)
plt.close()
print('  → ecpa_composition_c_grid.pdf', flush=True)


# ── 4. TIMING HEATMAP (wall time per call as proxy for solver effort) ─────────
print('Generating timing heatmap...', flush=True)

# Pivot metrics to (T, P, ms) array of mean wall time (two-phase only)
# Build lookup
ms_vals_sorted = sorted(metrics['ms_feed'].unique())
T_vals_sorted  = sorted(metrics['T'].unique())
P_vals_sorted  = sorted(metrics['P'].unique())

ms_sel = []
for m in MS_COLS:
    best = min(ms_vals_sorted, key=lambda x: abs(x-m))
    ms_sel.append(best)

fig, axes = plt.subplots(1, 6, figsize=(14, 4), sharey=True)
fig.suptitle(r'Mean wall time per flash call (ms), two-phase points only', fontsize=11)

norm_t = LogNorm(vmin=1, vmax=200)
cmap_t = plt.cm.YlOrRd

for col, (ms_v, label) in enumerate(zip(ms_sel, ms_labels)):
    ax = axes[col]
    sub = metrics[(metrics['ms_feed'] == ms_v) & metrics['is_two_phase']].copy()
    # pivot
    piv = sub.pivot_table(index='T', columns='P', values='wall_time_ms', aggfunc='mean')
    T_arr = np.array(sorted(piv.index)) - 273.15
    P_arr = np.array(sorted(piv.columns))
    W = np.full((len(T_arr), len(P_arr)), np.nan)
    for i, T in enumerate(sorted(piv.index)):
        for j, P in enumerate(sorted(piv.columns)):
            try: W[i, j] = piv.loc[T, P]
            except: pass
    im = ax.pcolormesh(T_arr, np.log10(P_arr), W.T,
                       norm=norm_t, cmap=cmap_t, shading='nearest', rasterized=True)
    ax.set_xlabel(r'$T$ (°C)', fontsize=9)
    ax.set_title(label, fontsize=9)
    if col == 0:
        ax.set_ylabel(r'$\log_{10}P$ (bar)', fontsize=9)
        ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.301])
        ax.set_yticklabels(['1', '3', '10', '30', '100', '300', '1000', '2000'], fontsize=7)

cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm_t, cmap=cmap_t), cax=cbar_ax)
cb.set_label('Wall time (ms)', fontsize=9)
fig.tight_layout(rect=[0, 0, 0.91, 1])
fig.savefig('figures/scan_v4/ecpa_timing_heatmap.pdf', bbox_inches='tight', dpi=150)
plt.close()
print('  → ecpa_timing_heatmap.pdf', flush=True)

print('All scan_v4 figures done.')
