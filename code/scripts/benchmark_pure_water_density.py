"""
benchmark_pure_water_density.py
================================
Compare eCPA pure-water (m_s → 0) density predictions against the IAPWS-95
reference equation over the full (T, P) range explored in the paper.

IAPWS-95 (International Association for the Properties of Water and Steam,
Revised Release 2016) is the international standard for water properties,
fitted to a comprehensive body of experimental measurements and valid up to
1000 MPa (10,000 bar) and 1273 K.  It covers the full 1–2000 bar pressure
range of the paper.

Outputs
-------
- Console table: AARE by phase and by temperature
- figures/density/iapws_parity.png      — parity plot (eCPA vs IAPWS-95)
- figures/density/iapws_error_map.png   — signed error (%) over (T, P) plane
- figures/density/iapws_error_vs_T.png  — error vs T at fixed pressures
- results/pure_water_density_iapws.parquet
"""
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

os.makedirs('figures/density', exist_ok=True)
os.makedirs('results', exist_ok=True)

from iapws import IAPWS95
from ecpa.stability import _lnphi_aq_inner
from ecpa.constants import peneloux_h2o, Mw as M_H2O  # M_H2O in kg/mol

R_bar_cm3 = 83.14           # bar·cm³/(mol·K)
M_H2O_g   = M_H2O * 1000         # kg/mol → g/mol     (= 18.015 g/mol)


# ── Grid ─────────────────────────────────────────────────────────────────────
# Temperatures covering the full paper range, staying ≥ 273.15 K (IAPWS lower bound)
T_vals = [273.15, 278, 283, 288, 293, 298, 303, 308, 313, 318, 323, 333,
          348, 353, 363, 373, 393, 413, 423, 453, 473, 498, 523, 548, 573, 623]

# Pressures 1–2000 bar (full paper range)
P_vals = [1, 2, 5, 10, 25, 50, 75, 100, 150, 200, 300, 400,
          500, 600, 700, 800, 1000, 1200, 1500, 2000]

print(f"Grid: {len(T_vals)} T × {len(P_vals)} P = {len(T_vals)*len(P_vals)} conditions")
print(f"Reference: IAPWS-95 (valid to 1000 MPa / 10 000 bar, covers full paper range)")


# ── eCPA density for pure water ───────────────────────────────────────────────
def ecpa_pure_water_density(T: float, P_bar: float, x0=None):
    """
    eCPA aqueous-phase density [kg/m³] for pure water (x_{H2O}=1, m_s → 0).

    Uses _lnphi_aq_inner with x1w=1, ms=1e-4 (negligible salt keeps the
    3×3 Newton system well-conditioned while leaving DH/Born terms negligible).
    Density is computed from the converged Z_w plus the H₂O Péneloux shift.

    Returns (rho_kg_m3, sol_aq).
    """
    ms_eff = 1e-4
    try:
        _, _, sol = _lnphi_aq_inner(x1w=1.0, ms=ms_eff, T=T, P=P_bar, x0=x0)
        Zw = float(sol[0])
        if not (np.isfinite(Zw) and 0.0 < Zw < 20.0):
            return np.nan, None
        Vm      = Zw * R_bar_cm3 * T / P_bar          # cm³/mol (EoS volume)
        Vm_corr = Vm + peneloux_h2o(T) * 1e6          # Péneloux-corrected [cm³/mol]
        if Vm_corr <= 0:
            return np.nan, None
        rho = (M_H2O_g / Vm_corr) * 1000.0     # kg/m³
        return rho, np.asarray(sol, dtype=float)
    except Exception:
        return np.nan, None


# ── Main scan ─────────────────────────────────────────────────────────────────
print("Running scan …")
rows = []
x0_cache = {}   # warm-start per T (carry across pressures at same T)

for T in T_vals:
    x0 = x0_cache.get(T)
    for P in P_vals:
        # IAPWS-95 reference (P in MPa = bar / 10)
        try:
            w = IAPWS95(T=float(T), P=float(P) * 0.1)
            rho_ref   = float(w.rho)   if (w.rho   is not None and np.isfinite(w.rho))   else np.nan
            phase_ref = str(w.phase)   if hasattr(w, 'phase') else 'unknown'
        except Exception:
            rho_ref   = np.nan
            phase_ref = 'out_of_range'

        # eCPA prediction
        rho_ecpa, sol = ecpa_pure_water_density(T, P, x0=x0)
        if sol is not None:
            x0_cache[T] = sol

        rows.append(dict(
            T_K=T, P_bar=P, T_C=T - 273.15,
            rho_ref=rho_ref,
            rho_ecpa=rho_ecpa,
            phase_ref=phase_ref,
        ))

df = pd.DataFrame(rows)
df['err_pct'] = (df['rho_ecpa'] - df['rho_ref']) / df['rho_ref'] * 100.0
df['are_pct'] = df['err_pct'].abs()
df.to_parquet('results/pure_water_density_iapws.parquet', index=False)
print(f"  Saved {len(df)} rows → results/pure_water_density_iapws.parquet")


# ── Filter: liquid / compressed water only (eCPA is a liquid-phase EoS) ──────
# Vapor and near-critical steam are outside the scope of the aqueous-phase model.
LIQUID_PHASES = {'Liquid', 'Compressible liquid', 'liquid', 'compressible liquid'}
df['is_liquid'] = df['phase_ref'].apply(
    lambda p: any(lp.lower() in p.lower() for lp in LIQUID_PHASES))

# ── Summary statistics ────────────────────────────────────────────────────────
print("\n" + "="*68)
print("DENSITY ACCURACY  (eCPA vs IAPWS-95)")
print("="*68)

ok = df[df['is_liquid']].dropna(subset=['rho_ecpa', 'rho_ref'])
print(f"\nOverall  N={len(ok)}")
print(f"  AARE = {ok['are_pct'].mean():.2f}%")
print(f"  bias = {ok['err_pct'].mean():+.2f}%")
worst = ok.loc[ok['are_pct'].idxmax()]
print(f"  max ARE = {worst['are_pct']:.1f}%  "
      f"(T={worst['T_K']:.1f} K, P={worst['P_bar']:.0f} bar, "
      f"phase={worst['phase_ref']})")

print("\nBy phase (IAPWS-95 classification):")
for phase, g in ok.groupby('phase_ref'):
    print(f"  {phase:<22s}  N={len(g):4d}  "
          f"AARE={g['are_pct'].mean():6.2f}%  "
          f"bias={g['err_pct'].mean():+6.2f}%")

print("\nBy temperature:")
print(f"  {'T [K]':>8}  {'T [°C]':>6}  {'N':>4}  {'AARE':>7}  {'bias':>7}  phase(s)")
for T, g in ok.groupby('T_K'):
    phases = '/'.join(sorted(g['phase_ref'].unique()))
    print(f"  {T:8.2f}  {T-273.15:6.1f}  {len(g):4d}  "
          f"{g['are_pct'].mean():6.2f}%  {g['err_pct'].mean():+6.2f}%  {phases}")


# ── Figure 1: parity plot (liquid phase only vs all) ─────────────────────────
fig = plt.figure(figsize=(12, 5.4))
gs  = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05],
                        left=0.08, right=0.93, top=0.93, bottom=0.13,
                        wspace=0.38)
ax_left  = fig.add_subplot(gs[0, 0])
ax_right = fig.add_subplot(gs[0, 1])
cax      = fig.add_subplot(gs[0, 2])
axes = [ax_left, ax_right]

cmap = plt.cm.plasma
T_unique = sorted(ok['T_K'].unique())
norm = plt.Normalize(vmin=min(T_unique)-273.15, vmax=max(T_unique)-273.15)

P_max_all = int(ok['P_bar'].max())
for ax, mask, title_suffix in zip(
        axes,
        [ok['P_bar'] <= 300,
         pd.Series(True, index=ok.index)],
        ['$P$ ≤ 300 bar', f'All liquid conditions ($P$ up to {P_max_all} bar)']):
    sub = ok[mask]
    if sub.empty:
        ax.set_visible(False)
        continue
    for T in T_unique:
        g = sub[sub['T_K'] == T]
        if g.empty:
            continue
        ax.scatter(g['rho_ref'], g['rho_ecpa'],
                   color=cmap(norm(T-273.15)), s=20, zorder=4, alpha=0.85)
    lo = sub[['rho_ref','rho_ecpa']].min().min() - 10
    hi = sub[['rho_ref','rho_ecpa']].max().max() + 10
    lv = np.linspace(lo, hi, 300)
    ax.plot(lv, lv, 'k-', lw=1.0, zorder=3)
    ax.fill_between(lv, lv*0.99, lv*1.01, color='green',  alpha=0.15, label='±1%')
    ax.fill_between(lv, lv*0.98, lv*1.02, color='orange', alpha=0.12, label='±2%')
    aare_sub = sub['are_pct'].mean()
    ax.set_xlabel(r'IAPWS-95 $\rho$ [kg m$^{-3}$]', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'eCPA $\rho$ [kg m$^{-3}$]', fontsize=12, fontweight='bold')
    ax.set_title(f'{title_suffix}\nAARE = {aare_sub:.2f}%', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left', prop={'size': 9, 'weight': 'bold'})
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight('bold')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cax, label='$T$ (°C)')
cbar.ax.tick_params(labelsize=9)
cbar.set_label('$T$ (°C)', fontsize=11, fontweight='bold')
for tick in cbar.ax.get_yticklabels():
    tick.set_fontweight('bold')
fig.savefig('figures/density/iapws_parity.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("\nSaved figures/density/iapws_parity.png")


# ── Figure 2: signed error map over (T, P) plane ─────────────────────────────
T_arr = sorted(df['T_K'].unique())
P_arr = sorted(df['P_bar'].unique())
err_grid = np.full((len(P_arr), len(T_arr)), np.nan)

for i, T in enumerate(T_arr):
    for j, P in enumerate(P_arr):
        row = df[(df['T_K'] == T) & (df['P_bar'] == P)]
        if not row.empty and bool(row['is_liquid'].iloc[0]):
            err_grid[j, i] = float(row['err_pct'].iloc[0])
        # else: leave as NaN (vapor / out-of-range — shown as gray)

T_C_arr = [T - 273.15 for T in T_arr]

fig, ax = plt.subplots(figsize=(11, 5))
vmax = min(10.0, np.nanpercentile(np.abs(err_grid), 97))
pcm = ax.pcolormesh(T_C_arr, P_arr, err_grid,
                    cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='nearest')
cbar = fig.colorbar(pcm, ax=ax, extend='both',
                    label='Signed error  (eCPA − IAPWS-95) / IAPWS-95  (%)')
cbar.set_label('Signed error  (eCPA − IAPWS-95) / IAPWS-95  (%)',
               fontsize=11, fontweight='bold')
cbar.ax.tick_params(labelsize=9)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight('bold')
ax.set_yscale('log')
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_minor_formatter(mticker.NullFormatter())
ax.set_yticks([1, 10, 100, 1000])
ax.set_xlabel('$T$ (°C)', fontsize=13, fontweight='bold')
ax.set_ylabel('$P$ (bar)', fontsize=13, fontweight='bold')
ax.set_title('eCPA pure-water density error vs IAPWS-95  '
             r'(blue = eCPA denser, red = eCPA lighter)',
             fontsize=11, fontweight='bold')
ax.axvline(100.0, color='white', lw=1.0, ls='--', alpha=0.6, label='100°C')
ax.legend(fontsize=9, prop={'weight': 'bold'})
ax.grid(True, which='major', ls=':', alpha=0.25, color='white')
for t in ax.get_xticklabels() + ax.get_yticklabels():
    t.set_fontweight('bold')
fig.tight_layout()
fig.savefig('figures/density/iapws_error_map.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved figures/density/iapws_error_map.png")


# ── Figure 3: error vs T at representative pressures ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Left: AARE by T, split by pressure band
P_bands = [
    ('$P$ = 1–50 bar',    (1,   50)),
    ('$P$ = 51–300 bar',  (51,  300)),
    ('$P$ = 301–1000 bar',(301, 1001)),
    ('$P$ > 1000 bar',    (1001,2001)),
]
colors = ['steelblue', 'seagreen', 'darkorange', 'tomato']

ax = axes[0]
for (label, (Plo, Phi)), col in zip(P_bands, colors):
    sub = ok[(ok['P_bar'] >= Plo) & (ok['P_bar'] < Phi)]  # ok already liquid-only
    T_grp = sorted(sub['T_K'].unique())
    aare_t = [sub[sub['T_K'] == T]['are_pct'].mean() if len(sub[sub['T_K']==T]) else np.nan
              for T in T_grp]
    ax.plot([T-273.15 for T in T_grp], aare_t, 'o-', color=col, lw=1.4, ms=5, label=label)
ax.axhline(1, ls='--', color='gray', lw=0.8, label='1%')
ax.axhline(2, ls=':',  color='gray', lw=0.8, label='2%')
ax.set_xlabel('$T$ (°C)', fontsize=13)
ax.set_ylabel('AARE (%)', fontsize=13)
ax.set_title('AARE in $\\rho$ vs temperature', fontsize=12)
ax.legend(fontsize=8, ncol=2)
ax.grid(True, ls=':', alpha=0.4)

# Right: signed error at fixed pressures (liquid only)
ax = axes[1]
for P, col in zip([10, 100, 500, 1000, 2000],
                  ['steelblue','seagreen','darkorange','tomato','purple']):
    g = ok[ok['P_bar'] == P].sort_values('T_K')  # ok already liquid-only
    if g.empty:
        continue
    ax.plot(g['T_C'], g['err_pct'], 'o-', color=col, lw=1.4, ms=5,
            label=f'$P$ = {P} bar')
ax.axhline(0,  color='k',    lw=0.8)
ax.axhline( 1, color='gray', lw=0.8, ls='--')
ax.axhline(-1, color='gray', lw=0.8, ls='--')
ax.set_xlabel('$T$ (°C)', fontsize=13)
ax.set_ylabel('Signed error (%)', fontsize=13)
ax.set_title('Signed error vs temperature at fixed $P$', fontsize=12)
ax.legend(fontsize=8)
ax.grid(True, ls=':', alpha=0.4)

fig.tight_layout()
fig.savefig('figures/density/iapws_error_vs_T.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved figures/density/iapws_error_vs_T.png")

print("\nDone.")
