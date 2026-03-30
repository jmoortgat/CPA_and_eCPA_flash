"""
benchmark_simplified_flash.py
==============================
Compare the simplified flash (y_{H₂O} = 0 assumption) against the full
K-value SSI flash for CO₂ + H₂O (CPA) and CO₂ + H₂O + NaCl (eCPA).

Outputs
-------
- Console table: AARE, AAE, bias on x_{CO₂}^{aq}, m_c, β, and y_{H₂O}
- figures/simplified/accuracy_vs_T.pdf  — error vs temperature panel
- figures/simplified/efficiency_vs_T.pdf — n_calls vs temperature panel
- results/simplified_comparison.parquet — full per-point results
"""
import warnings
warnings.filterwarnings('ignore')

import os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

os.makedirs('figures/simplified', exist_ok=True)
os.makedirs('results', exist_ok=True)

from ecpa.parameters import make_params
from ecpa.flash import flash_co2_h2o_salt_kv
from ecpa.flash_simplified import flash_co2_h2o_simplified
from ecpa.constants import Mw

params = make_params()

# ── Scan grid ────────────────────────────────────────────────────────────────
# Cover typical CCS reservoir conditions + some high-T to show breakdown.
T_vals  = [283, 298, 313, 323, 333, 348, 353, 373, 393, 413, 423, 453, 473, 523]
P_vals  = [25, 50, 75, 100, 150, 200, 300, 400, 600, 800]
z_vals  = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70]
ms_vals = [0.0, 1.0, 2.0, 4.0, 6.0]

print("Building comparison grid …")
print(f"  {len(T_vals)} T × {len(P_vals)} P × {len(z_vals)} z × {len(ms_vals)} ms "
      f"= {len(T_vals)*len(P_vals)*len(z_vals)*len(ms_vals):,} conditions")

rows = []
for T in T_vals:
    for P in P_vals:
        for z in z_vals:
            for ms in ms_vals:
                # ── Full K-value flash (reference) ──────────────────────────
                t0 = time.perf_counter()
                try:
                    r_full = flash_co2_h2o_salt_kv(
                        T=T, P_bar=P, z_co2=z, m_tot=ms, params=params)
                    ok_full = True
                except Exception:
                    r_full  = None
                    ok_full = False
                t_full = time.perf_counter() - t0

                # ── Simplified flash ────────────────────────────────────────
                t0 = time.perf_counter()
                try:
                    r_simp = flash_co2_h2o_simplified(
                        T=T, P_bar=P, z_co2=z, m_tot=ms)
                    ok_simp = True
                except Exception:
                    r_simp  = None
                    ok_simp = False
                t_simp = time.perf_counter() - t0

                if not (ok_full and ok_simp):
                    continue

                # ── Reference quantities ────────────────────────────────────
                # flash_co2_h2o_salt_kv always returns two-phase (no 'phase' key);
                # flash_co2_h2o_simplified uses phase='two_phase'|'single_phase'.
                phase_full = r_full.get('phase', 'two_phase')
                is_2ph_full = (phase_full != 'single_phase') and r_full.get('beta', 0) > 0
                is_2ph_simp = r_simp.get('phase') == 'two_phase'

                x4w_full = r_full['x_aq']['x4w'] if is_2ph_full else float('nan')
                x4w_simp = r_simp['x_aq']['x4w'] if is_2ph_simp else float('nan')
                x1w_full = r_full['x_aq']['x1w'] if is_2ph_full else float('nan')
                x1w_simp = r_simp['x_aq']['x1w'] if is_2ph_simp else float('nan')

                # Convert x_{CO₂}^{aq} → molality m_c [mol/kg H₂O]
                def x4w_to_mc(x4w_, x1w_):
                    if x1w_ > 0:
                        return x4w_ / (x1w_ * Mw)
                    return float('nan')

                mc_full = x4w_to_mc(x4w_full, x1w_full)
                mc_simp = x4w_to_mc(x4w_simp, x1w_simp)

                beta_full = r_full.get('beta', float('nan')) if is_2ph_full else 0.0
                beta_simp = r_simp.get('beta', float('nan')) if is_2ph_simp else 0.0

                x1c_full = r_full['x_c']['x1c'] if is_2ph_full else float('nan')  # y_H2O in CO2-rich
                x1c_simp = 0.0  # assumed zero

                ms_aq_full = r_full.get('ms_aq', float('nan'))
                ms_aq_simp = r_simp.get('ms_aq', float('nan'))

                n_iter_full = r_full.get('n_iter_ms', float('nan'))
                n_iter_simp = r_simp.get('n_iter_ms', float('nan'))

                rows.append(dict(
                    T_K=T, P_bar=P, z_co2=z, ms=ms,
                    phase_full=phase_full,
                    is_2ph_full=is_2ph_full,
                    is_2ph_simp=is_2ph_simp,
                    # Aqueous CO2 content
                    x4w_full=x4w_full, x4w_simp=x4w_simp,
                    mc_full=mc_full,   mc_simp=mc_simp,
                    # H2O in CO2-rich phase
                    y1c_full=x1c_full, y1c_simp=x1c_simp,
                    # Phase fraction
                    beta_full=beta_full, beta_simp=beta_simp,
                    # Aqueous molality
                    ms_aq_full=ms_aq_full, ms_aq_simp=ms_aq_simp,
                    # Performance
                    n_iter_full=n_iter_full, n_iter_simp=n_iter_simp,
                    t_full_ms=t_full * 1e3, t_simp_ms=t_simp * 1e3,
                ))

df = pd.DataFrame(rows)
df.to_parquet('results/simplified_comparison.parquet', index=False)
print(f"  Saved {len(df)} rows → results/simplified_comparison.parquet")

# ── Accuracy analysis (two-phase conditions only) ─────────────────────────────
tp = df[df['is_2ph_full'] & df['is_2ph_simp'] & df['mc_full'].notna()].copy()
tp['agree_phase'] = df['is_2ph_full'] == df['is_2ph_simp']

def rel_err(pred, ref):
    return (pred - ref) / np.where(np.abs(ref) > 1e-12, np.abs(ref), 1e-12)

def aare(pred, ref):
    return np.abs(rel_err(pred, ref)).mean() * 100

def aae(pred, ref):
    return np.abs(pred - ref).mean()

def bias(pred, ref):
    return rel_err(pred, ref).mean() * 100

print("\n" + "="*70)
print("ACCURACY SUMMARY (two-phase conditions where both flashes agree)")
print("="*70)

metrics = {
    'x_{CO2}^{aq}':  ('x4w_simp', 'x4w_full'),
    'm_c [mol/kg]':  ('mc_simp',  'mc_full'),
    'beta':          ('beta_simp','beta_full'),
    'ms_aq [mol/kg]':('ms_aq_simp','ms_aq_full'),
}
for label, (pred_col, ref_col) in metrics.items():
    sub = tp[tp[pred_col].notna() & tp[ref_col].notna()]
    if len(sub) == 0:
        continue
    print(f"\n  {label}  (N={len(sub)})")
    print(f"    AARE = {aare(sub[pred_col], sub[ref_col]):.2f}%")
    print(f"    AAE  = {aae(sub[pred_col],  sub[ref_col]):.4e}")
    print(f"    bias = {bias(sub[pred_col], sub[ref_col]):+.2f}%")

# By temperature
print("\n── Error in m_c by temperature (two-phase, all ms) ──────────────────")
print(f"{'T [K]':>8}  {'N':>5}  {'AARE(mc)%':>10}  {'AAE(mc)':>10}  {'y_H2O_full':>10}")
for T in sorted(tp['T_K'].unique()):
    g = tp[tp['T_K'] == T]
    if len(g) < 2:
        continue
    # Mean y_H2O in CO2-rich phase from full flash (tells us how valid simplification is)
    y1c_mean = g['y1c_full'].mean()
    print(f"{T:8.0f}  {len(g):5d}  {aare(g['mc_simp'], g['mc_full']):10.2f}  "
          f"{aae(g['mc_simp'], g['mc_full']):10.4e}  {y1c_mean:10.4f}")

# ── Phase identification agreement ───────────────────────────────────────────
print(f"\n── Phase agreement ───────────────────────────────────────────────────")
n_agree = (df['is_2ph_full'] == df['is_2ph_simp']).sum()
print(f"  Phase (2-phase vs single-phase) agreement: {n_agree}/{len(df)} "
      f"({100*n_agree/len(df):.1f}%)")

# ── Efficiency ────────────────────────────────────────────────────────────────
print(f"\n── Computational efficiency ──────────────────────────────────────────")
tp_both = df[df['is_2ph_full'] & df['is_2ph_simp']]
speedup = tp_both['t_full_ms'].mean() / tp_both['t_simp_ms'].mean()
print(f"  Mean wall time (two-phase): full={tp_both['t_full_ms'].mean():.3f} ms, "
      f"simplified={tp_both['t_simp_ms'].mean():.3f} ms")
print(f"  Mean speedup: {speedup:.1f}×")
print(f"  Mean Brent calls (simplified): {tp_both['n_iter_simp'].mean():.1f}")
print(f"  Mean SSI iters  (full):        {tp_both['n_iter_full'].mean():.1f}")

# ── Figures ───────────────────────────────────────────────────────────────────
T_unique = sorted(tp['T_K'].unique())
aare_by_T = [aare(tp[tp['T_K']==T]['mc_simp'], tp[tp['T_K']==T]['mc_full'])
             for T in T_unique]
aae_by_T  = [aae(tp[tp['T_K']==T]['mc_simp'],  tp[tp['T_K']==T]['mc_full'])
             for T in T_unique]
y1c_by_T  = [tp[tp['T_K']==T]['y1c_full'].mean() for T in T_unique]
speedup_T = [(df[df['T_K']==T]['t_full_ms'].mean() /
              df[df['T_K']==T]['t_simp_ms'].mean())
             for T in T_unique]

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

ax = axes[0]
ax.plot([T-273.15 for T in T_unique], aare_by_T, 'o-', color='steelblue')
ax.set_xlabel('T (°C)'); ax.set_ylabel('AARE in $m_c$ (%)');
ax.set_title('Accuracy: AARE in dissolved CO₂')
ax.axhline(5, ls='--', color='gray', lw=0.8, label='5%')
ax.legend(fontsize=9); ax.grid(True, ls=':', alpha=0.4)

ax = axes[1]
ax.plot([T-273.15 for T in T_unique], [v*100 for v in y1c_by_T], 's-', color='tomato')
ax.set_xlabel('T (°C)'); ax.set_ylabel('Mean $y_{H_2O}$ in CO₂-rich phase (mol%)');
ax.set_title('Validity: $y_{H_2O}$ of full flash')
ax.axhline(1, ls='--', color='gray', lw=0.8, label='1 mol%')
ax.legend(fontsize=9); ax.grid(True, ls=':', alpha=0.4)

ax = axes[2]
ax.plot([T-273.15 for T in T_unique], speedup_T, '^-', color='seagreen')
ax.set_xlabel('T (°C)'); ax.set_ylabel('Speedup (full / simplified)')
ax.set_title('Efficiency: wall-time speedup')
ax.axhline(1, ls='--', color='gray', lw=0.8)
ax.grid(True, ls=':', alpha=0.4)

fig.tight_layout()
fig.savefig('figures/simplified/accuracy_vs_T.pdf', bbox_inches='tight', dpi=150)
fig.savefig('figures/simplified/accuracy_vs_T.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print("\nSaved figures/simplified/accuracy_vs_T.{pdf,png}")

# Accuracy vs T for different ms values (mc AARE)
fig, ax = plt.subplots(figsize=(7, 4))
T_C = [T-273.15 for T in T_unique]
for ms_v in sorted(tp['ms'].unique()):
    sub = tp[tp['ms'] == ms_v]
    err = [aare(sub[sub['T_K']==T]['mc_simp'], sub[sub['T_K']==T]['mc_full'])
           if len(sub[sub['T_K']==T]) > 0 else float('nan')
           for T in T_unique]
    ax.plot(T_C, err, 'o-', lw=1.2, ms=5,
            label=f'$m_s$ = {ms_v:.1g} mol/kg')
ax.set_xlabel('T (°C)'); ax.set_ylabel('AARE in $m_c$ (%)')
ax.set_title('Simplified vs full flash: accuracy by salinity')
ax.axhline(5, ls='--', color='gray', lw=0.8)
ax.legend(fontsize=9, ncol=2)
ax.grid(True, ls=':', alpha=0.4)
fig.tight_layout()
fig.savefig('figures/simplified/accuracy_by_ms.pdf', bbox_inches='tight', dpi=150)
fig.savefig('figures/simplified/accuracy_by_ms.png', bbox_inches='tight', dpi=150)
plt.close(fig)
print("Saved figures/simplified/accuracy_by_ms.{pdf,png}")
print("\nDone.")
