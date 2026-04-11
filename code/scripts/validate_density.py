"""
Validate CPA and eCPA aqueous-phase density predictions against experimental
data from King (1992) and Nighswander (1989) for the CO2 + H2O binary.

Experimental data: rho_W [kg/m³] from EXP/CO2-WATER/ files that contain
a rho_W or rho_w column (ms = 0, no salt).

CPA density  : CPA.flash_co2_h2o_tpz() with Péneloux shifts from ecpa/constants.py
eCPA density : ELV fsolve at ms=0 → rho_W = M_mix / (Vm_EoS + Σ xᵢcᵢ) × 1000 [kg/m³]
               Shifts read from params (ecpa/constants.py): Peneloux_H2O, Peneloux_CO2

Phase compositions are z_co2-independent in the two-phase region, so
z_co2 = 0.3 is used as a fixed value for all CPA calls.
"""
import warnings; warnings.filterwarnings('ignore')

if __name__ == '__main__':
    import re
    import os
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.interpolate import RegularGridInterpolator

    import CPA

    # ── Constants ──────────────────────────────────────────────────────────────
    R_CGS  = 83.14          # bar·cm³/(mol·K)
    M_H2O  = 18.015         # g/mol
    M_CO2  = 44.010         # g/mol

    os.makedirs('results', exist_ok=True)
    os.makedirs('figures/density', exist_ok=True)

    # ── Parse density files ────────────────────────────────────────────────────
    print("Parsing density data from EXP/CO2-WATER/ …")
    data_dir = Path('EXP/CO2-WATER')
    records = []

    for txt_path in sorted(data_dir.rglob('EXP*.txt')):
        if txt_path.stem.endswith('_X') or '(copy)' in txt_path.name.lower():
            continue
        m_T = re.search(r'T(\d+)K', str(txt_path))
        if not m_T:
            continue
        T_K = int(m_T.group(1))

        lines = txt_path.read_text(encoding='utf-8', errors='replace').splitlines()
        reference = ''
        hdr_idx = None
        for i, line in enumerate(lines):
            s = line.strip()
            if s.startswith('#') and not reference:
                reference = s.lstrip('#').strip()
            elif s.lower().startswith('p ['):
                hdr_idx = i
                break
        if hdr_idx is None:
            continue

        # Identify rho_W column (case-insensitive)
        hdr = lines[hdr_idx]
        hdr_clean = re.sub(r'\[.*?\]', '', hdr).lower().split()
        rho_col = None
        for j, tok in enumerate(hdr_clean):
            if tok in ('rho_w', 'rho_c'):
                rho_col = (j, tok)   # take first density column found
                break
        if rho_col is None:
            continue

        col_idx, col_name = rho_col
        for line in lines[hdr_idx + 1:]:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            parts = s.split()
            if len(parts) <= col_idx:
                continue
            try:
                P_bar = float(parts[0])
            except ValueError:
                continue
            val_s = parts[col_idx]
            if val_s.lower() in ('x', 'nan', ''):
                continue
            try:
                rho_exp = float(val_s)
            except ValueError:
                continue
            records.append({
                'T_K': T_K, 'P_bar': P_bar,
                'rho_exp': rho_exp,      # kg/m³
                'qty': col_name,         # 'rho_w' or 'rho_c'
                'reference': reference,
                'source_file': txt_path.name,
            })

    exp_df = pd.DataFrame(records).sort_values(['T_K', 'P_bar']).reset_index(drop=True)
    print(f"  Found {len(exp_df)} density points across "
          f"{exp_df['T_K'].nunique()} temperatures:")
    for T, grp in exp_df.groupby('T_K'):
        refs = ', '.join(grp['reference'].unique())
        print(f"    T={T}K  {len(grp)} pts  [{refs}]")

    # Keep only aqueous-phase density (rho_w) for now
    exp_aq = exp_df[exp_df['qty'] == 'rho_w'].copy()

    # ── Build eCPA warm-start interpolator from CPA_ELV_all.parquet ───────────
    # Used only as initial guess for fsolve; exact ELV solve avoids interpolation
    # artifacts (non-monotonic jumps near CO2 critical region at high T).
    print("\nBuilding eCPA warm-start interpolator from CPA_ELV_all.parquet …")
    from scipy.optimize import fsolve
    from ecpa.elv import ELV, ELV_jac, USE_COMPLEX_JAC
    from ecpa.parameters import make_params

    params = make_params()
    elv_df = pd.read_parquet('CPA_ELV_all.parquet')

    SOL_COLS = ['Z_W', 'xw_W', 'eps_r', 'Z_C', 'xw_C',
                'chiw_W', 'chiw_C', 'Ndchi1w_dNw', 'Ndchi1w_dNc', 'Vdchi1w_dV']

    # Cache ELV solutions per T for fast lookup
    elv_cache = {}
    for T_K, grp in elv_df.groupby('T_K'):
        grp = grp.sort_values('P_bar')
        elv_cache[float(T_K)] = {
            'P':   grp['P_bar'].values,
            'sol': grp[SOL_COLS].values,
        }
    elv_T_arr = np.array(sorted(elv_cache.keys()))

    def _interp_elv_guess(T, P_bar):
        """Bilinear (T, P) interpolation of CPA_ELV_all → 10-vector initial guess."""
        idx_T = np.searchsorted(elv_T_arr, T)
        if idx_T == 0:
            T_lo = T_hi = elv_T_arr[0]; w_hi = 0.0
        elif idx_T >= len(elv_T_arr):
            T_lo = T_hi = elv_T_arr[-1]; w_hi = 0.0
        else:
            T_lo, T_hi = elv_T_arr[idx_T - 1], elv_T_arr[idx_T]
            w_hi = (T - T_lo) / (T_hi - T_lo)

        def _at_P(Tk):
            e = elv_cache[Tk]
            Pa, Sa = e['P'], e['sol']
            if P_bar <= Pa[0]:  return Sa[0].copy()
            if P_bar >= Pa[-1]: return Sa[-1].copy()
            i = max(1, int(np.searchsorted(Pa, P_bar)))
            f = (P_bar - Pa[i - 1]) / (Pa[i] - Pa[i - 1])
            return (1 - f) * Sa[i - 1] + f * Sa[i]

        s_lo = _at_P(T_lo)
        return s_lo if T_lo == T_hi else (1 - w_hi) * s_lo + w_hi * _at_P(T_hi)

    def ecpa_density(T, P_bar):
        """Exact eCPA aqueous-phase density [kg/m³] at ms=0 via ELV fsolve."""
        guess = _interp_elv_guess(T, P_bar)
        try:
            sol, info, ier, _ = fsolve(
                ELV, guess,
                args=(float(T), float(P_bar) * 1e5, 0.0, params),
                fprime=ELV_jac if USE_COMPLEX_JAC else None,
                full_output=True, xtol=1e-10, maxfev=2000,
            )
            sol = np.asarray(sol, dtype=np.float64)
            res_norm = np.linalg.norm(
                ELV(sol, float(T), float(P_bar) * 1e5, 0.0, params))
            ok = (ier == 1 and np.isfinite(sol).all() and res_norm < 1e-6
                  and 0.0 < sol[1] < 1.0 and 0.0 < sol[4] < 1.0
                  and sol[1] - sol[4] > 0.01)
        except Exception:
            ok = False
        if not ok:
            return np.nan
        Z_W  = float(sol[0])
        xw_W = float(sol[1])   # H2O mole fraction in aqueous phase
        xc_W = 1.0 - xw_W      # CO2 mole fraction in aqueous phase
        M_mix = xw_W * M_H2O + xc_W * M_CO2   # g/mol
        Vm    = Z_W * R_CGS * T / P_bar        # cm³/mol
        # No Péneloux volume shift for H2O or CO2
        if Vm <= 0:
            return np.nan
        return (M_mix / Vm) * 1000.0           # kg/m³

    # ── Run CPA and eCPA for each experimental point ──────────────────────────
    print("\nRunning CPA and eCPA at experimental conditions …")
    Z_CO2 = 0.3   # z-independent in two-phase region

    # No Péneloux volume shift for H2O or CO2
    print("  No Péneloux volume shift applied for H₂O or CO₂")

    rows_out = []
    for _, row in exp_aq.iterrows():
        T, P = float(row['T_K']), float(row['P_bar'])

        # CPA — no volume shift
        rho_cpa = np.nan
        try:
            r = CPA.flash_co2_h2o_tpz(T=T, P_bar=P, z_co2=Z_CO2)
            if r['phase'] == 'two_phase' and r['tie']['converged']:
                rho_cpa = float(r['tie']['rho_mass'][0]) * 1000.0  # kg/L → kg/m³
        except Exception:
            pass

        # eCPA
        rho_ecpa = ecpa_density(T, P)

        rows_out.append({
            **row.to_dict(),
            'rho_cpa':  rho_cpa,
            'rho_ecpa': rho_ecpa,
        })
        print(f"  T={T:.0f}K  P={P:.1f}bar  "
              f"exp={row['rho_exp']:.1f}  "
              f"CPA={rho_cpa:.1f}  "
              f"eCPA={rho_ecpa:.1f}")

    res = pd.DataFrame(rows_out)
    res.to_parquet('results/density_co2h2o.parquet', index=False)
    print(f"\nSaved results/density_co2h2o.parquet  ({len(res)} rows)")

    # ── Metrics ────────────────────────────────────────────────────────────────
    print("\n── Density metrics (rho_W, aqueous phase) ─────────────────────────")
    for model, col in [('CPA', 'rho_cpa'), ('eCPA', 'rho_ecpa')]:
        ok = res.dropna(subset=[col, 'rho_exp'])
        are = (ok[col] - ok['rho_exp']).abs() / ok['rho_exp']
        bias = ((ok[col] - ok['rho_exp']) / ok['rho_exp']).mean() * 100
        print(f"  {model}:  N={len(ok)}  "
              f"AARE={are.mean()*100:.2f}%  "
              f"bias={bias:+.2f}%  "
              f"max_ARE={are.max()*100:.1f}%")

    # By temperature
    print("\n  By temperature:")
    print(f"  {'T [K]':>6}  {'N':>3}  {'CPA AARE':>9}  {'eCPA AARE':>10}")
    for T, grp in res.groupby('T_K'):
        def aare(col):
            ok = grp.dropna(subset=[col, 'rho_exp'])
            if ok.empty:
                return float('nan')
            return (ok[col] - ok['rho_exp']).abs().mean() / ok['rho_exp'].mean() * 100
        print(f"  {T:>6.0f}  {len(grp):>3}  "
              f"{aare('rho_cpa'):>8.2f}%  {aare('rho_ecpa'):>9.2f}%")

    # ── Single eCPA parity plot (no volume shift for H2O) ─────────────────────
    print("\nGenerating eCPA parity plot …")
    T_all = sorted(res['T_K'].unique())
    cmap  = plt.cm.viridis
    T_norm = plt.Normalize(vmin=min(T_all), vmax=max(T_all))

    fig, ax = plt.subplots(figsize=(5.5, 5.0))

    ok = res.dropna(subset=['rho_ecpa', 'rho_exp'])
    for T in T_all:
        g = ok[ok['T_K'] == T]
        if g.empty:
            continue
        ax.scatter(g['rho_exp'], g['rho_ecpa'], color=cmap(T_norm(T)),
                   s=50, zorder=4)

    vmin = ok[['rho_exp', 'rho_ecpa']].min().min() - 5
    vmax = ok[['rho_exp', 'rho_ecpa']].max().max() + 5
    lv = np.linspace(vmin, vmax, 200)
    ax.plot(lv, lv, 'k-', lw=1.0)
    ax.fill_between(lv, lv * 0.99, lv * 1.01, color='green',  alpha=0.15, label='±1%')
    ax.fill_between(lv, lv * 0.98, lv * 1.02, color='orange', alpha=0.12, label='±2%')

    aare = (ok['rho_ecpa'] - ok['rho_exp']).abs().mean() / ok['rho_exp'].mean() * 100
    ax.set_xlabel(r'Experimental $\rho_W$ [kg m$^{-3}$]', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'eCPA $\rho_W$ [kg m$^{-3}$]', fontsize=12, fontweight='bold')
    ax.set_title(f'eCPA  (AARE = {aare:.2f}%)', fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    ax.set_xlim(vmin, vmax); ax.set_ylim(vmin, vmax)
    ax.set_aspect('equal')

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=T_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label='T [K]', shrink=0.85, pad=0.02)
    cbar.ax.tick_params(labelsize=9)

    fig.tight_layout()
    fig.savefig('figures/density/parity_ecpa.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved figures/density/parity_ecpa.png  (AARE = {aare:.2f}%)")

    print("\nDone.")
