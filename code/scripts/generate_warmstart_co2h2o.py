"""
CO2-H2O warm-start validation with ribbon plots — parallelized.

Key design:
  1. CPA computed ONCE for all (T, P) in parallel across T values.
  2. eCPA computed for each ms in parallel across T values.
     Uses ecpa_stability (always, independent of any table hint) +
     flash_co2_h2o_salt_kv with ScanTableWarmStart (scan_v3_table.npz,
     50 P-grid points, auto-loads cpa_table.npz for ms≈0).
  3. CPA columns merged into each ms-specific parquet.
  4. Per-T ribbon plots with fill_between bands.

Usage:
  python _run_warmstart_co2h2o.py             # use cached parquets if available
  python _run_warmstart_co2h2o.py --recompute # delete smooth-curve caches and rerun

Output:
  results/ws_cpa_smooth.parquet              — CPA only (200 P pts, all T)
  results/ws_smooth_co2h2o_ms*.parquet       — CPA + eCPA per ms (200 P pts)
  results/ws_validation_co2h2o.parquet       — validation at experimental T,P
  figures/co2h2o_ws/T*K.png                  — per-temperature ribbon figures
"""
import warnings
warnings.filterwarnings('ignore')

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

N_WORKERS      = 8       # leave 2 cores free
N_P            = 200
P_MIN          = 1.0
P_MAX          = 1500.0
Z_CO2          = 0.5     # default feed
SCAN_TABLE_PATH = "results/scan_v3_table.npz"

MS_RIBBON = [1e-5, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]


def _ms_tag(ms_val):
    return f'{ms_val:.4f}' if ms_val < 0.1 else f'{ms_val:.1f}'


# ── Worker: CPA for one T, all P ──────────────────────────────────────────────

def _cpa_T_worker(args):
    import warnings; warnings.filterwarnings('ignore')
    T, P_grid = args
    from ecpa.validate_co2h2o import run_cpa_binary, Z_CO2_RETRY
    records = []
    for P in P_grid:
        cpa = run_cpa_binary(T, P, z_co2=Z_CO2)
        if np.isnan(cpa['xc_W']):
            for z_r in Z_CO2_RETRY:
                cpa_r = run_cpa_binary(T, P, z_co2=z_r)
                if not np.isnan(cpa_r['xc_W']):
                    cpa = cpa_r
                    break
        records.append({
            'T_K': float(T), 'P_bar': float(P),
            'cpa_xc_W':     cpa['xc_W'],
            'cpa_yw_C':     cpa['yw_C'],
            'cpa_converged': cpa['converged'],
            'cpa_n_iter':   cpa['n_iter'],
            'cpa_t_ms':     cpa['t_ms'],
        })
    return T, records


# ── Worker: eCPA for one (T, ms), all P ───────────────────────────────────────

def _ecpa_T_worker(args):
    import warnings; warnings.filterwarnings('ignore')
    import time as _time
    T, ms, P_grid, _unused = args
    from ecpa.solution_table import make_solution_guess_fn, load_solution_table
    from ecpa.parameters import make_params
    from ecpa.flash import flash_co2_h2o_salt_fast_kv
    from ecpa.validate_co2h2o import Z_CO2_RETRY, T_MAX_ECPA

    grid_data = load_solution_table()
    guess_fn  = make_solution_guess_fn(grid_data)
    params    = make_params()
    ecpa_ok   = float(T) <= T_MAX_ECPA

    Z_CANDIDATES = [Z_CO2] + list(Z_CO2_RETRY)

    records = []
    for P in P_grid:
        rec = {'T_K': float(T), 'P_bar': float(P), 'ecpa_available': ecpa_ok}
        if ecpa_ok:
            t0 = _time.perf_counter()
            out = None
            for z in Z_CANDIDATES:
                try:
                    r = flash_co2_h2o_salt_fast_kv(
                        T=T, P_bar=P, z_co2=z, m_tot=max(ms, 1e-9),
                        solution_guess_fn=guess_fn, params=params,
                    )
                    if r.get('phase') == 'two_phase':
                        out = r
                        break
                except Exception:
                    continue
            t_ms = (_time.perf_counter() - t0) * 1e3

            if out is not None:
                x_aq = out['x_aq']
                x_c  = out['x_c']
                rec.update({
                    'ecpa_xc_W':          float(x_aq['x4w']),        # CO2 in aq
                    'ecpa_yw_C':          float(x_c['x1c']),          # H2O in CO2-rich
                    'ecpa_converged':     True,
                    'ecpa_n_iter_ms':     int(out.get('n_iter_ms', 0)),
                    'ecpa_stability_run': out.get('stable') is not None,
                    'ecpa_t_ms':          t_ms,
                })
            else:
                rec.update({
                    'ecpa_xc_W': np.nan, 'ecpa_yw_C': np.nan,
                    'ecpa_converged': False, 'ecpa_n_iter_ms': 0,
                    'ecpa_stability_run': False, 'ecpa_t_ms': t_ms,
                })
        else:
            rec.update({
                'ecpa_xc_W': np.nan, 'ecpa_yw_C': np.nan,
                'ecpa_converged': False, 'ecpa_n_iter_ms': 0,
                'ecpa_stability_run': False, 'ecpa_t_ms': 0.0,
            })
        records.append(rec)
    return T, records


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--recompute', action='store_true',
                    help='Delete cached smooth-curve parquets and recompute from scratch')
    args_cli = ap.parse_args()

    os.makedirs('results', exist_ok=True)
    os.makedirs('figures/co2h2o_ws', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    from ecpa.parameters import make_params
    from ecpa.validate_co2h2o import run_validation, T_MAX_ECPA

    if args_cli.recompute:
        print("--recompute: removing cached smooth-curve parquets …")
        removed = 0
        for tag_ms in [_ms_tag(m) for m in MS_RIBBON]:
            p = Path(f'results/ws_smooth_co2h2o_ms{tag_ms}.parquet')
            if p.exists():
                p.unlink()
                removed += 1
        print(f"  Removed {removed} parquet file(s)")

    params = make_params()

    exp_df = pd.read_parquet('CO2_WATER_exp.parquet')
    T_unique     = sorted(exp_df['T_K'].unique())
    curve_T_vals = [T for T in T_unique if T <= T_MAX_ECPA]
    P_grid       = np.logspace(np.log10(P_MIN), np.log10(P_MAX), N_P)

    print(f"T range: {curve_T_vals[0]:.0f}–{curve_T_vals[-1]:.0f} K "
          f"({len(curve_T_vals)} values), {N_P} P points, {N_WORKERS} workers")

    # ── Phase 1: CPA (once, parallel across T) ────────────────────────────────
    cpa_cache = Path('results/ws_cpa_smooth.parquet')
    if cpa_cache.exists():
        print(f"\nLoading cached CPA curves …")
        cpa_df = pd.read_parquet(cpa_cache)
    else:
        print(f"\n=== CPA smooth curves ({len(curve_T_vals)} T × {N_P} P) ===")
        t0 = time.perf_counter()
        args_list = [(T, P_grid) for T in curve_T_vals]
        rows_all = []
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            futs = {pool.submit(_cpa_T_worker, a): a[0] for a in args_list}
            done = 0
            for fut in as_completed(futs):
                T_done, recs = fut.result()
                rows_all.extend(recs)
                done += 1
                print(f"  CPA T={T_done:.0f}K  [{done}/{len(curve_T_vals)}]",
                      flush=True)
        cpa_df = pd.DataFrame(rows_all).sort_values(['T_K', 'P_bar'])
        cpa_df.to_parquet(cpa_cache, index=False)
        elapsed = time.perf_counter() - t0
        print(f"  → saved {cpa_cache}  ({elapsed:.1f}s)")

    # ── Phase 2: eCPA per ms (parallel across T, ms sequential) ──────────────
    print(f"\n=== eCPA smooth curves ({len(MS_RIBBON)} ms values) ===")
    smooth_data = {}   # ms_val → DataFrame

    for ms_val in MS_RIBBON:
        tag   = _ms_tag(ms_val)
        cache = Path(f'results/ws_smooth_co2h2o_ms{tag}.parquet')
        if cache.exists():
            print(f"  Loading cached ms={ms_val} …")
            smooth_data[ms_val] = pd.read_parquet(cache)
            continue

        print(f"  Computing eCPA ms={ms_val} mol/kg …", flush=True)
        t0 = time.perf_counter()
        args_list = [(T, ms_val, P_grid, None) for T in curve_T_vals]
        rows_ecpa = []
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            futs = {pool.submit(_ecpa_T_worker, a): a[0] for a in args_list}
            done = 0
            for fut in as_completed(futs):
                T_done, recs = fut.result()
                rows_ecpa.extend(recs)
                done += 1
                if done % 10 == 0 or done == len(curve_T_vals):
                    print(f"    [{done}/{len(curve_T_vals)}]", flush=True)

        ecpa_df = pd.DataFrame(rows_ecpa).sort_values(['T_K', 'P_bar'])
        # Merge CPA columns
        df = ecpa_df.merge(cpa_df, on=['T_K', 'P_bar'], how='left')
        df.to_parquet(cache, index=False)
        elapsed = time.perf_counter() - t0
        print(f"    → saved {cache}  ({elapsed:.1f}s)")
        smooth_data[ms_val] = df

    # ── Phase 3: Validation at experimental T, P ──────────────────────────────
    # The validation parquet provides scatter points for the ribbon plots.
    # It uses the old solution_guess_fn; the cached result is fine since
    # validation is at specific experimental (T, P) — not a smooth P sweep.
    from ecpa.solution_table import load_solution_table, make_solution_guess_fn
    grid_data         = load_solution_table()
    solution_guess_fn = make_solution_guess_fn(grid_data)

    val_cache = Path('results/ws_validation_co2h2o.parquet')
    if val_cache.exists():
        print(f"\nLoading cached validation …")
        results_df = pd.read_parquet(val_cache)
    else:
        print(f"\n=== Validation at {len(exp_df)} experimental points ===")
        results_df = run_validation(
            exp_df=exp_df,
            solution_guess_fn=solution_guess_fn,
            params=params,
            ms=1e-5,
            verbose=True,
        )
        results_df.to_parquet(val_cache, index=False)
        print(f"  Saved {val_cache}")

    # Iteration/convergence summary
    if 'ecpa_n_iter_ms' in results_df.columns:
        avail = results_df[results_df['ecpa_available'].fillna(False)]
        n_conv = avail['ecpa_converged'].fillna(False).sum()
        iters  = avail['ecpa_n_iter_ms'].dropna()
        print(f"\neCPA validation: {n_conv}/{len(avail)} converged  "
              f"SSI median={iters.median():.1f}  max={iters.max():.0f}")

    # ── Phase 4: Ribbon plots ─────────────────────────────────────────────────
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    CMAP  = plt.cm.rainbow
    CNORM = mcolors.Normalize(vmin=0.0, vmax=max(m for m in MS_RIBBON if m > 0.01))
    P_FINE = np.logspace(np.log10(P_MIN), np.log10(P_MAX), 3000)

    # ── Figure style: larger, bold fonts ────────────────────────────────────────
    matplotlib.rcParams.update({
        'font.size':          15,
        'font.weight':        'bold',
        'axes.labelsize':     15,
        'axes.labelweight':   'bold',
        'xtick.labelsize':    13,
        'ytick.labelsize':    13,
        'legend.fontsize':    11,
        'axes.titlesize':     13,
    })

    _COLORS = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    ]

    def _short_ref(ref):
        import re
        m = re.search(r'(\w+)\s+(?:et al\.?\s*)?\((\d{4})\)', str(ref))
        return f"{m.group(1)} ({m.group(2)})" if m else str(ref)[:25]

    def _interp_curve(df, T_K, col):
        sub  = df[np.abs(df['T_K'] - T_K) < 0.5].sort_values('P_bar')
        if len(sub) < 2:
            return np.full(len(P_FINE), np.nan)
        mask = sub[col].notna() & (sub[col] > 0) & (sub['P_bar'] > 0)
        if mask.sum() < 4:
            return np.full(len(P_FINE), np.nan)
        logP = np.log10(sub.loc[mask, 'P_bar'].values)
        logY = np.log10(sub.loc[mask, col].values)

        # Remove isolated bad-convergence spikes: a point is an outlier if it
        # deviates by more than 2× the local variation from the linear trend
        # through its two neighbours (in log-log space).  Only interior points
        # are tested; phase-boundary endpoints are kept as-is.
        n = len(logY)
        keep = np.ones(n, dtype=bool)
        for i in range(1, n - 1):
            y_trend = (logY[i-1] * (logP[i+1] - logP[i])
                       + logY[i+1] * (logP[i] - logP[i-1])) / (logP[i+1] - logP[i-1])
            local_var = abs(logY[i+1] - logY[i-1]) + 1e-6
            if abs(logY[i] - y_trend) > 2.0 * local_var:
                keep[i] = False
        logP = logP[keep]
        logY = logY[keep]

        if len(logP) < 2:
            return np.full(len(P_FINE), np.nan)
        logPf = np.log10(P_FINE)
        within = (logPf >= logP.min()) & (logPf <= logP.max())
        out = np.full(len(P_FINE), np.nan)
        if within.sum() >= 2:
            out[within] = 10.0 ** np.interp(logPf[within], logP, logY)
        return out

    sorted_ms = sorted(smooth_data.keys())

    def plot_ribbon_T(T_K):
        # Layout: two panels, colorbar right of right panel only.
        # Gap of 0.055 gives the right panel's y-label enough room to clear the left panel.
        fig = plt.figure(figsize=(12, 5))
        ax0 = fig.add_axes([0.07,  0.12, 0.39, 0.80])   # left panel
        ax1 = fig.add_axes([0.545, 0.12, 0.39, 0.80])   # right panel (gap = 0.085)
        cax = fig.add_axes([0.950, 0.12, 0.018, 0.80])  # colorbar (right panel only)

        panels = [
            (ax0, 'ecpa_xc_W', 'cpa_xc_W', 'exp_xc_W',
             r'$x_{\mathrm{CO_2}}$ (aqueous)'),
            (ax1, 'ecpa_yw_C', 'cpa_yw_C', 'exp_yw_C',
             r'$y_{\mathrm{H_2O}}$ (CO$_2$-rich phase)'),
        ]

        for ax, ecpa_col, cpa_col, exp_col, ylabel in panels:
            curves = {ms_v: _interp_curve(df, T_K, ecpa_col)
                      for ms_v, df in smooth_data.items()}

            # Fill between adjacent ms curves
            for i in range(len(sorted_ms) - 1):
                ms_lo, ms_hi = sorted_ms[i], sorted_ms[i + 1]
                y_lo, y_hi   = curves[ms_lo], curves[ms_hi]
                valid = np.isfinite(y_lo) & np.isfinite(y_hi)
                if valid.sum() < 2:
                    continue
                y1 = np.where(valid, np.maximum(y_lo, y_hi), np.nan)
                y2 = np.where(valid, np.minimum(y_lo, y_hi), np.nan)
                ax.fill_between(P_FINE, y1, y2, where=valid,
                                color=CMAP(CNORM(0.5 * (ms_lo + ms_hi))),
                                alpha=0.45, linewidth=0, zorder=2)

            # Discrete ms lines
            for ms_v in sorted_ms:
                y = curves[ms_v]
                valid = np.isfinite(y)
                if valid.sum() < 2:
                    continue
                ax.plot(P_FINE[valid], y[valid],
                        '-', color=CMAP(CNORM(ms_v if ms_v > 0.01 else 0.0)),
                        lw=1.2, alpha=0.85, zorder=3)

            # CPA (from first smooth_data entry — same for all ms)
            cpa_y = _interp_curve(smooth_data[sorted_ms[0]], T_K, cpa_col)
            valid = np.isfinite(cpa_y)
            if valid.sum() > 1:
                ax.plot(P_FINE[valid], cpa_y[valid],
                        'k-', lw=2.5, zorder=5, label='CPA (salt-free)')

            # Experimental scatter
            sub_val = results_df[np.abs(results_df['T_K'] - T_K) < 0.5]
            for ci, ref in enumerate(sub_val['reference'].unique()):
                grp  = sub_val[sub_val['reference'] == ref]
                mask = grp[exp_col].notna() & (grp[exp_col] > 0)
                if mask.sum() == 0:
                    continue
                ax.scatter(grp.loc[mask, 'P_bar'], grp.loc[mask, exp_col],
                           color=_COLORS[ci % len(_COLORS)],
                           s=30, zorder=6, label=_short_ref(ref))

            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_xlabel(r'$P$ [bar]', fontsize=15, fontweight='bold')
            ax.set_ylabel(ylabel,       fontsize=15, fontweight='bold')
            ax.tick_params(labelsize=13, which='both')
            ax.legend(fontsize=11, loc='best', framealpha=0.85)
            ax.grid(True, which='both', ls=':', alpha=0.35)

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=CMAP, norm=CNORM)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax, label=r'$m_s$ [mol kg$^{-1}$]')
        cb.ax.tick_params(labelsize=12)
        cb.ax.yaxis.label.set_size(13)
        cb.ax.yaxis.label.set_weight('bold')
        ms_ticks = [ms for ms in sorted_ms if ms > 0.01]
        cb.set_ticks(ms_ticks)
        cb.set_ticklabels([str(m) for m in ms_ticks])

        # Right panel: when y-values are all in the 0.1–1 range the default
        # log formatter produces "6×10⁻¹" labels that are too wide.
        # Switch to plain decimal ticks in that case.
        import matplotlib.ticker as mticker
        ylo, yhi = ax1.get_ylim()
        if ylo > 0.05:
            ax1.set_yticks([0.1, 0.2, 0.4, 0.6])
            ax1.yaxis.set_major_formatter(mticker.ScalarFormatter())
            ax1.yaxis.set_minor_formatter(mticker.NullFormatter())

        return fig

    print(f"\n=== Ribbon figures for {len(T_unique)} temperatures ===")
    for T_K in T_unique:
        fpath = f'figures/co2h2o_ws/T{int(T_K)}K.png'
        fig = plot_ribbon_T(T_K)
        fig.savefig(fpath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  {fpath}", flush=True)

    print("\nDone.")
