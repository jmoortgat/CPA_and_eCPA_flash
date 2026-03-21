"""
Compare CPA flash performance: with vs without solution-table warm-start.

Method A — CPA (no table):
    CPA2.flash_co2_h2o_tpz_robust — 6-trial Michelsen stability +
    K-value SSI; robust but expensive (~200–300 ms/call).
    Results already in results/ws_cpa_smooth.parquet.

Method B — CPA/eCPA (table warm-start, ms=0):
    flash_co2_h2o_salt_fast_kv at ms→0, table queried at ms=0.
    Single-guess K-value SSI warm-started from the solution table;
    no stability test when the table says two-phase.

Same (T, P, z) grid as the smooth curves: 35 T × 200 P = 7000 conditions.

Output:
    results/ws_cpa_warmstart.parquet    — Method B results with timing
    figures/co2h2o_ws/cpa_warmstart_comparison.png
"""
import warnings
warnings.filterwarnings('ignore')

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

N_WORKERS = 8
Z_CO2     = 0.5
MS_TINY   = 1e-9    # effectively ms=0; table queried at ms=0 boundary


# ── Worker: K-value SSI with ms=0 table warm-start ───────────────────────────

def _kv_T_worker(args):
    import warnings; warnings.filterwarnings('ignore')
    import time as _time
    T, P_grid, grid_data = args
    from ecpa.solution_table import make_solution_guess_fn
    from ecpa.parameters import make_params
    from ecpa.flash import flash_co2_h2o_salt_fast_kv
    from ecpa.validate_co2h2o import Z_CO2_RETRY

    guess_fn = make_solution_guess_fn(grid_data)
    params   = make_params()
    Z_CANDS  = [Z_CO2] + list(Z_CO2_RETRY)

    records = []
    for P in P_grid:
        t0  = _time.perf_counter()
        out = None
        for z in Z_CANDS:
            try:
                r = flash_co2_h2o_salt_fast_kv(
                    T=T, P_bar=P, z_co2=z, m_tot=MS_TINY,
                    solution_guess_fn=guess_fn, params=params,
                )
                if r.get('phase') == 'two_phase':
                    out = r
                    break
                elif r.get('phase') == 'single_phase':
                    # single-phase confirmed — stop trying
                    out = r
                    break
            except Exception:
                continue
        t_ms = (_time.perf_counter() - t0) * 1e3

        if out is not None and out.get('phase') == 'two_phase':
            x_aq = out['x_aq']
            x_c  = out['x_c']
            records.append({
                'T_K': float(T), 'P_bar': float(P),
                'phase':       'two_phase',
                'kv_xc_W':    float(x_aq['x4w']),
                'kv_yw_C':    float(x_c['x1c']),
                'kv_n_iter':  int(out.get('n_iter_ms', -1)),
                'kv_t_ms':    t_ms,
                'kv_stab':    out.get('stable') is not None,
            })
        else:
            records.append({
                'T_K': float(T), 'P_bar': float(P),
                'phase':      'single_phase' if out else 'failed',
                'kv_xc_W':   np.nan,
                'kv_yw_C':   np.nan,
                'kv_n_iter': -1,
                'kv_t_ms':   t_ms,
                'kv_stab':   False,
            })
    return T, records


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures/co2h2o_ws', exist_ok=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from ecpa.solution_table import load_solution_table, make_solution_guess_fn
    from ecpa.validate_co2h2o import T_MAX_ECPA

    print("Loading solution table …")
    grid_data = load_solution_table()

    # Reference grid: same T, P as smooth curves
    cpa_df = pd.read_parquet('results/ws_cpa_smooth.parquet')
    T_vals = sorted(cpa_df['T_K'].unique())
    P_vals = sorted(cpa_df['P_bar'].unique())
    P_grid = np.array(P_vals)
    print(f"Grid: {len(T_vals)} T × {len(P_grid)} P = {len(T_vals)*len(P_grid)} conditions")

    # ── Run Method B ──────────────────────────────────────────────────────────
    cache = Path('results/ws_cpa_warmstart.parquet')
    if cache.exists():
        print("Loading cached Method B results …")
        kv_df = pd.read_parquet(cache)
    else:
        print(f"\n=== K-value SSI (table warm-start, ms=0)  [{N_WORKERS} workers] ===")
        t_total = time.perf_counter()
        args_list = [(T, P_grid, grid_data) for T in T_vals]
        rows = []
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            futs = {pool.submit(_kv_T_worker, a): a[0] for a in args_list}
            done = 0
            for fut in as_completed(futs):
                T_done, recs = fut.result()
                rows.extend(recs)
                done += 1
                if done % 5 == 0 or done == len(T_vals):
                    print(f"  [{done}/{len(T_vals)}]  T={T_done:.0f}K", flush=True)
        kv_df = pd.DataFrame(rows).sort_values(['T_K', 'P_bar'])
        kv_df.to_parquet(cache, index=False)
        elapsed = time.perf_counter() - t_total
        print(f"  → saved {cache}  (total {elapsed:.1f}s)")

    # ── Merge and compare ─────────────────────────────────────────────────────
    merged = cpa_df.merge(kv_df, on=['T_K', 'P_bar'], how='inner')

    A_2ph = merged['cpa_xc_W'].notna()   # Method A two-phase
    B_2ph = merged['kv_xc_W'].notna()    # Method B two-phase
    both  = A_2ph & B_2ph

    print(f"\n{'='*60}")
    print(f"{'Method':<35}  {'two-phase %':>11}  {'t_ms mean':>9}  {'t_ms median':>11}  {'iters median':>12}")
    print(f"{'-'*60}")
    print(f"{'A: CPA2 robust (no table)':<35}  "
          f"{A_2ph.mean()*100:>10.1f}%  "
          f"{merged['cpa_t_ms'].mean():>9.1f}  "
          f"{merged['cpa_t_ms'].median():>11.1f}  "
          f"{merged.loc[A_2ph,'cpa_n_iter'].median():>12.1f}")
    print(f"{'B: K-value SSI (table ms=0)':<35}  "
          f"{B_2ph.mean()*100:>10.1f}%  "
          f"{merged['kv_t_ms'].mean():>9.1f}  "
          f"{merged['kv_t_ms'].median():>11.1f}  "
          f"{merged.loc[B_2ph,'kv_n_iter'].median():>12.1f}")
    print(f"{'='*60}")
    print(f"Speed-up (mean):   {merged['cpa_t_ms'].mean() / merged['kv_t_ms'].mean():.1f}×")
    print(f"Speed-up (median): {merged['cpa_t_ms'].median() / merged['kv_t_ms'].median():.1f}×")

    # Agreement between methods (where both find two-phase)
    xc_err = np.abs(merged.loc[both,'cpa_xc_W'] - merged.loc[both,'kv_xc_W']) \
             / merged.loc[both,'cpa_xc_W']
    yw_err = np.abs(merged.loc[both,'cpa_yw_C'] - merged.loc[both,'kv_yw_C']) \
             / merged.loc[both,'cpa_yw_C']
    print(f"\nComposition agreement (CPA vs table-KV, two-phase only):")
    print(f"  xc_W AARE = {xc_err.mean()*100:.3f}%  max = {xc_err.max()*100:.3f}%")
    print(f"  yw_C AARE = {yw_err.mean()*100:.3f}%  max = {yw_err.max()*100:.3f}%")

    print(f"\nK-value SSI iteration distribution (Method B, two-phase):")
    iters_B = merged.loc[B_2ph, 'kv_n_iter']
    for pct in [50, 75, 90, 95, 99]:
        print(f"  p{pct:02d} = {np.percentile(iters_B, pct):.0f}")
    print(f"  max = {iters_B.max():.0f}")
    print(f"  stability test run: {merged.loc[B_2ph,'kv_stab'].mean()*100:.1f}% of two-phase calls")

    # ── Figure ────────────────────────────────────────────────────────────────
    try:
        import scienceplots; plt.style.use(['science'])
    except ImportError:
        pass
    plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 300,
                         'savefig.bbox': 'tight'})

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2),
                             gridspec_kw=dict(left=0.08, right=0.97,
                                              top=0.88, bottom=0.15,
                                              wspace=0.35))

    # Panel 1: wall-time distributions
    ax = axes[0]
    bins_t = np.logspace(np.log10(1), np.log10(5000), 60)
    ax.hist(merged['cpa_t_ms'], bins=bins_t, density=True,
            color='steelblue', alpha=0.65, label='CPA2 robust')
    ax.hist(merged['kv_t_ms'],  bins=bins_t, density=True,
            color='tomato',    alpha=0.65, label='KV (table)')
    ax.axvline(merged['cpa_t_ms'].median(), color='steelblue', lw=1.2, ls='--')
    ax.axvline(merged['kv_t_ms'].median(),  color='tomato',    lw=1.2, ls='--')
    ax.set_xscale('log')
    ax.set_xlabel('Wall time per call (ms)', fontsize=9)
    ax.set_ylabel('Probability density', fontsize=9)
    ax.legend(fontsize=8, framealpha=0)
    ax.set_title('Timing', fontsize=9)

    # Panel 2: SSI/flash iterations
    ax = axes[1]
    iters_A = merged.loc[A_2ph, 'cpa_n_iter']
    iters_B = merged.loc[B_2ph, 'kv_n_iter']
    bins_i = np.arange(0.5, max(iters_A.max(), iters_B.max()) + 1.5)
    ax.hist(iters_A, bins=bins_i, density=True,
            color='steelblue', alpha=0.65, label='CPA2 robust')
    ax.hist(iters_B, bins=bins_i, density=True,
            color='tomato',    alpha=0.65, label='KV (table)')
    ax.set_xlabel('Iterations per call', fontsize=9)
    ax.set_ylabel('Probability density', fontsize=9)
    ax.legend(fontsize=8, framealpha=0)
    ax.set_title('Iterations', fontsize=9)
    ax.set_xlim(0, min(50, bins_i[-1]))

    # Panel 3: composition agreement
    ax = axes[2]
    ax.scatter(merged.loc[both, 'cpa_xc_W'],
               merged.loc[both, 'kv_xc_W'],
               s=2, alpha=0.3, color='steelblue', label=r'$x_{\rm CO_2}$ (aq)')
    ax.scatter(merged.loc[both, 'cpa_yw_C'],
               merged.loc[both, 'kv_yw_C'],
               s=2, alpha=0.3, color='tomato',    label=r'$y_{\rm H_2O}$ (CO$_2$-rich)')
    lims = [1e-5, 1]
    ax.plot(lims, lims, 'k-', lw=0.8, zorder=5)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(*lims); ax.set_ylim(*lims)
    ax.set_xlabel('CPA2 robust', fontsize=9)
    ax.set_ylabel('KV (table warm-start)', fontsize=9)
    ax.legend(fontsize=7, framealpha=0, markerscale=3)
    ax.set_title('Composition agreement', fontsize=9)

    fig.suptitle('CPA flash: CPA2 robust vs K-value SSI with table warm-start  '
                 r'(35 T $\times$ 200 P, $z_{\rm CO_2}=0.5$)', fontsize=9)

    outpath = 'figures/co2h2o_ws/cpa_warmstart_comparison.png'
    fig.savefig(outpath)
    plt.close(fig)
    print(f"\nFigure saved: {outpath}")
    print("Done.")
