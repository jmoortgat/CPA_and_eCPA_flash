"""
CO2-NaCl warm-start validation using K-value SSI (flash_co2_h2o_salt_fast_kv).

Replaces the ms_aq outer-loop SSI (flash_co2_h2o_salt_fast) used in
_run_validation_full.py with the K-value SSI solver that uses the solution
table for warm-starting both the K-values and the inner Newton solvers.

Tracks for every flash call:
  - n_iter_ms  : K-value SSI outer iterations (typically 3–5 with warm start)
  - converged  : whether the flash converged (expected 100%)

Output
------
  results/ws_validation_co2nacl.parquet   — full results with iteration stats
  figures/co2nacl_ws/T*.png               — per-temperature VLE figures
  figures/co2nacl_ws/iter_hist.png        — SSI iteration histogram
"""
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    import os
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from ecpa.parameters import make_params
    from ecpa.solution_table import load_solution_table, make_solution_guess_fn
    from ecpa.flash import flash_co2_h2o_salt_fast_kv
    from ecpa.validate_nacl import (
        load_co2nacl_exp, compute_metrics, print_metrics,
        plot_nacl_T_figures,
    )
    from ecpa.constants import Mw

    os.makedirs('results', exist_ok=True)
    os.makedirs('figures/co2nacl_ws', exist_ok=True)

    # ── Model + solution table ─────────────────────────────────────────────────
    params = make_params()

    print("Loading solution table …")
    gd       = load_solution_table('results/solution_table.npz')
    guess_fn = make_solution_guess_fn(gd)
    print(f"  T : {gd['T_grid'].min():.0f}–{gd['T_grid'].max():.0f} K")
    print(f"  ms: {gd['ms_grid'].min():.1f}–{gd['ms_grid'].max():.1f} mol/kg")

    # ── Experimental data ──────────────────────────────────────────────────────
    exp_df = load_co2nacl_exp(force_reparse=True)
    print(f"\nTotal experimental points: {len(exp_df)}")
    print(exp_df.groupby('T_K')['ms'].agg(['min', 'max', 'count'])
          .rename(columns={'min': 'ms_min', 'max': 'ms_max', 'count': 'n'})
          .to_string())

    # ── z candidates for each measurement type ────────────────────────────────
    def _z_candidates(qty, val_exp):
        if qty == 'mc':
            x4w = val_exp * Mw / (1.0 + val_exp * Mw)
            z_mid = float(np.clip(x4w * 1.5, 0.05, 0.85))
            return [z_mid, 0.3, 0.5, 0.15, 0.65, 0.8]
        elif qty in ('xc_W_SALTfree', 'xc_W_SALTincl'):
            x4w = float(val_exp)
            z_mid = float(np.clip(x4w * 1.5, 0.05, 0.85))
            return [z_mid, 0.3, 0.5, 0.15, 0.65]
        elif qty == 'xc_C':
            return [0.7, 0.5, 0.85, 0.3]
        return [0.3, 0.5, 0.15, 0.65]

    def _pred_to_unit(qty, x_aq, x_c):
        x1w = x_aq['x1w']
        x4w = x_aq['x4w']
        x1c = x_c['x1c']
        if qty == 'mc':
            return x4w / (x1w * Mw) if x1w > 0 else np.nan
        elif qty == 'xc_W_SALTincl':
            return x4w
        elif qty == 'xc_W_SALTfree':
            denom = x4w + x1w
            return x4w / denom if denom > 0 else np.nan
        elif qty == 'xc_C':
            return 1.0 - x1c
        return np.nan

    # ── Flash loop ────────────────────────────────────────────────────────────
    rows    = exp_df.to_dict('records')
    results = []

    print(f"\nRunning K-value SSI flash on {len(rows)} points …")
    n_converged = 0

    for k, row in enumerate(rows):
        T       = float(row['T_K'])
        P       = float(row['P_bar'])
        ms      = float(row['ms'])
        qty     = row['qty']
        val_exp = float(row['value'])

        base = {
            'T_K': T, 'P_bar': P, 'ms': ms, 'qty': qty,
            'value_exp': val_exp, 'value_pred': np.nan,
            'rel_err': np.nan, 'abs_rel_err': np.nan,
            'status': 'flash_failed', 'z_co2_used': np.nan,
            'ms_aq_pred': np.nan, 'x1w_pred': np.nan,
            'x1c_pred': np.nan, 'n_iter_ms': -1,
            'reference': row.get('reference', ''),
            'source_file': row.get('source_file', ''),
            'exp_id': row.get('exp_id', k),
        }

        candidates = _z_candidates(qty, val_exp)

        for z in candidates:
            try:
                out = flash_co2_h2o_salt_fast_kv(
                    T=T, P_bar=P, z_co2=z, m_tot=ms,
                    solution_guess_fn=guess_fn,
                    params=params,
                )
            except Exception as e:
                continue

            if out.get('phase') == 'single_phase':
                base['status'] = 'single_phase'
                continue

            ms_aq = float(out['ms_aq'])
            if ms_aq <= 0 or abs(ms_aq - ms) / max(ms, 0.1) > 0.6:
                continue

            x_aq = out['x_aq']
            x_c  = out['x_c']
            if x_aq['x1w'] <= 0:
                base['status'] = 'nonphysical'
                continue

            pred = _pred_to_unit(qty, x_aq, x_c)
            if pred is None or not np.isfinite(pred) or pred <= 0:
                base['status'] = 'nonphysical'
                continue

            rel_err = (pred - val_exp) / val_exp
            base.update({
                'value_pred': pred, 'rel_err': rel_err,
                'abs_rel_err': abs(rel_err), 'status': 'ok',
                'z_co2_used': z, 'ms_aq_pred': ms_aq,
                'x1w_pred': x_aq['x1w'], 'x1c_pred': x_c['x1c'],
                'n_iter_ms': int(out.get('n_iter_ms', -1)),
            })
            n_converged += 1
            break

        results.append(base)
        if (k + 1) % 100 == 0:
            ok_so_far = sum(r['status'] == 'ok' for r in results)
            print(f'  {k+1}/{len(rows)}  converged: {ok_so_far}', flush=True)

    res_df = pd.DataFrame(results)
    ok_n   = (res_df['status'] == 'ok').sum()
    print(f"\nTotal: {ok_n}/{len(res_df)} two-phase converged  "
          f"({100*ok_n/len(res_df):.1f}%)")
    print("\nStatus breakdown:")
    print(res_df['status'].value_counts().to_string())

    # ── Convergence by temperature ─────────────────────────────────────────────
    print("\nConvergence by T:")
    print(res_df.groupby('T_K')['status'].value_counts().to_string())

    # ── SSI iteration statistics ───────────────────────────────────────────────
    ok_df    = res_df[res_df['status'] == 'ok']
    n_iter_s = ok_df['n_iter_ms']
    print(f"\nK-value SSI iterations (converged two-phase points):")
    print(f"  min    = {n_iter_s.min():.0f}")
    print(f"  median = {n_iter_s.median():.1f}")
    print(f"  mean   = {n_iter_s.mean():.2f}")
    print(f"  max    = {n_iter_s.max():.0f}")
    print(f"  frac<=5 = {(n_iter_s <= 5).mean()*100:.1f}%")
    print(f"  frac<=8 = {(n_iter_s <= 8).mean()*100:.1f}%")

    # ── Save results ───────────────────────────────────────────────────────────
    res_df.to_parquet('results/ws_validation_co2nacl.parquet', index=False)
    print("\nSaved results/ws_validation_co2nacl.parquet")

    # ── Metrics ───────────────────────────────────────────────────────────────
    print("\n=== Metrics (all T, all ms) ===")
    metrics = compute_metrics(res_df)
    print_metrics(metrics)

    print("\nPer-quantity metrics:")
    for qty, grp in ok_df.groupby('qty'):
        aare = grp['abs_rel_err'].mean() * 100
        bias = grp['rel_err'].mean() * 100
        print(f"  {qty:20s}  N={len(grp):4d}  AARE={aare:.1f}%  bias={bias:+.1f}%")

    print("\nPer-temperature metrics:")
    for T, grp in ok_df.groupby('T_K'):
        aare = grp['abs_rel_err'].mean() * 100
        bias = grp['rel_err'].mean() * 100
        iters_med = grp['n_iter_ms'].median()
        print(f"  T={T:.0f}K  N={len(grp):4d}  AARE={aare:.1f}%  "
              f"bias={bias:+.1f}%  SSI_median={iters_med:.1f}")

    # ── VLE figures ────────────────────────────────────────────────────────────
    from _run_smooth_co2h2o_robust import _ms_tag, MS_RIBBON
    smooth_data = {}
    for ms in MS_RIBBON:
        cache = Path(f'results/ws2_smooth_co2h2o_ms{_ms_tag(ms)}.parquet')
        if cache.exists():
            smooth_data[ms] = pd.read_parquet(cache)

    plot_nacl_T_figures(
        res_df,
        fig_dir='figures/co2nacl_ws',
        T_max=730.0,
        smooth_data=smooth_data if smooth_data else None,
    )
    print("\nSaved per-temperature figures → figures/co2nacl_ws/")

    # ── SSI iteration histogram ────────────────────────────────────────────────
    try:
        import scienceplots  # noqa: F401
        plt.style.use(['science'])
    except ImportError:
        pass

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bins = np.arange(0.5, n_iter_s.max() + 1.5, 1.0)
    ax.hist(n_iter_s, bins=bins, density=True, color='steelblue', alpha=0.75,
            edgecolor='white', linewidth=0.4)
    ax.axvline(n_iter_s.median(), color='navy', lw=1.5, ls='--',
               label=rf'Median = {n_iter_s.median():.0f}')
    ax.set_xlabel('K-value SSI iterations per flash', fontsize=11)
    ax.set_ylabel('Probability density', fontsize=11)
    ax.set_title(f'CO$_2$–NaCl flash: SSI iterations  (N={len(ok_df)})',
                 fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', ls=':', alpha=0.4)
    fig.tight_layout()
    fig.savefig('figures/co2nacl_ws/iter_hist.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved figures/co2nacl_ws/iter_hist.png")

    print("\nDone.")
