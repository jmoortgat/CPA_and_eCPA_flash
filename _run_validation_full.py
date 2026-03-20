"""
Full CO2+NaCl validation using the extended solution table.

Covers ALL experimental conditions:
  T = 278–723 K, P = 1–1400 bar, ms = 0.17–6.0 mol/kg

Strategy
--------
Uses flash_co2_h2o_salt_fast (warm-started SSI from the extended solution
table) for all points.  For conditions outside the solution table range the
function automatically falls back to cold-start SSI.

Output
------
  results/validation_co2nacl.parquet   — full results (replaces old file)
  figures/co2nacl/T*.png               — per-temperature VLE figures
  figures/validation_heatmap.png       — AARE heatmap (T × ms)
"""
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    import os
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from ecpa.parameters import make_params
    from ecpa.solution_table import load_solution_table, make_solution_guess_fn
    from ecpa.flash import flash_co2_h2o_salt_fast
    from ecpa.validate_nacl import (
        load_co2nacl_exp, compute_metrics, print_metrics,
        plot_nacl_T_figures, plot_error_heatmap,
    )

    Mw = 0.018015   # kg/mol

    # ── load model & solution table ───────────────────────────────────────────
    params = make_params()

    print("Loading solution table …")
    gd       = load_solution_table('results/solution_table.npz')
    guess_fn = make_solution_guess_fn(gd)
    print(f"  T range : {gd['T_grid'].min():.0f}–{gd['T_grid'].max():.0f} K")
    print(f"  ms range: {gd['ms_grid'].min():.1f}–{gd['ms_grid'].max():.1f} mol/kg")

    # ── experimental data ─────────────────────────────────────────────────────
    exp_df = load_co2nacl_exp(force_reparse=True)
    print(f"\nTotal experimental points : {len(exp_df)}")
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

    # ── run flash over all experimental points ────────────────────────────────
    rows  = exp_df.to_dict('records')
    results = []

    print(f"\nRunning flash on {len(rows)} points …")
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
        last_err   = ''

        for z in candidates:
            try:
                out = flash_co2_h2o_salt_fast(
                    T=T, P_bar=P, z_co2=z, m_tot=ms,
                    solution_guess_fn=guess_fn,
                    params=params,
                )
            except Exception as e:
                last_err = str(e)
                continue

            if out.get('phase') == 'single_phase':
                base['status'] = 'single_phase'
                continue

            ms_aq = float(out['ms_aq'])
            if ms_aq <= 0 or abs(ms_aq - ms) / max(ms, 0.1) > 0.6:
                last_err = f'ms_aq={ms_aq:.3f} far from ms={ms:.3f}'
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
                'n_iter_ms': out.get('n_iter_ms', -1),
            })
            break

        results.append(base)
        if (k + 1) % 100 == 0:
            ok_so_far = sum(r['status'] == 'ok' for r in results)
            print(f'  {k+1}/{len(rows)}  converged: {ok_so_far}', flush=True)

    res_df = pd.DataFrame(results)
    ok_n   = (res_df['status'] == 'ok').sum()
    print(f"\nTotal: {ok_n}/{len(res_df)} converged")

    # ── convergence summary by T ──────────────────────────────────────────────
    print("\nConvergence by T:")
    print(res_df.groupby('T_K')['status'].value_counts().to_string())

    # ── save results ──────────────────────────────────────────────────────────
    res_df.to_parquet('results/validation_co2nacl.parquet', index=False)
    print("\nSaved results/validation_co2nacl.parquet")

    # ── metrics ───────────────────────────────────────────────────────────────
    print("\n=== Metrics (all T, all ms) ===")
    metrics = compute_metrics(res_df)
    print_metrics(metrics)

    # Per-quantity breakdown
    ok_df = res_df[res_df['status'] == 'ok']
    print("\nPer-quantity metrics:")
    for qty, grp in ok_df.groupby('qty'):
        aare = grp['abs_rel_err'].mean() * 100
        bias = grp['rel_err'].mean() * 100
        print(f"  {qty:20s}  N={len(grp):4d}  AARE={aare:.1f}%  bias={bias:+.1f}%")

    print("\nPer-temperature metrics (converged points only):")
    for T, grp in ok_df.groupby('T_K'):
        aare = grp['abs_rel_err'].mean() * 100
        bias = grp['rel_err'].mean() * 100
        n    = len(grp)
        print(f"  T={T:.0f}K  N={n:4d}  AARE={aare:.1f}%  bias={bias:+.1f}%")

    # ── figures ───────────────────────────────────────────────────────────────
    os.makedirs('figures/co2nacl', exist_ok=True)

    plot_nacl_T_figures(
        ok_df,
        fig_dir='figures/co2nacl',
        T_max=730.0,
    )
    print("\nSaved per-temperature figures → figures/co2nacl/")

    fig = plot_error_heatmap(
        res_df, save_path='figures/validation_heatmap.png'
    )
    plt.close(fig)
    print("Saved figures/validation_heatmap.png")

    print("\nDone.")
