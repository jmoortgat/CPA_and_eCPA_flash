"""
Extend CO2+NaCl validation to high temperatures (T > 523 K).

Strategy:
  - T <= 623 K, ms <= 3.0  : warm-start SSI from solution table
  - T <= 623 K, ms >  3.0  : cold-start SSI with Wilson K-values init
  - T >  623 K             : cold-start SSI (solution table extrapolates to 623 K boundary)

Merges results with the existing validation_co2nacl.parquet (T<=523K)
and saves combined results + new per-T figures.
"""
import warnings; warnings.filterwarnings('ignore')

if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import os

    from ecpa.parameters import make_params
    from ecpa.solution_table import make_solution_guess_fn
    from ecpa.flash import flash_co2_h2o_salt_ssi, flash_co2_h2o_salt_fast
    from ecpa.guess_table import load_cpa_guess_table, make_guess_fn
    from ecpa.validate_nacl import (
        load_co2nacl_exp, compute_metrics, print_metrics, plot_nacl_T_figures,
        plot_error_heatmap,
    )

    # ── load model & tables ──────────────────────────────────────────────────
    params = make_params()

    npz      = np.load('results/solution_table.npz')
    guess_fn = make_solution_guess_fn(dict(npz))   # warm start (T<=623K, ms<=3)

    CPA_GROUPS, CPA_TEMPS = load_cpa_guess_table()
    cpa_guess_fn = make_guess_fn(CPA_GROUPS, CPA_TEMPS)   # binary fallback (T<=533K)

    # ── experimental data at T > 523 K ──────────────────────────────────────
    exp_df = load_co2nacl_exp()
    high_T = exp_df[exp_df['T_K'] > 523.0].copy()
    print(f"High-T points (T > 523 K): {len(high_T)}")
    print(high_T.groupby(['T_K','ms'])['T_K'].count().rename('n').reset_index().to_string())

    TABLE_T_MAX = 623.0
    TABLE_MS_MAX = 3.0
    Mw = 0.018015  # kg/mol

    def _z_candidates(qty, val_exp):
        """Same logic as in validate_nacl._z_candidates."""
        if qty == 'mc':
            # convert mc [mol/kg] to approximate z_co2
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

    def flash_one(T, P, ms, params, guess_fn, cpa_guess_fn):
        """Try warm-start flash, fall back to cold SSI."""
        use_warm = (T <= TABLE_T_MAX) and (ms <= TABLE_MS_MAX)
        if use_warm:
            # Use solution table warm start at a representative z
            def _ssi(z):
                return flash_co2_h2o_salt_ssi(
                    T=T, P_bar=P, z_co2=z, m_tot=ms,
                    params=params,
                    maxiter_ms=50,
                    guess_table_fn=None,
                    initial_sol=None,
                    initial_ms_aq=None,
                )
            # Actually use guess_fn to get warm start, then pass to ssi
            def _ssi_warm(z):
                sol0, ms0, _ = guess_fn(T, P, z, ms)
                return flash_co2_h2o_salt_ssi(
                    T=T, P_bar=P, z_co2=z, m_tot=ms,
                    params=params,
                    maxiter_ms=50,
                    initial_sol=sol0,
                    initial_ms_aq=ms0,
                )
            return _ssi_warm
        else:
            # Cold start with binary guess table (clamped to max T=533K internally)
            def _ssi_cold(z):
                return flash_co2_h2o_salt_ssi(
                    T=T, P_bar=P, z_co2=z, m_tot=ms,
                    params=params,
                    maxiter_ms=50,
                    guess_table_fn=cpa_guess_fn,
                )
            return _ssi_cold

    results = []
    for k, row in enumerate(high_T.to_dict('records')):
        T   = float(row['T_K'])
        P   = float(row['P_bar'])
        ms  = float(row['ms'])
        qty = row['qty']
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

        ssi_fn = flash_one(T, P, ms, params, guess_fn, cpa_guess_fn)
        candidates = _z_candidates(qty, val_exp)
        last_err = ''

        for z in candidates:
            try:
                out = ssi_fn(z)
            except Exception as e:
                last_err = str(e)
                continue

            ms_aq = float(out['ms_aq'])
            if ms_aq <= 0 or abs(ms_aq - ms) / ms > 0.5:
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
        if (k + 1) % 10 == 0:
            print(f'  {k+1}/{len(high_T)}', flush=True)

    new_df = pd.DataFrame(results)
    ok = (new_df['status'] == 'ok').sum()
    print(f"\nHigh-T results: {ok}/{len(new_df)} converged")
    print(new_df.groupby('T_K')['status'].value_counts().to_string())

    # ── merge with existing T<=523K results ─────────────────────────────────
    old_df = pd.read_parquet('results/validation_co2nacl.parquet')
    combined = pd.concat([old_df, new_df], ignore_index=True)
    combined = combined.sort_values(['T_K', 'ms', 'P_bar']).reset_index(drop=True)
    combined.to_parquet('results/validation_co2nacl_extended.parquet', index=False)
    print(f"\nSaved {len(combined)} rows to results/validation_co2nacl_extended.parquet")

    # ── metrics ─────────────────────────────────────────────────────────────
    print("\n=== Metrics (all T) ===")
    metrics_all = compute_metrics(combined)
    print_metrics(metrics_all)

    print("\n=== Metrics (T > 523 K only) ===")
    metrics_highT = compute_metrics(new_df)
    print_metrics(metrics_highT)

    # ── figures for new high-T panels ───────────────────────────────────────
    os.makedirs('figures/co2nacl', exist_ok=True)
    plot_nacl_T_figures(
        new_df[new_df['status'] == 'ok'],
        fig_dir='figures/co2nacl',
        T_max=730.0,
    )
    print("Saved high-T per-temperature figures to figures/co2nacl/")

    # Updated heatmap with all T
    fig = plot_error_heatmap(
        combined, save_path='figures/validation_heatmap_extended.png'
    )
    plt.close(fig)
    print("Saved figures/validation_heatmap_extended.png")
