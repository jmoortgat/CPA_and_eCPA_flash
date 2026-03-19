"""
Re-generate per-temperature CO2-H2O figures using saved parquets.
Adds an 'eCPA (ms=0, ELV)' curve from CPA_ELV_all.parquet to each panel.

Optionally re-runs smooth curves at temperatures with incomplete eCPA coverage
(pass --resmooth to enable; requires solution table).
"""
import warnings; warnings.filterwarnings('ignore')

if __name__ == '__main__':
    import os
    import sys
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from ecpa.validate_co2h2o import (
        plot_validation_T, plot_co2h2o_parity, plot_error_heatmap,
        flag_outliers, compute_regime_metrics, regime_metrics_to_latex,
        run_smooth_curves, MS_EVAL,
    )

    os.makedirs('figures/co2h2o', exist_ok=True)

    print("Loading saved parquets …")
    results = pd.read_parquet('results/validation_co2h2o.parquet')
    smooth  = pd.read_parquet('results/smooth_co2h2o.parquet')
    elv_df  = pd.read_parquet('CPA_ELV_all.parquet')

    print(f"  results: {len(results)} rows")
    print(f"  smooth : {len(smooth)} rows")
    print(f"  ELV    : {len(elv_df)} rows  "
          f"T={elv_df['T_K'].min():.0f}–{elv_df['T_K'].max():.0f} K")

    exp_df = pd.read_parquet('CO2_WATER_exp.parquet')
    exp_T_vals = sorted(exp_df['T_K'].unique())

    # ── Optional: re-run smooth curves at temperatures with low eCPA coverage ──
    if '--resmooth' in sys.argv:
        from ecpa.parameters import make_params
        from ecpa.solution_table import load_solution_table, make_solution_guess_fn

        params = make_params()
        grid_data = load_solution_table()
        solution_guess_fn = make_solution_guess_fn(grid_data)

        # Identify temperatures with <50% eCPA coverage
        redo_T = []
        for T in smooth['T_K'].unique():
            sub = smooth[np.abs(smooth['T_K'] - T) < 0.5]
            if sub['ecpa_xc_W'].notna().mean() < 0.5:
                redo_T.append(float(T))
        if redo_T:
            print(f"\nRe-running smooth curves at {len(redo_T)} temperatures "
                  f"with low eCPA coverage …")
            new_smooth = run_smooth_curves(
                redo_T, solution_guess_fn, params,
                ms=MS_EVAL, n_P=100, verbose=True)
            # Replace old rows
            keep = smooth[~smooth['T_K'].isin(redo_T)]
            smooth = pd.concat([keep, new_smooth], ignore_index=True)
            smooth.to_parquet('results/smooth_co2h2o.parquet', index=False)
            print(f"Updated smooth_co2h2o.parquet  ({len(smooth)} rows)")
        else:
            print("\nAll smooth curves have ≥50% eCPA coverage — skipping resmooth.")

    # ── Outlier flagging ────────────────────────────────────────────────────────
    print("\nFlagging outliers …")
    results = flag_outliers(results)
    n_out_xc = results['outlier_xc'].sum()
    n_out_yw = results['outlier_yw'].sum()
    print(f"  Outlier xc_W: {n_out_xc} points")
    print(f"  Outlier yw_C: {n_out_yw} points")

    # ── Regime metrics ──────────────────────────────────────────────────────────
    print("\nRegime-specific metrics:")
    regime_rows = compute_regime_metrics(results)
    for row in regime_rows:
        a_xc = row['AARE_xc_W']
        a_yw = row['AARE_yw_C']
        a_xc_s = f'{a_xc:.1f}%' if np.isfinite(a_xc) else '---'
        a_yw_s = f'{a_yw:.1f}%' if np.isfinite(a_yw) else '---'
        print(f"  {row['regime']:40s}  xc_W: N={row['N_xc_W']:>3d} AARE={a_xc_s:>6s}  "
              f"yw_C: N={row['N_yw_C']:>3d} AARE={a_yw_s:>6s}")

    tex = regime_metrics_to_latex(regime_rows, path='results/co2h2o_regime_metrics.tex')
    print("  Wrote results/co2h2o_regime_metrics.tex")

    # ── Per-temperature figures ──────────────────────────────────────────────────
    print(f"\nRegenerating {len(exp_T_vals)} figures …")
    for T_K in exp_T_vals:
        fpath = f'figures/co2h2o/T{int(T_K)}K.png'
        fig = plot_validation_T(
            T_K        = T_K,
            results_df = results,
            smooth_df  = smooth,
            save_path  = fpath,
            ms         = MS_EVAL,
            elv_df     = elv_df,
        )
        plt.close(fig)
        print(f"  Saved {fpath}")

    # ── Parity plot ─────────────────────────────────────────────────────────────
    print("\nRegenerating parity plot …")
    fig_p = plot_co2h2o_parity(
        results,
        save_path='figures/validation_parity.png',
    )
    plt.close(fig_p)
    print("Saved figures/validation_parity.png")

    # ── Error heatmap (for SI) ──────────────────────────────────────────────────
    print("\nGenerating error heatmap …")
    fig_h = plot_error_heatmap(
        results,
        save_path='figures/co2h2o/error_heatmap.png',
    )
    plt.close(fig_h)
    print("Saved figures/co2h2o/error_heatmap.png")

    print("\nDone.")
