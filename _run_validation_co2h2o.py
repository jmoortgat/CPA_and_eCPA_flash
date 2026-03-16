import warnings; warnings.filterwarnings('ignore')

if __name__ == '__main__':
    import os
    import subprocess
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from ecpa.parameters import make_params
    from ecpa.solution_table import load_solution_table, make_solution_guess_fn
    from ecpa.validate_co2h2o import (
        run_validation, run_smooth_curves,
        compute_metrics, plot_validation_T,
        metrics_to_latex, print_perf_summary,
        MS_EVAL,
    )

    os.makedirs('results', exist_ok=True)
    os.makedirs('figures/co2h2o', exist_ok=True)

    # ── Load solution table ────────────────────────────────────────────────────
    print("Loading solution table …")
    grid_data = load_solution_table()
    solution_guess_fn = make_solution_guess_fn(grid_data)
    params = make_params()

    # ── Load experimental data ─────────────────────────────────────────────────
    exp_df = pd.read_parquet('CO2_WATER_exp.parquet')
    print(f"Loaded {len(exp_df)} experimental points  "
          f"T={exp_df['T_K'].min()}–{exp_df['T_K'].max()} K")
    print(f"  xc_W available: {exp_df['xc_W'].notna().sum()}  "
          f"yw_C available: {exp_df['yw_C'].notna().sum()}")

    # ── Validation at experimental conditions ──────────────────────────────────
    print("\nRunning CPA + eCPA at experimental conditions …")
    results = run_validation(
        exp_df           = exp_df,
        solution_guess_fn= solution_guess_fn,
        params           = params,
        ms               = MS_EVAL,
        verbose          = True,
    )
    results.to_parquet('results/validation_co2h2o.parquet', index=False)
    print(f"Saved results/validation_co2h2o.parquet  ({len(results)} rows)")

    # ── Smooth curves for plotting ─────────────────────────────────────────────
    # Use the 17 solution-table T values (where eCPA is available) plus any
    # additional experimental T values for CPA-only curves
    table_T_vals = sorted(grid_data['T_grid'])
    extra_T      = sorted(
        set(exp_df['T_K'].unique()) - set(int(t) for t in table_T_vals)
    )
    all_T_vals   = sorted(set(table_T_vals) | set(extra_T))

    print(f"\nRunning smooth curves at {len(all_T_vals)} temperatures "
          f"({len(all_T_vals)} × 100 P points) …")
    smooth = run_smooth_curves(
        T_vals            = all_T_vals,
        solution_guess_fn = solution_guess_fn,
        params            = params,
        ms                = MS_EVAL,
        n_P               = 100,
        verbose           = True,
    )
    smooth.to_parquet('results/smooth_co2h2o.parquet', index=False)
    print(f"Saved results/smooth_co2h2o.parquet  ({len(smooth)} rows)")

    # ── Performance summary ────────────────────────────────────────────────────
    print_perf_summary(results, smooth)

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = compute_metrics(results)
    metrics.to_csv('results/metrics_co2h2o.csv', index=False)

    print("\n── Overall metrics ────────────────────────────────────────────────")
    overall = metrics[metrics['T_K'] == 'All']
    for _, row in overall.iterrows():
        print(f"  {row['qty']:6s}  CPA: AARE={row['AARE_CPA']:.2f}%  "
              f"bias={row['bias_CPA']:+.2f}%  R²={row['R2_CPA']:.4f}  (n={row['n_cpa']:.0f})  |  "
              f"eCPA: AARE={row['AARE_eCPA']:.2f}%  "
              f"bias={row['bias_eCPA']:+.2f}%  R²={row['R2_eCPA']:.4f}  (n={row['n_ecpa']:.0f})  |  "
              f"CPA vs eCPA: AARE={row['AARE_CPA_eCPA']:.2f}%")

    # ── LaTeX table ───────────────────────────────────────────────────────────
    tex_path = 'results/co2h2o_metrics.tex'
    metrics_to_latex(metrics, path=tex_path, ms=MS_EVAL)

    # Try to compile to PDF
    try:
        result = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', '-output-directory=results',
             'results/co2h2o_metrics.tex'],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            print("Compiled results/co2h2o_metrics.pdf")
        else:
            print(f"pdflatex failed (code {result.returncode}); .tex saved.")
            if result.stdout:
                # Show first error line
                for line in result.stdout.splitlines():
                    if line.startswith('!'):
                        print(f"  LaTeX error: {line}")
                        break
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"PDF compilation skipped ({e}); .tex saved.")

    # ── Per-temperature plots ─────────────────────────────────────────────────
    print("\nGenerating per-temperature plots …")
    exp_T_vals = sorted(exp_df['T_K'].unique())
    saved_figs = []

    for T_K in exp_T_vals:
        fpath = f'figures/co2h2o/T{int(T_K)}K.png'
        fig = plot_validation_T(
            T_K        = T_K,
            results_df = results,
            smooth_df  = smooth,
            save_path  = fpath,
            ms         = MS_EVAL,
        )
        plt.close(fig)
        saved_figs.append(fpath)

    print(f"Saved {len(saved_figs)} figures to figures/co2h2o/")
    print("Done.")
