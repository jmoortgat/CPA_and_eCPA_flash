import warnings; warnings.filterwarnings('ignore')

if __name__ == '__main__':
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from ecpa.parameters import make_params
    from ecpa.guess_table import load_cpa_guess_table, make_guess_fn
    from ecpa.validate_nacl import (
        load_co2nacl_exp, run_validation, compute_metrics, print_metrics,
        plot_validation_T, plot_validation_parity, plot_error_heatmap,
        plot_nacl_T_figures,
    )

    params = make_params()
    CPA_GROUPS, CPA_TEMPS = load_cpa_guess_table()
    guess_table_fn = make_guess_fn(CPA_GROUPS, CPA_TEMPS)

    # Load experimental data
    exp_df = load_co2nacl_exp()
    print(f"Loaded {len(exp_df)} experimental points")
    print(f"After filtering (T≤523K, ms≤3.5): "
          f"{len(exp_df[(exp_df.T_K<=523) & (exp_df.ms<=3.5)])} points")

    # Run validation
    results = run_validation(
        exp_df=exp_df,
        guess_table_fn=guess_table_fn,
        params=params,
        T_max=523.0,
        ms_max=3.5,
        n_workers=1,
        verbose=True,
    )

    results.to_parquet('results/validation_co2nacl.parquet', index=False)
    print("\nResults saved to results/validation_co2nacl.parquet")

    # Metrics
    metrics = compute_metrics(results)
    print_metrics(metrics)

    # Plots
    import os
    os.makedirs('figures', exist_ok=True)
    os.makedirs('figures/co2nacl', exist_ok=True)

    # Per-temperature figures (one file each, clean 2-panel layout)
    plot_nacl_T_figures(results, fig_dir='figures/co2nacl', T_max=523.0)

    # Overview / summary figures
    fig1 = plot_validation_T(results, save_path='figures/validation_T.png')
    plt.close(fig1)
    print("Saved figures/validation_T.png")

    fig2 = plot_validation_parity(results, save_path='figures/validation_parity.png')
    plt.close(fig2)
    print("Saved figures/validation_parity.png")

    fig3 = plot_error_heatmap(results, save_path='figures/validation_heatmap.png')
    plt.close(fig3)
    print("Saved figures/validation_heatmap.png")
