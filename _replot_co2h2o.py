"""
Re-generate per-temperature CO2-H2O figures using saved parquets.
Adds an 'eCPA (ms=0, ELV)' curve from CPA_ELV_all.parquet to each panel.
No flash calculations are re-run.
"""
import warnings; warnings.filterwarnings('ignore')

if __name__ == '__main__':
    import os
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from ecpa.validate_co2h2o import plot_validation_T, plot_co2h2o_parity, MS_EVAL

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

    print("\nRegenerating parity plot …")
    fig_p = plot_co2h2o_parity(
        results,
        save_path='figures/validation_parity.png',
    )
    plt.close(fig_p)
    print("Saved figures/validation_parity.png")

    print("Done.")
