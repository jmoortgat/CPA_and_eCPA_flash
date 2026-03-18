"""
Regenerate CO2-H2O per-temperature figures with rainbow salinity bands.

For each temperature in the experimental dataset, generates a two-panel figure
(x_CO2 left, y_H2O right) with:
  - Open symbols: experimental data (coloured by reference)
  - Black solid line: CPA (salt-free)
  - Rainbow dashed lines: eCPA at ms = 0, 1, 2, 3, 4, 5, 6 mol/kg
  - Colorbar: ms 0 → 6 mol/kg

Smooth curves for ms = 0, 1, 2 are loaded from existing parquets.
Curves for ms = 3, 4, 5, 6 are computed on first run and cached to parquet.
"""
import warnings; warnings.filterwarnings('ignore')

if __name__ == '__main__':
    import os
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from ecpa.parameters import make_params
    from ecpa.solution_table import load_solution_table, make_solution_guess_fn
    from ecpa.validate_co2h2o import run_smooth_curves, plot_validation_T, MS_EVAL

    os.makedirs('results', exist_ok=True)
    os.makedirs('figures/co2h2o', exist_ok=True)

    # ── Load table + params ────────────────────────────────────────────────────
    print("Loading solution table …")
    grid_data = load_solution_table()
    solution_guess_fn = make_solution_guess_fn(grid_data)
    params = make_params()

    # ── Load existing smooth/results parquets ──────────────────────────────────
    print("Loading existing parquets …")
    results  = pd.read_parquet('results/validation_co2h2o.parquet')
    smooth0  = pd.read_parquet('results/smooth_co2h2o.parquet')    # ms ≈ 1e-4
    exp_df   = pd.read_parquet('CO2_WATER_exp.parquet')

    # T values with solution-table coverage (≤ T_MAX_ECPA)
    from ecpa.validate_co2h2o import T_MAX_ECPA
    curve_T_vals = sorted(
        T for T in smooth0['T_K'].unique() if T >= 273.0 and T <= T_MAX_ECPA
    )

    # ── Build / load rainbow smooth curves for ms = 0, 1, 2, 3, 4, 5, 6 ──────
    # ms = 0  →  reuse smooth0 (ms ≈ 1e-4, effectively salt-free)
    MS_RAINBOW = [0, 1, 2, 3, 4, 5, 6]
    rainbow_data = {0: smooth0}    # ms_val → DataFrame

    for ms_val in MS_RAINBOW[1:]:   # skip ms=0, already loaded
        cache = Path(f'results/smooth_co2h2o_ms{ms_val:.1f}.parquet')
        if cache.exists():
            print(f"Loading cached smooth curves  ms={ms_val} …")
            rainbow_data[ms_val] = pd.read_parquet(cache)
        else:
            print(f"Computing smooth curves  ms={ms_val} mol/kg …")
            df = run_smooth_curves(
                T_vals            = curve_T_vals,
                solution_guess_fn = solution_guess_fn,
                params            = params,
                ms                = float(ms_val),
                n_P               = 100,
                verbose           = True,
            )
            df.to_parquet(cache, index=False)
            print(f"  Saved {cache}  ({len(df)} rows)")
            rainbow_data[ms_val] = df

    # ── Generate figures ───────────────────────────────────────────────────────
    ms_rainbow_list = [(mv, rainbow_data[mv]) for mv in MS_RAINBOW]

    exp_T_vals = sorted(exp_df['T_K'].unique())
    print(f"\nGenerating {len(exp_T_vals)} rainbow figures …")

    for T_K in exp_T_vals:
        fpath = f'figures/co2h2o/T{int(T_K)}K.png'
        # Only include rainbow curves for T within solution-table range
        sc = ms_rainbow_list if T_K <= T_MAX_ECPA else None
        fig = plot_validation_T(
            T_K          = T_K,
            results_df   = results,
            smooth_df    = smooth0,
            save_path    = fpath,
            ms           = MS_EVAL,
            salty_curves = sc,
        )
        plt.close(fig)
        print(f"  Saved {fpath}")

    print("\nDone.")
