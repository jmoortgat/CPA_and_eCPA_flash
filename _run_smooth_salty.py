"""
Compute eCPA smooth curves at ms=0.5, 1.0, 2.0 mol/kg for T>=293K,
then regenerate all per-temperature figures with the salty curves added.

Existing CPA / eCPA-ms~0 results are loaded from saved parquets — no
re-running of those calculations.
"""
import warnings; warnings.filterwarnings('ignore')

if __name__ == '__main__':
    import os
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from ecpa.parameters import make_params
    from ecpa.solution_table import load_solution_table, make_solution_guess_fn
    from ecpa.validate_co2h2o import (
        run_smooth_curves, plot_validation_T, print_perf_summary, MS_EVAL,
    )

    os.makedirs('results', exist_ok=True)
    os.makedirs('figures/co2h2o', exist_ok=True)

    # ── Load table + params ────────────────────────────────────────────────────
    print("Loading solution table …")
    grid_data = load_solution_table()
    solution_guess_fn = make_solution_guess_fn(grid_data)
    params = make_params()

    # ── Load existing results ──────────────────────────────────────────────────
    print("Loading existing parquets …")
    results = pd.read_parquet('results/validation_co2h2o.parquet')
    smooth0 = pd.read_parquet('results/smooth_co2h2o.parquet')
    elv_df  = pd.read_parquet('CPA_ELV_all.parquet')
    exp_df  = pd.read_parquet('CO2_WATER_exp.parquet')

    # T values for salty curves: same set used for smooth0, filtered to T>=293K
    # and within solution-table range (T<=523K).
    from ecpa.validate_co2h2o import T_MAX_ECPA
    salty_T_vals = sorted(
        T for T in smooth0['T_K'].unique() if T >= 293.0 and T <= T_MAX_ECPA
    )
    print(f"Salty curves will be computed for {len(salty_T_vals)} temperatures "
          f"({min(salty_T_vals):.0f}–{max(salty_T_vals):.0f} K)")

    # ── Run smooth curves for each salty ms ───────────────────────────────────
    MS_SALTY = [0.5, 1.0, 2.0]
    salty_smooth = {}   # ms → DataFrame

    for ms_val in MS_SALTY:
        print(f"\nRunning smooth curves at ms={ms_val} mol/kg …")
        df = run_smooth_curves(
            T_vals            = salty_T_vals,
            solution_guess_fn = solution_guess_fn,
            params            = params,
            ms                = ms_val,
            n_P               = 100,
            verbose           = True,
        )
        out_path = f'results/smooth_co2h2o_ms{ms_val:.1f}.parquet'
        df.to_parquet(out_path, index=False)
        print(f"Saved {out_path}  ({len(df)} rows)")
        salty_smooth[ms_val] = df

        print(f"\n── Performance at ms={ms_val} ──")
        print_perf_summary(results.iloc[0:0], df)   # pass empty results_df

    # ── Regenerate all figures ─────────────────────────────────────────────────
    print("\nRegenerating figures …")
    exp_T_vals = sorted(exp_df['T_K'].unique())
    salty_curve_list = [(ms_val, salty_smooth[ms_val]) for ms_val in MS_SALTY]

    for T_K in exp_T_vals:
        fpath = f'figures/co2h2o/T{int(T_K)}K.png'
        # Only pass salty curves for T >= 293
        sc = salty_curve_list if T_K >= 293.0 else None
        fig = plot_validation_T(
            T_K        = T_K,
            results_df = results,
            smooth_df  = smooth0,
            save_path  = fpath,
            ms         = MS_EVAL,
            elv_df     = elv_df,
            salty_curves = sc,
        )
        plt.close(fig)
        print(f"  Saved {fpath}")

    print("\nDone.")
