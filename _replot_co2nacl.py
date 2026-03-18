"""
Regenerate CO2-NaCl validation figures from saved parquet.
No flash re-running required.

Generates:
  - Per-temperature T-panel figures (T <= 523 K) in figures/co2nacl/
  - AARE heatmap (individual ms values) in figures/validation_heatmap_extended.png
"""
import warnings; warnings.filterwarnings('ignore')

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    import pandas as pd
    from ecpa.validate_nacl import plot_nacl_T_figures, plot_error_heatmap

    results = pd.read_parquet('results/validation_co2nacl.parquet')
    print(f"Loaded {len(results)} rows  ({(results.status=='ok').sum()} ok)")

    saved = plot_nacl_T_figures(
        results,
        fig_dir='figures/co2nacl',
        T_max=523.0,
        ms_max=7.0,
    )

    print("Regenerating heatmap …")
    plot_error_heatmap(
        results,
        save_path='figures/validation_heatmap_extended.png',
    )
    print("Saved figures/validation_heatmap_extended.png")

    print("Done.")
