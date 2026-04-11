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

    import numpy as np
    import pandas as pd
    from pathlib import Path
    from ecpa.validate_nacl import plot_nacl_T_figures, plot_error_heatmap
    from _run_smooth_co2h2o_robust import _ms_tag, MS_RIBBON

    # Use K-value SSI results (post B1/B4 bug fix, 708 points)
    results = pd.read_parquet('results/ws_validation_co2nacl.parquet')
    print(f"Loaded {len(results)} rows  ({(results.status=='ok').sum()} ok)")

    # Load smooth-curve parquets for rainbow ribbon background
    smooth_data = {}
    for ms in MS_RIBBON:
        cache = Path(f'results/ws2_smooth_co2h2o_ms{_ms_tag(ms)}.parquet')
        if cache.exists():
            smooth_data[ms] = pd.read_parquet(cache)
    print(f"Loaded smooth_data for {len(smooth_data)} ms values: {list(smooth_data)}")

    for fig_dir, T_max in [('figures/co2nacl_ws', 730.0),
                            ('figures/co2nacl',    523.0)]:
        saved = plot_nacl_T_figures(
            results,
            fig_dir=fig_dir,
            T_max=T_max,
            ms_max=7.0,
            smooth_data=smooth_data if smooth_data else None,
        )
        print(f"Saved {len(saved)} figures → {fig_dir}/")

    print("Regenerating heatmap …")
    plot_error_heatmap(
        results,
        save_path='figures/validation_heatmap_extended.png',
    )
    print("Saved figures/validation_heatmap_extended.png")

    print("Done.")
