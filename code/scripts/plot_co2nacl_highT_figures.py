"""Regenerate T573K.png and T623K.png without rainbow ribbon background."""
import warnings; warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, '/Users/moortgat/Software/2026/eCPA_SALTbasis/Claude_code')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from ecpa.validate_nacl import plot_nacl_T_figures

results = pd.read_parquet('results/ws_validation_co2nacl.parquet')
# Keep only the two high-T panels
results_highT = results[results['T_K'].isin([573.0, 623.0])].copy()
print(f"Rows for T=573/623 K: {len(results_highT)}")

saved = plot_nacl_T_figures(
    results_highT,
    fig_dir='figures/co2nacl_ws',
    T_max=730.0,
    ms_max=7.0,
    smooth_data=None,   # no rainbow background
)
print(f"Saved: {saved}")
