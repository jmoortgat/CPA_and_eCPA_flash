# Scripts

All scripts needed to reproduce the figures, tables, and numerical results
reported in the paper. Each script is self-contained; inputs are read from
`results/` or `EXP/` and outputs go to `results/` or `figures/` (created on
demand).

Run every script from the `code/` directory with the package on the path:

```bash
cd code
PYTHONPATH=. python scripts/<script>.py
```

The authoritative figure-by-figure reproduction guide (including which
scripts must run first) is [`REPRODUCING_FIGURES.md`](../../REPRODUCING_FIGURES.md)
at the repository root.

---

## Solution table

| Script | Purpose |
|--------|---------|
| `build_solution_table.py` | Build the 3-D $(T,P,m_s)$ warm-start solution table from scratch (~1–4 h on HPC). Writes `results/solution_table.npz`. |
| `extend_solution_table_highT.py` | Extend the table to $T=538$–623 K. |
| `extend_solution_table_full.py` | Extend to the full $(T,P,m_s)$ coverage used in the paper. |
| `extend_solution_table_ms0.py` | Add a salt-free ($m_s=0$) slice to the table. |

---

## Validation

| Script | Purpose |
|--------|---------|
| `validate_co2h2o.py` | Validate the salt-free CPA binary against experimental CO₂+H₂O VLE data. Writes `results/validation_co2h2o.parquet`. |
| `validate_co2nacl.py` | Validate the eCPA ternary against experimental CO₂+NaCl VLE data. Writes `results/validation_co2nacl.parquet`. |
| `validate_co2nacl_full.py` | Full eCPA ternary validation over all experimental conditions. |
| `validate_co2nacl_highT.py` | Extend validation to high-temperature conditions ($T>523$ K). |
| `benchmark_pure_water_density.py` | eCPA pure-water ($m_s \to 0$) density vs IAPWS-95 over the full $(T,P)$ range, with the temperature-dependent Péneloux shift; produces the density parity and error-map figures (paper Figs. S5, S6). Requires the optional `iapws` package. |
| `validate_density.py` | Companion check of CPA/eCPA aqueous density against the 37 experimental CO₂-saturated data points (no volume shift; not a paper figure). Writes `results/density_co2h2o.parquet`. |
| `generate_warmstart_co2h2o.py` | Run CO₂+H₂O validation with warm-started K-value flash. Writes `results/ws_validation_co2h2o.parquet`. |
| `generate_warmstart_co2nacl.py` | Run CO₂+NaCl validation with warm-started K-value flash. Writes `results/ws_validation_co2nacl.parquet`. |
| `_run_smooth_co2h2o_robust.py` | Helper: recompute the smooth CO₂+H₂O model curves used by the plotting scripts. |

---

## Figure generation

See `REPRODUCING_FIGURES.md` for the complete mapping and required
prerequisites. In brief:

| Script | Output | Paper figures |
|--------|--------|---------------|
| `plot_co2h2o_figures.py` | `figures/co2h2o/T*.png`, `error_heatmap.png` | Figs. 1, S1, S7 |
| `plot_co2nacl_figures.py` | `figures/co2nacl*/T*.png`, `validation_heatmap_extended.png` | Figs. 2, 3, S2, S3, S4 |
| `plot_co2nacl_highT_figures.py` | `figures/co2nacl/T573K.png`, `T623K.png` | high-T panels of Fig. S2 |
| `plot_flash_vs_z.py` | `figures/flash_vs_z.png` | Fig. 4 |
| `plot_newton_figures.py` | `figures/scan/newton_heatmap.png`, `strategy_comparison_bar.png` | Figs. 5, S11 |
| `plot_speedup_figures.py` | `figures/scan/speedup_ratio_heatmap.png`, `stability_best_trial.png` | Fig. 6 |
| `plot_scan_figures.py` | `figures/scan_v4/*.png` | Figs. 7, 8, 9 |
| `plot_newton_stats.py` | `figures/scan_v4/ecpa_newton_stats.png` | Fig. S10 |
| `benchmark_pure_water_density.py` | `figures/density/iapws_parity.png`, `iapws_error_map.png` | Figs. S5, S6 |
| `benchmark_simplified_flash.py` | `figures/simplified/accuracy_vs_T.png`, `accuracy_by_ms.png` | Figs. S8, S9 |
| `run_simulator_paper.py` | `figures/simulator/fig_compositions.png`, `fig_perf.png` | Figs. 10, S12 |

---

## Parameter-space scan

| Script | Purpose |
|--------|---------|
| `run_parameter_scan.py` | Extended CPA parameter-space scan (86 T × 18 P × 19 z = 29,412 conditions). Writes `results/scan_results_extended.npz` with the convergence statistics reported in the paper. |
| `run_newton_scan.py` | SSI + Newton polish on the full two-phase scan grid; produces the reported iteration statistics. |
| `scan_experimental_points.py` | CPA stability + flash at all 631 experimental CO₂+H₂O data points. |
| `run_warmstart_scan.py` | Add the warm-start strategy to the scan comparison; generates `results/scan_v4_table.npz`. |

---

## Benchmarks and timing

| Script | Purpose |
|--------|---------|
| `benchmark_cpa_vs_srk.py` | Per-SSI-iteration cost: CPA vs plain cubic (SRK). Quantifies the ~3× overhead. |
| `benchmark_warmstart.py` | Warm-start vs cold-start flash performance (reported speedup factors). |
| `benchmark_cold_start.py` | Cold-start stability + flash benchmark (no solution table). |
| `benchmark_simplified_flash.py` | Simplified flash ($y_{\mathrm{H_2O}}=0$) vs full K-value flash accuracy and speed. |
| `compare_cpa_warmstart.py` | CPA flash with vs without solution-table warm-start. |
| `benchmark_newton.py` | Analytical vs finite-difference Jacobian Newton solve (inner eCPA). |
| `benchmark_newton_tol.py` | Sensitivity of K-value Newton polish to convergence tolerance. |
| `benchmark_z_rootfinding.py` | Z-factor root-finding speed comparison. |
| `time_co2h2o.py` | Wall-time for the CO₂+H₂O validation flash loop. |
| `time_co2nacl.py` | Wall-time for the CO₂+NaCl validation flash loop. |
| `time_cpa_warmstart.py` | Timing for warm-started CPA flash at experimental conditions. |

---

## Volume-shift studies

| Script | Purpose |
|--------|---------|
| `optimize_peneloux_h2o.py` | Optimize a temperature-dependent Péneloux volume shift for H₂O against IAPWS-95 density data (exploratory; the published model uses no shift). |
| `optimize_peneloux_co2.py` | Same for CO₂ against the Span–Wagner reference (requires the optional `CoolProp` package). |

---

## Simulation

| Script | Purpose |
|--------|---------|
| `run_demo_simulations.py` | Prototype IMPEC reservoir simulator demo. |
| `run_simulator_paper.py` | Full simulator runs for the paper (Figs. 10, S12) and throughput benchmarks. |
| `run_benchmark.py` | Comprehensive flash performance benchmark. |
