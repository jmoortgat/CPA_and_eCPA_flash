# Scripts

All scripts needed to reproduce the figures, tables, and numerical results
reported in the paper. Each script is self-contained; inputs are read from
`results/` or `EXP/` and outputs go to `results/` or `figures/` (created on demand).

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
| `validate_density.py` | Validate aqueous-phase density and optimise the Péneloux volume shift. Writes `results/density_co2h2o.parquet`. |
| `generate_warmstart_co2h2o.py` | Run CO₂+H₂O validation with warm-started K-value flash. Writes `results/ws_validation_co2h2o.parquet`. |
| `generate_warmstart_co2nacl.py` | Run CO₂+NaCl validation with warm-started K-value flash. Writes `results/ws_validation_co2nacl.parquet`. |

---

## Figure generation

| Script | Output | Paper figure |
|--------|--------|-------------|
| `plot_co2h2o_figures.py` | `figures/co2h2o_ws/T*.png` | Figs 1, S1 |
| `plot_co2nacl_figures.py` | `figures/co2nacl_ws/T*.png` | Figs 2, 3, S2 |
| `plot_co2nacl_highT_figures.py` | `figures/co2nacl_ws/T573K.png`, `T623K.png` | Fig S3 |
| `plot_flash_vs_z.py` | `figures/flash_vs_z.png` | Fig 4 |
| `plot_scan_figures.py` | `figures/scan_v4/*.pdf` | Figs 5–9 |
| `plot_newton_figures.py` | `figures/scan/newton_heatmap.png` | Fig S7 |
| `plot_newton_stats.py` | `figures/scan_v4/ecpa_newton_stats.pdf` | Fig S7 (alt) |

---

## Parameter-space scan

| Script | Purpose |
|--------|---------|
| `run_parameter_scan.py` | Extended CPA parameter-space scan (86 T × 18 P × 19 z = 29,412 conditions). Generates convergence statistics reported in Section 5.1.1. |
| `run_newton_scan.py` | SSI + Newton polish on the full two-phase scan grid; produces iteration statistics for Tables 3–4. |
| `scan_experimental_points.py` | CPA stability + flash at all 631 experimental CO₂+H₂O data points. |
| `run_warmstart_scan.py` | Add the warm-start strategy to the scan comparison. |

---

## Benchmarks and timing

| Script | Purpose / paper result |
|--------|----------------------|
| `benchmark_cpa_vs_srk.py` | Per-SSI-iteration cost: CPA vs plain cubic (SRK). Quantifies the ~3× overhead. |
| `benchmark_warmstart.py` | Warm-start vs cold-start flash performance (Table 5 speedup factors). |
| `benchmark_cold_start.py` | Cold-start stability + flash benchmark (no solution table). |
| `compare_cpa_warmstart.py` | CPA flash with vs without solution-table warm-start. |
| `benchmark_newton.py` | Analytical vs finite-difference Jacobian Newton solve (inner eCPA). |
| `benchmark_newton_tol.py` | Sensitivity of K-value Newton polish to convergence tolerance. |
| `benchmark_z_rootfinding.py` | Z-factor root-finding speed comparison. |
| `time_co2h2o.py` | Wall-time for the CO₂+H₂O validation flash loop (throughput in Section 5.3). |
| `time_co2nacl.py` | Wall-time for the CO₂+NaCl validation flash loop. |
| `time_cpa_warmstart.py` | Timing for warm-started CPA flash at experimental conditions. |

---

## Simulation

| Script | Purpose |
|--------|---------|
| `run_demo_simulations.py` | Prototype IMPEC reservoir simulator demo (Fig 10). |
| `run_simulator_paper.py` | Full simulator runs for paper figures and throughput benchmarks. |
| `run_benchmark.py` | Comprehensive flash performance benchmark (paper Table 5). |
