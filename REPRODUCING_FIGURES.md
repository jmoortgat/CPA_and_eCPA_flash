# Reproducing the Figures

All commands are run from the `code/` subdirectory unless noted otherwise, with the
repository importable — either `pip install -e .` from the repo root (recommended),
or prefix each command with `PYTHONPATH=.`:

```bash
cd code
export PYTHONPATH=.
```

---

## Overview

Figure generation is a two-step process:

1. **Data generation** — run validation, benchmark, or scan scripts that compute
   flash results and save them to `results/` as parquet or npz files.
2. **Plotting** — run lightweight plot scripts that load those saved results and
   write figures to `figures/`.

The sections below are organised by figure number and list both steps.
The precomputed solution table (`results/CPA_ELV_all.parquet`, included in the
repository) is required by most scripts — no additional setup is needed for it.

---

## Main paper figures

---

### Figure 1 — CO₂ + H₂O phase equilibria at 25, 150, and 250 °C

**Data generation** (~5 min):
```bash
python scripts/validate_co2h2o.py
```
Saves `results/validation_co2h2o.parquet` and `results/smooth_co2h2o.parquet`.

**Plotting**:
```bash
python scripts/plot_co2h2o_figures.py
```
Outputs per-temperature panels to `figures/co2h2o/T*.png`.
Panels used in Fig. 1: `T298K.png` (25 °C), `T423K.png` (150 °C), `T523K.png` (250 °C).

---

### Figure 2 — CO₂ solubility in NaCl brine at 40, 80, 120, and 250 °C

**Data generation** (~30 min):
```bash
python scripts/validate_co2nacl_full.py
```
Saves `results/ws_validation_co2nacl.parquet` and warm-start smooth-curve parquets
(`results/ws2_smooth_co2h2o_ms*.parquet`).

**Plotting**:
```bash
python scripts/plot_co2nacl_figures.py
```
Outputs per-temperature panels to `figures/co2nacl/T*.png`.
Panels used in Fig. 2: `T313K.png` (40 °C), `T353K.png` (80 °C), `T393K.png` (120 °C),
`T523K.png` (250 °C).

---

### Figure 3 — CO₂ + NaCl two-panel at 100 °C (x-CO₂ and y-CO₂)

Same data and script as Fig. 2. Panels used:
- `figures/co2nacl/T373K.png` (100 °C aqueous panel)
- `figures/co2nacl_ws/T373K.png` (100 °C CO₂-rich panel, from the same script run)

---

### Figure 4 — eCPA flash results vs. CO₂ feed mole fraction *z*

**Data generation**: requires `results/scan_v4_table.npz` (see Figs. 7–9 below).

**Plotting**:
```bash
python scripts/plot_flash_vs_z.py
```
Output: `figures/flash_vs_z.png`.

---

### Figure 5 — Mean SSI iteration count (CPA binary parameter-space scan)

**Data generation — CPA parameter-space scan** (~20 min):
```bash
python scripts/run_parameter_scan.py
```
Saves `results/scan_results_extended.npz` (86T × 18P × 19z grid).

**Data generation — Newton polish scan** (~10 min):
```bash
python scripts/run_newton_scan.py
```
Saves `results/scan_newton_results.npz`.

**Plotting**:
```bash
python scripts/plot_newton_figures.py
```
Output: `figures/scan/newton_heatmap.pdf` and `figures/scan/newton_heatmap.png`.

---

### Figure 6 — SSI speedup ratio and best stability initial guess

Same data prerequisites as Fig. 5 (`run_parameter_scan.py`; also reads
`results/scan_v4_table.npz`, see Figs. 7–9).

**Plotting**:
```bash
python scripts/plot_speedup_figures.py
```
Outputs (in `figures/scan/`):
- `speedup_ratio_heatmap.png`  — Fig. 6a (standard/accelerated SSI iteration ratio)
- `stability_best_trial.png`   — Fig. 6b (distribution of the best stability initial guess)

---

### Figures 7, 8, 9 — eCPA ternary composition and timing grids

**Data generation — eCPA warm-start scan** (~27 min, uses all CPU cores):
```bash
python scripts/run_warmstart_scan.py
```
Saves `results/scan_v4_table.npz` and `results/scan_v4_metrics.parquet`.

**Plotting**:
```bash
python scripts/plot_scan_figures.py
```
Outputs (in `figures/scan_v4/`):
- `ecpa_composition_aq_grid.pdf` — Fig. 7 (CO₂ mol% in aqueous phase)
- `ecpa_composition_c_grid.pdf`  — Fig. 8 (H₂O mol% in CO₂-rich phase)
- `ecpa_timing_heatmap.pdf`      — Fig. 9 (mean wall time per flash call)

---

### Figure 10 — Reservoir simulator: final spatial fields (CPA vs. eCPA)

**Run the simulator** (~10 min on a laptop, 50×50 grid, 300 time steps):
```bash
python scripts/run_simulator_paper.py
```
Output: `figures/simulator/fig_compositions.png`.
The same run also writes `fig_simulator.png` (spatial maps at multiple time
steps) and `fig_perf.png` (used in Fig. S12).

---

## Supplemental Information figures

---

### Figure S1 — CO₂ + H₂O panels at further temperatures

Same data and script as Fig. 1. All panels in `figures/co2h2o/T*.png` are used.

---

### Figure S2 — CO₂ + NaCl brine panels at further temperatures

Same data and script as Fig. 2 (requires `validate_co2nacl_full.py`). Panels
from `figures/co2nacl/T*.png`; the two hottest panels (300 and 350 °C) are
written by:

```bash
python scripts/plot_co2nacl_highT_figures.py
```

---

### Figure S3 — CO₂ + NaCl at 100 °C, molality and mole-fraction views

Same data and script as Fig. 2. Panels used: `figures/co2nacl/T373K.png` and
`figures/co2nacl_ws/T373K.png`.

---

### Figure S4 — AARE heatmap for CO₂ + NaCl (all T, all m_s)

Same data and script as Fig. 2. Output: `figures/validation_heatmap_extended.png`.

---

### Figures S5 and S6 — Pure-water density parity plot and signed error map

**Data generation and plotting** (~1 min, single script; requires the optional
`iapws` package, `pip install iapws`):
```bash
python scripts/benchmark_pure_water_density.py
```
Saves `results/pure_water_density_iapws.parquet` and writes
`figures/density/iapws_parity.png` (Fig. S5) and
`figures/density/iapws_error_map.png` (Fig. S6).

A companion comparison against the 37 experimental CO₂-saturated aqueous
density points (not a paper figure) is produced by
`python scripts/validate_density.py`.

---

### Figure S7 — ARE scatter plot and error heatmap (CO₂ + H₂O)

Same data and script as Fig. 1. Output: `figures/co2h2o/error_heatmap.png`.

---

### Figures S8 and S9 — Simplified-flash benchmark

**Data generation and plotting** (~10 min, 3,285 conditions, single script):
```bash
python scripts/benchmark_simplified_flash.py
```
Outputs: `figures/simplified/accuracy_vs_T.png` (Fig. S8) and
`figures/simplified/accuracy_by_ms.png` (Fig. S9).

---

### Figure S10 — SSI + Newton polish iteration breakdown

Same prerequisites as Figs. 5–6 (requires `run_warmstart_scan.py`).

**Plotting**:
```bash
python scripts/plot_newton_stats.py
```
Output: `figures/scan_v4/ecpa_newton_stats.png`.

---

### Figure S11 — CPA flash strategy comparison (all 9,825 two-phase conditions)

Same prerequisites and script as Fig. 5. The strategy comparison bar chart is
generated alongside the Fig. 5 heatmap:

```bash
python scripts/plot_newton_figures.py
```
Output: `figures/scan/strategy_comparison_bar.png`.

---

### Figure S12 — CO₂ trapping fractions over time

Same script and run as Fig. 10:
```bash
python scripts/run_simulator_paper.py
```
Output: `figures/simulator/fig_perf.png`.

---

## Complete reproduction sequence

To regenerate every figure from scratch, run the following in order.
Steps marked *(slow)* take more than a few minutes.

```bash
cd code

# 1. Build solution table (if not using the pre-built CPA_ELV_all.parquet)  [slow: 1–4 h]
# python scripts/build_solution_table.py

# 2. CO₂ + H₂O binary validation                                            [~5 min]
python scripts/validate_co2h2o.py

# 3. CO₂ + NaCl ternary validation                                           [~30 min]
python scripts/validate_co2nacl_full.py

# 4. Pure-water density vs IAPWS-95 (writes Figs. S5, S6; needs `pip install iapws`)
python scripts/benchmark_pure_water_density.py

# 5. CPA binary parameter-space scan                                         [~20 min]
python scripts/run_parameter_scan.py

# 6. CPA Newton polish scan                                                   [~10 min]
python scripts/run_newton_scan.py

# 7. eCPA warm-start scan (generates scan_v4_table.npz)                     [~27 min]
python scripts/run_warmstart_scan.py

# 8. Reservoir simulator
python scripts/run_simulator_paper.py

# ── Plotting ──────────────────────────────────────────────────────────────────

python scripts/plot_co2h2o_figures.py          # Figs. 1, S1, S7
python scripts/plot_co2nacl_figures.py         # Figs. 2, 3, S2, S3, S4
python scripts/plot_co2nacl_highT_figures.py   # high-T panels of Fig. S2
python scripts/plot_flash_vs_z.py              # Fig. 4
python scripts/plot_newton_figures.py          # Figs. 5, S11
python scripts/plot_speedup_figures.py         # Fig. 6
python scripts/plot_scan_figures.py            # Figs. 7, 8, 9
python scripts/plot_newton_stats.py            # Fig. S10
python scripts/benchmark_simplified_flash.py   # Figs. S8, S9
```
