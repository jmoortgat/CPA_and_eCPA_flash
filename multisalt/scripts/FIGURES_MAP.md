# Paper figures: script → output PNG → LaTeX reference

This is the mapping between the `fig_*.py` scripts in this folder and the
published figures of Paper II (Energy Fuels 2026, 40, 23472-23484,
doi:10.1021/acs.energyfuels.6c04568) and Paper III (Energy Fuels 2026,
doi:10.1021/acs.energyfuels.6c04569). To regenerate everything, run `python3 scripts/make_all_paper_figures.py`
(use `--force` to recompute caches).

The cache parquet associated with each script lives in
`results_cache/<script_stem>_data.parquet`.

## Phase 1 — Headline flash & stability figures

| Script | Output PNG | Published figure | Cache |
|---|---|---|---|
| `fig_flash_predictions.py` | `flash_parity_CO2_brines_main.png` | Paper III Figure 2 | `fig_flash_predictions_data.parquet` |
|  | `per_salt_parity_SI.png` | Paper III Figure S2 |  |
|  | `experimental_variance_SI.png` | Paper III Figure S3 |  |
| `fig_stability_isotherm.py` | `stability_tpd_isotherm_NaCl.png` | Paper III Figure S1 | `fig_stability_isotherm_data.parquet` |

## Phase 2 — Single-salt / mixed-brine / ternary figures

| Script | Output PNG | Published figure | Cache |
|---|---|---|---|
| `fig_brine_density.py` | `brine_density_main.png` | Paper II Figure 1 | `fig_brine_density_data.parquet` |
|  | `brine_density_SI_KCl.png` | Paper II Figure S6 |  |
|  | `brine_density_SI_CaCl.png` | Paper II Figure S7 |  |
|  | `brine_density_SI_MgCl.png` | Paper II Figure S8 |  |
|  | `brine_density_SI_NaSO.png` | Paper II Figure S9 |  |
| `fig_colligative.py` | `colligative_main.png` | Paper II Figure 2 | `fig_colligative_data.parquet` |
|  | `colligative_SI.png` | Paper II Figure S5 |  |
| `fig_binaries.py` | `binaries_main.png` | Paper II Figure 3 | `fig_binaries_data.parquet` |
|  | `binaries_SI.png` | not in the published papers |  |
| `fig_co2_singlesalt.py` | `co2_singlesalt_main.png` | Paper II Figure 4 | `fig_co2_singlesalt_data.parquet` |
|  | `co2_singlesalt_SI_NaCl.png` | Paper II Figure S10 |  |
|  | `co2_singlesalt_SI_KCl.png` | Paper II Figure S11 |  |
|  | `co2_singlesalt_SI_CaCl.png` | Paper II Figure S12 |  |
|  | `co2_singlesalt_SI_MgCl.png` | Paper II Figure S13 |  |
|  | `co2_singlesalt_SI_NaSO.png` | Paper II Figure S14 |  |
| `fig_co2_mixed_brines.py` | `co2_mixed_brines_main.png` | Paper II Figure 6 | `fig_co2_mixed_brines_data.parquet` |
|  | `co2_mixed_brines_SI_LIU_2011.png` | Paper II Figure S19 |  |
|  | `co2_mixed_brines_SI_WANG_2014.png` | Paper II Figure S20 |  |
|  | `co2_mixed_brines_SI_POULANI_2019.png` | Paper II Figure S21 |  |
|  | `co2_mixed_brines_SI_ZHAOc_2015.png` | Paper II Figure S22 |  |
|  | `co2_mixed_brines_SI_other.png` | Paper II Figure S23 |  |
| `fig_ch4_singlesalt.py` | `ch4_singlesalt_main.png` | Paper II Figure 5 | `fig_ch4_singlesalt_data.parquet` |
|  | `ch4_singlesalt_SI_NaCl.png` | Paper II Figure S15 |  |
|  | `ch4_singlesalt_SI_KCl.png` | Paper II Figure S16 |  |
|  | `ch4_singlesalt_SI_CaCl2.png` | Paper II Figure S17 |  |
|  | `ch4_singlesalt_SI_other.png` | Paper II Figure S18 |  |
| `fig_ch4_henry.py` | `henry_water_brine_SI.png` | Paper II Figure S24 | `fig_ch4_henry_data.parquet` |
| `fig_ccw_ternary.py` | `ccw_ternary_main.png` | not in the published papers | `fig_ccw_ternary_data.parquet` |
|  | `ccw_ternary_SI.png` | Paper III Figure 5 |  |
| `fig_ccb_ternary.py` | `ccb_ternary_SI.png` | Paper III Figure S23 | `fig_ccb_ternary_data.parquet` |
| `fig_additional_gases.py` | `n2_brine_main.png` | Paper III Figure 3 | `fig_additional_gases_data.parquet` |
|  | `h2_brine_main.png` | Paper III Figure 4 |  |

## Phase 3 — Parameter-fit figures (analytic, no flash)

| Script | Output PNG | Published figure | Cache |
|---|---|---|---|
| `fig_param_fits.py` | `kij_h2o_ch4.png` | Paper II Figure S1 | `fig_param_fits_data.parquet` |
|  | `delta_U_ch4_ions.png` | Paper II Figure S2 |  |
|  | `delta_U_co2_ions.png` | Paper II Figure S3 |  |
|  | `peneloux_ions.png` | Paper II Figure S4 |  |

## Infrastructure (not figure scripts)

| File | Purpose |
|---|---|
| `paper_style.py` | Centralized `rcParams`, WONG palette, `LINESTYLES`, panel helpers, `PAPER_FIGURES` path constant. |
| `data_loader.py` | `load_experimental(kind)` (reads parquet), `load_or_compute(name, fn, force=...)` (cache pattern), `add_cache_flag(parser)`. |
| `build_experimental_parquets.py` | One-time consolidator from F. M. Coelho's `EXP_*.txt` source files into `data/<kind>.parquet`. Re-run only if upstream data changes. |
| `make_all_paper_figures.py` | Orchestrator. Calls every `fig_*.py` in dependency order. Pass `--force` to recompute every cache. |

## How to regenerate

```bash
# Everything (fast: uses caches; ~5 s when hot)
python3 scripts/make_all_paper_figures.py

# Recompute every cache (slow: ~5 min for flash + density)
python3 scripts/make_all_paper_figures.py --force

# A single family
python3 scripts/fig_brine_density.py
python3 scripts/fig_brine_density.py --force
```
