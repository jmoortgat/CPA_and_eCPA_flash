# ms-eCPA: gases in multisalt brines

The ion-specific electrolyte Cubic-Plus-Association equation of state
(ms-eCPA) for gases in water and multisalt brines, and the scripts that
produce the figures and tables of Papers II and III of this series:

- **Paper II.** J. Moortgat, F. Mourão Coelho, A. Firoozabadi. Phase Equilibria
  of CO2 and CH4 in Multisalt Brines: Parametrization and Validation of an
  Ion-Specific Electrolyte Cubic-Plus-Association Equation of State.
  *Energy & Fuels* **2026**, 40 (42), 23472–23484.
  [doi:10.1021/acs.energyfuels.6c04568](https://doi.org/10.1021/acs.energyfuels.6c04568)
- **Paper III.** J. Moortgat, F. M. Coelho, A. Firoozabadi. Phase Equilibria of
  CO2, CH4, N2, H2, C2H6, C3H8, and Mixtures in Multisalt Brines: Robust
  Stability and Flash Computations with an Ion-Specific Electrolyte
  Cubic-Plus-Association Equation of State. *Energy & Fuels* **2026**.
  [doi:10.1021/acs.energyfuels.6c04569](https://doi.org/10.1021/acs.energyfuels.6c04569)

Paper I (CO2 + H2O + NaCl) is the `code/` package at the root of this
repository. ms-eCPA extends it to the components

    H2O, CH4, CO2, Na+, Cl-, K+, Ca2+, Mg2+, SO4 2-

with the SRK physical term, Wertheim association, Debye–Hückel and Born
terms, a dielectric-constant closure, and ion-specific Péneloux volume
shifts. Paper III adds N2, H2, C2H6 and C3H8 (one gas at a time, in the CH4
slot), a Michelsen tangent-plane-distance stability test, and two- and
three-phase flash calculations.

## Layout

```
ms_ecpa/        the equation of state, stability test, two- and three-phase flash
ecpa_ms/        alias of ms_ecpa (older scripts import this name)
fortran/        a compiled implementation of the same kernel (optional, ~190x faster)
data/           cached experimental tables and the CH4-H2O start-value table
fitting/        objective, data sets and parameter-fitting tools
scripts/        figure and table scripts (Paper II; Paper III validation figures)
pilot_studies/  per-gas and per-topic studies behind Paper II/III figures:
                  n2_pilot_*, h2_pilot_*     N2 and H2 (Paper III)
                  c2_pilot_*, c3_pilot_*     C2H6 and C3H8 (Paper III)
                  three_phase_*              three-phase locus (Paper III)
                  vpsat_salt_*, ywater_lowP_*  SI figures (Paper III)
                  koschel_enthalpy_*         enthalpy of solution (Paper II)
sweep/          the stability-and-flash grid over (T, P, salt, feed) (Paper III)
reproduce/      checks: reference states, convergence counts, convention comparison
tests/          pytest suite
```

## Requirements

Python 3.11 with numpy, scipy, pandas, pyarrow, matplotlib and CoolProp.
The package is used from the source tree; no installation is needed:

```bash
cd multisalt
export PYTHONPATH=$PWD
python3 -c "from ms_ecpa import stability_flash"
```

The Fortran library is optional. With gfortran:

```bash
make -C fortran
MS_ECPA_BACKEND=fortran python3 scripts/make_all_paper_figures.py
```

The compiled backend gives the same results as the Python kernel (see
`fortran/README.md`) for the default salinity convention, which is the one
used in Papers II and III.

## Experimental data

All measurements come from **GasBrineBench**
([github.com/jmoortgat/GasBrineBench](https://github.com/jmoortgat/GasBrineBench)),
release **v1.1.1** ([doi:10.5281/zenodo.22898153](https://doi.org/10.5281/zenodo.22898153)).
`ms_ecpa/gbb.py` finds a local checkout (a sibling directory named
`GasBrineBench`, or the path in `ECPA_GBB`) or downloads v1.1.1 into
`~/.cache/ecpa/`. `ECPA_GBB_REFRESH=never` keeps it offline.

Start values for the CO2–H2O and CH4–H2O binaries are precomputed CPA
solutions, not data: `data/guess_tables/CH4.parquet`, and for CO2 the Paper I
table `code/results/CPA_ELV_all.parquet`.

## Reproducing the papers

The two executable papers in `../notebooks/` (`ms_ecpa_parametrization_paper.ipynb`
for Paper II, `ms_ecpa_flash_paper.ipynb` for Paper III) call these scripts for every
figure and table and show the result beside the published text. Run the commands from this
directory with `PYTHONPATH=.:scripts`.

Figures are written to `multisalt/figures/` (or to `$ECPA_PAPER_FIGURES`). The model
results behind them are shipped in `results_cache/` and beside the pilot scripts; by
default each script draws from those results, and `--force` (figure scripts) or
`--recompute` (pilot scripts) recomputes them from the equation of state.

| What | Command |
|---|---|
| Paper II figures (brine density, osmotic coefficients, binaries, CO2 and CH4 in single-salt and mixed brines, Henry constants, parameter plots); Paper III stability, flash-parity, N2, H2 and CH4–CO2–H2O/brine ternary figures | `python3 scripts/make_all_paper_figures.py` |
| Paper II tables and number macros | `python3 scripts/build_paper_tables.py`, `python3 scripts/build_paper_stats_tex.py` |
| Paper II enthalpy of solution (Figure 7) | `python3 pilot_studies/koschel_enthalpy_2026_08/koschel_pilot.py --stage figure` (`--stage all` recomputes) |
| Paper II mixed-brine predictions (Tables S13, S14) | `python3 reproduce/mixed_brine_ch4_prediction.py` |
| Paper III C2H6 and C3H8 | `pilot_studies/c2_pilot_2026_08/c2_fig_split.py`; `pilot_studies/c3_pilot_2026_08/c3_fig_final.py` |
| Paper III three-phase locus (Figure 6) | `python3 pilot_studies/three_phase_2026_08/vlle_locus.py` (writes `fig_vlle_locus.png` beside the script) |
| Paper III water vapor pressure of brines, low-pressure water content (Figures S7, S9) | `pilot_studies/vpsat_salt_2026_08/vpsat.py`, `pilot_studies/ywater_lowP_2026_08/fig_ywater.py` |
| Paper III flash grid: figures and Tables 4, S8, S11–S13 | `python3 -m sweep.figures.make_sweep_figures --merged results_cache/sweep/merged.parquet --out-dir figures/sweep` |
| Paper III ternary statistics and k_hc scan (Tables S9, S10) | `python3 reproduce/ternary_stats.py` |
| Paper III threshold sensitivity (Table S15) | `python3 reproduce/threshold_sensitivity.py` (`--rerun` recomputes the 276 boundary states) |
| Paper III convergence counts | `python3 reproduce/convergence_census.py`, `python3 reproduce/sweep_census.py` |
| Salinity convention: effect of the equilibrium vs prepared-brine molality (Papers II and III) | `python3 reproduce/convention_comparison.py` |

The flash over the 151,632-state grid itself runs on an HPC cluster:
`sweep/README.md` gives the grid definitions and run times, and `sweep/slurm/`
holds the SLURM scripts. Its merged results are shipped in `results_cache/sweep/`.

## Published and released numbers

The released code reproduces the CO2 tables and per-salt CO2 results of
Paper II. It does not reproduce the two CH4 tables of the Paper II
Supporting Information, which were computed with an earlier version of the
code; the released code gives a lower AAD for CH4 in water (7.6 % instead
of 11.1 %). For Paper III, the CH4 entries of Table 3 come from the same
earlier code, and the bound on the CO2–H2O three-phase pressure stated in the
text does not cover the 302.15 K point of Figure 6, which the released code
reproduces.
[PUBLISHED_VS_RELEASED.md](PUBLISHED_VS_RELEASED.md) lists every value that
differs and the reasons.

## Checking the code

```bash
python3 -m pytest tests                    # 99 tests; 4 marked as known deviations
python3 -m pytest fortran/tools            # Python vs Fortran, after make -C fortran
python3 reproduce/verify_reference.py      # 1,920 stored states, compared point by point
python3 reproduce/verify_baseline.py       # family scores against the published values
```

The four tests marked as known deviations concern the Paper I parameter
configuration of this kernel (agreement with the original Paper I code to
2e-4 rather than 3e-5 at two states) and one pure-water vapor-pressure state
at 283 K. None of them enters a Paper II or III result.

## Scope

This directory holds the model of Papers II and III only. The parameters are
those published in Paper II (`ms_ecpa/parameters.py`) and, for the additional
gases, in Paper III (`pilot_studies/*/` result files). `ms_ecpa/extensions.py`
defines hooks through which a separate package can add terms to the aqueous
model; nothing is registered here, and with no extension registered the
kernel evaluates exactly the published model.
