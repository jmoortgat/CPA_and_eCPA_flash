# The papers and their code

This repository holds the code of three journal articles by J. Moortgat,
F. M. Coelho and A. Firoozabadi. Throughout the repository they are called
**Paper I**, **Paper II** and **Paper III**, in order of publication. The
numbering is ours; the journals do not use it.

| Name | Article | Model | Code | Executable paper |
|---|---|---|---|---|
| Paper I | *Ind. Eng. Chem. Res.* 2026, 65, 20081–20103 | eCPA for CO₂ + H₂O + NaCl | [`code/`](../code/README.md) (package `ecpa`) | [`ecpa_flash_paper.ipynb`](../notebooks/ecpa_flash_paper.ipynb) |
| Paper II | *Energy Fuels* 2026, 40, 23472–23484 | ms-eCPA, parametrization and validation | [`multisalt/`](../multisalt/README.md) (package `ms_ecpa`) | [`ms_ecpa_parametrization_paper.ipynb`](../notebooks/ms_ecpa_parametrization_paper.ipynb) |
| Paper III | *Energy Fuels* 2026, 40, 24867–24882 | ms-eCPA, stability and flash computations, more gases | [`multisalt/`](../multisalt/README.md) (package `ms_ecpa`) | [`ms_ecpa_flash_paper.ipynb`](../notebooks/ms_ecpa_flash_paper.ipynb) |

Each executable paper is a Jupyter notebook with the full text of the
article and its Supporting Information, and code cells that regenerate every
figure and table. [`models.md`](models.md) compares the two models and
explains which one to use.

All three articles are open access under the
[Creative Commons Attribution 4.0 license](https://creativecommons.org/licenses/by/4.0/).

---

## Paper I

J. Moortgat, F. M. Coelho, A. Firoozabadi. Fast and Robust Phase Equilibrium
Computations for CO₂ + H₂O + NaCl Mixtures Using the Electrolyte
Cubic-Plus-Association Equation of State. *Industrial & Engineering Chemistry
Research* **2026**, 65 (37), 20081–20103.
[doi:10.1021/acs.iecr.6c02626](https://doi.org/10.1021/acs.iecr.6c02626)

- **Content.** Phase stability and two-phase flash for CO₂ + H₂O (CPA) and
  CO₂ + H₂O + NaCl (eCPA, with the parameters of Coelho et al., 2025), a
  precomputed solution table for warm starts, and a prototype reservoir
  simulator.
- **Code.** [`code/`](../code/README.md); installable from PyPI as
  `ecpa-flash`.
- **Notebooks.** [`ecpa_flash_paper.ipynb`](../notebooks/ecpa_flash_paper.ipynb)
  (the executable paper) and
  [`ecpa_flash_tutorial.ipynb`](../notebooks/ecpa_flash_tutorial.ipynb)
  (a tutorial on using the package).
- **Reproduction.** [`REPRODUCING_FIGURES.md`](../REPRODUCING_FIGURES.md) gives
  the command for each figure.
- **Published and released numbers.** Releases 1.0.1 and 1.0.2 correct
  defects found after publication. Every reported average absolute relative
  error changes by at most 0.05 percentage points;
  [`CHANGELOG.md`](../CHANGELOG.md) gives the measured differences.

## Paper II

J. Moortgat, F. Mourão Coelho, A. Firoozabadi. Phase Equilibria of CO₂ and
CH₄ in Multisalt Brines: Parametrization and Validation of an Ion-Specific
Electrolyte Cubic-Plus-Association Equation of State. *Energy & Fuels*
**2026**, 40 (42), 23472–23484.
[doi:10.1021/acs.energyfuels.6c04568](https://doi.org/10.1021/acs.energyfuels.6c04568)
([article page](https://pubs.acs.org/enfuem/article/40/42/23472/5449187/Phase-Equilibria-of-CO2-and-CH4-in-Multisalt))

- **Content.** The ion-specific electrolyte CPA model ms-eCPA for CO₂ and
  CH₄ in brines of Na⁺, K⁺, Ca²⁺, Mg²⁺, Cl⁻ and SO₄²⁻: the parameter
  hierarchy, the fitted parameters, and the validation against brine
  densities, osmotic coefficients, gas solubilities in single-salt and
  mixed-salt brines, and enthalpies of CO₂ dissolution.
- **Code.** [`multisalt/`](../multisalt/README.md). The parameters are in
  `multisalt/ms_ecpa/parameters.py`.
- **Notebooks.** [`ms_ecpa_parametrization_paper.ipynb`](../notebooks/ms_ecpa_parametrization_paper.ipynb)
  (the executable paper) and
  [`ms_ecpa_parametrization_tutorial.ipynb`](../notebooks/ms_ecpa_parametrization_tutorial.ipynb)
  (a tutorial on the functions behind it).
- **Reproduction.** The table in [`multisalt/README.md`](../multisalt/README.md)
  gives the command for each figure and table;
  [`multisalt/scripts/FIGURES_MAP.md`](../multisalt/scripts/FIGURES_MAP.md)
  maps each figure script to its published figure numbers.
- **Experimental data.** GasBrineBench v1.1.1
  ([doi:10.5281/zenodo.22898153](https://doi.org/10.5281/zenodo.22898153)),
  downloaded on first use.
- **Published and released numbers.** The released code reproduces the CO₂
  results. It does not reproduce the two CH₄ tables of the Supporting
  Information (Tables S6 and S7), which were computed with an earlier version
  of the code; [`multisalt/PUBLISHED_VS_RELEASED.md`](../multisalt/PUBLISHED_VS_RELEASED.md)
  lists every data point that differs. The notebook shows both sets of values.

## Paper III

J. Moortgat, F. M. Coelho, A. Firoozabadi. Phase Equilibria of CO₂, CH₄, N₂,
H₂, C₂H₆, C₃H₈, and Mixtures in Multisalt Brines: Robust Stability and Flash
Computations with an Ion-Specific Electrolyte Cubic-Plus-Association Equation
of State. *Energy & Fuels* **2026**, 40 (45), 24867–24882.
[doi:10.1021/acs.energyfuels.6c04569](https://doi.org/10.1021/acs.energyfuels.6c04569)
([article page](https://pubs.acs.org/enfuem/article/doi/10.1021/acs.energyfuels.6c04569/5462396/Phase-Equilibria-of-CO2-CH4-N2-H2-C2H6-C3H8-and))

- **Content.** The tangent-plane-distance stability test and the two- and
  three-phase flash for ms-eCPA, a flash over 151,632 states (283–623 K,
  1–1000 bar, eight salt configurations), and the extension of the model to
  N₂, H₂, C₂H₆, C₃H₈ and CH₄–CO₂ mixtures.
- **Code.** [`multisalt/`](../multisalt/README.md), the same package as
  Paper II. The flash is in `multisalt/ms_ecpa/flash.py` (two phases) and
  `multisalt/ms_ecpa/flash3.py` (three phases); the parameters for the
  additional gases are in the result files of `multisalt/pilot_studies/`.
- **Notebooks.** [`ms_ecpa_flash_paper.ipynb`](../notebooks/ms_ecpa_flash_paper.ipynb)
  (the executable paper) and
  [`ms_ecpa_flash_tutorial.ipynb`](../notebooks/ms_ecpa_flash_tutorial.ipynb)
  (a tutorial on the stability and flash functions).
- **Reproduction.** As for Paper II; the flash over the full grid is
  described in [`multisalt/sweep/README.md`](../multisalt/sweep/README.md).
  It is run on an HPC cluster with the SLURM scripts in
  `multisalt/sweep/slurm/`; the notebook plots the shipped results of that
  run.
- **Published and released numbers.** The CH₄ entries of Table 3 come from
  the earlier code of the Paper II CH₄ tables. The released code reproduces
  Figure 6; the bound on the three-phase pressure stated in the text does not
  cover its 302.15 K point. See
  [`multisalt/PUBLISHED_VS_RELEASED.md`](../multisalt/PUBLISHED_VS_RELEASED.md).

---

## Which release reproduces which paper

| Paper | Release | Notes |
|---|---|---|
| Paper I | 1.0.0 (published code), 1.0.2 and later (corrected) | see `CHANGELOG.md` |
| Paper II | 1.1.0 and later | CH₄ tables S6 and S7: see `multisalt/PUBLISHED_VS_RELEASED.md` |
| Paper III | 1.1.0 and later | |

The releases are archived on Zenodo
([doi:10.5281/zenodo.22212018](https://doi.org/10.5281/zenodo.22212018), all
versions).

## How to cite

Cite the paper whose model or results you use, and the software release
(see [`CITATION.cff`](../CITATION.cff)). The BibTeX entries are in the
[repository README](../README.md#citation).
