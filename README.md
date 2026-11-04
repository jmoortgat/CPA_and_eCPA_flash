# CPA and eCPA flash

### Phase equilibria of gases in water and brines with electrolyte Cubic-Plus-Association equations of state

[![Documentation website](https://img.shields.io/badge/documentation-website-0A7D5C.svg?style=flat-square)](https://jmoortgat.github.io/CPA_and_eCPA_flash/)
[![PyPI](https://img.shields.io/pypi/v/ecpa-flash.svg?style=flat-square&color=3775A9&logo=pypi&logoColor=white&label=PyPI%20ecpa-flash)](https://pypi.org/project/ecpa-flash/)
[![License: MIT](https://img.shields.io/badge/License-MIT-BA0C2F.svg?style=flat-square)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![tests](https://github.com/jmoortgat/CPA_and_eCPA_flash/actions/workflows/tests.yml/badge.svg)](https://github.com/jmoortgat/CPA_and_eCPA_flash/actions/workflows/tests.yml)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22212018-1682D4.svg?style=flat-square)](https://doi.org/10.5281/zenodo.22212018)

This repository holds the code, data and executable papers of a series of
three articles by J. Moortgat, F. M. Coelho and A. Firoozabadi on phase
equilibria of CO₂, CH₄, N₂, H₂, C₂H₆, C₃H₈ and their mixtures in water and
brines. In the repository the articles are called Paper I, Paper II and
Paper III, in order of publication:

| Name | Article | Model and code | Executable paper |
|---|---|---|---|
| **Paper I** | Fast and Robust Phase Equilibrium Computations for CO₂ + H₂O + NaCl Mixtures Using the Electrolyte Cubic-Plus-Association Equation of State. *Ind. Eng. Chem. Res.* **2026**, 65, 20081–20103. [doi:10.1021/acs.iecr.6c02626](https://doi.org/10.1021/acs.iecr.6c02626) | eCPA for CO₂ + H₂O + NaCl: [`code/`](code/README.md) | [`ecpa_flash_paper.ipynb`](notebooks/ecpa_flash_paper.ipynb) |
| **Paper II** | Phase Equilibria of CO₂ and CH₄ in Multisalt Brines: Parametrization and Validation of an Ion-Specific Electrolyte Cubic-Plus-Association Equation of State. *Energy Fuels* **2026**, 40, 23472–23484. [doi:10.1021/acs.energyfuels.6c04568](https://doi.org/10.1021/acs.energyfuels.6c04568) | ms-eCPA: [`multisalt/`](multisalt/README.md) | [`ms_ecpa_parametrization_paper.ipynb`](notebooks/ms_ecpa_parametrization_paper.ipynb) |
| **Paper III** | Phase Equilibria of CO₂, CH₄, N₂, H₂, C₂H₆, C₃H₈, and Mixtures in Multisalt Brines: Robust Stability and Flash Computations with an Ion-Specific Electrolyte Cubic-Plus-Association Equation of State. *Energy Fuels* **2026**, available online. [doi:10.1021/acs.energyfuels.6c04569](https://doi.org/10.1021/acs.energyfuels.6c04569) | ms-eCPA: [`multisalt/`](multisalt/README.md) | [`ms_ecpa_flash_paper.ipynb`](notebooks/ms_ecpa_flash_paper.ipynb) |

- [`docs/papers.md`](docs/papers.md): each paper, its code, how to
  reproduce its figures and tables, and which published values the released
  code does not reproduce.
- [`docs/models.md`](docs/models.md): the two models compared, and which one
  to use.
- [API reference](https://jmoortgat.github.io/CPA_and_eCPA_flash/api/) on the documentation website: the public functions of both packages, generated from their docstrings.

---

## The two models

**eCPA of Paper I** ([`code/`](code/README.md), package `ecpa`, on PyPI as
`ecpa-flash`). CO₂ + H₂O + NaCl with the eCPA parameters of Coelho, Franco
and Firoozabadi (2025), at 273–698 K, 1–1500 bar and 0–6 mol/kg NaCl; a
salt-free CPA for CO₂ + H₂O; stability test and two-phase flash with warm
starts from a precomputed solution table; a prototype reservoir simulator.

**ms-eCPA of Papers II and III** ([`multisalt/`](multisalt/README.md),
package `ms_ecpa`). The ion-specific eCPA for CO₂, CH₄, N₂, H₂, C₂H₆, C₃H₈
and CH₄–CO₂ mixtures in brines of Na⁺, K⁺, Ca²⁺, Mg²⁺, Cl⁻ and SO₄²⁻;
stability test and two- and three-phase flash; an optional Fortran library
with the same results; the scripts behind every figure and table of both
papers. The experimental data come from
[GasBrineBench](https://github.com/jmoortgat/GasBrineBench) v1.1.1
([doi:10.5281/zenodo.22898153](https://doi.org/10.5281/zenodo.22898153)).

---

## Installation

Python 3.11 or later. The Paper I solvers are on PyPI:

```bash
pip install ecpa-flash
```

For ms-eCPA, the notebooks, the data and the figure scripts, clone the
repository:

```bash
git clone https://github.com/jmoortgat/CPA_and_eCPA_flash.git
cd CPA_and_eCPA_flash
pip install -e ".[test]"        # the ecpa package (Paper I) and pytest
pip install CoolProp            # used by ms-eCPA
```

`code/environment.yml` defines a conda environment with everything both
packages and the notebooks need:

```bash
conda env create -f code/environment.yml
```

ms-eCPA is used from the source tree: run its scripts from `multisalt/` with
`PYTHONPATH=multisalt` (see [`multisalt/README.md`](multisalt/README.md)).

---

## Notebooks

All notebooks are in [`notebooks/`](notebooks/). They run locally or on
Google Colab, where the setup cell clones the repository.

| Notebook | Content | |
|---|---|---|
| [`ecpa_flash_paper.ipynb`](notebooks/ecpa_flash_paper.ipynb) | Paper I with its Supporting Information, executable | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jmoortgat/CPA_and_eCPA_flash/blob/main/notebooks/ecpa_flash_paper.ipynb) |
| [`ecpa_flash_tutorial.ipynb`](notebooks/ecpa_flash_tutorial.ipynb) | Tutorial for the `ecpa` package of Paper I | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jmoortgat/CPA_and_eCPA_flash/blob/main/notebooks/ecpa_flash_tutorial.ipynb) |
| [`ms_ecpa_parametrization_paper.ipynb`](notebooks/ms_ecpa_parametrization_paper.ipynb) | Paper II with its Supporting Information, executable | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jmoortgat/CPA_and_eCPA_flash/blob/main/notebooks/ms_ecpa_parametrization_paper.ipynb) |
| [`ms_ecpa_parametrization_tutorial.ipynb`](notebooks/ms_ecpa_parametrization_tutorial.ipynb) | Tutorial for ms-eCPA: parameters, brine density, osmotic coefficients, gas solubility in single and mixed brines (Paper II) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jmoortgat/CPA_and_eCPA_flash/blob/main/notebooks/ms_ecpa_parametrization_tutorial.ipynb) |
| [`ms_ecpa_flash_paper.ipynb`](notebooks/ms_ecpa_flash_paper.ipynb) | Paper III with its Supporting Information, executable | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jmoortgat/CPA_and_eCPA_flash/blob/main/notebooks/ms_ecpa_flash_paper.ipynb) |
| [`ms_ecpa_flash_tutorial.ipynb`](notebooks/ms_ecpa_flash_tutorial.ipynb) | Tutorial for ms-eCPA: stability test, two- and three-phase flash, gas mixtures, other gases, Fortran backend (Paper III) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jmoortgat/CPA_and_eCPA_flash/blob/main/notebooks/ms_ecpa_flash_tutorial.ipynb) |

Each executable paper holds the full text of the article and its Supporting
Information, with hidden code cells that regenerate every figure and table.
By default the cells plot from result files shipped with the repository; set
`RUN_LONG = True` in the setup cell to recompute them from the equation of
state.

---

## Repository structure

```
CPA_and_eCPA_flash/
├── README.md                  # this page
├── docs/
│   ├── papers.md              # the three papers and their code
│   └── models.md              # the two models compared
├── notebooks/                 # executable papers and tutorials
├── code/                      # Paper I: eCPA for CO2 + H2O + NaCl (package ecpa)
│   └── README.md
├── EXP/, CO2/                 # experimental data of Paper I
├── REPRODUCING_FIGURES.md     # Paper I: command for each figure
├── multisalt/                 # Papers II and III: ms-eCPA (package ms_ecpa)
│   ├── README.md
│   └── PUBLISHED_VS_RELEASED.md
├── tests/                     # tests of the Paper I package
├── paper.md, paper.bib        # JOSS paper on the Paper I software
├── pyproject.toml             # pip install of the Paper I package
└── LICENSE, CITATION.cff, CHANGELOG.md, CONTRIBUTING.md, CODE_OF_CONDUCT.md
```

---

## Citation

Please cite the paper whose model or results you use, and the software
(see [`CITATION.cff`](CITATION.cff); the Zenodo DOI above covers all
releases).

```bibtex
@article{moortgat2026ecpa,
  author  = {Moortgat, Joachim and Coelho, Felipe Mour{\~a}o and Firoozabadi, Abbas},
  title   = {Fast and Robust Phase Equilibrium Computations for {CO}$_2$ + {H}$_2${O} + {NaCl}
             Mixtures Using the Electrolyte Cubic-Plus-Association Equation of State},
  journal = {Industrial \& Engineering Chemistry Research},
  year    = {2026},
  volume  = {65},
  number  = {37},
  pages   = {20081--20103},
  doi     = {10.1021/acs.iecr.6c02626}
}

@article{moortgat2026msecpa1,
  author  = {Moortgat, Joachim and Mour{\~a}o Coelho, Felipe and Firoozabadi, Abbas},
  title   = {Phase Equilibria of {CO}$_2$ and {CH}$_4$ in Multisalt Brines: Parametrization
             and Validation of an Ion-Specific Electrolyte Cubic-Plus-Association Equation of State},
  journal = {Energy \& Fuels},
  year    = {2026},
  volume  = {40},
  number  = {42},
  pages   = {23472--23484},
  doi     = {10.1021/acs.energyfuels.6c04568}
}

@article{moortgat2026msecpa2,
  author  = {Moortgat, Joachim and Coelho, Felipe Mour{\~a}o and Firoozabadi, Abbas},
  title   = {Phase Equilibria of {CO}$_2$, {CH}$_4$, {N}$_2$, {H}$_2$, {C}$_2${H}$_6$, {C}$_3${H}$_8$,
             and Mixtures in Multisalt Brines: Robust Stability and Flash Computations with an
             Ion-Specific Electrolyte Cubic-Plus-Association Equation of State},
  journal = {Energy \& Fuels},
  year    = {2026},
  doi     = {10.1021/acs.energyfuels.6c04569}
}
```

The eCPA of Paper I uses the parametrization of

```bibtex
@article{coelho2025ecpa,
  author  = {Coelho, Felipe Mour{\~a}o and Franco, Lu{\'i}s Fernando Mercier and Firoozabadi, Abbas},
  title   = {Phase Equilibria of {CO}$_2$--Water and {CO}$_2$--Brine at High Temperatures:
             From {Monte Carlo} Simulations to the Equation of State},
  journal = {Industrial \& Engineering Chemistry Research},
  volume  = {64},
  number  = {16},
  pages   = {8492--8505},
  year    = {2025},
  doi     = {10.1021/acs.iecr.5c00134}
}
```

---

## Contributing and license

Bug reports, questions and pull requests are welcome; see
[CONTRIBUTING.md](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md).
The code is under the [MIT license](LICENSE). The three articles are open
access under the
[Creative Commons Attribution 4.0 license](https://creativecommons.org/licenses/by/4.0/).
