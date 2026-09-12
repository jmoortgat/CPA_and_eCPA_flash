# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Removed
- The experimental neural-network warm-start (`ecpa/nn_flash.py`, the
  `NNWarmStart` provider, and the `[nn]`/PyTorch extra). It never
  outperformed the solution-table warm-start, its trained checkpoint was
  not distributed, and it is not used in the companion paper. The
  solution-table warm-start (`ScanTableWarmStart`) is unaffected.

## [1.0.1] — 2026-09-12

### Fixed
- Indexing slip in the electrolyte permittivity chain of the aqueous eCPA
  kernels: the Wertheim chain-derivative terms `dFdchic` and `dFdV` used
  the Na+ mole fraction `x2w` where the CO2 fraction `x4w` belongs
  (`code/ecpa/elv.py` lines 152/155; `code/ecpa/stability.py` lines
  540/542, 701/703 and the second-derivative lines 877/880;
  `code/benchmark_flash.py` lines 630/633).
- Sign slip in the closed-form solution for `Ndchi1WdNc` (the composition
  derivative of the water association fraction) in both aqueous kernels of
  `code/ecpa/stability.py`: the chain term read
  `dFdchic*(-dGddelta*NddeltadN4)` where the correct form is
  `dFdchic*(dGddelta*NddeltadN4)`, as in `code/ecpa/elv.py` and
  `code/benchmark_flash.py`, which solve the same relation iteratively and
  were unaffected. This slip only entered the K-value flash / stability
  route, not the ELV route used for the published figures.
- Both slips enter only through the salt-dependent permittivity term, so
  all salt-free results are unchanged. With salt, predicted compositions
  shift by roughly 1e-4 to 1e-2 relative at 4-6 mol/kg NaCl through the
  ELV route, and by up to ~3e-2 relative through the K-value flash — well
  within the ~5% experimental scatter and the reported AAREs (6.9-8.2%
  for CO2 solubility). After the fix, the aqueous kernels agree with an
  independent reimplementation of the same model to better than 4e-9
  relative on solved equilibrium compositions at 1-6 mol/kg (previously
  up to 4e-3).
- Regression reference values in `tests/test_flash.py` recomputed for the
  corrected model (x_CO2_aq shifts of +3.1e-3, +1.8e-2, and +2.8e-2
  relative at the 1, 3, and 6 mol/kg pinned points).

## [1.0.0] — 2026-08-27

First public release, accompanying the acceptance of the companion journal
article in *Industrial & Engineering Chemistry Research*.

### Added
- Packaging (`pyproject.toml`): the `ecpa` package is installable with
  `pip install -e .`; optional extras `[nn]` (PyTorch warm-start) and
  `[test]` (pytest).
- Automated test suite (`tests/`) with regression checks of the eCPA
  K-value flash against pinned reference values, and continuous
  integration via GitHub Actions.
- Community and citation files: `LICENSE` (MIT), `CITATION.cff`,
  `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`.

### Fixed
- `ecpa.envelope.build_cpa2_envelope` imported the parallel worker under
  its pre-rename module name (`cpa2_worker`); it now imports
  `cpa_worker`, matching the module shipped in the repository.
- Several benchmark/plot scripts contained machine-specific absolute
  paths; they now resolve paths relative to the repository.
- Restored `scripts/plot_speedup_figures.py` (generates the two panels of
  paper Figure 6), which had been dropped during the repository
  reorganization; `run_parameter_scan.py` now writes its npz output to
  `results/` where the plotting scripts expect it.
- `REPRODUCING_FIGURES.md` and `code/scripts/README.md` updated to the
  accepted paper's final figure numbering (S1–S12), including the correct
  scripts for the density figures S5/S6
  (`benchmark_pure_water_density.py`, optional `iapws` dependency).
- Removed stale committed PDF copies of `README.md` and
  `REPRODUCING_FIGURES.md` (GitHub renders the markdown natively).

### Earlier development

The full development history (100+ commits, March–August 2026) is
preserved in the git log: CPA salt-free flash with Michelsen stability
testing and accelerated SSI; eCPA extension to CO2 + H2O + NaCl;
warm-start solution tables; validation against experimental
CO2-solubility and density data; and the prototype reservoir-simulation
demonstration.
