# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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

### Earlier development

The full development history (95+ commits, 2026) is preserved in the git
log: CPA salt-free flash with Michelsen stability testing and accelerated
SSI; eCPA extension to CO2 + H2O + NaCl; warm-start solution tables;
validation against experimental CO2-solubility and density data; and the
prototype reservoir-simulation demonstration.
