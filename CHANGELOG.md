# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.2] — 2026-09-13

### Fixed
- **Phase-stability SSI did not iterate.** In `_stability_ssi`
  (`code/ecpa/stability.py`) the "previous" iterate `lnW_old` was recomputed
  from the current `lnphi` and was therefore algebraically identical to the
  direct-substitution update `lnW_new`. The step `lnW_new - lnW_old` was
  identically zero, so every trial reported convergence on its first pass,
  `max_iter`, `tol` and the Jex et al. (2024) acceleration were dead code,
  and the reported tangent-plane distance was a one-step estimate from the
  initial guess rather than a stationary-point value. The iteration now
  carries `W` as its state, as Michelsen's method requires.
  Consequence of the defect: the TPD returned at some conditions was too
  weakly negative to clear the `-0.02` hard threshold in
  `ecpa_stability_flash`, which then classified a genuinely two-phase state
  as single-phase. Example: T = 423.15 K, P = 1000 bar, z_CO2 = 0.5,
  m_NaCl = 1 mol/kg returned `single_phase` with tpd = -0.0139 while the
  K-value flash finds a valid split at beta = 0.532; the corrected
  iteration returns tpd = -0.0382 and the point is correctly two-phase.
- **Phase-stability verdict could rest on a non-converged trial.**
  `ecpa_stability` scored every trial by `1 - sum_W` regardless of whether
  the trial had reached a stationary point. With the iteration now live, an
  aborted trial carries a transient `sum_W`; such trials are no longer
  allowed to drive the stable/unstable decision, and the "stable" message
  now states how many trials actually converged.
- **Permittivity chain derivatives (remaining defects).** In all four
  aqueous kernels (`code/ecpa/elv.py`, `code/ecpa/stability.py` x2,
  `code/benchmark_flash.py`) the `chi1w**2` / `chi4w**2` prefactors of the
  implicit chi-system Jacobian were swapped between the F and G rows
  (`dFdchic`, `dGdchiw`, `dGdV`, `dGdNw`), and `dFddelta` / `dGddelta`
  carried a spurious `-(1 + (delta-1)*...)` grouping instead of the plain
  derivative. The dependent second-derivative lines in
  `code/ecpa/stability.py` were updated to match.
- **Inner lnφ solvers accepted fsolve stagnation points as roots.** Both
  `_lnphi_aq_inner` and `_lnphi_c_inner` accepted a result on `ier == 1`
  alone. That flag only reports that the iterates stopped moving, which can
  happen far from a root, so a non-solution could be returned silently.
  Observed for pure water at T = 273.15 K, P = 75 bar: a returned "root"
  with residual infinity-norm ~5e1 gave a density of 481 kg/m3 against an
  IAPWS-95 reference of 1004 kg/m3. Both solvers now require the residual
  itself to be below 1e-6 and otherwise continue to the next starting point.
- **`_cpa2_label` argument order** (`code/ecpa/scan.py`): `(z, T, P)` was
  passed into a `(T, P_bar, z_co2)` signature. Because the callee swallows
  exceptions, the CPA cross-check silently never fired, so the
  `single_phase_gas` / `single_phase_liquid` buckets in the scan failure
  statistics were not what they claimed to be. Diagnostics only; no
  computed equilibrium was affected.
- **Brent flash returned `inf`/`NaN` as success.** When the salt bracketing
  collapsed onto `ms_aq = 0`, `N_aq = n_salt / x2w` divided by zero and the
  function returned `beta = NaN` without raising. It now raises.
- **`scripts/validate_co2h2o.py`** read `CO2_WATER_exp.parquet` from the
  working directory; the tracked database lives at `CO2/` in the repository
  root, so the script could not run from a fresh clone.
- **The CPA scan chain was broken by three missing `results/` prefixes.**
  `run_parameter_scan.py` writes `results/scan_results_extended.npz`, but
  `run_newton_scan.py` read `scan_results_extended.npz` and wrote
  `scan_newton_results.npz` (both relative to `code/`), while
  `plot_newton_figures.py` reads them from `results/`; and
  `run_warmstart_scan.py` read the bare filename too. The documented
  reproduction route for Figs. 5, 6 and S11 therefore could not run from a
  clean clone — including the `RUN_LONG = True` full-recompute path in
  `notebooks/ecpa_flash_paper.ipynb`, which chains
  `run_parameter_scan.py` -> `run_newton_scan.py` -> `plot_newton_figures.py`.
  All three paths corrected.
- **`code/scripts/README.md`** claimed `run_warmstart_scan.py` "generates
  `results/scan_v4_table.npz`". It does not, and cannot: it works on the
  salt-free CPA (T, P, z) grid of 86x18x19 and reports convergence and
  iteration counts, whereas `scan_v4_table.npz` holds eCPA *ternary*
  compositions on a (T, P, m_s) grid of 361x100x14 with no z axis. The two
  share no grid, no axes and no variables. The script is reporting-only and
  writes nothing; the README entry and the script's stale docstring now say
  so.
- Regression reference values in `tests/test_flash.py` recomputed.

### Changed
- `code/ecpa/__init__.py` now exports `__version__`.
- `REPRODUCING_FIGURES.md`: `results/solution_table.npz` is required by the
  two validation drivers and is *not* distributed (it must be built with
  `scripts/build_solution_table.py`); and `run_warmstart_scan.py` does not
  write `results/scan_v4_table.npz` — that table ships with the repository
  and no script here regenerates it.

### Verification

The corrected chain derivatives were checked against complex-step
differentiation of the code's own defining equations for chi1w and chi4w
(`code/ecpa/elv.py`), which shares no derivative algebra with the kernels.
The corrected expressions agree with the complex-step reference to 3e-14
relative; the previous expressions were in error by up to 1.6 (i.e. 160%)
relative on the chi4w derivatives.

The whole validation and figure pipeline was then run twice — once under
1.0.1 and once under 1.0.2 — and the artifacts compared point by point.
Measured effect of the corrections:

| Quantity | 1.0.1 | 1.0.2 |
|:---|---:|---:|
| AARE, CO2 molality in brine (N=436) | 6.7163% | 6.7152% |
| AARE, x_CO2 salt-free basis (N=99) | 6.8614% | 6.8729% |
| AARE, x_CO2 salt-inclusive (N=36) | 8.5496% | 8.5574% |
| AARE, x_CO2 in the CO2-rich phase (N=28) | 0.3905% | 0.3905% |
| AARE, CO2 + H2O binary, all four quantities | 9.6025 / 24.2607 / 9.6028 / 24.3400% | unchanged to 4 decimals |
| Regime AARE table (Figs. 1, S1, S7) | — | byte-identical |
| Pure-water density vs IAPWS-95 (N=475) | 0.30% | 0.30% (all 514 points agree to 6e-10) |
| Simplified-flash AARE, x_CO2 / m_c / beta / ms_aq | 15.602 / 14.484 / 10.086 / 4.359% | 15.643 / 14.518 / 10.087 / 4.374% |
| Pinned regression x_CO2 at 1, 3, 6 mol/kg | 1.675571e-2, 1.129073e-2, 9.729840e-3 | 1.675865e-2, 1.129390e-2, 9.732861e-3 |

Largest change in any predicted composition: 8.5e-4 relative. Largest change
in any reported AARE: 0.05 percentage points. The converged two-phase set is
unchanged (599/708 brine conditions, 631/631 binary conditions); two brine
conditions move from `flash_failed` to a definite `single_phase` verdict.

Of the figures, 13 of the 15 PDFs are identical once their embedded creation
timestamps are stripped, and 84 of 115 image files are byte-identical. Four
differ by more than 0.1% of their pixels: Figs. S8 and S9 (the simplified-flash
benchmark, 1.8% and 1.4%), one high-temperature panel of Fig. S2 (0.18%), and
Fig. 4 (0.13%) — the last because one isotherm's two-phase count moves from 263
to 262 of 300 sampled feed compositions.

Both notebooks execute end to end with no errors under either version.

### Removed
- The experimental neural-network warm-start (`ecpa/nn_flash.py`, the
  `NNWarmStart` provider, and the `[nn]`/PyTorch extra). It never
  outperformed the solution-table warm-start, its trained checkpoint was
  not distributed, and it is not used in the companion paper. The
  solution-table warm-start (`ScanTableWarmStart`) is unaffected.
- Two stale duplicate files in the experimental database
  (`EXP/CO2-WATER/T423K/EXP4_T423K (copy).txt` and the T473K equivalent).
  The loader already skipped them, so no data changed.

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
  for CO2 solubility).

  *Note added in 1.0.2:* the original entry supported this fix by
  reporting agreement "to better than 4e-9 relative" with an independent
  reimplementation of the same model. That agreement is real, but it is
  not evidence of correctness: the reimplementation was derived from the
  same hand-differentiated chain and inherits the same algebra, so it
  agrees with the kernels whether or not that algebra is right. It did
  not, and could not, detect the further defects fixed in 1.0.2. The
  verification that does carry weight is complex-step differentiation of
  the defining equations themselves, which shares no derivative algebra
  with either implementation; see the 1.0.2 entry.
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
