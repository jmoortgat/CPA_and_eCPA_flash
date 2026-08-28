---
title: 'eCPA Flash: Phase stability and flash calculations for CO2 + H2O + NaCl mixtures with the electrolyte Cubic-Plus-Association equation of state'
tags:
  - Python
  - thermodynamics
  - equation of state
  - phase equilibrium
  - flash calculations
  - electrolytes
  - CO2 storage
  - carbon sequestration
  - geothermal energy
  - reservoir simulation
authors:
  # TODO(user): verify/fill ORCIDs. Moortgat's ORCID below is copied from the
  # geoai-datacubes JOSS submission; Coelho and Firoozabadi ORCIDs must be added.
  - name: Joachim Moortgat
    orcid: 0000-0002-0259-3597
    corresponding: true
    affiliation: '1'
  - name: Felipe Mourão Coelho
    # TODO(user): add ORCID for Felipe Mourão Coelho
    affiliation: '2'
  - name: Abbas Firoozabadi
    # TODO(user): add ORCID for Abbas Firoozabadi
    affiliation: '3'
affiliations:
  - name: 'School of Earth Sciences, The Ohio State University, Columbus, OH, USA'
    index: 1
  - name: 'Universidade Estadual de Campinas (UNICAMP), Campinas, SP, Brazil'
    index: 2
  - name: 'Rice University, Houston, TX, USA'
    index: 3
date: 2026-08-27
bibliography: paper.bib
---

# Summary

`eCPA Flash` is an open-source Python package for phase-stability and
two-phase flash (phase-split) calculations in mixtures of carbon dioxide,
water, and sodium chloride, using the electrolyte Cubic-Plus-Association
(eCPA) equation of state (EoS). The package implements the complete eCPA
model — a Soave–Redlich–Kwong cubic term [@Soave1972], Wertheim association
with $\mathrm{CO_2\text{–}H_2O}$ cross-association [@Kontogeorgis1996], and Debye–Hückel
and Born electrostatic terms [@MaribMogensen2015] — with the recent
parameterization of @Coelho2025. A hierarchical algorithm couples Michelsen
tangent-plane-distance (TPD) stability analysis [@Michelsen1982a] with
accelerated successive substitution and multiple stability initial guesses
[@Jex2024], backed by analytical-Jacobian Newton inner solvers. The result
is 100% flash convergence across more than 9,800 two-phase conditions
spanning $T$ = 0–425 °C and $P$ = 1–1500 bar, at salinities from fresh
water to 6 mol/kg NaCl.

That coverage — from shallow $\mathrm{CO_2}$-storage aquifers to deep hydrothermal
systems — makes the package useful to researchers modeling geological
carbon storage, geothermal energy extraction, and coupled flow and
transport in saline reservoirs, where equilibrium phase compositions and
densities govern injectivity, trapping, and pressure evolution. Beyond the
flash routines themselves, the repository includes a precomputed solution
table for millisecond-scale warm-started calls and a prototype reservoir
simulator demonstrating production-level throughput.

# Statement of need

Practical $\mathrm{CO_2}\text{–brine}$ phase-behavior modeling still leans on
activity-coefficient correlations [@DuanSun2003; @SpycherPruess2005] that
are accurate inside their fitted ranges but are not a single consistent
EoS valid across storage and geothermal conditions; reference equations
of state exist for the pure end-members [@SpanWagner1996; @IAPWS2016] but
not for the mixture. Association EoS such as eCPA capture hydrogen
bonding, cross-association, and ion electrostatics in one framework, yet
published eCPA studies have focused on equilibrium conditions (equality of
fugacities) without solving the coupled material-balance problem. In the
salt-containing ternary this coupling is essential: as $\mathrm{CO_2}$ partitions
into its own phase it removes water, so the equilibrium aqueous molality
depends on the phase fraction (salting-out), and stability testing plus a
full flash must be solved self-consistently. To our knowledge no open,
production-grade stability-plus-flash implementation for an electrolyte
EoS was previously available; `eCPA Flash` fills that gap for the $\mathrm{CO_2 + H_2O + NaCl}$ system with the algorithmic rigor that is standard in
hydrocarbon flash computations.

# Functionality

The repository provides five deliverables:

- **Complete eCPA EoS** (`code/ecpa/`): Debye–Hückel, Born, association,
  and permittivity terms with the parameters of @Coelho2025, plus
  analytical Jacobians for the Newton inner solvers.
- **Salt-free CPA flash** (`code/CPA.py`): Michelsen TPD stability testing
  with six initial guesses and accelerated successive substitution
  [@Jex2024]; 100% convergence over >29,000 binary conditions.
- **eCPA stability + flash** (`code/ecpa/flash.py`): warm-started K-value
  successive substitution with optional Newton polish for the full ternary,
  100% convergence over >9,800 conditions at $T$ = 0–425 °C and
  $P$ = 1–1500 bar.
- **Precomputed 4D solution table** (temperature, pressure, feed
  composition, salinity) shipped as a Parquet file, providing warm starts
  that yield a 3.2× speedup over cold-started successive substitution.
- **Prototype reservoir simulator** (`code/co2brine_simulator.py`): an
  IMPEC scheme with a mixed finite element pressure solver that ran a
  50×50 grid for 300 time steps with zero flash failures.

# Validation and reuse

The algorithms, their derivation, and the full validation study are
documented in the peer-reviewed companion paper [@Moortgat2026]. Headline
results: 6.5% average absolute relative error (AARE) for $\mathrm{CO_2}$ solubility
against more than 1,100 experimental data points, and 0.33% AARE for
aqueous-phase density against the IAPWS-95 standard [@IAPWS2016]. Every
figure in the companion paper can be regenerated from the repository;
`REPRODUCING_FIGURES.md` maps each figure to its data-generation and
plotting scripts.

# Acknowledgements

Abbas Firoozabadi was supported by the member companies of the Reservoir
Engineering Research Institute. Felipe Mourão Coelho thanks FAPESP for
grants #2018/02713-8, #2020/13300-6, and #2021/13068-9.

# References
