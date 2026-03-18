# eCPA Flash — Open-Source Phase Equilibrium for CO₂ + H₂O + NaCl

Fast, robust phase stability and flash calculations for the
**CO₂ + H₂O + NaCl** ternary system using the electrolyte Cubic-Plus-Association
(eCPA) equation of state of [Coelho, Franco & Firoozabadi (2025)][coelho2025].

---

## Overview

This repository implements:

1. **A complete eCPA EoS** (`ecpa/`) — all Debye–Hückel, Born, association, and permittivity
   terms, with the parameters and temperature-dependent binary interaction coefficients from
   Coelho et al. (2025).

2. **A generalized salt-free CPA flash** (`CPA2.py`) — Michelsen TPD stability test + SSI flash
   for the CO₂ + H₂O binary, using the same eCPA parameters.

3. **eCPA stability + flash** (`ecpa/stability.py`, `ecpa/flash.py`) — extension to the
   full ternary system. Replaces the earlier outer Brent loop over aqueous molality with
   a direct warm-started SSI approach.

4. **A 4D precomputed solution table** — 31T × 30P × 18*z* × 14*m*ₛ = 234,360 cells covering
   *T* = 283–728 K, *P* = 1–1500 bar, CO₂ mole fraction *z* = 0.05–0.90, NaCl molality
   *m*ₛ = 0–6 mol/kg. Enables warm-started flash that is **3.2× faster** than cold-start SSI.

5. **Validated VLE and density predictions** — AARE 8.2% (CO₂ in water), 6.8% (CO₂ in brine
   at moderate conditions), 0.33% (aqueous density after Péneloux H₂O shift optimisation).

All code is open source; see [paper/main.tex](paper/main.tex) for the full journal-paper
description and the supplemental information section for function-level documentation.

---

## Requirements

```
python >= 3.11
numpy
scipy
pandas
pyarrow        # for parquet I/O
matplotlib
jupyter        # optional, for the notebook
```

Install with:
```bash
pip install numpy scipy pandas pyarrow matplotlib jupyter
```

---

## Quick start

```python
from ecpa.parameters import make_params
from ecpa.solution_table import make_solution_guess_fn
from ecpa.flash import flash_co2_h2o_salt_fast
import numpy as np

params   = make_params()
npz      = np.load('results/solution_table.npz')
guess_fn = make_solution_guess_fn(
    npz['T_grid'], npz['logP_grid'], npz['z_grid'], npz['ms_grid'],
    npz['sol_filled'], npz['stable'],
)

# Flash at T=350 K, P=100 bar, z_CO2=0.3, ms=1.0 mol/kg
result = flash_co2_h2o_salt_fast(
    T=350.0, P=100.0, z=0.3, ms=1.0,
    params=params, guess_fn=guess_fn,
)
print(result)
```

For the salt-free binary:
```python
import CPA2
r = CPA2.flash_co2_h2o_tpz(T=323.15, P_bar=100.0, z_co2=0.3)
print(r['phase'], r['x'], r['tie']['rho_mass'] * 1000, 'kg/m³')
```

---

## Repository structure

```
Claude_code/
├── ecpa/                          # Main eCPA package
│   ├── constants.py               # All EoS parameters (edit Péneloux shifts here)
│   ├── parameters.py              # Assembles params dict (make_params())
│   ├── elv.py                     # ELV residual system + Jacobian
│   ├── flash.py                   # flash_co2_h2o_salt_ssi, flash_co2_h2o_salt_fast
│   ├── stability.py               # ecpa_stability, stability_map, stability_ms_scan
│   ├── solution_table.py          # build_solution_table, make_solution_guess_fn
│   ├── scan.py                    # scan_flash — grid scan
│   ├── envelope.py                # find_envelope_from_scan — phase boundary
│   ├── guess_table.py             # make_guess_fn — CPA2 lookup table init
│   ├── validate_co2h2o.py         # CO2+H2O binary validation
│   ├── validate_nacl.py           # CO2+NaCl ternary validation
│   ├── plotting.py                # Shared plot utilities
│   ├── exp_data.py                # Experimental data parsers
│   └── utils.py                   # Numerical helpers
│
├── CPA2.py                        # Self-contained salt-free CPA binary flash
│
├── eCPA_notebook.ipynb            # Interactive Jupyter notebook (sections 7–8)
│
├── _run_solution_table.py         # Rebuild 4D solution table (~1–4 h)
├── _add_ms0_to_solution_table.py  # Add ms=0 slice from binary database
├── _extend_solution_table_high_T.py  # Extend to 623 K
├── _run_validation.py             # CO2+NaCl validation end-to-end
├── _run_validation_co2h2o.py      # CO2+H2O validation end-to-end
├── _validate_density_co2h2o.py    # Aqueous density validation + Péneloux optimisation
├── _run_benchmark.py              # Benchmark: fast flash vs cold SSI
├── _run_smooth_salty.py           # Smooth eCPA curves at fixed ms
├── _replot_co2nacl.py             # Replot CO2+NaCl figures from saved parquet
├── _replot_co2h2o.py              # Replot CO2+H2O figures from saved parquet
│
├── results/
│   ├── solution_table.npz         # 4D table (24T×30P×18z×8ms, 103,680 cells)
│   ├── validation_co2h2o.parquet  # 631-row CO2+H2O validation results
│   ├── validation_co2nacl.parquet # 423-row CO2+NaCl validation results
│   └── density_co2h2o.parquet     # 37-row density validation results
│
├── figures/
│   ├── co2h2o/                    # Per-temperature CO2+H2O VLE figures
│   ├── co2nacl/                   # Per-temperature CO2+NaCl VLE figures
│   └── density/                   # Density validation figures
│
├── EXP/
│   ├── CO2-WATER/                 # Binary experimental data (txt files)
│   └── CO2-NaCl/                  # Ternary experimental data (txt files)
│
└── paper/
    ├── main.tex                   # Journal paper (elsarticle)
    └── refs.bib                   # BibTeX references
```

---

## Key functions

### `ecpa/flash.py`

| Function | Description |
|---|---|
| `flash_co2_h2o_salt_ssi(T, P, z, ms, params)` | Robust cold-start SSI flash (ω=0.7). Use for single calls or unknown conditions. |
| `flash_co2_h2o_salt_fast(T, P, z, ms, params, guess_fn)` | Fast warm-started flash using the solution table. **Use this in production.** |

### `ecpa/stability.py`

| Function | Description |
|---|---|
| `ecpa_stability(T, P, z, ms, params)` | Michelsen TPD test. Returns `(tpd_min, trial_comp, converged)`. |
| `stability_map(T_range, P_range, z, ms, params, n_workers)` | Parallel 2D stability scan. |
| `stability_ms_scan(T, P_range, ms_range, z, params)` | Serial salinity scan at fixed T. |

### `ecpa/solution_table.py`

| Function | Description |
|---|---|
| `build_solution_table(T_grid, logP_grid, z_grid, ms_grid, params, n_workers)` | Build table from scratch. |
| `make_solution_guess_fn(T_grid, logP_grid, z_grid, ms_grid, sol, stable)` | Build interpolating guess function from saved table. |

### `CPA2.py`

| Function | Description |
|---|---|
| `flash_co2_h2o_tpz(T, P_bar, z_co2, vshift_h2o, vshift_co2)` | Salt-free CO₂+H₂O binary flash. Returns phase, compositions, Z-factors, `rho_mass` [kg/L]. |

---

## Péneloux volume shifts

All shifts are defined in `ecpa/constants.py` and propagated through `make_params()`:

| Parameter | Value | Description |
|---|---|---|
| `Penelouxs` | −53.5 cm³/mol | NaCl shift — from Coelho et al. (2025) |
| `Peneloux_H2O` | +0.1105 cm³/mol | H₂O shift — **optimised in this work** |
| `Peneloux_CO2` | 0 | CO₂ shift — off by default |

The H₂O shift reduces aqueous-phase density AARE from 0.76% to 0.33% (37 experimental
points, *T* = 288–473 K). The shift is isofugacity-preserving and does **not** affect
phase compositions or VLE predictions.

For `CPA2.py`, pass shifts explicitly:
```python
r = CPA2.flash_co2_h2o_tpz(T=323, P_bar=100, z_co2=0.3,
                             vshift_h2o=1.105e-7,   # m³/mol
                             vshift_co2=0.0)
rho_kg_m3 = r['tie']['rho_mass'][0] * 1000
```

---

## Rebuilding the solution table

The pre-built table is stored in `results/solution_table.npz`. To rebuild from scratch:

```bash
python _run_solution_table.py
```

This takes 1–4 hours depending on hardware (uses all available CPU cores via
`multiprocessing`). After completion, add the ms=0 slice, the high-T extension,
and finally the full extension to T=728K and ms=6:

```bash
python _add_ms0_to_solution_table.py
python _extend_solution_table_high_T.py
python _extend_solution_table_full.py
```

---

## Running validations

```bash
# CO₂ + H₂O binary VLE
python _run_validation_co2h2o.py

# CO₂ + NaCl ternary VLE (all T, all ms — requires extended solution table)
python _run_validation_full.py

# Aqueous-phase density
python _validate_density_co2h2o.py

# Performance benchmark
python _run_benchmark.py
```

Results are saved to `results/` (parquet files) and figures to `figures/`.

---

## Performance summary

| Method | Mean SSI iters | Time per call | Speedup |
|---|---|---|---|
| Cold-start SSI (ω=0.7, Wilson init) | 11.7 | ~10 ms | 1× |
| **Fast flash (table + selective stability)** | **3.3** | **~3 ms** | **3.2×** |
| Fast flash + forced stability check | 3.3 | ~18 ms | 0.6× |

Benchmark conditions: *T* = 398 K, *z* = 0.5, *m*ₛ = 1.0 mol/kg, 30 pressure points.

---

## Validation summary

| System | Quantity | *N* | AARE |
|---|---|---|---|
| CO₂ + H₂O (CPA) | *x*_CO₂ in water | 460 | 8.9% |
| CO₂ + H₂O (CPA) | *y*_H₂O in CO₂ phase | 363 | 25.3% |
| CO₂ + H₂O (eCPA) | *x*_CO₂ in water | 451 | 8.2% |
| CO₂ + H₂O (eCPA) | *y*_H₂O in CO₂ phase | 357 | 23.0% |
| CO₂ + NaCl (eCPA) | *m*_CO₂ [mol/kg] | 440 | 6.9% |
| CO₂ + NaCl (eCPA) | *x*_CO₂ [salt-free] | 99 | 7.0% |
| CO₂ + NaCl (eCPA) | *x*_CO₂ in CO₂ phase | 28 | 0.4% |
| Aqueous density (CPA+eCPA) | ρ_W [kg/m³] | 37 | **0.33%** |

---

## Citation

If you use this code, please cite:

> [Authors]. Fast and Robust Phase Equilibrium Computation for CO₂ + H₂O + NaCl Mixtures
> Using the eCPA Equation of State. *[Journal]*, [Year].

And the underlying eCPA parametrisation:

> Coelho, L., Franco, L. F. M., & Firoozabadi, A. (2025). Phase Equilibria of CO₂–Water
> and CO₂–Brine at High Temperatures: From Monte Carlo Simulations to the Equation of State.
> *Ind. Eng. Chem. Res.*, **64**(16), 8492–8505. https://doi.org/10.1021/acs.iecr.5c00134

---

## License

[TODO: add license file]

[coelho2025]: https://doi.org/10.1021/acs.iecr.5c00134
