# eCPA Flash — Open-Source Phase Equilibrium for CO₂ + H₂O + NaCl

Fast, robust phase stability and flash calculations for the
**CO₂ + H₂O + NaCl** ternary system using the electrolyte Cubic-Plus-Association
(eCPA) equation of state of [Coelho, Franco & Firoozabadi (2025)][coelho2025].

---

## Overview

This repository implements:

1. **A complete eCPA EoS** (`code/ecpa/`) — all Debye–Hückel, Born, association, and permittivity
   terms, with the parameters and temperature-dependent binary interaction coefficients from
   Coelho et al. (2025).

2. **A generalized salt-free CPA flash** (`code/CPA.py`) — Michelsen TPD stability test with six
   initial guesses and accelerated SSI ([Jex et al., 2024](https://doi.org/10.2118/219490-PA))
   for the CO₂ + H₂O binary. The hierarchical algorithm (stability → flash with K from lowest
   TPD) achieves 100% convergence across >29,000 conditions and a 2× iteration-count reduction
   over standard SSI.

3. **eCPA stability + flash** (`code/ecpa/stability.py`, `code/ecpa/flash.py`) — extension to the
   full ternary system using a direct warm-started SSI approach.

4. **A precomputed solution table** (`code/results/CPA_ELV_all.parquet`) — 31T × 30P × 18*z* × 14*m*ₛ
   cells covering *T* = 283–728 K, *P* = 1–1500 bar, CO₂ mole fraction *z* = 0.05–0.90, NaCl
   molality *m*ₛ = 0–6 mol/kg. Enables warm-started flash that is **3.2× faster** than cold-start SSI.

5. **Validated VLE and density predictions** — AARE 6.5% (CO₂ solubility in brine),
   0.33% (aqueous density after Péneloux H₂O shift optimisation), 100% flash convergence
   across >9,800 conditions spanning *T* = 0–425 °C, *P* = 1–1500 bar.

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

```bash
cd code/
```

```python
import sys
sys.path.insert(0, 'code')   # or: cd code/ before launching Python

from ecpa.parameters import make_params
from ecpa.guess_table import make_guess_fn
from ecpa.flash import flash_co2_h2o_salt_kv
import pandas as pd

params   = make_params()
df       = pd.read_parquet('code/results/CPA_ELV_all.parquet')
guess_fn = make_guess_fn(df)

# Flash at T=350 K, P=100 bar, z_CO2=0.3, ms=1.0 mol/kg NaCl
result = flash_co2_h2o_salt_kv(
    T=350.0, P=100.0, z=0.3, ms=1.0,
    params=params, guess_fn=guess_fn,
)
print(result)
```

For the salt-free binary:
```python
import sys; sys.path.insert(0, 'code')
import CPA
r = CPA.flash_co2_h2o_tpz(T=323.15, P_bar=100.0, z_co2=0.3)
print(r['phase'], r['x'], r['tie']['rho_mass'] * 1000, 'kg/m³')
```

---

## Repository structure

```
CPA_and_eCPA_flash/
├── README.md
│
└── code/
    ├── ecpa/                          # Main eCPA package
    │   ├── constants.py               # All EoS parameters (edit Péneloux shifts here)
    │   ├── parameters.py              # Assembles params dict (make_params())
    │   ├── elv.py                     # ELV residual system + Jacobian
    │   ├── flash.py                   # flash_co2_h2o_salt_kv (production flash)
    │   ├── stability.py               # ecpa_stability, stability_map
    │   ├── solution_table.py          # build_solution_table, make_solution_guess_fn
    │   ├── guess_table.py             # make_guess_fn — parquet-based warm-start init
    │   ├── flash_simplified.py        # Simplified flash for T < 80 °C
    │   ├── scan.py                    # scan_flash — grid scan
    │   ├── envelope.py                # find_envelope_from_scan — phase boundary
    │   ├── validate_co2h2o.py         # CO2+H2O binary validation helpers
    │   ├── validate_nacl.py           # CO2+NaCl ternary validation helpers
    │   ├── plotting.py                # Shared plot utilities
    │   ├── exp_data.py                # Experimental data loader (parquet)
    │   └── utils.py                   # Numerical helpers
    │
    ├── CPA.py                         # Salt-free CPA binary: flash, stability, accelerated SSI
    ├── co2brine_simulator.py          # Prototype IMPEC reservoir simulator
    ├── eCPA_notebook.ipynb            # Interactive Jupyter notebook
    │
    ├── scripts/                       # Validation, benchmark, and figure scripts
    │   ├── validate_co2h2o.py         # CO2+H2O binary VLE validation
    │   ├── validate_co2nacl_full.py   # CO2+NaCl ternary VLE validation
    │   ├── validate_density.py        # Aqueous-phase density validation
    │   ├── build_solution_table.py    # Rebuild 4D solution table (~1–4 h)
    │   ├── run_benchmark.py           # Benchmark: fast flash vs cold SSI
    │   ├── run_parameter_scan.py      # CPA grid scan (T, P, z parameter space)
    │   ├── plot_co2h2o_figures.py     # Publication figures: CO2+H2O VLE
    │   ├── plot_co2nacl_figures.py    # Publication figures: CO2+NaCl VLE
    │   └── ...                        # Additional benchmark and plot scripts
    │
    └── results/
        └── CPA_ELV_all.parquet        # Precomputed solution table (25 MB)
```

---

## Key functions

### `ecpa/flash.py`

| Function | Description |
|---|---|
| `flash_co2_h2o_salt_kv(T, P, z, ms, params, guess_fn)` | Hierarchical stability+flash with K-value SSI. **Use this in production.** |
| `flash_co2_h2o_salt_ssi(T, P, z, ms, params)` | Robust cold-start SSI flash (ω=0.7). For single calls without a guess table. |

### `ecpa/stability.py`

| Function | Description |
|---|---|
| `ecpa_stability(T, P, z, ms, params)` | Michelsen TPD test. Returns `(tpd_min, trial_comp, converged)`. |
| `stability_map(T_range, P_range, z, ms, params, n_workers)` | Parallel 2D stability scan. |

### `ecpa/solution_table.py`

| Function | Description |
|---|---|
| `build_solution_table(T_grid, logP_grid, z_grid, ms_grid, params, n_workers)` | Build table from scratch. |
| `make_solution_guess_fn(T_grid, logP_grid, z_grid, ms_grid, sol, stable)` | Build interpolating guess function from an npz table. |

### `CPA.py`

| Function | Description |
|---|---|
| `flash_co2_h2o_tpz(T, P_bar, z_co2, vshift_h2o, vshift_co2)` | Salt-free CO₂+H₂O binary flash. Returns phase, compositions, Z-factors, `rho_mass` [kg/L]. |
| `flash_co2_h2o_tpz_robust(T, P_bar, z_co2, **kwargs)` | Hierarchical flash: stability → best-K flash → Wilson fallback. **100% convergence.** |
| `stability_test(T, P_bar, z, ..., accelerated=True)` | Michelsen TPD test with 6 initial guesses ([Jex et al., 2024](https://doi.org/10.2118/219490-PA)). |
| `tie_line_two_comp(T, P_bar, ..., accelerated=True)` | SSI flash with dominant-eigenvalue acceleration. |

---

## Péneloux volume shifts

All shifts are defined in `code/ecpa/constants.py` and propagated through `make_params()`:

| Parameter | Value | Description |
|---|---|---|
| `Penelouxs` | −53.5 cm³/mol | NaCl shift — from Coelho et al. (2025) |
| `Peneloux_H2O` | +0.1105 cm³/mol | H₂O shift — **optimised in this work** |
| `Peneloux_CO2` | 0 | CO₂ shift — off by default |

The H₂O shift reduces aqueous-phase density AARE from 0.76% to 0.33% (37 experimental
points, *T* = 288–473 K). The shift is isofugacity-preserving and does **not** affect
phase compositions or VLE predictions.

To apply the H₂O shift in `CPA.py`:
```python
r = CPA.flash_co2_h2o_tpz(T=323, P_bar=100, z_co2=0.3,
                            vshift_h2o=1.105e-7,   # m³/mol
                            vshift_co2=0.0)
rho_kg_m3 = r['tie']['rho_mass'][0] * 1000
```

---

## Running validations

From the repo root:

```bash
cd code

# CO₂ + H₂O binary VLE
python scripts/validate_co2h2o.py

# CO₂ + NaCl ternary VLE
python scripts/validate_co2nacl_full.py

# Aqueous-phase density
python scripts/validate_density.py

# Performance benchmark
python scripts/run_benchmark.py
```

Results are saved to `code/results/` (parquet files).

### Rebuilding the solution table

The pre-built table is `code/results/CPA_ELV_all.parquet`. To rebuild from scratch (1–4 h):

```bash
cd code
python scripts/build_solution_table.py
```

---

## Performance summary

### eCPA ternary flash (solution-table warm start)

| Method | Mean SSI iters | Time per call | Speedup |
|---|---|---|---|
| Cold-start SSI (ω=0.7, Wilson init) | 11.7 | ~10 ms | 1× |
| **Fast flash (table warm-start)** | **3.3** | **~3 ms** | **3.2×** |

Benchmark conditions: *T* = 398 K, *z* = 0.5, *m*ₛ = 1.0 mol/kg, 30 pressure points.

### Salt-free CPA binary flash (accelerated SSI)

| Strategy | Convergence | Mean iters | Description |
|---|---|---|---|
| Standard SSI + Wilson K | 96.0% | 46.6 | Baseline |
| Accelerated SSI + Wilson K | 97.1% | 17.0 | Jex acceleration only |
| **Robust (hierarchical)** | **100%** | **12.0** | Stability → best-K flash → Wilson fallback |

Tested at all 631 experimental CO₂+H₂O data points (*T* = 273–623 K, *P* = 5–3500 bar).

---

## Validation summary

| System | Quantity | *N* | AARE |
|---|---|---|---|
| CO₂ + H₂O (CPA) | *x*_CO₂ in water | 460 | 8.9% |
| CO₂ + H₂O (eCPA) | *x*_CO₂ in water | 451 | 8.2% |
| CO₂ + NaCl (eCPA) | *m*_CO₂ [mol/kg] | 440 | 6.9% |
| CO₂ + NaCl (eCPA) | *x*_CO₂ [salt-free] | 99 | 7.0% |
| Aqueous density (CPA+eCPA) | ρ_W [kg/m³] | 37 | **0.33%** |

---

## Citation

If you use this code, please cite:

> Moortgat, J., Coelho, F. M., & Firoozabadi, A. Fast and Robust Phase Equilibrium
> Computations for CO₂ + H₂O + NaCl Mixtures Using the Electrolyte Cubic-Plus-Association
> Equation of State. *Chemical Engineering Journal* (2026, under review).

And the underlying eCPA parametrisation:

> Coelho, L., Franco, L. F. M., & Firoozabadi, A. (2025). Phase Equilibria of CO₂–Water
> and CO₂–Brine at High Temperatures: From Monte Carlo Simulations to the Equation of State.
> *Ind. Eng. Chem. Res.*, **64**(16), 8492–8505. https://doi.org/10.1021/acs.iecr.5c00134

---

## License

[TODO: add license file]

[coelho2025]: https://doi.org/10.1021/acs.iecr.5c00134
