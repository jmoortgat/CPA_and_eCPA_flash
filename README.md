# eCPA Flash

### Fast, robust phase equilibrium for CO₂ + H₂O + NaCl

[![License: MIT](https://img.shields.io/badge/License-MIT-BA0C2F.svg?style=flat-square)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![tests](https://github.com/jmoortgat/CPA_and_eCPA_flash/actions/workflows/tests.yml/badge.svg)](https://github.com/jmoortgat/CPA_and_eCPA_flash/actions/workflows/tests.yml)
[![Paper](https://img.shields.io/badge/paper-in%20press%20%40%20I%26ECR-2563EB.svg?style=flat-square)](https://pubs.acs.org/journal/iecred)
![Coverage](https://img.shields.io/badge/T%20range-0–425%20°C-teal.svg?style=flat-square)
![Coverage](https://img.shields.io/badge/P%20range-1–1500%20bar-teal.svg?style=flat-square)
<!-- Add after Zenodo archive is minted:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
-->
<!-- Add after JOSS submission:
[![status](https://joss.theoj.org/papers/XXXX/status.svg)](https://joss.theoj.org/papers/XXXX)
-->

Phase stability and flash calculations for the **CO₂ + H₂O + NaCl** ternary system
using the electrolyte Cubic-Plus-Association (eCPA) equation of state of
[Coelho, Franco & Firoozabadi (2025)][coelho2025].
Covers the full spectrum from shallow CO₂ storage aquifers to deep geothermal reservoirs.

---

## Highlights

| | |
|:---|:---|
| ✅ **100% flash convergence** | across >9,800 conditions, *T* = 0–425 °C, *P* = 1–1500 bar |
| ⚡ **3.2× faster** than cold-start SSI | via precomputed 4D solution table warm-start |
| 🎯 **6.5% AARE** on CO₂ solubility | validated against >1,100 experimental data points |
| 🔬 **0.30% AARE** on pure-water density | 475 IAPWS-95 reference conditions |
| 🏗️ **Reservoir simulator demo** | 50×50 grid, 300 time steps, zero flash failures |

---

## What this repository provides

1. **Complete eCPA EoS** (`code/ecpa/`) — Debye–Hückel, Born, association, and permittivity
   terms with parameters from Coelho et al. (2025); analytical Jacobians for both Newton
   inner solvers.

2. **Salt-free CPA flash** (`code/CPA.py`) — Michelsen TPD stability test with six
   initial guesses and accelerated SSI ([Jex et al., 2024](https://doi.org/10.2118/219490-PA)).
   Hierarchical algorithm achieves **100% convergence** across >29,000 conditions with a
   **2.96×** iteration-count reduction over standard SSI.

3. **eCPA stability + flash** (`code/ecpa/flash.py`) — extension to the full ternary
   CO₂ + H₂O + NaCl system via warm-started K-value SSI with optional Newton polish.

4. **Precomputed solution table** (`code/results/CPA_ELV_all.parquet`) — 31T × 30P × 18*z* × 14*m*ₛ
   grid covering *T* = 283–728 K, *P* = 1–1500 bar, *z* = 0.05–0.90, *m*ₛ = 0–6 mol/kg.

5. **Prototype reservoir simulator** (`code/co2brine_simulator.py`) — IMPEC scheme with
   MFE pressure solver demonstrating production-level flash throughput.

See [REPRODUCING_FIGURES.md](REPRODUCING_FIGURES.md) for the complete script-by-script
guide to regenerating every figure in the paper.

---

## Installation

Requires Python ≥ 3.11.

```bash
git clone https://github.com/jmoortgat/CPA_and_eCPA_flash.git
cd CPA_and_eCPA_flash
pip install -e ".[test]"     # editable install + pytest
pytest                       # run the test suite
```

Optional extras: `pip install -e ".[nn]"` adds PyTorch for the
neural-network warm-start experiments; a conda environment is provided in
`code/environment.yml`.

---

## Quick start

**eCPA ternary flash** (CO₂ + H₂O + NaCl) — after `pip install -e .`,
from anywhere:

```python
from ecpa.flash import flash_co2_h2o_salt_kv

# T [K], P [bar], overall CO2 mole fraction, NaCl molality [mol/kg]
r = flash_co2_h2o_salt_kv(323.15, 100.0, 0.3, 1.0)

print(r["beta"])            # vapor fraction              -> 0.2814
print(r["x_aq"]["x4w"])     # CO2 solubility in brine     -> 1.670e-02
print(r["x_c"]["x1c"])      # H2O content of CO2 phase    -> 2.727e-03
print(r["ms_aq"])           # equilibrium aqueous molality
```

**Salt-free CPA binary** (CO₂ + H₂O) — run from the `code/` directory:

```python
import CPA
r = CPA.flash_co2_h2o_tpz(T=323.15, P_bar=100.0, z_co2=0.3)
print(r['phase'], r['beta'], r['x'])
```

---

## Notebooks

**Interactive tutorial** — a hands-on user manual covering the basic
workflows: single CPA and eCPA flash calculations, pressure sweeps compared
against experimental data, CO₂ solubility in NaCl brines of different
salinity, phase densities, and warm-started high-throughput calls:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jmoortgat/CPA_and_eCPA_flash/blob/main/notebooks/ecpa_flash_tutorial.ipynb)
&nbsp;[`notebooks/ecpa_flash_tutorial.ipynb`](notebooks/ecpa_flash_tutorial.ipynb)

**The executable paper** — the complete journal article and its Supporting
Information as a single executable notebook — full text, table of contents,
and collapsed code cells that regenerate every figure from the code and data
in this repository:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jmoortgat/CPA_and_eCPA_flash/blob/main/notebooks/ecpa_flash_paper.ipynb)
&nbsp;[`notebooks/ecpa_flash_paper.ipynb`](notebooks/ecpa_flash_paper.ipynb)

Fast figures render from result caches shipped with the repository; the
long parameter-space scans and the reservoir-simulation demo can be
recomputed from scratch by setting `RUN_LONG = True` in the setup cell
(30–60 min total). See [REPRODUCING_FIGURES.md](REPRODUCING_FIGURES.md)
for the script-by-script equivalent.

---

## Repository structure

```
CPA_and_eCPA_flash/
├── README.md
├── REPRODUCING_FIGURES.md
├── LICENSE, CITATION.cff, CHANGELOG.md
├── CONTRIBUTING.md, CODE_OF_CONDUCT.md
├── pyproject.toml                 # pip install -e .
├── paper.md, paper.bib            # JOSS paper
├── tests/                         # pytest regression suite
├── notebooks/
│   ├── ecpa_flash_tutorial.ipynb  # interactive user manual
│   └── ecpa_flash_paper.ipynb     # executable paper
├── EXP/                           # experimental VLE database (raw)
└── code/
    ├── ecpa/                      # eCPA package
    │   ├── flash.py               #   flash_co2_h2o_salt_kv  ← production flash
    │   ├── stability.py           #   ecpa_stability, ecpa_stability_flash
    │   ├── guess_table.py         #   parquet warm-start lookup
    │   ├── solution_table.py      #   build_solution_table
    │   ├── warmstart.py           #   ScanTableWarmStart (4D npz lookup)
    │   ├── constants.py           #   EoS parameters & Péneloux shifts
    │   ├── parameters.py          #   make_params()
    │   ├── elv.py                 #   ELV residual + analytical Jacobian
    │   ├── flash_simplified.py    #   low-T simplified flash (T < 80 °C)
    │   ├── envelope.py            #   phase-boundary tracing
    │   ├── exp_data.py            #   experimental data loader
    │   └── ...
    ├── CPA.py                     # Salt-free CPA binary flash
    ├── co2brine_simulator.py      # IMPEC reservoir simulator
    ├── ...
    ├── scripts/                   # Validation, benchmark & figure scripts
    │   ├── validate_co2h2o.py
    │   ├── validate_co2nacl_full.py
    │   ├── benchmark_pure_water_density.py
    │   ├── run_benchmark.py
    │   ├── plot_co2h2o_figures.py
    │   ├── plot_co2nacl_figures.py
    │   └── ...
    └── results/
        ├── CPA_ELV_all.parquet    # Precomputed solution table (25 MB)
        └── scan_v4_table.npz      # 4D warm-start table (23 MB)
```

---

## API reference

### `ecpa/flash.py`

| Function | Description |
|:---|:---|
| `flash_co2_h2o_salt_kv(T, P, z, ms, params, guess_fn)` | **Production flash.** Hierarchical stability+flash with K-value SSI and optional Newton polish. |
| `flash_co2_h2o_salt_ssi(T, P, z, ms, params)` | Legacy cold-start SSI flash (ω = 0.7). Kept for reference — prefer `flash_co2_h2o_salt_kv`. |

### `ecpa/stability.py`

| Function | Description |
|:---|:---|
| `ecpa_stability(T, P, z, ms, params)` | Michelsen TPD test → `(tpd_min, trial_comp, converged)`. |
| `stability_map(T_range, P_range, z, ms, params, n_workers)` | Parallel 2-D stability scan. |

### `CPA.py`

| Function | Description |
|:---|:---|
| `flash_co2_h2o_tpz_robust(T, P_bar, z_co2)` | **100% convergence.** Stability → best-K flash → Wilson fallback. |
| `flash_co2_h2o_tpz(T, P_bar, z_co2, vshift_h2o, vshift_co2)` | Single-call binary flash. Returns compositions, Z-factors, `rho_mass` [kg/L]. |
| `stability_test(T, P_bar, z, accelerated=True)` | TPD with 6 initial guesses ([Jex et al., 2024](https://doi.org/10.2118/219490-PA)). |

---

## Performance

### eCPA ternary flash

| Method | SSI iters | Time / call | Speedup |
|:---|:---:|:---:|:---:|
| Cold-start SSI (Wilson init) | 11.7 | ~10 ms | 1× |
| **Warm-start (solution table)** | **3.3** | **~3 ms** | **3.2×** |

*Benchmark: T = 398 K, z = 0.5, m*ₛ *=  1.0 mol/kg, 30 pressure points.*

### Salt-free CPA binary flash

| Strategy | Convergence | Mean iters |
|:---|:---:|:---:|
| Standard SSI + Wilson K | 96.0% | 46.6 |
| Accelerated SSI + Wilson K | 97.1% | 17.0 |
| **Hierarchical (robust)** | **100%** | **12.0** |

*Tested at 631 experimental CO₂+H₂O points (T = 273–623 K, P = 5–3500 bar).*

---

## Validation

| System | Quantity | *N* | AARE |
|:---|:---|:---:|:---:|
| CO₂ + H₂O (CPA) | *x*_CO₂ in water | 460 | 8.9% |
| CO₂ + H₂O (eCPA) | *x*_CO₂ in water | 451 | 8.2% |
| CO₂ + NaCl (eCPA) | *m*_CO₂ [mol/kg] | 440 | 6.9% |
| CO₂ + NaCl (eCPA) | *x*_CO₂ [mole fraction] | 99 | 7.0% |
| Pure-water density vs. IAPWS-95 | ρ_W [kg/m³] | 475 | **0.30%** |
| CO₂-saturated aqueous density (exp.) | ρ_W [kg/m³] | 37 | 0.76% |

---

## Péneloux volume shifts

All shifts live in `code/ecpa/constants.py`.

| Parameter | Value | Source |
|:---|:---:|:---|
| `Penelouxs` (NaCl) | −53.5 cm³/mol | Coelho et al. (2025); applied via `make_params()` |
| `peneloux_h2o(T)` | degree-4 polynomial in *T*/*T*c | Fitted in this work to 475 IAPWS-95 liquid-water conditions |
| `Peneloux_CO2` | 0 | Evaluated against Span–Wagner; unshifted performs best |

The temperature-dependent H₂O shift is isofugacity-preserving — it improves
density predictions without affecting phase compositions or VLE results. It is
applied in the density validation (`scripts/benchmark_pure_water_density.py`,
which produces the paper's density figures); flash routines report unshifted
EoS densities unless the shift is passed explicitly (`vshift_h2o`/`vshift_co2`
in the CPA flash functions).

---

## Reproducing the figures

See **[REPRODUCING_FIGURES.md](REPRODUCING_FIGURES.md)** for the complete script-by-script
commands to regenerate all figures in the paper and supplemental information.

Quick summary:

```bash
cd code
python scripts/validate_co2h2o.py        # ~5 min
python scripts/validate_co2nacl_full.py  # ~30 min
python scripts/run_parameter_scan.py     # ~20 min
python scripts/run_warmstart_scan.py     # ~27 min
python scripts/plot_co2h2o_figures.py    # Figs. 1, S1, S6
python scripts/plot_co2nacl_figures.py   # Figs. 2, 3, S2, S4
python scripts/plot_scan_figures.py      # Figs. 7, 8, 9
# ... see REPRODUCING_FIGURES.md for the full list
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{moortgat2026ecpa,
  author  = {Moortgat, Joachim and Coelho, Felipe Mour{\~a}o and Firoozabadi, Abbas},
  title   = {Fast and Robust Phase Equilibrium Computations for {CO}$_2$ + {H}$_2${O} + {NaCl}
             Mixtures Using the Electrolyte Cubic-Plus-Association Equation of State},
  journal = {Industrial \& Engineering Chemistry Research},
  year    = {2026},
  note    = {in press}
}
```

And the underlying eCPA parametrisation:

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

## Contributing

Bug reports, questions, and pull requests are welcome — see
[CONTRIBUTING.md](CONTRIBUTING.md). Participation is governed by the
[Code of Conduct](CODE_OF_CONDUCT.md).

## License

[MIT](LICENSE). If you use this software, please cite it (see
[CITATION.cff](CITATION.cff)) together with the companion paper above.

[coelho2025]: https://doi.org/10.1021/acs.iecr.5c00134
