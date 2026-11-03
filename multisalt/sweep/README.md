# Multi-salt eCPA stability+flash sweep harness

A self-contained driver for a wide-area parameter scan of
`stability_flash` across the multi-salt eCPA window. Every input is
recorded; every physical output is preserved; every internal warm-start
state (compressibility factors, association fractions, permittivity)
is saved; every iteration count and wall-clock measurement is
captured. **Failures are never silenced** — they become rows with
`converged=False` and a populated `error_type` / `error_msg` so the
failure inventory directly drives the next round of robustness work.

## Quick start

### Local smoke test

```bash
cd code/
PYTHONPATH=. python -m sweep.sweep_runner \
    --config smoke --chunk-id 0 --n-chunks 1 \
    --workers 4 \
    --out-dir /tmp/sweep_smoke

PYTHONPATH=. python -m sweep.sweep_postprocess \
    --in-dir /tmp/sweep_smoke \
    --out-dir /tmp/sweep_smoke_summary
```

Runs ~270 points in roughly a minute on a laptop, writes one parquet
and prints the failure mode table.

### HPC standard run (single node, 48 cores, ~50 k points)

```bash
sbatch --array=0-7 \
    --export=CONFIG=standard,NCHUNKS=8,CONVENTION=as-aqueous,OUTDIR=$PWD/results_cache/sweep_standard \
    sweep/slurm/run_sweep.slurm

# After it finishes, merge + summarise:
sbatch --dependency=afterany:<sweep_jobid> \
    --export=INDIR=$PWD/results_cache/sweep_standard,OUTDIR=$PWD/results_cache/sweep_standard_summary \
    sweep/slurm/run_postprocess.slurm
```

### HPC full run (multiple nodes, ~530 k points)

```bash
sbatch --array=0-31 \
    --export=CONFIG=full,NCHUNKS=32,CONVENTION=as-aqueous,OUTDIR=$PWD/results_cache/sweep_full \
    sweep/slurm/run_sweep.slurm
```

The SLURM array distributes 32 chunks across whatever nodes the
partition has available. With 4 nodes × 48 cores active the
whole sweep finishes in ~30 min; with one node it takes ~3 h.

## Cluster context

* **Partition:** set `--partition` in `slurm/run_sweep.slurm` for your cluster.

## Grid design (in `sweep_config.py`)

Three named grids; see `sweep_config.estimate_size()` for live sizes.

| Config | Salt configs | T pts | P pts | Binary z | Ternary z | Total |
|---|---:|---:|---:|---:|---:|---:|
| smoke    | 2  | 3  | 5  | 3  | 3  | ~270 |
| standard | 8  | 13 | 20 | 7  | 10 | ~50 k |
| full     | 14 | 19 | 30 | 19 | 28 | ~530 k |

Every `(salt, T, P)` triple is hit with three feed families:
* **`binary_CO2`** — CO2 + H2O, `z_CO2 ∈ ZBIN`
* **`binary_CH4`** — CH4 + H2O, `z_CH4 ∈ ZBIN`
* **`ternary`**    — CH4 + CO2 + H2O, `(z_CH4, z_CO2)` triangle

Salt set covers no-salt, five single salts at one-to-three molalities
each, and four mixed-brine recipes from the paper validation set.

To shrink/grow without rewriting the structure, edit the resolution
arrays (`T_FULL`, `P_FULL`, `ZBIN_FULL`, `ZTRI_FULL`,
`SALT_CONFIGS_FULL`) at the top of `sweep_config.py`.

## Output schema

Each chunk writes `chunk_NNNNN_of_NNNNN.parquet`. Columns (≈ 50 in
total; full list in `sweep_core.output_schema()`):

| Group | Columns |
|---|---|
| Inputs (echoed) | `point_id, config_name, feed_family, salt_name, T_K, P_bar, z_H2O, z_CH4, z_CO2, m_Na, m_Cl, m_K, m_Ca, m_Mg, m_SO4, salinity_convention` |
| Outcome | `converged, phases, tpd_min, error_type, error_msg` |
| Phase equilibrium | `beta, x_H2O_aq, x_CH4_aq, x_CO2_aq, y_H2O_gas, y_CH4_gas, y_CO2_gas, K_H2O, K_CH4, K_CO2` |
| Equilibrium aq molality | `m_Na_aq, …, m_SO4_aq` (`= m_prep` under `as-aqueous`; amplified under `chain`) |
| Fugacities (bar) | `f_*_aq, f_*_gas` |
| Warm-start state | `Z_aq, Z_gas, eps_r, chi_H2O_aq, chi_H2O_gas` |
| Densities (kg/m³) | `rho_aq, rho_gas` |
| Iteration counts | `n_iter_ssi, n_iter_newton_polish, n_iter_aq_inner_diag, n_iter_gas_inner_diag, max_residual` |
| Timing | `t_wall_ms` |

Iteration counts and timings are the dataset to mine when we want to
hunt down slow spots and pull more state into warm-start lookups.

## Post-processing artefacts

`sweep_postprocess.py` writes into the output directory:

* `merged.parquet`     — all chunks concatenated
* `summary.json`       — overall convergence rate, wall-time
  percentiles, iteration-count percentiles
* `failures.parquet`   — every non-converged row, ordered by
  `(feed_family, salt_name, T_K, P_bar)`
* `failure_modes.csv`  — counts grouped by
  `(error_type, feed_family, salt_name)`; the natural starting point
  for the next iteration on robustness

## Iterating to 100 % convergence

The expected workflow is:

1. Run `--config smoke` on the cluster once to confirm the harness
   itself is healthy (we expect ~85 % convergence in the current
   baseline).
2. Run `--config standard` and inspect `failure_modes.csv`.
3. For the top few rows of `failure_modes.csv`, pull the failing
   `(T_K, P_bar, z, mi)` tuples from `failures.parquet`, reproduce
   locally with `stability_flash(...)`, fix the underlying issue
   (better trial seeds, looser inner-Newton initial guesses, a Brent
   fallback, etc.).
4. Re-run `--config standard` and confirm the failure inventory
   shrinks.
5. When the standard sweep reaches 100 %, launch `--config full` as
   the final benchmark + warm-start corpus.

## Reusing the sweep output as warm-start

Each converged row carries the full inner-Newton state at convergence:
`Z_aq, Z_gas, eps_r, chi_H2O_aq, chi_H2O_gas`. To seed a new flash
call from a near-by sweep row:

```python
import pandas as pd
sweep = pd.read_parquet("results_cache/sweep_full_summary/merged.parquet")
# Pick the nearest (T, P, m_NaCl, z_CO2) row in the sweep and pass its
# (Z_aq, eps_r, chi_H2O_aq) as the initial guess to the aqueous inner
# Newton; ditto (Z_gas, chi_H2O_gas) for the gas side. See
# ms_ecpa.lnphi._aq_residual_3var / _gas_residual for the signatures.
```

This is the same warm-start scheme we use inside the SSI loop today;
the sweep just provides a dense pre-computed lookup table that any
downstream call (reservoir simulation, batch experiment fit, sweep
re-run) can interpolate from.

## Files

```
code/sweep/
├── __init__.py
├── sweep_config.py         # named grids, salt recipes
├── sweep_core.py           # single-point worker (run_one), output schema
├── sweep_runner.py         # chunk runner with multiprocessing.Pool
├── sweep_postprocess.py    # merge chunks, summary.json, failures.csv
├── README.md               # (this file)
└── slurm/
    ├── run_sweep.slurm     # SLURM array job
    └── run_postprocess.slurm
```
