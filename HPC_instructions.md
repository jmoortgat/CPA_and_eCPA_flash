# Running the CO₂-brine demo simulations on an HPC cluster

## 1 · Transfer the code

From your local machine (run once):

```bash
# Set your cluster login — adjust to your actual hostname/username
CLUSTER=unity

# Sync the entire project (excluding large generated outputs)
rsync -avz --exclude='__pycache__' \
           --exclude='*.pyc' \
           --exclude='figures/' \
           --exclude='results/' \
           --exclude='.git/' \
  ~/Software/2026/eCPA_SALTbasis/Claude_code/ \
  ${CLUSTER}:/home/moortgat.1/Claude_code/

# Also sync the parquet data files (needed by flash)
rsync -avz ~/Software/2026/eCPA_SALTbasis/Claude_code/*.parquet \
           ~/Software/2026/eCPA_SALTbasis/Claude_code/*.npz  \
  ${CLUSTER}:/home/moortgat.1/Claude_code/
```

## 2 · Set up the Python environment on the cluster

The `CPA` and `ecpa` packages are not formally installed — they are plain
directories at the project root. The SLURM scripts set `PYTHONPATH` so Python
can find them. If your `h2oval` mamba environment already has numpy, scipy,
pandas, matplotlib, pyarrow, etc., no further setup is needed.

If `h2oval` is missing any dependency, install it once:

```bash
mamba activate h2oval
mamba install -n h2oval numpy scipy pandas matplotlib pyarrow -y
```

## 3 · SLURM job scripts

`run_cpa.slurm` and `run_ecpa.slurm` are in the project root and will be
copied by rsync automatically.

## 4 · Submit both jobs simultaneously

```bash
cd /home/moortgat.1/Claude_code
mkdir -p logs
sbatch run_cpa.slurm    # → JOB_ID_A
sbatch run_ecpa.slurm   # → JOB_ID_B

# Monitor
squeue -u $USER
```

Each job runs on its own node (1 node × 48 cores). The flash pool inside the
simulator picks up `os.cpu_count()` automatically, so no extra flags are needed.

## 5 · Generate the comparison figure after both jobs finish

Once both `final_state.npz` files exist on the cluster:

```bash
python _run_demo_simulations.py figures
```

Or retrieve the results first and generate the figure locally:

```bash
# On your local machine:
rsync -avz ${CLUSTER}:/home/moortgat.1/Claude_code/figures/ \
           /Users/moortgat/Software/2026/eCPA_SALTbasis/Claude_code/figures/

cd /Users/moortgat/Software/2026/eCPA_SALTbasis/Claude_code
python _run_demo_simulations.py figures
```

## 6 · Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: CPA` | `pip install -e .` not run, or wrong env active |
| `FileNotFoundError: *.parquet` | Re-run the rsync step to copy data files |
| Flash pool hangs at startup | Add `--cpus-per-task=1` and set `n_workers=1` in `main()` calls inside `_run_demo_simulations.py` |
| Job killed (OOM) | Increase `--mem` to 64G or reduce `Nx_`/`Ny_` |
| Very slow (> 8 h) | Reduce `n_steps_` from 300 to 150, or reduce grid to 40×40 |
