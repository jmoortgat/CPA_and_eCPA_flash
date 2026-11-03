#!/bin/bash
#SBATCH --job-name=chain2-sweep
# Set the partition (and account, if your cluster needs one) for your site:
##SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=logs/chain2_sweep_%A_%a.out
#SBATCH --error=logs/chain2_sweep_%A_%a.err
# ---------------------------------------------------------------------
# Stage 2: the 151,632-state convergence sweep (config valid_dense).
# One array task per chunk, 48 workers each. A task that dies takes only
# its own chunk down; stage 3 checks the chunk count before trusting the
# total, because a silently missing chunk would shrink the denominator
# and make the convergence rate look better than it is.
# ---------------------------------------------------------------------
set -euo pipefail
CODE="${CODE:?export CODE=<repo root>}"
PYBIN="${PYBIN:-python3}"
NCHUNKS="${NCHUNKS:-8}"
OUTDIR="${OUTDIR:?export OUTDIR=<absolute path>}"
cd "$CODE"
mkdir -p "$OUTDIR" logs
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "=== chunk ${SLURM_ARRAY_TASK_ID}/${NCHUNKS} on $(hostname), 48 workers"
$PYBIN -m sweep.sweep_runner \
    --config valid_dense \
    --chunk-id "${SLURM_ARRAY_TASK_ID}" \
    --n-chunks "${NCHUNKS}" \
    --workers "${SLURM_CPUS_PER_TASK:-48}" \
    --convention as-aqueous \
    --out-dir "$OUTDIR"
echo "=== chunk ${SLURM_ARRAY_TASK_ID} DONE"
