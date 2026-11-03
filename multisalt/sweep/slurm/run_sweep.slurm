#!/bin/bash
# ----------------------------------------------------------------------
# Multi-salt eCPA stability+flash sweep — SLURM array job
#
# Launches one task per chunk; each task runs N workers in parallel on
# one node and writes its own parquet file. Failed tasks affect only
# their own chunk; the rest of the sweep keeps making progress.
#
# Usage
# -----
#   sbatch --array=0-$((NCHUNKS-1)) \
#          --export=CONFIG=standard,NCHUNKS=$NCHUNKS,CONVENTION=as-aqueous,OUTDIR=$PWD/results_cache/sweep_standard \
#          slurm/run_sweep.slurm
#
# Recommended chunk counts:
#   smoke      NCHUNKS=1     (single laptop task is enough)
#   standard   NCHUNKS=8     ( ~6k pts/chunk; finishes well under an hour)
#   full       NCHUNKS=32    (~16k pts/chunk; one node per task at 48 cores)
#
# Environment overrides via --export:
#   CONFIG       : smoke | standard | full         (REQUIRED)
#   NCHUNKS      : integer — total number of chunks (REQUIRED)
#   CONVENTION   : as-aqueous | chain   (default: as-aqueous)
#   OUTDIR       : output directory     (REQUIRED, absolute)
#   PYBIN        : python executable    (default: python)
# ----------------------------------------------------------------------
#SBATCH --job-name=ms-sweep
# Set the partition (and account, if your cluster needs one) for your site:
##SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0                        # reserve all memory on the node
#SBATCH --time=08:00:00
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
# Add this manually when you want a specific subset (otherwise let the
# scheduler pick from the partition):
##SBATCH --nodelist=<nodes>

set -euo pipefail

# ----------------------------------------------------------------------
# Required inputs
# ----------------------------------------------------------------------
: "${CONFIG:?must export CONFIG=smoke|standard|full}"
: "${NCHUNKS:?must export NCHUNKS=<integer>}"
: "${OUTDIR:?must export OUTDIR=<absolute path>}"
CONVENTION="${CONVENTION:-as-aqueous}"
PYBIN="${PYBIN:-python}"

CHUNK_ID="${SLURM_ARRAY_TASK_ID:-0}"
WORKERS="${SLURM_CPUS_PER_TASK:-48}"

mkdir -p "$OUTDIR" logs

echo "============================================================"
echo " ms-sweep  chunk=$CHUNK_ID/$NCHUNKS  workers=$WORKERS"
echo " config=$CONFIG  convention=$CONVENTION"
echo " host=$(hostname)  job=$SLURM_JOB_ID  array=$SLURM_ARRAY_JOB_ID"
echo " outdir=$OUTDIR"
echo "============================================================"

# Locate the repo root: prefer SLURM_SUBMIT_DIR (cwd at sbatch time, which
# should be the code/ root); fall back to BASH_SOURCE if not set or if it
# does not look right. SLURM may stage the script outside the repo tree,
# which breaks the BASH_SOURCE-relative resolution we used originally.
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "$SLURM_SUBMIT_DIR/sweep" ] && [ -d "$SLURM_SUBMIT_DIR/ms_ecpa" ]; then
    CODE_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    CODE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Diagnostic: confirm Python will find sweep/ + ms_ecpa/.
echo "[ms-sweep] SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}"
echo "[ms-sweep] CODE_DIR=$CODE_DIR"
echo "[ms-sweep] PYTHONPATH=$PYTHONPATH"
ls -d "$CODE_DIR/sweep" "$CODE_DIR/ms_ecpa" 2>&1 | sed 's/^/[ms-sweep] /'

# Cap BLAS/MKL/OMP thread counts so they don't fight the worker pool. The
# inner Newton/SSI loops are scalar and benefit from 1 thread per worker.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$CODE_DIR"
$PYBIN -m sweep.sweep_runner \
    --config "$CONFIG" \
    --chunk-id "$CHUNK_ID" \
    --n-chunks "$NCHUNKS" \
    --workers "$WORKERS" \
    --convention "$CONVENTION" \
    --out-dir "$OUTDIR"

echo "============================================================"
echo " ms-sweep  chunk=$CHUNK_ID  DONE"
echo "============================================================"
