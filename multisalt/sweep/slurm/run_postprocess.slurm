#!/bin/bash
# ----------------------------------------------------------------------
# Multi-salt eCPA sweep — post-processing (single-task; runs after the
# array job to merge per-chunk parquets, build a failure inventory, and
# write a top-level summary.json).
#
# Usage:
#   sbatch --dependency=afterany:<sweep_jobid> \
#          --export=INDIR=$PWD/results_cache/sweep_standard,OUTDIR=$PWD/results_cache/sweep_standard_summary \
#          slurm/run_postprocess.slurm
# ----------------------------------------------------------------------
#SBATCH --job-name=ms-sweep-post
# Set the partition (and account, if your cluster needs one) for your site:
##SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/sweep_post_%j.out
#SBATCH --error=logs/sweep_post_%j.err

set -euo pipefail
: "${INDIR:?must export INDIR=<absolute path>}"
: "${OUTDIR:?must export OUTDIR=<absolute path>}"
PYBIN="${PYBIN:-python}"

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "$SLURM_SUBMIT_DIR/sweep" ] && [ -d "$SLURM_SUBMIT_DIR/ms_ecpa" ]; then
    CODE_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    CODE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "[ms-sweep-post] SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}"
echo "[ms-sweep-post] CODE_DIR=$CODE_DIR"
echo "[ms-sweep-post] PYTHONPATH=$PYTHONPATH"
ls -d "$CODE_DIR/sweep" "$CODE_DIR/ms_ecpa" 2>&1 | sed 's/^/[ms-sweep-post] /'

mkdir -p logs "$OUTDIR"

cd "$CODE_DIR"
$PYBIN -m sweep.sweep_postprocess --in-dir "$INDIR" --out-dir "$OUTDIR"
