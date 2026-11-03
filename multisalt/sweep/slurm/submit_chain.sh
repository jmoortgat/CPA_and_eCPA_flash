#!/bin/bash
# ---------------------------------------------------------------------
# Submit the five stages as one dependent chain.
#
#   1 verify   build, parameter files, acceptance gate, baseline numbers
#   2 sweep    151,632 states, 8 array tasks x 48 workers
#   3 census   merge + honest route breakdown
#   4 figures  every figure and table, cold, + paper_stats.tex
#   5 sweep figures  merge the sweep; Paper III grid figures and tables
#
# Each stage runs afterok of the previous one, so a failure anywhere stops
# everything downstream rather than producing numbers from a tree that did
# not pass its own checks.
#
# Usage (from the multisalt/ directory):  bash sweep/slurm/submit_chain.sh
# ---------------------------------------------------------------------
set -euo pipefail
CODE="${CODE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYBIN="${PYBIN:-python3}"
NCHUNKS="${NCHUNKS:-8}"
OUTDIR="${OUTDIR:-$CODE/results/sweep_valid_dense}"
EXPORTS="CODE=$CODE,PYBIN=$PYBIN,NCHUNKS=$NCHUNKS,OUTDIR=$OUTDIR"

cd "$CODE"
mkdir -p logs "$OUTDIR"

J1=$(sbatch --parsable --export="$EXPORTS" sweep/slurm/chain_01_verify.slurm)
echo "stage 1 verify   : $J1"

J2=$(sbatch --parsable --dependency=afterok:$J1 --array=0-$((NCHUNKS-1)) \
        --export="$EXPORTS" sweep/slurm/chain_02_sweep.slurm)
echo "stage 2 sweep    : $J2  (array 0-$((NCHUNKS-1)), afterok:$J1)"

J3=$(sbatch --parsable --dependency=afterok:$J2 --export="$EXPORTS" \
        sweep/slurm/chain_03_census.slurm)
echo "stage 3 census   : $J3  (afterok:$J2)"

J4=$(sbatch --parsable --dependency=afterok:$J3 --export="$EXPORTS" \
        sweep/slurm/chain_04_figures.slurm)
echo "stage 4 figures  : $J4  (afterok:$J3)"

J5=$(sbatch --parsable --dependency=afterok:$J4 --export="$EXPORTS" \
        sweep/slurm/chain_05_sweepfigs.slurm)
echo "stage 5 sweep figures : $J5  (afterok:$J4)"

echo
echo "watch:   squeue -u \$USER"
echo "logs:    $CODE/logs/"
echo "results: $CODE/results/"
