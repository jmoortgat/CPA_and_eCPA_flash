#!/bin/bash
#SBATCH --job-name=chain3-census
# Set the partition (and account, if your cluster needs one) for your site:
##SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/chain3_census_%j.out
#SBATCH --error=logs/chain3_census_%j.err
# ---------------------------------------------------------------------
# Stage 3: merge the chunks and report the convergence counts.
#
# sweep_core records converged = (phases == 1), so a state where every
# flash attempt failed and the soft fallback declared it single-phase is
# indistinguishable from a genuine single-phase verdict in that column.
# This stage reports the route breakdown, counts the failure routes
# separately, and fails the chain if any appear, so the convergence rate
# reported in Paper III counts only flashes that converged.
# ---------------------------------------------------------------------
set -uo pipefail
CODE="${CODE:?export CODE=<repo root>}"
PYBIN="${PYBIN:-python3}"
OUTDIR="${OUTDIR:?export OUTDIR=<absolute path>}"
NCHUNKS="${NCHUNKS:-8}"
cd "$CODE"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"

NFOUND=$(ls "$OUTDIR"/*.parquet 2>/dev/null | wc -l)
echo "=== chunk files: $NFOUND of $NCHUNKS expected"
if [ "$NFOUND" -ne "$NCHUNKS" ]; then
    echo "FATAL: $((NCHUNKS - NFOUND)) chunk(s) missing. The denominator would"
    echo "be wrong and the convergence rate would look better than it is."
    exit 1
fi

$PYBIN reproduce/sweep_census.py --in-dir "$OUTDIR" \
       --out "$CODE/results/sweep_census.json" || exit 1
echo "=== STAGE 3 PASSED"
