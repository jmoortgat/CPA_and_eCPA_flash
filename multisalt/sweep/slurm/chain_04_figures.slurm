#!/bin/bash
#SBATCH --job-name=chain4-figs
# Set the partition (and account, if your cluster needs one) for your site:
##SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/chain4_figs_%j.out
#SBATCH --error=logs/chain4_figs_%j.err
# ---------------------------------------------------------------------
# Stage 4: every paper figure and table, cold, plus paper_stats.tex.
# --force recomputes every cached model result, so no figure is drawn
# from a cache built with other code or data.
# ---------------------------------------------------------------------
set -uo pipefail
CODE="${CODE:?export CODE=<repo root>}"
PYBIN="${PYBIN:-python3}"
cd "$CODE"
mkdir -p logs
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "=== experimental source in use:"
$PYBIN -c "import sys; sys.path.insert(0,'scripts'); import gasbrinebench_source as g; print('   '+g.describe_source())"

$PYBIN scripts/make_all_paper_figures.py --force
RC=$?
echo "=== figures exit code: $RC"
[ $RC -ne 0 ] && { echo "FATAL: figure generation failed"; exit 1; }
echo "=== STAGE 4 PASSED"
