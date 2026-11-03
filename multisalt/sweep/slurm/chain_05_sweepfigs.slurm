#!/bin/bash
#SBATCH --job-name=chain5-sweepfigs
# Set the partition (and account, if your cluster needs one) for your site:
##SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/chain5_sweepfigs_%j.out
#SBATCH --error=logs/chain5_sweepfigs_%j.err
# ---------------------------------------------------------------------
# Stage 5: merge the sweep and build the Paper III surface figures and
# performance tables.
#
# This stage exists because stage 4 does NOT cover them.
# make_all_paper_figures.py drives thirteen fig_*.py scripts over the
# experimental data; make_sweep_figures.py is a separate entry point that
# takes the merged sweep and produces the heatmaps and the performance
# tables -- including the convergence cell and the SSI statistics. Without
# this stage the chain would regenerate every figure built from experiment
# but not the grid figures.
#
# sweep_postprocess is run with --config so its own row-count check fires:
# it compares the merged total against what valid_dense should yield, which
# catches a chunk that produced a short file rather than no file.
# ---------------------------------------------------------------------
set -uo pipefail
CODE="${CODE:?export CODE=<repo root>}"
PYBIN="${PYBIN:-python3}"
OUTDIR="${OUTDIR:?export OUTDIR=<chunk dir>}"
MERGED="${MERGED:-$CODE/results/sweep_merged}"
FIGS="${FIGS:-$CODE/results/sweep_figures}"
cd "$CODE"
mkdir -p logs "$MERGED" "$FIGS"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "=== 1/2 merge + row-count check against valid_dense"
$PYBIN -m sweep.sweep_postprocess --in-dir "$OUTDIR" --out-dir "$MERGED" \
       --config valid_dense || { echo "FATAL: postprocess/row-count failed"; exit 1; }
test -f "$MERGED/merged.parquet" || { echo "FATAL: merged.parquet missing"; exit 1; }

echo "=== 2/2 sweep figures and performance tables"
$PYBIN sweep/figures/make_sweep_figures.py --merged "$MERGED/merged.parquet" \
       --out-dir "$FIGS" || { echo "FATAL: sweep figures failed"; exit 1; }

echo "=== convergence cell as it will appear in the paper:"
grep -h "Converged\|Grid points" "$FIGS"/tables/*.tex 2>/dev/null | head -4
echo "=== STAGE 5 PASSED"
