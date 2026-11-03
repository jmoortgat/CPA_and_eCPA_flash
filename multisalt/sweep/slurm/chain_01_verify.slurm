#!/bin/bash
#SBATCH --job-name=chain1-verify
# Set the partition (and account, if your cluster needs one) for your site:
##SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/chain1_verify_%j.out
#SBATCH --error=logs/chain1_verify_%j.err
# ---------------------------------------------------------------------
# Stage 1 of the chain: build and check the code.
#
# Everything after this stage runs under afterok, so if anything here is
# wrong the sweep and the figures do not start.
#
# It fails deliberately on:
#   * a compile error
#   * any acceptance-test failure
#   * a family AAD that has moved from the verified values
# ---------------------------------------------------------------------
set -uo pipefail
CODE="${CODE:?export CODE=<repo root>}"
PYBIN="${PYBIN:-python3}"
cd "$CODE"
mkdir -p logs results
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "=== host $(hostname)  commit $(git rev-parse --short HEAD)"

echo "=== 1/4 build"
make -C fortran clean >/dev/null 2>&1
if ! make -C fortran 2>&1 | tail -20; then
    echo "FATAL: Fortran build failed"; exit 1
fi
test -f fortran/build/libecpa.so || { echo "FATAL: libecpa.so missing"; exit 1; }

echo "=== 2/4 parameter files"
$PYBIN fortran/tools/gen_params.py || { echo "FATAL: gen_params failed"; exit 1; }

echo "=== 3/4 acceptance gate"
$PYBIN -m pytest fortran/tools/test_fortran_port.py \
                 fortran/tools/test_fortran_colligative.py \
                 fortran/tools/test_fortran_chain.py -q > logs/gate_$SLURM_JOB_ID.txt 2>&1
grep -E "passed|failed" logs/gate_$SLURM_JOB_ID.txt | tail -2
BAD=$(grep -c "^FAILED" logs/gate_$SLURM_JOB_ID.txt || true)
if [ "$BAD" -ne 0 ]; then
    echo "FATAL: $BAD acceptance failure(s):"
    grep "^FAILED" logs/gate_$SLURM_JOB_ID.txt
    exit 1
fi
echo "acceptance tests OK"

echo "=== 4/4 family AADs + density, against the verified values"
$PYBIN reproduce/verify_baseline.py || { echo "FATAL: baseline regression failed"; exit 1; }

echo "=== STAGE 1 PASSED"
