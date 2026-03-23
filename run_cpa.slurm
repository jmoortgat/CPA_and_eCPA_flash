#!/bin/bash
#SBATCH --job-name=co2brine_cpa
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/cpa_%j.out
#SBATCH --error=logs/cpa_%j.err

source $(conda info --base)/etc/profile.d/conda.sh
mamba activate h2oval

cd /home/moortgat.1/Claude_code
mkdir -p logs figures/demo/cpa_ms0

# Make CPA and ecpa importable from this directory
export PYTHONPATH=/home/moortgat.1/Claude_code:$PYTHONPATH

# Let numpy/scipy use only 1 thread per worker so the Python process pool
# gets all 48 cores without competition from BLAS threading
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python _run_demo_simulations.py cpa
