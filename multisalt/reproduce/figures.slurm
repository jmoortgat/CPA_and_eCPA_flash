#!/bin/bash
#SBATCH --job-name=ms-figs
# Set the partition (and account, if your cluster needs one) for your site:
##SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
# Generous by design: a job killed at the time limit leaves a half-written
# cache. The regeneration on the Python kernel takes over 2.5 h on a
# workstation, and a cold run can be far slower than a warm one.
#SBATCH --time=24:00:00
#SBATCH --output=logs/figs_%j.out
#SBATCH --error=logs/figs_%j.err
set -euo pipefail
source /etc/profile.d/lmod.sh 2>/dev/null || true
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD:$PWD/scripts"
export MS_ECPA_BACKEND="${MS_ECPA_BACKEND:-fortran}"
PY="${PYBIN:-python3}"
echo "host=$(hostname) backend=$MS_ECPA_BACKEND"
echo "gbb=${ECPA_GBB:-<pinned version, fetched on first use>}"
"$PY" -c "import ms_ecpa, ms_ecpa.backend as b; print(\"kernel ok, backend:\", b.mode(), \"available:\", b.available())"
"$PY" scripts/make_all_paper_figures.py --force
