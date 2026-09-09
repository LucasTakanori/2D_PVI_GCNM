#!/usr/bin/env bash
# Export one patient's ten-beat six-channel image package.
#SBATCH --job-name=sixch-images
#SBATCH --output=logs/sixch-images_%A_%a.out
#SBATCH --error=logs/sixch-images_%A_%a.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=08:00:00

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT must be set}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"${GCNM_PYTHON}" -u scripts/export_all_patient_six_channel_images.py \
  export-subject \
  --output-root "${OUTPUT_ROOT}" \
  --subject-index "${SLURM_ARRAY_TASK_ID}" \
  --png-workers "${SLURM_CPUS_PER_TASK}"
