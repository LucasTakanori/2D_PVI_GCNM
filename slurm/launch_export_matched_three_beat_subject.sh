#!/bin/bash
#SBATCH --job-name=fig26-match
#SBATCH --output=logs/fig26-match_%A_%a.out
#SBATCH --error=logs/fig26-match_%A_%a.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=04:00:00

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
OUTPUT_ROOT="${MATCHED_OUTPUT_ROOT:?MATCHED_OUTPUT_ROOT is required}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
source "$(dirname "$(dirname "${GCNM_PYTHON}")")/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

python -u scripts/export_matched_three_beat_comparison.py export-subject \
  --output-root "${OUTPUT_ROOT}" \
  --subject-index "${SLURM_ARRAY_TASK_ID:?array task ID required}" \
  --png-workers "${PNG_WORKERS:-8}"
