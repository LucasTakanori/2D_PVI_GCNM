#!/bin/bash
#SBATCH --job-name=fig26-match-final
#SBATCH --output=logs/fig26-match-final_%j.out
#SBATCH --error=logs/fig26-match-final_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=02:00:00

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
OUTPUT_ROOT="${MATCHED_OUTPUT_ROOT:?MATCHED_OUTPUT_ROOT is required}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
source "$(dirname "$(dirname "${GCNM_PYTHON}")")/bin/activate"

python -u scripts/export_matched_three_beat_comparison.py finalize \
  --output-root "${OUTPUT_ROOT}"
