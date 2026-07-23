#!/bin/bash
#SBATCH --job-name=compare-s010-channels
#SBATCH --output=logs/compare-s010-channels_%j.out
#SBATCH --error=logs/compare-s010-channels_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=00:20:00

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
python scripts/compare_subject010_channel_contracts.py \
  --repo-root "${REPO_ROOT}" \
  --output reports/subject010_coordinate_3ch_vs_6ch_v1.json
python scripts/build_coordinate_bp_results_dashboard.py
