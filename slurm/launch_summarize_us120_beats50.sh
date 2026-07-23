#!/bin/bash
# CPU-only aggregation of the paired 50-samples-per-beat reports.
#SBATCH --job-name=us120-beats50-report
#SBATCH --output=logs/us120-beats50-report_%j.out
#SBATCH --error=logs/us120-beats50-report_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=01:00:00
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env" 2>/dev/null || true
python -u scripts/summarize_us120_beats50_comparison.py \
  --new-root "${NEW_ROOT}" \
  --baseline-root "${BASELINE_ROOT}" \
  --output "${REPORT_DIR}/comparison_50_samples.json"
