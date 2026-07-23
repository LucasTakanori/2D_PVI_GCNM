#!/bin/bash
# CPU-only result validation and paired comparison after all 24 BP runs.
#SBATCH --job-name=us120-bp-summary
#SBATCH --output=logs/us120-bp-summary_%j.out
#SBATCH --error=logs/us120-bp-summary_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=04:00:00

set -euo pipefail
module load gcc/11.2.0
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
source "$(dirname "$(dirname "${GCNM_PYTHON}")")/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${REPO_ROOT}/artifacts/pvi_bp_us120_hp_lp_pilot_v1}"
REPORT_ROOT="${REPORT_ROOT:-${REPO_ROOT}/reports/hp_lp_us120_v1/bp_pilot}"
python -u -m gcnm_pvi.summarize_bp_pilot \
  --artifact-root "${ARTIFACT_ROOT}" \
  --output "${REPORT_ROOT}/summary.json" \
  --csv "${REPORT_ROOT}/paired_differences.csv"
