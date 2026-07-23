#!/usr/bin/env bash
# Submit the subject006/010 BP pilot only after both Parquet exports exist.
# This wrapper is intentionally dependent on the export array.
#SBATCH --job-name=us120-submit-bp
#SBATCH --output=logs/us120-submit-bp_%j.out
#SBATCH --error=logs/us120-submit-bp_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=01:00:00

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
export PARQUET_ROOT="${PARQUET_ROOT:-${REPO_ROOT}/gcnm_parquet}"
bash scripts/submit_us120_hp_lp_bp_pilot.sh
