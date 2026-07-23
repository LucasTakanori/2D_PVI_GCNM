#!/bin/bash
# CPU-only preflight/release job submitted after both Parquet exports finish.
#SBATCH --job-name=us120-bp-release
#SBATCH --output=logs/us120-bp-release_%j.out
#SBATCH --error=logs/us120-bp-release_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=04:00:00

set -euo pipefail
module load gcc/11.2.0
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
source "$(dirname "$(dirname "${GCNM_PYTHON}")")/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
bash scripts/submit_us120_hp_lp_bp_pilot.sh
