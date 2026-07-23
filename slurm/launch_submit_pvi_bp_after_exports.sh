#!/bin/bash
# Validate coordinate and global-voltage-slot exports, then submit 728 runs.
#SBATCH --job-name=gcnm-bp-submit
#SBATCH --output=logs/gcnm-bp-submit_%j.out
#SBATCH --error=logs/gcnm-bp-submit_%j.err
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
export SPLIT_MANIFEST="${SPLIT_MANIFEST:-${REPO_ROOT}/data/manifests/pvi_subject_splits_mask05_v1.json}"
export COORDINATE_PARQUET_ROOT="${COORDINATE_PARQUET_ROOT:-${REPO_ROOT}/gcnm_parquet/coordinate_v2}"
export GLOBAL_VOLTAGE_SLOTS_PARQUET_ROOT="${GLOBAL_VOLTAGE_SLOTS_PARQUET_ROOT:-${REPO_ROOT}/gcnm_parquet/global_voltage_slots_v2}"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-${REPO_ROOT}/artifacts/pvi_bp_gcnm_v1}"
export MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
bash scripts/submit_pvi_bp_matrix.sh
