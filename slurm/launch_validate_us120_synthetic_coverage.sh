#!/bin/bash
# CPU-only gate between synthetic generation and GCNM training.
#SBATCH --job-name=us120-coverage
#SBATCH --output=logs/us120-coverage_%j.out
#SBATCH --error=logs/us120-coverage_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=10-00:00:00

set -euo pipefail
module load gcc/11.2.0
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
source "$(dirname "$(dirname "${GCNM_PYTHON}")")/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data/hp_lp_beats_US120_v1}"
AUDIT="${COHORT_AUDIT:-${REPO_ROOT}/reports/cohort_voltage_audit_v2.json}"
python -u -m gcnm_pvi.validate_hp_lp_beat_dataset \
  --root "${DATASET_ROOT}" --output "${DATASET_ROOT}/validation_recheck.json"
python -u -m gcnm_pvi.validate_synthetic_cohort_coverage \
  --dataset-root "${DATASET_ROOT}" --cohort-audit "${AUDIT}" \
  --output "${DATASET_ROOT}/cohort_coverage.json" \
  --minimum-scale-ratio 0.02 --maximum-scale-ratio 50 \
  --minimum-negative-fraction 0.001 --maximum-negative-fraction 0.999
