#!/bin/bash
# Cohort-wide read-only coverage audit. No GPU is requested.
#SBATCH --job-name=pvi-voltage-audit
#SBATCH --output=logs/pvi-voltage-audit_%j.out
#SBATCH --error=logs/pvi-voltage-audit_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250GB
#SBATCH --time=10-00:00:00

set -euo pipefail
module load gcc/11.2.0

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
source "$(dirname "$(dirname "${GCNM_PYTHON}")")/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PVI_SOLVER_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

OUTPUT="${OUTPUT:-${REPO_ROOT}/reports/cohort_voltage_audit_v2.json}"
if [[ -e "${OUTPUT}" ]]; then
  echo "ERROR: immutable cohort audit exists: ${OUTPUT}" >&2
  exit 1
fi
python -u -m gcnm_pvi.audit_voltage_distributions \
  --registry "${REGISTRY:-${REPO_ROOT}/data/registries/main_b045_v1.json}" \
  --output "${OUTPUT}" \
  --reuse-source-hashes-from "${REPO_ROOT}/reports/cohort_voltage_audit_v1.json"
