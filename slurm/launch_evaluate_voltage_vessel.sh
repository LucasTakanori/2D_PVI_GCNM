#!/bin/bash
#SBATCH --job-name=gcnm-vessel-eval
#SBATCH --output=logs/gcnm-vessel-eval_%j.out
#SBATCH --error=logs/gcnm-vessel-eval_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250GB
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

module load CUDA/12.9.0
module load gcc/11.2.0

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
if [[ -f "${REPO_ROOT}/env/cluster.env" ]]; then source "${REPO_ROOT}/env/cluster.env"; fi
VENV_ROOT="${GCNM_VENV_ROOT:-$(dirname "$(dirname "${GCNM_PYTHON:-${REPO_ROOT}/.venv/bin/python}")")}"
source "${VENV_ROOT}/bin/activate"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}:${PVI_SOLVER_ROOT:-}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/gcnm_vessel_eval_${SLURM_JOB_ID}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

CONFIG="${CONFIG:-${REPO_ROOT}/configs/subject006_anatomical_gcnm.yaml}"
DATASET_DIR="${DATASET_DIR:?set DATASET_DIR}"
EXPERIMENT="${EXPERIMENT:?set EXPERIMENT}"
TARGET_KIND="${TARGET_KIND:?set TARGET_KIND}"
EVALUATION_NAME="${EVALUATION_NAME:?set EVALUATION_NAME}"
MODELS_DIR="${MODELS_DIR:-${REPO_ROOT}/models/faithful/${EXPERIMENT}}"
OUT_DIR="${REPO_ROOT}/data/faithful_results/${EXPERIMENT}/${EVALUATION_NAME}"
TEST_FILE="${TEST_FILE:-${DATASET_DIR}/test.npz}"

python -u -m gcnm_pvi.evaluate_voltage_vessel_gcnm \
  --config "${CONFIG}" \
  --test "${TEST_FILE}" \
  --localizer "${MODELS_DIR}/${EXPERIMENT}_localizer.pt" \
  --refiner "${MODELS_DIR}/${EXPERIMENT}_refiner.pt" \
  --out-dir "${OUT_DIR}" \
  --target-kind "${TARGET_KIND}"

deactivate
