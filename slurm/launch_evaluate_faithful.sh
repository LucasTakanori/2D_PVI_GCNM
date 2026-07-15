#!/bin/bash
#SBATCH --job-name=gcnm-faithful-eval
#SBATCH --output=logs/gcnm-faithful-eval_%j.out
#SBATCH --error=logs/gcnm-faithful-eval_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250GB
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

module load CUDA/12.9.0
module load gcc/11.2.0

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
if [[ -f "${REPO_ROOT}/env/cluster.env" ]]; then
  source "${REPO_ROOT}/env/cluster.env"
fi
VENV_ROOT="${GCNM_VENV_ROOT:-$(dirname "$(dirname "${GCNM_PYTHON:-${REPO_ROOT}/.venv/bin/python}")")}"
source "${VENV_ROOT}/bin/activate"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}:${PVI_SOLVER_ROOT:-}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/gcnm_mpl_${SLURM_JOB_ID}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

CONFIG="${CONFIG:-${REPO_ROOT}/configs/subject006_anatomical_gcnm.yaml}"
DATASET_DIR="${DATASET_DIR:?set DATASET_DIR to the evaluation pack directory}"
TEST_FILE="${TEST_FILE:-${DATASET_DIR}/test.npz}"
EXPERIMENT="${EXPERIMENT:?set EXPERIMENT to the trained result name}"
EVALUATION_NAME="${EVALUATION_NAME:-evaluation}"
TARGET_KIND="${TARGET_KIND:-clean}"
MODELS_DIR="${MODELS_DIR:-${REPO_ROOT}/models/faithful/${EXPERIMENT}}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/data/faithful_results/${EXPERIMENT}/${EVALUATION_NAME}}"
MAX_TEST="${MAX_TEST:-}"
ITERATIONS="${ITERATIONS:-}"
SKIP_LM_CONTROL="${SKIP_LM_CONTROL:-0}"

ARGS=(
  --config "${CONFIG}"
  --test "${TEST_FILE}"
  --model-name "${EXPERIMENT}"
  --models-dir "${MODELS_DIR}"
  --out-dir "${OUT_DIR}"
  --target-kind "${TARGET_KIND}"
  --save-examples 8
)
if [[ -n "${MAX_TEST}" ]]; then ARGS+=(--max-test "${MAX_TEST}"); fi
if [[ -n "${ITERATIONS}" ]]; then ARGS+=(--iterations "${ITERATIONS}"); fi
if [[ "${SKIP_LM_CONTROL}" == "1" ]]; then ARGS+=(--skip-lm-control); fi

echo "job=${SLURM_JOB_ID} host=$(hostname) experiment=${EXPERIMENT} target=${TARGET_KIND} test=${TEST_FILE}"
nvidia-smi
python -u -m gcnm_pvi.evaluate_faithful_gcnm "${ARGS[@]}"

deactivate
