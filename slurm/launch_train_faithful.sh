#!/bin/bash
#SBATCH --job-name=gcnm-faithful-train
#SBATCH --output=logs/gcnm-faithful-train_%j.out
#SBATCH --error=logs/gcnm-faithful-train_%j.err
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
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/subject006_anatomical_full}"
EXPERIMENT="${EXPERIMENT:?set EXPERIMENT to a unique result name}"
OUTPUT_MODE="${OUTPUT_MODE:-direct}"
USE_COORDINATES="${USE_COORDINATES:-0}"
USE_VOLTAGE_MLP="${USE_VOLTAGE_MLP:-0}"
POSITIVE_WEIGHT="${POSITIVE_WEIGHT:-0}"
BACKGROUND_WEIGHT="${BACKGROUND_WEIGHT:-0}"
DICE_WEIGHT="${DICE_WEIGHT:-0}"
HARD_BACKGROUND_WEIGHT="${HARD_BACKGROUND_WEIGHT:-0}"
HARD_BACKGROUND_FRACTION="${HARD_BACKGROUND_FRACTION:-0.05}"
CHECKPOINT_MODE="${CHECKPOINT_MODE:-loss}"
MAX_TRAIN="${MAX_TRAIN:-}"
MAX_VALIDATION="${MAX_VALIDATION:-}"
EPOCHS="${EPOCHS:-}"
ITERATIONS="${ITERATIONS:-}"
SEED="${SEED:-0}"
BASELINE_MODE="${BASELINE_MODE:-homogeneous}"
BASELINE_CONDUCTIVITY="${BASELINE_CONDUCTIVITY:-0.7}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
MODELS_DIR="${MODELS_DIR:-${REPO_ROOT}/models/faithful/${EXPERIMENT}}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/data/faithful_results/${EXPERIMENT}}"

ARGS=(
  --config "${CONFIG}"
  --train "${DATASET_DIR}/train.npz"
  --validation "${DATASET_DIR}/validation.npz"
  --model-name "${EXPERIMENT}"
  --models-dir "${MODELS_DIR}"
  --results-dir "${RESULTS_DIR}"
  --output-mode "${OUTPUT_MODE}"
  --positive-weight "${POSITIVE_WEIGHT}"
  --background-weight "${BACKGROUND_WEIGHT}"
  --dice-weight "${DICE_WEIGHT}"
  --hard-background-weight "${HARD_BACKGROUND_WEIGHT}"
  --hard-background-fraction "${HARD_BACKGROUND_FRACTION}"
  --checkpoint-mode "${CHECKPOINT_MODE}"
  --seed "${SEED}"
  --baseline-mode "${BASELINE_MODE}"
  --baseline-conductivity "${BASELINE_CONDUCTIVITY}"
)
if [[ "${USE_COORDINATES}" == "1" ]]; then ARGS+=(--use-coordinates); fi
if [[ "${USE_VOLTAGE_MLP}" == "1" ]]; then ARGS+=(--use-voltage-mlp); fi
if [[ -n "${MAX_TRAIN}" ]]; then ARGS+=(--max-train "${MAX_TRAIN}"); fi
if [[ -n "${MAX_VALIDATION}" ]]; then ARGS+=(--max-validation "${MAX_VALIDATION}"); fi
if [[ -n "${EPOCHS}" ]]; then ARGS+=(--epochs "${EPOCHS}"); fi
if [[ -n "${ITERATIONS}" ]]; then ARGS+=(--iterations "${ITERATIONS}"); fi
if [[ "${ALLOW_OVERWRITE}" == "1" ]]; then ARGS+=(--allow-overwrite); fi

echo "job=${SLURM_JOB_ID} host=$(hostname) experiment=${EXPERIMENT}"
echo "mode=${OUTPUT_MODE} coordinates=${USE_COORDINATES} voltage_mlp=${USE_VOLTAGE_MLP} positive=${POSITIVE_WEIGHT} background=${BACKGROUND_WEIGHT} dice=${DICE_WEIGHT} hard_bg=${HARD_BACKGROUND_WEIGHT} checkpoint=${CHECKPOINT_MODE}"
nvidia-smi
python -u -m gcnm_pvi.train_faithful_gcnm "${ARGS[@]}"

deactivate
