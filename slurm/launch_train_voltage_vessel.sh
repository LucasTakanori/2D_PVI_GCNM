#!/bin/bash
#SBATCH --job-name=gcnm-vessel-train
#SBATCH --output=logs/gcnm-vessel-train_%j.out
#SBATCH --error=logs/gcnm-vessel-train_%j.err
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
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/gcnm_vessel_${SLURM_JOB_ID}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

CONFIG="${CONFIG:-${REPO_ROOT}/configs/subject006_anatomical_gcnm.yaml}"
DATASET_DIR="${DATASET_DIR:?set DATASET_DIR}"
EXPERIMENT="${EXPERIMENT:?set EXPERIMENT}"
MODELS_DIR="${MODELS_DIR:-${REPO_ROOT}/models/faithful/${EXPERIMENT}}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/data/faithful_results/${EXPERIMENT}}"
EPOCHS="${EPOCHS:-}"

ARGS=(
  --config "${CONFIG}"
  --train "${DATASET_DIR}/train.npz"
  --validation "${DATASET_DIR}/validation.npz"
  --train-anatomy "${DATASET_DIR}/train_anatomy.json"
  --validation-anatomy "${DATASET_DIR}/validation_anatomy.json"
  --model-name "${EXPERIMENT}"
  --models-dir "${MODELS_DIR}"
  --results-dir "${RESULTS_DIR}"
  --baseline-conductivity "${BASELINE_CONDUCTIVITY:-0.7}"
  --baseline-mode "${BASELINE_MODE:-homogeneous}"
  --positive-weight "${POSITIVE_WEIGHT:-1.0}"
  --background-weight "${BACKGROUND_WEIGHT:-0.25}"
  --dice-weight "${DICE_WEIGHT:-0.05}"
  --slot-weight "${SLOT_WEIGHT:-0.2}"
  --separation-weight "${SEPARATION_WEIGHT:-0.1}"
  --attention-weight "${ATTENTION_WEIGHT:-0.05}"
  --correlation-weight "${CORRELATION_WEIGHT:-0.0}"
  --physics-weight "${PHYSICS_WEIGHT:-0.0}"
  --minimum-center-separation "${MINIMUM_CENTER_SEPARATION:-0.25}"
  --minimum-vessel-axis "${MINIMUM_VESSEL_AXIS:-0.05}"
  --maximum-vessel-axis "${MAXIMUM_VESSEL_AXIS:-0.23}"
  --conductivity-scale-mode "${CONDUCTIVITY_SCALE_MODE:-global_p995}"
  --architecture "${ARCHITECTURE:-mean_pool_dense_refiner}"
  --seed "${SEED:-0}"
)
if [[ -n "${EPOCHS}" ]]; then ARGS+=(--epochs "${EPOCHS}"); fi
if [[ "${ALLOW_OVERWRITE:-0}" == "1" ]]; then ARGS+=(--allow-overwrite); fi

echo "job=${SLURM_JOB_ID} host=$(hostname) experiment=${EXPERIMENT}"
nvidia-smi
python -u -m gcnm_pvi.train_voltage_vessel_gcnm "${ARGS[@]}"

deactivate
