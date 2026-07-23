#!/bin/bash
#SBATCH --job-name=gcnm-physcal-train
#SBATCH --output=logs/gcnm-physcal-train_%j.out
#SBATCH --error=logs/gcnm-physcal-train_%j.err
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

DATASET_DIR="${DATASET_DIR:?set DATASET_DIR}"
REAL_TRAIN_FILE="${REAL_TRAIN_FILE:?set REAL_TRAIN_FILE}"
REAL_VALIDATION_FILE="${REAL_VALIDATION_FILE:?set REAL_VALIDATION_FILE}"
EXPERIMENT="${EXPERIMENT:?set EXPERIMENT}"
ARGS=(
  --config "${CONFIG:-${REPO_ROOT}/configs/subject006_anatomical_gcnm.yaml}"
  --train "${DATASET_DIR}/train.npz"
  --validation "${DATASET_DIR}/validation.npz"
  --train-anatomy "${DATASET_DIR}/train_anatomy.json"
  --validation-anatomy "${DATASET_DIR}/validation_anatomy.json"
  --real-train "${REAL_TRAIN_FILE}"
  --real-validation "${REAL_VALIDATION_FILE}"
  --model-name "${EXPERIMENT}"
  --models-dir "${REPO_ROOT}/models/faithful/${EXPERIMENT}"
  --results-dir "${REPO_ROOT}/data/faithful_results/${EXPERIMENT}"
  --epochs "${EPOCHS:-100}"
  --baseline-conductivity "${BASELINE_CONDUCTIVITY:-0.7}"
  --maximum-amplitude "${MAXIMUM_AMPLITUDE:-0.25}"
  --real-voltage-weight "${REAL_VOLTAGE_WEIGHT:-0.25}"
  --synthetic-voltage-weight "${SYNTHETIC_VOLTAGE_WEIGHT:-0.05}"
  --amplitude-factor-min "${AMPLITUDE_FACTOR_MIN:-0.25}"
  --amplitude-factor-max "${AMPLITUDE_FACTOR_MAX:-6.0}"
  --seed "${SEED:-0}"
)
if [[ -n "${MAX_TRAIN:-}" ]]; then ARGS+=(--max-train "${MAX_TRAIN}"); fi
if [[ -n "${MAX_VALIDATION:-}" ]]; then ARGS+=(--max-validation "${MAX_VALIDATION}"); fi
if [[ -n "${MAX_REAL_TRAIN:-}" ]]; then ARGS+=(--max-real-train "${MAX_REAL_TRAIN}"); fi
if [[ -n "${MAX_REAL_VALIDATION:-}" ]]; then ARGS+=(--max-real-validation "${MAX_REAL_VALIDATION}"); fi
if [[ "${ALLOW_OVERWRITE:-0}" == "1" ]]; then ARGS+=(--allow-overwrite); fi

echo "job=${SLURM_JOB_ID} host=$(hostname) experiment=${EXPERIMENT}"
echo "synthetic=${DATASET_DIR} real_train=${REAL_TRAIN_FILE} real_validation=${REAL_VALIDATION_FILE}"
echo "real_voltage_weight=${REAL_VOLTAGE_WEIGHT:-0.25} maximum_amplitude=${MAXIMUM_AMPLITUDE:-0.25}"
nvidia-smi
python -u -m gcnm_pvi.train_physics_calibrated_gcnm "${ARGS[@]}"

deactivate
