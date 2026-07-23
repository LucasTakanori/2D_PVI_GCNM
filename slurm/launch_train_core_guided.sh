#!/bin/bash
#SBATCH --job-name=gcnm-core-train
#SBATCH --output=logs/gcnm-core-train_%j.out
#SBATCH --error=logs/gcnm-core-train_%j.err
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
EXPERIMENT="${EXPERIMENT:?set EXPERIMENT}"
python -u -m gcnm_pvi.train_core_guided_gcnm \
  --config "${CONFIG:-${REPO_ROOT}/configs/subject006_anatomical_gcnm.yaml}" \
  --train "${DATASET_DIR}/train.npz" \
  --validation "${DATASET_DIR}/validation.npz" \
  --train-anatomy "${DATASET_DIR}/train_anatomy.json" \
  --validation-anatomy "${DATASET_DIR}/validation_anatomy.json" \
  --model-name "${EXPERIMENT}" \
  --models-dir "${REPO_ROOT}/models/faithful/${EXPERIMENT}" \
  --results-dir "${REPO_ROOT}/data/faithful_results/${EXPERIMENT}" \
  --epochs "${EPOCHS:-100}" \
  --baseline-conductivity "${BASELINE_CONDUCTIVITY:-0.7}" \
  --seed "${SEED:-0}"

deactivate
