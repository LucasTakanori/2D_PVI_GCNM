#!/bin/bash
# Loss-only ablation on the best one-frame global-voltage coordinate model.
# Compare these two tasks with coordinate_global from job 432936.
#SBATCH --job-name=gcnm-1beat-loss
#SBATCH --output=logs/gcnm-1beat-loss_%A_%a.out
#SBATCH --error=logs/gcnm-1beat-loss_%A_%a.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

set -euo pipefail
module load CUDA/12.9.0
module load gcc/11.2.0
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
source "$(dirname "$(dirname "${GCNM_PYTHON}")")/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PVI_SOLVER_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export GCNM_PHYSICS_WORKERS="${SLURM_CPUS_PER_TASK}"

TASK_ID="${SLURM_ARRAY_TASK_ID:?submit as array 0-1%2}"
NAMES=(coordinate_global_element_mse coordinate_global_balanced_mse)
NAME="${NAMES[${TASK_ID}]}"
MODEL_ROOT="${REPO_ROOT}/models/full_band_one_beat_overfit_v1/${NAME}"
RESULT_ROOT="${REPO_ROOT}/data/full_band_one_beat_overfit_results_v1/${NAME}"
DATA_ROOT="${REPO_ROOT}/data/full_band_overfit_US120_anatomy8_beat40_v1"
if [[ -e "${MODEL_ROOT}" || -e "${RESULT_ROOT}" ]]; then
  echo "immutable pilot output already exists for ${NAME}" >&2
  exit 2
fi

LOSS_FLAGS=()
if [[ "${TASK_ID}" -eq 1 ]]; then
  LOSS_FLAGS+=(--balanced-support-loss)
fi
python -u -m gcnm_pvi.train_faithful_gcnm \
  --config "${REPO_ROOT}/configs/rings_b045/US120.yaml" \
  --train "${DATA_ROOT}/train.npz" --validation "${DATA_ROOT}/validation.npz" \
  --model-name "${NAME}" --models-dir "${MODEL_ROOT}" --results-dir "${RESULT_ROOT}" \
  --output-mode direct --use-coordinates --use-voltage-mlp "${LOSS_FLAGS[@]}" \
  --checkpoint-mode loss --seed 0 --baseline-mode homogeneous \
  --baseline-conductivity 0.7 --iterations 1 --epochs 500 --patience 100

python -u -m gcnm_pvi.evaluate_faithful_gcnm \
  --config "${REPO_ROOT}/configs/rings_b045/US120.yaml" \
  --test "${DATA_ROOT}/validation.npz" --model-name "${NAME}" \
  --models-dir "${MODEL_ROOT}" --out-dir "${RESULT_ROOT}/overfit_evaluation" \
  --iterations 1 --target-kind clean --baseline-mode homogeneous \
  --baseline-conductivity 0.7 --skip-lm-control --save-examples 50
