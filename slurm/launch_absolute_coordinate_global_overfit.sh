#!/bin/bash
# First corrected architecture gate: absolute voltage -> absolute conductivity.
#SBATCH --job-name=abs-gcnm-fit
#SBATCH --output=logs/abs-gcnm-fit_%j.out
#SBATCH --error=logs/abs-gcnm-fit_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
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

DATA_ROOT="${REPO_ROOT}/data/full_band_absolute_overfit_US120_smoke_v2"
MODEL_ROOT="${REPO_ROOT}/models/full_band_absolute_overfit_v2/coordinate_global"
RESULT_ROOT="${REPO_ROOT}/data/full_band_absolute_overfit_results_v2/coordinate_global"
if [[ -e "${MODEL_ROOT}" || -e "${RESULT_ROOT}" ]]; then
  echo "ERROR: immutable corrected overfit output already exists" >&2
  exit 2
fi

python -u -m gcnm_pvi.train_faithful_gcnm \
  --config "${REPO_ROOT}/configs/rings_b045/US120.yaml" \
  --train "${DATA_ROOT}/train.npz" --validation "${DATA_ROOT}/validation.npz" \
  --model-name coordinate_global_absolute \
  --models-dir "${MODEL_ROOT}" --results-dir "${RESULT_ROOT}" \
  --physics-contract absolute --output-mode direct \
  --use-coordinates --use-voltage-mlp \
  --positive-weight 1.0 --background-weight 0.25 \
  --correlation-weight 0.1 --relative-amplitude-weight 0.05 \
  --checkpoint-mode loss --seed 0 --baseline-mode homogeneous \
  --baseline-conductivity 0.7 --iterations 1 --epochs 500 --patience 100

python -u -m gcnm_pvi.evaluate_absolute_gcnm \
  --config "${REPO_ROOT}/configs/rings_b045/US120.yaml" \
  --test "${DATA_ROOT}/validation.npz" \
  --checkpoint "${MODEL_ROOT}/coordinate_global_absolute_0.pt" \
  --output "${RESULT_ROOT}/absolute_evaluation"
