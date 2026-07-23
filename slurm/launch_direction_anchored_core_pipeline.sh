#!/bin/bash
# One-allocation, apples-to-apples architecture comparison.
# Training protocol matches finger_default_core_guided_hom_seed0 exactly.
#SBATCH --job-name=gcnm-dir-anchor
#SBATCH --output=logs/gcnm-dir-anchor_%j.out
#SBATCH --error=logs/gcnm-dir-anchor_%j.err
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
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/gcnm_mpl_${SLURM_JOB_ID}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

EXPERIMENT="${EXPERIMENT:-finger_default_direction_anchored_core_residual_hom_seed0}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/subject006_anatomical_gcnm.yaml}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/finger_default_anatomical_exact}"
REAL_TEST_FILE="${REAL_TEST_FILE:-${REPO_ROOT}/data/subject006_gcnm_hdf/test.npz}"
MODEL_DIR="${REPO_ROOT}/models/faithful/${EXPERIMENT}"
RESULT_DIR="${REPO_ROOT}/data/faithful_results/${EXPERIMENT}"

if [[ -e "${MODEL_DIR}" ]] || [[ -e "${RESULT_DIR}" ]]; then
  echo "ERROR: output exists for ${EXPERIMENT}" >&2
  exit 1
fi

python -u -m gcnm_pvi.train_core_guided_gcnm \
  --config "${CONFIG}" \
  --train "${DATASET_DIR}/train.npz" \
  --validation "${DATASET_DIR}/validation.npz" \
  --train-anatomy "${DATASET_DIR}/train_anatomy.json" \
  --validation-anatomy "${DATASET_DIR}/validation_anatomy.json" \
  --model-name "${EXPERIMENT}" \
  --models-dir "${MODEL_DIR}" \
  --results-dir "${RESULT_DIR}" \
  --architecture direction_anchored \
  --epochs 100 \
  --baseline-conductivity 0.7 \
  --maximum-amplitude-scale 2.0 \
  --seed 0

python -u -m gcnm_pvi.evaluate_core_guided_gcnm \
  --config "${CONFIG}" \
  --test "${DATASET_DIR}/test.npz" \
  --anatomy "${DATASET_DIR}/test_anatomy.json" \
  --stage1 "${MODEL_DIR}/${EXPERIMENT}_stage1.pt" \
  --stage2 "${MODEL_DIR}/${EXPERIMENT}_stage2.pt" \
  --out-dir "${RESULT_DIR}/evaluation_nonlinear" \
  --target-kind clean

python -u -m gcnm_pvi.evaluate_core_guided_gcnm \
  --config "${CONFIG}" \
  --test "${REAL_TEST_FILE}" \
  --stage1 "${MODEL_DIR}/${EXPERIMENT}_stage1.pt" \
  --stage2 "${MODEL_DIR}/${EXPERIMENT}_stage2.pt" \
  --out-dir "${RESULT_DIR}/evaluation_real_pvi" \
  --target-kind real_subject006

EXPERIMENT="${EXPERIMENT}" \
EXPECTED_STAGES=2 \
SAMPLES="0 10 20" \
SYNTHETIC_ANATOMY_JSON="${DATASET_DIR}/test_anatomy.json" \
bash scripts/make_stage_comparison_gifs.sh

echo "Completed one-job fair comparison: ${EXPERIMENT}"
deactivate
