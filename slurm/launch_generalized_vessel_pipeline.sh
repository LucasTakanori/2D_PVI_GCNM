#!/bin/bash
#SBATCH --job-name=gcnm-beat-slots
#SBATCH --output=logs/gcnm-beat-slots_%j.out
#SBATCH --error=logs/gcnm-beat-slots_%j.err
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
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/gcnm_beat_slots_${SLURM_JOB_ID}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

CONFIG="${CONFIG:-${REPO_ROOT}/configs/subject006_anatomical_gcnm.yaml}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/finger_generalized_beats_US120_v2}"
EXPERIMENT="${EXPERIMENT:?set EXPERIMENT}"
ARCHITECTURE="${ARCHITECTURE:?set ARCHITECTURE}"
MODELS_DIR="${REPO_ROOT}/models/faithful/${EXPERIMENT}"
RESULTS_DIR="${REPO_ROOT}/data/faithful_results/${EXPERIMENT}"
REAL_TEST_FILE="${REAL_TEST_FILE:-${REPO_ROOT}/data/finger_default_anatomical_exact/subject006_test_default_finger_baseline.npz}"

python -u -m gcnm_pvi.train_voltage_vessel_gcnm \
  --config "${CONFIG}" \
  --train "${DATASET_DIR}/train.npz" \
  --validation "${DATASET_DIR}/validation.npz" \
  --train-anatomy "${DATASET_DIR}/train_anatomy.json" \
  --validation-anatomy "${DATASET_DIR}/validation_anatomy.json" \
  --model-name "${EXPERIMENT}" \
  --models-dir "${MODELS_DIR}" \
  --results-dir "${RESULTS_DIR}" \
  --architecture "${ARCHITECTURE}" \
  --baseline-mode homogeneous \
  --baseline-conductivity 0.7 \
  --conductivity-scale-mode positive_p995 \
  --positive-weight 1.0 \
  --background-weight 0.25 \
  --dice-weight 0.05 \
  --slot-weight 0.30 \
  --separation-weight 0.10 \
  --attention-weight 0.05 \
  --correlation-weight 0.05 \
  --physics-weight 0.01 \
  --minimum-center-separation 0.25 \
  --minimum-vessel-axis 0.025 \
  --maximum-vessel-axis 0.23 \
  --seed 0

python -u -m gcnm_pvi.evaluate_voltage_vessel_gcnm \
  --config "${CONFIG}" \
  --test "${DATASET_DIR}/test.npz" \
  --localizer "${MODELS_DIR}/${EXPERIMENT}_localizer.pt" \
  --refiner "${MODELS_DIR}/${EXPERIMENT}_refiner.pt" \
  --out-dir "${RESULTS_DIR}/evaluation_nonlinear" \
  --target-kind clean

python -u -m gcnm_pvi.evaluate_voltage_vessel_gcnm \
  --config "${CONFIG}" \
  --test "${REAL_TEST_FILE}" \
  --localizer "${MODELS_DIR}/${EXPERIMENT}_localizer.pt" \
  --refiner "${MODELS_DIR}/${EXPERIMENT}_refiner.pt" \
  --out-dir "${RESULTS_DIR}/evaluation_real_pvi" \
  --target-kind real_subject006

EXPERIMENT="${EXPERIMENT}" \
EXPECTED_STAGES=2 \
SYNTHETIC_ANATOMY_JSON="${DATASET_DIR}/test_anatomy.json" \
bash "${REPO_ROOT}/scripts/make_stage_comparison_gifs.sh"

deactivate
