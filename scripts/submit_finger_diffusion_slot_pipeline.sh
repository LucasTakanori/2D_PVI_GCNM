#!/usr/bin/env bash
# Train two spatial vessel cores with learned tissue-aware diffusion halos.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs

EXPERIMENT="${EXPERIMENT:-finger_default_diffusion_slots_corr010_mlp_seed0}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/subject006_anatomical_gcnm.yaml}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/finger_default_anatomical_exact}"
REAL_TEST_FILE="${REAL_TEST_FILE:-${DATASET_DIR}/subject006_test_default_finger_baseline.npz}"

if [[ -e "${REPO_ROOT}/models/faithful/${EXPERIMENT}" ]] || \
   [[ -e "${REPO_ROOT}/data/faithful_results/${EXPERIMENT}" ]]; then
  echo "ERROR: experiment output already exists: ${EXPERIMENT}" >&2
  exit 1
fi

TRAIN_JOB=$(sbatch --parsable \
  --time=02:00:00 \
  --job-name="${EXPERIMENT}-train" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},DATASET_DIR=${DATASET_DIR},EXPERIMENT=${EXPERIMENT},ARCHITECTURE=diffusion_slots,BASELINE_MODE=saved,CONDUCTIVITY_SCALE_MODE=positive_p995,POSITIVE_WEIGHT=1.0,BACKGROUND_WEIGHT=0.25,DICE_WEIGHT=0.05,SLOT_WEIGHT=0.5,SEPARATION_WEIGHT=0.15,ATTENTION_WEIGHT=0.05,CORRELATION_WEIGHT=0.10,MINIMUM_CENTER_SEPARATION=0.25,MINIMUM_VESSEL_AXIS=0.025,MAXIMUM_VESSEL_AXIS=0.23,SEED=0" \
  slurm/launch_train_voltage_vessel.sh)

SYNTHETIC_JOB=$(sbatch --parsable \
  --time=02:00:00 \
  --dependency="afterok:${TRAIN_JOB}" \
  --job-name="${EXPERIMENT}-synthetic" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},DATASET_DIR=${DATASET_DIR},EXPERIMENT=${EXPERIMENT},EVALUATION_NAME=evaluation_nonlinear,TARGET_KIND=clean" \
  slurm/launch_evaluate_voltage_vessel.sh)

REAL_JOB=$(sbatch --parsable \
  --time=02:00:00 \
  --dependency="afterok:${TRAIN_JOB}" \
  --job-name="${EXPERIMENT}-real" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},DATASET_DIR=${DATASET_DIR},TEST_FILE=${REAL_TEST_FILE},EXPERIMENT=${EXPERIMENT},EVALUATION_NAME=evaluation_real_pvi,TARGET_KIND=real_subject006" \
  slurm/launch_evaluate_voltage_vessel.sh)

GIF_JOB=$(sbatch --parsable \
  --time=02:00:00 \
  --dependency="afterok:${SYNTHETIC_JOB}:${REAL_JOB}" \
  --job-name="${EXPERIMENT}-gifs" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},EXPERIMENT=${EXPERIMENT},EXPECTED_STAGES=2,SYNTHETIC_ANATOMY_JSON=${DATASET_DIR}/test_anatomy.json" \
  slurm/launch_make_stage_gifs.sh)

echo "experiment: ${EXPERIMENT}"
echo "core-plus-diffusion training: ${TRAIN_JOB}"
echo "synthetic evaluation: ${SYNTHETIC_JOB}"
echo "real subject-006 evaluation: ${REAL_JOB}"
echo "waveform comparison GIFs: ${GIF_JOB}"
