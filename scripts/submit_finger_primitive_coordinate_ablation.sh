#!/usr/bin/env bash
# Retrain the pre-voltage-MLP selected coordinate GCNM on the finger dataset.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs

CONFIG="${CONFIG:-${REPO_ROOT}/configs/subject006_anatomical_gcnm.yaml}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/finger_default_anatomical_exact}"
REAL_TEST_FILE="${REAL_TEST_FILE:-${DATASET_DIR}/subject006_test_default_finger_baseline.npz}"
EXPERIMENT="${EXPERIMENT:-finger_default_primitive_coords_a1_b025_hom_seed0}"

if [[ -e "${REPO_ROOT}/models/faithful/${EXPERIMENT}" ]] || \
   [[ -e "${REPO_ROOT}/data/faithful_results/${EXPERIMENT}" ]]; then
  echo "ERROR: experiment output already exists: ${EXPERIMENT}" >&2
  exit 1
fi

# CPU, memory, GPU, partition, and account remain defined by the established
# Slurm launchers. Only the wall-time override is supplied here.
TRAIN_JOB=$(sbatch --parsable \
  --time=02:00:00 \
  --job-name="${EXPERIMENT}-train" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},DATASET_DIR=${DATASET_DIR},EXPERIMENT=${EXPERIMENT},OUTPUT_MODE=direct,USE_COORDINATES=1,USE_VOLTAGE_MLP=0,POSITIVE_WEIGHT=1.0,BACKGROUND_WEIGHT=0.25,DICE_WEIGHT=0.0,CHECKPOINT_MODE=composite,BASELINE_MODE=homogeneous,BASELINE_CONDUCTIVITY=0.7,ITERATIONS=2,SEED=0" \
  slurm/launch_train_faithful.sh)

SYNTHETIC_JOB=$(sbatch --parsable \
  --time=02:00:00 \
  --dependency="afterok:${TRAIN_JOB}" \
  --job-name="${EXPERIMENT}-synthetic" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},DATASET_DIR=${DATASET_DIR},TEST_FILE=${DATASET_DIR}/test.npz,EXPERIMENT=${EXPERIMENT},EVALUATION_NAME=evaluation_nonlinear,TARGET_KIND=clean,ITERATIONS=2" \
  slurm/launch_evaluate_faithful.sh)

REAL_JOB=$(sbatch --parsable \
  --time=02:00:00 \
  --dependency="afterok:${TRAIN_JOB}" \
  --job-name="${EXPERIMENT}-real" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},DATASET_DIR=${DATASET_DIR},TEST_FILE=${REAL_TEST_FILE},EXPERIMENT=${EXPERIMENT},EVALUATION_NAME=evaluation_real_pvi,TARGET_KIND=real_subject006,ITERATIONS=2,SKIP_LM_CONTROL=1" \
  slurm/launch_evaluate_faithful.sh)

GIF_JOB=$(sbatch --parsable \
  --time=02:00:00 \
  --dependency="afterok:${SYNTHETIC_JOB}:${REAL_JOB}" \
  --job-name="${EXPERIMENT}-gifs" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},EXPERIMENT=${EXPERIMENT},EXPECTED_STAGES=2" \
  slurm/launch_make_stage_gifs.sh)

echo "experiment: ${EXPERIMENT}"
echo "training: ${TRAIN_JOB}"
echo "synthetic evaluation: ${SYNTHETIC_JOB}"
echo "real subject-006 evaluation: ${REAL_JOB}"
echo "comparison GIFs: ${GIF_JOB}"
