#!/usr/bin/env bash
# Train and evaluate the core-guided two-stage model on the original
# default-finger dataset used by the earlier PVI-GCNM experiments.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs

CONFIG="${CONFIG:-${REPO_ROOT}/configs/subject006_anatomical_gcnm.yaml}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/finger_default_anatomical_exact}"
EXPERIMENT="${EXPERIMENT:-finger_default_core_guided_hom_seed0}"
REAL_TEST_FILE="${REAL_TEST_FILE:-${REPO_ROOT}/data/subject006_gcnm_hdf/test.npz}"

for required in \
  "${DATASET_DIR}/train.npz" \
  "${DATASET_DIR}/validation.npz" \
  "${DATASET_DIR}/test.npz" \
  "${DATASET_DIR}/train_anatomy.json" \
  "${DATASET_DIR}/validation_anatomy.json" \
  "${DATASET_DIR}/test_anatomy.json" \
  "${REAL_TEST_FILE}"; do
  if [[ ! -f "${required}" ]]; then
    echo "ERROR: missing required comparison input: ${required}" >&2
    exit 1
  fi
done

if [[ -e "${REPO_ROOT}/models/faithful/${EXPERIMENT}" ]] || \
   [[ -e "${REPO_ROOT}/data/faithful_results/${EXPERIMENT}" ]]; then
  echo "ERROR: experiment output already exists; choose a new EXPERIMENT" >&2
  exit 1
fi

TRAIN_JOB=$(sbatch --parsable \
  --time=02:00:00 \
  --job-name="${EXPERIMENT}-train" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},DATASET_DIR=${DATASET_DIR},EXPERIMENT=${EXPERIMENT},EPOCHS=100,BASELINE_CONDUCTIVITY=0.7,SEED=0" \
  slurm/launch_train_core_guided.sh)

SYNTHETIC_JOB=$(sbatch --parsable \
  --time=02:00:00 \
  --dependency="afterok:${TRAIN_JOB}" \
  --job-name="${EXPERIMENT}-synthetic" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},EXPERIMENT=${EXPERIMENT},TEST_FILE=${DATASET_DIR}/test.npz,ANATOMY_FILE=${DATASET_DIR}/test_anatomy.json,EVALUATION_NAME=evaluation_nonlinear,TARGET_KIND=clean" \
  slurm/launch_evaluate_core_guided.sh)

REAL_JOB=$(sbatch --parsable \
  --time=02:00:00 \
  --dependency="afterok:${TRAIN_JOB}" \
  --job-name="${EXPERIMENT}-real" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},EXPERIMENT=${EXPERIMENT},TEST_FILE=${REAL_TEST_FILE},EVALUATION_NAME=evaluation_real_pvi,TARGET_KIND=real_subject006" \
  slurm/launch_evaluate_core_guided.sh)

GIF_JOB=$(sbatch --parsable \
  --time=01:00:00 \
  --dependency="afterok:${SYNTHETIC_JOB}:${REAL_JOB}" \
  --job-name="${EXPERIMENT}-gifs" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},EXPERIMENT=${EXPERIMENT},EXPECTED_STAGES=2,SAMPLES=0 10 20,SYNTHETIC_ANATOMY_JSON=${DATASET_DIR}/test_anatomy.json" \
  slurm/launch_make_stage_gifs.sh)

echo "experiment: ${EXPERIMENT}"
echo "dataset: ${DATASET_DIR}"
echo "two-stage core-guided training: ${TRAIN_JOB}"
echo "same-holdout synthetic evaluation: ${SYNTHETIC_JOB}"
echo "subject-006 voltage evaluation: ${REAL_JOB}"
echo "comparison GIFs: ${GIF_JOB}"
